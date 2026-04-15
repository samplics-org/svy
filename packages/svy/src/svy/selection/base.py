# src/svy/selection/base.py
from __future__ import annotations

import copy
import logging

from typing import TYPE_CHECKING, Iterable, Literal, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from svy.core.constants import (
    SVY_CERTAINTY,
    SVY_HIT,
    SVY_PROB,
    SVY_PROB_STAGE1,
    SVY_ROW_INDEX,
    SVY_WEIGHT,
)
from svy.core.enumerations import PPSMethod
from svy.core.types import (
    DF,
    Category,
    Number,
)
from svy.engine.sampling.combine_stages import (
    CombineResult,
    _apply_chaining_writeback,
    _combine_stages,
)
from svy.engine.sampling.pps import _select_pps
from svy.engine.sampling.srs import _select_srs
from svy.utils.checks import assert_no_missing, drop_missing
from svy.utils.helpers import _colspec_to_list
from svy.utils.random_state import RandomState, resolve_random_state, seed_from_random_state


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from svy.core.sample import Sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _psu_list(psu) -> list[str]:
    """Normalise a PSU spec (str | tuple | None) to a list of column names."""
    if psu is None:
        return []
    return [psu] if isinstance(psu, str) else list(psu)


def _apply_order(
    data: pl.DataFrame,
    *,
    order_by: str | Sequence[str] | None,
    order_type: Literal["ascending", "descending", "random"],
    seed: int | None,
) -> pl.DataFrame:
    """
    Apply frame ordering before selection.

    Parameters
    ----------
    order_by   : column(s) to sort/group by
    order_type : "ascending" | "descending" | "random"

                 * "ascending" / "descending" — sort by ``order_by``;
                   no-op when ``order_by`` is None.
                 * "random" without ``order_by`` — full random shuffle.
                 * "random" with ``order_by`` — sort by ``order_by``,
                   then shuffle within each group (implicit stratification).
                   The standard approach for systematic PPS.
    seed       : random seed for "random" order_type
    """
    if order_type == "random":
        if order_by is not None:
            sort_cols = [order_by] if isinstance(order_by, str) else list(order_by)
            return (
                data.with_columns(
                    pl.arange(0, pl.len()).shuffle(seed=seed).over(sort_cols).alias("_shuffle_col")
                )
                .sort(*sort_cols, "_shuffle_col")
                .drop("_shuffle_col")
            )
        return data.sample(fraction=1.0, with_replacement=False, shuffle=True, seed=seed)
    if order_by is not None:
        sort_cols = [order_by] if isinstance(order_by, str) else list(order_by)
        return data.sort(by=sort_cols, descending=(order_type == "descending"))
    return data


# ---------------------------------------------------------------------------
# Selection facade
# ---------------------------------------------------------------------------


class Selection:
    def __init__(self, sample: "Sample") -> None:
        self._sample = sample

    # ── bookkeeping ──────────────────────────────────────────────────────

    def _ensure_row_index(self) -> None:
        df = self._sample._data
        if SVY_ROW_INDEX not in df.columns:
            try:
                df = df.with_row_index(name=SVY_ROW_INDEX)
            except Exception:
                df = df.with_row_count(SVY_ROW_INDEX)
            self._sample._data = df
        if getattr(self._sample._design, "row_index", None) != SVY_ROW_INDEX:
            self._sample._design = self._sample._design.fill_missing(row_index=SVY_ROW_INDEX)

    @staticmethod
    def _unique_as_str(a: Iterable[object]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for v in a:
            s = str(v)
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # ── group-key helpers ────────────────────────────────────────────────

    def _build_group_keys(
        self,
        data: DF,
        *,
        stratum_col: str | None,
        by_cols: list[str],
        suffix: str,
    ) -> tuple[
        str | None,
        npt.NDArray[np.object_] | None,
        list[str],
        list[str],
        list[str],
        pl.DataFrame,
    ]:
        if isinstance(data, pl.LazyFrame):
            data = cast(pl.DataFrame, data.collect())
        group_parts: list[str] = []
        if stratum_col is not None:
            group_parts.append(stratum_col)
        group_parts.extend(by_cols)

        stratum_by_col: str | None = None
        stratum_by_arr: npt.NDArray[np.object_] | None = None

        if group_parts:
            data = cast(
                pl.DataFrame,
                self._sample._concatenate_cols(
                    data,
                    sep="__by__",
                    null_token="__Null__",
                    categorical=True,
                    drop_original=False,
                    rename_suffix=suffix,
                    stratum_by=group_parts,
                ),
            )
            stratum_by_col = f"stratum_by{suffix}"
            stratum_by_arr = data[stratum_by_col].to_numpy()

        G = self._unique_as_str(data[stratum_by_col].unique().to_list()) if stratum_by_col else []

        if by_cols:
            data = cast(
                pl.DataFrame,
                self._sample._concatenate_cols(
                    data,
                    sep="__by__",
                    null_token="__Null__",
                    categorical=True,
                    drop_original=False,
                    rename_suffix=suffix,
                    only_by=by_cols,
                ),
            )
            by_only_col = f"only_by{suffix}"
            B = self._unique_as_str(data[by_only_col].unique().to_list())
        else:
            B = []

        S = self._unique_as_str(data[stratum_col].unique().to_list()) if stratum_col else []

        return stratum_by_col, stratum_by_arr, G, B, S, data

    @staticmethod
    def _normalize_n_for_groups(
        n: Number | Mapping[Category, Number],
        *,
        G: list[str],
        B: list[str],
        S: list[str],
    ) -> int | dict[str, int]:
        """
        Normalize `n` to match the combined universe (G).

        - scalar -> broadcast to G (or scalar if ungrouped)
        - mapping keys == G -> pass-through
        - mapping keys ⊂ G -> fill missing with 0
        - mapping keys ==/⊂ B -> broadcast within each stratum
        - mapping keys ==/⊂ S -> broadcast across by
        - mapping keys match sub-level components of combined keys
        - ungrouped -> scalar
        """

        def _broadcast(levels: list[str], v: Number) -> dict[str, int]:
            return {lvl: int(v) for lvl in levels}

        def _coerce(m: Mapping[Category, Number]) -> dict[str, int]:
            return {str(k): int(v) for k, v in m.items()}

        def _fill_zero(universe: Iterable[str], src: dict[str, int]) -> dict[str, int]:
            return {k: int(src.get(k, 0)) for k in universe}

        def _match_sublevel(n_map: dict[str, int], G: list[str]) -> dict[str, int] | None:
            n_keys = set(n_map.keys())
            result: dict[str, int] = {}
            any_match = False
            for g in G:
                parts = set(g.split("__by__"))
                matches = parts & n_keys
                if len(matches) == 1:
                    result[g] = n_map[matches.pop()]
                    any_match = True
                elif len(matches) == 0:
                    result[g] = 0
                else:
                    return None
            return result if any_match else None

        if isinstance(n, (int, float)):
            return _broadcast(G, n) if G else int(n)

        n_map = _coerce(n)
        n_keys = set(n_map.keys())
        G_set, B_set, S_set = set(G), set(B), set(S)

        if G:
            if n_keys == G_set:
                return n_map
            if n_keys.issubset(G_set):
                return _fill_zero(G, n_map)
            if B and n_keys.issubset(B_set):
                bf = _fill_zero(B, n_map)
                return {f"{s}__by__{b}": bf[b] for s in S for b in B} if S else bf
            if S and n_keys.issubset(S_set):
                sf = _fill_zero(S, n_map)
                return {f"{s}__by__{b}": sf[s] for s in S for b in B} if B else sf
            sub = _match_sublevel(n_map, G)
            if sub is not None:
                return sub
            raise ValueError(
                "The keys of `n` must match either the combined groups (stratum×by), "
                "a subset of the by-levels, a subset of the stratum levels, "
                "or sub-component values of the combined group keys "
                '(e.g., {"Urban": 30, "Rural": 20}). '
                f"Got keys={sorted(n_keys)}, combined={G}, by={B}, stratum={S}."
            )
        else:
            return int(sum(n_map.values()))

    # ── add_stage ────────────────────────────────────────────────────────

    def add_stage(
        self,
        next_stage: "pl.DataFrame | pl.LazyFrame | Sample",
        *,
        prob_name: str | None = None,
        wgt_name: str | None = None,
    ) -> "Sample":
        """
        Chain a second-stage frame or selected sample onto the current stage.

        The combined ``Sample`` represents the second-stage population.
        Unconditional probabilities ($\\pi_{hij} = \\pi_{hi} \\times
        \\pi_{j|hi}$) are computed immediately when ``next_stage`` is a
        pre-selected ``Sample``, or deferred to the next selection call
        otherwise.

        Parameters
        ----------
        next_stage : pl.DataFrame | pl.LazyFrame | Sample
            Second-stage data. Three cases:

            * **DataFrame / LazyFrame** — bare frame. Stage-1 columns are
              joined on the stage-1 PSU column (must exist in the data).
              The returned ``Sample`` is ready for a subsequent selection.
            * **Sample without selection probs** — treated as a bare frame.
            * **Sample with selection probs** — selection already done.
              Combined probabilities computed immediately.

        prob_name : str | None
            Name for the combined probability column.
            Defaults to ``svy_prob_selection``.
        wgt_name : str | None
            Name for the combined weight column.
            Defaults to ``svy_sample_weight``.

        Returns
        -------
        Sample
            Combined ``Sample`` with stage-1 columns joined in.

        Notes
        -----
        **Returned Design:**

        * ``stratum``  — from stage-1 design
        * ``psu``      — from stage-1 design
        * ``ssu``      — from ``next_stage.design.psu`` (if Sample), else None
        * ``prob``     — ``svy_prob_selection_stage1`` (chaining trigger)
        * ``wgt``      — None; recomputed after stage-2 selection
        * ``hit``, ``mos``, ``pop_size``, ``rep_wgts`` — None
        * ``wr``       — False

        **Stage-1 column renaming (always applied):**

        * ``svy_prob_selection`` → ``svy_prob_selection_stage1``
        * ``svy_sample_weight``  → ``svy_sample_weight_stage1``
        * ``svy_certainty``      → ``svy_certainty_stage1`` (if present)
        * ``svy_number_of_hits`` → ``svy_number_of_hits_stage1`` (if present)

        All other stage-1 columns: kept as-is unless the name exists in
        ``next_stage``, in which case ``_stage1`` suffix is appended.
        ``svy_row_index`` and internal design columns are excluded.
        """
        from svy.core.design import Design as _Design
        from svy.core.sample import Sample as _Sample

        s1 = self._sample
        s1_design = s1._design

        if s1_design.prob is None:
            raise ValueError(
                "Current stage has no selection probabilities. "
                "Run a selection method (pps_sys, srs, etc.) before add_stage()."
            )
        if s1_design.psu is None:
            raise ValueError("Current stage Design has no PSU. Set psu= before add_stage().")

        if isinstance(next_stage, (pl.DataFrame, pl.LazyFrame)):
            ns_df = next_stage if isinstance(next_stage, pl.DataFrame) else next_stage.collect()
            ns_df = cast(pl.DataFrame, ns_df)
            ns_design = _Design()
            already_selected = False
        elif isinstance(next_stage, _Sample):
            ns_df = next_stage._data
            if isinstance(ns_df, pl.LazyFrame):
                ns_df = ns_df.collect()
            ns_df = cast(pl.DataFrame, ns_df)
            ns_design = next_stage._design
            already_selected = ns_design.prob is not None
        else:
            raise TypeError(
                f"next_stage must be pl.DataFrame, pl.LazyFrame, or Sample; "
                f"got {type(next_stage).__name__}"
            )

        s1_df = s1._data
        if isinstance(s1_df, pl.LazyFrame):
            s1_df = s1_df.collect()
        s1_df = cast(pl.DataFrame, s1_df)

        s1_psu = _psu_list(s1_design.psu)
        ns_psu = _psu_list(ns_design.psu) if ns_design.psu else s1_psu

        exclude: set[str] = {SVY_ROW_INDEX}
        internal = s1._internal_design or {}
        for key in ("stratum", "psu", "ssu"):
            val = internal.get(key)
            if isinstance(val, str):
                exclude.add(val)

        result: CombineResult = _combine_stages(
            stage1_df=s1_df,
            stage1_prob_col=s1_design.prob,
            stage1_wgt_col=s1_design.wgt,
            stage1_hit_col=s1_design.hit,
            stage1_psu=s1_psu,
            stage1_exclude_cols=exclude,
            next_df=ns_df,
            next_psu=ns_psu,
            next_prob_col=ns_design.prob if already_selected else None,
            next_wgt_col=ns_design.wgt if already_selected else None,
            already_selected=already_selected,
            out_prob_col=prob_name or SVY_PROB,
            out_wgt_col=wgt_name or SVY_WEIGHT,
        )

        combined_design = _Design(
            row_index=ns_design.row_index,
            stratum=s1_design.stratum,
            psu=s1_design.psu,
            ssu=ns_design.psu if ns_design.psu else None,
            wgt=result.out_wgt_col if result.already_selected else None,
            prob=result.chaining_prob_col,
            hit=ns_design.hit if already_selected else None,
            mos=None,
            pop_size=None,
            wr=False,
            rep_wgts=None,
        )

        combined = _Sample.__new__(_Sample)
        combined._data = result.data
        combined._design = combined_design
        combined._fpc = 1
        combined._metadata = copy.deepcopy(
            next_stage._metadata if isinstance(next_stage, _Sample) else s1._metadata
        )
        combined._internal_design = {
            "stratum": None,
            "psu": None,
            "ssu": None,
            "suffix": s1._internal_design.get("suffix", "_svy_internal_cols_concatenated"),
        }
        combined._warnings = (
            next_stage._warnings if isinstance(next_stage, _Sample) else s1._warnings
        )
        combined._print_width = None
        combined._singleton_result = None
        combined._stage_out_prob = result.out_prob_col
        combined._stage_out_wgt = result.out_wgt_col

        return combined

    # ── SRS ──────────────────────────────────────────────────────────────

    def srs(
        self,
        n: int | Mapping[Category, Number],
        *,
        by: str | Sequence[str] | None = None,
        wr: bool = False,
        order_by: str | Sequence[str] | None = None,
        order_type: Literal["ascending", "descending", "random"] = "ascending",
        wgt_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
    ) -> "Sample":
        """
        Simple Random Sample (SRS), optionally stratified by (stratum × by).

        Parameters
        ----------
        n          : sample size per group (scalar or mapping)
        by         : additional grouping beyond design stratum
        wr         : sample with replacement
        order_by   : column(s) to order the frame before selection.
                     Must be None when order_type="random".
        order_type : "ascending" | "descending" | "random".
                     Controls frame ordering before selection.
        rstate     : random state for reproducibility
        drop_nulls : drop rows with missing values before selection

        `n` semantics:
        - scalar -> broadcast to each (stratum×by) group
        - mapping keyed by combined (stratum×by) -> pass-through
        - mapping keyed by by-only -> broadcast within each stratum
        - mapping keyed by stratum-only -> broadcast across by
        """
        self._ensure_row_index()
        df: DF = self._sample._data
        assert isinstance(df, pl.DataFrame), "selection requires an eager DataFrame"
        design = self._sample._design

        cols: list[str] = design.specified_fields()
        cols += _colspec_to_list(by)
        cols += _colspec_to_list(
            [
                c
                for c in [
                    self._sample._internal_design["stratum"],
                    self._sample._internal_design["psu"],
                    self._sample._internal_design["ssu"],
                ]
                if isinstance(c, str)
            ]
        )
        # order_by column(s) must be in the slice so _apply_order can sort on them
        if order_by is not None:
            cols += _colspec_to_list(order_by)
        cols = self._sample._dedup_preserve_order(cols)

        df_reduced: DF = df.select(cols)
        names = (
            set(df_reduced.columns) if hasattr(df_reduced, "columns") else set(df_reduced.schema)
        )
        missing_cols = [c for c in cols if c not in names]
        if missing_cols:
            raise KeyError(f"missing columns {missing_cols}; available: {sorted(names)}")
        assert isinstance(df_reduced, pl.DataFrame), "selection requires an eager DataFrame"

        if drop_nulls:
            data: pl.DataFrame = drop_missing(
                df=df_reduced, cols=cols, treat_infinite_as_missing=True
            )
        else:
            assert_no_missing(df=df_reduced, subset=cols)
            data = df_reduced

        rng = resolve_random_state(rstate)
        seed = seed_from_random_state(rstate)

        data = _apply_order(data, order_by=order_by, order_type=order_type, seed=seed)

        suffix: str = cast(str, self._sample._internal_design["suffix"])
        stratum_col: str | None = self._sample._internal_design["stratum"]
        by_cols: list[str] = self._sample._to_cols(by) if by is not None else []

        (
            stratum_by_col,
            stratum_by_arr,
            G,
            B,
            S,
            data,
        ) = self._build_group_keys(
            data,
            stratum_col=stratum_col,
            by_cols=by_cols,
            suffix=suffix,
        )

        n_norm = self._normalize_n_for_groups(n, G=G, B=B, S=S)

        row_col = design.row_index or SVY_ROW_INDEX
        frame: npt.NDArray[np.int_] = data[row_col].to_numpy().astype(np.int_, copy=False)

        sel_idx, hits, probs = _select_srs(
            frame=frame,
            n=n_norm,
            stratum=stratum_by_arr,
            wr=wr,
            rstate=rng,
        )

        # ── write-back ───────────────────────────────────────────────────
        is_chaining = design.prob == SVY_PROB_STAGE1
        out_prob_col = getattr(self._sample, "_stage_out_prob", None) or SVY_PROB
        wgt_col = (
            wgt_name or getattr(self._sample, "_stage_out_wgt", None) or design.wgt or SVY_WEIGHT
        )
        hit_col = design.hit or SVY_HIT

        if is_chaining:
            assert design.prob is not None
            temp = _apply_chaining_writeback(
                src_df=df,
                sel_idx=sel_idx,
                hits=hits,
                probs=probs,
                certainty=None,
                row_col=row_col,
                prev_prob_col=design.prob,
                out_prob_col=out_prob_col,
                out_wgt_col=wgt_col,
                hit_col=hit_col,
                is_pps=False,
            )
            df_new = df.join(
                other=temp,
                left_on=design.row_index,
                right_on=row_col,
                how="inner",
            )
        else:
            prob_col = design.prob or SVY_PROB
            if design.prob is not None:
                prev = pl.DataFrame({row_col: sel_idx}).join(
                    other=data.select(row_col, prob_col),
                    on=row_col,
                    how="left",
                )
                prev_probs = prev[prob_col].fill_null(1.0).to_numpy().astype(np.float64)
                probs = probs * prev_probs

            design = design.fill_missing(prob=prob_col)
            temp = pl.DataFrame(
                {
                    row_col: sel_idx,
                    prob_col: probs.astype(np.float64, copy=False),
                    hit_col: hits.astype(np.int_, copy=False),
                }
            )
            df_new = df.join(
                other=temp,
                left_on=design.row_index,
                right_on=row_col,
                how="inner",
            ).with_columns((1.0 / pl.col(prob_col)).alias(wgt_col))
            out_prob_col = prob_col

        self._sample._data = df_new
        if is_chaining:
            self._sample._design = design.update(
                wgt=wgt_col,
                prob=out_prob_col,
                hit=hit_col,
                wr=wr,
            )
        else:
            self._sample._design = design.fill_missing(
                wgt=wgt_col,
                prob=out_prob_col,
                hit=hit_col,
                wr=wr,
            )
        for attr in ("_stage_out_prob", "_stage_out_wgt"):
            if hasattr(self._sample, attr):
                delattr(self._sample, attr)

        return self._sample

    # ── PPS public methods ───────────────────────────────────────────────

    def pps_sys(
        self,
        n: int | Mapping[Category, Number],
        *,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
        order_type: Literal["ascending", "descending", "random"] = "ascending",
        wgt_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
    ) -> "Sample":
        """
        PPS systematic sampling without replacement.

        Parameters
        ----------
        n                   : sample size per group (scalar or mapping)
        certainty_threshold : units with pi_i >= threshold selected with
                              certainty; must be in (0, 1], default 1.0
        by                  : additional grouping beyond design stratum
        order_by            : column(s) to order the frame before the
                              systematic grid is applied. Frame order affects
                              which units fall into each systematic interval.
                              Must be None when order_type="random".
        order_type          : "ascending" | "descending" | "random".
                              Controls frame ordering before the systematic
                              grid. "random" shuffles the frame first,
                              which is recommended for implicit stratification.
        rstate              : random state for reproducibility
        drop_nulls          : drop rows with missing values before selection
        """
        return self._pps(
            n=n,
            method=PPSMethod.SYS,
            certainty_threshold=certainty_threshold,
            by=by,
            wr=False,
            order_by=order_by,
            order_type=order_type,
            wgt_name=wgt_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
        )

    def pps_wr(
        self,
        n: int | Mapping[Category, Number],
        *,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        wgt_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
    ) -> "Sample":
        """PPS sampling with replacement."""
        return self._pps(
            n=n,
            method=PPSMethod.WR,
            certainty_threshold=certainty_threshold,
            by=by,
            wr=True,
            order_by=None,
            order_type="ascending",
            wgt_name=wgt_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
        )

    def pps_brewer(
        self,
        n: int | Mapping[Category, Number],
        *,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        wgt_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
    ) -> "Sample":
        """Brewer PPS sampling without replacement."""
        return self._pps(
            n=n,
            method=PPSMethod.BREWER,
            certainty_threshold=certainty_threshold,
            by=by,
            wr=False,
            order_by=None,
            order_type="ascending",
            wgt_name=wgt_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
        )

    def pps_murphy(
        self,
        n: int | Mapping[Category, Number],
        *,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        wgt_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
    ) -> "Sample":
        """Murphy PPS sampling without replacement (n=2 only)."""
        return self._pps(
            n=n,
            method=PPSMethod.MURPHY,
            certainty_threshold=certainty_threshold,
            by=by,
            wr=False,
            order_by=None,
            order_type="ascending",
            wgt_name=wgt_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
        )

    def pps_rs(
        self,
        n: int | Mapping[Category, Number],
        *,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        wgt_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
    ) -> "Sample":
        """Rao-Sampford PPS sampling without replacement."""
        return self._pps(
            n=n,
            method=PPSMethod.RS,
            certainty_threshold=certainty_threshold,
            by=by,
            wr=False,
            order_by=None,
            order_type="ascending",
            wgt_name=wgt_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
        )

    # ── _pps (internal) ──────────────────────────────────────────────────

    def _pps(
        self,
        n: int | Mapping[Category, Number],
        *,
        method: PPSMethod,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        wr: bool = False,
        order_by: str | Sequence[str] | None = None,
        order_type: Literal["ascending", "descending", "random"] = "ascending",
        wgt_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
    ) -> "Sample":
        """
        PPS selection with n broadcasting semantics.

        n semantics:
          - scalar -> each (stratum×by) group
          - mapping keyed by combined (stratum×by) -> pass-through
          - mapping keyed by by-only -> broadcast within each stratum
          - mapping keyed by stratum-only -> broadcast across by
          - mapping keyed by sub-level components -> auto-broadcast
        """
        self._ensure_row_index()

        src_df = self._sample._data
        assert isinstance(src_df, pl.DataFrame), "selection requires an eager DataFrame"
        design = self._sample._design

        cols = design.specified_fields()
        cols += _colspec_to_list(by)
        cols += _colspec_to_list(
            [
                c
                for c in [
                    self._sample._internal_design["stratum"],
                    self._sample._internal_design["psu"],
                    self._sample._internal_design["ssu"],
                    design.mos,
                ]
                if isinstance(c, str)
            ]
        )
        # order_by column(s) must be in the slice so _apply_order can sort on them
        if order_by is not None:
            cols += _colspec_to_list(order_by)
        cols = [c for c in cols if c is not None]
        cols = self._sample._dedup_preserve_order(cols)

        df = src_df.select(cols)
        names = set(df.columns) if hasattr(df, "columns") else set(df.schema)
        missing_cols = [c for c in cols if c not in names]
        if missing_cols:
            raise KeyError(f"missing columns {missing_cols}; available: {sorted(names)}")
        assert isinstance(df, pl.DataFrame), "selection requires an eager DataFrame"

        if drop_nulls:
            data: pl.DataFrame = drop_missing(df=df, cols=cols, treat_infinite_as_missing=True)
        else:
            assert_no_missing(df=df, subset=cols)
            data = df

        rng = resolve_random_state(rstate)
        seed = seed_from_random_state(rstate)

        data = _apply_order(data, order_by=order_by, order_type=order_type, seed=seed)

        suffix: str = cast(str, self._sample._internal_design["suffix"])
        stratum_col = self._sample._internal_design["stratum"]
        by_cols = self._sample._to_cols(by) if by is not None else []

        (
            stratum_by_col,
            stratum_by_arr,
            G,
            B,
            S,
            data,
        ) = self._build_group_keys(
            data,
            stratum_col=stratum_col,
            by_cols=by_cols,
            suffix=suffix,
        )

        n_norm = self._normalize_n_for_groups(n, G=G, B=B, S=S)

        row_col = design.row_index or SVY_ROW_INDEX
        if design.mos is None:
            raise ValueError("Missing mandatory MOS for PPS sampling. Please update the design!")
        mos_arr = data[design.mos].to_numpy()
        frame = data[row_col].to_numpy()

        sel_idx, hits, probs, certainty = _select_pps(
            method=method,
            frame=frame,
            n=n_norm,
            mos=mos_arr,
            stratum=stratum_by_arr,
            certainty_threshold=certainty_threshold,
            rstate=rng,
        )

        # ── write-back ───────────────────────────────────────────────────
        is_chaining = design.prob == SVY_PROB_STAGE1
        out_prob_col = getattr(self._sample, "_stage_out_prob", None) or SVY_PROB
        wgt_col = (
            wgt_name or getattr(self._sample, "_stage_out_wgt", None) or design.wgt or SVY_WEIGHT
        )
        hit_col = design.hit or SVY_HIT

        if is_chaining:
            assert design.prob is not None
            temp = _apply_chaining_writeback(
                src_df=src_df,
                sel_idx=sel_idx,
                hits=hits,
                probs=probs,
                certainty=certainty,
                row_col=row_col,
                prev_prob_col=design.prob,
                out_prob_col=out_prob_col,
                out_wgt_col=wgt_col,
                hit_col=hit_col,
                is_pps=True,
            )
            df_new = src_df.join(other=temp, on=row_col, how="inner").with_columns(
                pl.col(row_col).cast(pl.Int64)
            )
        else:
            prev_prob_col = design.prob
            if prev_prob_col is not None:
                prev = pl.DataFrame({row_col: sel_idx}).join(
                    other=src_df.select(row_col, prev_prob_col),
                    on=row_col,
                    how="left",
                )
                prev_probs = (
                    prev[prev_prob_col].fill_null(1.0).to_numpy().astype(np.float64, copy=False)
                )
                probs = probs * prev_probs

            design = design.fill_missing(prob=out_prob_col)

            temp = pl.DataFrame(
                {
                    row_col: sel_idx,
                    out_prob_col: probs.astype(np.float64, copy=False),
                    hit_col: hits.astype(np.int_, copy=False),
                    SVY_CERTAINTY: certainty,
                }
            ).cast(
                {
                    row_col: src_df[row_col].dtype,
                    out_prob_col: pl.Float64,
                    hit_col: pl.Int64,
                    SVY_CERTAINTY: pl.Boolean,
                }
            )

            df_new = src_df.join(other=temp, on=row_col, how="inner").with_columns(
                [
                    (1.0 / pl.col(out_prob_col)).alias(wgt_col),
                    pl.col(row_col).cast(pl.Int64),
                ]
            )

        self._sample._data = df_new
        if is_chaining:
            self._sample._design = design.update(
                wgt=wgt_col,
                prob=out_prob_col,
                hit=hit_col,
                wr=wr,
            )
        else:
            self._sample._design = design.fill_missing(
                wgt=wgt_col,
                prob=out_prob_col,
                hit=hit_col,
                wr=wr,
            )
        for attr in ("_stage_out_prob", "_stage_out_wgt"):
            if hasattr(self._sample, attr):
                delattr(self._sample, attr)

        return self._sample
