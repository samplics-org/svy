# src/svy/selection/pps.py
"""
Probability-Proportional-to-Size (PPS) sampling.

Public variants: pps_sys, pps_wr, pps_brewer, pps_murphy, pps_rs.
All dispatch to the shared _pps() engine function.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, Literal, Mapping, NamedTuple, Sequence, cast

import numpy as np
import polars as pl

from svy_rs import select_pps_rs as _select_pps_rs


try:
    from svy_rs import select_pps_batch_rs as _select_pps_batch_rs
except ImportError:
    _select_pps_batch_rs = None

from svy.core.constants import (
    SVY_CERTAINTY,
    SVY_HIT,
    SVY_PROB,
    SVY_PROB_STAGE1,
    SVY_ROW_INDEX,
    SVY_WEIGHT,
)
from svy.core.enumerations import PPSMethod
from svy.core.types import Category, Number, WhereArg
from svy.errors import SelectionError
from svy.selection._group_keys import (
    _build_group_keys,
    _compute_pop_sizes,
    _normalize_n_for_groups,
)
from svy.selection._helpers import (
    _apply_order,
    _child_seeds,
    _warn_empty_strata,
    _warn_n_exceeds_population,
    _warn_zero_mos,
)
from svy.selection.combine_stages import _apply_chaining_writeback
from svy.selection.srs import (
    _apply_where,
    _check_output_col_names,
    _encode_stratum,
    _ensure_row_index,
    _remap_n_map,
)
from svy.utils.checks import assert_no_missing, drop_missing
from svy.utils.helpers import _colspec_to_list
from svy.utils.random_state import (
    RandomState,
    advance_generator,
    resolve_random_state,
    seed_from_random_state,
)


_PPS_METHOD = {
    PPSMethod.SYS: "sys",
    PPSMethod.WR: "wr",
    PPSMethod.BREWER: "brewer",
    PPSMethod.MURPHY: "murphy",
    PPSMethod.RS: "rs",
}


def _pps_kernel_args(*, method, frame, n, mos, stratum, certainty_threshold) -> dict:
    """The Rust kernel's arguments other than the seed, shared by every draw."""
    if isinstance(n, dict):
        n_scalar = None
        n_map = {str(k): int(v) for k, v in n.items()}
    else:
        n_scalar = int(n)
        n_map = None
    strat_int, strat_codes = _encode_stratum(stratum)
    return {
        "frame": np.ascontiguousarray(frame, dtype=np.int64),
        "mos": np.ascontiguousarray(mos, dtype=np.float64),
        "n_scalar": n_scalar,
        "n_map": _remap_n_map(n_map, strat_int) if n_map is not None else None,
        "stratum": strat_codes,
        "method": _PPS_METHOD[method],
        # None takes nothing with certainty; _pps_prepare has refused any unit at 1.
        "certainty_threshold": 1.0 if certainty_threshold is None else certainty_threshold,
    }


def _pps_arrays(sel, hits, probs, cert) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(sel, dtype=np.int64),
        np.asarray(hits, dtype=np.int64),
        np.asarray(probs, dtype=np.float64),
        np.asarray(cert, dtype=bool),
    )


def _kernel_error(method: PPSMethod, e: ValueError) -> SelectionError:
    # Kernel-side validation (Murphy n != 2, Sampford n*p >= 1 or
    # degenerate acceptance, invalid MOS, ...) as a typed svy error.
    name = _PPS_METHOD[method]
    return SelectionError.not_applicable(
        where=f"Sample.selection.pps_{name}", method=f"pps_{name}", reason=str(e)
    )


def _refuse_units_at_one(method: PPSMethod, n, mos: np.ndarray, stratum: pl.Series | None) -> None:
    """Raise when a unit's n x share of size reaches 1: a draw cannot give it a probability above 1."""
    groups = stratum.cast(pl.String) if stratum is not None else pl.Series(["__all__"] * len(mos))
    agg = (
        pl.DataFrame({"g": groups, "m": mos})
        .group_by("g")
        .agg(pl.col("m").max().alias("top"), pl.col("m").sum().alias("total"))
    )
    over = []
    for g, top, total in agg.iter_rows():
        n_g = n if isinstance(n, int) else n.get(g, 0)
        # The kernel's own tolerance, so the two agree on a unit exactly at 1.
        if n_g > 0 and total > 0 and n_g * top / total >= 1.0 - 1e-12:
            over.append(g)
    if over:
        name = _PPS_METHOD[method]
        raise SelectionError.not_applicable(
            where=f"Sample.sampling.pps_{name}",
            method=f"pps_{name}",
            reason=(
                f"in {', '.join(map(repr, sorted(over)[:5]))}"
                f"{' and others' if len(over) > 5 else ''}, a unit's size share times n reaches 1, "
                "so its selection probability would exceed 1"
            ),
            hint="Pass certainty_threshold=1.0 to take such units with certainty, or lower n.",
        )


def _select_pps(*, method, frame, n, mos, stratum, certainty_threshold, rstate):
    """One PPS draw from the Rust kernel."""
    args = _pps_kernel_args(
        method=method,
        frame=frame,
        n=n,
        mos=mos,
        stratum=stratum,
        certainty_threshold=certainty_threshold,
    )
    try:
        return _pps_arrays(*_select_pps_rs(**args, seed=seed_from_random_state(rstate)))
    except ValueError as e:
        raise _kernel_error(method, e) from None


def _select_pps_draws(*, method, frame, n, mos, stratum, certainty_threshold, seeds):
    """One PPS draw per seed; older svy-rs has no batch kernel and draws them in turn."""
    args = _pps_kernel_args(
        method=method,
        frame=frame,
        n=n,
        mos=mos,
        stratum=stratum,
        certainty_threshold=certainty_threshold,
    )
    try:
        if _select_pps_batch_rs is not None:
            draws = _select_pps_batch_rs(**args, seeds=seeds)
        else:
            draws = [_select_pps_rs(**args, seed=seed) for seed in seeds]
    except ValueError as e:
        raise _kernel_error(method, e) from None
    return [_pps_arrays(*d) for d in draws]


if TYPE_CHECKING:
    from svy.core.sample import Sample

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public variants -- each is a one-liner delegating to _pps
# ---------------------------------------------------------------------------


def pps_sys(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    certainty_threshold: float | None = None,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    order_by: str | Sequence[str] | None = None,
    order_type: Literal["ascending", "descending", "random"] = "ascending",
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
    n_draws: int | None = None,
) -> "Sample | list[Sample]":
    """PPS systematic sampling without replacement."""
    return _pps(
        sample,
        n,
        method=PPSMethod.SYS,
        certainty_threshold=certainty_threshold,
        by=by,
        where=where,
        wr=False,
        order_by=order_by,
        order_type=order_type,
        prob_name=prob_name,
        wgt_name=wgt_name,
        hit_name=hit_name,
        rstate=rstate,
        drop_nulls=drop_nulls,
        n_draws=n_draws,
    )


def pps_wr(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    certainty_threshold: float | None = None,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
    n_draws: int | None = None,
) -> "Sample | list[Sample]":
    """PPS sampling with replacement."""
    return _pps(
        sample,
        n,
        method=PPSMethod.WR,
        certainty_threshold=certainty_threshold,
        by=by,
        where=where,
        wr=True,
        order_by=None,
        order_type="ascending",
        prob_name=prob_name,
        wgt_name=wgt_name,
        hit_name=hit_name,
        rstate=rstate,
        drop_nulls=drop_nulls,
        n_draws=n_draws,
    )


def pps_brewer(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    certainty_threshold: float | None = None,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
    n_draws: int | None = None,
) -> "Sample | list[Sample]":
    """Brewer PPS sampling without replacement."""
    return _pps(
        sample,
        n,
        method=PPSMethod.BREWER,
        certainty_threshold=certainty_threshold,
        by=by,
        where=where,
        wr=False,
        order_by=None,
        order_type="ascending",
        prob_name=prob_name,
        wgt_name=wgt_name,
        hit_name=hit_name,
        rstate=rstate,
        drop_nulls=drop_nulls,
        n_draws=n_draws,
    )


def pps_murphy(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    certainty_threshold: float | None = None,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
    n_draws: int | None = None,
) -> "Sample | list[Sample]":
    """Murphy PPS sampling without replacement (n=2 only)."""
    return _pps(
        sample,
        n,
        method=PPSMethod.MURPHY,
        certainty_threshold=certainty_threshold,
        by=by,
        where=where,
        wr=False,
        order_by=None,
        order_type="ascending",
        prob_name=prob_name,
        wgt_name=wgt_name,
        hit_name=hit_name,
        rstate=rstate,
        drop_nulls=drop_nulls,
        n_draws=n_draws,
    )


def pps_rs(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    certainty_threshold: float | None = None,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
    n_draws: int | None = None,
) -> "Sample | list[Sample]":
    """Rao-Sampford PPS sampling without replacement."""
    return _pps(
        sample,
        n,
        method=PPSMethod.RS,
        certainty_threshold=certainty_threshold,
        by=by,
        where=where,
        wr=False,
        order_by=None,
        order_type="ascending",
        prob_name=prob_name,
        wgt_name=wgt_name,
        hit_name=hit_name,
        rstate=rstate,
        drop_nulls=drop_nulls,
        n_draws=n_draws,
    )


# ---------------------------------------------------------------------------
# Shared _pps engine
# ---------------------------------------------------------------------------


def _pps(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    method: PPSMethod,
    certainty_threshold: float | None = None,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    wr: bool = False,
    order_by: str | Sequence[str] | None = None,
    order_type: Literal["ascending", "descending", "random"] = "ascending",
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
    n_draws: int | None = None,
) -> "Sample | list[Sample]":
    """
    PPS selection with n broadcasting semantics.

    where semantics mirror those of srs() -- eligible rows participate in
    the draw; non-eligible rows are kept in the output with null selection
    columns.
    """
    kwargs = {
        "certainty_threshold": certainty_threshold,
        "by": by,
        "where": where,
        "wr": wr,
        "order_by": order_by,
        "prob_name": prob_name,
        "wgt_name": wgt_name,
        "hit_name": hit_name,
        "drop_nulls": drop_nulls,
    }
    if n_draws is not None:
        return _pps_draws(
            sample,
            n,
            method=method,
            n_draws=n_draws,
            order_type=order_type,
            rstate=rstate,
            **kwargs,
        )
    rstate = advance_generator(rstate)
    prep = _pps_prepare(
        sample,
        n,
        method=method,
        order_type=order_type,
        seed=seed_from_random_state(rstate),
        **kwargs,
    )
    sel_idx, hits, probs, certainty = _select_pps(
        method=method,
        frame=prep.frame,
        n=prep.n,
        mos=prep.mos,
        stratum=prep.stratum,
        certainty_threshold=certainty_threshold,
        rstate=resolve_random_state(rstate),
    )
    return _pps_writeback(
        sample,
        prep.src_df,
        prep.data,
        prep.design,
        sel_idx,
        hits,
        probs,
        certainty,
        row_col=SVY_ROW_INDEX,
        prob_name=prob_name,
        wgt_name=wgt_name,
        hit_name=hit_name,
        wr=wr,
        where_mask=prep.where_mask,
    )


class _PpsPrepared(NamedTuple):
    """What a draw needs from the frame, computed once for every draw."""

    src_df: pl.DataFrame
    data: pl.DataFrame
    design: Any
    where_mask: Any
    frame: np.ndarray
    n: int | dict[str, int]
    stratum: pl.Series | None
    mos: np.ndarray


def _pps_prepare(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    method: PPSMethod,
    certainty_threshold: float | None,
    by: str | Sequence[str] | None,
    where: WhereArg,
    wr: bool,
    order_by: str | Sequence[str] | None,
    order_type: Literal["ascending", "descending", "random"],
    prob_name: str | None,
    wgt_name: str | None,
    hit_name: str | None,
    drop_nulls: bool,
    seed: int | None,
) -> _PpsPrepared:
    """Checks, eligible rows, order, groups and size measures."""
    _ensure_row_index(sample)

    src_df = sample._data
    assert isinstance(src_df, pl.DataFrame), "selection requires an eager DataFrame"
    design = sample._design

    # -- Apply where mask -------------------------------------------------
    eligible_df, where_mask = _apply_where(src_df, where)

    # -- Guard: reject names that already exist in the frame -------------
    _check_output_col_names(
        src_df,
        prob_name=prob_name,
        wgt_name=wgt_name,
        hit_name=hit_name,
        where="Sample.sampling.pps",
    )

    cols = [SVY_ROW_INDEX, *design.specified_fields()]
    cols += _colspec_to_list(by)
    cols += _colspec_to_list(
        [
            c
            for c in [
                sample._internal_design["stratum"],
                sample._internal_design["psu"],
                sample._internal_design["ssu"],
                design.mos,
            ]
            if isinstance(c, str)
        ]
    )
    if order_by is not None:
        cols += _colspec_to_list(order_by)
    cols = [c for c in cols if c is not None]
    cols = sample._dedup_preserve_order(cols)

    df = eligible_df.select(cols)
    names = set(df.columns) if hasattr(df, "columns") else set(df.schema)
    missing_cols = [c for c in cols if c not in names]
    if missing_cols:
        raise SelectionError.invalid_choice(
            where="Sample.sampling.pps",
            param="columns",
            got=missing_cols,
            allowed=sorted(names),
            hint="Check that all design and by= columns exist in the data.",
        )
    assert isinstance(df, pl.DataFrame)

    if drop_nulls:
        data: pl.DataFrame = drop_missing(df=df, cols=cols, treat_infinite_as_missing=True)
    else:
        assert_no_missing(df=df, subset=cols)
        data = df

    data = _apply_order(data, order_by=order_by, order_type=order_type, seed=seed)

    stratum_col = sample._internal_design["stratum"]
    by_cols = sample._to_cols(by) if by is not None else []

    stratum_by_col, stratum_by_arr, G, B, S, data = _build_group_keys(
        data, stratum_col=stratum_col, by_cols=by_cols, sample_ref=sample
    )
    n_norm = _normalize_n_for_groups(n, G=G, B=B, S=S)

    pop_sizes = _compute_pop_sizes(data, stratum_by_col, G)
    _warn_empty_strata(n_norm, pop_sizes, sample)
    _warn_n_exceeds_population(n_norm, pop_sizes, wr=wr, pps=True, sample=sample)

    if design.mos is None:
        raise SelectionError.not_applicable(
            where="Sample.sampling.pps",
            method="pps",
            reason="Missing mandatory MOS for PPS sampling.",
            hint="Set mos= on the Design before calling any pps_* method.",
        )
    mos_arr = data[design.mos].to_numpy()
    _warn_zero_mos(mos_arr, stratum_by_arr, drop_nulls=drop_nulls, sample=sample)
    if certainty_threshold is None and method in (PPSMethod.SYS, PPSMethod.BREWER, PPSMethod.RS):
        _refuse_units_at_one(method, n_norm, mos_arr, stratum_by_arr)

    row_col = SVY_ROW_INDEX
    return _PpsPrepared(
        src_df, data, design, where_mask, data[row_col].to_numpy(), n_norm, stratum_by_arr, mos_arr
    )


def _pps_draws(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    method: PPSMethod,
    n_draws: int,
    order_type: Literal["ascending", "descending", "random"],
    rstate: RandomState,
    **kwargs: Any,
) -> "list[Sample]":
    """``n_draws`` samples; the input is left as it was."""
    children = _child_seeds(rstate, n_draws, where=f"Sample.sampling.pps_{_PPS_METHOD[method]}")
    if order_type == "random":
        # The shuffle depends on the seed, so each draw prepares its own frame.
        return [
            cast(
                "Sample",
                _pps(sample._fork(), n, method=method, order_type=order_type, rstate=c, **kwargs),
            )
            for c in children
        ]
    base = sample._fork()
    prep = _pps_prepare(base, n, method=method, order_type=order_type, seed=None, **kwargs)
    seeds = [seed_from_random_state(resolve_random_state(c)) for c in children]
    draws = _select_pps_draws(
        method=method,
        frame=prep.frame,
        n=prep.n,
        mos=prep.mos,
        stratum=prep.stratum,
        certainty_threshold=kwargs["certainty_threshold"],
        seeds=seeds,
    )
    return [
        _pps_writeback(
            base._fork(),
            prep.src_df,
            prep.data,
            prep.design,
            sel_idx,
            hits,
            probs,
            certainty,
            row_col=SVY_ROW_INDEX,
            prob_name=kwargs["prob_name"],
            wgt_name=kwargs["wgt_name"],
            hit_name=kwargs["hit_name"],
            wr=kwargs["wr"],
            where_mask=prep.where_mask,
        )
        for sel_idx, hits, probs, certainty in draws
    ]


def _pps_writeback(
    sample,
    src_df,
    data,
    design,
    sel_idx,
    hits,
    probs,
    certainty,
    *,
    row_col,
    prob_name,
    wgt_name,
    hit_name,
    wr,
    where_mask,
):
    """Merge PPS selection results back onto the Sample."""
    is_chaining = design.prob == SVY_PROB_STAGE1
    out_prob_col = prob_name or getattr(sample, "_stage_out_prob", None) or SVY_PROB
    wgt_col = wgt_name or getattr(sample, "_stage_out_wgt", None) or design.wgt or SVY_WEIGHT
    hit_col = hit_name or design.hit or SVY_HIT
    join_how = "left" if where_mask is not None else "inner"

    if is_chaining:
        temp = _apply_chaining_writeback(
            src_df=src_df,
            sel_idx=sel_idx,
            hits=hits,
            probs=probs,
            certainty=certainty,
            row_col=row_col,
            out_prob_col=out_prob_col,
            out_wgt_col=wgt_col,
            hit_col=hit_col,
            is_pps=True,
        )
        df_new = src_df.join(
            other=temp, on=row_col, how=join_how, maintain_order="left"
        ).with_columns(pl.col(row_col).cast(pl.Int64))
    else:
        prev_prob_col = design.prob
        if prev_prob_col is not None:
            row_arr = src_df[row_col].to_numpy()
            prob_arr = src_df[prev_prob_col].to_numpy().astype(np.float64)
            order = np.argsort(row_arr, kind="stable")
            positions = np.searchsorted(row_arr[order], sel_idx)
            prev_probs = prob_arr[order[positions]]
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
        df_new = src_df.join(
            other=temp, on=row_col, how=join_how, maintain_order="left"
        ).with_columns(pl.col(row_col).cast(pl.Int64))
        if join_how == "left":
            df_new = df_new.with_columns(
                pl.when(pl.col(out_prob_col).is_not_null())
                .then(1.0 / pl.col(out_prob_col))
                .otherwise(None)
                .alias(wgt_col)
            )
        else:
            df_new = df_new.with_columns((1.0 / pl.col(out_prob_col)).alias(wgt_col))

    sample._data = df_new
    if is_chaining:
        sample._design = design.update(wgt=wgt_col, prob=out_prob_col, hit=hit_col, wr=wr)
    else:
        sample._design = design.fill_missing(wgt=wgt_col, prob=out_prob_col, hit=hit_col, wr=wr)
    for attr in ("_stage_out_prob", "_stage_out_wgt"):
        if hasattr(sample, attr):
            delattr(sample, attr)
    return sample
