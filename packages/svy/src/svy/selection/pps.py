# src/svy/selection/pps.py
"""
Probability-Proportional-to-Size (PPS) sampling.

Public variants: pps_sys, pps_wr, pps_brewer, pps_murphy, pps_rs.
All dispatch to the shared _pps() engine function.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Mapping, Sequence, cast

import numpy as np
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
from svy.core.types import DF, Category, Number, WhereArg
from svy.selection.combine_stages import _apply_chaining_writeback
from svy_rs import select_pps_rs as _select_pps_rs


def _select_pps(*, method, frame, n, mos, stratum, certainty_threshold, rstate):
    """Adapter: translate old Python engine calling convention to Rust."""
    import numpy as np
    from svy.core.enumerations import PPSMethod
    from svy.utils.random_state import seed_from_random_state

    method_str = {
        PPSMethod.SYS: "sys",
        PPSMethod.WR: "wr",
        PPSMethod.BREWER: "brewer",
        PPSMethod.MURPHY: "murphy",
        PPSMethod.RS: "rs",
    }[method]

    seed = seed_from_random_state(rstate)

    if isinstance(n, dict):
        n_scalar = None
        n_map = {str(k): int(v) for k, v in n.items()}
    else:
        n_scalar = int(n)
        n_map = None

    # stratum_by_arr contains the composite group key strings (e.g. "North__by__urban")
    # encoded as object dtype numpy array; convert to i64 index for Rust
    strat_int, strat_list = _encode_stratum(stratum)

    sel, hits, probs, cert = _select_pps_rs(
        frame=frame.tolist(),
        mos=mos.tolist(),
        n_scalar=n_scalar,
        n_map=(_remap_n_map(n_map, strat_int) if n_map is not None else None),
        stratum=strat_list,
        method=method_str,
        certainty_threshold=certainty_threshold,
        seed=seed,
    )
    return (
        np.asarray(sel, dtype=np.int64),
        np.asarray(hits, dtype=np.int64),
        np.asarray(probs, dtype=np.float64),
        np.asarray(cert, dtype=bool),
    )


from svy.utils.checks import assert_no_missing, drop_missing
from svy.utils.helpers import _colspec_to_list
from svy.utils.random_state import RandomState, resolve_random_state, seed_from_random_state

from svy.selection._group_keys import (
    _build_group_keys,
    _compute_pop_sizes,
    _normalize_n_for_groups,
)
from svy.selection._helpers import (
    _apply_order,
    _warn_empty_strata,
    _warn_n_exceeds_population,
    _warn_zero_mos,
)
from svy.selection.srs import (
    _apply_where,
    _check_output_col_names,
    _encode_stratum,
    _ensure_row_index,
    _remap_n_map,
)
from svy.errors import MethodError

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
    certainty_threshold: float = 1.0,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    order_by: str | Sequence[str] | None = None,
    order_type: Literal["ascending", "descending", "random"] = "ascending",
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
) -> "Sample":
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
    )


def pps_wr(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    certainty_threshold: float = 1.0,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
) -> "Sample":
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
    )


def pps_brewer(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    certainty_threshold: float = 1.0,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
) -> "Sample":
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
    )


def pps_murphy(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    certainty_threshold: float = 1.0,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
) -> "Sample":
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
    )


def pps_rs(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    certainty_threshold: float = 1.0,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    prob_name: str | None = None,
    wgt_name: str | None = None,
    hit_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
) -> "Sample":
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
    )


# ---------------------------------------------------------------------------
# Shared _pps engine
# ---------------------------------------------------------------------------


def _pps(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    method: PPSMethod,
    certainty_threshold: float = 1.0,
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
) -> "Sample":
    """
    PPS selection with n broadcasting semantics.

    where semantics mirror those of srs() -- eligible rows participate in
    the draw; non-eligible rows are kept in the output with null selection
    columns.
    """
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

    cols = design.specified_fields()
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
        raise MethodError.invalid_choice(
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

    rng = resolve_random_state(rstate)
    seed = seed_from_random_state(rstate)
    data = _apply_order(data, order_by=order_by, order_type=order_type, seed=seed)

    suffix: str = cast(str, sample._internal_design["suffix"])
    stratum_col = sample._internal_design["stratum"]
    by_cols = sample._to_cols(by) if by is not None else []

    stratum_by_col, stratum_by_arr, G, B, S, data = _build_group_keys(
        data, stratum_col=stratum_col, by_cols=by_cols, suffix=suffix, sample_ref=sample
    )
    n_norm = _normalize_n_for_groups(n, G=G, B=B, S=S)

    pop_sizes = _compute_pop_sizes(data, stratum_by_col, G)
    _warn_empty_strata(n_norm, pop_sizes)
    _warn_n_exceeds_population(n_norm, pop_sizes, wr=wr, pps=True)

    if design.mos is None:
        raise MethodError.not_applicable(
            where="Sample.sampling.pps",
            method="pps",
            reason="Missing mandatory MOS for PPS sampling.",
            hint="Set mos= on the Design before calling any pps_* method.",
        )
    mos_arr = data[design.mos].to_numpy()
    _warn_zero_mos(mos_arr, stratum_by_arr, drop_nulls=drop_nulls)

    row_col = design.row_index or SVY_ROW_INDEX
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

    return _pps_writeback(
        sample,
        src_df,
        data,
        design,
        sel_idx,
        hits,
        probs,
        certainty,
        row_col=row_col,
        prob_name=prob_name,
        wgt_name=wgt_name,
        hit_name=hit_name,
        wr=wr,
        where_mask=where_mask,
    )


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
        df_new = src_df.join(other=temp, on=row_col, how=join_how).with_columns(
            pl.col(row_col).cast(pl.Int64)
        )
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
        df_new = src_df.join(other=temp, on=row_col, how=join_how).with_columns(
            pl.col(row_col).cast(pl.Int64)
        )
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
