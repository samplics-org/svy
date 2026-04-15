# src/svy/selection/srs.py
"""
Simple Random Sampling (SRS) -- public method + internal engine glue.

Called from the Selection facade in base.py.  All SRS-specific logic
(column prep, group-key construction, write-back) lives here; the
combinatorial draw is delegated to svy.engine.sampling.srs._select_srs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from svy.core.constants import (
    SVY_HIT,
    SVY_PROB,
    SVY_PROB_STAGE1,
    SVY_ROW_INDEX,
    SVY_WEIGHT,
)
from svy.core.types import DF, Category, Number, WhereArg
from svy.engine.sampling.combine_stages import _apply_chaining_writeback
from svy.engine.sampling.srs import _select_srs
from svy.utils.checks import assert_no_missing, drop_missing
from svy.utils.helpers import _colspec_to_list
from svy.utils.random_state import RandomState, resolve_random_state, seed_from_random_state
from svy.utils.where import _compile_where
from svy.errors import MethodError

from svy.selection._group_keys import (
    _build_group_keys,
    _compute_pop_sizes,
    _normalize_n_for_groups,
)
from svy.selection._helpers import (
    _apply_order,
    _warn_empty_strata,
    _warn_n_exceeds_population,
)

if TYPE_CHECKING:
    from svy.core.sample import Sample

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row-index helper  (shared by pps.py via import)
# ---------------------------------------------------------------------------


def _ensure_row_index(sample) -> None:
    """Add SVY_ROW_INDEX to the frame and design if missing."""
    df = sample._data
    if SVY_ROW_INDEX not in df.columns:
        try:
            df = df.with_row_index(name=SVY_ROW_INDEX)
        except Exception:
            df = df.with_row_count(SVY_ROW_INDEX)
        sample._data = df
    if getattr(sample._design, "row_index", None) != SVY_ROW_INDEX:
        sample._design = sample._design.fill_missing(row_index=SVY_ROW_INDEX)


# ---------------------------------------------------------------------------
# Where-mask helper
# ---------------------------------------------------------------------------


def _apply_where(
    src_df: pl.DataFrame,
    where: WhereArg,
) -> tuple[pl.DataFrame, pl.Series | None]:
    """
    Split src_df into eligible and non-eligible rows.

    Parameters
    ----------
    src_df : pl.DataFrame
        Full source frame (all rows).
    where  : WhereArg
        User-supplied filter condition.  None means all rows eligible.

    Returns
    -------
    eligible : pl.DataFrame
        Rows where the condition is True -- these participate in the draw.
    mask : pl.Series | None
        Boolean Series aligned to src_df (True = eligible).
        None when where is None (all rows eligible, no splitting needed).
    """
    if where is None:
        return src_df, None

    expr = _compile_where(where)
    if expr is None:
        return src_df, None

    mask: pl.Series = src_df.select(expr.alias("__where__"))["__where__"]
    eligible = src_df.filter(mask)
    return eligible, mask


# ---------------------------------------------------------------------------
# Public SRS function
# ---------------------------------------------------------------------------


def srs(
    sample: "Sample",
    n: int | Mapping[Category, Number],
    *,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    wr: bool = False,
    order_by: str | Sequence[str] | None = None,
    order_type: Literal["ascending", "descending", "random"] = "ascending",
    wgt_name: str | None = None,
    rstate: RandomState = None,
    drop_nulls: bool = False,
) -> "Sample":
    """
    Simple Random Sample (SRS), optionally stratified by (stratum x by).

    Parameters
    ----------
    sample     : Sample instance (injected by the Selection facade)
    n          : sample size per group (scalar or mapping)
    by         : additional grouping beyond design stratum
    where      : row filter applied before selection.
                 Rows where the condition is True are eligible for the
                 draw; rows where it is False are kept in the output
                 with prob=null, weight=null, hit=null.
    wr         : sample with replacement
    order_by   : column(s) to sort the frame before selection
    order_type : "ascending" | "descending" | "random"
    rstate     : random state / seed
    drop_nulls : drop rows with missing values in design columns before
                 drawing (applied only within the eligible subset)

    ``n`` semantics
    ---------------
    scalar               -> broadcast to each (stratum x by) group
    mapping by combined  -> pass-through
    mapping by by-only   -> broadcast within each stratum
    mapping by stratum   -> broadcast across by

    Notes on ``where``
    ------------------
    The output frame always has the same row count as the input.
    Non-eligible rows (where=False) receive null for all selection
    columns (prob, weight, hit).  This correctly represents
    "not in scope for this selection stage" -- distinct from prob=0
    which would imply "eligible but never selected".
    """
    _ensure_row_index(sample)

    src_df: pl.DataFrame = sample._data
    assert isinstance(src_df, pl.DataFrame), "selection requires an eager DataFrame"
    design = sample._design

    # -- Apply where mask -------------------------------------------------
    eligible_df, where_mask = _apply_where(src_df, where)

    # -- Column slicing on eligible subset --------------------------------
    cols: list[str] = design.specified_fields()
    cols += _colspec_to_list(by)
    cols += _colspec_to_list(
        [c for c in [
            sample._internal_design["stratum"],
            sample._internal_design["psu"],
            sample._internal_design["ssu"],
        ] if isinstance(c, str)]
    )
    if order_by is not None:
        cols += _colspec_to_list(order_by)
    cols = sample._dedup_preserve_order(cols)

    df_reduced: DF = eligible_df.select(cols)
    names = set(df_reduced.columns) if hasattr(df_reduced, "columns") else set(df_reduced.schema)
    missing_cols = [c for c in cols if c not in names]
    if missing_cols:
        raise MethodError.invalid_choice(
            where="Sample.sampling.srs",
            param="columns",
            got=missing_cols,
            allowed=sorted(names),
            hint="Check that all design and by= columns exist in the data.",
        )
    assert isinstance(df_reduced, pl.DataFrame)

    if drop_nulls:
        data: pl.DataFrame = drop_missing(df=df_reduced, cols=cols, treat_infinite_as_missing=True)
    else:
        assert_no_missing(df=df_reduced, subset=cols)
        data = df_reduced

    rng = resolve_random_state(rstate)
    seed = seed_from_random_state(rstate)
    data = _apply_order(data, order_by=order_by, order_type=order_type, seed=seed)

    suffix: str = cast(str, sample._internal_design["suffix"])
    stratum_col: str | None = sample._internal_design["stratum"]
    by_cols: list[str] = sample._to_cols(by) if by is not None else []

    stratum_by_col, stratum_by_arr, G, B, S, data = _build_group_keys(
        data, stratum_col=stratum_col, by_cols=by_cols, suffix=suffix, sample_ref=sample
    )
    n_norm = _normalize_n_for_groups(n, G=G, B=B, S=S)

    pop_sizes = _compute_pop_sizes(data, stratum_by_col, G)
    _warn_empty_strata(n_norm, pop_sizes)
    _warn_n_exceeds_population(n_norm, pop_sizes, wr=wr)

    row_col = design.row_index or SVY_ROW_INDEX

    # Guard: if where filtered out all rows, return src_df with null selection columns.
    if len(data) == 0:
        return _srs_empty_writeback(
            sample, src_df, design, wgt_name=wgt_name, wr=wr,
        )

    frame: npt.NDArray[np.int_] = data[row_col].to_numpy().astype(np.int_, copy=False)

    sel_idx, hits, probs = _select_srs(
        frame=frame, n=n_norm, stratum=stratum_by_arr, wr=wr, rstate=rng
    )

    return _srs_writeback(
        sample, src_df, data, design, sel_idx, hits, probs,
        row_col=row_col, wgt_name=wgt_name, wr=wr, where_mask=where_mask,
    )



def _srs_empty_writeback(sample, src_df, design, *, wgt_name, wr):
    """
    Return src_df unchanged with null selection columns added.

    Called when where= filters out all rows so the eligible frame is empty.
    Rather than raising, we add null prob/weight/hit columns to preserve
    the invariant that the output always has the same rows as the input.
    """
    prob_col = design.prob or SVY_PROB
    wgt_col = wgt_name or design.wgt or SVY_WEIGHT
    hit_col = design.hit or SVY_HIT

    n = len(src_df)
    df_new = src_df.with_columns([
        pl.lit(None).cast(pl.Float64).alias(prob_col),
        pl.lit(None).cast(pl.Float64).alias(wgt_col),
        pl.lit(None).cast(pl.Int64).alias(hit_col),
    ])
    sample._data = df_new
    sample._design = design.fill_missing(prob=prob_col, wgt=wgt_col, hit=hit_col, wr=wr)
    return sample


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


def _srs_writeback(
    sample, src_df, data, design, sel_idx, hits, probs,
    *, row_col, wgt_name, wr, where_mask,
):
    """Merge SRS selection results back onto the Sample."""
    is_chaining = design.prob == SVY_PROB_STAGE1
    out_prob_col = getattr(sample, "_stage_out_prob", None) or SVY_PROB
    wgt_col = wgt_name or getattr(sample, "_stage_out_wgt", None) or design.wgt or SVY_WEIGHT
    hit_col = design.hit or SVY_HIT

    if is_chaining:
        assert design.prob is not None
        temp = _apply_chaining_writeback(
            src_df=src_df, sel_idx=sel_idx, hits=hits, probs=probs, certainty=None,
            row_col=row_col, prev_prob_col=design.prob,
            out_prob_col=out_prob_col, out_wgt_col=wgt_col, hit_col=hit_col, is_pps=False,
        )
        # left join so non-eligible rows stay with null selection columns
        join_how = "left" if where_mask is not None else "inner"
        df_new = src_df.join(other=temp, left_on=design.row_index, right_on=row_col, how=join_how)
    else:
        prob_col = design.prob or SVY_PROB
        if design.prob is not None:
            prev = pl.DataFrame({row_col: sel_idx}).join(
                other=data.select(row_col, prob_col), on=row_col, how="left"
            )
            prev_probs = prev[prob_col].fill_null(1.0).to_numpy().astype(np.float64)
            probs = probs * prev_probs
        design = design.fill_missing(prob=prob_col)
        temp = pl.DataFrame({
            row_col: sel_idx,
            prob_col: probs.astype(np.float64, copy=False),
            hit_col: hits.astype(np.int_, copy=False),
        })
        # left join: non-eligible rows get null for prob/hit/weight
        join_how = "left" if where_mask is not None else "inner"
        df_new = src_df.join(
            other=temp, left_on=design.row_index, right_on=row_col, how=join_how
        )
        if join_how == "left":
            # weight is 1/prob only where prob is not null
            df_new = df_new.with_columns(
                pl.when(pl.col(prob_col).is_not_null())
                .then(1.0 / pl.col(prob_col))
                .otherwise(None)
                .alias(wgt_col)
            )
        else:
            df_new = df_new.with_columns((1.0 / pl.col(prob_col)).alias(wgt_col))
        out_prob_col = prob_col

    sample._data = df_new
    if is_chaining:
        sample._design = design.update(wgt=wgt_col, prob=out_prob_col, hit=hit_col, wr=wr)
    else:
        sample._design = design.fill_missing(wgt=wgt_col, prob=out_prob_col, hit=hit_col, wr=wr)
    for attr in ("_stage_out_prob", "_stage_out_wgt"):
        if hasattr(sample, attr):
            delattr(sample, attr)
    return sample
