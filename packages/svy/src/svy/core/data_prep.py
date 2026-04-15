# src/svy/core/data_prep.py
"""
Unified data preparation for all svy Rust backend calls.

Centralizes the common operations that every public API (estimation and
categorical) needs before calling Rust:

  1. Materialize LazyFrame
  2. Column selection (only needed columns for efficiency)
  3. Missing value handling (drop or assert)
  4. Concatenated design columns (stratum, psu, ssu, by)
  5. Paired difference (y - y_pair)
  6. Weight column creation (if no design weight)
  7. Type casting (y→Float64, group/strata/psu→String)
  8. Singleton filtering
  9. FPC column computation
  10. Where clause → domain column + zero weights for non-domain

Both ``estimation.base.Estimation`` and ``categorical.base.Categorical``
call :func:`prepare_data` rather than maintaining their own prep logic.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import polars as pl

from svy.core.constants import _INTERNAL_CONCAT_SUFFIX
from svy.core.types import WhereArg
from svy.utils.checks import assert_no_missing, drop_missing


if TYPE_CHECKING:
    from svy.core.sample import Sample

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Return type
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PreparedData:
    """Result of :func:`prepare_data`."""

    df: pl.DataFrame
    y_col: str
    weight_col: str
    strata_col: str | None = None
    psu_col: str | None = None
    ssu_col: str | None = None
    fpc_col: str | None = None
    fpc_ssu_col: str | None = None
    domain_col: str | None = None
    domain_val: str | None = None
    by_col: str | None = None
    by_cols: list[str] = field(default_factory=list)
    singleton_method: str | None = None


# ═══════════════════════════════════════════════════════════════════════
# Where-clause column extraction
# ═══════════════════════════════════════════════════════════════════════


def extract_where_cols(where: WhereArg) -> list[str]:
    """Extract column names referenced by a where clause.

    Handles all WhereArg forms:
    - None → []
    - Mapping[str, Any] (dict) → list of keys
    - pl.Expr → expr.meta.root_names()
    - HasExpr (svy.col wrapper) → unwrap ._e then root_names()
    - Sequence[ExprLike] → union of all root_names()
    """
    if where is None:
        return []

    # Dict form: {"sex": 2, "region": "North"}
    if isinstance(where, Mapping):
        return list(where.keys())

    # svy Expr wrapper (HasExpr protocol): unwrap to pl.Expr
    if hasattr(where, "_e"):
        where = where._e

    # Single Polars expression
    if isinstance(where, pl.Expr):
        try:
            return where.meta.root_names()
        except Exception:
            return []

    # Sequence of expressions (combined by AND)
    if isinstance(where, Sequence) and not isinstance(where, (str, bytes)):
        cols: list[str] = []
        for w in where:
            if hasattr(w, "_e"):
                w = w._e
            if isinstance(w, pl.Expr):
                try:
                    cols.extend(w.meta.root_names())
                except Exception:
                    pass
            elif isinstance(w, Mapping):
                cols.extend(w.keys())
        return cols

    return []


# ═══════════════════════════════════════════════════════════════════════
# Main data preparation function
# ═══════════════════════════════════════════════════════════════════════


def prepare_data(
    sample: Sample,
    *,
    y: str,
    x: str | None = None,
    group: str | None = None,
    y_pair: str | None = None,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    extra_cols: list[str] | None = None,
    drop_nulls: bool,
    cast_y_float: bool,
    select_columns: bool,
    apply_singleton_filter: bool,
) -> PreparedData:
    """
    Unified data preparation for all Rust backend calls.

    Parameters
    ----------
    sample : Sample
        The survey sample object.
    y : str
        Primary variable name.
    x : str | None
        Secondary variable (e.g. denominator for ratio estimation).
    group : str | None
        Grouping variable (two-sample tests).
    y_pair : str | None
        Second variable for paired tests.
    by : str | Sequence[str] | None
        Domain / stratification variable(s).
    where : WhereArg
        Subpopulation filter.
    extra_cols : list[str] | None
        Additional columns to include (e.g. colvar for tabulate).
    drop_nulls : bool
        If True, drop rows with missing values. Required — must be explicit.
    cast_y_float : bool
        If True, cast y to Float64. Set False for categorical y (tabulate). Required — must be explicit.
    select_columns : bool
        If True, select only needed columns for efficiency. Required — must be explicit.
    apply_singleton_filter : bool
        If True, apply singleton stratum exclusion filter. Required — must be explicit.

    Returns
    -------
    PreparedData
        Dataclass with all prepared data and column names.
    """
    # ── Materialize ──────────────────────────────────────────────────────
    _raw = sample._data
    local_data: pl.DataFrame = (
        cast(pl.DataFrame, _raw)
        if not isinstance(_raw, pl.LazyFrame)
        else cast(pl.DataFrame, _raw.collect())
    )
    design = sample._design

    # ── Build analysis-relevant column list ─────────────────────────────
    # This list is ALWAYS built, regardless of select_columns, because:
    #   1. drop_nulls/assert_no_missing should only check columns involved
    #      in the analysis, not every column in the dataset.
    #   2. select_columns controls whether the dataframe is subsetted for
    #      efficiency — it should never change analytical results.
    needed = [y]
    if x:
        needed.append(x)
    if y_pair:
        needed.append(y_pair)
    if group:
        needed.append(group)
    if isinstance(by, str):
        needed.append(by)
    elif isinstance(by, (list, tuple)):
        needed.extend(by)
    if extra_cols:
        needed.extend(extra_cols)
    needed.extend(extract_where_cols(where))
    needed.extend(design.specified_fields())
    # Include singleton variance/exclude columns if present in data
    _sr_pre = getattr(sample, "_singleton_result", None)
    _sc_pre = _sr_pre.config if _sr_pre else None
    if _sc_pre:
        for _col in [_sc_pre.var_stratum_col, _sc_pre.var_psu_col, _sc_pre.var_exclude_col]:
            if _col and _col in local_data.columns:
                needed.append(_col)
    needed = sample._dedup_preserve_order(needed)

    # ── Missing handling ─────────────────────────────────────────────────
    # Applied BEFORE column selection so that the same rows are dropped
    # regardless of select_columns. Only analysis-relevant columns are
    # checked — irrelevant columns (e.g., replicate weights) are ignored.
    if drop_nulls:
        local_data = drop_missing(
            df=local_data,
            cols=needed,
            treat_infinite_as_missing=True,
        )
    else:
        assert_no_missing(df=local_data, subset=needed)

    # ── Column selection (optional optimization) ─────────────────────────
    if select_columns:
        local_data = local_data.select(needed)

    # ── Singleton filter ─────────────────────────────────────────────────
    if apply_singleton_filter:
        from svy.core.singleton import _VAR_EXCLUDE_COL

        if _VAR_EXCLUDE_COL in local_data.columns:
            local_data = local_data.filter(~pl.col(_VAR_EXCLUDE_COL))

    # ── Concatenated design columns ──────────────────────────────────────
    _cc, _ = sample._create_concatenated_cols_from_lists(
        data=local_data,
        design=design,
        by=by,
        null_token="__Null__",
        suffix=_INTERNAL_CONCAT_SUFFIX,
        categorical=True,
        drop_original=False,
    )
    df: pl.DataFrame = cast(pl.DataFrame, _cc)

    # ── Design column names ──────────────────────────────────────────────
    # Use singleton variance columns if present (scale/center/collapse methods
    # create __svy_var_stratum__ / __svy_var_psu__ as the correct variance cols).
    # Otherwise use the freshly created concatenated columns from the current
    # design — NOT the stale _internal_design from Sample.__init__, which may
    # reference pre-rename column names.
    _singleton_result = getattr(sample, "_singleton_result", None)
    _singleton_config = _singleton_result.config if _singleton_result else None

    suffix = _INTERNAL_CONCAT_SUFFIX

    if _singleton_config and _singleton_config.var_stratum_col:
        strata_col = _singleton_config.var_stratum_col
        psu_col = _singleton_config.var_psu_col
    else:
        strata_col = f"stratum{suffix}" if design.stratum else None
        psu_col = f"psu{suffix}" if design.psu else None

    ssu_col = f"ssu{suffix}" if design.ssu else None

    # ── By column resolution ─────────────────────────────────────────────
    if isinstance(by, str):
        by_cols_list = [by]
        by_col = f"by{_INTERNAL_CONCAT_SUFFIX}"
    elif isinstance(by, (list, tuple)) and len(by) > 0:
        by_cols_list = list(by)
        by_col = f"by{_INTERNAL_CONCAT_SUFFIX}"
    else:
        by_cols_list = []
        by_col = None

    # Validate: by and where cannot reference the same column
    if by_cols_list and where is not None:
        where_cols = set(extract_where_cols(where))
        by_cols_set = set(by_cols_list)
        overlap = by_cols_set & where_cols
        if overlap:
            raise ValueError(
                f"Column(s) {sorted(overlap)} appear in both 'by' and 'where'. "
                f"Use 'by' to stratify or 'where' to restrict, not both on the same variable."
            )

    # ── Paired difference ────────────────────────────────────────────────
    y_col = y
    if y_pair:
        diff_name = f"__svy_{y}_minus_{y_pair}__"
        df = df.with_columns(
            (pl.col(y).cast(pl.Float64) - pl.col(y_pair).cast(pl.Float64)).alias(diff_name)
        )
        y_col = diff_name

    # ── Weight column ────────────────────────────────────────────────────
    weight_col = design.wgt if design.wgt else "__svy_ones__"
    if not design.wgt:
        df = df.with_columns(pl.lit(1.0).alias(weight_col))
    else:
        # Cast to Float64 — Rust backend requires it
        if df[weight_col].dtype != pl.Float64:
            df = df.with_columns(pl.col(weight_col).cast(pl.Float64))

    # ── Where clause → domain column + zero weights ──────────────────────
    # Must happen BEFORE type casting so where expressions reference original
    # column types (e.g. pl.col("sex") == 2 where sex is still Int32).
    domain_col = None
    domain_val = None
    if where is not None:
        where_expr = sample.estimation._compile_where_expr(where)
        df = df.with_columns(
            where_expr.cast(pl.String).alias("__svy_domain__"),
            # Zero weights for non-domain observations.
            # This ensures degrees_of_freedom, taylor_variance, and all
            # downstream Rust functions see only domain-active observations,
            # matching R's subset() behavior.
            pl.when(where_expr).then(pl.col(weight_col)).otherwise(0.0).alias(weight_col),
        )
        domain_col = "__svy_domain__"
        domain_val = "true"

    # ── Type casting ─────────────────────────────────────────────────────
    casts = []
    if cast_y_float:
        casts.append(pl.col(y_col).cast(pl.Float64))
    if x:
        casts.append(pl.col(x).cast(pl.Float64))
    if group:
        casts.append(pl.col(group).cast(pl.String))
    if strata_col and strata_col in df.columns:
        if df[strata_col].dtype != pl.String:
            casts.append(pl.col(strata_col).cast(pl.String))
    if psu_col and psu_col in df.columns:
        if df[psu_col].dtype != pl.String:
            casts.append(pl.col(psu_col).cast(pl.String))
    if ssu_col and ssu_col in df.columns:
        if df[ssu_col].dtype != pl.String:
            casts.append(pl.col(ssu_col).cast(pl.String))
    if by_col and by_col in df.columns:
        if df[by_col].dtype != pl.String:
            casts.append(pl.col(by_col).cast(pl.String))
    if casts:
        df = df.with_columns(casts)

    # ── Singleton method ─────────────────────────────────────────────────
    singleton_method = getattr(design, "singleton_method", None)

    return PreparedData(
        df=df,
        y_col=y_col,
        weight_col=weight_col,
        strata_col=strata_col,
        psu_col=psu_col,
        ssu_col=ssu_col,
        fpc_col=None,  # FPC computed separately by estimation facade
        fpc_ssu_col=None,  # when needed (not all APIs use FPC)
        domain_col=domain_col,
        domain_val=domain_val,
        by_col=by_col,
        by_cols=by_cols_list,
        singleton_method=singleton_method,
    )
