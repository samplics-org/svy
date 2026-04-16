# src/svy/wrangling/values.py
"""
Value-transformation operations: coding, recoding, categorizing,
casting, and null-filling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import polars as pl

from svy.engine.wrangling.cleaning import (
    _bottom_and_top_code,
    _bottom_code,
    _categorize,
    _recode,
    _top_code,
)
from svy.errors import DimensionError, MethodError
from svy.wrangling._helpers import (
    _eager_df,
    _rebuild_concat_if_touched,
    _resolve_target,
)


if TYPE_CHECKING:
    from svy.core.sample import Sample


def top_code(
    sample: "Sample",
    top_codes: Mapping[str, float],
    *,
    replace: bool = False,
    into: str | Mapping[str, str] | None = None,
    inplace: bool = False,
) -> "Sample":
    """Cap values at upper bounds (top coding)."""
    _df = _eager_df(sample)
    new_data = _top_code(_df, top_codes=top_codes, replace=replace, into=into)
    target = _resolve_target(sample, new_data, inplace=inplace)
    if replace:
        _rebuild_concat_if_touched(sample, target, set(top_codes.keys()))
    return target


def bottom_code(
    sample: "Sample",
    bottom_codes: Mapping[str, float],
    *,
    replace: bool = False,
    into: str | Mapping[str, str] | None = None,
    inplace: bool = False,
) -> "Sample":
    """Cap values at lower bounds (bottom coding)."""
    _df = _eager_df(sample)
    new_data = _bottom_code(_df, bottom_codes=bottom_codes, replace=replace, into=into)
    target = _resolve_target(sample, new_data, inplace=inplace)
    if replace:
        _rebuild_concat_if_touched(sample, target, set(bottom_codes.keys()))
    return target


def bottom_and_top_code(
    sample: "Sample",
    bottom_and_top_codes: Mapping[str, tuple[float, float] | list[float]],
    *,
    replace: bool = False,
    into: str | Mapping[str, str] | None = None,
    inplace: bool = False,
) -> "Sample":
    """Cap values at both lower and upper bounds."""
    _df = _eager_df(sample)
    new_data = _bottom_and_top_code(
        _df,
        bottom_and_top_codes=bottom_and_top_codes,
        replace=replace,
        into=into,
    )
    target = _resolve_target(sample, new_data, inplace=inplace)
    if replace:
        _rebuild_concat_if_touched(sample, target, set(bottom_and_top_codes.keys()))
    return target


def recode(
    sample: "Sample",
    cols: str | list[str],
    recodes: Mapping[Any, Sequence[Any]],
    *,
    replace: bool = False,
    into: Mapping[str, str] | str | None = None,
    inplace: bool = False,
) -> "Sample":
    """Map old values to new labels."""
    _df = _eager_df(sample)
    new_data = _recode(_df, cols=cols, recodes=recodes, replace=replace, into=into)
    target = _resolve_target(sample, new_data, inplace=inplace)
    if replace:
        touched = {cols} if isinstance(cols, str) else set(cols)
        _rebuild_concat_if_touched(sample, target, touched)
    return target


def categorize(
    sample: "Sample",
    col: str,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
    *,
    percentiles: int | tuple[float, ...] | None = None,
    right: bool = True,
    replace: bool = False,
    into: str | None = None,
    inplace: bool = False,
) -> "Sample":
    """Bin continuous values into labeled categories."""
    from svy.utils.quantiles import weighted_quantile_bins, weighted_quantiles

    # -- Validate: exactly one of bins / percentiles ---------------------
    if bins is not None and percentiles is not None:
        raise MethodError(
            title="Ambiguous bin specification",
            detail="Provide either 'bins' or 'percentiles', not both.",
            code="AMBIGUOUS_BINS",
            where="wrangling.categorize",
            param="bins / percentiles",
            hint="Use 'bins' for explicit edges or 'percentiles' to compute from data.",
        )
    if bins is None and percentiles is None:
        raise MethodError(
            title="No bin specification",
            detail="One of 'bins' or 'percentiles' must be provided.",
            code="MISSING_BINS",
            where="wrangling.categorize",
            param="bins / percentiles",
            hint="Use 'bins' for explicit edges or 'percentiles' to compute from data.",
        )

    # -- Compute bins from percentiles if needed -------------------------
    if percentiles is not None:
        _df = _eager_df(sample)

        if col not in _df.columns:
            raise DimensionError.missing_columns(
                where="wrangling.categorize",
                param="col",
                missing=[col],
                available=_df.columns,
            )

        wgt_col = getattr(sample._design, "wgt", None)
        if wgt_col and wgt_col in _df.columns:
            # Fix 2: single filter pass — extract both columns at once
            filtered = _df.filter(pl.col(col).is_not_null()).select([col, wgt_col])
            values = filtered[col].to_numpy().astype(float)
            weights = filtered[wgt_col].to_numpy().astype(float)
        else:
            values = _df[col].drop_nulls().to_numpy().astype(float)
            weights = None

        if isinstance(percentiles, int):
            bins = weighted_quantile_bins(values, percentiles, weights)
        else:
            cuts = weighted_quantiles(values, list(percentiles), weights)
            bins = [float("-inf")] + cuts + [float("inf")]

    # -- Validate labels vs bins -----------------------------------------
    n_bins = len(bins) - 1  # type: ignore[arg-type]
    if labels is not None and len(labels) != n_bins:
        raise MethodError(
            title="Label count mismatch",
            detail=f"Expected {n_bins} labels for {n_bins} bins, got {len(labels)}.",
            code="LABELS_LENGTH_MISMATCH",
            where="wrangling.categorize",
            param="labels",
            expected=n_bins,
            got=len(labels),
        )

    # -- Delegate to existing _categorize --------------------------------
    # Fix 3: reuse _df if already materialised, otherwise get it now
    _df_cat = _df if percentiles is not None else _eager_df(sample)
    new_data = _categorize(
        data=_df_cat,
        varname=col,
        bins=bins,  # type: ignore[arg-type]
        labels=labels,
        right=right,
        replace=replace,
        into=into,
    )
    target = _resolve_target(sample, new_data, inplace=inplace)
    if replace:
        _rebuild_concat_if_touched(sample, target, {col})
    return target


def cast_columns(
    sample: "Sample",
    cols: str | Sequence[str] | Mapping[str, pl.DataType],
    dtype: pl.DataType | None = None,
    *,
    strict: bool = True,
    inplace: bool = False,
) -> "Sample":
    """Cast columns to specified data type(s)."""

    def _cast_expr(col_name: str, target_dt: pl.DataType, strict: bool) -> pl.Expr:
        base = pl.col(col_name)
        if target_dt in (pl.Categorical, pl.Enum) or (
            isinstance(target_dt, pl.Categorical) or isinstance(target_dt, pl.Enum)
        ):
            return base.cast(pl.Utf8).cast(target_dt, strict=strict).alias(col_name)
        return base.cast(target_dt, strict=strict).alias(col_name)

    if isinstance(cols, Mapping):
        col_names = set(cols.keys())
        exprs = [_cast_expr(c, dt, strict) for c, dt in cols.items()]  # type: ignore[arg-type]
    else:
        if dtype is None:
            raise MethodError(
                title="dtype required when cols is not a mapping",
                detail="Provide dtype parameter or pass a dict mapping columns to types",
                code="CAST_DTYPE_REQUIRED",
                where="wrangling.cast",
            )
        col_list = [cols] if isinstance(cols, str) else list(cols)
        col_names = set(col_list)
        exprs = [_cast_expr(c, dtype, strict) for c in col_list]

    new_data = sample._data.with_columns(exprs)
    target = _resolve_target(sample, new_data, inplace=inplace)
    _rebuild_concat_if_touched(sample, target, col_names)
    return target


def fill_null(
    sample: "Sample",
    cols: str | Sequence[str],
    value: Any = None,
    *,
    strategy: Literal["forward", "backward", "mean", "min", "max", "zero", "one"] | None = None,
    inplace: bool = False,
) -> "Sample":
    """Fill null values in specified columns."""
    col_list = [cols] if isinstance(cols, str) else list(cols)

    if strategy is not None:
        exprs = [pl.col(c).fill_null(strategy=strategy).alias(c) for c in col_list]
    else:
        exprs = [pl.col(c).fill_null(value).alias(c) for c in col_list]

    new_data = sample._data.with_columns(exprs)
    target = _resolve_target(sample, new_data, inplace=inplace)
    _rebuild_concat_if_touched(sample, target, set(col_list))
    return target
