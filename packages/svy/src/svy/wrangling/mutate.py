# src/svy/wrangling/mutate.py
"""
Column creation and transformation via ``mutate()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import polars as pl

from svy.core.expr import Expr, to_polars_expr
from svy.core.types import MutateValue
from svy.errors import DimensionError, MethodError
from svy.wrangling._helpers import (
    _rebuild_concat_if_touched,
    _resolve_target,
)

if TYPE_CHECKING:
    from svy.core.sample import Sample


# Scalar types that can be broadcast to all rows via pl.lit()
SCALAR_TYPES = (bool, int, float, str)


# -------------------------------------------------------------------
# Value coercion helper
# -------------------------------------------------------------------


def _as_output_obj(obj: object, out_name: str, n_rows: int) -> object:
    """
    Convert various input types to Polars expressions or Series for
    mutate().

    Handles:
    - None: creates null column
    - bool, int, float, str: broadcast scalar to all rows via pl.lit()
    - pl.Expr: Polars expression
    - svy.Expr: svy expression (converted to Polars)
    - pl.Series: renamed if length matches
    - np.ndarray, list, tuple: converted to Series if length matches
    """
    if obj is None:
        return pl.lit(None).alias(out_name)

    if type(obj) in SCALAR_TYPES:
        return pl.lit(obj).alias(out_name)

    if isinstance(obj, pl.Expr):
        return obj.alias(out_name)

    if isinstance(obj, Expr):
        return to_polars_expr(obj).alias(out_name)

    if isinstance(obj, pl.Series):
        if obj.len() != n_rows:
            raise ValueError(
                f"Series length {obj.len()} != n_rows {n_rows} "
                f"for column '{out_name}'"
            )
        return obj.rename(out_name)

    try:
        is_np = isinstance(obj, np.ndarray)
    except Exception:
        is_np = False

    if is_np or isinstance(obj, (list, tuple)):
        vals = obj.tolist() if is_np else list(obj)  # type: ignore[arg-type]
        if len(vals) != n_rows:
            raise ValueError(
                f"Sequence length {len(vals)} != n_rows {n_rows} "
                f"for column '{out_name}'"
            )
        return pl.Series(out_name, vals)

    try:
        return to_polars_expr(obj).alias(out_name)
    except Exception as ex:
        raise TypeError(
            f"Unsupported type for column '{out_name}': "
            f"{type(obj).__name__}. "
            f"Expected: int, float, str, bool, None, Expr, pl.Expr, "
            f"pl.Series, np.ndarray, list, tuple, or callable."
        ) from ex


# -------------------------------------------------------------------
# Public function
# -------------------------------------------------------------------


def mutate(
    sample: "Sample",
    specs: Mapping[str, MutateValue],
    *,
    inplace: bool = False,
) -> "Sample":
    """Create or transform columns using expressions, scalars, or arrays."""
    _raw_ld = sample._data
    local_data: pl.DataFrame = (
        cast(pl.DataFrame, _raw_ld)
        if not isinstance(_raw_ld, pl.LazyFrame)
        else cast(pl.DataFrame, _raw_ld.collect())
    )
    n_rows = local_data.height
    existing_cols = set(local_data.columns)

    env = {name: pl.col(name) for name in local_data.columns}

    compiled: dict[str, tuple[object, set[str]]] = {}

    def _root_names_safe(expr: pl.Expr) -> set[str]:
        try:
            return set(expr.meta.root_names())
        except Exception:
            return set()

    for out_name, spec in specs.items():
        try:
            raw = spec(env) if callable(spec) else spec  # type: ignore[operator]
            out_obj = _as_output_obj(raw, out_name, n_rows)
        except Exception as ex:
            raise MethodError(
                title="Column transformation failed",
                detail=str(ex),
                code="MUTATE_COMPILE_FAILED",
                where="wrangling.mutate",
                hint=(
                    "Check names, syntax, that array/series lengths "
                    "match n_rows, and finalize when/then with "
                    ".otherwise(...)."
                ),
            ) from ex

        deps: set[str] = (
            _root_names_safe(out_obj)
            if isinstance(out_obj, pl.Expr)
            else set()
        )
        compiled[out_name] = (out_obj, deps)

    # Batch by readiness (topological sort)
    current_cols = set(local_data.columns)
    pending = set(compiled.keys())
    max_iters = len(pending) + 8

    for _ in range(max_iters):
        ready = [
            name
            for name in pending
            if compiled[name][1] <= current_cols
        ]
        if not ready:
            break

        batch_objs = [compiled[name][0] for name in ready]
        try:
            local_data = local_data.with_columns(batch_objs)
        except pl.exceptions.ColumnNotFoundError as ex:
            missing_by_out = {
                name: sorted(compiled[name][1] - current_cols)
                for name in ready
                if compiled[name][1] - current_cols
            }
            raise DimensionError.missing_columns(
                missing=sum(missing_by_out.values(), []),
                param="columns",
                where="wrangling.mutate",
            ) from ex
        except Exception as ex:
            raise MethodError(
                title="Column transformation failed",
                detail=str(ex),
                code="MUTATE_WITH_COLUMNS_FAILED",
                where="wrangling.mutate",
            ) from ex

        for name in ready:
            pending.remove(name)
            current_cols.add(name)

        if not pending:
            break

    if pending:
        unresolved = {
            name: sorted(compiled[name][1] - current_cols)
            for name in pending
        }
        missing_outside = sorted(
            {
                col
                for deps in unresolved.values()
                for col in deps
                if col not in pending and col not in current_cols
            }
        )

        if missing_outside:
            raise DimensionError.missing_columns(
                where="wrangling.mutate",
                param="specs",
                missing=missing_outside,
                available=local_data.columns,
            )

        raise MethodError(
            title="Dependency resolution failed",
            detail=f"Circular or mutually dependent outputs: {unresolved}",
            code="MUTATE_DEPENDENCY_ERROR",
            where="wrangling.mutate",
            hint=(
                "Split into multiple mutate() calls or break the "
                "circular reference."
            ),
        )

    target = _resolve_target(sample, local_data, inplace=inplace)

    # Detect which existing design-source columns were overwritten
    overwritten = set(specs.keys()) & existing_cols
    _rebuild_concat_if_touched(sample, target, overwritten)

    return target
