# python/svy_io/utils.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List
import polars as pl


# ───────────────────────── text helpers ─────────────────────────


def cat_line(*parts: Any) -> str:
    """
    Concatenate parts and append a single newline.

    OPTIMIZED: Direct string concatenation with join.
    """
    return "".join(str(p) for p in parts) + "\n"


# ───────────────────────── label helpers ─────────────────────────


def combine_labels(
    x_labels: Dict[Any, Any],
    y_labels: Dict[Any, Any],
) -> Dict[Any, Any]:
    """
    Merge two value-label dicts, preferring LHS (x) on conflicts.

    OPTIMIZED: Early exits, single dict construction.
    """
    # Fast paths
    if not x_labels:
        return dict(y_labels) if y_labels else {}
    if not y_labels:
        return dict(x_labels)

    # Merge: start with y, then overlay x (so x wins on conflicts)
    merged = dict(y_labels)
    merged.update(x_labels)
    return merged


# ───────────────────────── timezone helpers (Polars) ─────────────────────────


def force_utc(series: pl.Series) -> pl.Series:
    """
    Ensure Datetime series has UTC timezone.

    OPTIMIZED: Early type check, minimal branching.
    """
    dt = series.dtype
    if not isinstance(dt, pl.Datetime):
        return series  # Fast path for non-datetime

    tz = dt.time_zone
    if tz == "UTC":
        return series  # Already UTC
    if tz is None:
        return series.dt.replace_time_zone("UTC")
    return series.dt.convert_time_zone("UTC")


def adjust_tz(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply force_utc to all Datetime columns.

    OPTIMIZED: Single with_columns call, early exit.
    """
    # Collect all datetime column expressions
    exprs = [
        force_utc(df[name]).alias(name)
        for name, dt in df.schema.items()
        if isinstance(dt, pl.Datetime)
    ]

    # Single batch update or return unchanged
    return df.with_columns(exprs) if exprs else df


# ───────────────────────── selection helpers ─────────────────────────


def var_names(df: pl.DataFrame, i: int | Iterable[int]) -> List[str] | str:
    """
    R-like helper: get column name(s) by index.

    OPTIMIZED: Direct indexing, list comprehension.
    """
    cols = df.columns
    if isinstance(i, int):
        return cols[i]
    return [cols[idx] for idx in i]


def skip_cols(
    df: pl.DataFrame,
    select: Iterable[str] | None = None,
) -> List[str]:
    """
    Return columns NOT in select list.

    OPTIMIZED: Early exit, set membership (O(1) lookup).
    """
    if select is None:
        return []

    # Convert to set for O(1) lookup
    sel = set(select)
    return [c for c in df.columns if c not in sel]
