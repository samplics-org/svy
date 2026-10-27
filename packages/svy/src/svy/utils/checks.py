# src/svy/utils/checks.py
from __future__ import annotations

import math
import numbers

from typing import Any, Sequence, cast

import numpy as np
import polars as pl

from svy.core.types import Category
from svy.errors import Code, DataError, UsageError


# ------------------------------------------------------------------ #
# Parameter Checks
# ------------------------------------------------------------------ #


def validate_alpha(alpha: Any, *, where: str) -> float:
    """``alpha`` as a float strictly inside (0, 1); NaN, bools and strings are refused."""
    hint = "alpha is the significance level: 0.05 gives 95% confidence intervals."
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise UsageError.invalid_type(
            where=where, param="alpha", got=alpha, expected="a float in (0, 1)", hint=hint
        )
    if not 0.0 < float(alpha) < 1.0:
        raise UsageError.invalid_range(
            where=where, param="alpha", got=alpha, min_=0.0, max_=1.0, hint=hint
        )
    return float(alpha)


# ------------------------------------------------------------------ #
# Numeric Conversion
# ------------------------------------------------------------------ #


def to_stringnumber(*, token: object) -> Category:
    """
    Robustly convert input to int, then float (if integer-like), else string.
    Optimized to reduce try/except overhead for common cases.
    """
    # 1. Booleans are ints in Python, handle explicitly first
    if isinstance(token, bool):
        return token

    # 2. Already numeric? Normalize floats like 1.0 -> 1
    if isinstance(token, (int, float, np.integer, np.floating)):
        if isinstance(token, (float, np.floating)):
            # is_integer() is fast on floats
            return int(token) if float(token).is_integer() else float(token)
        return int(token)

    # 3. String / Bytes processing
    if isinstance(token, (str, bytes, bytearray)):
        s_val = token.decode() if isinstance(token, (bytes, bytearray)) else token

        # Optimization: Check digits before try-except to avoid exception overhead
        # This handles positive integers "123" instantly
        if s_val.isdigit():
            return int(s_val)

        # Fallback to general parsing (handles negatives "-5", floats "1.5", etc.)
        try:
            f = float(s_val)
            if math.isfinite(f):
                return int(f) if f.is_integer() else f
        except ValueError:
            pass  # Not a number, return as string

        return s_val

    # 4. Fallback for other objects
    return str(token)


# ------------------------------------------------------------------ #
# Polars Missing Data Handling
# ------------------------------------------------------------------ #


def any_missing(df: pl.DataFrame, cols: Sequence[str], *, infinite: bool) -> bool:
    """Whether any of ``cols`` holds a NULL, a NaN or (with ``infinite``) a ±∞,
    from null counts and float buffers rather than a query."""
    for c in cols:
        s = df.get_column(c)
        if s.null_count():
            return True
        if s.dtype in (pl.Float32, pl.Float64):
            v = s.to_numpy()
            if (not np.isfinite(v).all()) if infinite else np.isnan(v).any():
                return True
    return False


def drop_missing(
    *,
    df: pl.DataFrame,
    cols: Sequence[str],
    treat_infinite_as_missing: bool = False,
    streaming: bool = False,
) -> pl.DataFrame:
    """
    Drop rows with NULLs in `cols`, plus NaNs (and optionally ±∞) for float columns.
    """
    if not cols:
        return df

    # Validate columns existence
    # Using set intersection is faster than list iteration for existence check
    df_cols = set(df.columns)
    missing = [c for c in cols if c not in df_cols]
    if missing:
        raise DataError.KeyError(
            title="Column(s) not found",
            detail=f"Columns not found: {missing}",
            code=Code.MISSING_COLUMNS,
            where="drop_missing",
            param="columns",
            got=missing,
        )
    if not any_missing(df, cols, infinite=treat_infinite_as_missing):
        return df

    # Build the lazy query
    lf = df.lazy().drop_nulls(subset=cols)

    # Float columns among the subset (only floats can have NaN/±∞)
    # We check schema once
    schema = df.schema
    float_cols = [c for c in cols if schema[c] in (pl.Float32, pl.Float64)]

    if float_cols:
        # Chain filters efficiently
        lf = lf.drop_nans(subset=float_cols)
        if treat_infinite_as_missing:
            # Combine infinite checks into one expression
            lf = lf.filter(pl.all_horizontal([pl.col(c).is_finite() for c in float_cols]))

    if streaming:
        return cast(pl.DataFrame, lf.collect(engine="streaming"))

    return cast(pl.DataFrame, lf.collect())


def assert_no_missing(
    *,
    df: pl.DataFrame,
    subset: Sequence[str],
    scope: str | None = None,
    hint: str = "Set drop_nulls=True to leave those rows out.",
) -> None:
    """
    Raise with a helpful message if any column in subset has NULL/NaN/±∞.
    Optimized to perform a SINGLE scan of the data rather than multiple.
    `scope` names the rows checked when they are not the whole frame.
    """
    if not subset or df.is_empty() or not any_missing(df, subset, infinite=True):
        return

    schema = df.schema
    check_exprs = []

    # 1. Build expressions for NULL checks
    for c in subset:
        check_exprs.append(pl.col(c).is_null().any().alias(f"{c}_null"))

    # 2. Build expressions for NaN/Inf checks (only valid for floats)
    float_cols = [c for c in subset if schema[c] in (pl.Float32, pl.Float64)]
    for c in float_cols:
        check_exprs.append(pl.col(c).is_nan().any().alias(f"{c}_nan"))
        check_exprs.append(pl.col(c).is_infinite().any().alias(f"{c}_inf"))

    if not check_exprs:
        return

    # 3. Execute ALL checks in a single pass (returns a 1-row DataFrame)
    # Using row(0) immediately materializes the result efficiently
    stats = df.select(check_exprs).row(0, named=True)

    # 4. Analyze results
    missing_errors = []

    # Check NULLs
    for c in subset:
        if stats[f"{c}_null"]:
            missing_errors.append(f"{c} (NULL)")

    # Check NaN/Inf
    for c in float_cols:
        if stats[f"{c}_nan"]:
            missing_errors.append(f"{c} (NaN)")
        if stats[f"{c}_inf"]:
            missing_errors.append(f"{c} (±∞)")

    if missing_errors:
        raise DataError.ValueError(
            title="Missing values",
            detail="Missing or invalid values found in required columns"
            + (f" {scope}" if scope else "")
            + f"; affected: {', '.join(missing_errors)}.",
            code=Code.MISSING_VALUES,
            got=missing_errors,
            hint=hint,
        )
