# src/svy/utils/checks.py
from __future__ import annotations

import math

from typing import Sequence, cast, overload

import numpy as np
import polars as pl

from svy.core.types import Category, RandomState


# ------------------------------------------------------------------ #
# Array Checks
# ------------------------------------------------------------------ #


def as_1d(*, a: np.ndarray, name: str) -> np.ndarray:
    """Ensure array is 1-D."""
    # Fast path: check dimension directly if it's already a numpy array
    if isinstance(a, np.ndarray):
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1-D.")
        return a

    # Conversion path
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D.")
    return arr


def as_float64_1d(*, a: np.ndarray, name: str) -> np.ndarray:
    """Ensure array is 1-D and float64."""
    # Combine checks to avoid redundant asarray calls
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D.")
    return arr.astype(float, copy=False)


def check_same_length(*arrays: tuple[np.ndarray, str]) -> None:
    """Verify all provided arrays have the same first dimension."""
    if not arrays:
        return

    # Check against the first array's length
    n = arrays[0][0].shape[0]

    for arr, name in arrays[1:]:
        if arr.shape[0] != n:
            # Only construct the error message on failure
            labels = ", ".join(f"{nm}={a.shape[0]}" for a, nm in arrays)
            raise ValueError(f"Arrays must share the same length: {labels}")


def check_weights_finite_positive(*, w: np.ndarray) -> None:
    """Verify weights are finite and sum > 0."""
    # np.min is faster than checking all elements for positivity
    # w.sum() <= 0 handles the empty case implicitly (sum of empty is 0.0)
    if w.size > 0 and w.min() < 0:
        raise ValueError("Weights must be finite and sum to a positive value.")

    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Weights must be finite and sum to a positive value.")


def validate_xyw(
    *,
    y: np.ndarray,
    w: np.ndarray,
    x: np.ndarray | None = None,
    require_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Standardize and validate target, weight, and auxiliary variables."""
    y = as_float64_1d(a=y, name="y")
    w = as_float64_1d(a=w, name="samp_weight")

    check_same_length((y, "y"), (w, "samp_weight"))
    check_weights_finite_positive(w=w)

    if x is None:
        if require_x:
            raise ValueError("Parameter x must be provided.")
    else:
        x = as_float64_1d(a=x, name="x")
        check_same_length((y, "y"), (x, "x"))

    return y, w, x


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
        raise KeyError(f"Columns not found: {missing}")

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


def assert_no_missing(*, df: pl.DataFrame, subset: Sequence[str]) -> None:
    """
    Raise with a helpful message if any column in subset has NULL/NaN/±∞.
    Optimized to perform a SINGLE scan of the data rather than multiple.
    """
    if not subset or df.is_empty():
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
        detail = ", ".join(missing_errors)
        raise ValueError(
            "Missing or invalid values found in required columns; "
            "set drop_missing=True to automatically drop them. "
            f"Affected: {detail}"
        )


# ------------------------------------------------------------------ #
# Random State Generator
# ------------------------------------------------------------------ #


@overload
def check_random_state() -> np.random.Generator: ...
@overload
def check_random_state(rstate: None = ...) -> np.random.Generator: ...
@overload
def check_random_state(rstate: int) -> np.random.Generator: ...
@overload
def check_random_state(rstate: np.random.RandomState) -> np.random.Generator: ...
@overload
def check_random_state(rstate: np.random.Generator) -> np.random.Generator: ...


def check_random_state(rstate: RandomState = None) -> np.random.Generator:
    """
    Normalize various seeds / RNGs to a numpy Generator.
    """
    if rstate is None:
        return np.random.default_rng()

    if isinstance(rstate, np.random.Generator):
        return rstate

    if isinstance(rstate, (int, np.integer)):
        return np.random.default_rng(int(rstate))

    if isinstance(rstate, np.random.RandomState):
        # Make a stable seed from the legacy RNG
        seed = int(rstate.randint(0, 2**32, dtype=np.uint32))
        return np.random.default_rng(seed)

    raise TypeError(f"Unsupported random state type: {type(rstate)!r}")
