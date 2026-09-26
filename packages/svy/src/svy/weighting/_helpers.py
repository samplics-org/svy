# src/svy/weighting/_helpers.py
"""
Shared internal helpers for the weighting namespace.

These are module-level utilities used across adjustment, normalization,
poststratification, raking, and calibration modules. Nothing here
should import from Sample or touch survey design directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from svy.core.terms import Cat, Cross, Feature


if TYPE_CHECKING:
    from svy.core.types import NDArray


# ---------------------------------------------------------------------------
# Array extraction helpers
# ---------------------------------------------------------------------------


def _to_int_array(
    data: pl.DataFrame,
    col_spec: str | tuple[str, ...] | None,
) -> np.ndarray | None:
    """
    Convert a single or multi-column design field to a 1-D int64 array
    suitable for passing to Rust functions.

    Parameters
    ----------
    data     : source DataFrame
    col_spec : str, tuple[str, ...], or None
               - None          → returns None
               - str           → encodes that single column
               - tuple[str,…]  → concatenates columns into composite labels,
                                  then integer-encodes

    Returns
    -------
    np.ndarray[int64] | None
    """
    if col_spec is None:
        return None
    cols = [col_spec] if isinstance(col_spec, str) else list(col_spec)
    if len(cols) == 1:
        vec = data[cols[0]].to_numpy()
    else:
        vec = (
            data.select(
                pl.concat_str([pl.col(c).cast(pl.Utf8) for c in cols], separator="__").alias(
                    "__key__"
                )
            )
            .get_column("__key__")
            .to_numpy()
        )
    if vec.dtype.kind in ("U", "S", "O"):
        _, codes = np.unique(vec, return_inverse=True)
        return codes.astype(np.int64)
    return vec.astype(np.int64)


def _to_float_array(
    data: pl.DataFrame,
    col: str | None,
    n: int,
) -> np.ndarray:
    """
    Return a float64 weight array from a column, or ones if col is None.
    Always returns float64 regardless of source dtype.
    """
    if col and col in data.columns:
        return data[col].to_numpy().astype(np.float64, copy=False)
    return np.ones(n, dtype=np.float64)


# ---------------------------------------------------------------------------
# Column naming helpers
# ---------------------------------------------------------------------------


def _canonical_psu_table(df: pl.DataFrame, by_cols: list[str]) -> pl.DataFrame:
    return df.select(by_cols).unique(maintain_order=True).sort(by_cols)


def _name_rep_cols(prefix: str, n_reps: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, n_reps + 1)]


def _unique_name(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    k = 1
    while f"{base}_{k}" in existing:
        k += 1
    return f"{base}_{k}"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _ensure_columns(df: pl.DataFrame, cols: list[str], *, where: str) -> None:
    from svy.errors import MethodError

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MethodError.invalid_choice(
            where=where,
            param="columns",
            got=missing,
            allowed=df.columns,
            hint="Check spelling or pass existing column names.",
        )


# ---------------------------------------------------------------------------
# Term / by-variable helpers
# ---------------------------------------------------------------------------


def _extract_cols_from_term(term: Feature) -> list[str]:
    if isinstance(term, str):
        return [term]
    if isinstance(term, Cat):
        return [term.name]
    if isinstance(term, Cross):
        return _extract_cols_from_term(term.left) + _extract_cols_from_term(term.right)
    raise TypeError(f"Unsupported term type for grouping: {type(term)}")


def _normalize_by_term(
    df: pl.DataFrame, by: Feature | None, *, param_name: str = "by", where: str
) -> tuple[list[str] | None, NDArray | None]:
    if by is None:
        return None, None

    if isinstance(by, (list, tuple)):
        raise TypeError(
            f"`{param_name}` must be a single Feature (str, Cat, Cross). "
            "Lists are not allowed; use Cross() for interactions."
        )

    adj_cols = _extract_cols_from_term(by)

    missing = [c for c in adj_cols if c not in df.columns]
    if missing:
        raise TypeError(f"`{param_name}` term refers to columns not present in data: {missing!r}")

    if len(adj_cols) == 1:
        adj_class_arr = df.get_column(adj_cols[0]).to_numpy()
    else:
        cols_np = [df.get_column(c).to_numpy() for c in adj_cols]
        adj_class_arr = np.fromiter(
            (tuple(row) for row in zip(*cols_np)),
            dtype=object,
            count=df.height,
        )

    return adj_cols, adj_class_arr


# ---------------------------------------------------------------------------
# Sort key helpers
# ---------------------------------------------------------------------------


def _num_sort_key_token(tok: str) -> tuple[int, float | str]:
    try:
        return (0, float(tok))
    except Exception:
        return (1, tok)
