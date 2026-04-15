# src/svy/core/types.py
from __future__ import annotations

import logging

from typing import (
    Any,
    Callable,
    Final,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import polars as pl

from numpy.typing import NDArray


log = logging.getLogger(__name__)

# --- Generic Type Variables ---
DT = TypeVar("DT", bound=np.generic)

# --- Core Primitives ---
Number: TypeAlias = int | float
Category: TypeAlias = str | int | float | bool  # matches to_Categoryber-normalized labels

# --- Basic Structures ---
DF: TypeAlias = pl.DataFrame | pl.LazyFrame
Array: TypeAlias = NDArray
IntArray: TypeAlias = NDArray[np.int64]
FloatArray: TypeAlias = NDArray[np.float64]

DictStrArray: TypeAlias = dict[str, Array]
DictStrArrayFloat: TypeAlias = dict[str, FloatArray]

# --- Reusable Dictionary Types ---
DomainScalarMap: TypeAlias = dict[Category, Number]  # e.g., {"North": 50.0, 1: 24.0}
DomainCatMap: TypeAlias = dict[Category, DomainScalarMap]

# --- Raking Specific Types ---
ControlsType: TypeAlias = dict[str, DomainScalarMap]  # {"region": {...}, "age": {...}}


# Accept any mapping-like input for validators/converters
ControlsLike: TypeAlias = Mapping[str, Mapping[Category, Number]]

# --- Random state (sklearn-style) ---
RandomState: TypeAlias = np.random.Generator | np.random.RandomState | int | None


# -----------------------------
# Missing-sentinel machinery
# -----------------------------


class _MissingType:
    __slots__ = ()

    def __repr__(self) -> str:  # optional, nice for debugging
        return "<MISSING>"


# Singleton sentinel value
_MISSING: Final[_MissingType] = _MissingType()


def _is_MissingType(x: object) -> TypeGuard[_MissingType]:
    # Be explicit so static analyzers are happy and humans can read it
    return isinstance(x, _MissingType) and x is _MISSING


# --- Expression-like protocols & aliases --------------------------------------


@runtime_checkable
class HasExpr(Protocol):
    """Minimal protocol for svy Expr wrappers (must expose a ._e pl.Expr)."""

    @property
    def _e(self) -> pl.Expr: ...


# A thing that can behave like an expression we can compile to polars.
ExprLike: TypeAlias = pl.Expr | HasExpr

# Callable that receives an env mapping of column names -> pl.col(name),
# and returns something expression-like.
ExprCallable: TypeAlias = Callable[[Mapping[str, pl.Expr]], ExprLike]

# A mutate spec value can be an Expr-like, a callable producing one, or a columnar literal.
SeriesLike: TypeAlias = pl.Series | NDArray | Sequence[Any]
MutateValue: TypeAlias = ExprLike | ExprCallable | SeriesLike

# --- Common API argument shapes ------------------------------------------------

# WHERE can be:
# - None
# - a mapping of {col: value|sequence-of-values}
# - a sequence of ExprLike combined by AND
# - a single ExprLike
WhereArg: TypeAlias = None | Mapping[str, Any] | Sequence[ExprLike] | ExprLike

# Column selection / sorting
ColumnsArg: TypeAlias = str | Sequence[str] | None
OrderByArg: TypeAlias = str | Sequence[str] | None
DescendingArg: TypeAlias = bool | Sequence[bool]
