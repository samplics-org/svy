"""
Survey Expression DSL - A Polars expression wrapper for survey data analysis.

Provides a fluent, chainable API for building Polars expressions with
survey-specific conveniences and improved error messages.
"""

from __future__ import annotations

import logging
import math
import typing as _typing

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, TypeVar

import polars as pl


log = logging.getLogger(__name__)

T = TypeVar("T", bound="Expr")


# =============================================================================
# Helper Functions
# =============================================================================


def to_polars_expr(x: Any) -> pl.Expr:
    """
    Converts a Python object, a raw pl.Expr, or a wrapped Expr object
    into a canonical Polars expression (pl.Expr).
    """
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, Expr):
        return x._e

    # Common wrappers: try a few conventional accessors
    for attr in ("to_polars", "to_pl", "as_pl", "expr", "_e"):
        if hasattr(x, attr):
            val = getattr(x, attr)
            val = val() if callable(val) else val
            if isinstance(val, pl.Expr):
                return val

    if isinstance(x, _Then):
        raise TypeError(
            "Incomplete conditional expression: call .otherwise(...) to finalize "
            "your when(...).then(...) expression."
        )

    raise TypeError(f"Unsupported expression type: {type(x).__name__!r}")


def _compile_to_polars(e: Any) -> pl.Expr:
    """Safely compile/extract Expr wrapper into pl.Expr."""
    if isinstance(e, Expr):
        return e._e
    if isinstance(e, pl.Expr):
        return e
    raise TypeError(f"Unsupported expression type: {type(e)!r}")


def _box(x: Any) -> pl.Expr:
    """Converts scalars, sequences, or wrapped Expr into a pl.Expr."""
    if isinstance(x, Expr):
        return x._e
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, (list, tuple)):
        return pl.lit(pl.Series(x))
    return pl.lit(x)


# =============================================================================
# Conditional Expression Builders
# =============================================================================


class _When:
    """Builder for conditional expressions - created by when()."""

    __slots__ = ("_cond", "_chain")

    def __init__(self, cond: pl.Expr, chain=None):
        self._cond = cond
        self._chain = chain

    def then(self, x: Any) -> _Then:
        """Specify the value when condition is true."""
        if self._chain is None:
            return _Then(pl.when(self._cond).then(_box(x)))
        else:
            return _Then(self._chain.when(self._cond).then(_box(x)))

    def __repr__(self) -> str:
        return "_When(condition pending .then())"


class _Then:
    """Intermediate builder after .then(), awaiting .when() or .otherwise()."""

    __slots__ = ("_w",)

    def __init__(self, w):
        self._w = w

    def when(self, condition: Expr | pl.Expr) -> _When:
        """Chain another condition."""
        if isinstance(condition, Expr):
            cond_expr = condition._e
        elif isinstance(condition, pl.Expr):
            cond_expr = condition
        else:
            raise TypeError(
                f"when() expects an Expr, got {type(condition).__name__!r}. "
                "Use svy.col('name') to reference columns."
            )
        return _When(cond_expr, self._w)

    def otherwise(self, x: Any) -> Expr:
        """Specify the default value when no conditions match."""
        return Expr(self._w.otherwise(_box(x)))

    def __repr__(self) -> str:
        return "_Then(awaiting .when() or .otherwise())"


# =============================================================================
# Main Expression Class
# =============================================================================


def _as_membership_rhs(values: Any) -> Any:
    """Normalize is_in's right-hand-side for Polars compatibility.

    Routes Series and expression values through .implode() to avoid the
    same-dtype ambiguity deprecation; Python iterables pass through as lists.
    """
    if isinstance(values, pl.Series):
        return values.implode()
    if isinstance(values, pl.Expr):
        return values.implode()
    if isinstance(values, Expr):
        return values._e.implode()
    return list(values)


@dataclass(frozen=True)
class Expr:
    """
    A wrapped Polars expression with a fluent, survey-friendly API.

    Supports arithmetic, comparisons, boolean logic, string operations,
    aggregations, and window functions.
    """

    _e: pl.Expr

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Expr({self._e!r})"

    # -------------------------------------------------------------------------
    # Unary Operations
    # -------------------------------------------------------------------------
    def __neg__(self: T) -> T:
        """Negation (-Expr)"""
        return _typing.cast(T, Expr(self._e.neg()))

    def __invert__(self: T) -> T:
        """Boolean NOT (~Expr)"""
        return _typing.cast(T, Expr(~self._e))

    def __abs__(self: T) -> T:
        """Absolute value (abs(Expr))"""
        return _typing.cast(T, Expr(self._e.abs()))

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------
    def __add__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e + _box(other)))

    def __radd__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(_box(other) + self._e))

    def __sub__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e - _box(other)))

    def __rsub__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(_box(other) - self._e))

    def __mul__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e * _box(other)))

    def __rmul__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(_box(other) * self._e))

    def __truediv__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e / _box(other)))

    def __rtruediv__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(_box(other) / self._e))

    def __floordiv__(self: T, other: Any) -> T:
        """Floor division (Expr // other)"""
        return _typing.cast(T, Expr(self._e // _box(other)))

    def __rfloordiv__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(_box(other) // self._e))

    def __mod__(self: T, other: Any) -> T:
        """Modulo (Expr % other)"""
        return _typing.cast(T, Expr(self._e % _box(other)))

    def __rmod__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(_box(other) % self._e))

    def __pow__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e ** _box(other)))

    def __rpow__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(_box(other) ** self._e))

    # -------------------------------------------------------------------------
    # Comparisons
    # -------------------------------------------------------------------------
    def __lt__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e < _box(other)))

    def __le__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e <= _box(other)))

    def __gt__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e > _box(other)))

    def __ge__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e >= _box(other)))

    def __eq__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e == _box(other)))

    def __ne__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(self._e != _box(other)))

    # -------------------------------------------------------------------------
    # Boolean Composition
    # -------------------------------------------------------------------------
    def __and__(self: T, other: Any) -> T:
        """Element-wise logical AND (Expr & other)"""
        return _typing.cast(T, Expr(self._e & to_polars_expr(other)))

    def __rand__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(to_polars_expr(other) & self._e))

    def __or__(self: T, other: Any) -> T:
        """Element-wise logical OR (Expr | other)"""
        return _typing.cast(T, Expr(self._e | to_polars_expr(other)))

    def __ror__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(to_polars_expr(other) | self._e))

    def __xor__(self: T, other: Any) -> T:
        """Element-wise logical XOR (Expr ^ other)"""
        return _typing.cast(T, Expr(self._e ^ to_polars_expr(other)))

    def __rxor__(self: T, other: Any) -> T:
        return _typing.cast(T, Expr(to_polars_expr(other) ^ self._e))

    # -------------------------------------------------------------------------
    # Naming / Aliasing
    # -------------------------------------------------------------------------
    def alias(self: T, name: str) -> T:
        """Rename the expression output."""
        return _typing.cast(T, Expr(self._e.alias(name)))

    def name_prefix(self: T, prefix: str) -> T:
        """Add a prefix to the column name."""
        return _typing.cast(T, Expr(self._e.name.prefix(prefix)))

    def name_suffix(self: T, suffix: str) -> T:
        """Add a suffix to the column name."""
        return _typing.cast(T, Expr(self._e.name.suffix(suffix)))

    # -------------------------------------------------------------------------
    # Null Handling
    # -------------------------------------------------------------------------
    def fill_null(self: T, value: Any) -> T:
        """Replace null values with the given value."""
        return _typing.cast(T, Expr(self._e.fill_null(_box(value))))

    def fill_nan(self: T, value: Any) -> T:
        """Replace NaN values with the given value."""
        return _typing.cast(T, Expr(self._e.fill_nan(_box(value))))

    def is_null(self: T) -> T:
        """Check if values are null."""
        return _typing.cast(T, Expr(self._e.is_null()))

    def is_not_null(self: T) -> T:
        """Check if values are not null."""
        return _typing.cast(T, Expr(self._e.is_not_null()))

    def is_nan(self: T) -> T:
        """Check if values are NaN."""
        return _typing.cast(T, Expr(self._e.is_nan()))

    def is_not_nan(self: T) -> T:
        """Check if values are not NaN."""
        return _typing.cast(T, Expr(self._e.is_not_nan()))

    def is_finite(self: T) -> T:
        """Check if values are finite (not inf or nan)."""
        return _typing.cast(T, Expr(self._e.is_finite()))

    def is_infinite(self: T) -> T:
        """Check if values are infinite."""
        return _typing.cast(T, Expr(self._e.is_infinite()))

    def drop_nulls(self: T) -> T:
        """Remove null values."""
        return _typing.cast(T, Expr(self._e.drop_nulls()))

    def drop_nans(self: T) -> T:
        """Remove NaN values."""
        return _typing.cast(T, Expr(self._e.drop_nans()))

    # -------------------------------------------------------------------------
    # Mathematical Functions
    # -------------------------------------------------------------------------
    def abs(self: T) -> T:
        """Absolute value."""
        return _typing.cast(T, Expr(self._e.abs()))

    def log(self: T, base: float | None = None) -> T:
        """Natural logarithm (or log with specified base)."""
        if base is None:
            return _typing.cast(T, Expr(self._e.log()))
        return _typing.cast(T, Expr(self._e.log(base)))

    def log10(self: T) -> T:
        """Base-10 logarithm."""
        return _typing.cast(T, Expr(self._e.log(10)))

    def log2(self: T) -> T:
        """Base-2 logarithm."""
        return _typing.cast(T, Expr(self._e.log(2)))

    def exp(self: T) -> T:
        """Exponential (e^x)."""
        return _typing.cast(T, Expr(self._e.exp()))

    def sqrt(self: T) -> T:
        """Square root."""
        return _typing.cast(T, Expr(self._e.sqrt()))

    def cbrt(self: T) -> T:
        """Cube root."""
        return _typing.cast(T, Expr(self._e.cbrt()))

    def pow(self: T, exponent: Any) -> T:
        """Raise to power."""
        return _typing.cast(T, Expr(self._e.pow(_box(exponent))))

    def round(self: T, decimals: int = 0) -> T:
        """Round to specified decimal places."""
        return _typing.cast(T, Expr(self._e.round(decimals)))

    def floor(self: T) -> T:
        """Round down to nearest integer."""
        return _typing.cast(T, Expr(self._e.floor()))

    def ceil(self: T) -> T:
        """Round up to nearest integer."""
        return _typing.cast(T, Expr(self._e.ceil()))

    def sign(self: T) -> T:
        """Sign of the value (-1, 0, or 1)."""
        return _typing.cast(T, Expr(self._e.sign()))

    def clip(self: T, lower: Any = None, upper: Any = None) -> T:
        """Clip values to a range."""
        lower_expr = _box(lower) if lower is not None else None
        upper_expr = _box(upper) if upper is not None else None
        return _typing.cast(T, Expr(self._e.clip(lower_expr, upper_expr)))

    # -------------------------------------------------------------------------
    # Trigonometric Functions
    # -------------------------------------------------------------------------
    def sin(self: T) -> T:
        """Sine."""
        return _typing.cast(T, Expr(self._e.sin()))

    def cos(self: T) -> T:
        """Cosine."""
        return _typing.cast(T, Expr(self._e.cos()))

    def tan(self: T) -> T:
        """Tangent."""
        return _typing.cast(T, Expr(self._e.tan()))

    def arcsin(self: T) -> T:
        """Arc sine."""
        return _typing.cast(T, Expr(self._e.arcsin()))

    def arccos(self: T) -> T:
        """Arc cosine."""
        return _typing.cast(T, Expr(self._e.arccos()))

    def arctan(self: T) -> T:
        """Arc tangent."""
        return _typing.cast(T, Expr(self._e.arctan()))

    def radians(self: T) -> T:
        """Convert degrees to radians."""
        return _typing.cast(T, Expr(self._e.radians()))

    def degrees(self: T) -> T:
        """Convert radians to degrees."""
        return _typing.cast(T, Expr(self._e.degrees()))

    # -------------------------------------------------------------------------
    # Type Casting
    # -------------------------------------------------------------------------
    def cast(self: T, dtype: Any) -> T:
        """Cast to a different data type."""
        return _typing.cast(T, Expr(self._e.cast(dtype)))

    def to_float(self: T) -> T:
        """Cast to Float64."""
        return _typing.cast(T, Expr(self._e.cast(pl.Float64)))

    def to_int(self: T) -> T:
        """Cast to Int64."""
        return _typing.cast(T, Expr(self._e.cast(pl.Int64)))

    def to_str(self: T) -> T:
        """Cast to String."""
        return _typing.cast(T, Expr(self._e.cast(pl.Utf8)))

    def to_bool(self: T) -> T:
        """Cast to Boolean."""
        return _typing.cast(T, Expr(self._e.cast(pl.Boolean)))

    # -------------------------------------------------------------------------
    # Membership / Range Checks
    # -------------------------------------------------------------------------
    def is_in(self: T, values: Iterable[Any]) -> T:
        """Check if values are in a collection.

        Accepts Python iterables (list/tuple/set), Polars Series, and Polars
        or svy expressions.  Series and expression values are routed through
        ``.implode()`` to avoid Polars' same-dtype ambiguity warning
        (pola-rs/polars#22149).
        """
        return _typing.cast(T, Expr(self._e.is_in(_as_membership_rhs(values))))

    isin = is_in  # Pandas-style alias

    def is_not_in(self: T, values: Iterable[Any]) -> T:
        """Check if values are NOT in a collection.

        Accepts the same value types as :meth:`is_in`.
        """
        return _typing.cast(T, Expr(~self._e.is_in(_as_membership_rhs(values))))

    def between(self: T, lower: Any, upper: Any, closed: str = "both") -> T:
        """Check if values are within a range (closed: 'both', 'left', 'right', 'none')."""
        return _typing.cast(T, Expr(self._e.is_between(_box(lower), _box(upper), closed=closed)))  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    # Aggregations
    # -------------------------------------------------------------------------
    def sum(self: T) -> T:
        """Sum of values."""
        return _typing.cast(T, Expr(self._e.sum()))

    def mean(self: T) -> T:
        """Mean of values."""
        return _typing.cast(T, Expr(self._e.mean()))

    def median(self: T) -> T:
        """Median of values."""
        return _typing.cast(T, Expr(self._e.median()))

    def mode(self: T) -> T:
        """Mode (most frequent value)."""
        return _typing.cast(T, Expr(self._e.mode()))

    def min(self: T) -> T:
        """Minimum value."""
        return _typing.cast(T, Expr(self._e.min()))

    def max(self: T) -> T:
        """Maximum value."""
        return _typing.cast(T, Expr(self._e.max()))

    def std(self: T, ddof: int = 1) -> T:
        """Standard deviation."""
        return _typing.cast(T, Expr(self._e.std(ddof=ddof)))

    def var(self: T, ddof: int = 1) -> T:
        """Variance."""
        return _typing.cast(T, Expr(self._e.var(ddof=ddof)))

    def count(self: T) -> T:
        """Count of non-null values."""
        return _typing.cast(T, Expr(self._e.count()))

    def len(self: T) -> T:
        """Total count including nulls."""
        return _typing.cast(T, Expr(self._e.len()))

    def null_count(self: T) -> T:
        """Count of null values."""
        return _typing.cast(T, Expr(self._e.null_count()))

    def first(self: T) -> T:
        """First value."""
        return _typing.cast(T, Expr(self._e.first()))

    def last(self: T) -> T:
        """Last value."""
        return _typing.cast(T, Expr(self._e.last()))

    def n_unique(self: T) -> T:
        """Count of unique values."""
        return _typing.cast(T, Expr(self._e.n_unique()))

    def unique(self: T) -> T:
        """Unique values."""
        return _typing.cast(T, Expr(self._e.unique()))

    def arg_min(self: T) -> T:
        """Index of minimum value."""
        return _typing.cast(T, Expr(self._e.arg_min()))

    def arg_max(self: T) -> T:
        """Index of maximum value."""
        return _typing.cast(T, Expr(self._e.arg_max()))

    def quantile(self: T, q: float, interpolation: str = "nearest") -> T:
        """Value at the given quantile."""
        return _typing.cast(T, Expr(self._e.quantile(q, interpolation=interpolation)))  # type: ignore[arg-type]

    def product(self: T) -> T:
        """Product of all values."""
        return _typing.cast(T, Expr(self._e.product()))

    # -------------------------------------------------------------------------
    # Window / Grouping Functions
    # -------------------------------------------------------------------------
    def over(self: T, *partition_by: str | Expr) -> T:
        """Apply as a window function over partitions."""
        cols = [p._e if isinstance(p, Expr) else pl.col(p) for p in partition_by]
        return _typing.cast(T, Expr(self._e.over(cols)))

    def cum_sum(self: T) -> T:
        """Cumulative sum."""
        return _typing.cast(T, Expr(self._e.cum_sum()))

    def cum_prod(self: T) -> T:
        """Cumulative product."""
        return _typing.cast(T, Expr(self._e.cum_prod()))

    def cum_min(self: T) -> T:
        """Cumulative minimum."""
        return _typing.cast(T, Expr(self._e.cum_min()))

    def cum_max(self: T) -> T:
        """Cumulative maximum."""
        return _typing.cast(T, Expr(self._e.cum_max()))

    def cum_count(self: T) -> T:
        """Cumulative count."""
        return _typing.cast(T, Expr(self._e.cum_count()))

    def shift(self: T, n: int = 1, fill_value: Any = None) -> T:
        """Shift values by n positions."""
        return _typing.cast(T, Expr(self._e.shift(n, fill_value=fill_value)))

    def diff(self: T, n: int = 1, null_behavior: str = "ignore") -> T:
        """Difference between consecutive values."""
        return _typing.cast(T, Expr(self._e.diff(n, null_behavior=null_behavior)))  # type: ignore[arg-type]

    def pct_change(self: T, n: int = 1) -> T:
        """Percentage change between consecutive values."""
        return _typing.cast(T, Expr(self._e.pct_change(n)))

    def rank(self: T, method: str = "ordinal", descending: bool = False) -> T:
        """Rank values (method: 'ordinal', 'min', 'max', 'dense', 'average')."""
        return _typing.cast(T, Expr(self._e.rank(method=method, descending=descending)))  # type: ignore[arg-type]

    def rolling_mean(self: T, window_size: int) -> T:
        """Rolling mean."""
        return _typing.cast(T, Expr(self._e.rolling_mean(window_size)))

    def rolling_sum(self: T, window_size: int) -> T:
        """Rolling sum."""
        return _typing.cast(T, Expr(self._e.rolling_sum(window_size)))

    def rolling_std(self: T, window_size: int) -> T:
        """Rolling standard deviation."""
        return _typing.cast(T, Expr(self._e.rolling_std(window_size)))

    def rolling_min(self: T, window_size: int) -> T:
        """Rolling minimum."""
        return _typing.cast(T, Expr(self._e.rolling_min(window_size)))

    def rolling_max(self: T, window_size: int) -> T:
        """Rolling maximum."""
        return _typing.cast(T, Expr(self._e.rolling_max(window_size)))

    def ewm_mean(self: T, span: float | None = None, alpha: float | None = None) -> T:
        """Exponentially weighted moving average."""
        if span is not None:
            return _typing.cast(T, Expr(self._e.ewm_mean(span=span)))
        elif alpha is not None:
            return _typing.cast(T, Expr(self._e.ewm_mean(alpha=alpha)))
        raise ValueError("Either 'span' or 'alpha' must be provided")

    # -------------------------------------------------------------------------
    # String Operations
    # -------------------------------------------------------------------------
    def str_len(self: T) -> T:
        """Length of string in characters."""
        return _typing.cast(T, Expr(self._e.str.len_chars()))

    def str_len_bytes(self: T) -> T:
        """Length of string in bytes."""
        return _typing.cast(T, Expr(self._e.str.len_bytes()))

    def lower(self: T) -> T:
        """Convert to lowercase."""
        return _typing.cast(T, Expr(self._e.str.to_lowercase()))

    def upper(self: T) -> T:
        """Convert to uppercase."""
        return _typing.cast(T, Expr(self._e.str.to_uppercase()))

    def title(self: T) -> T:
        """Convert to title case."""
        return _typing.cast(T, Expr(self._e.str.to_titlecase()))

    def strip(self: T, chars: str | None = None) -> T:
        """Strip whitespace (or specified chars) from both ends."""
        return _typing.cast(T, Expr(self._e.str.strip_chars(chars)))

    def lstrip(self: T, chars: str | None = None) -> T:
        """Strip from left/start."""
        return _typing.cast(T, Expr(self._e.str.strip_chars_start(chars)))

    def rstrip(self: T, chars: str | None = None) -> T:
        """Strip from right/end."""
        return _typing.cast(T, Expr(self._e.str.strip_chars_end(chars)))

    def contains(self: T, pattern: str, literal: bool = False) -> T:
        """Check if string contains pattern (regex by default)."""
        return _typing.cast(T, Expr(self._e.str.contains(pattern, literal=literal)))

    def startswith(self: T, prefix: str) -> T:
        """Check if string starts with prefix."""
        return _typing.cast(T, Expr(self._e.str.starts_with(prefix)))

    def endswith(self: T, suffix: str) -> T:
        """Check if string ends with suffix."""
        return _typing.cast(T, Expr(self._e.str.ends_with(suffix)))

    def replace(self: T, pattern: str, replacement: str, literal: bool = False) -> T:
        """Replace first occurrence of pattern."""
        return _typing.cast(T, Expr(self._e.str.replace(pattern, replacement, literal=literal)))

    def replace_all(self: T, pattern: str, replacement: str, literal: bool = False) -> T:
        """Replace all occurrences of pattern."""
        return _typing.cast(
            T, Expr(self._e.str.replace_all(pattern, replacement, literal=literal))
        )

    def slice(self: T, offset: int, length: int | None = None) -> T:
        """Extract substring."""
        return _typing.cast(T, Expr(self._e.str.slice(offset, length)))

    def head(self: T, n: int) -> T:
        """Get first n characters."""
        return _typing.cast(T, Expr(self._e.str.head(n)))

    def tail(self: T, n: int) -> T:
        """Get last n characters."""
        return _typing.cast(T, Expr(self._e.str.tail(n)))

    def split(self: T, by: str) -> T:
        """Split string by separator."""
        return _typing.cast(T, Expr(self._e.str.split(by)))

    def concat(self: T, *others: Any, separator: str = "") -> T:
        """Concatenate strings."""
        other_exprs = [_box(o) for o in others]
        return _typing.cast(T, Expr(pl.concat_str([self._e] + other_exprs, separator=separator)))

    def pad_left(self: T, length: int, fill_char: str = " ") -> T:
        """Pad string on the left."""
        return _typing.cast(T, Expr(self._e.str.pad_start(length, fill_char)))

    def pad_right(self: T, length: int, fill_char: str = " ") -> T:
        """Pad string on the right."""
        return _typing.cast(T, Expr(self._e.str.pad_end(length, fill_char)))

    def extract(self: T, pattern: str, group_index: int = 1) -> T:
        """Extract regex group."""
        return _typing.cast(T, Expr(self._e.str.extract(pattern, group_index)))

    def count_matches(self: T, pattern: str, literal: bool = False) -> T:
        """Count pattern occurrences."""
        return _typing.cast(T, Expr(self._e.str.count_matches(pattern, literal=literal)))

    # -------------------------------------------------------------------------
    # Date/Time Operations
    # -------------------------------------------------------------------------
    def dt_year(self: T) -> T:
        """Extract year."""
        return _typing.cast(T, Expr(self._e.dt.year()))

    def dt_month(self: T) -> T:
        """Extract month."""
        return _typing.cast(T, Expr(self._e.dt.month()))

    def dt_day(self: T) -> T:
        """Extract day."""
        return _typing.cast(T, Expr(self._e.dt.day()))

    def dt_hour(self: T) -> T:
        """Extract hour."""
        return _typing.cast(T, Expr(self._e.dt.hour()))

    def dt_minute(self: T) -> T:
        """Extract minute."""
        return _typing.cast(T, Expr(self._e.dt.minute()))

    def dt_second(self: T) -> T:
        """Extract second."""
        return _typing.cast(T, Expr(self._e.dt.second()))

    def dt_weekday(self: T) -> T:
        """Day of week (0=Monday, 6=Sunday)."""
        return _typing.cast(T, Expr(self._e.dt.weekday()))

    def dt_week(self: T) -> T:
        """Week number of year."""
        return _typing.cast(T, Expr(self._e.dt.week()))

    def dt_ordinal_day(self: T) -> T:
        """Day of year (1-366)."""
        return _typing.cast(T, Expr(self._e.dt.ordinal_day()))

    def dt_quarter(self: T) -> T:
        """Quarter (1-4)."""
        return _typing.cast(T, Expr(self._e.dt.quarter()))

    def dt_date(self: T) -> T:
        """Extract date component."""
        return _typing.cast(T, Expr(self._e.dt.date()))

    def dt_time(self: T) -> T:
        """Extract time component."""
        return _typing.cast(T, Expr(self._e.dt.time()))

    def dt_epoch(self: T, unit: str = "s") -> T:
        """Convert to epoch timestamp (unit: 's', 'ms', 'us', 'ns')."""
        return _typing.cast(T, Expr(self._e.dt.epoch(unit)))  # type: ignore[arg-type]

    def dt_strftime(self: T, fmt: str) -> T:
        """Format datetime as string."""
        return _typing.cast(T, Expr(self._e.dt.strftime(fmt)))

    def dt_truncate(self: T, every: str) -> T:
        """Truncate datetime (e.g., '1d', '1h', '1mo')."""
        return _typing.cast(T, Expr(self._e.dt.truncate(every)))

    # -------------------------------------------------------------------------
    # List Operations
    # -------------------------------------------------------------------------
    def list_len(self: T) -> T:
        """Length of list."""
        return _typing.cast(T, Expr(self._e.list.len()))

    def list_get(self: T, index: int) -> T:
        """Get element at index."""
        return _typing.cast(T, Expr(self._e.list.get(index)))

    def list_first(self: T) -> T:
        """First element of list."""
        return _typing.cast(T, Expr(self._e.list.first()))

    def list_last(self: T) -> T:
        """Last element of list."""
        return _typing.cast(T, Expr(self._e.list.last()))

    def list_sum(self: T) -> T:
        """Sum of list elements."""
        return _typing.cast(T, Expr(self._e.list.sum()))

    def list_mean(self: T) -> T:
        """Mean of list elements."""
        return _typing.cast(T, Expr(self._e.list.mean()))

    def list_min(self: T) -> T:
        """Min of list elements."""
        return _typing.cast(T, Expr(self._e.list.min()))

    def list_max(self: T) -> T:
        """Max of list elements."""
        return _typing.cast(T, Expr(self._e.list.max()))

    def list_contains(self: T, value: Any) -> T:
        """Check if list contains value."""
        return _typing.cast(T, Expr(self._e.list.contains(_box(value))))

    def list_join(self: T, separator: str) -> T:
        """Join list elements with separator."""
        return _typing.cast(T, Expr(self._e.list.join(separator)))

    def list_unique(self: T) -> T:
        """Unique elements in list."""
        return _typing.cast(T, Expr(self._e.list.unique()))

    def list_sort(self: T, descending: bool = False) -> T:
        """Sort list elements."""
        return _typing.cast(T, Expr(self._e.list.sort(descending=descending)))

    def list_reverse(self: T) -> T:
        """Reverse list."""
        return _typing.cast(T, Expr(self._e.list.reverse()))

    def explode(self: T) -> T:
        """Explode list into rows."""
        return _typing.cast(T, Expr(self._e.explode()))

    # -------------------------------------------------------------------------
    # Sorting
    # -------------------------------------------------------------------------
    def sort(self: T, descending: bool = False, nulls_last: bool = False) -> T:
        """Sort values."""
        return _typing.cast(T, Expr(self._e.sort(descending=descending, nulls_last=nulls_last)))

    def arg_sort(self: T, descending: bool = False, nulls_last: bool = False) -> T:
        """Get indices that would sort the values."""
        return _typing.cast(
            T, Expr(self._e.arg_sort(descending=descending, nulls_last=nulls_last))
        )

    def reverse(self: T) -> T:
        """Reverse order."""
        return _typing.cast(T, Expr(self._e.reverse()))

    def shuffle(self: T, seed: int | None = None) -> T:
        """Randomly shuffle values."""
        return _typing.cast(T, Expr(self._e.shuffle(seed=seed)))

    def sample(
        self: T, n: int | None = None, fraction: float | None = None, seed: int | None = None
    ) -> T:
        """Sample values."""
        return _typing.cast(T, Expr(self._e.sample(n=n, fraction=fraction, seed=seed)))


# =============================================================================
# Public Constructors
# =============================================================================


def col(name: str) -> Expr:
    """
    Create a column expression.

    Example:
        svy.col('age') > 18
    """
    if not isinstance(name, str):
        raise TypeError(f"col() expects a string column name, got {type(name).__name__!r}")
    return Expr(pl.col(name))


def cols(*names: str) -> list[Expr]:
    """
    Create multiple column expressions.

    Example:
        svy.cols('age', 'income', 'weight')
    """
    return [col(name) for name in names]


def lit(value: Any) -> Expr:
    """
    Create a literal value expression.

    Example:
        svy.lit(100)
    """
    return Expr(pl.lit(value))


def when(condition: Expr | pl.Expr) -> _When:
    """
    Start a conditional expression.

    Example:
        svy.when(svy.col('age') < 18).then('child')
           .when(svy.col('age') < 65).then('adult')
           .otherwise('senior')
    """
    if isinstance(condition, Expr):
        cond_expr = condition._e
    elif isinstance(condition, pl.Expr):
        cond_expr = condition
    else:
        raise TypeError(
            f"when() expects an Expr, got {type(condition).__name__!r}. "
            "Use svy.col('name') to reference columns."
        )
    return _When(cond_expr)


def coalesce(*exprs: Expr | Any) -> Expr:
    """
    Return first non-null value from expressions.

    Example:
        svy.coalesce(svy.col('preferred_name'), svy.col('first_name'), svy.lit('Unknown'))
    """
    if not exprs:
        raise ValueError("coalesce() requires at least one expression")
    pl_exprs = [_box(e) for e in exprs]
    return Expr(pl.coalesce(pl_exprs))


def concat_str(*exprs: Expr | Any, separator: str = "") -> Expr:
    """
    Concatenate strings from multiple expressions.

    Example:
        svy.concat_str(svy.col('first'), svy.lit(' '), svy.col('last'))
    """
    pl_exprs = [_box(e) for e in exprs]
    return Expr(pl.concat_str(pl_exprs, separator=separator))


def all_horizontal(*exprs: Expr) -> Expr:
    """
    Check if all expressions are True (row-wise AND).

    Example:
        svy.all_horizontal(svy.col('a') > 0, svy.col('b') > 0)
    """
    pl_exprs = [_box(e) for e in exprs]
    return Expr(pl.all_horizontal(pl_exprs))


def any_horizontal(*exprs: Expr) -> Expr:
    """
    Check if any expression is True (row-wise OR).

    Example:
        svy.any_horizontal(svy.col('a') > 0, svy.col('b') > 0)
    """
    pl_exprs = [_box(e) for e in exprs]
    return Expr(pl.any_horizontal(pl_exprs))


def sum_horizontal(*exprs: Expr) -> Expr:
    """
    Sum values across columns (row-wise).

    Example:
        svy.sum_horizontal(svy.col('a'), svy.col('b'), svy.col('c'))
    """
    pl_exprs = [_box(e) for e in exprs]
    return Expr(pl.sum_horizontal(pl_exprs))


def min_horizontal(*exprs: Expr) -> Expr:
    """Minimum value across columns (row-wise)."""
    pl_exprs = [_box(e) for e in exprs]
    return Expr(pl.min_horizontal(pl_exprs))


def max_horizontal(*exprs: Expr) -> Expr:
    """Maximum value across columns (row-wise)."""
    pl_exprs = [_box(e) for e in exprs]
    return Expr(pl.max_horizontal(pl_exprs))


def struct(*exprs: Expr | tuple[str, Expr], **named_exprs: Expr) -> Expr:
    """
    Create a struct from expressions.

    Example:
        svy.struct(svy.col('a'), svy.col('b'))
        svy.struct(x=svy.col('a'), y=svy.col('b'))
    """
    pl_exprs = []
    for e in exprs:
        if isinstance(e, tuple):
            name, expr = e
            pl_exprs.append(_box(expr).alias(name))
        else:
            pl_exprs.append(_box(e))
    for name, expr in named_exprs.items():
        pl_exprs.append(_box(expr).alias(name))
    return Expr(pl.struct(pl_exprs))


# =============================================================================
# Safe Functions Mapping (for expression evaluation contexts)
# =============================================================================

SAFE_FUNCS: Mapping[str, Any] = {
    # Math operations
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "log": lambda x, base=None: (
        x.log(base)
        if isinstance(x, Expr)
        else (math.log(x) if base is None else math.log(x, base))
    ),
    "log10": lambda x: x.log10() if isinstance(x, Expr) else math.log10(x),
    "log2": lambda x: x.log2() if isinstance(x, Expr) else math.log2(x),
    "exp": lambda x: x.exp() if isinstance(x, Expr) else math.exp(x),
    "sqrt": lambda x: x.sqrt() if isinstance(x, Expr) else math.sqrt(x),
    "floor": lambda x: x.floor() if isinstance(x, Expr) else math.floor(x),
    "ceil": lambda x: x.ceil() if isinstance(x, Expr) else math.ceil(x),
    # Trig
    "sin": lambda x: x.sin() if isinstance(x, Expr) else math.sin(x),
    "cos": lambda x: x.cos() if isinstance(x, Expr) else math.cos(x),
    "tan": lambda x: x.tan() if isinstance(x, Expr) else math.tan(x),
    # Constructors
    "col": col,
    "cols": cols,
    "lit": lit,
    "when": when,
    "coalesce": coalesce,
    "concat_str": concat_str,
    # Horizontal operations
    "all_horizontal": all_horizontal,
    "any_horizontal": any_horizontal,
    "sum_horizontal": sum_horizontal,
    "min_horizontal": min_horizontal,
    "max_horizontal": max_horizontal,
    # Struct
    "struct": struct,
    # Constants
    "pi": math.pi,
    "e": math.e,
    "nan": float("nan"),
    "inf": float("inf"),
}
