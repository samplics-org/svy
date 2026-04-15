# src/svy/core/terms.py
"""
Declarative term specifications for svy methods.

Terms are lightweight, frozen objects that users pass to svy methods
to declare intent. Some reference columns (Cat, Cross, RE), others
describe computation rules (Cap). The method receiving the term
decides what to apply it to.

All terms inherit from Term for isinstance checks and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray


# A "Feature" can be a simple string (column name) or a Term object
Feature = Union[str, "Term"]

FloatArr = NDArray[np.float64]

_SUPPORTED_STATS = frozenset({"median", "mean", "sd", "iqr"})


class Term:
    """Base class for all model terms."""

    pass


@dataclass(frozen=True)
class Cat(Term):
    """
    Explicitly treat a variable as categorical.

    Args:
        name: The column name in the dataframe.
        ref: (Optional) A specific level to drop (reference level).
             If None, all levels are usually kept (typical for calibration/raking).
    """

    name: str
    ref: str | int | float | None = None

    def __repr__(self) -> str:
        ref_str = f", ref={self.ref!r}" if self.ref is not None else ""
        return f"Cat('{self.name}'{ref_str})"


@dataclass(frozen=True)
class Cross(Term):
    """
    Create an interaction (crossing) between two variables.

    Args:
        left: The first variable (string name or Term object).
        right: The second variable (string name or Term object).
    """

    left: Feature
    right: Feature

    def __repr__(self) -> str:
        return f"Cross({self.left!r}, {self.right!r})"


@dataclass(frozen=True)
class RE(Term):
    """
    Random Effect term.

    Used to define the nesting structure (Area) for SAE.
    """

    name: str

    def __repr__(self) -> str:
        return f"RE('{self.name}')"


@dataclass(frozen=True)
class Cap(Term):
    """
    Statistical threshold for capping values.

    Describes how to compute a data-driven bound from an array of values.
    Used in weight trimming, and anywhere a stat-based cap is needed
    (e.g., top_code, bottom_code).

    The column or weight array that Cap operates on is determined by
    the method that receives it — Cap itself is context-free.

    Supports composition via ``+``, ``-``, and ``*`` operators.

    Parameters
    ----------
    stat : {"median", "mean", "sd", "iqr"}
        Statistic to compute from the (positive) values.
    k : float
        Multiplier. Default 1.0.

    Examples
    --------
    Simple thresholds:

    >>> Cap("median", 3.5)                     # 3.5 × median
    >>> Cap("mean", 5.0)                       # 5.0 × mean

    Composed thresholds:

    >>> Cap("median") + 6 * Cap("iqr")         # median + 6 × IQR
    >>> Cap("mean") + 3 * Cap("sd")            # mean + 3 × SD
    >>> Cap("mean") - 2 * Cap("sd")            # mean − 2 × SD (lower bound)
    """

    stat: str
    k: float = 1.0

    def __post_init__(self) -> None:
        if self.stat not in _SUPPORTED_STATS:
            raise ValueError(
                f"Unsupported stat {self.stat!r}. Must be one of: {sorted(_SUPPORTED_STATS)}"
            )
        if self.k == 0:
            raise ValueError(f"k must be nonzero, got {self.k}")

    def compute(self, values: FloatArr) -> float:
        """Compute the threshold scalar from an array of values."""
        v = values[values > 0]
        if v.size == 0:
            return 0.0
        if self.stat == "median":
            return self.k * float(np.median(v))
        if self.stat == "mean":
            return self.k * float(np.mean(v))
        if self.stat == "sd":
            return self.k * float(np.std(v, ddof=1))
        if self.stat == "iqr":
            q75, q25 = np.percentile(v, [75.0, 25.0])
            return self.k * float(q75 - q25)
        raise AssertionError(f"Unhandled stat: {self.stat}")  # pragma: no cover

    # -- Composition operators ------------------------------------------------

    def __rmul__(self, k: float) -> "Cap":
        """Scalar * Cap: returns a new Cap with scaled k."""
        return Cap(self.stat, float(k) * self.k)

    def __mul__(self, k: float) -> "Cap":
        """Cap * scalar: returns a new Cap with scaled k."""
        return Cap(self.stat, self.k * float(k))

    def __add__(self, other: "Cap") -> "_ComposedCap":
        """Cap + Cap: returns a composed threshold (sum)."""
        if not isinstance(other, (Cap, _ComposedCap)):
            return NotImplemented
        left = _ComposedCap([self]) if isinstance(self, Cap) else self
        if isinstance(other, Cap):
            return _ComposedCap(left._parts + [other])
        return _ComposedCap(left._parts + other._parts)

    def __sub__(self, other: "Cap") -> "_ComposedCap":
        """Cap - Cap: returns a composed threshold (difference)."""
        if not isinstance(other, Cap):
            return NotImplemented
        negated = Cap(other.stat, -other.k)
        return self.__add__(negated)

    def __repr__(self) -> str:
        if self.k == 1.0:
            return f"Cap('{self.stat}')"
        return f"Cap('{self.stat}', {self.k})"


class _ComposedCap:
    """
    Internal: result of composing multiple Caps via + and -.

    Acts as a callable, so it satisfies ThresholdSpec directly.
    Users never instantiate this — they get it from Cap(...) + Cap(...).
    """

    __slots__ = ("_parts",)

    def __init__(self, parts: list[Cap]) -> None:
        self._parts = parts

    def compute(self, values: FloatArr) -> float:
        return sum(cap.compute(values) for cap in self._parts)

    def __call__(self, values: FloatArr) -> float:
        """Makes _ComposedCap a valid ThresholdSpec (callable)."""
        return self.compute(values)

    def __add__(self, other: "Cap | _ComposedCap") -> "_ComposedCap":
        if isinstance(other, Cap):
            return _ComposedCap(self._parts + [other])
        if isinstance(other, _ComposedCap):
            return _ComposedCap(self._parts + other._parts)
        return NotImplemented

    def __sub__(self, other: Cap) -> "_ComposedCap":
        if not isinstance(other, Cap):
            return NotImplemented
        negated = Cap(other.stat, -other.k)
        return _ComposedCap(self._parts + [negated])

    def __repr__(self) -> str:
        parts = []
        for i, cap in enumerate(self._parts):
            if i == 0:
                parts.append(repr(cap))
            elif cap.k < 0:
                pos = Cap(cap.stat, -cap.k)
                parts.append(f"- {repr(pos)}")
            else:
                parts.append(f"+ {repr(cap)}")
        return " ".join(parts)
