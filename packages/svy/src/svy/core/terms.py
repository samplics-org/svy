# src/svy/core/terms.py
"""
Declarative term specifications for svy methods.

Terms are lightweight, frozen objects that users pass to svy methods
to declare intent. Some reference columns (Cat, Cross, RE), others
describe computation rules (Threshold). The method receiving the term
decides what to apply it to.

All terms inherit from Term for isinstance checks and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Union

import numpy as np

from numpy.typing import NDArray


# A "Feature" can be a simple string (column name) or a Term object
Feature = Union[str, "Term"]

FloatArr = NDArray[np.float64]

_SUPPORTED_STATS = frozenset({"median", "mean", "sd", "iqr", "quantile", "absolute"})


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
class Threshold(Term):
    """
    A bound computed from the values it is applied to (the weights, in trimming).

    Context-free: the method that receives it supplies the values. A bare
    number passed where a threshold is expected is an absolute bound.

    Kinds
    -----
    ``Threshold.absolute(v)``   the value ``v`` itself
    ``Threshold.quantile(p)``   the ``p`` quantile of the positive values, ``p`` in (0, 1]
    ``Threshold(stat, k)``      ``k`` × ``stat`` of the positive values, ``stat`` one of
                                ``"median"``, ``"mean"``, ``"sd"``, ``"iqr"``

    Thresholds compose with ``+``, ``-`` and scalar ``*``.

    Examples
    --------
    >>> Threshold.quantile(0.99)                     # 99th percentile
    >>> Threshold.absolute(0.9)                      # a cap at 0.9
    >>> Threshold("median", 3.5)                     # 3.5 × median
    >>> Threshold("median") + 6 * Threshold("iqr")   # median + 6 × IQR
    >>> Threshold("mean") - 2 * Threshold("sd")      # mean − 2 × SD (lower bound)
    """

    stat: str
    k: float = 1.0
    p: float | None = None

    def __post_init__(self) -> None:
        if self.stat not in _SUPPORTED_STATS:
            raise ValueError(
                f"Unsupported stat {self.stat!r}. Must be one of: {sorted(_SUPPORTED_STATS)}"
            )
        if self.k == 0:
            raise ValueError(f"k must be nonzero, got {self.k}")
        if self.stat == "quantile":
            if self.p is None:
                raise ValueError("A quantile threshold needs p: use Threshold.quantile(p).")
            if not (0 < self.p <= 1):
                hint = (
                    f" For the {self.p:g}th percentile use {self.p / 100:g}."
                    if 1 < self.p <= 100
                    else ""
                )
                raise ValueError(f"Quantile p must be in (0, 1], got {self.p}.{hint}")
        elif self.p is not None:
            raise ValueError(f"p applies only to a quantile threshold, not {self.stat!r}.")

    @classmethod
    def quantile(cls, p: float) -> "Threshold":
        """The ``p`` quantile of the positive values, ``p`` in (0, 1]."""
        return cls("quantile", p=float(p))

    @classmethod
    def absolute(cls, value: float) -> "Threshold":
        """The value itself, whatever the data. Same as passing the bare number."""
        if not value > 0:
            raise ValueError(f"An absolute threshold must be > 0, got {value}.")
        return cls("absolute", float(value))

    def compute(self, values: FloatArr) -> float:
        """Compute the threshold scalar from an array of values."""
        if self.stat == "absolute":
            return self.k
        v = values[values > 0]
        if v.size == 0:
            return 0.0
        if self.stat == "quantile":
            return self.k * float(np.quantile(v, self.p))
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

    def __rmul__(self, k: float) -> "Threshold":
        """Scalar * Threshold: returns a new Threshold with scaled k."""
        return replace(self, k=float(k) * self.k)

    def __mul__(self, k: float) -> "Threshold":
        """Threshold * scalar: returns a new Threshold with scaled k."""
        return replace(self, k=self.k * float(k))

    def __add__(self, other: "Threshold") -> "_ComposedCap":
        """Threshold + Threshold: returns a composed threshold (sum)."""
        if not isinstance(other, (Threshold, _ComposedCap)):
            return NotImplemented
        left = _ComposedCap([self])
        if isinstance(other, Threshold):
            return _ComposedCap(left._parts + [other])
        return _ComposedCap(left._parts + other._parts)

    def __sub__(self, other: "Threshold") -> "_ComposedCap":
        """Threshold - Threshold: returns a composed threshold (difference)."""
        if not isinstance(other, Threshold):
            return NotImplemented
        return self.__add__(replace(other, k=-other.k))

    def __repr__(self) -> str:
        if self.stat == "absolute":
            return f"Threshold.absolute({self.k})"
        if self.stat == "quantile":
            base = f"Threshold.quantile({self.p})"
            return base if self.k == 1.0 else f"{self.k} * {base}"
        if self.k == 1.0:
            return f"Threshold('{self.stat}')"
        return f"Threshold('{self.stat}', {self.k})"


#: The earlier name, kept so existing code runs unchanged.
Cap = Threshold


class _ComposedCap:
    """
    Internal: result of composing multiple Caps via + and -.

    Acts as a callable, so it satisfies ThresholdSpec directly.
    Users never instantiate this — they get it from Threshold(...) + Threshold(...).
    """

    __slots__ = ("_parts",)

    def __init__(self, parts: list[Threshold]) -> None:
        self._parts = parts

    def compute(self, values: FloatArr) -> float:
        return sum(cap.compute(values) for cap in self._parts)

    def __call__(self, values: FloatArr) -> float:
        """Makes _ComposedCap a valid ThresholdSpec (callable)."""
        return self.compute(values)

    def __add__(self, other: "Threshold | _ComposedCap") -> "_ComposedCap":
        if isinstance(other, Threshold):
            return _ComposedCap(self._parts + [other])
        if isinstance(other, _ComposedCap):
            return _ComposedCap(self._parts + other._parts)
        return NotImplemented

    def __sub__(self, other: Threshold) -> "_ComposedCap":
        if not isinstance(other, Threshold):
            return NotImplemented
        return _ComposedCap(self._parts + [replace(other, k=-other.k)])

    def __repr__(self) -> str:
        parts = []
        for i, cap in enumerate(self._parts):
            if i == 0:
                parts.append(repr(cap))
            elif cap.k < 0:
                parts.append(f"- {repr(replace(cap, k=-cap.k))}")
            else:
                parts.append(f"+ {repr(cap)}")
        return " ".join(parts)
