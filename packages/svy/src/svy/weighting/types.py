# src/svy/weighting/types.py
"""
Weighting-namespace type definitions.

Types live here — not in the engine — so that:
  - svy/engine/weighting/adj_trimming.py can import them without a cycle
  - svy/weighting/trim.py imports from the same place as the public API
  - Users can do: from svy.weighting import Threshold, TrimConfig, Cap

Relationship to svy/core/types.py and svy/core/containers.py
-------------------------------------------------------------
svy/core/types.py    — generic primitives and type aliases (Number, Category, …)
svy/core/containers.py — generic statistical output containers (ChiSquare, FDist, …)
svy/core/terms.py    — declarative term specs (Cat, Cross, RE, Cap)
svy/weighting/types.py — weighting-specific domain objects (TrimConfig, TrimResult, …)
                         and backward-compatible Threshold alias
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from numpy.typing import NDArray

from svy.core.terms import Cap, _ComposedCap


FloatArr = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Threshold — backward-compatible alias for Cap
# ---------------------------------------------------------------------------

Threshold = Cap
"""Backward-compatible alias for :class:`svy.core.terms.Cap`."""


# Three ways to specify one bound
ThresholdSpec = float | Cap | _ComposedCap | Callable[[FloatArr], float]


# ---------------------------------------------------------------------------
# TrimConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrimConfig:
    """
    Full trimming specification — single source of truth for trim() parameters.

    trim() constructs one of these as its first action and delegates all logic
    to run_trim().  Other weighting methods (rake, calibrate) can accept a
    TrimConfig via their trimming= parameter.

    Parameters
    ----------
    upper : ThresholdSpec | None
        Upper bound spec.
        float > 1       → absolute cap
        float in (0, 1] → quantile of weight distribution
        Cap             → k * stat(w), e.g. Cap("median", 6.0)
        Cap + Cap       → composed, e.g. Cap("median") + 6 * Cap("iqr")
        callable        → f(w: FloatArr) -> float
    lower : ThresholdSpec | None
        Lower bound spec, same type rules as upper.
    by : str | list[str] | None
        Domain variable(s).  Thresholds computed per domain;
        redistribution also within each domain.
    redistribute : bool
        Redistribute trimmed mass proportionally to non-trimmed units.
        Default True.
    min_cell_size : int
        Skip (and warn) domains with fewer positive-weight units than this.
        Default 10.
    max_iter : int
        Maximum iterations.  Default 10.
    tol : float
        Convergence tolerance: fraction of weights changed between iterations.
        Default 1e-6.
    """

    upper: ThresholdSpec | None = None
    lower: ThresholdSpec | None = None
    by: str | Sequence[str] | None = None
    redistribute: bool = True
    min_cell_size: int = 10
    max_iter: int = 10
    tol: float = 1e-6

    def __post_init__(self) -> None:
        if self.upper is None and self.lower is None:
            raise ValueError("At least one of `upper` or `lower` must be specified.")
        if self.min_cell_size < 1:
            raise ValueError(f"min_cell_size must be >= 1, got {self.min_cell_size}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if not (0 < self.tol < 1):
            raise ValueError(f"tol must be in (0, 1), got {self.tol}")


# ---------------------------------------------------------------------------
# TrimResult
# ---------------------------------------------------------------------------


@dataclass
class TrimResult:
    """
    Output of run_trim() for a single weight array (one domain group).

    Attributes
    ----------
    weights : FloatArr
        Trimmed weight array (same length as input).
    upper_threshold : float | None
        Resolved upper cutoff value (None if not requested).
    lower_threshold : float | None
        Resolved lower cutoff value (None if not requested).
    n_trimmed_upper : int
        Number of units trimmed at the upper bound.
    n_trimmed_lower : int
        Number of units trimmed at the lower bound.
    weight_sum_before : float
    weight_sum_after : float
    ess_before : float
        Effective sample size = (Σw)² / Σw² before trimming.
    ess_after : float
    iterations : int
        Number of iterations actually run.
    converged : bool
    """

    weights: FloatArr
    upper_threshold: float | None
    lower_threshold: float | None
    n_trimmed_upper: int
    n_trimmed_lower: int
    weight_sum_before: float
    weight_sum_after: float
    ess_before: float
    ess_after: float
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# resolve_threshold — lives here because it operates on ThresholdSpec
# ---------------------------------------------------------------------------


def resolve_threshold(spec: ThresholdSpec, weights: FloatArr) -> float:
    """
    Convert any ThresholdSpec to a resolved scalar cutoff value.

    Rules
    -----
    Cap             → k * stat(positive weights)
    _ComposedCap    → sum of Cap.compute() results (from Cap + Cap)
    float > 1       → absolute cap (returned as-is)
    float in (0, 1] → quantile of positive weights
    callable        → f(positive weights) -> float

    Raises
    ------
    ValueError
        If the resolved threshold is not positive (e.g., from a
        composition like Cap("mean") - 10 * Cap("sd") that yields
        a negative value for the given weight distribution).
    """
    if isinstance(spec, Cap):
        result = spec.compute(weights)
    elif isinstance(spec, _ComposedCap):
        result = spec.compute(weights)
    elif callable(spec):
        w_pos = weights[weights > 0]
        result = float(spec(w_pos))
    elif isinstance(spec, (int, float)):
        v = float(spec)
        if v <= 0:
            raise ValueError(f"Threshold value must be > 0, got {v}")
        if v <= 1.0:
            w_pos = weights[weights > 0]
            if w_pos.size == 0:
                return 0.0
            return float(np.quantile(w_pos, v))
        return v
    else:
        raise TypeError(f"Unsupported ThresholdSpec type: {type(spec)}")

    if result < 0:
        raise ValueError(
            f"Resolved threshold must be >= 0, got {result:.4f}. "
            f"Check your Cap specification — the composed threshold "
            f"evaluated to a negative value for this weight distribution."
        )
    return result
