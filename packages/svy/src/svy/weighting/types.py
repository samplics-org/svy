# src/svy/weighting/types.py
"""
Weighting-namespace type definitions.

Types live here — not in the engine — so that:
  - svy/engine/weighting/adj_trimming.py can import them without a cycle
  - svy/weighting/trim.py imports from the same place as the public API
  - Users can do: from svy.weighting import Threshold, TrimConfig

Relationship to svy/core/types.py and svy/core/containers.py
-------------------------------------------------------------
svy/core/types.py    — generic primitives and type aliases (Number, Category, …)
svy/core/containers.py — generic statistical output containers (ChiSquare, FDist, …)
svy/core/terms.py    — declarative term specs (Cat, Cross, RE, Threshold)
svy/weighting/types.py — weighting-specific domain objects (TrimConfig, TrimResult, …)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from numpy.typing import NDArray

from svy.core.terms import Threshold, _ComposedCap


FloatArr = NDArray[np.float64]

# Ways to specify one bound; a bare number is absolute
ThresholdSpec = float | Threshold | _ComposedCap | Callable[[FloatArr], float]


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
        number                   → absolute bound, e.g. 40 or 0.9
        Threshold.quantile(p)    → p quantile of the positive weights
        Threshold(stat, k)       → k * stat(w), e.g. Threshold("median", 6.0)
        Threshold + Threshold    → composed, e.g. Threshold("median") + 6 * Threshold("iqr")
        callable                 → f(w: FloatArr) -> float
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
    number          → absolute bound (returned as-is); must be > 0
    Threshold       → Threshold.compute(weights): absolute, quantile or k * stat
    _ComposedCap    → sum of its parts (from Threshold + Threshold)
    callable        → f(positive weights) -> float

    A number is never read as a quantile: use ``Threshold.quantile(p)``.

    Raises
    ------
    ValueError
        If the resolved threshold is not positive (e.g., from a composition
        like Threshold("mean") - 10 * Threshold("sd") that yields a negative
        value for the given weight distribution).
    """
    if isinstance(spec, (Threshold, _ComposedCap)):
        result = spec.compute(weights)
    elif isinstance(spec, bool):
        raise TypeError("A threshold cannot be a bool.")
    elif isinstance(spec, (int, float, np.integer, np.floating)):
        v = float(spec)
        if not v > 0:
            raise ValueError(
                f"Threshold value must be > 0, got {v}. A number is an absolute bound; "
                "for a quantile use svy.Threshold.quantile(p)."
            )
        return v
    elif callable(spec):
        w_pos = weights[weights > 0]
        result = float(spec(w_pos))
    else:
        raise TypeError(f"Unsupported ThresholdSpec type: {type(spec)}")

    if result < 0:
        raise ValueError(
            f"Resolved threshold must be >= 0, got {result:.4f}. "
            f"Check your Threshold specification — the composed threshold "
            f"evaluated to a negative value for this weight distribution."
        )
    return result
