# src/svy/engine/weighting/adj_trimming.py
"""
Pure algorithmic core for survey weight trimming.

All inputs and outputs are flat Python/numpy primitives — no custom types.
This makes the Rust replacement straightforward: swap the function body,
keep the same signature.

Public surface
--------------
run_trim(weights, upper, lower, redistribute, max_iter, tol) -> tuple
"""

from __future__ import annotations

import numpy as np

from numpy.typing import NDArray


FloatArr = NDArray[np.float64]
BoolArr = NDArray[np.bool_]


# ---------------------------------------------------------------------------
# ESS
# ---------------------------------------------------------------------------


def _ess(weights: FloatArr) -> float:
    s = float(np.sum(weights))
    s2 = float(np.sum(weights**2))
    if s2 == 0:
        return 0.0
    return (s * s) / s2


# ---------------------------------------------------------------------------
# Single-iteration trim step
# ---------------------------------------------------------------------------


def _trim_once(
    weights: FloatArr,
    upper: float | None,
    lower: float | None,
    redistribute: bool,
    zero_mask: BoolArr,
) -> tuple[FloatArr, int, int]:
    """
    One round of trimming.  Returns (new_weights, n_upper, n_lower).
    zero_mask marks units that were zero before trimming — never touched.
    """
    w = weights.copy()
    active = ~zero_mask
    n_upper = 0
    n_lower = 0

    if upper is not None:
        trimmed_upper = active & (w > upper)
        n_upper = int(trimmed_upper.sum())
        if n_upper > 0 and redistribute:
            excess = float(np.sum(w[trimmed_upper] - upper))
            w[trimmed_upper] = upper
            non_trimmed = active & (w <= upper) & ~trimmed_upper
            if non_trimmed.any():
                total = float(np.sum(w[non_trimmed]))
                if total > 0:
                    w[non_trimmed] += excess * (w[non_trimmed] / total)
        elif n_upper > 0:
            w[trimmed_upper] = upper

    if lower is not None:
        trimmed_lower = active & (w < lower) & (w > 0)
        n_lower = int(trimmed_lower.sum())
        if n_lower > 0 and redistribute:
            deficit = float(np.sum(lower - w[trimmed_lower]))
            w[trimmed_lower] = lower
            non_trimmed = active & (w > lower)
            if non_trimmed.any():
                total = float(np.sum(w[non_trimmed]))
                if total > deficit:
                    w[non_trimmed] -= deficit * (w[non_trimmed] / total)
        elif n_lower > 0:
            w[trimmed_lower] = lower

    return w, n_upper, n_lower


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_trim(
    weights: FloatArr,
    upper: float | None,
    lower: float | None,
    redistribute: bool,
    max_iter: int,
    tol: float,
) -> tuple[FloatArr, int, int, float, float, float, float, int, bool]:
    """
    Iterative weight trimming on a single flat weight array.

    Parameters
    ----------
    weights     : float64 array, shape (n,)
    upper       : resolved upper cap scalar, or None
    lower       : resolved lower cap scalar, or None
    redistribute: redistribute trimmed mass to non-trimmed units
    max_iter    : maximum iterations
    tol         : convergence tolerance (fraction of weights changed)

    Returns (tuple, positional)
    -------
    trimmed_weights   : float64 array
    n_trimmed_upper   : int
    n_trimmed_lower   : int
    weight_sum_before : float
    weight_sum_after  : float
    ess_before        : float
    ess_after         : float
    iterations        : int
    converged         : bool
    """
    w = np.asarray(weights, dtype=np.float64).copy()

    if np.any(w < 0):
        raise ValueError(
            "Negative weights found. Negative weights must be caught before calling run_trim()."
        )

    zero_mask: BoolArr = w == 0.0
    weight_sum_before = float(np.sum(w))
    ess_before = _ess(w)

    # Degenerate threshold: return unchanged
    if upper is not None and upper <= 0:
        return w, 0, 0, weight_sum_before, weight_sum_before, ess_before, ess_before, 0, True
    if lower is not None and lower <= 0:
        lower = None

    if upper is not None and lower is not None and lower >= upper:
        raise ValueError(
            f"lower ({lower:.6g}) >= upper ({upper:.6g}). Check threshold specifications."
        )

    n_upper_total = 0
    n_lower_total = 0
    iterations = 0
    converged = False
    n_active = int((~zero_mask).sum())

    for i in range(max_iter):
        iterations = i + 1
        w_new, n_up, n_lo = _trim_once(w, upper, lower, redistribute, zero_mask)
        n_upper_total += n_up
        n_lower_total += n_lo

        frac_changed = float(np.sum(w_new != w)) / max(1, n_active)
        w = w_new

        if frac_changed <= tol:
            converged = True
            break

    return (
        w,
        n_upper_total,
        n_lower_total,
        weight_sum_before,
        float(np.sum(w)),
        ess_before,
        _ess(w),
        iterations,
        converged,
    )
