# src/svy/utils/quantiles.py
"""
Weighted quantile computation.

Used by:
  - svy/wrangling/base.py  (categorize with percentiles=)
  - svy/core/describe.py   (run_describe percentiles)
"""

from __future__ import annotations

import numpy as np

from numpy.typing import NDArray


def weighted_quantiles(
    values: NDArray[np.float64],
    probs: list[float] | tuple[float, ...] | NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> list[float]:
    """
    Compute weighted quantiles.

    Parameters
    ----------
    values : 1-d array
        Data values (NaN/null should be removed before calling).
    probs : sequence of float
        Quantile probabilities, each in (0, 1).
    weights : 1-d array or None
        Observation weights. If None, equal weights are assumed.

    Returns
    -------
    list[float]
        Quantile values, one per element of ``probs``.

    Notes
    -----
    The algorithm sorts by value, computes the cumulative weight share,
    and linearly interpolates at each requested probability.  This matches
    the "Type 4" interpolation used by most survey software.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return [float("nan")] * len(probs)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
    else:
        w = np.ones_like(vals)

    # Drop pairs where value or weight is NaN
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    vals = vals[mask]
    w = w[mask]

    if vals.size == 0:
        return [float("nan")] * len(probs)

    # Sort by value
    order = np.argsort(vals, kind="mergesort")
    vals = vals[order]
    w = w[order]

    # Cumulative weight share, normalised to [0, 1]
    cum_w = np.cumsum(w)
    cum_w /= cum_w[-1]

    # Interpolate at all requested probabilities in one vectorised call
    return np.interp(probs, cum_w, vals).tolist()


def weighted_quantile_bins(
    values: NDArray[np.float64],
    n_groups: int,
    weights: NDArray[np.float64] | None = None,
) -> list[float]:
    """
    Compute bin edges for ``n_groups`` equal-mass quantile groups.

    Parameters
    ----------
    values : 1-d array
        Data values.
    n_groups : int
        Number of groups (e.g. 5 for quintiles, 10 for deciles).
    weights : 1-d array or None
        Observation weights. If None, equal weights are assumed.

    Returns
    -------
    list[float]
        Bin edges of length ``n_groups + 1``.  The first edge is ``-inf``
        and the last is ``inf`` so that all observations are captured.
    """
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")

    probs = [i / n_groups for i in range(1, n_groups)]  # e.g. [0.2, 0.4, 0.6, 0.8]
    cuts = weighted_quantiles(values, probs, weights)

    return [float("-inf")] + cuts + [float("inf")]
