# src/svy/selection/allocation.py
"""
Sample-size allocation helpers.

Pure functions that derive per-group n mappings from a target total size
or sampling rate.  Results feed directly into any selection method as n=.

Methods
-------
proportional  n_h proportional to N_h
neyman        optimal allocation (n_h proportional to N_h * SD_h)
equal         equal n per stratum
rate          fixed sampling rate per stratum

All use Hamilton (largest-remainder) integer rounding so that
sum(result.values()) == n_total exactly for proportional / neyman.
"""

from __future__ import annotations

import math
import warnings


# ---------------------------------------------------------------------------
# Core allocation functions  (pure -- no Sample dependency)
# ---------------------------------------------------------------------------


def _apply_population_caps(
    alloc: dict[str, int],
    group_sizes: dict[str, int],
    measure: dict[str, float],
    *,
    label: str,
) -> dict[str, int]:
    """
    Cap each group's allocation at its frame count, redistributing the
    surplus to groups with headroom (proportional to `measure`, Hamilton
    largest-remainder rounding). Warns if the surplus cannot be placed.
    """
    import numpy as np

    out = dict(alloc)
    for _ in range(len(out)):
        over = {g: out[g] - group_sizes[g] for g in out if out[g] > group_sizes[g]}
        if not over:
            return out
        surplus = sum(over.values())
        for g in over:
            out[g] = group_sizes[g]
        receivers = [g for g in out if out[g] < group_sizes[g]]
        if not receivers:
            warnings.warn(
                f"{label}: total allocation reduced by {surplus} because every "
                "group is capped at its frame size.",
                stacklevel=4,
            )
            return out
        m = np.array([max(measure.get(g, 0.0), 0.0) for g in receivers], dtype=np.float64)
        if m.sum() <= 0:
            m = np.ones(len(receivers), dtype=np.float64)
        raw = m / m.sum() * surplus
        base = np.floor(raw)
        rem = surplus - int(base.sum())
        order = np.argsort(-(raw - base))
        for i in range(max(0, rem)):
            base[order[i % len(order)]] += 1
        for i, g in enumerate(receivers):
            out[g] += int(base[i])
    return out


def _proportional_allocation(
    group_sizes: dict[str, int],
    n_total: int,
    *,
    min_n: int = 1,
    cap_at_population: bool = True,
) -> dict[str, int]:
    """
    Allocate n_total units proportional to group size.

    Parameters
    ----------
    group_sizes       : {group_key: frame_count}
    n_total           : target overall sample size
    min_n             : floor allocation per non-empty group (default 1)
    cap_at_population : cap n_h <= N_h, redistributing surplus (default True)
    """
    import numpy as np

    total_pop = sum(group_sizes.values())
    if total_pop == 0:
        raise ValueError("proportional_allocation: total frame size is 0.")
    if n_total <= 0:
        raise ValueError(f"proportional_allocation: n_total must be > 0, got {n_total}.")
    if n_total > total_pop:
        warnings.warn(
            f"proportional_allocation: n_total={n_total} exceeds the total frame "
            f"size of {total_pop}. Capping at frame size.",
            stacklevel=3,
        )
        n_total = total_pop

    groups = list(group_sizes.keys())
    sizes = np.array([group_sizes[g] for g in groups], dtype=np.float64)
    non_empty = sizes > 0
    raw = sizes / total_pop * n_total

    # Apply floor with min_n for non-empty groups
    floored = np.where(non_empty, np.maximum(np.floor(raw), min_n), 0.0)

    # Hamilton largest-remainder to distribute rounding residual
    remainder = n_total - int(floored.sum())
    if remainder < 0:
        # min_n forced over-allocation -- scale back
        warnings.warn(
            f"proportional_allocation: min_n={min_n} floors exceed n_total="
            f"{n_total}; min_n is not honored and allocation is rescaled.",
            stacklevel=3,
        )
        floored = np.floor(raw * (n_total / raw.sum()))
        remainder = n_total - int(floored.sum())

    fractional = raw - floored
    order = np.argsort(-fractional)
    for i in range(max(0, remainder)):
        floored[order[i % len(order)]] += 1

    result = {g: int(floored[i]) for i, g in enumerate(groups)}
    if cap_at_population:
        result = _apply_population_caps(
            result,
            group_sizes,
            {g: float(group_sizes[g]) for g in groups},
            label="proportional_allocation",
        )
    return result


def _neyman_allocation(
    group_sizes: dict[str, int],
    group_sds: dict[str, float],
    n_total: int,
    *,
    min_n: int = 1,
    cap_at_population: bool = True,
) -> dict[str, int]:
    """
    Neyman / optimal allocation: n_h proportional to N_h * SD_h.

    Parameters
    ----------
    group_sizes       : {group_key: frame_count}
    group_sds         : {group_key: within-group SD of target variable};
                        must cover every non-empty group
    n_total           : target overall sample size
    min_n             : floor per non-empty group
    cap_at_population : cap n_h <= N_h, redistributing surplus (default True)
    """
    import numpy as np

    groups = list(group_sizes.keys())
    missing = [g for g in groups if group_sizes[g] > 0 and g not in group_sds]
    if missing:
        raise ValueError(
            f"neyman_allocation: group_sds is missing entries for non-empty "
            f"groups {missing!r}. Provide an SD for every group."
        )

    if n_total <= 0:
        raise ValueError(f"neyman_allocation: n_total must be > 0, got {n_total}.")
    total_pop = sum(group_sizes.values())
    if cap_at_population and n_total > total_pop:
        warnings.warn(
            f"neyman_allocation: n_total={n_total} exceeds the total frame "
            f"size of {total_pop}. Capping at frame size.",
            stacklevel=3,
        )
        n_total = total_pop

    N = np.array([group_sizes[g] for g in groups], dtype=np.float64)
    S = np.array([group_sds.get(g, 0.0) for g in groups], dtype=np.float64)

    measure = N * S
    total_measure = measure.sum()
    if total_measure == 0:
        raise ValueError(
            "neyman_allocation: all N*SD products are zero. "
            "Check that group_sds contains positive values."
        )

    raw = measure / total_measure * n_total
    non_empty = N > 0
    floored = np.where(non_empty, np.maximum(np.floor(raw), min_n), 0.0)
    remainder = n_total - int(floored.sum())
    if remainder < 0:
        # min_n forced over-allocation -- scale back (mirror proportional)
        warnings.warn(
            f"neyman_allocation: min_n={min_n} floors exceed n_total="
            f"{n_total}; min_n is not honored and allocation is rescaled.",
            stacklevel=3,
        )
        floored = np.floor(raw * (n_total / raw.sum()))
        remainder = n_total - int(floored.sum())

    fractional = raw - floored
    order = np.argsort(-fractional)
    for i in range(max(0, remainder)):
        floored[order[i % len(order)]] += 1

    result = {g: int(floored[i]) for i, g in enumerate(groups)}
    if cap_at_population:
        result = _apply_population_caps(
            result,
            group_sizes,
            {g: float(measure[i]) for i, g in enumerate(groups)},
            label="neyman_allocation",
        )
    return result


def _equal_allocation(
    group_sizes: dict[str, int],
    n_per_group: int,
    *,
    cap_at_population: bool = True,
) -> dict[str, int]:
    """Allocate exactly n_per_group to every non-empty group."""
    result: dict[str, int] = {}
    for g, size in group_sizes.items():
        if size == 0:
            result[g] = 0
        elif cap_at_population:
            result[g] = min(n_per_group, size)
        else:
            result[g] = n_per_group
    return result


def _rate_allocation(
    group_sizes: dict[str, int],
    rate: float | dict[str, float],
    *,
    min_n: int = 1,
    cap_at_population: bool = True,
) -> dict[str, int]:
    """Allocate n = ceil(rate * N_h) per group."""
    result: dict[str, int] = {}
    for g, size in group_sizes.items():
        if size == 0:
            result[g] = 0
            continue
        r = rate[g] if isinstance(rate, dict) else float(rate)
        if not (0 < r <= 1.0):
            raise ValueError(f"rate_allocation: rate must be in (0, 1], got {r} for group {g!r}.")
        n = max(min_n, math.ceil(r * size))
        result[g] = min(n, size) if cap_at_population else n
    return result


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


def allocate(
    group_sizes: dict[str, int],
    *,
    method: str = "proportional",
    n_total: int | None = None,
    n_per_group: int | None = None,
    rate: float | dict[str, float] | None = None,
    group_sds: dict[str, float] | None = None,
    min_n: int = 1,
    cap_at_population: bool = True,
) -> dict[str, int]:
    """
    Compute a per-group n mapping using a named allocation method.

    This is a pure function -- pass the returned dict directly as n= to
    any selection method (srs, pps_sys, etc.).

    Parameters
    ----------
    group_sizes       : {group_key: frame_count} from Selection.group_sizes().
    method            : "proportional" | "neyman" | "equal" | "rate"
    n_total           : target total sample size (proportional / neyman)
    n_per_group       : target per-group size (equal)
    rate              : sampling rate in (0, 1] -- scalar or per-group dict
    group_sds         : within-group SDs of target variable (neyman only)
    min_n             : minimum per non-empty group (default 1)
    cap_at_population : cap n_h <= N_h for WOR consistency (default True)

    Returns
    -------
    dict[str, int]
        Per-group sample sizes, ready to pass as n=.

    Examples
    --------
    Proportional allocation::

        sizes = sample.selection.group_sizes(by="region")
        n_map = sample.selection.allocate(sizes, method="proportional", n_total=500)
        sample = sample.selection.srs(n_map, by="region")

    Fixed 10% sampling rate::

        n_map = sample.selection.allocate(sizes, method="rate", rate=0.10)
        sample = sample.selection.pps_sys(n_map, by="region")

    Neyman with known within-stratum SDs::

        sds = {"North": 12.4, "South": 9.1, "West": 11.0}
        n_map = sample.selection.allocate(sizes, method="neyman",
                                          n_total=300, group_sds=sds)

    Equal allocation (50 per stratum, capped at stratum size)::

        n_map = sample.selection.allocate(sizes, method="equal", n_per_group=50)
    """
    _METHODS = ("proportional", "neyman", "equal", "rate")
    if method not in _METHODS:
        raise ValueError(f"allocate: unknown method {method!r}. Choose from {_METHODS}.")

    if method == "proportional":
        if n_total is None:
            raise ValueError("allocate(method='proportional') requires n_total=.")
        return _proportional_allocation(
            group_sizes, n_total, min_n=min_n, cap_at_population=cap_at_population
        )

    if method == "neyman":
        if n_total is None:
            raise ValueError("allocate(method='neyman') requires n_total=.")
        if group_sds is None:
            raise ValueError("allocate(method='neyman') requires group_sds=.")
        return _neyman_allocation(
            group_sizes, group_sds, n_total, min_n=min_n, cap_at_population=cap_at_population
        )

    if method == "equal":
        if n_per_group is None:
            raise ValueError("allocate(method='equal') requires n_per_group=.")
        return _equal_allocation(group_sizes, n_per_group, cap_at_population=cap_at_population)

    # method == "rate"
    if rate is None:
        raise ValueError("allocate(method='rate') requires rate=.")
    return _rate_allocation(group_sizes, rate, min_n=min_n, cap_at_population=cap_at_population)
