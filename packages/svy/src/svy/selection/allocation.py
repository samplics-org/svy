# src/svy/selection/allocation.py
"""
Sample-size allocation helpers.

Pure functions that derive per-group n mappings from a target total size
or sampling rate.  Results feed directly into any selection method as n=.

Methods
-------
proportional  n_h proportional to N_h (or N_h ** power)
neyman        optimal allocation (n_h proportional to N_h * SD_h)
size          n_h proportional to the group's total measure of size (** power)
equal         equal n per stratum
rate          fixed sampling rate per stratum

All use Hamilton (largest-remainder) integer rounding so that
sum(result.values()) == n_total exactly for proportional / neyman / size.
"""

from __future__ import annotations

import math

from typing import Literal, get_args

from svy.core.warnings import warn_no_sample


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
            warn_no_sample(
                f"{label}: total allocation reduced by {surplus} because every "
                "group is capped at its frame size."
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


def _allocate_by_measure(
    group_sizes: dict[str, int],
    measure: dict[str, float],
    n_total: int,
    *,
    min_n: int,
    cap_at_population: bool,
    label: str,
) -> dict[str, int]:
    """
    Allocate n_total in proportion to `measure`, at least min_n per non-empty
    group, Hamilton largest-remainder rounding, optionally capped at N_h.
    """
    import numpy as np

    groups = list(group_sizes.keys())
    N = np.array([group_sizes[g] for g in groups], dtype=np.float64)
    m = np.array([measure.get(g, 0.0) for g in groups], dtype=np.float64)

    raw = m / m.sum() * n_total
    floored = np.where(N > 0, np.maximum(np.floor(raw), min_n), 0.0)
    remainder = n_total - int(floored.sum())
    if remainder < 0:
        warn_no_sample(
            f"{label}: min_n={min_n} floors exceed n_total="
            f"{n_total}; min_n is not honored and allocation is rescaled."
        )
        floored = np.floor(raw * (n_total / raw.sum()))
        remainder = n_total - int(floored.sum())

    fractional = raw - floored
    order = np.argsort(-fractional)
    for i in range(max(0, remainder)):
        floored[order[i % len(order)]] += 1

    result = {g: int(floored[i]) for i, g in enumerate(groups)}
    if cap_at_population:
        result = _apply_population_caps(result, group_sizes, measure, label=label)
    return result


def _check_power(power: float, label: str) -> float:
    power = float(power)
    if not math.isfinite(power) or power < 0:
        raise ValueError(f"{label}: power must be a finite number >= 0, got {power}.")
    return power


def _proportional_allocation(
    group_sizes: dict[str, int],
    n_total: int,
    *,
    power: float = 1.0,
    min_n: int = 1,
    cap_at_population: bool = True,
) -> dict[str, int]:
    """
    Allocate n_total units proportional to group size raised to `power`.

    Parameters
    ----------
    group_sizes       : {group_key: frame_count}
    n_total           : target overall sample size
    power             : n_h proportional to N_h ** power (default 1; 0.5 is
                        square-root allocation)
    min_n             : floor allocation per non-empty group (default 1)
    cap_at_population : cap n_h <= N_h, redistributing surplus (default True)
    """
    power = _check_power(power, "proportional_allocation")
    total_pop = sum(group_sizes.values())
    if total_pop == 0:
        raise ValueError("proportional_allocation: total frame size is 0.")
    if n_total <= 0:
        raise ValueError(f"proportional_allocation: n_total must be > 0, got {n_total}.")
    if n_total > total_pop:
        warn_no_sample(
            f"proportional_allocation: n_total={n_total} exceeds the total frame "
            f"size of {total_pop}. Capping at frame size."
        )
        n_total = total_pop

    measure = {g: float(n) ** power if n > 0 else 0.0 for g, n in group_sizes.items()}
    return _allocate_by_measure(
        group_sizes,
        measure,
        n_total,
        min_n=min_n,
        cap_at_population=cap_at_population,
        label="proportional_allocation",
    )


def _size_allocation(
    group_sizes: dict[str, int],
    group_mos: dict[str, float],
    n_total: int,
    *,
    power: float = 1.0,
    min_n: int = 1,
    cap_at_population: bool = True,
) -> dict[str, int]:
    """
    Allocate n_total proportional to each group's total measure of size,
    raised to `power`.

    Parameters
    ----------
    group_sizes       : {group_key: frame_count}
    group_mos         : {group_key: total measure of size}; must cover every
                        non-empty group
    n_total           : target overall sample size
    power             : n_h proportional to MOS_h ** power (default 1)
    min_n             : floor per non-empty group
    cap_at_population : cap n_h <= N_h, redistributing surplus (default True)
    """
    power = _check_power(power, "size_allocation")
    groups = list(group_sizes.keys())
    missing = [g for g in groups if group_sizes[g] > 0 and g not in group_mos]
    if missing:
        raise ValueError(
            f"size_allocation: group_mos is missing entries for non-empty "
            f"groups {missing!r}. Provide a size total for every group."
        )
    bad = {
        g: v
        for g, v in group_mos.items()
        if g in group_sizes and not (math.isfinite(float(v)) and float(v) >= 0)
    }
    if bad:
        raise ValueError(f"size_allocation: group_mos must be finite and >= 0; got {bad!r}.")
    if n_total <= 0:
        raise ValueError(f"size_allocation: n_total must be > 0, got {n_total}.")
    # A group with no size cannot be drawn with probability proportional to
    # size, so it gets 0 rather than the min_n floor.
    drawable = {g: group_sizes[g] for g in groups if group_sizes[g] > 0 and group_mos[g] > 0}
    if not drawable:
        raise ValueError(
            "size_allocation: every group's size total is zero. Check the measure of size column."
        )
    total_pop = sum(drawable.values())
    if cap_at_population and n_total > total_pop:
        warn_no_sample(
            f"size_allocation: n_total={n_total} exceeds the {total_pop} units in "
            "groups with a positive size total. Capping at that size."
        )
        n_total = total_pop

    out = _allocate_by_measure(
        drawable,
        {g: float(group_mos[g]) ** power for g in drawable},
        n_total,
        min_n=min_n,
        cap_at_population=cap_at_population,
        label="size_allocation",
    )
    return {g: out.get(g, 0) for g in groups}


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
        warn_no_sample(
            f"neyman_allocation: n_total={n_total} exceeds the total frame "
            f"size of {total_pop}. Capping at frame size."
        )
        n_total = total_pop

    measure = {g: float(group_sizes[g]) * float(group_sds.get(g, 0.0)) for g in groups}
    if sum(measure.values()) == 0:
        raise ValueError(
            "neyman_allocation: all N*SD products are zero. "
            "Check that group_sds contains positive values."
        )
    return _allocate_by_measure(
        group_sizes,
        measure,
        n_total,
        min_n=min_n,
        cap_at_population=cap_at_population,
        label="neyman_allocation",
    )


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


AllocationMethod = Literal["proportional", "neyman", "size", "equal", "rate"]


def allocate(
    group_sizes: dict[str, int],
    *,
    method: AllocationMethod = "proportional",
    n_total: int | None = None,
    n_per_group: int | None = None,
    rate: float | dict[str, float] | None = None,
    group_sds: dict[str, float] | None = None,
    group_mos: dict[str, float] | None = None,
    power: float = 1.0,
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
    method            : "proportional" | "neyman" | "size" | "equal" | "rate"
    n_total           : target total sample size (proportional / neyman / size)
    n_per_group       : target per-group size (equal)
    rate              : sampling rate in (0, 1] -- scalar or per-group dict
    group_sds         : within-group SDs of target variable (neyman only)
    group_mos         : per-group totals of the measure of size, from
                        Selection.group_totals() (size only)
    power             : n_h proportional to N_h ** power (proportional) or
                        MOS_h ** power (size); default 1, 0.5 is square-root
                        allocation
    min_n             : minimum per non-empty group (default 1)
    cap_at_population : cap n_h <= N_h for WOR consistency (default True)

    Returns
    -------
    dict[str, int]
        Per-group sample sizes, ready to pass as n=.

    Examples
    --------
    Proportional allocation::

        sizes = sample.sampling.group_sizes(by="region")
        n_map = sample.sampling.allocate(sizes, method="proportional", n_total=500)
        sample = sample.sampling.srs(n_map, by="region")

    Square-root allocation (n_h proportional to N_h ** 0.5)::

        n_map = sample.sampling.allocate(sizes, n_total=500, power=0.5)

    In proportion to each stratum's total measure of size, for a PPS design::

        mos = sample.sampling.group_totals("hh_count", by="region")
        n_map = sample.sampling.allocate(sizes, method="size", n_total=60, group_mos=mos)
        sample = sample.sampling.pps_sys(n_map, by="region")

    Fixed 10% sampling rate::

        n_map = sample.sampling.allocate(sizes, method="rate", rate=0.10)
        sample = sample.sampling.pps_sys(n_map, by="region")

    Neyman with known within-stratum SDs::

        sds = {"North": 12.4, "South": 9.1, "West": 11.0}
        n_map = sample.sampling.allocate(sizes, method="neyman",
                                          n_total=300, group_sds=sds)

    Equal allocation (50 per stratum, capped at stratum size)::

        n_map = sample.sampling.allocate(sizes, method="equal", n_per_group=50)
    """
    _METHODS = get_args(AllocationMethod)
    if method not in _METHODS:
        raise ValueError(f"allocate: unknown method {method!r}. Choose from {_METHODS}.")
    if power != 1.0 and method not in ("proportional", "size"):
        raise ValueError(
            f"allocate(method={method!r}) does not take power=; it applies to "
            "'proportional' and 'size'."
        )
    if group_mos is not None and method != "size":
        raise ValueError(
            f"allocate(method={method!r}) does not take group_mos=; use method='size'."
        )

    if method == "proportional":
        if n_total is None:
            raise ValueError("allocate(method='proportional') requires n_total=.")
        return _proportional_allocation(
            group_sizes, n_total, power=power, min_n=min_n, cap_at_population=cap_at_population
        )

    if method == "size":
        if n_total is None:
            raise ValueError("allocate(method='size') requires n_total=.")
        if group_mos is None:
            raise ValueError(
                "allocate(method='size') requires group_mos=, the per-group totals "
                "of the measure of size: group_mos=sample.sampling.group_totals(mos, by=...)."
            )
        return _size_allocation(
            group_sizes,
            group_mos,
            n_total,
            power=power,
            min_n=min_n,
            cap_at_population=cap_at_population,
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
