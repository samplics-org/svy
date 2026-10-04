# src/svy/engine/size_and_power/allocation.py
"""
Splitting an overall sample size n across strata, behind SampleSize.allocate().

proportional  n_h proportional to N_h (or N_h ** power)
neyman        n_h proportional to N_h * S_h
equal         n / H per stratum

All use Hamilton (largest-remainder) integer rounding, so the strata sum to n,
floor non-empty strata at min_n, and can cap n_h at N_h.
"""

from __future__ import annotations

import math

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
    n_total: int,
    *,
    min_n: int = 1,
    cap_at_population: bool = True,
) -> dict[str, int]:
    """Split n_total evenly over the non-empty strata (n / H each)."""
    if n_total <= 0:
        raise ValueError(f"equal_allocation: n_total must be > 0, got {n_total}.")
    total_pop = sum(group_sizes.values())
    if total_pop == 0:
        raise ValueError("equal_allocation: total frame size is 0.")
    if cap_at_population and n_total > total_pop:
        warn_no_sample(
            f"equal_allocation: n_total={n_total} exceeds the total frame "
            f"size of {total_pop}. Capping at frame size."
        )
        n_total = total_pop
    measure = {g: 1.0 if size > 0 else 0.0 for g, size in group_sizes.items()}
    return _allocate_by_measure(
        group_sizes,
        measure,
        n_total,
        min_n=min_n,
        cap_at_population=cap_at_population,
        label="equal_allocation",
    )
