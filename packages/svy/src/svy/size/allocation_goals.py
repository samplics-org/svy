# src/svy/size/allocation_goals.py
"""SampleSize.allocate: split an overall sample size n across strata."""

from __future__ import annotations

import math

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, get_args

from svy.core.types import Number
from svy.engine.size_and_power.allocation import (
    _equal_allocation,
    _neyman_allocation,
    _proportional_allocation,
)
from svy.errors import MethodError
from svy.size.types import Allocation, TargetAllocation


if TYPE_CHECKING:
    from svy.size.base import SampleSize


AllocationMethod = Literal["proportional", "neyman", "equal"]

_WHERE = "SampleSize.allocate"


def _engine_error(ex: ValueError) -> MethodError:
    return MethodError(
        title="Allocation not possible",
        detail=str(ex),
        code="ALLOCATION_INVALID",
        where=_WHERE,
    )


def _n_from_previous_goal(ss: SampleSize) -> tuple[int, str | None, Number]:
    """The overall n of the goal ``ss`` holds, rounded up, for ``allocate()`` without n."""
    if ss._allocation is not None and ss._target is not None:
        return ss._target.n, None, ss._target.n
    sizes = ss._iter_sizes()
    goal = getattr(ss._param, "name", None)
    if not sizes:
        raise MethodError.not_applicable(
            where=_WHERE,
            method="allocate()",
            reason="no n to split: pass n, or compute one first",
            param="n",
            hint="svy.SampleSize().estimate_mean(...).allocate(pop_size=...), or allocate(n, ...).",
        )
    if len(sizes) > 1 or sizes[0].stratum is not None:
        raise MethodError.not_applicable(
            where=_WHERE,
            method="allocate()",
            reason="the previous goal is stratified: it already gives n per stratum",
            param="n",
            hint="Compute an overall n (scalar inputs), or pass n.",
        )
    total = sizes[0].n
    if isinstance(total, tuple):
        raise MethodError.not_applicable(
            where=_WHERE,
            method="allocate()",
            reason="the previous goal gives one n per comparison group, not an overall n",
            param="n",
            hint="Pass n.",
        )
    return math.ceil(float(total) - 1e-9), (goal.lower() if goal else None), total


def allocate(
    ss: SampleSize,
    n: int | None = None,
    *,
    pop_size: Mapping[object, Number],
    method: AllocationMethod = "proportional",
    sigma: Number | Mapping[object, Number] | None = None,
    power: Number = 1.0,
    min_n: int = 1,
    cap_at_population: bool = True,
) -> SampleSize:
    from_goal: str | None = None
    from_n: Number | None = None
    if n is None:
        n, from_goal, from_n = _n_from_previous_goal(ss)
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise MethodError.invalid_type(
            where=_WHERE, param="n", got=n, expected="a positive integer"
        )
    if method not in get_args(AllocationMethod):
        raise MethodError.invalid_choice(
            where=_WHERE, param="method", got=method, allowed=list(get_args(AllocationMethod))
        )
    if not isinstance(pop_size, Mapping) or not pop_size:
        raise MethodError.invalid_type(
            where=_WHERE,
            param="pop_size",
            got=pop_size,
            expected="a non-empty dict {stratum: N_h}",
            hint="Allocation splits n across strata, so pop_size names them: "
            "{'North': 52_000, 'South': 31_000}.",
        )
    if power != 1.0 and method != "proportional":
        raise MethodError.not_applicable(
            where=_WHERE,
            method=f"allocate(method={method!r})",
            reason="power= applies to proportional allocation (n_h ~ N_h ** power)",
            param="power",
        )
    if method == "neyman":
        if sigma is None:
            raise MethodError.not_applicable(
                where=_WHERE,
                method="allocate(method='neyman')",
                reason="Neyman allocation needs each stratum's standard deviation S_h",
                param="sigma",
                hint="Pass sigma={stratum: S_h}, e.g. from a previous survey.",
            )
        if isinstance(sigma, Mapping):
            missing = [k for k in pop_size if k not in sigma and pop_size[k] > 0]
            extra = [k for k in sigma if k not in pop_size]
            if missing or extra:
                raise MethodError.invalid_choice(
                    where=_WHERE,
                    param="sigma",
                    got=list(sigma),
                    allowed=list(pop_size),
                    hint="sigma needs every non-empty stratum of pop_size, and no others.",
                )
        else:
            sigma = dict.fromkeys(pop_size, sigma)
    elif sigma is not None:
        raise MethodError.not_applicable(
            where=_WHERE,
            method=f"allocate(method={method!r})",
            reason="sigma= is used by Neyman allocation only",
            param="sigma",
        )

    bad = {
        k: v
        for k, v in pop_size.items()
        if isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not v >= 0
        or v == float("inf")
    }
    if bad:
        raise MethodError.invalid_type(
            where=_WHERE,
            param="pop_size",
            got=bad,
            expected="finite numbers >= 0: unit counts or size totals per stratum",
        )

    # The engine works on string keys; keep the caller's keys for the result.
    keys = {str(k): k for k in pop_size}
    if len(keys) != len(pop_size):
        raise MethodError.invalid_choice(
            where=_WHERE,
            param="pop_size",
            got=list(pop_size),
            allowed=["distinct strata"],
            hint="Two strata print the same; give them distinct keys.",
        )
    sizes = {s: pop_size[k] for s, k in keys.items()}
    try:
        if method == "proportional":
            out = _proportional_allocation(
                sizes, n, power=power, min_n=min_n, cap_at_population=cap_at_population
            )
        elif method == "neyman":
            assert isinstance(sigma, Mapping)  # noqa: S101
            sds = {s: float(sigma[k]) for s, k in keys.items() if k in sigma}
            out = _neyman_allocation(
                sizes, sds, n, min_n=min_n, cap_at_population=cap_at_population
            )
        else:
            out = _equal_allocation(sizes, n, min_n=min_n, cap_at_population=cap_at_population)
    except ValueError as ex:
        raise _engine_error(ex) from ex

    ss._param = None
    ss._target = TargetAllocation(
        n=n, method=method, power=power, from_goal=from_goal, from_n=from_n
    )
    ss._size = None
    ss._allocation = [
        Allocation(
            stratum=k,
            pop_size=pop_size[k],
            n=out[s],
            sigma=None if method != "neyman" else sigma.get(k),  # type: ignore[union-attr]
        )
        for s, k in keys.items()
    ]
    return ss
