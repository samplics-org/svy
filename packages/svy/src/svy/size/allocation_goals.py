# src/svy/size/allocation_goals.py
"""SampleSize.allocate: split an overall sample size n across strata."""

from __future__ import annotations

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


def allocate(
    ss: SampleSize,
    n: int,
    *,
    pop_size: Mapping[object, Number],
    method: AllocationMethod = "proportional",
    sigma: Number | Mapping[object, Number] | None = None,
    power: Number = 1.0,
    min_n: int = 1,
    cap_at_population: bool = True,
) -> SampleSize:
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
            got=type(pop_size).__name__,
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
    ss._target = TargetAllocation(n=n, method=method, power=power)
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
