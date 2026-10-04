# src/svy/size/types.py
"""
Size-namespace type definitions.

Types live here so that:
  - svy/size/estimation_goals.py and svy/size/comparison_goals.py can import
    them without a cycle
  - Users can do: from svy.size import Size

Relationship to svy/core/types.py
----------------------------------
svy/core/types.py  — generic primitives and type aliases (Number, DomainScalarMap, …)
svy/size/types.py  — size-specific domain objects (Size, Target*, …)
"""

from __future__ import annotations

from typing import Any

import msgspec

from svy.core.types import Number


# =============================================================================
# Targets
# =============================================================================


# A goal's input is one number, or one per stratum when the goal is stratified.
NumberOrMap = Number | dict[Any, Number]


class TargetProp(msgspec.Struct, frozen=True, tag="prop"):
    """Every argument ``estimate_prop`` was called with."""

    p: NumberOrMap
    moe: NumberOrMap
    alpha: NumberOrMap = 0.05
    method: Any = "wald"
    pop_size: NumberOrMap | None = None
    deff: NumberOrMap = 1.0
    resp_rate: NumberOrMap = 1.0


class TargetMean(msgspec.Struct, frozen=True, tag="mean"):
    """Every argument ``estimate_mean`` was called with."""

    sigma: NumberOrMap
    moe: NumberOrMap
    alpha: NumberOrMap = 0.05
    method: Any = "wald"
    pop_size: NumberOrMap | None = None
    deff: NumberOrMap = 1.0
    resp_rate: NumberOrMap = 1.0


class TargetTwoProps(msgspec.Struct, frozen=True, tag="two_prop"):
    """Every argument ``compare_props`` was called with."""

    p1: NumberOrMap
    p2: NumberOrMap
    alloc_ratio: NumberOrMap = 1.0  # allocation ratio n2/n1
    alpha: NumberOrMap = 0.05
    power: NumberOrMap = 0.80
    method: str = "wald"
    var_mode: str = "alt-props"
    two_sides: bool = True
    delta: NumberOrMap = 0.0
    pop_size: NumberOrMap | None = None
    deff: NumberOrMap = 1.0
    resp_rate: NumberOrMap = 1.0


class TargetTwoMeans(msgspec.Struct, frozen=True, tag="two_mean"):
    """Every argument ``compare_means`` was called with."""

    mu1: NumberOrMap
    mu2: NumberOrMap
    sigma1: NumberOrMap | None = None
    sigma2: NumberOrMap | None = None
    alloc_ratio: NumberOrMap = 1.0  # allocation ratio n2/n1
    alpha: NumberOrMap = 0.05
    power: NumberOrMap = 0.80
    method: str = "wald"
    two_sides: bool = True
    delta: NumberOrMap = 0.0
    pop_size: NumberOrMap | None = None
    deff: NumberOrMap = 1.0
    resp_rate: NumberOrMap = 1.0


class TargetAllocation(msgspec.Struct, frozen=True, tag="allocation"):
    n: int
    method: str = "proportional"
    power: Number = 1.0
    #: The goal n was read from ("mean", "prop") when allocate() was called
    #: without n, and that goal's unrounded n; None when n was given.
    from_goal: str | None = None
    from_n: Number | None = None


Target = TargetProp | TargetMean | TargetTwoProps | TargetTwoMeans | TargetAllocation


# =============================================================================
# Result container
# =============================================================================


class Size(msgspec.Struct, frozen=True, tag="size"):
    #: The stratum as keyed in the goal's inputs (a tuple for several stratum
    #: columns); None when unstratified.
    stratum: Any = None
    n0: Number | tuple[Number, Number] = 0  # base (no DEFF/FPC/nonresponse)
    n1_deff: Number | tuple[Number, Number] | None = None  # after DEFF
    n2_fpc: Number | tuple[Number, Number] | None = None  # after FPC (if pop_size provided)
    n: Number | tuple[Number, Number] = 0  # final after nonresponse adjustment


class Allocation(msgspec.Struct, frozen=True, tag="allocation"):
    """One stratum's share of an allocated sample size.

    ``stratum`` is the key as given in ``pop_size`` (a value, or a tuple for
    several stratum columns); ``sigma`` is set for Neyman allocation.
    """

    stratum: object
    pop_size: Number
    n: int
    sigma: Number | None = None
