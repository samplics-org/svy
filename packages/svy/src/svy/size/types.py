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

import msgspec

from svy.core.types import Number


# =============================================================================
# Targets
# =============================================================================


class TargetProp(msgspec.Struct, frozen=True, tag="prop"):
    p: Number
    moe: Number
    alpha: Number = 0.05


class TargetMean(msgspec.Struct, frozen=True, tag="mean"):
    sigma: Number
    moe: Number
    alpha: Number = 0.05


class TargetTwoProps(msgspec.Struct, frozen=True, tag="two_prop"):
    p1: Number
    p2: Number
    alloc_ratio: Number = 1.0  # allocation ratio n2/n1
    alpha: Number = 0.05
    power: Number = 0.80


class TargetTwoMeans(msgspec.Struct, frozen=True, tag="two_mean"):
    mu1: Number
    mu2: Number
    alloc_ratio: Number = 1.0  # allocation ratio n2/n1
    alpha: Number = 0.05
    power: Number = 0.80


Target = TargetProp | TargetMean | TargetTwoProps | TargetTwoMeans


# =============================================================================
# Result container
# =============================================================================


class Size(msgspec.Struct, frozen=True, tag="size"):
    stratum: str | None = None
    n0: Number | tuple[Number, Number] = 0  # base (no FPC/DEFF/nonresponse)
    n1_fpc: Number | tuple[Number, Number] | None = None  # after FPC (if pop_size provided)
    n2_deff: Number | tuple[Number, Number] | None = None  # after DEFF
    n: Number | tuple[Number, Number] = 0  # final after nonresponse adjustment
