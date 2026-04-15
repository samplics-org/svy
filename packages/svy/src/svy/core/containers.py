# src/svy/core/containers.py
from __future__ import annotations

import msgspec

from svy.core.types import Number


################################################
#
# -------------- DISTRIBUTIONS -----------------
#
# ##############################################


class ChiSquare(msgspec.Struct, frozen=True):
    df: Number
    value: Number
    p_value: Number


class FDist(msgspec.Struct, frozen=True):
    df_num: Number
    df_den: Number
    value: Number
    p_value: Number


class TDist(msgspec.Struct, frozen=True):
    df: Number
    value: Number
    p_value: Number


################################################
#
# ------------- OTHER CONTAINERS ---------------
#
# ##############################################


# class RepWeights(msgspec.Struct):
#     method: EstMethod | None = None
#     weights: list[str] = []
#     n_reps: int = 0
#     fay_coef: float = 0.0
#     degrees_of_freedom: int = 0
