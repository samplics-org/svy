# src/svy/core/containers.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec

from svy.core.types import Number


if TYPE_CHECKING:
    import polars as pl


################################################
#
# -------------- DISTRIBUTIONS -----------------
#
# ##############################################


class ChiSquare(msgspec.Struct, frozen=True):
    df: Number
    value: Number
    p_value: Number

    def to_polars(self) -> pl.DataFrame:
        """One-row frame: df, value, p_value."""
        return _chisq_frame(self)


def _chisq_frame(c: Any) -> pl.DataFrame:
    """Shared by ChiSquare and svy.serialize.to_polars (same field names).

    Floats throughout, as the serialized struct stores them, so an integer df
    reads the same from either side.
    """
    import polars as pl

    return pl.DataFrame(
        {"df": [float(c.df)], "value": [float(c.value)], "p_value": [float(c.p_value)]}
    )


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
