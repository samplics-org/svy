# src/svy/core/containers.py
from __future__ import annotations

import math

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


def _fmt_df(df: Number) -> str:
    return str(int(df)) if float(df).is_integer() else f"{df:.2f}"


def _fmt_test(name: str, dfs: tuple[Number, ...], value: Number, p_value: Number) -> str:
    """``F(3.30, 49.48) = 11.67, p < 0.001``."""
    stat = "nan" if math.isnan(value) else f"{value:.2f}"
    if math.isnan(p_value):
        p = "p = nan"
    elif p_value < 0.001:
        p = "p < 0.001"
    else:
        p = f"p = {p_value:.3f}"
    return f"{name}({', '.join(_fmt_df(d) for d in dfs)}) = {stat}, {p}"


class ChiSquare(msgspec.Struct, frozen=True):
    df: Number
    value: Number
    p_value: Number

    def __str__(self) -> str:
        return _fmt_test("chi2", (self.df,), self.value, self.p_value)

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

    def __str__(self) -> str:
        return _fmt_test("F", (self.df_num, self.df_den), self.value, self.p_value)


class TDist(msgspec.Struct, frozen=True):
    df: Number
    value: Number
    p_value: Number

    def __str__(self) -> str:
        return _fmt_test("t", (self.df,), self.value, self.p_value)


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
