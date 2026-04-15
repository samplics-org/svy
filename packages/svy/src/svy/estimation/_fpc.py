# src/svy/estimation/_fpc.py
"""
Finite Population Correction (FPC) column construction.

Module-level functions that take an Estimation instance as first argument.
Used by taylor.py for FPC-corrected variance estimation.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

import polars as pl

from svy.core.design import PopSize
from svy.errors import DimensionError, MethodError


if TYPE_CHECKING:
    from svy.estimation.base import Estimation

log = logging.getLogger(__name__)


def compute_fpc_columns(
    est: Estimation,
    data: pl.DataFrame,
    pop_size: str | PopSize,
    strata_col: str | None,
    psu_col: str | None,
    ssu_col: str | None = None,
) -> tuple[pl.DataFrame, str | None, str | None]:
    """
    Compute per-row FPC columns from population size specification.

    For single-stage FPC (pop_size is str):
        fpc_psu = (N_h - n_h) / N_h per stratum, where n_h = number of PSUs in stratum h.
        If no strata, uses a single global FPC.

    For two-stage FPC (pop_size is PopSize):
        fpc_psu = (N_h - n_h) / N_h per stratum (PSU level)
        fpc_ssu = (M_hi - m_hi) / M_hi per PSU (SSU level)

    Returns
    -------
    tuple[pl.DataFrame, str | None, str | None]
        (data_with_fpc_cols, fpc_col_name, fpc_ssu_col_name)
    """
    _FPC_PSU_COL = "__svy_fpc_psu__"
    _FPC_SSU_COL = "__svy_fpc_ssu__"

    fpc_col_name: str | None = None
    fpc_ssu_col_name: str | None = None

    if isinstance(pop_size, str):
        pop_col = pop_size
        if pop_col not in data.columns:
            log.warning("FPC column '%s' not found in data; skipping FPC.", pop_col)
            return data, None, None

        data, fpc_col_name = build_fpc_psu_column(data, pop_col, strata_col, psu_col, _FPC_PSU_COL)

    elif isinstance(pop_size, PopSize):
        psu_pop_col = pop_size.psu
        ssu_pop_col = pop_size.ssu

        if psu_pop_col not in data.columns:
            log.warning("FPC PSU column '%s' not found in data; skipping FPC.", psu_pop_col)
            return data, None, None

        data, fpc_col_name = build_fpc_psu_column(
            data, psu_pop_col, strata_col, psu_col, _FPC_PSU_COL
        )

        if ssu_pop_col in data.columns and psu_col is not None:
            data, fpc_ssu_col_name = build_fpc_ssu_column(
                data, ssu_pop_col, psu_col, ssu_col, _FPC_SSU_COL
            )
        elif ssu_pop_col not in data.columns:
            log.warning("FPC SSU column '%s' not found in data; skipping SSU FPC.", ssu_pop_col)

    return data, fpc_col_name, fpc_ssu_col_name


def build_fpc_psu_column(
    data: pl.DataFrame,
    pop_col: str,
    strata_col: str | None,
    psu_col: str | None,
    out_col: str,
) -> tuple[pl.DataFrame, str]:
    """
    Build per-row FPC column for PSU level: (N_h - n_h) / N_h.

    N_h is the population size (from pop_col, constant within stratum).
    n_h is the number of distinct PSUs in stratum h (counted from data).
    If no PSU column, n_h = number of rows in the stratum.
    """
    _WHERE = "estimation.fpc"

    # Validate: all values must be >= 1 (population counts, not fractions)
    min_val = data[pop_col].cast(pl.Float64).min()
    if min_val is not None and min_val < 1:
        raise MethodError(
            title="Invalid FPC values",
            detail=f"FPC column '{pop_col}' contains values < 1 ({min_val}). "
            f"Population sizes must be counts (N >= 1), not sampling fractions.",
            code="FPC_INVALID_VALUES",
            where=_WHERE,
            param="pop_size",
            expected="N >= 1 (population counts)",
            got=min_val,
            hint="Use population counts (e.g., 500), not sampling fractions (e.g., 0.2).",
        )

    if strata_col is not None:
        # Validate: pop_col must be constant within each stratum
        n_unique_per_stratum = data.group_by(strata_col).agg(
            pl.col(pop_col).n_unique().alias("__nuniq__")
        )
        non_constant = n_unique_per_stratum.filter(pl.col("__nuniq__") > 1)
        if non_constant.height > 0:
            bad_strata = non_constant[strata_col].to_list()
            raise DimensionError(
                title="FPC not constant within strata",
                detail=f"FPC column '{pop_col}' varies within strata: {bad_strata}. "
                f"Population size must be the same for all observations in a stratum.",
                code="FPC_NOT_CONSTANT",
                where=_WHERE,
                param="pop_size",
                expected="constant N within each stratum",
                got=f"varying values in strata {bad_strata}",
                hint="Ensure each row in a stratum has the same population size value.",
            )

        # Count distinct PSUs per stratum
        if psu_col is not None:
            n_per_stratum = data.group_by(strata_col).agg(
                pl.col(psu_col).n_unique().alias("__n_h__")
            )
        else:
            n_per_stratum = data.group_by(strata_col).agg(pl.len().alias("__n_h__"))

        # Get N_h per stratum
        pop_per_stratum = data.group_by(strata_col).agg(
            pl.col(pop_col).first().cast(pl.Float64).alias("__N_h__")
        )

        # Validate: N_h >= n_h for every stratum
        check = n_per_stratum.join(pop_per_stratum, on=strata_col)
        bad = check.filter(pl.col("__N_h__") < pl.col("__n_h__").cast(pl.Float64))
        if bad.height > 0:
            bad_strata = bad[strata_col].to_list()
            raise DimensionError(
                title="FPC population size < sample size",
                detail=f"FPC column '{pop_col}' has population size < sample size "
                f"in strata: {bad_strata}.",
                code="FPC_POP_LT_SAMPLE",
                where=_WHERE,
                param="pop_size",
                expected="N >= n (population >= sampled PSUs)",
                got=f"N < n in strata {bad_strata}",
                hint="Population size N must be >= the number of sampled PSUs n in every stratum.",
            )

        # Compute FPC
        stratum_fpc = (
            n_per_stratum.join(pop_per_stratum, on=strata_col)
            .with_columns(
                ((pl.col("__N_h__") - pl.col("__n_h__").cast(pl.Float64)) / pl.col("__N_h__"))
                .clip(0.0, 1.0)
                .alias(out_col)
            )
            .select([strata_col, out_col])
        )

        data = data.join(stratum_fpc, on=strata_col, how="left")
    else:
        # Unstratified: pop_col must be constant across all rows
        n_unique = data[pop_col].n_unique()
        if n_unique > 1:
            raise DimensionError(
                title="FPC not constant in unstratified design",
                detail=f"FPC column '{pop_col}' has {n_unique} distinct values, "
                f"but must be constant for unstratified designs.",
                code="FPC_NOT_CONSTANT",
                where=_WHERE,
                param="pop_size",
                expected="constant N across all observations",
                got=f"{n_unique} distinct values",
                hint="If different groups have different population sizes, "
                "specify strata in the Design.",
            )

        n_h = data[psu_col].n_unique() if psu_col is not None else data.height
        n_pop = data[pop_col].cast(pl.Float64).first()

        if n_pop is not None and n_pop < n_h:
            raise DimensionError(
                title="FPC population size < sample size",
                detail=f"FPC column '{pop_col}' has population size ({n_pop}) < "
                f"sample size ({n_h}).",
                code="FPC_POP_LT_SAMPLE",
                where=_WHERE,
                param="pop_size",
                expected=f"N >= {n_h}",
                got=n_pop,
                hint="Population size N must be >= the number of sampled PSUs n.",
            )

        if n_pop is not None and n_pop > 0:
            fpc_val = max(0.0, min(1.0, (n_pop - n_h) / n_pop))
        else:
            fpc_val = 1.0
        data = data.with_columns(pl.lit(fpc_val).alias(out_col))

    return data, out_col


def build_fpc_ssu_column(
    data: pl.DataFrame,
    pop_col: str,
    psu_col: str,
    ssu_col: str | None,
    out_col: str,
) -> tuple[pl.DataFrame, str]:
    """
    Build per-row FPC column for SSU level: (M_hi - m_hi) / M_hi.

    M_hi is the SSU population size within PSU i (from pop_col, constant within PSU).
    m_hi is the number of distinct SSUs sampled in PSU i.
    If ssu_col is None, falls back to counting rows per PSU.
    """
    _WHERE = "estimation.fpc"

    # Validate: all values must be >= 1 (population counts)
    min_val = data[pop_col].cast(pl.Float64).min()
    if min_val is not None and min_val < 1:
        raise MethodError(
            title="Invalid FPC values",
            detail=f"FPC column '{pop_col}' contains values < 1 ({min_val}). "
            f"Population sizes must be counts (N >= 1), not sampling fractions.",
            code="FPC_INVALID_VALUES",
            where=_WHERE,
            param="pop_size.ssu",
            expected="M >= 1 (population counts)",
            got=min_val,
            hint="Use population counts (e.g., 50), not sampling fractions (e.g., 0.3).",
        )

    # Validate: pop_col must be constant within each PSU
    n_unique_per_psu = data.group_by(psu_col).agg(pl.col(pop_col).n_unique().alias("__nuniq__"))
    non_constant = n_unique_per_psu.filter(pl.col("__nuniq__") > 1)
    if non_constant.height > 0:
        bad_psus = non_constant[psu_col].to_list()
        raise DimensionError(
            title="FPC not constant within PSUs",
            detail=f"FPC column '{pop_col}' varies within PSUs: {bad_psus}. "
            f"SSU population size must be the same for all observations in a PSU.",
            code="FPC_NOT_CONSTANT",
            where=_WHERE,
            param="pop_size.ssu",
            expected="constant M within each PSU",
            got=f"varying values in PSUs {bad_psus}",
            hint="Ensure each row in a PSU has the same SSU population size value.",
        )

    # Count distinct SSUs per PSU (or rows if no SSU column)
    if ssu_col is not None and ssu_col in data.columns:
        n_per_psu = data.group_by(psu_col).agg(pl.col(ssu_col).n_unique().alias("__m_hi__"))
    else:
        n_per_psu = data.group_by(psu_col).agg(pl.len().alias("__m_hi__"))

    # Get M_hi per PSU
    pop_per_psu = data.group_by(psu_col).agg(
        pl.col(pop_col).first().cast(pl.Float64).alias("__M_hi__")
    )

    # Validate: M_hi >= m_hi for every PSU
    check = n_per_psu.join(pop_per_psu, on=psu_col)
    bad = check.filter(pl.col("__M_hi__") < pl.col("__m_hi__").cast(pl.Float64))
    if bad.height > 0:
        bad_psus = bad[psu_col].to_list()
        raise DimensionError(
            title="FPC population size < sample size",
            detail=f"FPC column '{pop_col}' has population size < sample size "
            f"in PSUs: {bad_psus}.",
            code="FPC_POP_LT_SAMPLE",
            where=_WHERE,
            param="pop_size.ssu",
            expected="M >= m (population >= sampled SSUs)",
            got=f"M < m in PSUs {bad_psus}",
            hint="SSU population size M must be >= the number of sampled SSUs m in every PSU.",
        )

    # Compute FPC
    psu_fpc = (
        n_per_psu.join(pop_per_psu, on=psu_col)
        .with_columns(
            ((pl.col("__M_hi__") - pl.col("__m_hi__").cast(pl.Float64)) / pl.col("__M_hi__"))
            .clip(0.0, 1.0)
            .alias(out_col)
        )
        .select([psu_col, out_col])
    )

    data = data.join(psu_fpc, on=psu_col, how="left")
    return data, out_col
