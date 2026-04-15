# src/svy/engine/categorical/ttesting.py
"""Comparison module

The module implements comparisons of groups using design-based t-tests.
"""

from __future__ import annotations

import math

from typing import Dict, Union

import numpy as np

from scipy.stats import t

from svy.categorical.ttest import (
    DiffEst,
    GroupLevels,
    TtestEst,
    TTestOneGroup,
    TTestStats,
    TTestTwoGroups,
)
from svy.core.enumerations import PopParam
from svy.core.types import Array, Number
from svy.engine.estimation.taylor import _estimate_taylor


def _get_subpop_df(
    stratum: np.ndarray | None,
    psu: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """
    Calculates Design Degrees of Freedom for a subpopulation.
    df = n_PSU - n_Strata
    """
    if mask is not None:
        p_sub = psu[mask]
        s_sub = stratum[mask] if stratum is not None else None

        # If the mask is empty, DF is 0
        if p_sub.size == 0:
            return 0.0

        # We reuse the robust logic from taylor.py which handles singletons/etc if needed,
        # or stick to the simple definition used in most software:
        # DF = (Number of PSUs with observations) - (Number of Strata with observations)
        n_psu = len(np.unique(p_sub))
        n_strata = len(np.unique(s_sub)) if s_sub is not None else 1
    else:
        n_psu = len(np.unique(psu))
        n_strata = len(np.unique(stratum)) if stratum is not None else 1

    return max(1.0, float(n_psu - n_strata))


def _compute_p_value(p_left: float, alternative: str) -> float:
    """Compute p-value based on alternative hypothesis."""
    if alternative == "less":
        return p_left
    elif alternative == "greater":
        return 1 - p_left
    else:  # "two-sided"
        return 2 * min(p_left, 1 - p_left)


def _ttest_by(
    y: Array,
    y_name: str,
    mean_h0: Number,
    group: Array | None,
    group_name: str | None,
    wgt: Array,
    stratum: Array | None,
    psu: Array,
    ssu: Array | None,
    by: Array | None = None,
    by_name: str | None = None,
    fpc: Union[Dict, float] = 1,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> list[TTestOneGroup | TTestTwoGroups]:
    """
    Vectorized T-Test execution across domains.
    """
    if by is None:
        # Single test case
        return [
            _ttest(
                y=y,
                y_name=y_name,
                mean_h0=mean_h0,
                group=group,
                group_name=group_name,
                wgt=wgt,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
                alpha=alpha,
                alternative=alternative,
            )
        ]

    # --- Vectorized Estimation ---
    # Instead of looping and slicing data (slow), we estimate all domains at once.
    # We ignore the covariance matrix here because one-sample tests are univariate.
    est_list, _ = _estimate_taylor(
        param=PopParam.MEAN,
        y=y,
        y_name=y_name,
        wgt=wgt,
        stratum=stratum,
        psu=psu,
        ssu=ssu,
        by=by,
        by_name=by_name,
        fpc=fpc,
        alpha=alpha,
    )

    # Convert inputs to numpy for fast DF calculation filtering
    # We still need to calculate DF per domain, as DF depends on the specific
    # PSUs/Strata present in that domain's subpopulation.
    by_arr = np.asarray(by)
    psu_arr = np.asarray(psu)
    stratum_arr = np.asarray(stratum) if stratum is not None else None

    results = []

    # Map estimates back to TTestOneGroup objects
    # Note: _ttest_by currently only supports One-Sample tests within domains.
    # If `group` (two-sample) was provided with `by`, the complexity increases (2-way split).
    # Assuming standard `by` usage implies "Test H0 for each level of By".

    for est in est_list:
        # Calculate DF for this specific domain (subtract 1 for t-test)
        by_val = est.by_level
        mask = by_arr == by_val

        df = _get_subpop_df(stratum_arr, psu_arr, mask) - 1

        # Calculate difference and t-statistic
        diff_value = est.est - float(mean_h0)
        t_value = diff_value / est.se if est.se > 0 else 0.0

        if est.se <= 0:
            # Handle edge cases
            if est.est == mean_h0:
                t_value = 0.0
            else:
                t_value = float("inf") if est.est > mean_h0 else float("-inf")

        p_left = float(t.cdf(t_value, df))
        p_value = _compute_p_value(p_left, alternative)

        stats = TTestStats(df=df, t=t_value, p_value=p_value)

        # Calculate CI using t-test df (not estimation df)
        t_crit = float(t.ppf(1 - alpha / 2, df))
        diff_lci = diff_value - t_crit * est.se
        diff_uci = diff_value + t_crit * est.se

        # Create DiffEst for the difference
        diff_est = DiffEst(
            y=y_name,
            diff=diff_value,
            se=est.se,
            lci=diff_lci,
            uci=diff_uci,
            by=est.by,  # type: ignore[arg-type]
            by_level=est.by_level,  # type: ignore[arg-type]
        )

        # Construct Result
        res_obj = TTestOneGroup(
            y=y_name,
            mean_h0=mean_h0,
            alternative=alternative,  # type: ignore[arg-type]
            diff=[diff_est],
            estimates=[TtestEst.from_param(est)],
            stats=stats,
            alpha=alpha,
        )
        results.append(res_obj)

    return results


def _ttest(
    y: Array,
    y_name: str,
    mean_h0: Number,
    group: Array | None,
    group_name: str | None,
    wgt: Array,
    stratum: Array | None,
    psu: Array,
    ssu: Array | None,
    fpc: Union[Dict, float] = 1,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TTestOneGroup | TTestTwoGroups:
    if group is None:
        return _one_sample_one_group(
            y=y,
            y_name=y_name,
            mean_h0=mean_h0,
            wgt=wgt,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            fpc=fpc,
            alpha=alpha,
            alternative=alternative,
        )
    else:
        if group_name is None:
            raise ValueError("group_name must be provided when group is provided")

        return _one_sample_two_groups(
            y=y,
            y_name=y_name,
            mean_h0=mean_h0,
            group=group,
            group_name=group_name,
            wgt=wgt,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            fpc=fpc,
            alpha=alpha,
            alternative=alternative,
        )


def _one_sample_one_group(
    y: Array,
    y_name: str,
    mean_h0: Number,
    wgt: Array,
    stratum: Array | None,
    psu: Array,
    ssu: Array | None,
    fpc: Union[Dict, float] = 1,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TTestOneGroup:
    # 1. Estimate
    est_list, _ = _estimate_taylor(
        param=PopParam.MEAN,
        y=y,
        y_name=y_name,
        wgt=wgt,
        stratum=stratum,
        psu=psu,
        ssu=ssu,
        fpc=fpc,
        alpha=alpha,
    )
    est = est_list[0]

    # 2. Calculate difference and t-statistic
    diff_value = est.est - float(mean_h0)
    t_val = diff_value / est.se if est.se > 0 else 0.0

    # Use t-test DF (subtract 1 to match R's svyttest)
    df = _get_subpop_df(np.asarray(stratum) if stratum is not None else None, np.asarray(psu)) - 1

    p_left = float(t.cdf(t_val, df))
    p_value = _compute_p_value(p_left, alternative)

    # 3. Calculate CI using t-test df (not estimation df)
    t_crit = float(t.ppf(1 - alpha / 2, df))
    diff_lci = diff_value - t_crit * est.se
    diff_uci = diff_value + t_crit * est.se

    # 4. Create DiffEst for the difference
    diff_est = DiffEst(
        y=y_name,
        diff=diff_value,
        se=est.se,
        lci=diff_lci,
        uci=diff_uci,
        by=est.by,  # type: ignore[arg-type]
        by_level=est.by_level,  # type: ignore[arg-type]
    )

    return TTestOneGroup(
        y=y_name,
        mean_h0=mean_h0,
        alternative=alternative,  # type: ignore[arg-type]
        diff=[diff_est],
        estimates=[TtestEst.from_param(est)],
        stats=TTestStats(df=df, t=t_val, p_value=p_value),
        alpha=alpha,
    )


def _one_sample_two_groups(
    y: Array,
    y_name: str,
    mean_h0: float,
    group: Array,
    group_name: str,
    wgt: Array,
    stratum: Array | None,
    psu: Array,
    ssu: Array | None,
    fpc: Union[Dict, float] = 1,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TTestTwoGroups:
    # 1. Estimate Groups AND Covariance
    # We pass 'by=group' to get both estimates and their relationship matrix
    est_list, cov_mat = _estimate_taylor(
        param=PopParam.MEAN,
        y=y,
        y_name=y_name,
        wgt=wgt,
        stratum=stratum,
        psu=psu,
        ssu=ssu,
        by=group,
        by_name=group_name,
        fpc=fpc,
        alpha=alpha,
    )

    if len(est_list) != 2:
        raise ValueError(f"T-test requires exactly 2 groups, found {len(est_list)}.")

    est1, est2 = est_list[0], est_list[1]

    # 2. Calculate Difference Variance with Covariance
    # Var(A - B) = Var(A) + Var(B) - 2 * Cov(A, B)
    # cov_mat is typically 2x2: [[Var1, Cov12], [Cov21, Var2]]

    # Ensure cov_mat is numpy array (it should be from estimate_taylor)
    cov_mat = np.asarray(cov_mat)

    var1 = cov_mat[0, 0]
    var2 = cov_mat[1, 1]
    cov12 = cov_mat[0, 1]

    var_diff = var1 + var2 - 2 * cov12
    se_diff = math.sqrt(var_diff) if var_diff > 0 else 0.0

    # Difference: est2 - est1 (to match R's ordering convention)
    diff_value = (est2.est - est1.est) - mean_h0

    # 3. Degrees of Freedom (subtract 1 for t-test)
    df = _get_subpop_df(np.asarray(stratum) if stratum is not None else None, np.asarray(psu)) - 1

    # 4. T-Statistic
    t_val = diff_value / se_diff if se_diff > 0 else 0.0

    p_left = float(t.cdf(t_val, df))
    p_value = _compute_p_value(p_left, alternative)

    # 5. Calculate CI using t-test df
    t_crit = float(t.ppf(1 - alpha / 2, df))
    diff_lci = diff_value - t_crit * se_diff
    diff_uci = diff_value + t_crit * se_diff

    # 6. Create DiffEst
    diff_est = DiffEst(
        y=y_name,
        diff=diff_value,
        se=se_diff,
        lci=diff_lci,
        uci=diff_uci,
        by=None,  # Two-group test doesn't have by in the diff
        by_level=None,
    )

    # 7. Assemble
    test_stats = TTestStats(df=df, t=t_val, p_value=p_value)
    test_est = [TtestEst.from_param(e) for e in est_list]

    return TTestTwoGroups(
        y=y_name,
        groups=GroupLevels(var=group_name, levels=(str(est2.by_level), str(est1.by_level))),
        alternative=alternative,  # type: ignore[arg-type]
        diff=[diff_est],
        estimates=test_est,
        stats=test_stats,
        alpha=alpha,
    )
