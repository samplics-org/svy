# src/svy/estimation/taylor.py
"""
Taylor linearization estimation functions.

Each function takes an Estimation instance as its first argument.
Helper methods are called on the Estimation instance (est._*).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import svy_rs as rs

from svy.core.enumerations import EstimationMethod, PopParam, QuantileMethod
from svy.estimation.estimate import Estimate


if TYPE_CHECKING:
    from svy.core.data_prep import PreparedData
    from svy.estimation.base import Estimation


def taylor_mean(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    param: PopParam = PopParam.MEAN,
    deff: bool = False,
    as_factor: bool = False,
    alpha: float = 0.05,
) -> Estimate:
    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )

    center_arg = est._get_center_method()
    fn = rs.taylor_prop if as_factor else rs.taylor_mean

    result_df = fn(
        df,
        value_col=y,
        weight_col=prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        by_col=prep.by_col,
        singleton_method=center_arg,
    )

    if est._should_run_double_pass():
        cache = est._get_polars_design_info()
        df_full = cache["data"]
        if y in df_full.columns and df_full[y].dtype != pl.Float64:
            df_full = df_full.with_columns(pl.col(y).cast(pl.Float64))
        result_full = fn(
            df_full,
            value_col=y,
            weight_col=cache["weight_col"],
            strata_col=cache["strata_col"],
            psu_col=cache["psu_col"],
            ssu_col=cache.get("ssu_col"),
            fpc_col=cache.get("fpc_col"),
            fpc_ssu_col=cache.get("fpc_ssu_col"),
            by_col=prep.by_col,
            singleton_method=center_arg,
        )
        result_df = est._apply_scale_adjustment(result_full, result_df, param=param)

    est_list = est._polars_result_to_param_est(
        result_df, y, param, alpha, deff, prep.by_col, as_factor
    )
    est_cov = np.diag(result_df["var"].to_numpy())
    return est._build_estimate_result_light(
        est_list,
        est_cov,
        param,
        alpha,
        prep.by_cols,
        as_factor,
        method=EstimationMethod.TAYLOR,
        rust_df=int(result_df["df"].min()),
    )


def taylor_mean_multi(
    est: Estimation,
    prep: PreparedData,
    ys: list[str],
    *,
    deff: bool = False,
    alpha: float = 0.05,
    param: PopParam = PopParam.MEAN,
) -> list[Estimate]:
    """Ungrouped Taylor means for many variables, sharing one design build.

    Calls the batched Rust kernel once (design indexed a single time, variables
    fanned out in parallel), then slices the N-row result into one ``Estimate``
    per variable — each identical to the corresponding single-variable
    ``taylor_mean`` call. Returned in ``ys`` order.
    """
    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )
    # Every response column must be Float64 for the Rust kernel; prepare_data
    # only casts the primary y, so cast the rest here.
    df = est._ensure_float64(df, ys)
    center_arg = est._get_center_method()

    result_df = rs.taylor_mean_multi(
        df,
        value_cols=ys,
        weight_col=prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        singleton_method=center_arg,
    )

    results: list[Estimate] = []
    for y in ys:
        sub = result_df.filter(pl.col("y") == y)
        est_list = est._polars_result_to_param_est(sub, y, param, alpha, deff, None, False)
        est_cov = np.diag(sub["var"].to_numpy())
        results.append(
            est._build_estimate_result_light(
                est_list,
                est_cov,
                param,
                alpha,
                [],
                as_factor=False,
                method=EstimationMethod.TAYLOR,
                rust_df=int(sub["df"].min()),
            )
        )
    return results


def taylor_total(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    deff: bool = False,
    as_factor: bool = False,
    alpha: float = 0.05,
) -> Estimate:
    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )
    center_arg = est._get_center_method()

    result_df = rs.taylor_total(
        df,
        value_col=y,
        weight_col=prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        by_col=prep.by_col,
        singleton_method=center_arg,
    )

    if est._should_run_double_pass():
        cache = est._get_polars_design_info()
        df_full = cache["data"]
        if y in df_full.columns and df_full[y].dtype != pl.Float64:
            df_full = df_full.with_columns(pl.col(y).cast(pl.Float64))
        result_full = rs.taylor_total(
            df_full,
            value_col=y,
            weight_col=cache["weight_col"],
            strata_col=cache["strata_col"],
            psu_col=cache["psu_col"],
            ssu_col=cache.get("ssu_col"),
            fpc_col=cache.get("fpc_col"),
            fpc_ssu_col=cache.get("fpc_ssu_col"),
            by_col=prep.by_col,
            singleton_method=center_arg,
        )
        result_df = est._apply_scale_adjustment(result_full, result_df, param=PopParam.TOTAL)

    est_list = est._polars_result_to_param_est(
        result_df, y, PopParam.TOTAL, alpha, deff, prep.by_col, as_factor=False
    )
    est_cov = np.diag(result_df["var"].to_numpy())
    return est._build_estimate_result_light(
        est_list,
        est_cov,
        PopParam.TOTAL,
        alpha,
        prep.by_cols,
        as_factor=False,
        method=EstimationMethod.TAYLOR,
        rust_df=int(result_df["df"].min()),
    )


def taylor_total_multi(
    est: Estimation,
    prep: PreparedData,
    ys: list[str],
    *,
    deff: bool = False,
    alpha: float = 0.05,
) -> list[Estimate]:
    """Ungrouped Taylor totals for many variables, sharing one design build.

    See :func:`taylor_mean_multi`. One ``Estimate`` per variable, in ``ys``
    order, each identical to the corresponding single-variable ``taylor_total``.
    """
    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )
    df = est._ensure_float64(df, ys)
    center_arg = est._get_center_method()

    result_df = rs.taylor_total_multi(
        df,
        value_cols=ys,
        weight_col=prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        singleton_method=center_arg,
    )

    results: list[Estimate] = []
    for y in ys:
        sub = result_df.filter(pl.col("y") == y)
        est_list = est._polars_result_to_param_est(
            sub, y, PopParam.TOTAL, alpha, deff, None, as_factor=False
        )
        est_cov = np.diag(sub["var"].to_numpy())
        results.append(
            est._build_estimate_result_light(
                est_list,
                est_cov,
                PopParam.TOTAL,
                alpha,
                [],
                as_factor=False,
                method=EstimationMethod.TAYLOR,
                rust_df=int(sub["df"].min()),
            )
        )
    return results


def taylor_ratio(
    est: Estimation,
    prep: PreparedData,
    y: str,
    x: str,
    *,
    deff: bool = False,
    alpha: float = 0.05,
) -> Estimate:
    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )
    center_arg = est._get_center_method()

    result_df = rs.taylor_ratio(
        df,
        numerator_col=y,
        denominator_col=x,
        weight_col=prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        by_col=prep.by_col,
        singleton_method=center_arg,
    )

    if est._should_run_double_pass():
        cache = est._get_polars_design_info()
        df_full = cache["data"]
        if y in df_full.columns and df_full[y].dtype != pl.Float64:
            df_full = df_full.with_columns(pl.col(y).cast(pl.Float64))
        result_full = rs.taylor_ratio(
            df_full,
            numerator_col=y,
            denominator_col=x,
            weight_col=cache["weight_col"],
            strata_col=cache["strata_col"],
            psu_col=cache["psu_col"],
            ssu_col=cache.get("ssu_col"),
            fpc_col=cache.get("fpc_col"),
            fpc_ssu_col=cache.get("fpc_ssu_col"),
            by_col=prep.by_col,
            singleton_method=center_arg,
        )
        result_df = est._apply_scale_adjustment(result_full, result_df, param=PopParam.RATIO)

    est_list = est._polars_result_to_param_est(
        result_df, y, PopParam.RATIO, alpha, deff, prep.by_col, as_factor=False, x_name=x
    )
    est_cov = np.diag(result_df["var"].to_numpy())
    return est._build_estimate_result_light(
        est_list,
        est_cov,
        PopParam.RATIO,
        alpha,
        prep.by_cols,
        as_factor=False,
        method=EstimationMethod.TAYLOR,
        rust_df=int(result_df["df"].min()),
    )


def taylor_prop(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    deff: bool = False,
    alpha: float = 0.05,
    ci_method: str = "logit",
) -> Estimate:
    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )
    center_arg = est._get_center_method()

    df = est._coerce_y_for_prop(df, y)
    result_df = rs.taylor_prop(
        df,
        value_col=y,
        weight_col=prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        by_col=prep.by_col,
        singleton_method=center_arg,
    )

    if est._should_run_double_pass():
        cache = est._get_polars_design_info()
        df_full = cache["data"]
        result_full = rs.taylor_prop(
            df_full,
            value_col=y,
            weight_col=cache["weight_col"],
            strata_col=cache["strata_col"],
            psu_col=cache["psu_col"],
            ssu_col=cache.get("ssu_col"),
            fpc_col=cache.get("fpc_col"),
            fpc_ssu_col=cache.get("fpc_ssu_col"),
            by_col=prep.by_col,
            singleton_method=center_arg,
        )
        result_df = est._apply_scale_adjustment(result_full, result_df, param=PopParam.PROP)

    est_list = est._polars_result_to_param_est(
        result_df,
        y,
        PopParam.PROP,
        alpha,
        deff,
        prep.by_col,
        as_factor=True,
        ci_method=ci_method,
    )
    est_cov = np.diag(result_df["var"].to_numpy())
    return est._build_estimate_result_light(
        est_list,
        est_cov,
        PopParam.PROP,
        alpha,
        prep.by_cols,
        as_factor=True,
        method=EstimationMethod.TAYLOR,
        rust_df=int(result_df["df"].min()),
    )


def taylor_median(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    q_method: QuantileMethod = QuantileMethod.HIGHER,
    alpha: float = 0.05,
) -> Estimate:
    if prep.df[y].null_count() > 0:
        raise ValueError(f"Missing values found in '{y}'. Use drop_nulls=True to ignore.")

    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )
    center_arg = est._get_center_method()
    q_method_str = q_method.value if hasattr(q_method, "value") else str(q_method).lower()

    result_df = rs.taylor_median(
        df,
        value_col=y,
        weight_col=prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        by_col=prep.by_col,
        singleton_method=center_arg,
        quantile_method=q_method_str,
    )

    est_list = est._median_result_to_param_est(
        result_df, y, alpha, prep.by_col, df, prep.weight_col
    )
    est_cov = np.diag(result_df["var"].to_numpy())
    return est._build_estimate_result_light(
        est_list,
        est_cov,
        PopParam.MEDIAN,
        alpha,
        prep.by_cols,
        as_factor=False,
        method=EstimationMethod.TAYLOR,
        rust_df=int(result_df["df"].min()),
    )


def taylor_ratio_multi(
    est: Estimation,
    prep: PreparedData,
    ys: list[str],
    xs: list[str],
    *,
    deff: bool = False,
    alpha: float = 0.05,
) -> list[Estimate]:
    """Ungrouped Taylor ratios for paired (numerator, denominator) columns,
    sharing one design build. See :func:`taylor_mean_multi`."""
    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )
    df = est._ensure_float64(df, list(dict.fromkeys([*ys, *xs])))
    center_arg = est._get_center_method()

    result_df = rs.taylor_ratio_multi(
        df,
        numerator_cols=ys,
        denominator_cols=xs,
        weight_col=prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        singleton_method=center_arg,
    )

    results: list[Estimate] = []
    for i, (y, x) in enumerate(zip(ys, xs)):
        sub = result_df.slice(i, 1)
        est_list = est._polars_result_to_param_est(
            sub, y, PopParam.RATIO, alpha, deff, None, as_factor=False, x_name=x
        )
        est_cov = np.diag(sub["var"].to_numpy())
        results.append(
            est._build_estimate_result_light(
                est_list, est_cov, PopParam.RATIO, alpha, [], as_factor=False,
                method=EstimationMethod.TAYLOR, rust_df=int(sub["df"].min()),
            )
        )
    return results


def taylor_prop_multi(
    est: Estimation,
    prep: PreparedData,
    ys: list[str],
    *,
    deff: bool = False,
    alpha: float = 0.05,
    ci_method: str = "logit",
) -> list[Estimate]:
    """Ungrouped Taylor proportions for many category columns, sharing one design
    build. Each variable yields a multi-row (per-level) ``Estimate``."""
    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )
    for y in ys:
        df = est._coerce_y_for_prop(df, y)
    center_arg = est._get_center_method()

    result_df = rs.taylor_prop_multi(
        df,
        value_cols=ys,
        weight_col=prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        singleton_method=center_arg,
    )

    results: list[Estimate] = []
    for y in ys:
        sub = result_df.filter(pl.col("y") == y)
        est_list = est._polars_result_to_param_est(
            sub, y, PopParam.PROP, alpha, deff, None, as_factor=True, ci_method=ci_method
        )
        est_cov = np.diag(sub["var"].to_numpy())
        results.append(
            est._build_estimate_result_light(
                est_list, est_cov, PopParam.PROP, alpha, [], as_factor=True,
                method=EstimationMethod.TAYLOR, rust_df=int(sub["df"].min()),
            )
        )
    return results


def taylor_median_multi(
    est: Estimation,
    prep: PreparedData,
    ys: list[str],
    *,
    q_method: QuantileMethod = QuantileMethod.HIGHER,
    alpha: float = 0.05,
) -> list[Estimate]:
    """Ungrouped Taylor medians for many variables, run in parallel. See
    :func:`taylor_median` (median amortises no design; the win is parallelism)."""
    for y in ys:
        if prep.df[y].null_count() > 0:
            raise ValueError(f"Missing values found in '{y}'. Use drop_nulls=True to ignore.")

    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )
    df = est._ensure_float64(df, ys)
    center_arg = est._get_center_method()
    q_method_str = q_method.value if hasattr(q_method, "value") else str(q_method).lower()

    result_df = rs.taylor_median_multi(
        df,
        value_cols=ys,
        weight_col=prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        singleton_method=center_arg,
        quantile_method=q_method_str,
    )

    results: list[Estimate] = []
    for i, y in enumerate(ys):
        sub = result_df.slice(i, 1)
        est_list = est._median_result_to_param_est(sub, y, alpha, None, df, prep.weight_col)
        est_cov = np.diag(sub["var"].to_numpy())
        results.append(
            est._build_estimate_result_light(
                est_list, est_cov, PopParam.MEDIAN, alpha, [], as_factor=False,
                method=EstimationMethod.TAYLOR, rust_df=int(sub["df"].min()),
            )
        )
    return results
