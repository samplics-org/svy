# src/svy/estimation/replication.py
"""
Replication-based estimation functions.

Each function takes an Estimation instance as its first argument.
Helper methods are called on the Estimation instance (est._*).
"""

from __future__ import annotations

import re

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import svy_rs as rs

from svy.core.enumerations import EstimationMethod, PopParam, QuantileMethod
from svy.estimation.estimate import Estimate


if TYPE_CHECKING:
    from svy.core.data_prep import PreparedData
    from svy.estimation.base import Estimation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_rep_weight_cols(est: Estimation) -> list[str]:
    rw = est._sample._design.rep_wgts
    if rw is None:
        return []
    if hasattr(rw, "_cached_cols") and rw._cached_cols is not None:
        return rw._cached_cols

    _lraw = est._sample._data
    local_data: pl.DataFrame = (
        cast(pl.DataFrame, _lraw.collect())
        if isinstance(_lraw, pl.LazyFrame)
        else cast(pl.DataFrame, _lraw)
    )

    def natural_keys(text: str):
        return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]

    if rw.prefix:
        prefix_lower = rw.prefix.lower()
        cols = sorted(
            [
                c
                for c in local_data.columns
                if c.lower().startswith(prefix_lower) and c.lower() != prefix_lower
            ],
            key=lambda c: natural_keys(c.lower()),
        )
    elif hasattr(rw, "wgts") and rw.wgts:
        # Resolve explicit column names case-insensitively against actual columns.
        lower_index: dict[str, str] = {}
        for c in local_data.columns:
            # First occurrence wins; collisions are rare but possible.
            lower_index.setdefault(c.lower(), c)

        requested = list(cast(list[str], rw.wgts))
        cols = []
        missing = []
        for name in requested:
            actual = lower_index.get(name.lower())
            if actual is None:
                missing.append(name)
            else:
                cols.append(actual)
        if missing:
            raise ValueError(
                f"Replicate weight columns not found (case-insensitive match): "
                f"{missing}. Available columns: {local_data.columns}"
            )
    else:
        cols = []

    try:
        rw._cached_cols = cols
    except Exception:
        pass
    return cols


def get_rep_method_str(method: EstimationMethod) -> str:
    method_map = {
        EstimationMethod.BRR: "BRR",
        EstimationMethod.BOOTSTRAP: "Bootstrap",
        EstimationMethod.JACKKNIFE: "Jackknife",
        EstimationMethod.SDR: "SDR",
    }
    return method_map.get(method, "jackknife")


def _get_rep_params(est: Estimation, fay_coef: float = 0.0):
    design = est._sample._design
    rw = design.rep_wgts
    if rw is None:
        raise ValueError("Replication weights required for replication-based estimation.")
    rep_weight_cols = get_rep_weight_cols(est)
    if not rep_weight_cols:
        raise ValueError("No replicate weight columns found.")
    n_reps = len(rep_weight_cols)
    df_val = int(rw.df) if rw.df and rw.df > 0 else max(1, n_reps - 1)
    final_fay = float(fay_coef) if fay_coef != 0.0 else float(rw.fay_coef)
    return rep_weight_cols, df_val, final_fay


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def replicate_estimate(
    est: Estimation,
    prep: PreparedData,
    method: EstimationMethod,
    param: PopParam,
    y: str,
    *,
    x: str | None = None,
    fay_coef: float = 0.0,
    as_factor: bool = False,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
    ci_method: str = "logit",
) -> Estimate:
    if method not in (
        EstimationMethod.BRR,
        EstimationMethod.BOOTSTRAP,
        EstimationMethod.JACKKNIFE,
        EstimationMethod.SDR,
    ):
        raise ValueError(f"Method {method} is not a valid replication method.")
    if param == PopParam.MEAN:
        return replicate_mean(
            est,
            prep,
            y,
            method=method,
            fay_coef=fay_coef,
            variance_center=variance_center,
            alpha=alpha,
        )
    elif param == PopParam.TOTAL:
        return replicate_total(
            est,
            prep,
            y,
            method=method,
            fay_coef=fay_coef,
            variance_center=variance_center,
            alpha=alpha,
        )
    elif param == PopParam.RATIO:
        if x is None:
            raise ValueError("x must be provided for ratio estimation.")
        return replicate_ratio(
            est,
            prep,
            y,
            x,
            method=method,
            fay_coef=fay_coef,
            variance_center=variance_center,
            alpha=alpha,
        )
    elif param == PopParam.PROP:
        return replicate_prop(
            est,
            prep,
            y,
            method=method,
            fay_coef=fay_coef,
            variance_center=variance_center,
            alpha=alpha,
            ci_method=ci_method,
        )
    else:
        raise ValueError(f"Unsupported parameter {param} for replication estimation.")


# ---------------------------------------------------------------------------
# Individual estimators
# ---------------------------------------------------------------------------


def replicate_mean(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    method: EstimationMethod,
    fay_coef: float = 0.0,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
) -> Estimate:
    rep_weight_cols, df_val, final_fay = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    result_df = rs.replicate_mean(
        data,
        value_col=y,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        method=get_rep_method_str(method),
        fay_coef=final_fay,
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
    )
    est_list = est._polars_result_to_param_est(
        result_df, y, PopParam.MEAN, alpha, deff=False, by_col=prep.by_col, as_factor=False
    )
    return est._build_estimate_result_light(
        est_list,
        np.diag(result_df["var"].to_numpy()),
        PopParam.MEAN,
        alpha,
        prep.by_cols,
        as_factor=False,
        method=method,
        rust_df=int(result_df["df"].min()),
    )


def replicate_total(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    method: EstimationMethod,
    fay_coef: float = 0.0,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
) -> Estimate:
    rep_weight_cols, df_val, final_fay = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    result_df = rs.replicate_total(
        data,
        value_col=y,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        method=get_rep_method_str(method),
        fay_coef=final_fay,
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
    )
    est_list = est._polars_result_to_param_est(
        result_df, y, PopParam.TOTAL, alpha, deff=False, by_col=prep.by_col, as_factor=False
    )
    return est._build_estimate_result_light(
        est_list,
        np.diag(result_df["var"].to_numpy()),
        PopParam.TOTAL,
        alpha,
        prep.by_cols,
        as_factor=False,
        method=method,
        rust_df=int(result_df["df"].min()),
    )


def replicate_ratio(
    est: Estimation,
    prep: PreparedData,
    y: str,
    x: str,
    *,
    method: EstimationMethod,
    fay_coef: float = 0.0,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
) -> Estimate:
    rep_weight_cols, df_val, final_fay = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    result_df = rs.replicate_ratio(
        data,
        numerator_col=y,
        denominator_col=x,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        method=get_rep_method_str(method),
        fay_coef=final_fay,
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
    )
    est_list = est._polars_result_to_param_est(
        result_df,
        y,
        PopParam.RATIO,
        alpha,
        deff=False,
        by_col=prep.by_col,
        as_factor=False,
        x_name=x,
    )
    return est._build_estimate_result_light(
        est_list,
        np.diag(result_df["var"].to_numpy()),
        PopParam.RATIO,
        alpha,
        prep.by_cols,
        as_factor=False,
        method=method,
        rust_df=int(result_df["df"].min()),
    )


def replicate_prop(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    method: EstimationMethod,
    fay_coef: float = 0.0,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
    ci_method: str = "logit",
) -> Estimate:
    rep_weight_cols, df_val, final_fay = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    data = est._coerce_y_for_prop(data, y)
    result_df = rs.replicate_prop(
        data,
        value_col=y,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        method=get_rep_method_str(method),
        fay_coef=final_fay,
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
    )
    est_list = est._polars_result_to_param_est(
        result_df,
        y,
        PopParam.PROP,
        alpha,
        deff=False,
        by_col=prep.by_col,
        as_factor=True,
        ci_method=ci_method,
    )
    return est._build_estimate_result_light(
        est_list,
        np.diag(result_df["var"].to_numpy()),
        PopParam.PROP,
        alpha,
        prep.by_cols,
        as_factor=True,
        method=method,
        rust_df=int(result_df["df"].min()),
    )


def replicate_median(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    method: EstimationMethod,
    fay_coef: float = 0.0,
    q_method: QuantileMethod = QuantileMethod.HIGHER,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
) -> Estimate:
    rep_weight_cols, df_val, final_fay = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    q_method_str = q_method.value if hasattr(q_method, "value") else str(q_method).lower()
    result_df = rs.replicate_median(
        data,
        value_col=y,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        method=get_rep_method_str(method),
        fay_coef=final_fay,
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
        quantile_method=q_method_str,
    )
    est_list = est._replicate_median_result_to_param_est(result_df, y, alpha, prep.by_col)
    return est._build_estimate_result_light(
        est_list,
        np.diag(result_df["var"].to_numpy()),
        PopParam.MEDIAN,
        alpha,
        prep.by_cols,
        as_factor=False,
        method=method,
        rust_df=int(result_df["df"].min()),
    )
