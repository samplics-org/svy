# src/svy/estimation/replication.py
"""
Replication-based estimation functions.

Each function takes an Estimation instance as its first argument.
Helper methods are called on the Estimation instance (est._*).
"""

from __future__ import annotations

import re

from typing import TYPE_CHECKING, Sequence, cast

import msgspec
import numpy as np
import polars as pl
import svy_rs as rs

from svy.core.enumerations import PopParam, QuantileMethod
from svy.core.repwgts import BrrWgts, RepWgts, _RepWgtsBase
from svy.errors import DimensionError
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
        # Strict ^prefix\d+$ matching — see core.data_prep._resolve_rep_weight_cols.
        pattern = re.compile(rf"^{re.escape(rw.prefix)}\d+$", re.IGNORECASE)
        cols = sorted(
            [c for c in local_data.columns if pattern.match(c)],
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
    # Both coefficient channels are length-checked at construction against the
    # recorded n_reps; this catches the case that check cannot see -- a recorded
    # n_reps that disagrees with the columns actually resolved from the data.
    for _param in ("scale", "rep_coefs"):
        _supplied = getattr(rw, _param, None)
        if _supplied is not None and len(_supplied) != n_reps:
            raise DimensionError(
                title=f"{_param} length mismatch",
                detail=f"RepWeights.{_param} has {len(_supplied)} entries but "
                f"{n_reps} replicate weight columns were resolved.",
                code=f"{_param.upper()}_LENGTH_MISMATCH",
                where="estimation.replication",
                param=f"rep_wgts.{_param}",
                expected=n_reps,
                got=len(_supplied),
            )
    # Per-replicate variance coefficients, computed by the variant rather than
    # re-derived from a method label on the far side of the FFI boundary. The
    # resolved column count wins over the recorded n_reps, and an
    # estimation-time fay_coef overrides the one stored on the design.
    changes: dict[str, object] = {}
    if rw.n_reps != n_reps:
        changes["n_reps"] = n_reps
    if fay_coef != 0.0 and isinstance(rw, BrrWgts):
        changes["fay_coef"] = float(fay_coef)
    effective = msgspec.structs.replace(rw, **changes) if changes else rw

    return rep_weight_cols, df_val, effective.coefficients()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def replicate_estimate(
    est: Estimation,
    prep: PreparedData,
    method: RepWgts,
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
    # The variant *is* the method: anything that is not one has no replicate
    # weights and cannot be estimated by replication.
    if not isinstance(method, _RepWgtsBase):
        raise ValueError(
            f"replicate_estimate requires replicate weights, got {method!r}. "
            f"Create them with sample.weighting.create_*_wgts, or use "
            f"method='taylor'."
        )
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
    method: RepWgts,
    fay_coef: float = 0.0,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
) -> Estimate:
    rep_weight_cols, df_val, rep_coefs = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    result_df = rs.replicate_mean(
        data,
        value_col=y,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        rep_coefs=rep_coefs,
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
        domain_mask_col=prep.domain_mask_col,
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
    )


def replicate_total(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    method: RepWgts,
    fay_coef: float = 0.0,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
) -> Estimate:
    rep_weight_cols, df_val, rep_coefs = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    result_df = rs.replicate_total(
        data,
        value_col=y,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        rep_coefs=rep_coefs,
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
        domain_mask_col=prep.domain_mask_col,
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
    )


def replicate_ratio(
    est: Estimation,
    prep: PreparedData,
    y: str,
    x: str,
    *,
    method: RepWgts,
    fay_coef: float = 0.0,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
) -> Estimate:
    rep_weight_cols, df_val, rep_coefs = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    result_df = rs.replicate_ratio(
        data,
        numerator_col=y,
        denominator_col=x,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        rep_coefs=rep_coefs,
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
        domain_mask_col=prep.domain_mask_col,
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
    )


def replicate_prop(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    method: RepWgts,
    fay_coef: float = 0.0,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
    ci_method: str = "logit",
) -> Estimate:
    rep_weight_cols, df_val, rep_coefs = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    data = est._coerce_y_for_prop(data, y)
    result_df = rs.replicate_prop(
        data,
        value_col=y,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        rep_coefs=rep_coefs,
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
        domain_mask_col=prep.domain_mask_col,
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
    )


def replicate_median(
    est: Estimation,
    prep: PreparedData,
    y: str,
    *,
    method: RepWgts,
    fay_coef: float = 0.0,
    q_method: QuantileMethod = QuantileMethod.HIGHER,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
) -> Estimate:
    rep_weight_cols, df_val, rep_coefs = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    q_method_str = q_method.value if hasattr(q_method, "value") else str(q_method).lower()
    result_df = rs.replicate_median(
        data,
        value_col=y,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        rep_coefs=rep_coefs,
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
    )


def replicate_quantile(
    est: Estimation,
    prep: PreparedData,
    y: str,
    probs: Sequence[float],
    *,
    method: RepWgts,
    fay_coef: float = 0.0,
    q_method: QuantileMethod = QuantileMethod.HIGHER,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
) -> list[Estimate]:
    """Replicate-weight quantiles, one ``Estimate`` per probability.

    Every probability is estimated from the same pass over the replicate
    weights; the frame is then sliced by probability.
    """
    rep_weight_cols, df_val, rep_coefs = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, rep_weight_cols)
    q_method_str = q_method.value if hasattr(q_method, "value") else str(q_method).lower()
    result_df = rs.replicate_quantile(
        data,
        value_col=y,
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        rep_coefs=rep_coefs,
        probs=list(probs),
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
        quantile_method=q_method_str,
    )

    results: list[Estimate] = []
    for p in probs:
        sub = result_df.filter(pl.col("prob") == p)
        est_list = est._replicate_quantile_result_to_param_est(sub, y, alpha, prep.by_col)
        results.append(
            est._build_estimate_result_light(
                est_list,
                np.diag(sub["var"].to_numpy()),
                PopParam.QUANTILE,
                alpha,
                prep.by_cols,
                as_factor=False,
                method=method,
            )
        )
    return results
