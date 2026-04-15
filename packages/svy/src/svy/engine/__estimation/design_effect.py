# src/svy/engine/estimation/design_effect.py
from __future__ import annotations

import numpy as np
import polars as pl

from svy.core.enumerations import PopParam
from svy.core.types import Array, DomainScalarMap, FloatArray, Number
from svy.engine.estimation.taylor import _estimate_taylor
from svy.estimation import ParamEst


def _weighted_s2_vectorized(
    y: FloatArray, w: FloatArray, group_ids: np.ndarray, n_groups: int
) -> np.ndarray:
    """
    Vectorized Kish-style weighted variance per group.
    Returns array of s2 values of shape (n_groups,).
    """
    # 1. Group sums
    w_g = np.bincount(group_ids, weights=w, minlength=n_groups)
    wy_g = np.bincount(group_ids, weights=w * y, minlength=n_groups)

    # 2. Weighted means
    # Avoid divide by zero
    w_g_safe = np.where(w_g == 0, 1.0, w_g)
    mu_g = wy_g / w_g_safe

    # 3. Residuals squared * w
    # Expand mean back to observation size
    mu_expanded = mu_g[group_ids]
    diff_sq = (y - mu_expanded) ** 2
    w_diff_sq = w * diff_sq

    sum_sq_diff_g = np.bincount(group_ids, weights=w_diff_sq, minlength=n_groups)

    # 4. n/(n-1) correction based on sample size (unweighted counts)
    n_g = np.bincount(group_ids, minlength=n_groups)
    correction = np.zeros_like(n_g, dtype=float)
    mask = n_g > 1
    correction[mask] = n_g[mask] / (n_g[mask] - 1.0)

    # 5. Combine: (1 / sum_w) * sum(w * (y-mu)^2) * correction
    s2 = (sum_sq_diff_g / w_g_safe) * correction
    s2[~mask] = 0.0  # Variance is 0 or undefined for n < 2

    return s2


def _srs_variance_wor(
    *,
    param: PopParam,
    y: Array,
    x: Array | None = None,
    wgt: Array | None = None,
    by: Array | None = None,
) -> Number | DomainScalarMap:
    """
    SRS(WOR) comparator variance for MEAN/PROP/TOTAL/RATIO.
    Vectorized implementation.
    """
    y_arr = np.asarray(y, dtype=float)
    n = y_arr.size

    # Default inputs
    if wgt is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(wgt, dtype=float)

    x_arr = np.asarray(x, dtype=float) if x is not None else None

    # Handle grouping
    if by is None:
        group_ids = np.zeros(n, dtype=int)
        unique_doms = [None]
        n_groups = 1
    else:
        by_arr = np.asarray(by)
        # Factorize domains
        # We can use np.unique logic or the helper if imported
        unique_doms, group_ids = np.unique(by_arr, return_inverse=True)
        n_groups = len(unique_doms)

    # 1. Calculate Group Stats
    n_g = np.bincount(group_ids, minlength=n_groups)

    # 2. Calculate s2 based on param type
    s2_vals = np.zeros(n_groups)

    if param in (PopParam.MEAN, PopParam.PROP, PopParam.TOTAL):
        s2_vals = _weighted_s2_vectorized(y_arr, w, group_ids, n_groups)

        if param == PopParam.TOTAL:
            # For TOTAL: Var = (N^2 / n) * S^2
            # Estimate N_hat = sum(w)
            N_hat_g = np.bincount(group_ids, weights=w, minlength=n_groups)

            # Avoid divide by zero for n=0 groups
            n_g_safe = np.where(n_g == 0, 1.0, n_g)
            var_srs = (N_hat_g**2 / n_g_safe) * s2_vals
            # Zero out invalid groups
            var_srs[n_g < 2] = 0.0

        else:
            # For MEAN/PROP: Var = S^2 / n
            n_g_safe = np.where(n_g == 0, 1.0, n_g)
            var_srs = s2_vals / n_g_safe
            var_srs[n_g < 2] = 0.0

    elif param == PopParam.RATIO:
        if x_arr is None:
            raise ValueError("x must be provided for RATIO")

        # Calculate Ratio Estimate R = sum(wy) / sum(wx)
        wy_g = np.bincount(group_ids, weights=w * y_arr, minlength=n_groups)
        wx_g = np.bincount(group_ids, weights=w * x_arr, minlength=n_groups)

        # R_hat per group
        # Avoid divide by zero
        wx_g_safe = np.where(wx_g == 0, 1.0, wx_g)
        R_g = wy_g / wx_g_safe

        # Expand R to rows to calculate residuals e = y - R*x
        R_expanded = R_g[group_ids]
        e_arr = y_arr - R_expanded * x_arr

        # Variance of residuals s2_e
        s2_e = _weighted_s2_vectorized(e_arr, w, group_ids, n_groups)

        # Variance of Ratio: (1/X_bar^2) * (s2_e / n)
        # X_bar = mean(x) weighted
        w_g = np.bincount(group_ids, weights=w, minlength=n_groups)
        x_bar_g = wx_g / np.where(w_g == 0, 1.0, w_g)

        denom = n_g * (x_bar_g**2)
        denom_safe = np.where(denom == 0, 1.0, denom)

        var_srs = s2_e / denom_safe
        var_srs[n_g < 2] = 0.0
        var_srs[denom == 0] = float("nan")

    else:
        raise ValueError(f"Unsupported PopParam: {param}")

    # Return result
    if by is None:
        return float(var_srs[0])

    return {k: float(v) for k, v in zip(unique_doms, var_srs)}


def _deff_scalar(
    *,
    est: ParamEst,
    param: PopParam,
    y: Array,
    wgt: FloatArray,
    psu: Array | None,
    stratum: Array | None = None,
    x: Array | None = None,
    by: Array | None = None,
) -> Number | DomainScalarMap:
    sum_w = float(wgt.sum())
    if sum_w == 0:
        return float("nan")

    var_srs_val = _srs_variance_wor(param=param, y=y, x=x, wgt=wgt, by=None)

    # Simple FPC approximation for the whole population
    variance_srs_adj = float(var_srs_val)  # type: ignore[arg-type]

    if variance_srs_adj == 0:
        return float("nan")

    return float((est.se**2) / variance_srs_adj)


def _deff_t(
    *,
    param: PopParam,
    y: Array,
    wgt: FloatArray,
    psu: Array | None,
    stratum: Array | None = None,
    x: Array | None = None,
    by: Array | None = None,
) -> Number | DomainScalarMap:
    # 1. Get Taylor Estimates (SE)
    # We re-run estimation here. In a full pipeline, this might be passed in.
    est_list, _ = _estimate_taylor(
        param=param,
        y=y,
        y_name="y",
        x=x,
        x_name="x",
        wgt=wgt,
        stratum=stratum,
        psu=psu,
        by=by,
    )

    # 2. Get SRS Variances (Vectorized)
    var_srs_map = _srs_variance_wor(param=param, y=y, x=x, wgt=wgt, by=by)

    # 3. Match and Calculate DEFF
    if by is None:
        # Scalar case
        se_taylor = est_list[0].se
        var_srs = float(var_srs_map)  # type: ignore[arg-type]

        sum_w = float(wgt.sum())
        fpc = (1 - len(wgt) / sum_w) if sum_w > 0 else 0

        denom = var_srs * fpc
        if denom <= 0:
            return float("nan")
        return float(se_taylor**2 / denom)
    else:
        # Domain case
        deff_map: DomainScalarMap = {}

        # Pre-calculate domain sums for FPC
        # We assume 'by' aligns with wgt
        by_arr = np.asarray(by)
        unique_doms, group_ids = np.unique(by_arr, return_inverse=True)

        w_dom = np.bincount(group_ids, weights=wgt, minlength=len(unique_doms))
        n_dom = np.bincount(group_ids, minlength=len(unique_doms))

        dom_stats = {d: (w_val, n_val) for d, w_val, n_val in zip(unique_doms, w_dom, n_dom)}

        for e in est_list:
            d_val = e.by_level
            if d_val not in var_srs_map:  # type: ignore[operator]
                deff_map[d_val]  # type: ignore[index] = float("nan")
                continue

            v_srs = var_srs_map[d_val]  # type: ignore[index]
            sum_w_d, n_d = dom_stats.get(d_val, (0, 0))

            if sum_w_d == 0 or v_srs == 0:
                deff_map[d_val]  # type: ignore[index] = float("nan")
                continue

            fpc = 1.0 - (n_d / sum_w_d)
            deff_map[d_val]  # type: ignore[index] = float(e.se**2 / (fpc * v_srs))

        return deff_map


def _deff_w(
    *,
    wgt: FloatArray,
    by: Array | None = None,
) -> Number | DomainScalarMap:
    """Design effect due to weighting (Kish's approximation: 1 + CV^2 of weights)."""
    # Use Polars for fast grouping
    if by is None:
        n = wgt.shape[0]
        sum_wgt = np.sum(wgt)
        if sum_wgt == 0:
            return float("nan")
        return float(n * np.sum(wgt * wgt) / (sum_wgt**2))

    df = pl.DataFrame({"wgt": wgt, "by": by})
    # deff_w = n * sum(w^2) / (sum(w)^2)
    #        = n * sum(w^2) / sum_w^2
    res = df.group_by("by").agg(
        (pl.len() * (pl.col("wgt") ** 2).sum() / (pl.col("wgt").sum() ** 2)).alias("deff_w")
    )
    return dict(zip(res["by"].to_list(), res["deff_w"].to_list()))


def _deff_s(
    *,
    y: Array,
    wgt: FloatArray,
    stratum: Array,
    stratum_name: str,
    by: Array | None = None,
) -> Number | DomainScalarMap:
    if by is None:
        n = wgt.shape[0]
        N = np.sum(wgt)
        if N == 0:
            return float("nan")
        df = (
            pl.DataFrame({"y": y, "wgt": wgt, stratum_name: stratum})
            .group_by(stratum_name)
            .agg(
                pl.col("wgt").sum().alias("N_h"),
                pl.col("wgt").count().alias("n_h"),
                pl.col("y").mean().alias("mu_h"),
                pl.col("y").var(ddof=1).alias("sigma2_h"),
            )
            .with_columns((pl.col("N_h") / N).alias("W_h"))
            .with_columns((pl.col("mu_h") * pl.col("N_h") / N).sum().alias("mu_bar"))
            .with_columns(
                (pl.col("W_h") * pl.col("sigma2_h")).alias("s2_p1"),
                ((pl.col("W_h") * (pl.col("mu_h") - pl.col("mu_bar"))) ** 2).alias("s2_p2"),
            )
            .with_columns(sigma2=(pl.col("s2_p1").sum() + pl.col("s2_p2").sum()))
            .with_columns(
                (
                    (pl.col("W_h") ** 2)
                    * (n / pl.col("n_h"))
                    * (pl.col("sigma2_h") / pl.col("sigma2"))
                ).alias("deff_s")
            )
        )
        return float(df["deff_s"].sum())
    else:
        # Domain case
        df = (
            pl.DataFrame({"y": y, "wgt": wgt, stratum_name: stratum, "by": by})
            .group_by(["by", stratum_name])
            .agg(
                pl.col("wgt").sum().alias("N_h"),
                pl.col("wgt").count().alias("n_h"),
                pl.col("y").mean().alias("mu_h"),
                pl.col("y").var(ddof=1).alias("sigma2_h"),
            )
        )
        totals = df.group_by("by").agg(
            pl.col("N_h").sum().alias("N"), pl.col("n_h").sum().alias("n")
        )

        df = (
            df.join(totals, on="by")
            .with_columns((pl.col("N_h") / pl.col("N")).alias("W_h"))
            .with_columns(
                (pl.col("mu_h") * pl.col("N_h") / pl.col("N")).sum().over("by").alias("mu_bar")
            )
            .with_columns(
                (pl.col("W_h") * pl.col("sigma2_h")).alias("s2_p1"),
                ((pl.col("W_h") * (pl.col("mu_h") - pl.col("mu_bar"))) ** 2).alias("s2_p2"),
            )
            .with_columns(
                sigma2=(pl.col("s2_p1").sum().over("by") + pl.col("s2_p2").sum().over("by"))
            )
            .with_columns(
                (
                    (pl.col("W_h") ** 2)
                    * (pl.col("n") / pl.col("n_h"))
                    * (pl.col("sigma2_h") / pl.col("sigma2"))
                ).alias("deff_s")
            )
            .group_by("by")
            .agg(pl.col("deff_s").sum())
        )
        return dict(zip(df["by"].to_list(), df["deff_s"].to_list()))


def _n_star(
    *,
    wgt: FloatArray,
    psu: Array | None,
    by: Array | None = None,
) -> Number | DomainScalarMap:
    # Efficient aggregation of (sum(w)^2 / sum(w^2)) per PSU group
    if psu is None:
        # If no PSU, n_star is effectively 1 (or treated as element-level)
        return 1.0

    if by is None:
        df = (
            pl.DataFrame({"wgt": wgt, "psu": psu})
            .group_by("psu")
            .agg(
                (pl.col("wgt").sum() ** 2).alias("num"),
                (pl.col("wgt") ** 2).sum().alias("den"),
            )
        )
        return float(df["num"].sum() / df["den"].sum())
    else:
        df = (
            pl.DataFrame({"wgt": wgt, "psu": psu, "by": by})
            .group_by(["by", "psu"])
            .agg(
                (pl.col("wgt").sum() ** 2).alias("num"),
                (pl.col("wgt") ** 2).sum().alias("den"),
            )
            .group_by("by")
            .agg((pl.col("num").sum() / pl.col("den").sum()).alias("n_star"))
        )
        return dict(zip(df["by"].to_list(), df["n_star"].to_list()))


def _rho(
    *,
    deff_t: Number | DomainScalarMap,
    deff_w: Number | DomainScalarMap,
    n_star: Number | DomainScalarMap,
    by: Array | None = None,
) -> Number | DomainScalarMap:
    def calc(dt, dw, m):
        if m <= 1.0 or dw == 0:
            return 0.0
        return (dt / dw - 1.0) / (m - 1.0)

    if by is None:
        return calc(float(deff_t), float(deff_w), float(n_star))  # type: ignore[arg-type]

    rho_map = {}
    # All maps should share keys
    keys = n_star.keys()  # type: ignore[union-attr]
    for k in keys:
        rho_map[k] = calc(deff_t.get(k, 0), deff_w.get(k, 1), n_star.get(k, 1))  # type: ignore[union-attr]
    return rho_map


def _deff_c(
    *,
    n_star: Number | DomainScalarMap,
    rho: Number | DomainScalarMap,
    by: Array | None = None,
) -> Number | DomainScalarMap:
    def calc(m, r):
        return 1.0 + (m - 1.0) * r if m > 1.0 else 1.0

    if by is None:
        return float(calc(float(n_star), float(rho)))  # type: ignore[arg-type]

    return {k: calc(n_star[k], rho[k]) for k in n_star}  # type: ignore[index, union-attr]
