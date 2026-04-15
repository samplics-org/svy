# src/svy/engine/estimation/replication.py
from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np

from scipy.stats import t as student

from svy.core.enumerations import EstimationMethod, PopParam, QuantileMethod
from svy.core.types import (
    Array,
    DomainCatMap,
    DomainScalarMap,
    FloatArray,
    Number,
)
from svy.engine.estimation.taylor import (
    _estimate_taylor_no_domain,
    _get_dummies_and_categories,
    _get_point_by,
    _get_point_no_domain,
)
from svy.estimation.estimate import ParamEst


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def _as_float1d(*, a: Array, name: str) -> FloatArray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' must be 1-D, got shape {arr.shape}.")
    return arr


def _as_float2d(*, a: Array, name: str) -> FloatArray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"'{name}' must be 2-D, got shape {arr.shape}.")
    return arr


def _ensure_same_length(n: int, *arrays: Array) -> None:
    for i, arr in enumerate(arrays, 1):
        if arr.shape[0] != n:
            raise ValueError(f"Array #{i} length {arr.shape[0]} != {n}.")


# ---------------------------------------------------------------------
# Point Estimators (Vectorized)
# ---------------------------------------------------------------------


def _rep_point_no_domain(
    *,
    param: PopParam,
    y: FloatArray,
    rep_w: FloatArray,  # (n, R)
    x: FloatArray | None,
) -> FloatArray:
    """
    Compute estimates for all R replicates simultaneously (Global).
    Uses matrix multiplication: (n,) @ (n, R) -> (R,)
    """
    # y @ rep_w is fast BLAS
    wy = y @ rep_w

    if param in (PopParam.PROP, PopParam.MEAN):
        denom = rep_w.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = wy / denom
        return res

    if param == PopParam.TOTAL:
        return wy

    if param == PopParam.RATIO:
        if x is None:
            raise ValueError("x must be provided for RATIO.")
        wx = x @ rep_w
        with np.errstate(divide="ignore", invalid="ignore"):
            res = wy / wx
        return res

    raise AssertionError("Unsupported parameter for replication estimators.")


def _rep_point_by_domain(
    *,
    param: PopParam,
    y: FloatArray,
    rep_w: FloatArray,  # (N, R)
    domain_ids: np.ndarray,  # (N,) int 0..K-1
    n_domains: int,
    x: FloatArray | None,
) -> FloatArray:
    """
    Compute estimates for all R replicates across all K domains simultaneously.
    Returns matrix (K, R).

    Optimization: Uses Matrix Multiplication instead of looping over replicates.
    Constructs a domain indicator matrix D (N, K).
    Result ~ D.T @ (Weight_Matrix * Y)
    """
    N, R = rep_w.shape
    K = n_domains

    # 1. Create Domain Indicator Matrix D (N, K)
    # Since K is usually small (<1000), a dense matrix is often faster than sparse overhead.
    # D[i, k] = 1 if domain_ids[i] == k
    # We use advanced indexing for speed.
    D = np.zeros((N, K), dtype=np.float64)
    D[np.arange(N), domain_ids] = 1.0

    # 2. Compute Weighted Numerators (K, R)
    # Wy = (N, R) -- Broadcasting Y across replicates
    # Numerator = D.T @ (Y * W)
    # This sums the weighted Y for each domain, for each replicate.

    # Note: If memory is tight, we can construct (Y*W) in chunks,
    # but for typical survey data (N < 10M), this is fine.

    Wy = rep_w * y[:, None]  # (N, R)
    nums = D.T @ Wy  # (K, N) @ (N, R) -> (K, R)

    if param == PopParam.TOTAL:
        return nums

    # 3. Compute Denominators (K, R)
    denoms = np.zeros((K, R), dtype=np.float64)

    if param in (PopParam.PROP, PopParam.MEAN):
        # Denom is Sum of Weights per domain per replicate
        denoms = D.T @ rep_w  # (K, R)

    elif param == PopParam.RATIO:
        assert x is not None
        Wx = rep_w * x[:, None]
        denoms = D.T @ Wx

    # 4. Divide safely
    with np.errstate(divide="ignore", invalid="ignore"):
        result = nums / denoms

    return result


# ---------------------------------------------------------------------
# Replicate coefficients
# ---------------------------------------------------------------------
def _rep_coefs(
    *,
    method: EstimationMethod,
    rep_coefs: Array | float | int | None = None,
    n_reps: int | None = None,
    fay_coef: float = 0.0,
) -> FloatArray:
    if rep_coefs is not None:
        arr = np.asarray(rep_coefs, dtype=np.float64).ravel()
        return _as_float1d(a=arr, name="rep_coefs")

    if n_reps is None or n_reps <= 0:
        raise ValueError("Provide 'n_reps' > 0 when 'rep_coefs' is not given.")

    if method == EstimationMethod.BOOTSTRAP:
        return cast(FloatArray, np.full(n_reps, 1.0 / n_reps, dtype=np.float64))

    if method == EstimationMethod.BRR:
        scale = 1.0 / (n_reps * (1.0 - float(fay_coef)) ** 2)
        return cast(FloatArray, np.full(n_reps, scale, dtype=np.float64))

    if method == EstimationMethod.JACKKNIFE:
        return cast(FloatArray, np.full(n_reps, (n_reps - 1.0) / n_reps, dtype=np.float64))

    if method == EstimationMethod.SDR:
        # SDR (e.g. ACS): coefficient = 4/R for each replicate
        return cast(FloatArray, np.full(n_reps, 4.0 / n_reps, dtype=np.float64))

    raise ValueError(f"Unsupported replication method for coefficients: {method}")


# ---------------------------------------------------------------------
# Variance Calculation (Vectorized)
# ---------------------------------------------------------------------
def _compute_variance_vectorized(
    *,
    method: EstimationMethod,
    theta_hat: FloatArray,  # (K,) or scalar
    theta_reps: FloatArray,  # (K, R) or (R,)
    rep_coefs: FloatArray,  # (R,)
    conservative: bool = False,
) -> FloatArray | float:
    """
    Compute variance for replicate estimates using broadcasting.
    Handles both scalar estimates (global) and vector estimates (domains).
    """
    # Ensure inputs are at least 1D for broadcasting
    is_scalar = theta_hat.ndim == 0
    if is_scalar:
        theta_hat = theta_hat[np.newaxis]  # (1,)
        theta_reps = theta_reps[np.newaxis, :]  # (1, R)

    # theta_hat: (K,)
    # theta_reps: (K, R)
    # rep_coefs: (R,)

    if method == EstimationMethod.JACKKNIFE:
        # JK Variance Logic
        # factor = 1 / (1 - C)
        # For JK1, C = (n-1)/n -> 1-C = 1/n -> factor = n

        # Avoid division by zero if C=1 (unlikely for valid JK)
        jk_factor = 1.0 / (1.0 - rep_coefs)  # (R,)

        # Pseudo values: (K, R)
        # pseudo = factor * theta_full - (factor - 1) * theta_rep
        # Broadcast: (K, 1) * (1, R) - ...

        th_full_broad = theta_hat[:, np.newaxis]
        pseudo = (th_full_broad * jk_factor) - (theta_reps * (jk_factor - 1.0))

        if conservative:
            center = th_full_broad
        else:
            center = np.mean(pseudo, axis=1, keepdims=True)  # (K, 1)

        denom = jk_factor - 1.0  # (R,)
        # Var = sum( c * ((pseudo - center)/denom)^2 )
        diffs = (pseudo - center) / denom

    elif method == EstimationMethod.SDR:
        # SDR always uses MSE formula (center on full sample estimate)
        # This is required by the successive difference methodology
        center = theta_hat[:, np.newaxis]
        diffs = theta_reps - center
    else:
        # BRR / Bootstrap
        if conservative:
            center = theta_hat[:, np.newaxis]
        else:
            center = np.mean(theta_reps, axis=1, keepdims=True)

        diffs = theta_reps - center

    # Weighted Sum of Squares
    weighted_sq_diff = (diffs**2) * rep_coefs  # (K, R)
    variance = np.sum(weighted_sq_diff, axis=1)  # (K,)

    if is_scalar:
        return float(variance[0])
    return variance


# ---------------------------------------------------------------------
# Primary Variance Controller
# ---------------------------------------------------------------------
def _get_variance(
    *,
    method: EstimationMethod,
    param: PopParam,
    y: Array,
    wgt: Array,
    rep_w: Array,
    rep_coefs: Array,
    x: Array | None = None,
    by: Array | None = None,
    conservative: bool = False,
    est_full: Any = None,  # Passed in to avoid recomputing full-sample estimate
) -> Number | DomainScalarMap | DomainCatMap:
    wgt = _as_float1d(a=wgt, name="samp_weight")
    Y = _as_float1d(a=y, name="y")
    RW = _as_float2d(a=rep_w, name="rep_weights")
    RC = _as_float1d(a=rep_coefs, name="rep_coefs")

    _ensure_same_length(Y.size, wgt)

    X: FloatArray | None = None
    if param == PopParam.RATIO:
        if x is None:
            raise ValueError("x must be provided for RATIO.")
        X = _as_float1d(a=x, name="x")

    # --- Case 1: Proportions (Multi-Category) ---
    if param == PopParam.PROP:
        # We handle PROP by treating it as multiple binary variables
        y_dummies, categories = _get_dummies_and_categories(y=Y)
        y_dummies = np.asarray(y_dummies, dtype=np.float64)
        categories = list(categories)

        out_prop: Any = {}

        # For PROP, we loop categories.
        # Inside each category, we do the full domain vectorization.
        # This is essentially K_cats * (Operations).

        for k, cat in enumerate(categories):
            yk = y_dummies[:, k]

            # Recurse for this binary variable (treated as MEAN)
            # We pass param=MEAN to reuse the numeric logic

            est_k = None
            if est_full is not None:
                if by is not None:
                    # est_full is {domain: {cat: val}} -> {domain: val}
                    est_k = {d: est_full[d][cat] for d in est_full}
                else:
                    # est_full is {cat: val} -> val
                    est_k = est_full[cat]

            var_k = _get_variance(
                method=method,
                param=PopParam.MEAN,
                y=yk,
                wgt=wgt,
                rep_w=RW,
                rep_coefs=RC,
                x=None,
                by=by,
                conservative=conservative,
                est_full=est_k,
            )

            # Reconstruct structure: {cat: {dom: var}} or {cat: var}
            if by is None:
                out_prop[cat] = var_k
            else:
                # var_k is {dom: var}
                # We need to pivot to {dom: {cat: var}}
                # Let's accumulate temporarily and pivot at end.
                for d, v in cast(dict, var_k).items():
                    if d not in out_prop:
                        out_prop[d] = {}
                    out_prop[d][cat] = v

        return out_prop

    # --- Case 2: Numeric (Mean, Total, Ratio) ---

    # A. No Domain (Global)
    if by is None:
        # Full sample point estimate (if not provided)
        if est_full is None:
            est_full = _get_point_no_domain(param=param, y=Y, wgt=wgt, x=X)

        # Replicate estimates: (R,)
        theta_reps = _rep_point_no_domain(param=param, y=Y, rep_w=RW, x=X)

        # Variance
        return _compute_variance_vectorized(  # type: ignore[return-value]
            method=method,
            theta_hat=np.array(est_full),
            theta_reps=theta_reps,
            rep_coefs=RC,
            conservative=conservative,
        )

    # B. By Domain (Vectorized)
    dom = np.asarray(by)

    # Factorize domains to 0..K-1 integers
    unique_domains, domain_ids = np.unique(dom, return_inverse=True)
    n_domains = len(unique_domains)

    # Full sample estimates by domain (if not provided)
    if est_full is None:
        # This returns {dom: val}
        est_full = _get_point_by(param=param, y=Y, wgt=wgt, x=X, by=dom)

    # Convert est_full dict to array aligned with unique_domains
    theta_full_arr = np.array([est_full[d] for d in unique_domains])

    # Calculate Replicate Estimates Matrix: (K, R)
    theta_reps_mat = _rep_point_by_domain(
        param=param, y=Y, rep_w=RW, domain_ids=domain_ids, n_domains=n_domains, x=X
    )

    # Compute Variance Vector: (K,)
    var_arr = _compute_variance_vectorized(
        method=method,
        theta_hat=theta_full_arr,
        theta_reps=theta_reps_mat,
        rep_coefs=RC,
        conservative=conservative,
    )

    # Map back to domain labels
    return {unique_domains[i]: float(var_arr[i]) for i in range(n_domains)}  # type: ignore[index]


# -----------------------------
# Main Entry Point
# -----------------------------
def _estimate_replicates(
    *,
    method: EstimationMethod,
    param: PopParam,
    y: Array,
    y_name: str,
    wgt: Array,
    rep_w: Array,
    n_reps: int,
    degrees_of_freedom: int,
    fay_coef: float = 0.0,
    x: Array | None = None,
    x_name: str | None = None,
    rep_coefs: Array | float | None = None,
    by: Array | None = None,
    by_name: str | Sequence[str] | None = None,
    conservative: bool = False,
    as_factor: bool = False,
    q_method: QuantileMethod | None = None,
    alpha: float = 0.05,
) -> list[ParamEst]:
    # 1. Setup
    if param == PopParam.RATIO and x is None:
        raise ValueError("x must be provided for ratio estimation.")

    RC = _rep_coefs(method=method, rep_coefs=rep_coefs, n_reps=n_reps, fay_coef=fay_coef)

    # 2. Compute Point Estimates & Variance
    # We compute point estimates first to pass them into variance calc (optimization)

    if by is None:
        # --- No Domain ---
        est_val = _get_point_no_domain(
            param=param, y=y, wgt=wgt, x=x, q_method=q_method, as_factor=as_factor
        )

        var_val = _get_variance(
            method=method,
            param=param,
            y=y,
            wgt=wgt,
            rep_w=rep_w,
            rep_coefs=RC,
            x=x,
            by=None,
            conservative=conservative,
            est_full=est_val,
        )

        # Build Result
        df_eff = max(degrees_of_freedom, 1)
        t_q = float(student.ppf(1.0 - alpha / 2.0, df=df_eff))

        rows = _estimate_taylor_no_domain(
            param=param,
            y_name=y_name,
            x_name=x_name,
            estimate=est_val,
            variance=var_val,  # type: ignore[arg-type]
            variance_srswor_normalized=None,
            t_quantile=t_q,
            q_method=q_method or QuantileMethod.MIDDLE,
            as_factor=as_factor,
        )
        return [ParamEst(**r) for r in rows]

    else:
        # --- By Domain ---
        est_dom = _get_point_by(
            param=param, y=y, wgt=wgt, x=x, by=by, q_method=q_method, as_factor=as_factor
        )

        var_dom = _get_variance(
            method=method,
            param=param,
            y=y,
            wgt=wgt,
            rep_w=rep_w,
            rep_coefs=RC,
            x=x,
            by=by,
            conservative=conservative,
            est_full=est_dom,
        )

        # Assemble Results
        df_eff = max(degrees_of_freedom, 1)
        t_q = float(student.ppf(1.0 - alpha / 2.0, df=df_eff))

        final_rows = []

        # Ensure by_name is a tuple if it exists
        if by_name is not None:
            by_name_tuple = tuple(by_name) if not isinstance(by_name, str) else (by_name,)
        else:
            by_name_tuple = None

        # Iterate over the Variance dictionary keys (Domains)
        for d_key, v_stat in var_dom.items():  # type: ignore[union-attr]  # v_stat is var or {cat: var}
            e_stat = est_dom[d_key]

            # Reuse Taylor helper to format CI/CV/SE
            entries = _estimate_taylor_no_domain(
                param=param,
                y_name=y_name,
                x_name=x_name,
                estimate=e_stat,
                variance=v_stat,
                variance_srswor_normalized=None,
                t_quantile=t_q,
                q_method=q_method or QuantileMethod.MIDDLE,
                as_factor=as_factor,
            )

            # Attach Domain Info
            dom_val = d_key.tolist() if hasattr(d_key, "tolist") else d_key  # type: ignore[operator]
            for e in entries:
                e["by"] = by_name_tuple
                e["by_level"] = dom_val

            final_rows.extend(entries)

        return [ParamEst(**r) for r in final_rows]
