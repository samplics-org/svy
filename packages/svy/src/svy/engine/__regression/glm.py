# src/svy/engine/regression/glm.py
from __future__ import annotations

import logging

from typing import Any, NamedTuple, Sequence

import numpy as np
import polars as pl

from svy.core.types import FloatArray
from svy.errors.model_errors import ModelError
from svy.utils.helpers import _normalize_name_seq


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Engine Result DTO
# ---------------------------------------------------------------------


class GLMEngineResult(NamedTuple):
    """Container for the raw numerical results of the GLM fit."""

    params: FloatArray
    cov_params: FloatArray
    scale: float
    df_resid: float
    n_obs: int
    deviance: float
    null_deviance: float
    aic: float | None
    bic: float | None
    r_squared: float | None
    r_squared_adj: float | None
    iterations: int | None


# ---------------------------------------------------------------------
# Vectorized Math Helpers (Likelihood, Link, Variance)
# ---------------------------------------------------------------------


def _inverse_link(eta: FloatArray, link: str) -> FloatArray:
    """Map linear predictor (eta) to mean (mu)."""
    if link == "identity":
        return eta
    if link == "logit":
        # Numerical stability: overflow protection for exp
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    if link == "log":
        return np.exp(np.clip(eta, -30, 30))
    if link == "inverse":
        return 1.0 / eta
    if link == "inverse_squared":
        return 1.0 / np.sqrt(eta)
    raise ValueError(f"Unsupported link: {link}")


def _deriv_link(mu: FloatArray, link: str) -> FloatArray:
    """Derivative of the link function with respect to mu: d(eta)/d(mu)."""
    if link == "identity":
        return np.ones_like(mu)
    if link == "logit":
        # Protected division
        return 1.0 / (mu * (1.0 - mu))
    if link == "log":
        return 1.0 / mu
    if link == "inverse":
        return -1.0 / (mu**2)
    if link == "inverse_squared":
        return -2.0 / (mu**3)
    raise ValueError(f"Unsupported link: {link}")


def _variance_family(mu: FloatArray, family: str) -> FloatArray:
    """Variance function V(mu)."""
    if family == "gaussian":
        return np.ones_like(mu)
    if family == "binomial":
        return mu * (1.0 - mu)
    if family == "poisson":
        return mu
    if family == "gamma":
        return mu**2
    if family == "inversegaussian":
        return mu**3
    raise ValueError(f"Unsupported family: {family}")


def _calculate_deviance_vectorized(
    y: FloatArray, mu: FloatArray, w: FloatArray, family: str
) -> float:
    """Calculate Deviance (2 * (LL_sat - LL_model))."""
    eps = 1e-10

    if family == "gaussian":
        return float(np.sum(w * (y - mu) ** 2))

    if family == "binomial":
        mu = np.clip(mu, eps, 1 - eps)
        # FIX: Ensure 'd' is float, even if 'y' is int (0/1)
        d = np.zeros_like(y, dtype=np.float64)

        # For y > 0 term
        mask1 = y > 0
        d[mask1] += y[mask1] * np.log(y[mask1] / mu[mask1])

        # For y < 1 term
        mask2 = y < 1
        d[mask2] += (1 - y[mask2]) * np.log((1 - y[mask2]) / (1 - mu[mask2]))

        return 2.0 * float(np.sum(w * d))

    if family == "poisson":
        mu = np.maximum(mu, eps)
        # FIX: Ensure 'd' is float, even if 'y' is int counts
        d = np.zeros_like(y, dtype=np.float64)
        mask = y > 0
        d[mask] = y[mask] * np.log(y[mask] / mu[mask])
        return 2.0 * float(np.sum(w * (d - (y - mu))))

    if family == "gamma":
        mu = np.maximum(mu, eps)
        return 2.0 * float(np.sum(w * (-np.log(y / mu) + (y - mu) / mu)))

    if family == "inversegaussian":
        mu = np.maximum(mu, eps)
        return float(np.sum(w * ((y - mu) ** 2) / (y * mu**2)))

    return 0.0


# ---------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------


def _solve_wls_numpy(X: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Numerically stable Weighted Least Squares using SVD/QR (via lstsq).

    Instead of solving (X.T @ W @ X) @ beta = X.T @ W @ z  <-- Squares Condition Number (BAD)
    We solve:          sqrt(W) @ X @ beta = sqrt(W) @ z    <-- Preserves Condition Number (GOOD)
    """
    # 1. Calculate sqrt of weights
    # Safe sqrt: handle potential negative weights (though they shouldn't exist in IRLS)
    sqrt_w = np.sqrt(np.maximum(w, 0))

    # 2. Scale X and z
    # Broadcasting sqrt_w (n,) onto X (n, p) implies row-wise multiplication
    X_scaled = X * sqrt_w[:, None]
    z_scaled = z * sqrt_w

    # 3. Solve using Least Squares
    # rcond=None uses standard machine precision defaults to handle singular matrices
    beta, residuals, rank, s = np.linalg.lstsq(X_scaled, z_scaled, rcond=None)

    return beta


def _fit_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    w_samp: np.ndarray,
    family: str,
    link: str,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """IRLS solver with Canonical Optimizations, Step Halving, and Robust SVD."""

    # 1. Scale weights to sample size 'n' (Critical for numerical stability)
    w_sum = w_samp.sum()
    if w_sum > 0:
        w_run = w_samp * (len(y) / w_sum)
    else:
        w_run = w_samp

    # 2. Initialization
    n_feats = X.shape[1]
    y_mean = np.average(y, weights=w_run) if w_run.sum() > 0 else np.mean(y)
    beta = np.zeros(n_feats)

    # Smart start
    if link == "log":
        beta[0] = np.log(max(y_mean, 1e-3))
    elif link == "logit":
        p = np.clip(y_mean, 1e-3, 1 - 1e-3)
        beta[0] = np.log(p / (1 - p))
    elif link == "inverse":
        beta[0] = 1.0 / max(y_mean, 1e-3)

    # Initial Deviance
    mu = _inverse_link(X @ beta, link)
    dev_old = _calculate_deviance_vectorized(y, mu, w_run, family)

    converged = False
    i = 0
    diff = 0.0

    for i in range(max_iter):
        # Clip mu to prevent explosions in derivative at the edges
        if family == "binomial":
            mu = np.clip(mu, 1e-10, 1.0 - 1e-10)
        elif family == "poisson":
            mu = np.maximum(mu, 1e-10)

        # --- CANONICAL LINK OPTIMIZATION ---
        # Detect if we are using standard links (Logistic, Poisson, etc.)
        # In these cases, the math simplifies: Weights = Variance * w_run
        is_canonical = (
            (family == "binomial" and link == "logit")
            or (family == "poisson" and link == "log")
            or (family == "gaussian" and link == "identity")
        )

        if is_canonical:
            # 1. Calculate Variance term only
            if family == "binomial":
                var_weights = mu * (1.0 - mu)
            elif family == "poisson":
                var_weights = mu
            else:
                var_weights = np.ones_like(mu)

            # 2. Simplified Weights (No division by derivative squared)
            w_irls = w_run * var_weights

            # 3. Calculate Derivative (1/V) just for z
            # Note: We still divide here, but we've avoided the squared instability in 'w'
            deriv = 1.0 / var_weights

        else:
            # --- GENERIC FALLBACK (For custom links) ---
            deriv = _deriv_link(mu, link)
            var_func = _variance_family(mu, family)

            # Guard against division by zero
            deriv = np.where(np.abs(deriv) < 1e-9, 1e-9, deriv)
            var_func = np.where(var_func < 1e-9, 1e-9, var_func)

            w_irls = w_run / (var_func * (deriv**2))

        # Working response
        z = (X @ beta) + (y - mu) * deriv

        # --- ROBUST SOLVER CALL ---
        beta_new = _solve_wls_numpy(X, z, w_irls)

        # --- STEP HALVING (Backtracking) ---
        step_factor = 1.0
        beta_candidate = beta_new
        dev_new = dev_old

        for backtrack_i in range(10):
            mu_new = _inverse_link(X @ beta_candidate, link)
            dev_new = _calculate_deviance_vectorized(y, mu_new, w_run, family)

            # Allow tiny increase (1e-5) for numerical noise
            if dev_new < dev_old + 1e-5:
                break

            # Reject step: backtrack towards old beta
            step_factor *= 0.5
            beta_candidate = beta * (1 - step_factor) + beta_new * step_factor

        # Accept the best candidate found
        diff = np.linalg.norm(beta_candidate - beta)
        beta = beta_candidate
        mu = _inverse_link(X @ beta, link)

        # --- DEBUG LOGGING ---
        log.debug(
            "IRLS Iter %03d | Diff: %.6f | Deviance: %.4f -> %.4f | Step: %.4f",
            i + 1,
            diff,
            dev_old,
            dev_new,
            step_factor,
        )
        # ---------------------

        dev_old = dev_new

        if diff < tol:
            converged = True
            break

    if not converged and i >= max_iter - 1:
        raise ModelError.convergence_failed(
            where="GLM Engine (IRLS)",
            iterations=max_iter,
            tol=tol,
            last_diff=float(diff),
            hint="Try rescaling weights or checking for perfect separation.",
        )

    # Final Weights Calculation (Using ORIGINAL w_samp for correct Variance SE)
    if family == "binomial":
        mu = np.clip(mu, 1e-10, 1.0 - 1e-10)

    deriv = _deriv_link(mu, link)
    var_func = _variance_family(mu, family)

    deriv = np.where(np.abs(deriv) < 1e-10, 1e-10, deriv)
    var_func = np.where(var_func < 1e-10, 1e-10, var_func)

    w_final = w_samp / (var_func * (deriv**2))

    return beta, w_final, mu, i + 1


# ---------------------------------------------------------------------
# Variance & Diagnostics (Sandwich)
# ---------------------------------------------------------------------


def _compute_sandwich(X, y, mu, w_samp, w_irls, strata, psu, fpc):
    # Bread: (X' W X)^-1
    XtWX = (X.T * w_irls) @ X
    D = np.linalg.pinv(XtWX)

    # Meat: Sum of squares of scores within PSUs
    resid = y - mu
    scores = X * (resid * w_samp)[:, None]

    n_obs, n_params = scores.shape
    G = np.zeros((n_params, n_params))

    S = strata if strata is not None else np.zeros(n_obs)
    P = psu if psu is not None else np.arange(n_obs)
    fpc_map = fpc if isinstance(fpc, dict) else {"__all__": float(fpc)}

    # Fast path for no strata/clusters?
    # For now, keep robust loop
    unique_strata = np.unique(S)
    for s in unique_strata:
        mask = S == s
        scores_s = scores[mask]
        psu_s = P[mask]

        # Aggregated scores per PSU
        unique_psus, inv_idx = np.unique(psu_s, return_inverse=True)
        n_psu = len(unique_psus)

        if n_psu <= 1:
            continue  # Variance undefined or 0 contribution

        psu_totals = np.zeros((n_psu, n_params))
        np.add.at(psu_totals, inv_idx, scores_s)

        # Variance of totals
        center = psu_totals - np.mean(psu_totals, axis=0)
        var_h = (n_psu / (n_psu - 1)) * (center.T @ center)

        f = fpc_map.get(s, fpc_map.get(str(s), fpc_map.get("__all__", 1.0)))
        G += f * var_h

    return D @ G @ D


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------


def fit_glm_engine(
    *,
    data: pl.DataFrame | pl.LazyFrame,
    y: str,
    features: list[str],
    weight: str,
    family: str,
    link: str,
    stratum: str | Sequence[str] | None = None,
    psu: str | Sequence[str] | None = None,
    fpc: Any = 1.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> GLMEngineResult:
    # 1. Materialize
    cols = [y, weight] + features
    s_norm = _normalize_name_seq(stratum, param="stratum", where="fit_glm_engine")
    if s_norm:
        cols.extend(s_norm)
        stratum_col_name = s_norm[0]
    else:
        stratum_col_name = None
    p_norm = _normalize_name_seq(psu, param="psu", where="fit_glm_engine")
    if p_norm:
        cols.extend(p_norm)
        psu_col_name = p_norm[0]
    else:
        psu_col_name = None

    subset = data.select(cols)
    if isinstance(subset, pl.LazyFrame):
        mat: pl.DataFrame = subset.collect()  # type: ignore[assignment]
    else:
        mat = subset if isinstance(subset, pl.DataFrame) else subset.collect()

    y_arr = mat[y].to_numpy()
    X = mat.select(features).to_numpy()
    w = mat[weight].to_numpy()

    # 2. Fit (IRLS)
    beta, w_irls, mu, iters = _fit_coefficients(X, y_arr, w, family, link, tol, max_iter)

    # 3. Variance (Sandwich)
    S = mat[stratum_col_name].to_numpy() if stratum_col_name else None
    P = mat[psu_col_name].to_numpy() if psu_col_name else None
    cov = _compute_sandwich(X, y_arr, mu, w, w_irls, S, P, fpc)

    # 4. Stats
    n_psu = len(np.unique(P)) if P is not None else len(y)
    n_strata = len(np.unique(S)) if S is not None else 0
    df_resid = max(1.0, float(n_psu - n_strata))

    # Scale parameter (phi)
    # For Binomial/Poisson, scale is fixed at 1.0 (unless overdispersed quasi-model)
    # For Gaussian/Gamma, estimated from residuals.
    if family in ("gaussian", "gamma", "inversegaussian"):
        # Pearson chi2 / df
        var_func = _variance_family(mu, family)
        pearson = np.sum(w * (y - mu) ** 2 / var_func)
        scale = pearson / df_resid
    else:
        scale = 1.0

    model_deviance = _calculate_deviance_vectorized(y_arr, mu, w, family)

    # Null Deviance
    w_sum = w.sum()
    y_bar = float(np.average(y_arr, weights=w)) if w_sum > 0 else float(np.mean(y_arr))
    mu_null = np.full_like(y_arr, y_bar)
    null_deviance = _calculate_deviance_vectorized(y_arr, mu_null, w, family)

    # R2
    r_squared = None
    r_squared_adj = None
    if null_deviance > 1e-12:
        r_squared = 1.0 - (model_deviance / null_deviance)
        n_obs = len(y)
        # p_rank = num predictors (cols in X) - 1 (if intercept exists)
        has_intercept = "_intercept_" in features
        p_rank = X.shape[1] - (1 if has_intercept else 0)
        if n_obs > p_rank + 1:
            r_squared_adj = 1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - p_rank - 1)

    # AIC
    k = X.shape[1]
    aic = None
    bic = None
    if family == "gaussian":
        # AIC for Gaussian uses log-likelihood of normal
        # LL = -n/2 * (log(2*pi) + log(RSS/n) + 1)
        # Simpler approximation linked to deviance:
        aic = len(y) * np.log(model_deviance / len(y)) + 2 * k
    else:
        aic = model_deviance + 2 * k

    if aic is not None:
        bic = (aic - 2 * k) + k * np.log(len(y))

    return GLMEngineResult(
        params=beta,
        cov_params=cov,
        scale=scale,
        df_resid=df_resid,
        n_obs=len(y),
        deviance=model_deviance,
        null_deviance=null_deviance,
        aic=aic,
        bic=bic,
        r_squared=r_squared,
        r_squared_adj=r_squared_adj,
        iterations=iters,
    )
