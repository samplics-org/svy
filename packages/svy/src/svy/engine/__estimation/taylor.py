# src/svy/engine/estimation/taylor.py
from __future__ import annotations

import math
import re

import numpy as np

from scipy.stats import t as student

from svy.core.enumerations import PopParam, QuantileMethod
from svy.core.types import (
    Array,
    Category,
    DomainCatMap,
    DomainScalarMap,
    FloatArray,
    Number,
)
from svy.estimation.estimate import ParamEst
from svy.utils.checks import as_1d, to_stringnumber


# ---------------------------------------------------------------------
# HIGH-PERFORMANCE HELPERS
# ---------------------------------------------------------------------


def _factorize(arr: Array) -> tuple[np.ndarray, int]:
    """
    Fast factorization of an input array (str, int, object) into
    contiguous integers [0, 1, ... K-1].
    """
    if arr.ndim > 1:
        # Fallback for 2D object arrays:
        arr = np.array([tuple(row) for row in arr], dtype=object)

    _, indices = np.unique(arr, return_inverse=True)
    return indices, indices.max() + 1


def _sum_by_group_indices(
    values: FloatArray, group_indices: np.ndarray, n_groups: int
) -> FloatArray:
    """
    Vectorized group summation using np.bincount.
    """
    n, k = values.shape
    out = np.zeros((n_groups, k), dtype=float)

    for col in range(k):
        out[:, col] = np.bincount(group_indices, weights=values[:, col], minlength=n_groups)

    return out


# ---------------------------------------------------------------------
# Helpers for categorical handling and weighted quantiles
# ---------------------------------------------------------------------


def _labels1d(arr: Array | None) -> Array | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=object)
    if a.ndim == 1:
        return a
    return np.array([tuple(row) for row in a], dtype=object)


def _get_dummies_and_categories(
    *,
    y: Array,
    prop_positive: Category = 1,
    ensure_positive_first: bool = True,
) -> tuple[FloatArray, list[Category]]:
    """Dummy-encode y with a stable order."""
    y_arr = np.asarray(y, dtype=object)
    y_norm_list = [to_stringnumber(token=v) for v in y_arr]

    seen = set()
    cats = []
    for x in y_norm_list:
        if x not in seen:
            cats.append(x)
            seen.add(x)

    _NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")

    def sort_key(v: Category) -> tuple[int, float | str]:
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return (0, float(v))
        s = str(v)
        if _NUMERIC_RE.fullmatch(s.strip()):
            return (1, float(s))
        return (2, s)

    ordered = sorted(cats, key=sort_key)

    if ensure_positive_first:
        prop_pos_norm = to_stringnumber(token=prop_positive)
        if prop_pos_norm in seen:
            ordered.remove(prop_pos_norm)
            ordered.insert(0, prop_pos_norm)

    y_norm = np.array(y_norm_list, dtype=object)
    ordered_arr = np.array(ordered, dtype=object)
    dummies = (y_norm[:, None] == ordered_arr[None, :]).astype(float)

    return dummies, ordered


def _weighted_quantile(
    *, y_sorted: FloatArray, cdf: FloatArray, p: float, q_method: QuantileMethod
) -> float:
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")

    n = y_sorted.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(y_sorted[0])

    if p <= cdf[0]:
        left, right = 0, min(1, n - 1)
    elif p >= cdf[-1]:
        left, right = max(n - 2, 0), n - 1
    else:
        idx = int(np.searchsorted(cdf, p, side="right"))
        left, right = max(idx - 1, 0), min(idx, n - 1)

    if q_method == QuantileMethod.LOWER:
        return float(y_sorted[left])
    if q_method == QuantileMethod.HIGHER:
        return float(y_sorted[right])
    if q_method == QuantileMethod.MIDDLE:
        return float((y_sorted[left] + y_sorted[right]) / 2)
    if q_method == QuantileMethod.NEAREST:
        dl, dr = abs(p - cdf[left]), abs(cdf[right] - p)
        return float(y_sorted[left] if dl <= dr else y_sorted[right])

    denom = float(cdf[right] - cdf[left])
    if denom <= 0:
        return float(y_sorted[left])

    if p == float(cdf[left]) and right != left:
        return float((y_sorted[left] + y_sorted[right]) / 2)

    w = (p - float(cdf[left])) / denom
    return float((1 - w) * float(y_sorted[left]) + w * float(y_sorted[right]))


def _weighted_median(*, y: Array, wgt: FloatArray, q_method: QuantileMethod) -> float:
    y = np.asarray(y, dtype=float)
    wgt = np.asarray(wgt, dtype=float)
    order = np.argsort(y, kind="mergesort")
    ys = y[order]
    ws = wgt[order]
    csum = np.cumsum(ws)
    total_w = csum[-1]
    if total_w == 0:
        return float("nan")
    cdf = csum / total_w
    return _weighted_quantile(y_sorted=ys, cdf=cdf, p=0.5, q_method=q_method)


# ---------------------------------------------------------------------
# Point estimators
# ---------------------------------------------------------------------


def _point_mean(*, y: Array, wgt: FloatArray) -> float:
    den = wgt.sum()
    if den == 0:
        raise ZeroDivisionError("Sum of weights is zero.")
    return float(np.average(y, weights=wgt))


def _point_total(*, y: Array, wgt: FloatArray) -> float:
    return float(np.dot(wgt, y))


def _point_ratio(*, y: Array, wgt: FloatArray, x: Array) -> float:
    den = float(np.dot(wgt, x))
    if den == 0:
        raise ZeroDivisionError("Weighted sum of x is zero in ratio estimation.")
    return float(np.dot(wgt, y) / den)


def _get_point_scalar(
    *,
    param: PopParam,
    y: Array,
    wgt: FloatArray,
    x: Array | None = None,
    q_method: QuantileMethod | None = None,
) -> Number:
    y = np.asarray(y, dtype=float)
    wgt = np.asarray(wgt, dtype=float)
    if param in (PopParam.MEAN, PopParam.PROP):
        return _point_mean(y=y, wgt=wgt)
    if param == PopParam.TOTAL:
        return _point_total(y=y, wgt=wgt)
    if param == PopParam.RATIO:
        assert x is not None
        return _point_ratio(y=y, wgt=wgt, x=np.asarray(x, dtype=float))
    if param == PopParam.MEDIAN:
        if q_method is None:
            raise ValueError("q_method required for MEDIAN.")
        return _weighted_median(y=y, wgt=wgt, q_method=q_method)
    raise ValueError("Parameter not valid!")


def _get_point_no_domain(
    *,
    param: PopParam,
    y: Array,
    wgt: FloatArray,
    x: Array | None = None,
    q_method: QuantileMethod | None = None,
    as_factor: bool = False,
    prop_positive: Category = 1,
) -> Number | DomainScalarMap:
    y = np.asarray(y)
    wgt = np.asarray(wgt, dtype=float)
    if (param == PopParam.PROP) or as_factor:
        y_dum, cats = _get_dummies_and_categories(
            y=y, prop_positive=prop_positive, ensure_positive_first=True
        )
        return {
            cats[k]: _get_point_scalar(param=param, y=y_dum[:, k], wgt=wgt, x=x, q_method=q_method)
            for k in range(y_dum.shape[1])
        }
    return _get_point_scalar(param=param, y=y, wgt=wgt, x=x, q_method=q_method)


def _get_point_by(
    *,
    param: PopParam,
    y: Array,
    wgt: FloatArray,
    by: Array,
    x: Array | None = None,
    q_method: QuantileMethod | None = None,
    as_factor: bool = False,
    prop_positive: Category = 1,
    on_empty_weight: str = "nan",
) -> DomainScalarMap | DomainCatMap:
    y = np.asarray(y)
    wgt = np.asarray(wgt, dtype=float)
    d = as_1d(a=np.asarray(by), name="domain")
    if d.shape[0] != y.shape[0]:
        raise ValueError("domain must be the same length as y.")

    dom_ids, inv = np.unique(d, return_inverse=True)
    is_categorical = (param == PopParam.PROP) or as_factor

    if is_categorical:
        y_dum, cats = _get_dummies_and_categories(
            y=y, prop_positive=prop_positive, ensure_positive_first=True
        )
        nb = y_dum.shape[1]
        out: DomainCatMap = {}
        for j, dom in enumerate(dom_ids):
            mask = inv == j
            wj = wgt[mask]
            if wj.sum() <= 0:
                out[dom] = {c: float("nan") for c in cats}
                continue
            dom_map = {}
            for k in range(nb):
                dom_map[cats[k]] = _get_point_scalar(
                    param=param, y=y_dum[mask, k], wgt=wj, x=x, q_method=q_method
                )
            out[dom] = dom_map
        return out

    out_s: DomainScalarMap = {}
    x_arr = np.asarray(x) if x is not None else None
    for j, dom in enumerate(dom_ids):
        mask = inv == j
        wj = wgt[mask]
        if wj.sum() <= 0:
            out_s[dom] = float("nan")
            continue
        xx = x_arr[mask] if x_arr is not None else None
        out_s[dom] = _get_point_scalar(param=param, y=y[mask], wgt=wj, x=xx, q_method=q_method)
    return out_s


# ---------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------


def _scores_mean(*, y: Array, wgt: FloatArray) -> FloatArray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    den = wgt.sum()
    mu = np.sum(wgt[:, None] * y, axis=0) / den
    return (wgt[:, None] * (y - mu)) / den


def _scores_total(*, y: Array, wgt: FloatArray) -> FloatArray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    return wgt[:, None] * y


def _scores_ratio(*, y: Array, wgt: FloatArray, x: Array) -> FloatArray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim == 1:
        x = x[:, None]
    den = np.sum(wgt[:, None] * x, axis=0)
    if np.any(den == 0):
        raise ZeroDivisionError("Weighted sum of x is zero in ratio estimation.")
    rhat = np.sum(wgt[:, None] * y, axis=0) / den
    return (wgt[:, None] * (y - x * rhat)) / den


def _score_variable(
    *, param: PopParam, y: Array, wgt: FloatArray, x: Array | None = None
) -> FloatArray:
    require_x = param == PopParam.RATIO
    if require_x and x is None:
        raise ValueError("x must be provided for ratio estimation")
    y = np.asarray(y, dtype=float)
    wgt = as_1d(a=np.asarray(wgt, dtype=float), name="samp_weight")
    if y.ndim == 1:
        y = y[:, None]
    if require_x:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
    if param in (PopParam.MEAN, PopParam.PROP):
        return _scores_mean(y=y, wgt=wgt)
    if param == PopParam.TOTAL:
        return _scores_total(y=y, wgt=wgt)
    if param == PopParam.RATIO:
        assert x is not None
        return _scores_ratio(y=y, wgt=wgt, x=x)
    if param == PopParam.MEDIAN:
        raise NotImplementedError("Scores for MEDIAN are not implemented via this path.")
    raise ValueError(f"Parameter {param} not valid!")


# ---------------------------------------------------------------------
# VARIANCE OPTIMIZATION
# ---------------------------------------------------------------------


def _variance_stratum_between_optimized(
    *, y_score_s: Array, psu_indices: np.ndarray | None, n_psus: int = 0
) -> FloatArray:
    """
    Optimized variance calculation within a single stratum using bincount.
    Computes between-PSU (or between-unit) variance.
    """
    Y = np.asarray(y_score_s, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, k = Y.shape

    if n == 0:
        return np.zeros((k, k), dtype=float)

    # 1. CLUSTERED CASE (PSUs)
    if psu_indices is not None and n_psus > 1:
        # Sum scores by PSU
        T = _sum_by_group_indices(Y, psu_indices, n_psus)

        # Variance of PSU totals
        T_bar = T.mean(axis=0, keepdims=True)
        diff = T - T_bar

        m = n_psus
        cov = (m / (m - 1.0)) * (diff.T @ diff)
        return np.asarray(cov, dtype=float)

    # 2. SIMPLE RANDOM CASE
    if n <= 1:
        return np.zeros((k, k), dtype=float)

    Y_bar = Y.mean(axis=0, keepdims=True)
    diff = Y - Y_bar
    cov = (n / (n - 1.0)) * (diff.T @ diff)
    return np.asarray(cov, dtype=float)


def _variance_within_psu(
    *,
    y_score_psu: FloatArray,
    ssu_indices: np.ndarray | None,
    n_ssus: int,
) -> FloatArray:
    """
    Compute within-PSU variance (between SSUs) for second stage.

    This calculates the variance of SSU totals within a single PSU,
    which represents the second-stage sampling variance.

    Parameters
    ----------
    y_score_psu : FloatArray
        Score values for observations within this PSU
    ssu_indices : np.ndarray | None
        Integer indices identifying SSUs (0 to n_ssus-1)
    n_ssus : int
        Number of unique SSUs in this PSU

    Returns
    -------
    FloatArray
        Covariance matrix of shape (k, k) where k is the number of variables
    """
    Y = np.asarray(y_score_psu, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, k = Y.shape

    if n == 0 or n_ssus <= 1:
        return np.zeros((k, k), dtype=float)

    if ssu_indices is not None and n_ssus > 1:
        # Sum scores by SSU
        T = _sum_by_group_indices(Y, ssu_indices, n_ssus)
        T_bar = T.mean(axis=0, keepdims=True)
        diff = T - T_bar
        m = n_ssus
        cov = (m / (m - 1.0)) * (diff.T @ diff)
        return np.asarray(cov, dtype=float)

    return np.zeros((k, k), dtype=float)


def _taylor_variance(
    *,
    y_score: Array,
    wgt: FloatArray,
    stratum: Array | None = None,
    psu: Array | None = None,
    ssu: Array | None = None,
    fpc: dict[Category, Number] | Number = 1,
    fpc_stage2: dict[Category, Number] | Number = 1,
) -> FloatArray:
    """
    Taylor Variance calculation with optional two-stage support.

    When ssu is None: one-stage variance (between PSUs within strata)
    When ssu is provided: two-stage variance using the formula:

        V(θ̂) = V₁ + Σ_h (n_h/N_h) * V₂_h

    Where:
        V₁ = between-PSU variance (first stage)
        V₂_h = within-PSU variance for PSU h (second stage)
        n_h/N_h = first-stage sampling fraction for stratum containing PSU h

    Parameters
    ----------
    y_score : array-like
        Score variables (linearized)
    wgt : array-like
        Sampling weights
    stratum : array-like, optional
        Stratum identifiers
    psu : array-like, optional
        Primary sampling unit identifiers
    ssu : array-like, optional
        Secondary sampling unit identifiers (for two-stage designs)
    fpc : dict or Number
        Finite population correction for stage 1 (by stratum).
        If dict, keys are stratum values and values are FPC factors.
        FPC = (N - n) / N where N is population size and n is sample size.
    fpc_stage2 : dict or Number
        Finite population correction for stage 2 (by PSU).
        If dict, keys are PSU values and values are FPC factors.

    Returns
    -------
    FloatArray
        Covariance matrix
    """
    Y = np.asarray(y_score, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, k = Y.shape
    if n == 0:
        return np.zeros((k, k), dtype=float)

    # Factorize PSU once globally
    P_indices: np.ndarray | None = None
    psu_arr = _labels1d(psu)
    if psu_arr is not None:
        P_indices, _ = _factorize(psu_arr)

    # =========================================================================
    # STAGE 1: Between-PSU variance
    # =========================================================================

    S = _labels1d(stratum)

    # Single Stratum Case
    if S is None:
        m = (P_indices.max() + 1) if P_indices is not None else 0
        cov_stage1 = _variance_stratum_between_optimized(
            y_score_s=Y, psu_indices=P_indices, n_psus=m
        )
        fpc_val = fpc if isinstance(fpc, (int, float)) else 1
        cov_stage1 = fpc_val * cov_stage1
    else:
        # Multi-Stratum Case (Optimized Sort-Split)
        if S.shape[0] != n:
            raise ValueError("stratum must have the same length as y_score.")

        sort_idx = np.argsort(S)
        S_sorted = S[sort_idx]
        Y_sorted = Y[sort_idx]
        P_sorted = P_indices[sort_idx] if P_indices is not None else None

        unique_strata, start_indices = np.unique(S_sorted, return_index=True)
        cov_stage1 = np.zeros((k, k), dtype=float)

        for i, s_val in enumerate(unique_strata):
            start = start_indices[i]
            end = start_indices[i + 1] if i + 1 < len(unique_strata) else n

            y_slice = Y_sorted[start:end]
            p_slice = P_sorted[start:end] if P_sorted is not None else None

            m_local = 0
            if p_slice is not None:
                p_slice, m_local = _factorize(p_slice)

            cov_s = _variance_stratum_between_optimized(
                y_score_s=y_slice, psu_indices=p_slice, n_psus=m_local
            )

            fpc_s = fpc.get(s_val, 1) if isinstance(fpc, dict) else fpc
            cov_stage1 += fpc_s * cov_s

    # =========================================================================
    # STAGE 2: Within-PSU variance (if ssu provided)
    # =========================================================================

    if ssu is None:
        return np.asarray(cov_stage1, dtype=float)

    # Two-stage variance estimation
    ssu_arr = _labels1d(ssu)
    if ssu_arr is None:
        return np.asarray(cov_stage1, dtype=float)

    if psu_arr is None:
        raise ValueError("psu must be provided for two-stage variance estimation")

    unique_psus = np.unique(psu_arr)
    cov_stage2 = np.zeros((k, k), dtype=float)

    # Build a mapping from PSU to its stratum (for FPC lookup)
    psu_to_stratum: dict = {}
    if S is not None:
        for psu_val in unique_psus:
            mask = psu_arr == psu_val
            psu_to_stratum[psu_val] = S[mask][0]

    for psu_val in unique_psus:
        mask = psu_arr == psu_val
        Y_psu = Y[mask]
        ssu_psu = ssu_arr[mask]

        # Get unique SSUs within this PSU
        ssu_indices_local, n_ssus = _factorize(ssu_psu)

        if n_ssus <= 1:
            # Only one SSU in this PSU, no within-PSU variance contribution
            continue

        # Compute within-PSU variance (between SSUs)
        cov_within_psu = _variance_within_psu(
            y_score_psu=Y_psu,
            ssu_indices=ssu_indices_local,
            n_ssus=n_ssus,
        )

        # Get stage 1 FPC for scaling
        # The second-stage contribution is scaled by the first-stage sampling fraction
        # If fpc1 = (N - n) / N, then sampling fraction = n/N = 1 - fpc1
        if S is not None and psu_val in psu_to_stratum:
            stratum_for_psu = psu_to_stratum[psu_val]
            fpc1_val = fpc.get(stratum_for_psu, 1) if isinstance(fpc, dict) else fpc
        else:
            fpc1_val = fpc if isinstance(fpc, (int, float)) else 1

        # Get stage 2 FPC
        if isinstance(fpc_stage2, dict):
            fpc2_val = fpc_stage2.get(psu_val, 1)
        else:
            fpc2_val = fpc_stage2

        # The multiplier for V_2 is the first-stage sampling fraction: n_1/N_1 = 1 - fpc1
        # V_total = V_1 + (n_1/N_1) * V_2
        stage1_sampling_fraction = 1.0 - fpc1_val

        cov_stage2 += stage1_sampling_fraction * fpc2_val * cov_within_psu

    return np.asarray(cov_stage1 + cov_stage2, dtype=float)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _block_diag(*, blocks: list[FloatArray]) -> FloatArray:
    if not blocks:
        return np.zeros((0, 0), dtype=float)
    sizes = [b.shape[0] for b in blocks]
    total = int(sum(sizes))
    out = np.zeros((total, total), dtype=float)
    off = 0
    for B in blocks:
        k = B.shape[0]
        out[off : off + k, off : off + k] = B
        off += k
    return np.asarray(out, dtype=float)


def _scores_for_variance(
    *,
    param: PopParam,
    Y: FloatArray,
    wgt: FloatArray,
    x: FloatArray | None,
    q_method: QuantileMethod | None,
) -> FloatArray:
    if param == PopParam.MEDIAN:
        if Y.shape[1] != 1:
            raise ValueError("MEDIAN requires a single target variable (not categorical).")
        if q_method is None:
            raise ValueError("q_method required for MEDIAN.")
        p = 0.5
        order = np.argsort(Y[:, 0], kind="mergesort")
        y_sorted = np.asarray(Y[order, 0], dtype=float)
        w_sorted = np.asarray(wgt[order], dtype=float)
        W = float(np.sum(w_sorted))
        if W <= 0.0:
            raise ValueError("Weights must sum to a positive value.")
        cdf = np.cumsum(w_sorted) / W
        q = _weighted_quantile(y_sorted=y_sorted, cdf=cdf, p=p, q_method=q_method)
        u = (y_sorted > q).astype(float) - (1.0 - p)
        if_scores_sorted = (w_sorted / W) * u
        if_scores = np.empty_like(if_scores_sorted)
        if_scores[order] = if_scores_sorted
        return np.asarray(if_scores[:, None], dtype=float)
    return np.asarray(_score_variable(param=param, y=Y, wgt=wgt, x=x), dtype=float)


def _rstyle_quantile_ci_from_p(
    y: np.ndarray, w: np.ndarray, p0: float, se_p: float, tcrit: float, q_method: QuantileMethod
) -> tuple[float, float]:
    order = np.argsort(y, kind="mergesort")
    ys = np.asarray(y, float)[order]
    ws = np.asarray(w, float)[order]
    W = float(ws.sum())
    cdf = np.cumsum(ws) / W
    p_lo = max(0.0, p0 - tcrit * se_p)
    p_hi = min(1.0, p0 + tcrit * se_p)

    def q_of(p_target):
        if p_target <= cdf[0]:
            return float(ys[0])
        if p_target >= cdf[-1]:
            return float(ys[-1])
        idx = int(np.searchsorted(cdf, p_target, side="right"))
        left, right = idx - 1, idx
        x0, x1 = float(cdf[left]), float(cdf[right])
        y0, y1 = float(ys[left]), float(ys[right])
        if p_target == x0:
            return y0
        wlin = (p_target - x0) / (x1 - x0)
        return (1.0 - wlin) * y0 + wlin * y1

    return q_of(p_lo), q_of(p_hi)


def _srs_variance_wor_normalized_weights(
    *, param: PopParam, y: Array, wgt: Array, x: Array | None = None, by: Array | None = None
) -> Number | DomainScalarMap | DomainCatMap:
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(wgt, dtype=np.float64)
    n = y.size
    if n < 2:
        return float("nan") if by is None else {}
    x_arr = None if x is None else np.asarray(x, dtype=np.float64)
    by_arr = None if by is None else np.asarray(by)

    def _weighted_s2_local(a, wn):
        mu = np.sum(wn * a)
        return (a.size / (a.size - 1)) * np.sum(wn * (a - mu) ** 2)

    def _one_group(yy, ww, xx):
        ng = yy.size
        if ng < 2:
            return float("nan")
        sumw = ww.sum()
        if sumw <= 0:
            return float("nan")
        wn = ww / sumw
        if param in (PopParam.MEAN, PopParam.PROP):
            s2_y = _weighted_s2_local(yy, wn)
            vsrs = s2_y / ng
        elif param == PopParam.TOTAL:
            s2_y = _weighted_s2_local(yy, wn)
            vsrs = (sumw**2 / ng) * s2_y
        elif param == PopParam.RATIO:
            ybar = np.sum(wn * yy)
            xbar = np.sum(wn * xx)
            if xbar == 0:
                return float("nan")
            Rhat = ybar / xbar
            e = yy - Rhat * xx
            s2_e = _weighted_s2_local(e, wn)
            vsrs = s2_e / (ng * xbar**2)
        else:
            return float("nan")
        fpc_val = 1.0 - (ng / sumw)
        return float(vsrs * fpc_val)

    if by_arr is None:
        v = _one_group(y, w, x_arr)
        if param is PopParam.PROP:
            return {0: v, 1: v}  # type: ignore[return-value]
        return v
    cats = np.unique(by_arr).tolist()
    if param is PopParam.PROP:
        out: DomainCatMap = {}
        for c in cats:
            m = by_arr == c
            v = _one_group(y[m], w[m], None if x_arr is None else x_arr[m])
            out[c] = {0: v, 1: v}
        return out
    else:
        out2: DomainScalarMap = {}
        for c in cats:
            m = by_arr == c
            out2[c] = _one_group(y[m], w[m], None if x_arr is None else x_arr[m])
        return out2


# ---------------------------------------------------------------------
# Variance (no domain / by domain)
# ---------------------------------------------------------------------


def _get_variance_no_domain(
    *,
    param: PopParam,
    y: Array,
    wgt: FloatArray,
    x: Array | None = None,
    stratum: Array | None = None,
    psu: Array | None = None,
    ssu: Array | None = None,
    q_method: QuantileMethod | None = None,
    fpc: dict[Category, Number] | Number = 1,
    fpc_stage2: dict[Category, Number] | Number = 1,
    as_factor: bool = False,
) -> tuple[Number | DomainScalarMap, FloatArray]:
    y = np.asarray(y)
    wgt = np.asarray(wgt, dtype=float)
    require_x = param == PopParam.RATIO
    xx = np.asarray(x, dtype=float) if require_x and x is not None else None

    categories = []
    if (param == PopParam.PROP) or as_factor:
        Y, categories = _get_dummies_and_categories(
            y=y, prop_positive=1, ensure_positive_first=True
        )
        Y = Y.astype(float, copy=False)
    else:
        Y = y[:, None] if y.ndim == 1 else y
        Y = np.asarray(Y, dtype=float)

    scores = _scores_for_variance(param=param, Y=Y, wgt=wgt, x=xx, q_method=q_method)
    cov = _taylor_variance(
        y_score=scores,
        wgt=wgt,
        stratum=stratum,
        psu=psu,
        ssu=ssu,
        fpc=fpc,
        fpc_stage2=fpc_stage2,
    )

    if param == PopParam.MEDIAN:
        var_p = float(cov[0, 0])
        var_p = 0.0 if not np.isfinite(var_p) or var_p < 0 else var_p
        se_p = math.sqrt(var_p)
        df_eff = max(_degrees_of_freedom(wgt=wgt, stratum=stratum, psu=psu), 1)
        zcrit = float(student.ppf(1.0 - 0.05 / 2.0, df=df_eff))
        p0 = 0.5
        p_lo = max(0.0, p0 - zcrit * se_p)
        p_hi = min(1.0, p0 + zcrit * se_p)
        y1d = Y[:, 0] if Y.ndim == 2 else np.asarray(Y, dtype=float).ravel()
        order = np.argsort(y1d, kind="mergesort")
        ys = y1d[order]
        ws = np.asarray(wgt, dtype=float)[order]
        W = float(ws.sum())
        cdf = np.cumsum(ws) / W
        q_lo = _weighted_quantile(y_sorted=ys, cdf=cdf, p=p_lo, q_method=q_method)  # type: ignore[arg-type]
        q_hi = _weighted_quantile(y_sorted=ys, cdf=cdf, p=p_hi, q_method=q_method)  # type: ignore[arg-type]
        se_q = (q_hi - q_lo) / (2.0 * zcrit)
        cov = np.asarray([[float(se_q * se_q)]], dtype=float)

    if (param == PopParam.PROP) or as_factor:
        var_map = {categories[i]: float(cov[i, i]) for i in range(len(categories))}
        return var_map, cov
    else:
        return float(cov[0, 0]), cov


def _get_variance_by(
    *,
    param: PopParam,
    y: Array,
    wgt: FloatArray,
    by: Array,
    x: Array | None = None,
    stratum: Array | None = None,
    psu: Array | None = None,
    ssu: Array | None = None,
    q_method: QuantileMethod | None = None,
    fpc: dict[Category, Number] | Number = 1,
    fpc_stage2: dict[Category, Number] | Number = 1,
    as_factor: bool = False,
    prop_positive: Category = 1,
    # Kept for API compatibility
    on_empty_weight: str = "nan",
) -> tuple[DomainScalarMap | DomainCatMap, FloatArray]:
    """
    Vectorized domain variance estimation.
    Replaces the iterative O(G * N) loop with a single O(N) vectorized pass.
    """
    y_arr = np.asarray(y)
    w = np.asarray(wgt, dtype=float)
    n = len(w)

    # 1. Map domains to integer indices 0..G-1
    # We use numpy for speed; assuming 'by' is somewhat homogeneous or str
    dom_vals, group_ids = np.unique(np.asarray(by), return_inverse=True)
    n_groups = len(dom_vals)

    is_categorical = (param == PopParam.PROP) or as_factor

    if is_categorical:
        # --- CATEGORICAL (PROP / FACTOR) ---
        y_dum, cat_labels = _get_dummies_and_categories(
            y=y_arr, prop_positive=prop_positive, ensure_positive_first=True
        )
        n_cats = len(cat_labels)
        y_float = y_dum  # Shape (n, n_cats)

        # 2. Calculate Domain Totals (Vectorized)
        # Sum of weights per domain (G,)
        w_dom = np.bincount(group_ids, weights=w, minlength=n_groups)

        # Sum of Weighted Y per domain (G, n_cats)
        # IMPORTANT: Multiply dummies by weights before summing!
        y_w = y_float * w[:, None]
        wy_dom = _sum_by_group_indices(y_w, group_ids, n_groups)

        # 3. Calculate Point Estimates (Means/Props)
        # Avoid divide by zero
        w_dom_safe = np.where(w_dom == 0, 1.0, w_dom)[:, None]  # (G, 1)
        prop_dom = wy_dom / w_dom_safe  # (G, n_cats)

        # 4. Construct Score Matrix Z
        # Residuals: y_ik - p_g(i),k
        # Advanced indexing: prop_dom[group_ids] broadcasts domain props to every row
        residuals = y_float - prop_dom[group_ids]

        # Scale by weights/pop: w_i / N_g(i)
        scale_factors = w / w_dom_safe[group_ids, 0]  # (n,)

        # Raw scores for every observation
        scores_raw = residuals * scale_factors[:, None]  # (n, n_cats)

        # 5. Place scores into Sparse-Structure Matrix Z (n, G*K)
        # Each row i contributes ONLY to the columns for its domain group_ids[i]
        col_offsets = group_ids * n_cats
        n_cols_total = n_groups * n_cats
        Z = np.zeros((n, n_cols_total), dtype=float)

        # Vectorized placement into Z
        row_idx = np.repeat(np.arange(n), n_cats)
        col_base = np.repeat(col_offsets, n_cats)
        cat_shifts = np.tile(np.arange(n_cats), n)
        col_idx = col_base + cat_shifts

        Z[row_idx, col_idx] = scores_raw.ravel()

        # 6. Single Variance Call
        cov_mat = _taylor_variance(
            y_score=Z,
            wgt=w,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            fpc=fpc,
            fpc_stage2=fpc_stage2,
        )

        # 7. Format Output
        out_map: DomainCatMap = {}
        # The covariance matrix is block diagonal-ish. We only need the diagonals
        # corresponding to the specific domain blocks.
        diag = np.diagonal(cov_mat)

        for g, d_val in enumerate(dom_vals):
            start = g * n_cats
            if w_dom[g] <= 0:
                out_map[d_val] = {c: float("nan") for c in cat_labels}
            else:
                out_map[d_val] = {cat_labels[k]: float(diag[start + k]) for k in range(n_cats)}
        return out_map, cov_mat

    else:
        # --- SCALAR (MEAN / TOTAL / RATIO) ---
        x_arr = np.asarray(x, dtype=float) if x is not None else None
        y_float = y_arr.astype(float)

        # 2. Domain Stats
        w_dom = np.bincount(group_ids, weights=w, minlength=n_groups)
        wy_dom = np.bincount(group_ids, weights=w * y_float, minlength=n_groups)

        scores = np.zeros(n, dtype=float)
        valid_mask = w_dom[group_ids] > 0

        # 3. Calculate Scores based on Param
        if param in (PopParam.MEAN, PopParam.PROP):
            y_bar = np.zeros(n_groups)
            np.divide(wy_dom, w_dom, out=y_bar, where=w_dom > 0)

            N_g = w_dom[group_ids]
            mu_g = y_bar[group_ids]

            # Score: (w / N_d) * (y - y_bar)
            scores[valid_mask] = (w[valid_mask] * (y_float[valid_mask] - mu_g[valid_mask])) / N_g[
                valid_mask
            ]

        elif param == PopParam.TOTAL:
            # Score: w * y (simple)
            scores = w * y_float

        elif param == PopParam.RATIO:
            if x_arr is None:
                raise ValueError("x required for RATIO")
            wx_dom = np.bincount(group_ids, weights=w * x_arr, minlength=n_groups)

            r_hat = np.zeros(n_groups)
            np.divide(wy_dom, wx_dom, out=r_hat, where=wx_dom != 0)

            X_g = wx_dom[group_ids]
            R_g = r_hat[group_ids]

            # Score: (w / X_d) * (y - R * x)
            num = w * (y_float - R_g * x_arr)
            scores[valid_mask] = num[valid_mask] / X_g[valid_mask]

        elif param == PopParam.MEDIAN:
            # Vectorizing median is complex due to sorting; safe fallback or raise
            raise NotImplementedError("Vectorized MEDIAN not supported yet.")

        # 4. Construct Z Matrix (n, n_groups)
        Z = np.zeros((n, n_groups), dtype=float)
        Z[np.arange(n), group_ids] = scores

        # 5. Compute Variance
        cov_mat = _taylor_variance(
            y_score=Z,
            wgt=w,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            fpc=fpc,
            fpc_stage2=fpc_stage2,
        )

        # 6. Format Output
        out_scalar: DomainScalarMap = {}
        diag = np.diagonal(cov_mat)

        for g, d_val in enumerate(dom_vals):
            if w_dom[g] <= 0:
                out_scalar[d_val] = float("nan")
            else:
                out_scalar[d_val] = float(diag[g])

        return out_scalar, cov_mat


# ---------------------------------------------------------------------
# Degrees of freedom (Optimized sort)
# ---------------------------------------------------------------------


def _degrees_of_freedom(
    *, wgt: FloatArray, stratum: Array | None = None, psu: Array | None = None
) -> int:
    S = _labels1d(stratum)
    P = _labels1d(psu) if psu is not None else None
    n = int(wgt.shape[0])
    if n == 0:
        return 0
    if S is None:
        if P is None:
            return max(n - 1, 0)
        m = int(np.unique(P).size)
        return max(m - 1, 0)

    # Sort-based DF calculation (O(N log N))
    sort_idx = np.argsort(S)
    S_sorted = S[sort_idx]
    P_sorted = P[sort_idx] if P is not None else None

    unique_strata, start_indices = np.unique(S_sorted, return_index=True)
    df_sum = 0

    for i in range(len(unique_strata)):
        start = start_indices[i]
        end = start_indices[i + 1] if i + 1 < len(unique_strata) else n

        if P_sorted is None:
            m_h = end - start
        else:
            m_h = np.unique(P_sorted[start:end]).size

        df_sum += max(m_h - 1, 0)

    return int(df_sum)


# ---------------------------------------------------------------------
# Top-level estimator
# ---------------------------------------------------------------------


def _estimate_taylor_no_domain(
    *,
    param: PopParam,
    y_name: str,
    x_name: str | None,
    estimate: dict[Category, Number] | Array | Number,
    variance: dict[Category, Number] | Array | Number,
    variance_srswor_normalized: DomainScalarMap | DomainScalarMap | Number | None,
    t_quantile: float,
    q_method: QuantileMethod,
    as_factor: bool,
    raw_y: np.ndarray | None = None,
    raw_w: np.ndarray | None = None,
    med_se_p: float | None = None,
    p_target: float | None = None,
) -> list[dict]:
    def _to_float(x):
        if x is None:
            return None
        return float(x.item()) if isinstance(x, np.ndarray) and x.size == 1 else float(x)

    def _row(level, est_f, var_f, is_prop, var_srs):
        var_f = max(var_f, 0.0)
        se = math.sqrt(var_f)
        if param == PopParam.MEDIAN and raw_y is not None:
            try:
                p0 = p_target if p_target is not None else 0.5
                lci, uci = _rstyle_quantile_ci_from_p(
                    y=raw_y,
                    w=raw_w,  # type: ignore[arg-type]
                    p0=p0,
                    se_p=float(med_se_p or 0),
                    tcrit=t_quantile,
                    q_method=q_method,
                )
            except Exception:
                lci, uci = est_f - t_quantile * se, est_f + t_quantile * se
        elif is_prop:
            if est_f <= 0 or est_f >= 1:
                lci, uci = est_f, est_f
            else:
                scale = se / (est_f * (1.0 - est_f)) if se > 0 else 0
                logit = math.log(est_f / (1 - est_f))
                lci = 1.0 / (1.0 + math.exp(-(logit - t_quantile * scale)))
                uci = 1.0 / (1.0 + math.exp(-(logit + t_quantile * scale)))
        else:
            lci = est_f - t_quantile * se
            uci = est_f + t_quantile * se
        cv = se / est_f if est_f != 0 else float("inf")
        deff = (var_f / var_srs) if (var_srs is not None and var_srs > 0) else None
        return {
            "by": None,
            "by_level": None,
            "y": y_name,
            "y_level": level,
            "x": x_name,
            "x_level": None,
            "est": est_f,
            "se": se,
            "cv": cv,
            "lci": float(lci),
            "uci": float(uci),
            "deff": deff,
        }

    out = []
    if isinstance(estimate, dict) and isinstance(variance, dict):
        is_prop = (param == PopParam.PROP) or (as_factor and param == PopParam.MEAN)
        for lvl in variance.keys():
            est_f = _to_float(estimate.get(lvl, 0))  # type: ignore[union-attr, arg-type]
            var_f = _to_float(variance[lvl])  # type: ignore[index]
            var_srs = (
                _to_float(variance_srswor_normalized.get(lvl))  # type: ignore[arg-type]
                if isinstance(variance_srswor_normalized, dict)
                else None
            )
            out.append(_row(lvl, est_f, var_f, is_prop, var_srs))
        return out
    est_f = _to_float(estimate)
    var_f = _to_float(variance)
    is_prop = (param == PopParam.PROP) or (as_factor and param == PopParam.MEAN)
    var_srs = (
        _to_float(variance_srswor_normalized) if variance_srswor_normalized is not None else None
    )
    out.append(_row(None, est_f, var_f, is_prop, var_srs))
    return out


def _estimate_taylor(
    *,
    param: PopParam,
    y: Array,
    y_name: str,
    wgt: FloatArray,
    x: Array | None = None,
    x_name: str | None = None,
    stratum: Array | None = None,
    psu: Array | None = None,
    ssu: Array | None = None,
    by: Array | None = None,
    by_name: str | list[str] | None = None,
    deff: bool = False,
    fpc: dict[Category, Number] | Number = 1,
    fpc_stage2: dict[Category, Number] | Number = 1,
    q_method: QuantileMethod | None = None,
    as_factor: bool = False,
    alpha: float = 0.05,
) -> tuple[list[ParamEst], FloatArray]:
    # Normalize inputs
    if by is not None and by.dtype == np.object_:
        by = by.astype(str)
    if stratum is not None and stratum.dtype == np.object_:
        stratum = stratum.astype(str)
    if psu is not None and psu.dtype == np.object_:
        psu = psu.astype(str)
    S = _labels1d(stratum)
    P = _labels1d(psu)
    if S is not None and isinstance(fpc, (int, float)):
        fpc = {s: fpc for s in np.unique(S)}

    df_raw = _degrees_of_freedom(wgt=wgt, stratum=S, psu=P)
    df_eff = max(df_raw, 1)
    t_quantile = float(student.ppf(1.0 - alpha / 2.0, df=df_eff))

    rows: list[dict] = []
    y_all = np.asarray(y)
    w_all = np.asarray(wgt, float)

    if by is None:
        est_no_dom = _get_point_no_domain(
            param=param, y=y_all, wgt=w_all, x=x, q_method=q_method, as_factor=as_factor
        )
        var_no_dom, covariance = _get_variance_no_domain(
            param=param,
            y=y_all,
            wgt=w_all,
            x=x,
            stratum=S,
            psu=P,
            ssu=ssu,
            fpc=fpc,
            fpc_stage2=fpc_stage2,
            q_method=q_method,
            as_factor=as_factor,
        )
        var_srs = (
            _srs_variance_wor_normalized_weights(param=param, y=y_all, wgt=w_all, x=x, by=None)
            if deff
            else None
        )

        med_se_p, raw_y, raw_w = None, None, None
        if param == PopParam.MEDIAN and not isinstance(est_no_dom, dict):
            order = np.argsort(y_all, kind="mergesort")
            ys, ws = y_all[order], w_all[order]
            cdf = np.cumsum(ws) / ws.sum()
            qhat = _weighted_quantile(y_sorted=ys, cdf=cdf, p=0.5, q_method=QuantileMethod.LINEAR)
            U = (y_all < qhat).astype(float)
            var_p, _ = _get_variance_no_domain(
                param=PopParam.PROP,
                y=U,
                wgt=w_all,
                stratum=S,
                psu=P,
                ssu=ssu,
                fpc=fpc,
                fpc_stage2=fpc_stage2,
            )
            med_se_p = float(np.sqrt(var_p if isinstance(var_p, (int, float)) else var_p[1]))
            raw_y, raw_w = y_all, w_all

        rows = _estimate_taylor_no_domain(
            param=param,
            y_name=y_name,
            x_name=x_name,
            estimate=est_no_dom,
            variance=var_no_dom,
            variance_srswor_normalized=var_srs,  # type: ignore[arg-type]
            t_quantile=t_quantile,
            q_method=q_method or QuantileMethod.MIDDLE,
            as_factor=as_factor,
            raw_y=raw_y,
            raw_w=raw_w,
            med_se_p=med_se_p,
            p_target=0.5 if param == PopParam.MEDIAN else None,
        )
    else:
        by_vals = np.asarray(by)
        est_dom = _get_point_by(
            param=param,
            y=y_all,
            wgt=w_all,
            x=x,
            by=by_vals,
            q_method=q_method,
            as_factor=as_factor,
        )
        var_dom, covariance = _get_variance_by(
            param=param,
            y=y_all,
            wgt=w_all,
            by=by_vals,
            x=x,
            stratum=S,
            psu=P,
            ssu=ssu,
            fpc=fpc,
            fpc_stage2=fpc_stage2,
            q_method=q_method,
            as_factor=as_factor,
        )
        var_srs = (
            _srs_variance_wor_normalized_weights(param=param, y=y_all, wgt=w_all, x=x, by=by_vals)
            if deff
            else None
        )

        for d_key, var_d in var_dom.items():
            est_d = est_dom[d_key]
            med_se_p, y_dom, w_dom = None, None, None
            if param == PopParam.MEDIAN and not isinstance(est_d, dict):
                m = by_vals == d_key
                y_sub, w_sub = y_all[m], w_all[m]
                order = np.argsort(y_sub, kind="mergesort")
                ys, ws = y_sub[order], w_sub[order]
                cdf = np.cumsum(ws) / ws.sum()
                qhat = _weighted_quantile(
                    y_sorted=ys, cdf=cdf, p=0.5, q_method=QuantileMethod.LINEAR
                )
                U_mask = (y_all < qhat).astype(float) * m.astype(float)
                w_mask = w_all * m.astype(float)
                var_p, _ = _get_variance_no_domain(
                    param=PopParam.PROP,
                    y=U_mask,
                    wgt=w_mask,
                    stratum=S,
                    psu=P,
                    ssu=ssu,
                    fpc=fpc,
                    fpc_stage2=fpc_stage2,
                )
                med_se_p = float(np.sqrt(var_p if isinstance(var_p, (int, float)) else var_p[1]))
                y_dom, w_dom = y_sub, w_sub

            entries = _estimate_taylor_no_domain(
                param=param,
                y_name=y_name,
                x_name=x_name,
                estimate=est_d,
                variance=var_d,
                variance_srswor_normalized=var_srs[d_key] if var_srs else None,  # type: ignore[index]
                t_quantile=t_quantile,
                q_method=q_method or QuantileMethod.MIDDLE,
                as_factor=as_factor,
                raw_y=y_dom,
                raw_w=w_dom,
                med_se_p=med_se_p,
                p_target=0.5 if param == PopParam.MEDIAN else None,
            )
            dom_val = d_key.tolist() if hasattr(d_key, "tolist") else d_key  # type: ignore[operator]
            for e in entries:
                e["by"] = by_name
                e["by_level"] = dom_val
            rows.extend(entries)

    return [ParamEst(**r) for r in rows], np.asarray(covariance, dtype=float)
