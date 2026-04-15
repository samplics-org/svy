# src/svy/engine/categorical/ranktesting.py
"""
Engine for design-based rank tests under complex sampling.

Implements Lumley & Scott (2013), Biometrika 100(4), 831-842.

The algorithm:
    1. Compute estimated population mid-ranks using survey weights.
    2. Apply a rank-score transformation (Wilcoxon, van der Waerden, median, or custom).
    3. For two groups: estimate mean rank scores per group via _estimate_taylor,
       compute t-test from difference and design-based SE.
    4. For k groups: estimate mean rank scores per group via _estimate_taylor,
       compute Wald/F-test from the joint covariance of group means.

Refactored to reuse the existing Taylor variance infrastructure, which correctly
handles strata, PSUs, SSUs, FPC, singletons, and domain estimation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from scipy import stats as sp_stats

from svy.core.enumerations import PopParam, RankScoreMethod


# =============================================================================
# Score functions
# =============================================================================


def _wilcoxon_score(r: np.ndarray, N: float) -> np.ndarray:
    """Wilcoxon: g(r) = r / N  (proportional ranks)."""
    return r / N


def _vanderwaerden_score(r: np.ndarray, N: float) -> np.ndarray:
    """Van der Waerden: g(r) = Φ⁻¹(r / N)."""
    u = np.clip(r / N, 1e-10, 1.0 - 1e-10)
    return sp_stats.norm.ppf(u)


def _median_score(r: np.ndarray, N: float) -> np.ndarray:
    """Mood's median: g(r) = I(r > N/2)."""
    return (r > N / 2).astype(np.float64)


SCORE_FUNCTIONS: dict[RankScoreMethod, Callable[[np.ndarray, float], np.ndarray]] = {
    RankScoreMethod.KRUSKAL_WALLIS: _wilcoxon_score,
    RankScoreMethod.VANDER_WAERDEN: _vanderwaerden_score,
    RankScoreMethod.MEDIAN: _median_score,
}


# =============================================================================
# Estimated population mid-ranks
# =============================================================================


def _compute_estimated_ranks(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute estimated population mid-ranks (R-hat).

    For each observation i:
        R_hat_i = sum_j [ w_j * I(y_j < y_i) + 0.5 * w_j * I(y_j == y_i) ]

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response variable.
    w : ndarray of shape (n,)
        Sampling weights (positive).

    Returns
    -------
    rankhat : ndarray of shape (n,)
        Estimated population mid-ranks, in original order of y.
    """
    n = len(y)
    ii = np.argsort(y, kind="mergesort")
    y_sorted = y[ii]
    w_sorted = w[ii]

    cumw = np.cumsum(w_sorted)
    midrank_sorted = cumw - w_sorted / 2.0

    # Average over ties
    rankhat_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and y_sorted[j] == y_sorted[i]:
            j += 1
        avg_midrank = np.mean(midrank_sorted[i:j])
        rankhat_sorted[i:j] = avg_midrank
        i = j

    rankhat = np.empty(n, dtype=np.float64)
    rankhat[ii] = rankhat_sorted
    return rankhat


# =============================================================================
# Helper: compute rank scores for a dataset
# =============================================================================


def _compute_rank_scores(
    y: np.ndarray,
    w: np.ndarray,
    score_fn: Callable[[np.ndarray, float], np.ndarray],
) -> np.ndarray:
    """Compute rank scores: ranks → score transformation."""
    N_hat = np.sum(w)
    rankhat = _compute_estimated_ranks(y, w)
    return score_fn(rankhat, N_hat)


# =============================================================================
# Two-sample rank test (using _estimate_taylor)
# =============================================================================


def _ranktest_two_sample(
    y: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    stratum: np.ndarray | None,
    psu: np.ndarray,
    ssu: np.ndarray | None,
    fpc: float,
    score_fn: Callable[[np.ndarray, float], np.ndarray],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict:
    """
    Two-sample design-based rank test.

    Uses _estimate_taylor for design-based variance estimation, ensuring
    correct handling of strata, PSUs, FPC, and domain estimation.

    Returns a dict with keys: delta, se, t, df, p_value, levels,
    group_means, group_ses.
    """
    from svy.engine.categorical.ttesting import _get_subpop_df
    from svy.engine.estimation.taylor import _estimate_taylor

    # 1. Compute rank scores
    rankscore = _compute_rank_scores(y, w, score_fn)

    # 2. Estimate mean rank score per group using Taylor variance
    levels = np.unique(g)
    est_list, cov_mat = _estimate_taylor(
        param=PopParam.MEAN,
        y=rankscore,
        y_name="__rankscore__",
        wgt=w,
        stratum=stratum,
        psu=psu,
        ssu=ssu,
        by=g,
        by_name="__group__",
        fpc=fpc,
        alpha=alpha,
    )

    # Map estimates to group levels (est_list order matches sorted unique levels)
    est_by_level = {}
    for est in est_list:
        est_by_level[est.by_level] = est

    # Ensure we have exactly 2 groups
    if len(est_by_level) != 2:
        raise ValueError(f"Two-sample test requires exactly 2 groups, found {len(est_by_level)}")

    est0 = est_by_level.get(levels[0]) or est_by_level.get(str(levels[0]))
    est1 = est_by_level.get(levels[1]) or est_by_level.get(str(levels[1]))

    if est0 is None or est1 is None:
        # Fallback: use order from est_list
        est0, est1 = est_list[0], est_list[1]

    # 3. Compute difference and SE from covariance matrix
    # cov_mat is 2x2: [[Var(mean0), Cov], [Cov, Var(mean1)]]
    cov_mat = np.asarray(cov_mat)
    delta = est1.est - est0.est
    var_diff = cov_mat[0, 0] + cov_mat[1, 1] - 2 * cov_mat[0, 1]
    se_delta = np.sqrt(max(var_diff, 0.0))

    # 4. Degrees of freedom: degf(design) - 1
    df = _get_subpop_df(stratum, psu) - 1
    df = max(df, 1)

    # 5. Test statistic and p-value
    t_stat = delta / se_delta if se_delta > 0 else np.nan

    if alternative == "two-sided":
        p_value = 2.0 * sp_stats.t.sf(np.abs(t_stat), df=df)
    elif alternative == "less":
        p_value = sp_stats.t.cdf(t_stat, df=df)
    elif alternative == "greater":
        p_value = sp_stats.t.sf(t_stat, df=df)
    else:
        p_value = 2.0 * sp_stats.t.sf(np.abs(t_stat), df=df)

    # 6. Per-group results
    group_means = [float(est0.est), float(est1.est)]
    group_ses = [float(est0.se), float(est1.se)]

    return {
        "delta": float(delta),
        "se": float(se_delta),
        "t": float(t_stat),
        "df": float(df),
        "p_value": float(p_value),
        "levels": (levels[0], levels[1]),
        "group_means": group_means,
        "group_ses": group_ses,
    }


# =============================================================================
# K-sample rank test (using _estimate_taylor)
# =============================================================================


def _ranktest_k_sample(
    y: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    stratum: np.ndarray | None,
    psu: np.ndarray,
    ssu: np.ndarray | None,
    fpc: float,
    score_fn: Callable[[np.ndarray, float], np.ndarray],
    alpha: float = 0.05,
) -> dict:
    """
    K-sample design-based rank test (Kruskal-Wallis style).

    Uses _estimate_taylor for design-based variance estimation. The Wald/F
    test is computed from the joint covariance of group mean rank scores.

    Returns a dict with keys: ndf, ddf, chisq, f_stat, p_value, levels,
    group_means, group_ses.
    """
    from svy.engine.categorical.ttesting import _get_subpop_df
    from svy.engine.estimation.taylor import _estimate_taylor

    # 1. Compute rank scores
    rankscore = _compute_rank_scores(y, w, score_fn)

    # 2. Estimate mean rank score per group
    levels = np.sort(np.unique(g))
    k = len(levels)
    ndf = k - 1

    est_list, cov_mat = _estimate_taylor(
        param=PopParam.MEAN,
        y=rankscore,
        y_name="__rankscore__",
        wgt=w,
        stratum=stratum,
        psu=psu,
        ssu=ssu,
        by=g,
        by_name="__group__",
        fpc=fpc,
        alpha=alpha,
    )

    # Map estimates to group levels
    est_by_level = {}
    for est in est_list:
        est_by_level[est.by_level] = est

    group_means = []
    group_ses = []
    for level in levels:
        est = est_by_level.get(level) or est_by_level.get(str(level))
        if est is None:
            # Fallback
            est = est_list[len(group_means)]
        group_means.append(float(est.est))
        group_ses.append(float(est.se))

    # 3. Wald test using covariance matrix
    # The covariance matrix from _estimate_taylor is k×k for k groups.
    # We need to test H0: all group means are equal.
    # This is equivalent to testing contrasts: mu_j - mu_0 = 0 for j=1..k-1
    #
    # Contrast matrix C (k-1 × k): each row is [-1, 0, ..., 1, ..., 0]
    cov_mat = np.asarray(cov_mat)

    # Build contrast: differences from first group
    C = np.zeros((ndf, k))
    for j in range(ndf):
        C[j, 0] = -1.0
        C[j, j + 1] = 1.0

    # Contrast estimates and variance
    means_vec = np.array(group_means)
    contrast_est = C @ means_vec  # (k-1,) differences from reference
    contrast_cov = C @ cov_mat @ C.T  # (k-1, k-1) covariance of contrasts

    # Wald chi-square: contrast' @ V^{-1} @ contrast
    try:
        chisq = float(contrast_est @ np.linalg.solve(contrast_cov, contrast_est))
    except np.linalg.LinAlgError:
        chisq = float("nan")

    # 4. Degrees of freedom
    design_df = _get_subpop_df(stratum, psu)
    ddf = design_df - ndf
    ddf = max(ddf, 1)

    # F-statistic and p-value
    f_stat = chisq / ndf
    p_value = float(sp_stats.f.sf(f_stat, dfn=ndf, dfd=ddf))

    return {
        "ndf": ndf,
        "ddf": float(ddf),
        "chisq": chisq,
        "f_stat": float(f_stat),
        "p_value": p_value,
        "levels": list(levels),
        "group_means": group_means,
        "group_ses": group_ses,
    }
