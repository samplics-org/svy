# tests/svy/regression/test_glm_canonical_sandwich.py
"""
Correctness tests for svy.glm.fit against an independent NumPy
implementation of the canonical Taylor-linearization sandwich variance.

This is the *correctness* test for GLM sandwich variance. It asserts
svy implements the canonical formula at near-machine precision. If this
test breaks, svy's GLM math is wrong.

Comparison against R's svyglm and Stata's svy: logit are kept in
separate files (`test_glm_against_R_outputs.py`, `test_glm_against_stata_outputs.py`)
because those implementations disagree with the canonical formula at
~1e-5 (R) and ~1e-8 (Stata) on small samples. Those files serve as
documentation of how close svy lands relative to other software, not as
correctness tests.

See docs/notes/glm-precision.md for the full investigation.
"""

from pathlib import Path

import numpy as np
import polars as pl

from svy.core.enumerations import DistFamily, LinkFunction
from svy.core.sample import Design, Sample


DATA_DIR = Path(__file__).parent.parent.parent / "test_data"

# Strict tolerance: svy should match the canonical NumPy reconstruction
# at near-machine precision, since both implement the same arithmetic on
# the same inputs.
RTOL = 1e-7
ATOL = 1e-7


# ---------------------------------------------------------------------------
# Canonical sandwich reconstruction
# ---------------------------------------------------------------------------


def canonical_sandwich(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    beta: np.ndarray,
    family: str,
    link: str,
    stratum: np.ndarray | None = None,
    psu: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the design-based sandwich variance for a GLM from scratch.

    Implements the canonical Taylor-linearization formula:
        bread = (X' diag(w * mu_eta) X)^{-1}
        score_i = X_i * (y_i - mu_i) * w_i
        infl_i = score_i @ bread
        vcov = sum over strata of (n_h / (n_h - 1)) * Σ (infl_h - mean)(infl_h - mean)'

    For SRS (no strata, no PSU), each row is its own PSU and the design
    df correction is n/(n-1). For PSU-only, sums of influence functions
    are aggregated by PSU. For strata-only, every row is its own PSU
    within each stratum. For PSU + strata, both apply.

    Weights are normalized to sum to n (matching svy's convention; R's
    rescale=TRUE is equivalent for non-Gaussian families since the
    sandwich is invariant to constant rescale).
    """
    n, k = X.shape

    # Normalize weights to sum to n (svy's convention)
    w = w * (n / w.sum())

    # Compute mu, mu_eta from beta
    eta = X @ beta
    if link == "identity":
        mu = eta
        mu_eta = np.ones_like(eta)
    elif link == "logit":
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu_eta = mu * (1.0 - mu)
    elif link == "log":
        mu = np.exp(eta)
        mu_eta = mu
    else:
        raise ValueError(f"unsupported link: {link}")

    # Bread: (X' diag(w * mu_eta) X)^-1
    w_irls = w * mu_eta
    bread = np.linalg.inv(X.T @ (w_irls[:, None] * X))

    # Per-row weighted score
    score = X * ((y - mu) * w)[:, None]  # (n, k)
    infl = score @ bread  # (n, k)

    # Sum across strata
    if stratum is None:
        stratum_ids = np.zeros(n, dtype=np.int64)
    else:
        stratum_ids = np.asarray(stratum)
    if psu is None:
        psu_ids = np.arange(n, dtype=np.int64)  # each row is its own PSU
    else:
        psu_ids = np.asarray(psu)

    vcov = np.zeros((k, k))
    for h in np.unique(stratum_ids):
        in_stratum = stratum_ids == h
        # Aggregate influence functions by PSU within this stratum
        psu_in_h = psu_ids[in_stratum]
        infl_in_h = infl[in_stratum]
        unique_psus = np.unique(psu_in_h)
        psu_sums = np.zeros((len(unique_psus), k))
        for j, p in enumerate(unique_psus):
            psu_sums[j] = infl_in_h[psu_in_h == p].sum(axis=0)
        n_h = len(unique_psus)
        if n_h <= 1:
            continue
        centered = psu_sums - psu_sums.mean(axis=0)
        vcov += centered.T @ centered * (n_h / (n_h - 1))

    return vcov


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


api_strat = pl.read_csv(DATA_DIR / "apistrat.csv")
api_strat = api_strat.with_columns((pl.col("api00") > 743).cast(pl.Float64).alias("y_bin"))

X_COLS = ["ell", "meals", "mobility"]
TERMS = ["_intercept_", "ell", "meals", "mobility"]


def make_sample(df, *, weight=None, stratum=None, psu=None):
    if weight is None and stratum is None and psu is None:
        design = None
    else:
        design = Design(wgt=weight, stratum=stratum, psu=psu)
    return Sample(data=df, design=design)


def _design_matrix():
    return np.column_stack(
        [
            np.ones(api_strat.height),
            api_strat["ell"].to_numpy(),
            api_strat["meals"].to_numpy(),
            api_strat["mobility"].to_numpy(),
        ]
    )


def _assert_sandwich_matches(res, X, y, w, family, link, stratum=None, psu=None):
    """Verify svy's reported SEs match the canonical NumPy sandwich at RTOL."""
    beta = np.array([c.est for c in sorted(res.coefs, key=lambda c: TERMS.index(c.term))])
    se_svy = np.array([c.se for c in sorted(res.coefs, key=lambda c: TERMS.index(c.term))])

    vcov_canonical = canonical_sandwich(
        X,
        y,
        w,
        beta,
        family=family,
        link=link,
        stratum=stratum,
        psu=psu,
    )
    se_canonical = np.sqrt(np.diag(vcov_canonical))

    np.testing.assert_allclose(se_svy, se_canonical, rtol=RTOL, atol=ATOL)


# ===========================================================================
# Linear / Gaussian
# ===========================================================================


class TestLinearGaussian:
    Y = "api00"
    FAMILY = DistFamily.GAUSSIAN
    LINK = LinkFunction.IDENTITY

    def test_weights_only(self):
        sample = make_sample(api_strat, weight="pw")
        res = sample.glm.fit(y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK)

        X = _design_matrix()
        y = api_strat[self.Y].to_numpy()
        w = api_strat["pw"].to_numpy()
        _assert_sandwich_matches(res, X, y, w, family="gaussian", link="identity")


# ===========================================================================
# Logistic / Binomial
# ===========================================================================


class TestLogisticBinomial:
    Y = "y_bin"
    FAMILY = DistFamily.BINOMIAL
    LINK = LinkFunction.LOGIT

    def test_weights_only(self):
        sample = make_sample(api_strat, weight="pw")
        res = sample.glm.fit(y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK)

        X = _design_matrix()
        y = api_strat[self.Y].to_numpy()
        w = api_strat["pw"].to_numpy()
        _assert_sandwich_matches(res, X, y, w, family="binomial", link="logit")

    def test_psu_only(self):
        sample = make_sample(api_strat, weight="pw", psu="dnum")
        res = sample.glm.fit(y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK)

        X = _design_matrix()
        y = api_strat[self.Y].to_numpy()
        w = api_strat["pw"].to_numpy()
        psu = api_strat["dnum"].to_numpy()
        _assert_sandwich_matches(res, X, y, w, family="binomial", link="logit", psu=psu)

    def test_stratified(self):
        sample = make_sample(api_strat, weight="pw", stratum="stype")
        res = sample.glm.fit(y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK)

        X = _design_matrix()
        y = api_strat[self.Y].to_numpy()
        w = api_strat["pw"].to_numpy()
        stratum = api_strat["stype"].to_numpy()
        _assert_sandwich_matches(
            res,
            X,
            y,
            w,
            family="binomial",
            link="logit",
            stratum=stratum,
        )

    def test_psu_stratified(self):
        sample = make_sample(api_strat, weight="pw", stratum="stype", psu="dnum")
        res = sample.glm.fit(y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK)

        X = _design_matrix()
        y = api_strat[self.Y].to_numpy()
        w = api_strat["pw"].to_numpy()
        stratum = api_strat["stype"].to_numpy()
        psu = api_strat["dnum"].to_numpy()
        _assert_sandwich_matches(
            res,
            X,
            y,
            w,
            family="binomial",
            link="logit",
            stratum=stratum,
            psu=psu,
        )
