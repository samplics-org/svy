# tests/svy/regression/test_glm_dispersion.py
"""
`stats.scale` is the Pearson dispersion, for every family.

    phi = sum_i w_i (y_i - mu_i)^2 / V(mu_i) / (n_obs - k)

on the fit's own weight scale — R `glm()`'s and Stata `glm`'s "(1/df) Pearson".
Two deliberate departures from R `summary.svyglm`, both documented in the
changelog:

  * `summary.svyglm` reports `svyvar(resid(pearson))`, a design-weighted
    *variance* of the Pearson residuals (survey.R:1264). That is not the
    textbook estimator and not what someone comparing against `glm()` or Stata
    expects, so it is not copied.
  * Binomial and Poisson used to hard-code 1.0. Reporting the estimate is the
    overdispersion diagnostic a count model needs — the negative-binomial
    question in one number. Design-based SEs never use it.

Reference (R, apistrat, weights-only):

    g <- glm(y ~ ell + meals + mobility, data = d, weights = pw / mean(pw),
             family = quasi*, control = glm.control(epsilon = 1e-12))
    sum(w * resid(g, "response")^2 / family$variance(fitted(g))) / g$df.residual
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.sample import Design, Sample


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"

X_COLS = ["ell", "meals", "mobility"]

# family, link, response, phi at n - k = 196
R_PEARSON = [
    ("gaussian", "identity", "api00", 5251.1287319858),
    ("binomial", "logit", "y_bin", 0.70335496621545),
    ("binomial", "cauchit", "y_bin", 0.85230514997122),
    ("poisson", "log", "enroll", 333.94415162862),
    ("poisson", "sqrt", "enroll", 333.79163085318),
]


@pytest.fixture
def api():
    return pl.read_csv(DATA_DIR / "apistrat.csv").with_columns(
        (pl.col("api00") > 743).cast(pl.Int32).alias("y_bin")
    )


@pytest.mark.parametrize(("family", "link", "y", "phi"), R_PEARSON)
def test_matches_r_pearson_over_n_minus_k(api, family, link, y, phi):
    res = Sample(api, Design(wgt="pw")).glm.fit(
        y=y, x=X_COLS, family=family, link=link, tol=1e-12, max_iter=100
    )

    np.testing.assert_allclose(res.stats.scale, phi, rtol=1e-9)


class TestDivisor:
    """n_obs - k, not the design residual df."""

    def test_the_divisor_is_n_minus_k(self, api):
        res = Sample(api, Design(wgt="pw")).glm.fit(y="api00", x=X_COLS, tol=1e-12, max_iter=100)

        assert res.stats.n == 200
        # 5251.1287... x (200 - 4) recovers the raw Pearson sum.
        np.testing.assert_allclose(res.stats.scale * 196, 5251.1287319858 * 196, rtol=1e-9)

    def test_a_clustered_design_reports_the_same_dispersion(self, api):
        """
        The dispersion is a moment estimate; the design df belongs to the t
        reference distribution. It used to be divided by the design residual
        df, so the same model on the same rows reported a different scale for
        every design.
        """
        weights_only = Sample(api, Design(wgt="pw")).glm.fit(
            y="api00", x=X_COLS, tol=1e-12, max_iter=100
        )
        clustered = Sample(api, Design(wgt="pw", stratum="stype", psu="dnum")).glm.fit(
            y="api00", x=X_COLS, tol=1e-12, max_iter=100
        )

        np.testing.assert_allclose(clustered.stats.scale, weights_only.stats.scale, rtol=1e-12)


class TestEveryFamilyReportsIt:
    def test_binomial_and_poisson_are_no_longer_pinned_at_one(self, api):
        sample = Sample(api, Design(wgt="pw"))

        binom = sample.glm.fit(y="y_bin", x=X_COLS, family="binomial", tol=1e-12)
        pois = sample.glm.fit(y="enroll", x=X_COLS, family="poisson", tol=1e-12)

        assert binom.stats.scale != 1.0
        assert pois.stats.scale != 1.0

    def test_poisson_overdispersion_shows_up(self, api):
        """The point of reporting it: enroll is far from equidispersed."""
        res = Sample(api, Design(wgt="pw")).glm.fit(
            y="enroll", x=X_COLS, family="poisson", tol=1e-12
        )

        assert res.stats.scale > 100

    def test_a_domain_fit_uses_its_own_rows(self, api):
        """phi is over the in-domain rows, on the in-domain weight scale."""
        domain = Sample(api, Design(wgt="pw")).glm.fit(
            y="api00", x=X_COLS, where=svy.col("stype") == "E", tol=1e-12
        )
        filtered = Sample(api.filter(pl.col("stype") == "E"), Design(wgt="pw")).glm.fit(
            y="api00", x=X_COLS, tol=1e-12
        )

        np.testing.assert_allclose(domain.stats.scale, filtered.stats.scale, rtol=1e-9)
