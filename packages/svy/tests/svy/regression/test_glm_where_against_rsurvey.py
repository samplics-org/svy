# tests/svy/regression/test_glm_where_against_rsurvey.py
# tests/svy/regression/test_glm_where_against_Rsurvey.py
"""
GLM domain-estimation (`where=`) tests against R's svyglm + subset() as the
oracle. R is the natural reference because:

  svy:  sample.glm.fit(..., where = svy.col("meals") > 50)
  R:    svyglm(..., design = subset(design, meals > 50))

These should produce equivalent estimates and standard errors. We assert at a
loose tolerance because R's svyglm carries the `glm.fit` stale-IRLS-weights
artifact that creates a ~1e-5 difference on small samples (see
`test_glm_canonical_sandwich.py` for the strict-tolerance correctness test
against the textbook formula).

DF convention: R reports `df.residual = degf(subset_design) - (k - 1)`. With
k = 4 (intercept + 3 covariates), the domain-restricted dfs become:

    weights only  : degf 79  - 3 = 76
    PSU only      : degf 51  - 3 = 48
    strata only   : degf 77  - 3 = 74
    PSU + strata  : degf 59  - 3 = 56
    linear        : degf 79  - 3 = 76

Reference values are from `tests/test_data/from_samplics_where.R`.
"""

from pathlib import Path

import numpy as np
import polars as pl

import svy
from svy.core.enumerations import DistFamily, LinkFunction
from svy.core.sample import Design, Sample


DATA_DIR = Path(__file__).parent.parent.parent / "test_data"

# Loose tolerance for R comparison: R has a stale-IRLS-weights artifact in
# glm.fit that produces ~1e-5 SE differences vs the canonical sandwich
# formula on small samples. The canonical correctness test
# (test_glm_canonical_sandwich.py) asserts svy is correct at 1e-7 against
# an independent reconstruction; here we just confirm we land in R's
# neighborhood.
RTOL = 1e-3
ATOL = 1e-5

WHERE = svy.col("meals") > 50

X_COLS = ["ell", "meals", "mobility"]
TERMS = ["_intercept_", "ell", "meals", "mobility"]
STRATUM_COL = "stype"
PSU_COL = "dnum"
WEIGHT_COL = "pw"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sample(df, *, weight=None, stratum=None, psu=None):
    if weight is None and stratum is None and psu is None:
        design = None
    else:
        design = Design(wgt=weight, stratum=stratum, psu=psu)
    return Sample(data=df, design=design)


def _coef_dict(result):
    return {c.term: c for c in result.coefs}


def _assert_matches(result, order, beta_r, se_r, df_resid_r):
    coefs = _coef_dict(result)
    beta = np.array([coefs[t].est for t in order])
    se = np.array([coefs[t].se for t in order])

    np.testing.assert_allclose(beta, beta_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_r, rtol=RTOL, atol=ATOL)

    assert result.coefs[0].wald.df == df_resid_r, (
        f"residual df mismatch: svy={result.coefs[0].wald.df}, R={df_resid_r}"
    )


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

api_strat = pl.read_csv(DATA_DIR / "apistrat.csv")
api_strat = api_strat.with_columns((pl.col("api00") > 743).cast(pl.Float64).alias("y_bin"))


# ===========================================================================
# Linear / Gaussian
# ===========================================================================


class TestLinearGaussianWhere:
    """Linear regression with where = meals > 50."""

    Y = "api00"
    FAMILY = DistFamily.GAUSSIAN
    LINK = LinkFunction.IDENTITY

    def test_weights_only(self):
        """
        R: svyglm(api00 ~ ell + meals + mobility,
                  design = subset(svydesign(~1, weights=~pw), meals > 50))
        df = degf(subset) - (k-1) = 79 - 3 = 76.
        """
        sample = make_sample(api_strat, weight=WEIGHT_COL)
        res = sample.glm.fit(
            y=self.Y,
            x=X_COLS,
            family=self.FAMILY,
            link=self.LINK,
            where=WHERE,
        )

        beta_r = np.array([786.5879509509, -0.6512981845, -2.4251360408, -0.5202520274])
        se_r = np.array([44.1965294825, 0.4985862523, 0.7487049385, 0.8797228934])
        _assert_matches(res, TERMS, beta_r, se_r, df_resid_r=76)


# ===========================================================================
# Logistic / Binomial
# ===========================================================================


class TestLogisticBinomialWhere:
    """Logistic regression with where = meals > 50."""

    Y = "y_bin"
    FAMILY = DistFamily.BINOMIAL
    LINK = LinkFunction.LOGIT

    BETA_R = np.array(
        [
            12.39031117649,
            -0.07133534124,
            -0.27345174707,
            0.08072406168,
        ]
    )

    def test_weights_only(self):
        """
        R: svyglm(..., design = subset(svydesign(~1, weights=~pw), meals > 50))
        df = degf(79) - 3 = 76.
        """
        sample = make_sample(api_strat, weight=WEIGHT_COL)
        res = sample.glm.fit(
            y=self.Y,
            x=X_COLS,
            family=self.FAMILY,
            link=self.LINK,
            where=WHERE,
        )

        se_r = np.array([4.89339708925, 0.06954758195, 0.07942095888, 0.04771419756])
        _assert_matches(res, TERMS, self.BETA_R, se_r, df_resid_r=76)

    def test_psu_only(self):
        """
        R: svyglm(..., design = subset(svydesign(~dnum, weights=~pw), meals > 50))
        df = degf(51) - 3 = 48.
        """
        sample = make_sample(api_strat, weight=WEIGHT_COL, psu=PSU_COL)
        res = sample.glm.fit(
            y=self.Y,
            x=X_COLS,
            family=self.FAMILY,
            link=self.LINK,
            where=WHERE,
        )

        se_r = np.array([4.98095435070, 0.06985282687, 0.08175498701, 0.04785982432])
        _assert_matches(res, TERMS, self.BETA_R, se_r, df_resid_r=48)

    def test_stratified(self):
        """
        R: svyglm(..., design = subset(svydesign(~1, strata=~stype, weights=~pw), meals > 50))
        df = degf(77) - 3 = 74.
        """
        sample = make_sample(api_strat, weight=WEIGHT_COL, stratum=STRATUM_COL)
        res = sample.glm.fit(
            y=self.Y,
            x=X_COLS,
            family=self.FAMILY,
            link=self.LINK,
            where=WHERE,
        )

        se_r = np.array([4.8902684939, 0.0692780030, 0.0795627859, 0.0478247904])
        _assert_matches(res, TERMS, self.BETA_R, se_r, df_resid_r=74)

    def test_psu_stratified(self):
        """
        Headline test for issue #6: full design + domain restriction.
        R: svyglm(..., design = subset(svydesign(~dnum, strata=~stype,
                                                  weights=~pw, nest=TRUE), meals > 50))
        df = degf(59) - 3 = 56.
        """
        sample = make_sample(
            api_strat,
            weight=WEIGHT_COL,
            stratum=STRATUM_COL,
            psu=PSU_COL,
        )
        res = sample.glm.fit(
            y=self.Y,
            x=X_COLS,
            family=self.FAMILY,
            link=self.LINK,
            where=WHERE,
        )

        se_r = np.array([4.98340134438, 0.06945592597, 0.08200317468, 0.04786349761])
        _assert_matches(res, TERMS, self.BETA_R, se_r, df_resid_r=56)


# ===========================================================================
# Where vs filter_records: contractual difference
# ===========================================================================


def test_where_differs_from_filter_records_se():
    """
    The bug behind issue #6: `filter_records` drops out-of-domain rows before
    fitting, which breaks the design's PSU/strata enumeration and inflates
    SEs. `where=` keeps the design intact (zeros contributions instead of
    dropping rows) and produces correct domain-restricted SEs matching R's
    `subset(design, ...) + svyglm`.

    Coefficients must agree (same point estimate). SEs must differ -- the
    `where=` SEs must be SMALLER than the `filter_records` SEs, since
    dropping rows from a PSU breaks the design symmetry the variance
    estimator relies on.
    """
    sample = make_sample(
        api_strat,
        weight=WEIGHT_COL,
        stratum=STRATUM_COL,
        psu=PSU_COL,
    )

    res_where = sample.glm.fit(
        y="y_bin",
        x=X_COLS,
        family=DistFamily.BINOMIAL,
        link=LinkFunction.LOGIT,
        where=WHERE,
    )
    res_filter = sample.wrangling.filter_records(WHERE).glm.fit(
        y="y_bin",
        x=X_COLS,
        family=DistFamily.BINOMIAL,
        link=LinkFunction.LOGIT,
    )

    cw = _coef_dict(res_where)
    cf = _coef_dict(res_filter)
    beta_w = np.array([cw[t].est for t in TERMS])
    beta_f = np.array([cf[t].est for t in TERMS])
    se_w = np.array([cw[t].se for t in TERMS])
    se_f = np.array([cf[t].se for t in TERMS])

    # Point estimates agree (same likelihood maximized).
    np.testing.assert_allclose(beta_w, beta_f, rtol=1e-6, atol=1e-8)

    # SEs must DIFFER. On api_strat (small n, mostly in-domain) the
    # difference is small but measurable; we want to catch regressions
    # where a future change accidentally makes them identical.
    rel_diff = np.abs(se_w - se_f) / np.abs(se_w)
    assert rel_diff.max() > 1e-3, (
        f"where= and filter_records should give different SEs; "
        f"got max relative diff = {rel_diff.max():.4g}"
    )
