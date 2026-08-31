# tests/svy/regression/test_glm_offset.py
"""
GLM offset tests against R's svyglm.

An offset is a known term on the link scale with its coefficient fixed at 1.
The canonical use is a rate model — Poisson counts with log-exposure — so the
fixture is `api.stu ~ ell + meals` with `offset = log(enroll)`.

Reference values from R survey 4.5:

    d$logenroll <- log(d$enroll)
    f <- svyglm(api.stu ~ ell + meals + offset(logenroll), design = des,
                family = quasipoisson(), epsilon = 1e-12, maxit = 100)

One deliberate divergence, covered by `TestOffsetPrediction`. R's
`predict.svyglm(newdata = )` silently drops the offset — it returns exp(X b)
where `fitted(f)` on the same rows returns exp(X b + offset) — so R's own two
answers disagree and only `fitted()` matches the observed counts. svy includes
the offset, which is what `fitted()` and Stata's `predict` after
`glm, exposure()` both do. R's `predict` is therefore not usable as the oracle
here, and `fitted()` is used instead.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from svy.core.sample import Design, Sample


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"

RTOL = 1e-3
ATOL = 1e-5

# Both sides run to 1e-12 so the comparison measures the offset, not each
# side's default IRLS stopping rule.
TOL_TIGHT = 1e-12

X_COLS = ["ell", "meals"]
TERMS = ["_intercept_", "ell", "meals"]

BETA_R = np.array([-0.19922092185311, -0.0011793776520303, 0.0010089870779191])
DEVIANCE_R = 1554.7886172942
NULL_DEVIANCE_R = 1591.6101769894


@pytest.fixture
def api_rate():
    return (
        pl.read_csv(DATA_DIR / "apistrat.csv")
        .with_columns(pl.col("enroll").log().alias("logenroll"))
        .rename({"api.stu": "api_stu"})
    )


def _fit(df, design, **kw):
    return Sample(df, design).glm.fit(
        y="api_stu",
        x=X_COLS,
        family="poisson",
        offset="logenroll",
        tol=TOL_TIGHT,
        **kw,
    )


class TestOffsetAgainstRSurvey:
    """Coefficients and design-based SEs under an offset."""

    def test_weights_only(self, api_rate):
        """R: svydesign(ids = ~1, weights = ~pw, data = apistrat)."""
        res = _fit(api_rate, Design(wgt="pw"))
        coefs = {c.term: c for c in res.coefs}

        se_r = np.array([0.021349567083696, 0.00062045949098397, 0.00050999460827149])
        lci_r = np.array([-0.24132395547912, -0.0024029728055087, 3.2374036067931e-06])
        uci_r = np.array([-0.15711788822711, 4.4217501448044e-05, 0.0020147367522314])
        t_r = np.array([-9.3313799325352, -1.9008132991245, 1.9784269510982])
        p_r = np.array([2.2281480824178e-17, 0.058786418369338, 0.049274199476047])

        np.testing.assert_allclose([coefs[t].est for t in TERMS], BETA_R, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose([coefs[t].se for t in TERMS], se_r, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose([coefs[t].lci for t in TERMS], lci_r, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose([coefs[t].uci for t in TERMS], uci_r, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            [coefs[t].wald.value for t in TERMS], t_r, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            [coefs[t].wald.p_value for t in TERMS], p_r, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(res.stats.wald.value, 2.1412224020381, rtol=RTOL)
        assert coefs["ell"].wald.df == 197

    def test_psu_stratified(self, api_rate):
        """R: svydesign(ids = ~dnum, strata = ~stype, weights = ~pw, nest = TRUE)."""
        res = _fit(api_rate, Design(wgt="pw", stratum="stype", psu="dnum"))
        coefs = {c.term: c for c in res.coefs}

        se_r = np.array([0.021953429286621, 0.00057625199072815, 0.00049926833710728])

        np.testing.assert_allclose([coefs[t].est for t in TERMS], BETA_R, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose([coefs[t].se for t in TERMS], se_r, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(res.stats.wald.value, 2.4362972912435, rtol=RTOL)
        assert coefs["ell"].wald.df == 157

    def test_deviance_and_null_deviance(self, api_rate):
        """
        Null deviance is the intercept-only fit *carrying the offset*, not the
        weighted mean of y — the shortcut that holds without an offset.
        """
        res = _fit(api_rate, Design(wgt="pw"))

        np.testing.assert_allclose(res.stats.deviance, DEVIANCE_R, rtol=1e-9)
        # null deviance reaches the surface through pseudo-R^2
        null_dev = res.stats.deviance / (1.0 - res.stats.r_squared)
        np.testing.assert_allclose(null_dev, NULL_DEVIANCE_R, rtol=1e-9)


class TestOffsetPrediction:
    """Predictions and marginal effects must carry the offset."""

    # R: fitted(f)[1:5] — equivalently exp(X b + offset), and the values that
    # track the observed counts (241, 631, 415, 288, 319).
    FITTED_R = np.array([227.0095475, 694.6717227, 375.1303031, 246.4890819, 291.855914])

    # R: SE(predict(f, newdata = head(d, 5), type = "response", se.fit = TRUE)).
    # These are SEs of exp(X b) because R drops the offset; the offset enters
    # mu as a constant factor exp(offset) and se = |dmu/deta| * se(eta), so the
    # correct SE is exactly this scaled by exp(offset).
    SE_R_WITHOUT_OFFSET = np.array(
        [0.01172672008, 0.01829263171, 0.009105014092, 0.01437928773, 0.01205872306]
    )

    def test_predict_matches_r_fitted(self, api_rate):
        res = _fit(api_rate, Design(wgt="pw"))
        pred = res.predict(api_rate.head(5))

        np.testing.assert_allclose(pred.yhat, self.FITTED_R, rtol=1e-6)

    def test_predict_se_scales_by_exp_offset(self, api_rate):
        res = _fit(api_rate, Design(wgt="pw"))
        pred = res.predict(api_rate.head(5))

        expected = self.SE_R_WITHOUT_OFFSET * np.exp(
            api_rate.head(5).get_column("logenroll").to_numpy()
        )
        np.testing.assert_allclose(pred.se, expected, rtol=1e-6)

    def test_ame_uses_offset_adjusted_mu(self, api_rate):
        """
        For a log link AME_j = mean_w(mu) * beta_j, and mu must include the
        offset. R: mu <- fitted(f); sum(w * mu) / sum(w) * coef(f)[j].
        """
        res = _fit(api_rate, Design(wgt="pw"))
        ame = {m.term: m.margin[0] for m in res.margins()}

        np.testing.assert_allclose(ame["ell"], -0.587596001912, rtol=1e-6)
        np.testing.assert_allclose(ame["meals"], 0.502703075597, rtol=1e-6)

    def test_predict_requires_the_offset_column(self, api_rate):
        res = _fit(api_rate, Design(wgt="pw"))
        with pytest.raises(ValueError, match="logenroll"):
            res.predict(api_rate.head(5).drop("logenroll"))


class TestOffsetInvariants:
    def test_zero_offset_reproduces_the_plain_fit(self, api_rate):
        """An all-zero offset is the no-offset path, to the last bit."""
        df = api_rate.with_columns(pl.lit(0.0).alias("zero"))
        design = Design(wgt="pw")

        plain = Sample(df, design).glm.fit(
            y="api_stu", x=X_COLS, family="poisson", tol=TOL_TIGHT
        )
        zeroed = Sample(df, design).glm.fit(
            y="api_stu", x=X_COLS, family="poisson", offset="zero", tol=TOL_TIGHT
        )

        np.testing.assert_allclose(
            [c.est for c in zeroed.coefs], [c.est for c in plain.coefs], rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            [c.se for c in zeroed.coefs], [c.se for c in plain.coefs], rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            zeroed.stats.deviance, plain.stats.deviance, rtol=0, atol=1e-9
        )

    def test_offset_is_not_a_fitted_term(self, api_rate):
        res = _fit(api_rate, Design(wgt="pw"))

        assert [c.term for c in res.coefs] == TERMS
        assert res.fitted.offset == "logenroll"

    def test_unknown_column_raises(self, api_rate):
        with pytest.raises(ValueError, match="not found"):
            Sample(api_rate, Design(wgt="pw")).glm.fit(
                y="api_stu", x=X_COLS, family="poisson", offset="nope"
            )

    def test_non_string_offset_raises(self, api_rate):
        with pytest.raises(TypeError, match="column name"):
            Sample(api_rate, Design(wgt="pw")).glm.fit(
                y="api_stu", x=X_COLS, family="poisson", offset=3.0
            )
