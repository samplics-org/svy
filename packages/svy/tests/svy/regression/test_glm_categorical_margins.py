# tests/svy/regression/test_glm_categorical_margins.py
"""
Marginal effects for categorical predictors, and exponentiated coefficients.

A categorical predictor's marginal effect is a discrete contrast, not a
derivative: for each non-reference level k, averaged over the sample,

    AME_k = mean_w[ mu(x, var=k) - mu(x, var=ref) ]

References from R survey 4.5 + marginaleffects 0.32.0:

    d$y <- as.integer(d$api00 > 600)
    des <- svydesign(ids = ~1, weights = ~pw, data = d)
    f <- svyglm(y ~ ell + factor(stype), design = des,
                family = quasibinomial(), epsilon = 1e-12, maxit = 100)
    avg_slopes(f, wts = weights(f))
    exp(coef(f)); exp(confint(f, ddf = df.residual(f)))
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from svy.core.sample import Design, Sample
from svy.core.terms import Cat


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"

RTOL = 1e-6


@pytest.fixture
def api_binary():
    return pl.read_csv(DATA_DIR / "apistrat.csv").with_columns(
        (pl.col("api00") > 600).cast(pl.Int32).alias("y")
    )


@pytest.fixture
def model(api_binary):
    return Sample(api_binary, Design(wgt="pw")).glm.fit(
        y="y", x=["ell", Cat("stype")], family="binomial", tol=1e-12
    )


# R avg_slopes(): term / contrast / estimate / std.error
R_AME = {
    "ell": ("dY/dX", -0.01089883876, 0.001049377552),
    "H - E": ("H - E", -0.2367438424, 0.06016230516),
    "M - E": ("M - E", -0.1172798808, 0.0600559736),
}


class TestCategoricalAME:
    def test_default_margins_include_the_categorical(self, model):
        """It used to return only the continuous term, dropping stype silently."""
        terms = {m.term for m in model.margins()}
        assert terms == {"ell", "stype"}

    def test_contrasts_match_r(self, model):
        cat = next(m for m in model.margins() if m.term == "stype")

        assert list(cat.values) == ["H - E", "M - E"]
        np.testing.assert_allclose(
            cat.margin, [R_AME["H - E"][1], R_AME["M - E"][1]], rtol=RTOL
        )
        np.testing.assert_allclose(cat.se, [R_AME["H - E"][2], R_AME["M - E"][2]], rtol=1e-5)

    def test_continuous_term_is_unchanged(self, model):
        cont = next(m for m in model.margins() if m.term == "ell")

        np.testing.assert_allclose(cont.margin[0], R_AME["ell"][1], rtol=RTOL)
        np.testing.assert_allclose(cont.se[0], R_AME["ell"][2], rtol=1e-5)

    def test_explicit_request_returns_the_contrast(self, model):
        """`margins(variables=["stype"])` used to answer with an empty list."""
        res = model.margins(variables=["stype"])

        assert len(res) == 1
        assert res[0].term == "stype"
        assert len(res[0].margin) == 2

    def test_ci_brackets_the_estimate(self, model):
        cat = next(m for m in model.margins() if m.term == "stype")

        assert np.all(cat.lci < cat.margin)
        assert np.all(cat.margin < cat.uci)

    def test_exports_carry_the_contrast_labels(self, model):
        cat = next(m for m in model.margins() if m.term == "stype")

        assert cat.to_polars()["value"].to_list() == ["H - E", "M - E"]
        assert cat.to_dict()["values"] == ["H - E", "M - E"]

    def test_renders_without_assuming_numeric_values(self, model):
        cat = next(m for m in model.margins() if m.term == "stype")

        assert "H - E" in str(cat)
        assert "H - E" in cat.__plain_str__()

    def test_unknown_variable_still_raises(self, model):
        with pytest.raises(ValueError, match="not found in model"):
            model.margins(variables=["nope"])


class TestExponentiatedCoefficients:
    # R: exp(coef(f)) and exp(confint(f, ddf = df.residual(f)))
    OR = np.array([20.56123868, 0.926003511, 0.2126402339, 0.4350036793])
    LCI = np.array([8.249698613, 0.9017336375, 0.08955822689, 0.1871156328])
    UCI = np.array([51.24605831, 0.9509266003, 0.5048767783, 1.011290175])

    def test_odds_ratios_match_r(self, model):
        out = model.fitted.to_polars(exponentiate=True)

        np.testing.assert_allclose(out["odds_ratio"].to_numpy(), self.OR, rtol=RTOL)
        np.testing.assert_allclose(out["conf_low"].to_numpy(), self.LCI, rtol=RTOL)
        np.testing.assert_allclose(out["conf_high"].to_numpy(), self.UCI, rtol=RTOL)

    def test_interval_is_exp_of_the_link_scale_bounds(self, model):
        """Not est +/- t*se on the ratio scale — the interval is asymmetric."""
        plain = model.fitted.to_polars()
        out = model.fitted.to_polars(exponentiate=True)

        np.testing.assert_allclose(
            out["conf_low"].to_numpy(), np.exp(plain["conf_low"].to_numpy()), rtol=0, atol=1e-12
        )
        ratio = out["odds_ratio"].to_numpy()
        assert not np.allclose(ratio - out["conf_low"], out["conf_high"] - ratio)

    def test_inference_columns_stay_on_the_link_scale(self, model):
        plain = model.fitted.to_polars()
        out = model.fitted.to_polars(exponentiate=True)

        for col in ("std_err", "statistic", "p_value", "df"):
            np.testing.assert_array_equal(out[col].to_numpy(), plain[col].to_numpy())

    def test_column_is_named_for_the_link(self, api_binary):
        sample = Sample(api_binary, Design(wgt="pw"))

        logit = sample.glm.fit(y="y", x=["ell"], family="binomial")
        assert "odds_ratio" in logit.fitted.to_polars(exponentiate=True).columns

        pois = sample.glm.fit(y="enroll", x=["ell"], family="poisson")
        assert "rate_ratio" in pois.fitted.to_polars(exponentiate=True).columns

        cll = sample.glm.fit(y="y", x=["ell"], family="binomial", link="cloglog")
        assert "hazard_ratio" in cll.fitted.to_polars(exponentiate=True).columns

    @pytest.mark.parametrize(
        ("family", "link", "y"),
        [
            ("gaussian", "identity", "api00"),
            ("binomial", "probit", "y"),
            ("gamma", "inverse", "api00"),
        ],
    )
    def test_refuses_links_where_exp_is_not_a_ratio(self, api_binary, family, link, y):
        fit = Sample(api_binary, Design(wgt="pw")).glm.fit(
            y=y, x=["ell"], family=family, link=link
        )

        with pytest.raises(ValueError, match="not meaningful"):
            fit.fitted.to_polars(exponentiate=True)
        with pytest.raises(ValueError, match="not meaningful"):
            fit.fitted.show(exponentiate=True)

    def test_show_renders_the_ratio_header(self, model):
        text = model.fitted.__plain_str__(exponentiate=True)

        assert "Odds ratio" in text
        assert "link scale" in text
        assert "20.56124" in text

    def test_default_is_unchanged(self, model):
        assert "estimate" in model.fitted.to_polars().columns
        assert "Coef." in model.fitted.__plain_str__()
