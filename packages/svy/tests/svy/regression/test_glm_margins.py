# tests/svy/regression/test_glm_margins.py
"""
Tests for GLM margins against R.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from svy.core.enumerations import DistFamily
from svy.core.sample import Design, Sample


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"


# =============================================================================
# R Reference Values
# =============================================================================
# Generated with R survey 4.5 + marginaleffects 0.32.0 (2026-07-23):
#
#   api <- read.csv("apistrat.csv"); api$y_bin <- as.integer(api$api00 > 600)
#   d1 <- svydesign(ids=~1, weights=~pw, data=api)
#   m1 <- svyglm(y_bin ~ ell + meals + mobility, design=d1, family=quasibinomial())
#   avg_slopes(m1, wts=weights(m1))
#   avg_predictions(m1, variables=list(meals=c(20,50,80)), by="meals", wts=weights(m1))
#
# `wts=weights(fit)` makes marginaleffects average with the design weights
# (Stata `margins` convention). SEs are delta-method over vcov(svyglm),
# i.e. g'V(beta)g with covariates fixed — the convention svy implements.

R_LOGISTIC_AME = {
    "ell": -0.0009982936,
    "meals": -0.0080360454,
    "mobility": 0.0007552178,
}

R_LOGISTIC_AME_SE = {
    "ell": 0.0016047333,
    "meals": 0.0010179779,
    "mobility": 0.0019007082,
}

R_PREDICTIVE_MARGINS = {
    "values": np.array([20, 50, 80]),
    "margin": np.array([0.9698889120, 0.7722732242, 0.2655410705]),
    "se": np.array([0.0158168043, 0.0400183429, 0.0813343039]),
}

# R: interaction model m2 <- svyglm(y_bin ~ ell * meals, design=d1, ...)
R_INTERACTION_AME = {
    "ell": (-0.0014277197, 0.0017420001),
    "meals": (-0.0077599270, 0.0011161544),
}

R_INTERACTION_PREDICTIVE = {
    "values": np.array([20, 50, 80]),
    "margin": np.array([0.9621412052, 0.7593207994, 0.2591576034]),
    "se": np.array([0.0305195844, 0.0513012146, 0.0813251070]),
}

# R: Cat x continuous m3 <- svyglm(y_bin ~ awards * ell, design=d1, ...)
R_CAT_INTERACTION_AME_ELL = (-0.0101530922, 0.0010906246)

# R: stratified design + domain fit
#   d4 <- svydesign(ids=~1, strata=~stype, weights=~pw, data=api)
#   m4 <- svyglm(y_bin ~ ell + meals, design=subset(d4, awards=="Yes"),
#                family=quasibinomial())
R_DOMAIN_AME = {
    "ell": (-0.0002686150, 0.0016696858),
    "meals": (-0.0085233748, 0.0014792988),
}

R_DOMAIN_PREDICTIVE = {
    "values": np.array([20, 50, 80]),
    "margin": np.array([0.9935953493, 0.8893161765, 0.2943536384]),
    "se": np.array([0.0067214884, 0.0402467347, 0.1307085096]),
}

# R: Linear AME (just coefficients)
R_LINEAR_AME = {
    "ell": -0.4805866,
    "meals": -3.1415353,
    "mobility": 0.2257132,
}

RTOL = 1e-2  # 1% tolerance for margins
ATOL = 1e-4

# Margin point estimates should agree with R to numerical precision; SEs to
# ~1e-4 relative (marginaleffects uses a numerical Jacobian, svy an
# analytic gradient).
RTOL_EST = 1e-6
RTOL_SE = 5e-4


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def api_strat():
    return pl.read_csv(DATA_DIR / "apistrat.csv")


@pytest.fixture
def api_binary(api_strat):
    return api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))


# =============================================================================
# AME Tests
# =============================================================================


class TestGLMMarginAME:
    """Tests for Average Marginal Effects."""

    def test_linear_ame_equals_coefficients(self, api_strat):
        """For linear model, AME equals coefficients."""
        sample = Sample(api_strat, Design(wgt="pw"))
        model = sample.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
        )

        margins = model.margins()

        # AME should equal coefficients for linear model
        ame_dict = {m.term: m.margin[0] for m in margins}

        np.testing.assert_allclose(ame_dict["ell"], R_LINEAR_AME["ell"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ame_dict["meals"], R_LINEAR_AME["meals"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            ame_dict["mobility"], R_LINEAR_AME["mobility"], rtol=RTOL, atol=ATOL
        )

    def test_logistic_ame_vs_r(self, api_binary):
        """Logistic AME matches R."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        margins = model.margins()
        ame_dict = {m.term: m.margin[0] for m in margins}

        np.testing.assert_allclose(ame_dict["ell"], R_LOGISTIC_AME["ell"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            ame_dict["meals"], R_LOGISTIC_AME["meals"], rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            ame_dict["mobility"], R_LOGISTIC_AME["mobility"], rtol=RTOL, atol=ATOL
        )

    def test_logistic_ame_se_vs_r(self, api_binary):
        """Logistic AME delta-method SEs match R marginaleffects."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        margins = model.margins()
        se_dict = {m.term: m.se[0] for m in margins}

        for var, se in R_LOGISTIC_AME_SE.items():
            np.testing.assert_allclose(se_dict[var], se, rtol=RTOL_SE)

    def test_ame_returns_list(self, api_strat):
        """AME returns list of GLMMargins."""
        sample = Sample(api_strat, Design(wgt="pw"))
        model = sample.glm.fit(y="api00", x=["ell", "meals"])

        margins = model.margins()

        assert isinstance(margins, list)
        assert len(margins) == 2

        from svy.regression import GLMMargins

        for m in margins:
            assert isinstance(m, GLMMargins)
            assert m.margin_type == "ame"

    def test_ame_specific_variables(self, api_strat):
        """AME for specific variables only."""
        sample = Sample(api_strat, Design(wgt="pw"))
        model = sample.glm.fit(y="api00", x=["ell", "meals", "mobility"])

        margins = model.margins(variables=["meals"])

        assert len(margins) == 1
        assert margins[0].term == "meals"


# =============================================================================
# Predictive Margins Tests
# =============================================================================


class TestGLMMarginPredictive:
    """Tests for Predictive Margins."""

    def test_predictive_margins_vs_r(self, api_binary):
        """Predictive margins match R."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        margins = model.margins(at={"meals": [20, 50, 80]})

        np.testing.assert_allclose(margins.margin, R_PREDICTIVE_MARGINS["margin"], rtol=RTOL_EST)

    def test_predictive_margins_se_vs_r(self, api_binary):
        """Predictive margins SE matches R."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        margins = model.margins(at={"meals": [20, 50, 80]})

        np.testing.assert_allclose(margins.se, R_PREDICTIVE_MARGINS["se"], rtol=RTOL_SE)

    def test_predictive_margins_returns_single(self, api_binary):
        """Single at variable returns single GLMMargins."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        margins = model.margins(at={"meals": [20, 50]})

        from svy.regression import GLMMargins

        assert isinstance(margins, GLMMargins)
        assert margins.margin_type == "predictive"
        assert len(margins) == 2

    def test_predictive_margins_multiple_variables(self, api_binary):
        """Multiple at variables returns list."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        margins = model.margins(at={"meals": [20, 50], "ell": [10, 30]})

        assert isinstance(margins, list)
        assert len(margins) == 2

    def test_predictive_margins_has_values(self, api_binary):
        """Predictive margins include the at values."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        margins = model.margins(at={"meals": [20, 50, 80]})

        np.testing.assert_array_equal(margins.values, np.array([20, 50, 80]))


# =============================================================================
# Interaction, Domain, and Missing-Data Tests
# =============================================================================


class TestGLMMarginInteractions:
    """Margins for models with interaction terms (Round 8: R2/R3)."""

    def test_interaction_ame_vs_r(self, api_binary):
        """AME differentiates the full linear predictor incl. ell:meals."""
        from svy.core.terms import Cross

        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", Cross("ell", "meals")],
            family=DistFamily.BINOMIAL,
        )

        margins = {m.term: m for m in model.margins()}
        for var, (est, se) in R_INTERACTION_AME.items():
            np.testing.assert_allclose(margins[var].margin[0], est, rtol=RTOL_EST)
            np.testing.assert_allclose(margins[var].se[0], se, rtol=RTOL_SE)

    def test_interaction_predictive_margins_vs_r(self, api_binary):
        """Counterfactuals rebuild interaction columns (no stale ell:meals)."""
        from svy.core.terms import Cross

        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", Cross("ell", "meals")],
            family=DistFamily.BINOMIAL,
        )

        margins = model.margins(at={"meals": [20, 50, 80]})
        np.testing.assert_allclose(
            margins.margin, R_INTERACTION_PREDICTIVE["margin"], rtol=RTOL_EST
        )
        np.testing.assert_allclose(margins.se, R_INTERACTION_PREDICTIVE["se"], rtol=RTOL_SE)

    def test_cat_interaction_ame_vs_r(self, api_binary):
        """AME for a continuous variable interacted with a Cat term."""
        from svy.core.terms import Cat, Cross

        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(
            y="y_bin",
            x=[Cat("awards"), "ell", Cross(Cat("awards"), "ell")],
            family=DistFamily.BINOMIAL,
        )

        margins = model.margins(variables=["ell"])
        est, se = R_CAT_INTERACTION_AME_ELL
        np.testing.assert_allclose(margins[0].margin[0], est, rtol=RTOL_EST)
        np.testing.assert_allclose(margins[0].se[0], se, rtol=RTOL_SE)


class TestGLMMarginDomain:
    """Margins after a domain (where=) fit use the fitted rows (Round 8: R1)."""

    def _domain_model(self, api_binary):
        import svy

        sample = Sample(api_binary, Design(stratum="stype", wgt="pw"))
        return sample.glm.fit(
            y="y_bin",
            x=["ell", "meals"],
            family=DistFamily.BINOMIAL,
            where=svy.col("awards") == "Yes",
        )

    def test_domain_ame_vs_r(self, api_binary):
        """AME after where= matches R svyglm on subset(design, ...)."""
        model = self._domain_model(api_binary)

        margins = {m.term: m for m in model.margins()}
        for var, (est, se) in R_DOMAIN_AME.items():
            np.testing.assert_allclose(margins[var].margin[0], est, rtol=RTOL_EST)
            np.testing.assert_allclose(margins[var].se[0], se, rtol=RTOL_SE)

    def test_domain_predictive_margins_vs_r(self, api_binary):
        """Predictive margins after where= average over in-domain rows only."""
        model = self._domain_model(api_binary)

        margins = model.margins(at={"meals": [20, 50, 80]})
        np.testing.assert_allclose(margins.margin, R_DOMAIN_PREDICTIVE["margin"], rtol=RTOL_EST)
        np.testing.assert_allclose(margins.se, R_DOMAIN_PREDICTIVE["se"], rtol=RTOL_SE)


class TestGLMMarginInvariants:
    """Structural invariants of the delta-method margins."""

    def test_identity_link_ame_se_equals_coef_se(self, api_strat):
        """For the identity link, AME == coef and se(AME) == se(coef)."""
        sample = Sample(api_strat, Design(wgt="pw"))
        model = sample.glm.fit(y="api00", x=["ell", "meals", "mobility"])

        coef = {c.term: c for c in model.fitted.coefs}
        for m in model.margins():
            np.testing.assert_allclose(m.margin[0], coef[m.term].est, rtol=1e-12)
            np.testing.assert_allclose(m.se[0], coef[m.term].se, rtol=1e-12)

    def test_null_covariates_do_not_propagate_nan(self, api_binary):
        """Margins average over the fitted rows, so covariate nulls
        (dropped at fit time) cannot produce NaN margins."""
        data = api_binary.with_columns(
            pl.when(pl.int_range(pl.len()) < 5)
            .then(None)
            .otherwise(pl.col("ell"))
            .alias("ell")
        )
        sample = Sample(data, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        ame = model.margins(variables=["ell"])[0]
        assert np.isfinite(ame.margin[0])
        assert np.isfinite(ame.se[0])

        pred = model.margins(at={"meals": [20, 50]})
        assert np.all(np.isfinite(pred.margin))
        assert np.all(np.isfinite(pred.se))


# =============================================================================
# Output Tests
# =============================================================================


class TestGLMMarginOutput:
    """Tests for margins output formats."""

    def test_to_polars(self, api_binary):
        """to_polars() returns DataFrame."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        margins = model.margins(at={"meals": [20, 50, 80]})
        df = margins.to_polars()

        assert isinstance(df, pl.DataFrame)
        assert "margin" in df.columns
        assert "se" in df.columns
        assert "value" in df.columns
        assert df.height == 3

    def test_to_dict(self, api_binary):
        """to_dict() returns dictionary."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        margins = model.margins(at={"meals": [20, 50]})
        d = margins.to_dict()

        assert isinstance(d, dict)
        assert "margin" in d
        assert "se" in d
        assert "margin_type" in d

    def test_repr(self, api_binary):
        """__repr__ is informative."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        margins = model.margins(at={"meals": [20, 50]})
        r = repr(margins)

        assert "GLMMargins" in r
        assert "meals" in r
        assert "predictive" in r


# =============================================================================
# Edge Cases
# =============================================================================


class TestGLMMarginEdgeCases:
    """Edge case tests."""

    def test_margins_not_fitted_raises(self, api_strat):
        """Margins before fit raises error."""
        from svy.errors.model_errors import ModelError

        sample = Sample(api_strat, Design(wgt="pw"))

        with pytest.raises(ModelError):
            sample.glm.margins()

    def test_margins_ci_contains_estimate(self, api_binary):
        """CI contains the margin estimate."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        margins = model.margins(at={"meals": [20, 50, 80]})

        assert np.all(margins.lci <= margins.margin)
        assert np.all(margins.margin <= margins.uci)


class TestGLMMarginValidation:
    """Tests for variable validation."""

    def test_at_invalid_variable_raises(self, api_binary):
        """Invalid variable in at raises ValueError."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        with pytest.raises(ValueError, match="not found in model"):
            model.margins(at={"invalid_var": [1, 2, 3]})

    def test_at_invalid_variable_shows_available(self, api_binary):
        """Error message shows available variables."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        with pytest.raises(ValueError, match="Available variables"):
            model.margins(at={"mobility": [1, 2]})  # not in model

    def test_ame_invalid_variable_raises(self, api_binary):
        """Invalid variable in variables raises ValueError."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        with pytest.raises(ValueError, match="not found in model"):
            model.margins(variables=["invalid_var"])

    def test_at_valid_variable_works(self, api_binary):
        """Valid variable in at works."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)

        # Should not raise
        margins = model.margins(at={"meals": [20, 50]})
        assert margins.term == "meals"
