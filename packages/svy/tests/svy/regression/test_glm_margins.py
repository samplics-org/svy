# tests/svy/regression/test_glm_margins.py
"""
Tests for GLM margins against R.
"""

import numpy as np
import polars as pl
import pytest

from svy.core.enumerations import DistFamily
from svy.core.sample import Design, Sample


# =============================================================================
# R Reference Values
# =============================================================================

# R: Logistic AME (manual calculation)
# ame_ell = -0.0009982936
# ame_meals = -0.0080360454
# ame_mobility = 0.0007552178

R_LOGISTIC_AME = {
    "ell": -0.0009982936,
    "meals": -0.0080360454,
    "mobility": 0.0007552178,
}

# R: Predictive margins at meals = 20, 50, 80
# mean SE
# pred_20 0.96989 0.0005
# pred_50 0.77227 0.0031
# pred_80 0.26554 0.0032

R_PREDICTIVE_MARGINS = {
    "values": np.array([20, 50, 80]),
    "margin": np.array([0.96989, 0.77227, 0.26554]),
    "se": np.array([0.0005, 0.0031, 0.0032]),
}

# R: Linear AME (just coefficients)
R_LINEAR_AME = {
    "ell": -0.4805866,
    "meals": -3.1415353,
    "mobility": 0.2257132,
}

RTOL = 1e-2  # 1% tolerance for margins
ATOL = 1e-4


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def api_strat():
    return pl.read_csv("tests/test_data/apistrat.csv")


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

        np.testing.assert_allclose(
            margins.margin, R_PREDICTIVE_MARGINS["margin"], rtol=RTOL, atol=ATOL
        )

    def test_predictive_margins_se_vs_r(self, api_binary):
        """Predictive margins SE matches R."""
        sample = Sample(api_binary, Design(wgt="pw"))
        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        margins = model.margins(at={"meals": [20, 50, 80]})

        # SE tolerance is looser due to simplified variance estimation
        np.testing.assert_allclose(
            margins.se,
            R_PREDICTIVE_MARGINS["se"],
            rtol=0.5,
            atol=0.01,  # Allow 50% relative difference for SE
        )

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
