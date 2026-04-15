# tests/svy/regression/test_glm_families.py
"""
Tests for GLM families: Poisson and Gamma.
"""

import numpy as np
import polars as pl
import pytest

from svy.core.enumerations import DistFamily, LinkFunction
from svy.core.sample import Design, Sample


# =============================================================================
# Test Data Setup
# =============================================================================


@pytest.fixture
def api_strat():
    """Load apistrat dataset."""
    return pl.read_csv("tests/test_data/apistrat.csv")


@pytest.fixture
def sample_weighted(api_strat):
    """Sample with weights only."""
    return Sample(api_strat, Design(wgt="pw"))


# =============================================================================
# R Reference Values - POISSON
# =============================================================================

# R: svyglm(enroll ~ ell + meals + mobility, design = des, family = poisson())

R_POISSON_COEF = {
    "_intercept_": 6.405673,
    "ell": 0.001914,
    "meals": -0.001946,
    "mobility": 0.001897,
}

R_POISSON_SE = {
    "_intercept_": 0.099834,
    "ell": 0.002664,
    "meals": 0.002379,
    "mobility": 0.003718,
}

R_POISSON_PRED = {
    "yhat": np.array([608.01, 608.93, 576.76, 595.52, 630.45]),
    "se": np.array([37.415, 62.287, 34.620, 51.981, 52.420]),
}

R_POISSON_AME = {
    "ell": 1.139397,
    "meals": -1.158219,
}


# =============================================================================
# R Reference Values - GAMMA (log link)
# =============================================================================

# R: svyglm(api00 ~ ell + meals + mobility, design = des, family = Gamma(link = "log"))

R_GAMMA_LOG_COEF = {
    "_intercept_": 6.7268685,
    "ell": -0.0009556,
    "meals": -0.0047223,
    "mobility": 0.0003894,
}

R_GAMMA_LOG_SE = {
    "_intercept_": 0.0155390,
    "ell": 0.0006872,
    "meals": 0.0004792,
    "mobility": 0.0006220,
}

R_GAMMA_LOG_PRED = {
    "yhat": np.array([700.24, 493.06, 607.46, 533.67, 735.03]),
    "se": np.array([7.8230, 13.4877, 7.1267, 11.4848, 10.4767]),
}

R_GAMMA_LOG_AME = {
    "ell": -0.6328442,
    "meals": -3.1273913,
}


# =============================================================================
# R Reference Values - GAMMA (inverse link - canonical)
# =============================================================================

# R: svyglm(api00 ~ ell + meals + mobility, design = des, family = Gamma(link = "inverse"))

R_GAMMA_INV_COEF = {
    "_intercept_": 1.174e-03,
    "ell": 1.902e-06,
    "meals": 7.064e-06,
    "mobility": -7.456e-07,
}

R_GAMMA_INV_SE = {
    "_intercept_": 2.028e-05,
    "ell": 1.174e-06,
    "meals": 7.331e-07,
    "mobility": 9.128e-07,
}

R_GAMMA_INV_PRED = {
    "yhat": np.array([691.11, 501.55, 603.27, 534.50, 731.33]),
    "se": np.array([8.3076, 12.4709, 7.0980, 11.2555, 10.8495]),
}


RTOL = 1e-3
ATOL = 1e-4


# =============================================================================
# POISSON Tests
# =============================================================================


class TestGLMPoisson:
    """Tests for Poisson family."""

    def test_poisson_coefficients_vs_r(self, sample_weighted, api_strat):
        """Poisson coefficients match R."""
        model = sample_weighted.glm.fit(
            y="enroll",
            x=["ell", "meals", "mobility"],
            family=DistFamily.POISSON,
        )

        coef_dict = {c.term: c.est for c in model.coefs}

        for term, expected in R_POISSON_COEF.items():
            np.testing.assert_allclose(
                coef_dict[term],
                expected,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Coefficient mismatch for {term}",
            )

    def test_poisson_se_vs_r(self, sample_weighted, api_strat):
        """Poisson standard errors match R."""
        model = sample_weighted.glm.fit(
            y="enroll",
            x=["ell", "meals", "mobility"],
            family=DistFamily.POISSON,
        )

        se_dict = {c.term: c.se for c in model.coefs}

        for term, expected in R_POISSON_SE.items():
            np.testing.assert_allclose(
                se_dict[term], expected, rtol=RTOL, atol=ATOL, err_msg=f"SE mismatch for {term}"
            )

    def test_poisson_predictions_vs_r(self, sample_weighted, api_strat):
        """Poisson predictions match R."""
        model = sample_weighted.glm.fit(
            y="enroll",
            x=["ell", "meals", "mobility"],
            family=DistFamily.POISSON,
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.yhat, R_POISSON_PRED["yhat"], rtol=RTOL, atol=ATOL)

    def test_poisson_prediction_se_vs_r(self, sample_weighted, api_strat):
        """Poisson prediction SE matches R."""
        model = sample_weighted.glm.fit(
            y="enroll",
            x=["ell", "meals", "mobility"],
            family=DistFamily.POISSON,
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.se, R_POISSON_PRED["se"], rtol=RTOL, atol=ATOL)

    def test_poisson_predictions_positive(self, sample_weighted, api_strat):
        """Poisson predictions are always positive."""
        model = sample_weighted.glm.fit(
            y="enroll",
            x=["ell", "meals", "mobility"],
            family=DistFamily.POISSON,
        )

        pred = model.predict(api_strat)

        assert np.all(pred.yhat > 0)

    def test_poisson_ame_vs_r(self, sample_weighted, api_strat):
        """Poisson AME matches R."""
        model = sample_weighted.glm.fit(
            y="enroll",
            x=["ell", "meals", "mobility"],
            family=DistFamily.POISSON,
        )

        margins = model.margins()
        ame_dict = {m.term: m.margin[0] for m in margins}

        np.testing.assert_allclose(ame_dict["ell"], R_POISSON_AME["ell"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ame_dict["meals"], R_POISSON_AME["meals"], rtol=RTOL, atol=ATOL)


# =============================================================================
# GAMMA (log link) Tests
# =============================================================================


class TestGLMGammaLog:
    """Tests for Gamma family with log link."""

    def test_gamma_log_coefficients_vs_r(self, sample_weighted, api_strat):
        """Gamma (log link) coefficients match R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.LOG,
        )

        coef_dict = {c.term: c.est for c in model.coefs}

        for term, expected in R_GAMMA_LOG_COEF.items():
            np.testing.assert_allclose(
                coef_dict[term],
                expected,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Coefficient mismatch for {term}",
            )

    def test_gamma_log_se_vs_r(self, sample_weighted, api_strat):
        """Gamma (log link) standard errors match R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.LOG,
        )

        se_dict = {c.term: c.se for c in model.coefs}

        for term, expected in R_GAMMA_LOG_SE.items():
            np.testing.assert_allclose(
                se_dict[term], expected, rtol=RTOL, atol=ATOL, err_msg=f"SE mismatch for {term}"
            )

    def test_gamma_log_predictions_vs_r(self, sample_weighted, api_strat):
        """Gamma (log link) predictions match R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.LOG,
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.yhat, R_GAMMA_LOG_PRED["yhat"], rtol=RTOL, atol=ATOL)

    def test_gamma_log_prediction_se_vs_r(self, sample_weighted, api_strat):
        """Gamma (log link) prediction SE matches R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.LOG,
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.se, R_GAMMA_LOG_PRED["se"], rtol=RTOL, atol=ATOL)

    def test_gamma_log_predictions_positive(self, sample_weighted, api_strat):
        """Gamma (log link) predictions are always positive."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.LOG,
        )

        pred = model.predict(api_strat)

        assert np.all(pred.yhat > 0)

    def test_gamma_log_ame_vs_r(self, sample_weighted, api_strat):
        """Gamma (log link) AME matches R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.LOG,
        )

        margins = model.margins()
        ame_dict = {m.term: m.margin[0] for m in margins}

        np.testing.assert_allclose(ame_dict["ell"], R_GAMMA_LOG_AME["ell"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            ame_dict["meals"], R_GAMMA_LOG_AME["meals"], rtol=RTOL, atol=ATOL
        )


# =============================================================================
# GAMMA (inverse link - canonical) Tests
# =============================================================================


class TestGLMGammaInverse:
    """Tests for Gamma family with inverse link (canonical)."""

    def test_gamma_inverse_coefficients_vs_r(self, sample_weighted, api_strat):
        """Gamma (inverse link) coefficients match R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.INVERSE,
        )

        coef_dict = {c.term: c.est for c in model.coefs}

        for term, expected in R_GAMMA_INV_COEF.items():
            np.testing.assert_allclose(
                coef_dict[term],
                expected,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Coefficient mismatch for {term}",
            )

    def test_gamma_inverse_se_vs_r(self, sample_weighted, api_strat):
        """Gamma (inverse link) standard errors match R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.INVERSE,
        )

        se_dict = {c.term: c.se for c in model.coefs}

        for term, expected in R_GAMMA_INV_SE.items():
            np.testing.assert_allclose(
                se_dict[term], expected, rtol=RTOL, atol=ATOL, err_msg=f"SE mismatch for {term}"
            )

    def test_gamma_inverse_predictions_vs_r(self, sample_weighted, api_strat):
        """Gamma (inverse link) predictions match R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.INVERSE,
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.yhat, R_GAMMA_INV_PRED["yhat"], rtol=RTOL, atol=ATOL)

    def test_gamma_inverse_prediction_se_vs_r(self, sample_weighted, api_strat):
        """Gamma (inverse link) prediction SE matches R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.INVERSE,
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.se, R_GAMMA_INV_PRED["se"], rtol=RTOL, atol=ATOL)

    def test_gamma_inverse_predictions_positive(self, sample_weighted, api_strat):
        """Gamma (inverse link) predictions are always positive."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
            family=DistFamily.GAMMA,
            link=LinkFunction.INVERSE,
        )

        pred = model.predict(api_strat)

        assert np.all(pred.yhat > 0)


# =============================================================================
# Default Link Tests
# =============================================================================


class TestGLMDefaultLinks:
    """Tests for default link functions."""

    def test_poisson_default_link_is_log(self, sample_weighted, api_strat):
        """Poisson default link is log."""
        model = sample_weighted.glm.fit(
            y="enroll",
            x=["ell", "meals"],
            family=DistFamily.POISSON,
            # No link specified - should default to log
        )

        assert model.fitted.link == "log"

    def test_gamma_default_link_is_inverse(self, sample_weighted, api_strat):
        """Gamma default link is inverse."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals"],
            family=DistFamily.GAMMA,
            # No link specified - should default to inverse
        )

        assert model.fitted.link == "inverse"

    def test_gaussian_default_link_is_identity(self, sample_weighted, api_strat):
        """Gaussian default link is identity."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals"],
            family=DistFamily.GAUSSIAN,
        )

        assert model.fitted.link == "identity"

    def test_binomial_default_link_is_logit(self, api_strat):
        """Binomial default link is logit."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))
        sample = Sample(api_binary, Design(wgt="pw"))

        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals"],
            family=DistFamily.BINOMIAL,
        )

        assert model.fitted.link == "logit"


# =============================================================================
# Domain Validation Tests
# =============================================================================


class TestGLMDomainValidation:
    """Tests for response variable domain validation."""

    def test_poisson_negative_raises(self, api_strat):
        """Poisson with negative values raises error."""
        from svy.errors.model_errors import ModelError

        # Create data with negative values
        bad_data = api_strat.with_columns((pl.col("enroll") - 1000).alias("bad_enroll"))
        sample = Sample(bad_data, Design(wgt="pw"))

        with pytest.raises(ModelError):
            sample.glm.fit(
                y="bad_enroll",
                x=["ell", "meals"],
                family=DistFamily.POISSON,
            )

    def test_gamma_non_positive_raises(self, api_strat):
        """Gamma with non-positive values raises error."""
        from svy.errors.model_errors import ModelError

        # Create data with zero/negative values
        bad_data = api_strat.with_columns((pl.col("api00") - 1000).alias("bad_api"))
        sample = Sample(bad_data, Design(wgt="pw"))

        with pytest.raises(ModelError):
            sample.glm.fit(
                y="bad_api",
                x=["ell", "meals"],
                family=DistFamily.GAMMA,
            )


# =============================================================================
# Complex Design Tests
# =============================================================================


class TestGLMFamiliesComplexDesign:
    """Tests for families with complex survey designs."""

    def test_poisson_stratified(self, api_strat):
        """Poisson works with stratified design."""
        sample = Sample(api_strat, Design(wgt="pw", stratum="stype"))

        model = sample.glm.fit(
            y="enroll",
            x=["ell", "meals"],
            family=DistFamily.POISSON,
        )

        assert model.fitted is not None
        assert len(model.coefs) == 3  # intercept + 2 vars

    def test_poisson_clustered(self, api_strat):
        """Poisson works with clustered design."""
        sample = Sample(api_strat, Design(wgt="pw", stratum="stype", psu="dnum"))

        model = sample.glm.fit(
            y="enroll",
            x=["ell", "meals"],
            family=DistFamily.POISSON,
        )

        assert model.fitted is not None
        pred = model.predict(api_strat.head(5))
        assert len(pred) == 5

    def test_gamma_stratified(self, api_strat):
        """Gamma works with stratified design."""
        sample = Sample(api_strat, Design(wgt="pw", stratum="stype"))

        model = sample.glm.fit(
            y="api00",
            x=["ell", "meals"],
            family=DistFamily.GAMMA,
            link=LinkFunction.LOG,
        )

        assert model.fitted is not None

    def test_gamma_clustered(self, api_strat):
        """Gamma works with clustered design."""
        sample = Sample(api_strat, Design(wgt="pw", stratum="stype", psu="dnum"))

        model = sample.glm.fit(
            y="api00",
            x=["ell", "meals"],
            family=DistFamily.GAMMA,
            link=LinkFunction.LOG,
        )

        assert model.fitted is not None
        pred = model.predict(api_strat.head(5))
        assert np.all(pred.yhat > 0)
