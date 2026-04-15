# tests/svy/regression/test_glm_predict.py
"""
Tests for GLM prediction against R's predict.svyglm().
"""

import numpy as np
import polars as pl
import pytest

from svy.core.enumerations import DistFamily
from svy.core.sample import Design, Sample
from svy.core.terms import Cat


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


@pytest.fixture
def sample_stratified(api_strat):
    """Sample with strata and weights."""
    return Sample(api_strat, Design(wgt="pw", stratum="stype"))


@pytest.fixture
def sample_clustered(api_strat):
    """Sample with PSU, strata, and weights."""
    return Sample(api_strat, Design(wgt="pw", stratum="stype", psu="dnum"))


# =============================================================================
# R Reference Values - Weights Only
# =============================================================================

# R code:
# des <- svydesign(ids = ~1, weights = ~pw, data = apistrat)
# fit <- svyglm(api00 ~ ell + meals + mobility, design = des)
# pred <- predict(fit, newdata = head(apistrat, 5), se.fit = TRUE)

R_LINEAR_PRED = {
    "yhat": np.array([707.68, 481.88, 612.61, 532.80, 738.03]),
    "se": np.array([7.4485, 15.1970, 6.7514, 11.8960, 9.6580]),
}

# R code:
# apistrat$y_bin <- as.integer(apistrat$api00 > 600)
# des2 <- svydesign(ids = ~1, weights = ~pw, data = apistrat)
# fit <- svyglm(y_bin ~ ell + meals + mobility, design = des2, family = binomial())
# pred <- predict(fit, newdata = head(apistrat, 5), type = "response", se.fit = TRUE)

R_LOGISTIC_PRED = {
    "yhat": np.array([0.920569, 0.055786, 0.543685, 0.159838, 0.960614]),
    "se": np.array([0.0285, 0.0364, 0.0620, 0.0713, 0.0175]),
}


# =============================================================================
# R Reference Values - Stratified Design
# =============================================================================

# R code:
# des_strat <- svydesign(ids = ~1, strata = ~stype, weights = ~pw, data = apistrat)
# fit_strat <- svyglm(api00 ~ ell + meals + mobility, design = des_strat)
# pred_strat <- predict(fit_strat, newdata = head(apistrat, 5), se.fit = TRUE)

R_LINEAR_PRED_STRAT = {
    "yhat": np.array([707.68, 481.88, 612.61, 532.80, 738.03]),
    "se": np.array([6.7968, 15.1391, 6.4712, 11.7992, 9.0374]),
}

# R code:
# fit_strat_log <- svyglm(y_bin ~ ell + meals + mobility, design = des_strat2, family = binomial())
# pred_strat_log <- predict(fit_strat_log, newdata = head(apistrat, 5), type = "response", se.fit = TRUE)

R_LOGISTIC_PRED_STRAT = {
    "yhat": np.array([0.920569, 0.055786, 0.543685, 0.159838, 0.960614]),
    "se": np.array([0.0267, 0.0362, 0.0611, 0.0710, 0.0168]),
}


# =============================================================================
# R Reference Values - Clustered Design (PSU + Strata)
# =============================================================================

# R code:
# des_clust <- svydesign(ids = ~dnum, strata = ~stype, weights = ~pw, data = apistrat, nest = TRUE)
# fit_clust <- svyglm(api00 ~ ell + meals + mobility, design = des_clust)
# pred_clust <- predict(fit_clust, newdata = head(apistrat, 5), se.fit = TRUE)

R_LINEAR_PRED_CLUST = {
    "yhat": np.array([707.68, 481.88, 612.61, 532.80, 738.03]),
    "se": np.array([7.7563, 16.0752, 6.5254, 12.8384, 9.1637]),
}

# R code:
# fit_clust_log <- svyglm(y_bin ~ ell + meals + mobility, design = des_clust2, family = binomial())
# pred_clust_log <- predict(fit_clust_log, newdata = head(apistrat, 5), type = "response", se.fit = TRUE)

R_LOGISTIC_PRED_CLUST = {
    "yhat": np.array([0.920569, 0.055786, 0.543685, 0.159838, 0.960614]),
    "se": np.array([0.0290, 0.0349, 0.0574, 0.0726, 0.0176]),
}


RTOL = 1e-3
ATOL = 1e-4


# =============================================================================
# Basic Tests
# =============================================================================


class TestGLMPredictBasic:
    """Basic prediction functionality tests."""

    def test_predict_returns_glmpred(self, sample_weighted, api_strat):
        """Predict returns GLMPred object."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
        )

        pred = model.predict(api_strat.head(5))

        from svy.regression import GLMPred

        assert isinstance(pred, GLMPred)

    def test_predict_length_matches_data(self, sample_weighted, api_strat):
        """Prediction length matches input data."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
        )

        new_data = api_strat.head(10)
        pred = model.predict(new_data)

        assert len(pred) == 10
        assert len(pred.yhat) == 10
        assert len(pred.se) == 10
        assert len(pred.lci) == 10
        assert len(pred.uci) == 10

    def test_predict_has_expected_attributes(self, sample_weighted, api_strat):
        """GLMPred has all expected attributes."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
        )

        pred = model.predict(api_strat.head(5))

        assert hasattr(pred, "yhat")
        assert hasattr(pred, "se")
        assert hasattr(pred, "lci")
        assert hasattr(pred, "uci")
        assert hasattr(pred, "df")
        assert hasattr(pred, "alpha")
        assert hasattr(pred, "residuals")

    def test_predict_default_alpha(self, sample_weighted, api_strat):
        """Default alpha is 0.05."""
        model = sample_weighted.glm.fit(y="api00", x=["ell"])
        pred = model.predict(api_strat.head(5))

        assert pred.alpha == 0.05
        assert pred.conf_level == 0.95

    def test_predict_custom_alpha(self, sample_weighted, api_strat):
        """Custom alpha is respected."""
        model = sample_weighted.glm.fit(y="api00", x=["ell"])
        pred = model.predict(api_strat.head(5), alpha=0.10)

        assert pred.alpha == 0.10
        assert pred.conf_level == 0.90


# =============================================================================
# R Validation Tests - Weights Only
# =============================================================================


class TestGLMPredictVsR:
    """Validate predictions against R's predict.svyglm() - weights only."""

    def test_linear_yhat_vs_r(self, sample_weighted, api_strat):
        """Linear model yhat matches R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.yhat, R_LINEAR_PRED["yhat"], rtol=RTOL, atol=ATOL)

    def test_linear_se_vs_r(self, sample_weighted, api_strat):
        """Linear model SE matches R."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.se, R_LINEAR_PRED["se"], rtol=RTOL, atol=ATOL)

    def test_logistic_yhat_vs_r(self, api_strat):
        """Logistic model yhat matches R."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))
        sample = Sample(api_binary, Design(wgt="pw"))

        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        pred = model.predict(api_binary.head(5))

        np.testing.assert_allclose(pred.yhat, R_LOGISTIC_PRED["yhat"], rtol=RTOL, atol=ATOL)

    def test_logistic_se_vs_r(self, api_strat):
        """Logistic model SE matches R."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))
        sample = Sample(api_binary, Design(wgt="pw"))

        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        pred = model.predict(api_binary.head(5))

        np.testing.assert_allclose(pred.se, R_LOGISTIC_PRED["se"], rtol=RTOL, atol=ATOL)


# =============================================================================
# R Validation Tests - Stratified Design
# =============================================================================


class TestGLMPredictVsRStratified:
    """Validate predictions against R - stratified design."""

    def test_stratified_linear_yhat_vs_r(self, sample_stratified, api_strat):
        """Stratified linear model yhat matches R."""
        model = sample_stratified.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.yhat, R_LINEAR_PRED_STRAT["yhat"], rtol=RTOL, atol=ATOL)

    def test_stratified_linear_se_vs_r(self, sample_stratified, api_strat):
        """Stratified linear model SE matches R."""
        model = sample_stratified.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.se, R_LINEAR_PRED_STRAT["se"], rtol=RTOL, atol=ATOL)

    def test_stratified_logistic_yhat_vs_r(self, api_strat):
        """Stratified logistic model yhat matches R."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))
        sample = Sample(api_binary, Design(wgt="pw", stratum="stype"))

        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        pred = model.predict(api_binary.head(5))

        np.testing.assert_allclose(pred.yhat, R_LOGISTIC_PRED_STRAT["yhat"], rtol=RTOL, atol=ATOL)

    def test_stratified_logistic_se_vs_r(self, api_strat):
        """Stratified logistic model SE matches R."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))
        sample = Sample(api_binary, Design(wgt="pw", stratum="stype"))

        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        pred = model.predict(api_binary.head(5))

        np.testing.assert_allclose(pred.se, R_LOGISTIC_PRED_STRAT["se"], rtol=RTOL, atol=ATOL)


# =============================================================================
# R Validation Tests - Clustered Design (PSU + Strata)
# =============================================================================


class TestGLMPredictVsRClustered:
    """Validate predictions against R - clustered design."""

    def test_clustered_linear_yhat_vs_r(self, sample_clustered, api_strat):
        """Clustered linear model yhat matches R."""
        model = sample_clustered.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.yhat, R_LINEAR_PRED_CLUST["yhat"], rtol=RTOL, atol=ATOL)

    def test_clustered_linear_se_vs_r(self, sample_clustered, api_strat):
        """Clustered linear model SE matches R."""
        model = sample_clustered.glm.fit(
            y="api00",
            x=["ell", "meals", "mobility"],
        )

        pred = model.predict(api_strat.head(5))

        np.testing.assert_allclose(pred.se, R_LINEAR_PRED_CLUST["se"], rtol=RTOL, atol=ATOL)

    def test_clustered_logistic_yhat_vs_r(self, api_strat):
        """Clustered logistic model yhat matches R."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))
        sample = Sample(api_binary, Design(wgt="pw", stratum="stype", psu="dnum"))

        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        pred = model.predict(api_binary.head(5))

        np.testing.assert_allclose(pred.yhat, R_LOGISTIC_PRED_CLUST["yhat"], rtol=RTOL, atol=ATOL)

    def test_clustered_logistic_se_vs_r(self, api_strat):
        """Clustered logistic model SE matches R."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))
        sample = Sample(api_binary, Design(wgt="pw", stratum="stype", psu="dnum"))

        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals", "mobility"],
            family=DistFamily.BINOMIAL,
        )

        pred = model.predict(api_binary.head(5))

        np.testing.assert_allclose(pred.se, R_LOGISTIC_PRED_CLUST["se"], rtol=RTOL, atol=ATOL)


# =============================================================================
# Design Comparison Tests
# =============================================================================


class TestGLMPredictDesignComparison:
    """Tests comparing predictions across different survey designs."""

    def test_yhat_same_across_designs(self, api_strat):
        """yhat should be same regardless of design (same coefficients)."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))

        # Weights only
        sample_wgt = Sample(api_binary, Design(wgt="pw"))
        model_wgt = sample_wgt.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)
        pred_wgt = model_wgt.predict(api_binary.head(5))

        # Stratified
        sample_strat = Sample(api_binary, Design(wgt="pw", stratum="stype"))
        model_strat = sample_strat.glm.fit(
            y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL
        )
        pred_strat = model_strat.predict(api_binary.head(5))

        # Clustered
        sample_clust = Sample(api_binary, Design(wgt="pw", stratum="stype", psu="dnum"))
        model_clust = sample_clust.glm.fit(
            y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL
        )
        pred_clust = model_clust.predict(api_binary.head(5))

        # yhat should be very close (coefficients may differ slightly due to variance)
        np.testing.assert_allclose(pred_wgt.yhat, pred_strat.yhat, rtol=1e-2)
        np.testing.assert_allclose(pred_strat.yhat, pred_clust.yhat, rtol=1e-2)

    def test_se_differs_by_design(self, api_strat):
        """SE should differ across designs."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))

        # Weights only
        sample_wgt = Sample(api_binary, Design(wgt="pw"))
        model_wgt = sample_wgt.glm.fit(y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL)
        pred_wgt = model_wgt.predict(api_binary.head(5))

        # Clustered
        sample_clust = Sample(api_binary, Design(wgt="pw", stratum="stype", psu="dnum"))
        model_clust = sample_clust.glm.fit(
            y="y_bin", x=["ell", "meals"], family=DistFamily.BINOMIAL
        )
        pred_clust = model_clust.predict(api_binary.head(5))

        # SE should differ
        assert not np.allclose(pred_wgt.se, pred_clust.se, rtol=1e-3)

    def test_df_differs_by_design(self, api_strat):
        """Degrees of freedom differ by design complexity."""
        # Weights only: df = n - 1 = 199
        sample_wgt = Sample(api_strat, Design(wgt="pw"))
        model_wgt = sample_wgt.glm.fit(y="api00", x=["ell"])
        pred_wgt = model_wgt.predict(api_strat.head(5))

        # Stratified: df = n - n_strata = 200 - 3 = 197
        sample_strat = Sample(api_strat, Design(wgt="pw", stratum="stype"))
        model_strat = sample_strat.glm.fit(y="api00", x=["ell"])
        pred_strat = model_strat.predict(api_strat.head(5))

        # Clustered: df = n_psu - n_strata
        sample_clust = Sample(api_strat, Design(wgt="pw", stratum="stype", psu="dnum"))
        model_clust = sample_clust.glm.fit(y="api00", x=["ell"])
        pred_clust = model_clust.predict(api_strat.head(5))

        # All should have different df
        assert pred_wgt.df != pred_strat.df
        assert pred_strat.df != pred_clust.df
        assert pred_wgt.df != pred_clust.df


# =============================================================================
# Residuals Tests
# =============================================================================


class TestGLMPredictResiduals:
    """Tests for residual computation."""

    def test_residuals_none_by_default(self, sample_weighted, api_strat):
        """Residuals are None when y_col not provided."""
        model = sample_weighted.glm.fit(y="api00", x=["ell", "meals"])
        pred = model.predict(api_strat.head(5))

        assert pred.residuals is None

    def test_residuals_computed_with_y_col(self, sample_weighted, api_strat):
        """Residuals are computed when y_col is provided."""
        model = sample_weighted.glm.fit(y="api00", x=["ell", "meals"])
        pred = model.predict(api_strat.head(5), y_col="api00")

        assert pred.residuals is not None
        assert len(pred.residuals) == 5

    def test_residuals_equal_y_minus_yhat(self, sample_weighted, api_strat):
        """Residuals = y - yhat."""
        model = sample_weighted.glm.fit(y="api00", x=["ell", "meals"])

        new_data = api_strat.head(5)
        pred = model.predict(new_data, y_col="api00")

        y = new_data.get_column("api00").to_numpy().astype(float)
        expected_resid = y - pred.yhat

        np.testing.assert_allclose(pred.residuals, expected_resid)


# =============================================================================
# Confidence Interval Tests
# =============================================================================


class TestGLMPredictCI:
    """Tests for confidence interval computation."""

    def test_ci_contains_yhat(self, sample_weighted, api_strat):
        """Confidence intervals contain predicted values."""
        model = sample_weighted.glm.fit(y="api00", x=["ell", "meals"])
        pred = model.predict(api_strat.head(10))

        assert np.all(pred.lci <= pred.yhat)
        assert np.all(pred.yhat <= pred.uci)

    def test_ci_wider_with_smaller_alpha(self, sample_weighted, api_strat):
        """99% CI is wider than 95% CI."""
        model = sample_weighted.glm.fit(y="api00", x=["ell", "meals"])
        new_data = api_strat.head(5)

        pred_95 = model.predict(new_data, alpha=0.05)
        pred_99 = model.predict(new_data, alpha=0.01)

        width_95 = pred_95.uci - pred_95.lci
        width_99 = pred_99.uci - pred_99.lci

        assert np.all(width_99 > width_95)

    def test_ci_narrower_with_larger_alpha(self, sample_weighted, api_strat):
        """90% CI is narrower than 95% CI."""
        model = sample_weighted.glm.fit(y="api00", x=["ell", "meals"])
        new_data = api_strat.head(5)

        pred_95 = model.predict(new_data, alpha=0.05)
        pred_90 = model.predict(new_data, alpha=0.10)

        width_95 = pred_95.uci - pred_95.lci
        width_90 = pred_90.uci - pred_90.lci

        assert np.all(width_90 < width_95)


# =============================================================================
# Categorical Variable Tests
# =============================================================================


class TestGLMPredictCategorical:
    """Tests for prediction with categorical variables."""

    def test_predict_with_categorical(self, sample_weighted, api_strat):
        """Prediction works with categorical variables."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=["ell", Cat("stype")],
        )

        pred = model.predict(api_strat.head(5))

        assert len(pred) == 5
        assert not np.any(np.isnan(pred.yhat))

    def test_predict_categorical_different_levels(self, sample_weighted, api_strat):
        """Predictions differ by categorical level."""
        model = sample_weighted.glm.fit(
            y="api00",
            x=[Cat("stype")],
        )

        # Create data with different stype values
        new_data = pl.DataFrame(
            {
                "stype": ["E", "M", "H"],
            }
        )

        pred = model.predict(new_data)

        # Predictions should differ
        assert len(set(pred.yhat)) > 1


# =============================================================================
# Logistic Regression Tests
# =============================================================================


class TestGLMPredictLogistic:
    """Tests for logistic regression predictions."""

    def test_logistic_yhat_is_probability(self, api_strat):
        """Logistic predictions are probabilities in [0, 1]."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))
        sample = Sample(api_binary, Design(wgt="pw"))

        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals"],
            family=DistFamily.BINOMIAL,
        )

        pred = model.predict(api_binary.head(10))

        assert np.all(pred.yhat >= 0)
        assert np.all(pred.yhat <= 1)

    def test_logistic_ci_bounded(self, api_strat):
        """Logistic CI bounds are in [0, 1]."""
        api_binary = api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))
        sample = Sample(api_binary, Design(wgt="pw"))

        model = sample.glm.fit(
            y="y_bin",
            x=["ell", "meals"],
            family=DistFamily.BINOMIAL,
        )

        pred = model.predict(api_binary.head(10))

        assert np.all(pred.lci >= 0)
        assert np.all(pred.uci <= 1)


# =============================================================================
# Output Format Tests
# =============================================================================


class TestGLMPredictOutput:
    """Tests for output formats."""

    def test_to_polars(self, sample_weighted, api_strat):
        """to_polars() returns DataFrame with correct columns."""
        model = sample_weighted.glm.fit(y="api00", x=["ell"])
        pred = model.predict(api_strat.head(5))

        df = pred.to_polars()

        assert isinstance(df, pl.DataFrame)
        assert "yhat" in df.columns
        assert "se" in df.columns
        assert "lci" in df.columns
        assert "uci" in df.columns
        assert df.height == 5

    def test_to_polars_with_residuals(self, sample_weighted, api_strat):
        """to_polars() includes residuals when available."""
        model = sample_weighted.glm.fit(y="api00", x=["ell"])
        pred = model.predict(api_strat.head(5), y_col="api00")

        df = pred.to_polars()

        assert "residuals" in df.columns

    def test_to_dict(self, sample_weighted, api_strat):
        """to_dict() returns dictionary."""
        model = sample_weighted.glm.fit(y="api00", x=["ell"])
        pred = model.predict(api_strat.head(5))

        d = pred.to_dict()

        assert isinstance(d, dict)
        assert "yhat" in d
        assert "se" in d
        assert "df" in d
        assert "alpha" in d

    def test_repr(self, sample_weighted, api_strat):
        """__repr__ is informative."""
        model = sample_weighted.glm.fit(y="api00", x=["ell"])
        pred = model.predict(api_strat.head(5))

        r = repr(pred)

        assert "GLMPred" in r
        assert "n=5" in r
        assert "95%" in r


# =============================================================================
# Edge Cases
# =============================================================================


class TestGLMPredictEdgeCases:
    """Edge case tests."""

    def test_predict_single_row(self, sample_weighted, api_strat):
        """Prediction works with single row."""
        model = sample_weighted.glm.fit(y="api00", x=["ell"])
        pred = model.predict(api_strat.head(1))

        assert len(pred) == 1

    def test_predict_not_fitted_raises(self, sample_weighted, api_strat):
        """Predict before fit raises error."""
        from svy.errors.model_errors import ModelError

        with pytest.raises(ModelError):
            sample_weighted.glm.predict(api_strat.head(5))

    def test_predict_missing_column_raises(self, sample_weighted, api_strat):
        """Predict with missing predictor column raises error."""
        model = sample_weighted.glm.fit(y="api00", x=["ell", "meals"])

        bad_data = api_strat.head(5).drop("meals")

        with pytest.raises(KeyError):
            model.predict(bad_data)
