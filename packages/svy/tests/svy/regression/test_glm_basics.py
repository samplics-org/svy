# tests/svy/regression/test_glm_basics.py
import numpy as np
import polars as pl
import pytest

from svy.core.design import Design
from svy.core.sample import Sample
from svy.core.terms import Cat, Cross  # New import
from svy.regression.glm import GLMFit


@pytest.fixture
def survey_data():
    """
    Synthetic survey data with known relationships.
    y = 2 + 0.5*x + 1.5*z + 0.5*(x*z) + noise
    Stratified by 'stratum', clustered by 'psu'.
    """
    np.random.seed(42)
    n = 100

    # Design vars
    strata = np.random.choice(["A", "B"], n)
    psu = np.random.randint(1, 11, n)  # 10 clusters
    weights = np.random.uniform(0.5, 2.0, n)

    # Features
    x = np.random.normal(10, 2, n)
    z_cat = np.random.choice(["M", "F"], n)

    # Target (linear with interaction)
    # z_eff = 1.5 if F, 0 if M
    z_num = (z_cat == "F").astype(float)
    noise = np.random.normal(0, 1, n)
    # Interaction effect: 0.5 * x when z is F
    y = 2.0 + 0.5 * x + 1.5 * z_num + 0.5 * (x * z_num) + noise

    # Target (binary for logistic)
    # prob = sigmoid( -2 + 0.3*x )
    prob = 1 / (1 + np.exp(-(-2 + 0.3 * x)))
    y_bin = np.random.binomial(1, prob)

    return pl.DataFrame(
        {"y": y, "y_bin": y_bin, "x": x, "z": z_cat, "stratum": strata, "psu": psu, "wgt": weights}
    )


def test_glm_linear_gaussian(survey_data):
    """Test standard linear regression (Gaussian family) using the new unified API."""
    design = Design(stratum="stratum", psu="psu", wgt="wgt", row_index="id")
    df = survey_data.with_row_index("id")

    sample = Sample(df, design)

    # Fit model: y ~ x + z (categorical)
    # New API: Use Cat() wrapper inside the x list
    model = sample.glm.fit(
        y="y",
        x=["x", Cat("z")],  # Unified list mixing string (continuous) and Cat (categorical)
        family="Gaussian",
    )

    res = model.fitted

    assert isinstance(res, GLMFit)
    # Check Enums are resolved correctly from strings/inputs
    assert "Gaussian" in res.family
    assert "identity" in res.link
    assert res.stats.n == 100

    # Check dimensions
    # Intercept + x + z (one dummy, since ref is dropped) = 3 terms
    assert len(res.coefs) == 3

    terms = [c.term for c in res.coefs]
    assert "_intercept_" in terms
    assert "x" in terms

    # Check dummy name generation (e.g., "z_M" or "z_F")
    assert any("z_" in t for t in terms)

    # Check coefficients are valid
    for c in res.coefs:
        assert not np.isnan(c.est)
        assert not np.isnan(c.se)
        # SE should be positive
        assert c.se > 0


def test_glm_interaction(survey_data):
    """Test interaction terms using Cross()."""
    design = Design(stratum="stratum", psu="psu", wgt="wgt", row_index="id")
    df = survey_data.with_row_index("id")
    sample = Sample(df, design)

    # Fit model: y ~ x + z + x:z
    model = sample.glm.fit(
        y="y",
        x=[
            "x",
            Cat("z", ref="M"),  # Explicit reference level
            Cross("x", Cat("z", ref="M")),  # Interaction between continuous and categorical
        ],
        family="Gaussian",
    )

    res = model.fitted
    terms = [c.term for c in res.coefs]

    # Intercept + x + z_F + x:z_F = 4 terms
    assert len(res.coefs) == 4
    assert "_intercept_" in terms
    assert "x" in terms
    assert "z_F" in terms  # Since ref="M", F is the dummy

    # Check for interaction term
    # The name format depends on your implementation, usually "x:z_F" or similar
    interaction_term = next((t for t in terms if ":" in t), None)
    assert interaction_term is not None
    assert "x" in interaction_term and "z_F" in interaction_term


def test_glm_logistic_binomial(survey_data):
    """Test logistic regression (Binomial family)."""
    design = Design(stratum="stratum", psu="psu", wgt="wgt", row_index="id")
    df = survey_data.with_row_index("id")
    sample = Sample(df, design)

    # Fit model: y_bin ~ x (simple list of strings for continuous features)
    model = sample.glm.fit(y="y_bin", x=["x"], family="binomial", link="logit")

    res = model.fitted

    assert "Binomial" in res.family
    assert "logit" in res.link

    # Intercept + x
    assert len(res.coefs) == 2

    # Convergence check
    assert res.stats.iterations is not None
    assert res.stats.iterations > 0


def test_glm_design_handling(survey_data):
    """Test that design columns are correctly passed and used."""

    df = survey_data.clone().with_row_index("id")

    # Make stratum 'C' have only 1 PSU
    extra_row = df.head(1).with_columns(
        pl.lit("C").alias("stratum"),
        pl.lit(999, dtype=pl.Int64).alias("psu"),
        pl.lit(999999, dtype=pl.UInt32).alias("id"),
    )

    df_aug = pl.concat([df, extra_row])

    design = Design(stratum="stratum", psu="psu", wgt="wgt", row_index="id")
    sample = Sample(df_aug, design)

    # Fit simple model
    model = sample.glm.fit(y="y", x=["x"], family="gaussian")
    res = model.fitted

    # Should run successfully
    assert res.stats.n == 101

    # Check validity of inference
    for c in res.coefs:
        assert not np.isnan(c.est)
        assert not np.isnan(c.se)


def test_glm_missing_data_drop(survey_data):
    """Test that nulls in features are dropped automatically."""
    df = survey_data.with_row_index("id")

    # Introduce nulls in 'x'
    df = df.with_columns(
        pl.when(pl.col("id") < 10).then(None).otherwise(pl.col("x")).alias("x_miss")
    )

    design = Design(stratum="stratum", psu="psu", wgt="wgt", row_index="id")
    sample = Sample(df, design)

    # Fit with default drop_nulls=True
    model = sample.glm.fit(y="y", x=["x_miss"], drop_nulls=True)
    res = model.fitted

    assert res.stats.n == 90  # 100 - 10 nulls
