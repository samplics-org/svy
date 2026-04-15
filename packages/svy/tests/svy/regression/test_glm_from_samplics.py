# tests/svy/regression/test_glm_from_samplics.py
from pathlib import Path

import numpy as np
import polars as pl

from svy.core.enumerations import DistFamily, LinkFunction
from svy.core.sample import Design, Sample
from svy.core.terms import Cat


DATA_DIR = Path(__file__).parent.parent.parent / "test_data"

# Strict tolerance for all comparisons
RTOL = 1e-5
ATOL = 1e-7


def make_sample(
    df: pl.DataFrame,
    *,
    weight: str | None = None,
    stratum: str | None = None,
    psu: str | None = None,
):
    """Build Sample with optional design."""
    if weight is None and stratum is None and psu is None:
        design = None
    else:
        design = Design(wgt=weight, stratum=stratum, psu=psu)
    return Sample(data=df, design=design)


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

# Neuralgia data for categorical test
data_str = """
P  F  68   1  No   B  M  74  16  No  P  F  67  30  No
P  M  66  26  Yes  B  F  67  28  No  B  F  77  16  No
A  F  71  12  No   B  F  72  50  No  B  F  76   9  Yes
A  M  71  17  Yes  A  F  63  27  No  A  F  69  18  Yes
B  F  66  12  No   A  M  62  42  No  P  F  64   1  Yes
A  F  64  17  No   P  M  74   4  No  A  F  72  25  No
P  M  70   1  Yes  B  M  66  19  No  B  M  59  29  No
A  F  64  30  No   A  M  70  28  No  A  M  69   1  No
B  F  78   1  No   P  M  83   1  Yes B  F  69  42  No
B  M  75  30  Yes  P  M  77  29  Yes P  F  79  20  Yes
A  M  70  12  No   A  F  69  12  No  B  F  65  14  No
B  M  70   1  No   B  M  67  23  No  A  M  76  25  Yes
P  M  78  12  Yes  B  M  77   1  Yes B  F  69  24  No
P  M  66   4  Yes  P  F  65  29  No  P  M  60  26  Yes
A  M  78  15  Yes  B  M  75  21  Yes A  F  67  11  No
P  F  72  27  No   P  F  70  13  Yes A  M  75   6  Yes
B  F  65   7  No   P  F  68  27  Yes P  M  68  11  Yes
P  M  67  17  Yes  B  M  70  22  No  A  M  65  15  No
P  F  67   1  Yes  A  M  67  10  No  P  F  72  11  Yes
A  F  74   1  No   B  M  80  21  Yes A  F  69   3  No
"""

data_lines = data_str.strip().split("\n")
data_values: list[list[str]] = []
for line in data_lines:
    values = line.split()
    for i in range(0, len(values), 5):
        data_values.append(values[i : i + 5])

neuralgia = (
    pl.DataFrame(
        data_values,
        schema=[
            ("Treatment", pl.String),
            ("Sex", pl.String),
            ("Age", pl.Int32),
            ("Duration", pl.Int32),
            ("Pain", pl.String),
        ],
        orient="row",
    )
    .rename(mapping=str.lower)
    .with_columns(
        pl.when(pl.col("pain") == "Yes").then(pl.lit(0.0)).otherwise(pl.lit(1.0)).alias("y")
    )
)

# API strat data
api_strat = pl.read_csv(DATA_DIR / "api_strat.csv")
api_strat = api_strat.with_columns((pl.col("api00") > 743).cast(pl.Float64).alias("y_bin"))

X_COLS = ["ell", "meals", "mobility"]
STRATUM_COL = "stype"
PSU_COL = "dnum"
WEIGHT_COL = "pw"


def _coef_dict(result):
    """Map term -> GLMCoef for easy access."""
    return {c.term: c for c in result.coefs}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reg_logistic_categorical_factors():
    """Categorical predictors should work without error."""
    sample = make_sample(neuralgia)
    res = sample.glm.fit(
        y="y",
        x=["age", Cat("sex"), Cat("treatment")],
        family=DistFamily.BINOMIAL,
        link=LinkFunction.LOGIT,
    )
    terms = [c.term for c in res.coefs]
    assert "_intercept_" in terms
    assert any(t.startswith("sex_") for t in terms)
    assert any(t.startswith("treatment_") for t in terms)


def test_reg_logistic_not_stratified():
    """
    Weights only (no strata, no PSU)
    R: svydesign(ids = ~1, weights = ~pw, data = apistrat)
    Stata equivalent used for reference:
      svyset __psu_ids1 [pweight=pw]   (one PSU per obs)
      svy: logit y_bin ell meals mobility
    DF = 199
    """
    sample = make_sample(api_strat, weight=WEIGHT_COL)
    res = sample.glm.fit(y="y_bin", x=X_COLS, family=DistFamily.BINOMIAL, link=LinkFunction.LOGIT)
    coefs = _coef_dict(res)

    assert res.coefs[0].wald.df == 199

    beta = np.array(
        [coefs["_intercept_"].est, coefs["ell"].est, coefs["meals"].est, coefs["mobility"].est]
    )
    se = np.array(
        [coefs["_intercept_"].se, coefs["ell"].se, coefs["meals"].se, coefs["mobility"].se]
    )
    lci = np.array(
        [coefs["_intercept_"].lci, coefs["ell"].lci, coefs["meals"].lci, coefs["mobility"].lci]
    )
    uci = np.array(
        [coefs["_intercept_"].uci, coefs["ell"].uci, coefs["meals"].uci, coefs["mobility"].uci]
    )

    # Stata reference values (TEST 1)
    beta_ref = np.array([2.6657559, -0.03703249, -0.08468788, -0.00442859])
    se_ref = np.array([0.47183059, 0.03363392, 0.01460186, 0.01478849])
    lci_ref = np.array([1.7353260, -0.1033571, -0.1134821, -0.0335908])
    uci_ref = np.array([3.5961850, 0.0292921, -0.0558936, 0.0247337])

    np.testing.assert_allclose(beta, beta_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_ref, rtol=RTOL, atol=ATOL)


def test_reg_logistic_psu_not_stratified():
    """
    PSU only (no strata)
    R: svydesign(ids = ~dnum, weights = ~pw, data = apistrat)
    Stata reference:
      svyset dnum [pweight=pw]
      svy: logit y_bin ell meals mobility
    DF = 134
    """
    sample = make_sample(api_strat, weight=WEIGHT_COL, psu=PSU_COL)
    res = sample.glm.fit(y="y_bin", x=X_COLS, family=DistFamily.BINOMIAL, link=LinkFunction.LOGIT)
    coefs = _coef_dict(res)

    assert res.coefs[0].wald.df == 134

    beta = np.array(
        [coefs["_intercept_"].est, coefs["ell"].est, coefs["meals"].est, coefs["mobility"].est]
    )
    se = np.array(
        [coefs["_intercept_"].se, coefs["ell"].se, coefs["meals"].se, coefs["mobility"].se]
    )
    lci = np.array(
        [coefs["_intercept_"].lci, coefs["ell"].lci, coefs["meals"].lci, coefs["mobility"].lci]
    )
    uci = np.array(
        [coefs["_intercept_"].uci, coefs["ell"].uci, coefs["meals"].uci, coefs["mobility"].uci]
    )

    # Stata reference values (TEST 2)
    beta_ref = np.array([2.6657559, -0.03703249, -0.08468788, -0.00442859])
    se_ref = np.array([0.49495811, 0.03182350, 0.01438540, 0.01537275])
    lci_ref = np.array([1.6868150, -0.0999738, -0.1131397, -0.0348332])
    uci_ref = np.array([3.6446968, 0.0259088, -0.0562361, 0.0259760])

    np.testing.assert_allclose(beta, beta_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_ref, rtol=RTOL, atol=ATOL)


def test_reg_logistic_stratified():
    """
    Strata only (no PSU)
    R: svydesign(ids = ~1, strata = ~stype, weights = ~pw, data = apistrat)
    Stata reference:
      svyset __psu_ids1s [pweight=pw], strata(stype)  (one PSU per obs)
      svy: logit y_bin ell meals mobility
    DF = 197
    """
    sample = make_sample(api_strat, weight=WEIGHT_COL, stratum=STRATUM_COL)
    res = sample.glm.fit(y="y_bin", x=X_COLS, family=DistFamily.BINOMIAL, link=LinkFunction.LOGIT)
    coefs = _coef_dict(res)

    assert res.coefs[0].wald.df == 197

    beta = np.array(
        [coefs["_intercept_"].est, coefs["ell"].est, coefs["meals"].est, coefs["mobility"].est]
    )
    se = np.array(
        [coefs["_intercept_"].se, coefs["ell"].se, coefs["meals"].se, coefs["mobility"].se]
    )
    lci = np.array(
        [coefs["_intercept_"].lci, coefs["ell"].lci, coefs["meals"].lci, coefs["mobility"].lci]
    )
    uci = np.array(
        [coefs["_intercept_"].uci, coefs["ell"].uci, coefs["meals"].uci, coefs["mobility"].uci]
    )

    # Stata reference values (TEST 3)
    beta_ref = np.array([2.6657559, -0.03703249, -0.08468788, -0.00442859])
    se_ref = np.array([0.43944327, 0.03363709, 0.01433075, 0.01483659])
    lci_ref = np.array([1.7991390, -0.1033675, -0.1129492, -0.0336875])
    uci_ref = np.array([3.5323730, 0.0293025, -0.0564265, 0.0248303])

    np.testing.assert_allclose(beta, beta_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_ref, rtol=RTOL, atol=ATOL)


def test_reg_logistic_psu_stratified():
    """
    PSU + Strata (nested)
    R: svydesign(ids = ~dnum, strata = ~stype, weights = ~pw, data = apistrat, nest = TRUE)
    Stata reference:
      svyset dnum [pweight=pw], strata(stype)
      svy: logit y_bin ell meals mobility
    DF = 159
    """
    sample = make_sample(api_strat, weight=WEIGHT_COL, stratum=STRATUM_COL, psu=PSU_COL)
    res = sample.glm.fit(y="y_bin", x=X_COLS, family=DistFamily.BINOMIAL, link=LinkFunction.LOGIT)
    coefs = _coef_dict(res)

    assert res.coefs[0].wald.df == 159

    beta = np.array(
        [coefs["_intercept_"].est, coefs["ell"].est, coefs["meals"].est, coefs["mobility"].est]
    )
    se = np.array(
        [coefs["_intercept_"].se, coefs["ell"].se, coefs["meals"].se, coefs["mobility"].se]
    )
    lci = np.array(
        [coefs["_intercept_"].lci, coefs["ell"].lci, coefs["meals"].lci, coefs["mobility"].lci]
    )
    uci = np.array(
        [coefs["_intercept_"].uci, coefs["ell"].uci, coefs["meals"].uci, coefs["mobility"].uci]
    )

    # Stata reference values (TEST 4)
    beta_ref = np.array([2.6657559, -0.03703249, -0.08468788, -0.00442859])
    se_ref = np.array([0.47239532, 0.03241909, 0.01428377, 0.01546300])
    lci_ref = np.array([1.7327769, -0.1010601, -0.1128983, -0.0349680])
    uci_ref = np.array([3.5987349, 0.0269951, -0.0564775, 0.0261108])

    np.testing.assert_allclose(beta, beta_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_ref, rtol=RTOL, atol=ATOL)


def test_reg_linear_not_stratified():
    """
    Linear model with weights only
    R: svydesign(ids = ~1, weights = ~pw, data = apistrat)
    Stata reference:
      svyset __psu_ids1 [pweight=pw]  (one PSU per obs)
      svy: regress api00 ell meals mobility
    DF = 199
    """
    sample = make_sample(api_strat, weight=WEIGHT_COL)
    res = sample.glm.fit(
        y="api00", x=X_COLS, family=DistFamily.GAUSSIAN, link=LinkFunction.IDENTITY
    )
    coefs = _coef_dict(res)

    assert res.coefs[0].wald.df == 199

    beta = np.array(
        [coefs["_intercept_"].est, coefs["ell"].est, coefs["meals"].est, coefs["mobility"].est]
    )
    se = np.array(
        [coefs["_intercept_"].se, coefs["ell"].se, coefs["meals"].se, coefs["mobility"].se]
    )
    lci = np.array(
        [coefs["_intercept_"].lci, coefs["ell"].lci, coefs["meals"].lci, coefs["mobility"].lci]
    )
    uci = np.array(
        [coefs["_intercept_"].uci, coefs["ell"].uci, coefs["meals"].uci, coefs["mobility"].uci]
    )

    # Stata reference values (TEST 1 linear)
    beta_ref = np.array([820.88732, -0.48058661, -3.1415353, 0.22571321])
    se_ref = np.array([10.970909, 0.39717553, 0.29173325, 0.40124980])
    lci_ref = np.array([799.2532, -1.2637990, -3.7168210, -0.5655340])
    uci_ref = np.array([842.5215, 0.3026263, -2.5662500, 1.0169600])

    np.testing.assert_allclose(beta, beta_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_ref, rtol=RTOL, atol=ATOL)
