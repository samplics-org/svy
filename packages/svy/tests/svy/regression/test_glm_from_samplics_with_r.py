# tests/svy/regression/test_glm_from_samplics_with_R.py
from pathlib import Path

import numpy as np
import polars as pl

from svy.core.enumerations import DistFamily, LinkFunction
from svy.core.sample import Design, Sample


DATA_DIR = Path(__file__).parent.parent.parent / "test_data"

# Strict tolerance for all comparisons
RTOL = 1e-3
ATOL = 1e-5


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


def test_reg_logistic_not_stratified():
    """
    Weights only (no strata, no PSU)
    R: svydesign(ids = ~1, weights = ~pw, data = apistrat)
    DF = 199
    """
    sample = make_sample(api_strat, weight=WEIGHT_COL)
    res = sample.glm.fit(y="y_bin", x=X_COLS, family=DistFamily.BINOMIAL, link=LinkFunction.LOGIT)
    coefs = _coef_dict(res)

    # Check DF
    assert res.coefs[0].wald.df == 199

    # Extract results
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

    # R reference values
    beta_r = np.array([2.665755993, -0.037032496, -0.084687878, -0.004428592])
    se_r = np.array([0.47181967, 0.03363664, 0.01460217, 0.01478821])
    lci_r = np.array([1.73526097, -0.10336871, -0.11348542, -0.03359304])
    uci_r = np.array([3.59625101, 0.02930371, -0.05589034, 0.02473585])

    np.testing.assert_allclose(beta, beta_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_r, rtol=RTOL, atol=ATOL)


def test_reg_logistic_psu_not_stratified():
    """
    PSU only (no strata)
    R: svydesign(ids = ~dnum, weights = ~pw, data = apistrat)
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

    beta_r = np.array([2.665755993, -0.037032496, -0.084687878, -0.004428592])
    se_r = np.array([0.49495054, 0.03182633, 0.01438574, 0.01537255])
    lci_r = np.array([1.68662577, -0.09999256, -0.11314631, -0.03483916])
    uci_r = np.array([3.64488622, 0.02592757, -0.05622945, 0.02598197])

    np.testing.assert_allclose(beta, beta_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_r, rtol=RTOL, atol=ATOL)


def test_reg_logistic_stratified():
    """
    Strata only (no PSU)
    R: svydesign(ids = ~1, strata = ~stype, weights = ~pw, data = apistrat)
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

    beta_r = np.array([2.665755993, -0.037032496, -0.084687878, -0.004428592])
    se_r = np.array([0.43943188, 0.03363983, 0.01433106, 0.01483631])
    lci_r = np.array([1.79907877, -0.10337924, -0.11295256, -0.03368976])
    uci_r = np.array([3.53243322, 0.02931425, -0.05642319, 0.02483257])

    np.testing.assert_allclose(beta, beta_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_r, rtol=RTOL, atol=ATOL)


def test_reg_logistic_psu_stratified():
    """
    PSU + Strata (nested)
    R: svydesign(ids = ~dnum, strata = ~stype, weights = ~pw, data = apistrat, nest = TRUE)
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

    beta_r = np.array([2.665755993, -0.037032496, -0.084687878, -0.004428592])
    se_r = np.array([0.47238789, 0.03242197, 0.01428410, 0.01546279])
    lci_r = np.array([1.73265409, -0.10107520, -0.11290309, -0.03497204])
    uci_r = np.array([3.59885790, 0.02701021, -0.05647267, 0.02611486])

    np.testing.assert_allclose(beta, beta_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_r, rtol=RTOL, atol=ATOL)


def test_reg_linear_not_stratified():
    """
    Linear model with weights only
    R: svydesign(ids = ~1, weights = ~pw, data = apistrat)
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

    beta_r = np.array([820.8873159, -0.4805866, -3.1415353, 0.2257132])
    se_r = np.array([10.9709091, 0.3971755, 0.2917333, 0.4012498])
    lci_r = np.array([799.2511339, -1.2638728, -3.7168745, -0.5656081])
    uci_r = np.array([842.5234980, 0.3026996, -2.5661961, 1.0170345])

    np.testing.assert_allclose(beta, beta_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_r, rtol=RTOL, atol=ATOL)
