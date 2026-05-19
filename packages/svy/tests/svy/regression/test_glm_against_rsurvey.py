# tests/svy/regression/test_glm_against_rsurvey.py
"""
GLM tests against R's svyglm as the oracle.

Organization: one test class per family. Within each class, one test per
design configuration.

Reference values come from tests/test_data/from_samplics.R, which uses
R's *default* residual df for svyglm — i.e. df.residual(fit), which is
degf(design) - (k - 1) where k - 1 is the number of non-intercept
parameters. This is the standard regression convention.

Assertions cover: coefficients, standard errors, CIs, t-statistics,
per-coefficient p-values, the joint Wald F-statistic, and the residual df.
"""

from pathlib import Path

import numpy as np
import polars as pl

from svy.core.enumerations import DistFamily, LinkFunction
from svy.core.sample import Design, Sample


DATA_DIR = Path(__file__).parent.parent.parent / "test_data"

# Strict tolerance. Once df convention matches R, every reported quantity
# should agree at this level.
RTOL = 1e-3
ATOL = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sample(
    df: pl.DataFrame,
    *,
    weight: str | None = None,
    stratum: str | None = None,
    psu: str | None = None,
):
    if weight is None and stratum is None and psu is None:
        design = None
    else:
        design = Design(wgt=weight, stratum=stratum, psu=psu)
    return Sample(data=df, design=design)


def _coef_dict(result):
    return {c.term: c for c in result.coefs}


def _arrays(result, order):
    coefs = _coef_dict(result)
    beta = np.array([coefs[t].est for t in order])
    se = np.array([coefs[t].se for t in order])
    lci = np.array([coefs[t].lci for t in order])
    uci = np.array([coefs[t].uci for t in order])
    t_stat = np.array([coefs[t].wald.value for t in order])
    p_val = np.array([coefs[t].wald.p_value for t in order])
    return beta, se, lci, uci, t_stat, p_val


def _assert_matches(
    result,
    order,
    beta_r,
    se_r,
    lci_r,
    uci_r,
    t_r,
    p_r,
    f_r,
    df_resid_r,
):
    beta, se, lci, uci, t_stat, p_val = _arrays(result, order)

    np.testing.assert_allclose(beta, beta_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(se, se_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lci, lci_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(uci, uci_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(t_stat, t_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(p_val, p_r, rtol=RTOL, atol=ATOL)

    assert result.stats.wald is not None
    np.testing.assert_allclose(result.stats.wald.value, f_r, rtol=RTOL, atol=ATOL)

    assert result.coefs[0].wald.df == df_resid_r, (
        f"residual df mismatch: svy={result.coefs[0].wald.df}, R={df_resid_r}"
    )


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

api_strat = pl.read_csv(DATA_DIR / "apistrat.csv")
api_strat = api_strat.with_columns((pl.col("api00") > 743).cast(pl.Float64).alias("y_bin"))

X_COLS = ["ell", "meals", "mobility"]
TERMS = ["_intercept_", "ell", "meals", "mobility"]

STRATUM_COL = "stype"
PSU_COL = "dnum"
WEIGHT_COL = "pw"


# ===========================================================================
# Linear / Gaussian
# ===========================================================================


class TestLinearGaussian:
    """Linear regression: y = api00, identity link, gaussian family."""

    Y = "api00"
    FAMILY = DistFamily.GAUSSIAN
    LINK = LinkFunction.IDENTITY

    def test_weights_only(self):
        """
        R: svydesign(ids = ~1, weights = ~pw, data = apistrat)
        df.residual = degf(199) - (k-1=3) = 196.
        """
        sample = make_sample(api_strat, weight=WEIGHT_COL)
        res = sample.glm.fit(y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK)

        beta_r = np.array([820.887315905620, -0.480586612172, -3.141535309985, 0.225713210230])
        se_r = np.array([10.970909087305, 0.397175526628, 0.291733250884, 0.401249796712])
        lci_r = np.array([799.25113385544, -1.26387284047, -3.71687447978, -0.56560805405])
        uci_r = np.array([842.523497955800, 0.302699616126, -2.566196140186, 1.017034474509])
        t_r = np.array([74.824001308655, -1.210010637494, -10.768519873756, 0.562525419525])
        p_r = np.array(
            [
                4.23857675584e-146,
                2.27732109015e-01,
                1.53384850497e-21,
                5.74400906432e-01,
            ]
        )
        f_r = 127.9294
        df_resid_r = 196

        _assert_matches(res, TERMS, beta_r, se_r, lci_r, uci_r, t_r, p_r, f_r, df_resid_r)


# ===========================================================================
# Logistic / Binomial
# ===========================================================================


class TestLogisticBinomial:
    """Logistic regression: y = y_bin (api00 > 743), logit link, binomial family."""

    Y = "y_bin"
    FAMILY = DistFamily.BINOMIAL
    LINK = LinkFunction.LOGIT

    def test_weights_only(self):
        """
        R: svydesign(ids = ~1, weights = ~pw, data = apistrat)
        df.residual = degf(199) - 3 = 196.
        """
        sample = make_sample(api_strat, weight=WEIGHT_COL)
        res = sample.glm.fit(y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK)

        beta_r = np.array(
            [2.66575599278362, -0.03703249576700, -0.08468787757247, -0.00442859225861]
        )
        se_r = np.array([0.4718196688815, 0.0336366431167, 0.0146021693674, 0.0147882129185])
        lci_r = np.array([1.735260974591, -0.103368706216, -0.113485417708, -0.033593036550])
        uci_r = np.array([3.5962510109757, 0.0293037146817, -0.0558903374373, 0.0247358520327])
        t_r = np.array([5.649946724568, -1.100956942658, -5.799677804144, -0.299467710062])
        p_r = np.array(
            [
                5.59248119211e-08,
                2.72265633197e-01,
                2.62748928676e-08,
                7.64900528326e-01,
            ]
        )
        f_r = 17.90289
        df_resid_r = 196

        _assert_matches(res, TERMS, beta_r, se_r, lci_r, uci_r, t_r, p_r, f_r, df_resid_r)

    def test_psu_only(self):
        """
        R: svydesign(ids = ~dnum, weights = ~pw, data = apistrat)
        df.residual = degf(134) - 3 = 131.
        """
        sample = make_sample(api_strat, weight=WEIGHT_COL, psu=PSU_COL)
        res = sample.glm.fit(y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK)

        beta_r = np.array(
            [2.66575599278362, -0.03703249576700, -0.08468787757247, -0.00442859225861]
        )
        se_r = np.array([0.4949505366073, 0.0318263271683, 0.0143857416874, 0.0153725477059])
        lci_r = np.array([1.6866257662542, -0.0999925627336, -0.1131463061938, -0.0348391585765])
        uci_r = np.array([3.6448862193130, 0.0259275711996, -0.0562294489511, 0.0259819740593])
        t_r = np.array([5.385903834060, -1.163580565584, -5.886931616945, -0.288084470014])
        p_r = np.array(
            [
                3.23298314471e-07,
                2.46709009862e-01,
                3.11757045250e-08,
                7.73737368990e-01,
            ]
        )
        f_r = 18.2147
        df_resid_r = 131

        _assert_matches(res, TERMS, beta_r, se_r, lci_r, uci_r, t_r, p_r, f_r, df_resid_r)

    def test_stratified(self):
        """
        R: svydesign(ids = ~1, strata = ~stype, weights = ~pw, data = apistrat)
        df.residual = degf(197) - 3 = 194.
        """
        sample = make_sample(api_strat, weight=WEIGHT_COL, stratum=STRATUM_COL)
        res = sample.glm.fit(y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK)

        beta_r = np.array(
            [2.66575599278362, -0.03703249576700, -0.08468787757247, -0.00442859225861]
        )
        se_r = np.array([0.4394318774792, 0.0336398300623, 0.0143310610689, 0.0148363068282])
        lci_r = np.array([1.7990787680499, -0.1033792404005, -0.1129525640262, -0.0336897585194])
        uci_r = np.array([3.5324332175174, 0.0293142488665, -0.0564231911188, 0.0248325740021])
        t_r = np.array([6.066369167562, -1.100852641003, -5.909393391399, -0.298496944684])
        p_r = np.array(
            [
                6.72854969843e-09,
                2.72324819493e-01,
                1.51882026548e-08,
                7.65643382623e-01,
            ]
        )
        f_r = 18.75483
        df_resid_r = 194

        _assert_matches(res, TERMS, beta_r, se_r, lci_r, uci_r, t_r, p_r, f_r, df_resid_r)

    def test_psu_stratified(self):
        """
        R: svydesign(ids = ~dnum, strata = ~stype, weights = ~pw, data = apistrat,
                     nest = TRUE)
        df.residual = degf(159) - 3 = 156. Headline test: full design.
        """
        sample = make_sample(
            api_strat,
            weight=WEIGHT_COL,
            stratum=STRATUM_COL,
            psu=PSU_COL,
        )
        res = sample.glm.fit(y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK)

        beta_r = np.array(
            [2.66575599278362, -0.03703249576700, -0.08468787757247, -0.00442859225861]
        )
        se_r = np.array([0.4723878944699, 0.0324219677485, 0.0142841040197, 0.0154627861042])
        lci_r = np.array([1.7326540860469, -0.1010752036358, -0.1129030898012, -0.0349720404846])
        uci_r = np.array([3.5988578995203, 0.0270102121018, -0.0564726653437, 0.0261148559674])
        t_r = np.array([5.643150520983, -1.142203830880, -5.928819718477, -0.286403254159])
        p_r = np.array(
            [
                7.67036694615e-08,
                2.55119387199e-01,
                1.89371130877e-08,
                7.74949284984e-01,
            ]
        )
        f_r = 18.21718
        df_resid_r = 156

        _assert_matches(res, TERMS, beta_r, se_r, lci_r, uci_r, t_r, p_r, f_r, df_resid_r)


# ===========================================================================
# Poisson (placeholder for next iteration)
# ===========================================================================
#
# class TestPoisson:
#     Y = "enroll"
#     FAMILY = DistFamily.POISSON
#     LINK = LinkFunction.LOG
#     ...


# ===========================================================================
# Gamma (placeholder)
# ===========================================================================
#
# class TestGamma:
#     ...
