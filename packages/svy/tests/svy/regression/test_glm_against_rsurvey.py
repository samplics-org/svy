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
import pytest

from svy.core.enumerations import DistFamily, LinkFunction
from svy.core.sample import Design, Sample
from svy.regression.links import DEFAULT_LINKS, FAMILY_LINKS, resolve_link


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
api_strat = api_strat.with_columns(
    (pl.col("api00") > 743).cast(pl.Float64).alias("y_bin"),
    # Inverse Gaussian needs y > 0; the /100 keeps mu^3 off the overflow edge.
    (pl.col("api00") / 100.0).alias("y_pos"),
)

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
# Probit / cloglog binomial
# ===========================================================================
#
# References from R survey 4.5, same designs as the logit classes above:
#
#   svyglm(y_bin ~ ell + meals + mobility, design = d,
#          family = quasibinomial(link = "probit"),  # or "cloglog"
#          epsilon = 1e-12, maxit = 100)
#
# Both sides are run to epsilon/tol 1e-12 so the comparison measures the link
# implementation rather than each side's default IRLS stopping rule (at the
# defaults the two agree to ~2e-4, well inside RTOL but not informative).

TOL_TIGHT = 1e-12


class TestProbitBinomial:
    """Probit regression: y = y_bin (api00 > 743), probit link, binomial family."""

    Y = "y_bin"
    FAMILY = DistFamily.BINOMIAL
    LINK = LinkFunction.PROBIT

    BETA_R = np.array(
        [1.545373014095, -0.018660537450225, -0.049972554700703, -0.0023892990572185]
    )

    def _fit(self, sample):
        return sample.glm.fit(
            y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK, tol=TOL_TIGHT
        )

    def test_weights_only(self):
        """R: svydesign(ids = ~1, weights = ~pw, data = apistrat)."""
        res = self._fit(make_sample(api_strat, weight=WEIGHT_COL))

        se_r = np.array(
            [0.25946978991497, 0.018266487884698, 0.0081516079905719, 0.008016983440414]
        )
        lci_r = np.array(
            [1.0336619456277, -0.054684631182629, -0.066048676841739, -0.018199922573441]
        )
        uci_r = np.array(
            [2.0570840825623, 0.017363556282178, -0.033896432559667, 0.013421324459004]
        )
        t_r = np.array([5.9558880230391, -1.021572267642, -6.1303922807011, -0.29802968597564])
        p_r = np.array(
            [1.1784663118024e-08, 0.30824263793933, 4.7348158379932e-09, 0.7659961919275]
        )

        _assert_matches(
            res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 20.352391832101, 196
        )

    def test_psu_only(self):
        """R: svydesign(ids = ~dnum, weights = ~pw, data = apistrat)."""
        res = self._fit(make_sample(api_strat, weight=WEIGHT_COL, psu=PSU_COL))

        se_r = np.array(
            [0.27214308685295, 0.017390720280361, 0.0080605999540125, 0.0083338359738581]
        )
        lci_r = np.array(
            [1.0070090714974, -0.053063530533811, -0.065918344179049, -0.018875614560329]
        )
        uci_r = np.array(
            [2.0837369566927, 0.01574245563336, -0.034026765222357, 0.014097016445892]
        )
        t_r = np.array([5.6785312166685, -1.0730169394593, -6.1996073475681, -0.2866985941064])
        p_r = np.array(
            [8.3637500473788e-08, 0.2852361565922, 6.849418092999e-09, 0.77479606024661]
        )

        _assert_matches(
            res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 20.631883090086, 131
        )

    def test_stratified(self):
        """R: svydesign(ids = ~1, strata = ~stype, weights = ~pw, data = apistrat)."""
        res = self._fit(make_sample(api_strat, weight=WEIGHT_COL, stratum=STRATUM_COL))

        se_r = np.array(
            [0.24134839789596, 0.018274386328224, 0.0080182151796345, 0.0080416824193266]
        )
        lci_r = np.array(
            [1.0693694048841, -0.054702516283244, -0.065786620168753, -0.018249648200193]
        )
        uci_r = np.array(
            [2.021376623306, 0.017381441382793, -0.034158489232653, 0.013471050085756]
        )
        t_r = np.array([6.4030796457212, -1.0211307299225, -6.2323788500499, -0.29711432665834])
        p_r = np.array(
            [1.121990994939e-09, 0.30846418231095, 2.8030715513917e-09, 0.76669711262376]
        )

        _assert_matches(
            res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 21.264526553695, 194
        )

    def test_psu_stratified(self):
        """R: svydesign(ids = ~dnum, strata = ~stype, weights = ~pw, nest = TRUE)."""
        res = self._fit(
            make_sample(api_strat, weight=WEIGHT_COL, stratum=STRATUM_COL, psu=PSU_COL)
        )

        se_r = np.array(
            [0.25956921879724, 0.017722011522305, 0.0080214714017624, 0.0083702474482525]
        )
        lci_r = np.array(
            [1.0326491788205, -0.053666605421611, -0.065817266954005, -0.018922944278138]
        )
        uci_r = np.array(
            [2.0580968493696, 0.01634553052116, -0.034127842447401, 0.014144346163701]
        )
        t_r = np.array([5.9536065996415, -1.0529582054915, -6.229848889036, -0.28545142446383])
        p_r = np.array(
            [1.6740597473401e-08, 0.29398835765445, 4.1538142141744e-09, 0.77567696666917]
        )

        _assert_matches(res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 20.55223605927, 156)


class TestCloglogBinomial:
    """Complementary log-log: y = y_bin (api00 > 743), cloglog link, binomial."""

    Y = "y_bin"
    FAMILY = DistFamily.BINOMIAL
    LINK = LinkFunction.CLOGLOG

    BETA_R = np.array(
        [1.3172099219634, -0.029682102446269, -0.060246710875653, 0.0015228433408384]
    )

    def _fit(self, sample):
        return sample.glm.fit(
            y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK, tol=TOL_TIGHT
        )

    def test_weights_only(self):
        """R: svydesign(ids = ~1, weights = ~pw, data = apistrat)."""
        res = self._fit(make_sample(api_strat, weight=WEIGHT_COL))

        se_r = np.array(
            [0.26730883107832, 0.022657612268236, 0.0095946561929446, 0.011203378554088]
        )
        lci_r = np.array(
            [0.79003915727953, -0.074366113584892, -0.079168727861434, -0.020571801327564]
        )
        uci_r = np.array(
            [1.8443806866473, 0.015001908692355, -0.041324693889872, 0.02361748800924]
        )
        t_r = np.array([4.927670801783, -1.3100278217701, -6.2791943415289, 0.1359271521074])
        p_r = np.array(
            [1.7663244203052e-06, 0.19171969797287, 2.1475085684276e-09, 0.89201837440881]
        )

        _assert_matches(res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 23.07629199748, 196)

    def test_psu_only(self):
        """R: svydesign(ids = ~dnum, weights = ~pw, data = apistrat)."""
        res = self._fit(make_sample(api_strat, weight=WEIGHT_COL, psu=PSU_COL))

        se_r = np.array(
            [0.27571564602567, 0.021347176298304, 0.0093168463404812, 0.011489265628425]
        )
        lci_r = np.array(
            [0.77177860512661, -0.071911909303318, -0.07867765537048, -0.021205664712767]
        )
        uci_r = np.array(
            [1.8626412388002, 0.012547704410781, -0.041815766380826, 0.024251351394444]
        )
        t_r = np.array([4.7774217421116, -1.3904463068789, -6.4664274448623, 0.13254488059452])
        p_r = np.array(
            [4.6876313436165e-06, 0.16675112982428, 1.8223991505756e-09, 0.8947567012033]
        )

        _assert_matches(
            res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 24.462693439042, 131
        )

    def test_stratified(self):
        """R: svydesign(ids = ~1, strata = ~stype, weights = ~pw, data = apistrat)."""
        res = self._fit(make_sample(api_strat, weight=WEIGHT_COL, stratum=STRATUM_COL))

        se_r = np.array(
            [0.25201530309238, 0.022662572214824, 0.009565380639981, 0.011244648006955]
        )
        lci_r = np.array(
            [0.82016832196044, -0.074378757830365, -0.079112200548677, -0.020654610838594]
        )
        uci_r = np.array(
            [1.8142515219663, 0.015014552937827, -0.041381221202629, 0.023700297520271]
        )
        t_r = np.array([5.2267060999884, -1.3097411081542, -6.2984122789465, 0.1354282801824])
        p_r = np.array(
            [4.4390662650044e-07, 0.19183230554382, 1.9705800536261e-09, 0.89241368056042]
        )

        _assert_matches(
            res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 23.450872290336, 194
        )

    def test_psu_stratified(self):
        """R: svydesign(ids = ~dnum, strata = ~stype, weights = ~pw, nest = TRUE)."""
        res = self._fit(
            make_sample(api_strat, weight=WEIGHT_COL, stratum=STRATUM_COL, psu=PSU_COL)
        )

        se_r = np.array([0.26447949795533, 0.021755627273, 0.0093349494130275, 0.011578654716537])
        lci_r = np.array(
            [0.79478687360856, -0.072655721220878, -0.07868591983625, -0.021348328676749]
        )
        uci_r = np.array(
            [1.8396329703182, 0.013291516328341, -0.041807501915056, 0.024394015358426]
        )
        t_r = np.array([4.9803857468978, -1.3643413758566, -6.4538872370936, 0.13152161266743])
        p_r = np.array(
            [1.665788341527e-06, 0.17442512814572, 1.3072836423164e-09, 0.89553215933927]
        )

        _assert_matches(res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 23.6993739904, 156)


# ===========================================================================
# Inverse Gaussian
# ===========================================================================
#
# R: svyglm(y ~ ell + meals + mobility, design = d,
#           family = inverse.gaussian(link = "1/mu^2"), epsilon = 1e-12)
# y = api00 / 100 keeps mu^3 in the variance away from overflow; the family
# requires a strictly positive response either way.


class TestInverseGaussian:
    """Inverse Gaussian with its canonical 1/mu^2 link."""

    Y = "y_pos"
    FAMILY = DistFamily.INVERSE_GAUSSIAN
    LINK = LinkFunction.INVERSE_SQUARED

    BETA_R = np.array(
        [0.013143949502326, 7.1589585380775e-05, 0.00020973893496085, -2.647812390399e-05]
    )

    def _fit(self, sample):
        return sample.glm.fit(
            y=self.Y, x=X_COLS, family=self.FAMILY, link=self.LINK, tol=TOL_TIGHT
        )

    def test_weights_only(self):
        """R: svydesign(ids = ~1, weights = ~pw, data = apistrat)."""
        res = self._fit(make_sample(api_strat, weight=WEIGHT_COL))

        se_r = np.array(
            [
                0.00055210204845821,
                3.9958532946854e-05,
                2.3236733637848e-05,
                2.7001864330128e-05,
            ]
        )
        lci_r = np.array(
            [
                0.012055126293998,
                -7.2142846008576e-06,
                0.00016391281469687,
                -7.9729613611161e-05,
            ]
        )
        uci_r = np.array(
            [0.014232772710655, 0.00015039345536241, 0.00025556505522483, 2.6773365803181e-05]
        )
        t_r = np.array([23.807101493341, 1.7915969406582, 9.0261797647508, -0.98060354575023])
        p_r = np.array(
            [9.6619834661281e-60, 0.074740289532756, 1.6667313734843e-16, 0.32799713290278]
        )

        _assert_matches(
            res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 94.614459019273, 196
        )

    def test_psu_only(self):
        """R: svydesign(ids = ~dnum, weights = ~pw, data = apistrat)."""
        res = self._fit(make_sample(api_strat, weight=WEIGHT_COL, psu=PSU_COL))

        se_r = np.array(
            [
                0.00059593883638442,
                4.6664895084989e-05,
                2.5463858080353e-05,
                2.6865574975381e-05,
            ]
        )
        lci_r = np.array(
            [
                0.011965040329166,
                -2.0724708505491e-05,
                0.0001593653495488,
                -7.9624639698871e-05,
            ]
        )
        uci_r = np.array(
            [0.014322858675486, 0.00016390387926704, 0.00026011252037289, 2.6668391890891e-05]
        )
        t_r = np.array([22.055869998457, 1.5341207828795, 8.2367304396294, -0.98557815822867])
        p_r = np.array(
            [6.1618107219953e-46, 0.12741155553028, 1.5798749855106e-13, 0.32615699383655]
        )

        _assert_matches(
            res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 87.906076353248, 131
        )

    def test_stratified(self):
        """R: svydesign(ids = ~1, strata = ~stype, weights = ~pw, data = apistrat)."""
        res = self._fit(make_sample(api_strat, weight=WEIGHT_COL, stratum=STRATUM_COL))

        se_r = np.array(
            [
                0.00052941933301628,
                4.0033499252727e-05,
                2.3180190473814e-05,
                2.7117370793504e-05,
            ]
        )
        lci_r = np.array(
            [0.012099792940697, -7.36718541103e-06, 0.00016402139789027, -7.9960833889008e-05]
        )
        uci_r = np.array(
            [0.014188106063956, 0.00015054635617258, 0.00025545647203142, 2.7004586081028e-05]
        )
        t_r = np.array([24.827105250276, 1.7882420152392, 9.0481972181284, -0.97642666413417])
        p_r = np.array(
            [3.8895476043754e-62, 0.075297564833711, 1.5297502352673e-16, 0.33006906213443]
        )

        _assert_matches(
            res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 95.342259067271, 194
        )

    def test_psu_stratified(self):
        """R: svydesign(ids = ~dnum, strata = ~stype, weights = ~pw, nest = TRUE)."""
        res = self._fit(
            make_sample(api_strat, weight=WEIGHT_COL, stratum=STRATUM_COL, psu=PSU_COL)
        )

        se_r = np.array(
            [
                0.00051603914573776,
                4.3436147235601e-05,
                2.2450703160116e-05,
                2.580499617031e-05,
            ]
        )
        lci_r = np.array(
            [0.012124623824264, -1.4209293636476e-05, 0.00016539234146951, -7.745041047554e-05]
        )
        uci_r = np.array(
            [0.014163275180388, 0.00015738846439802, 0.00025408552845218, 2.4494162667559e-05]
        )
        t_r = np.array([25.470838037946, 1.6481568909063, 9.3421989264663, -1.026085170842])
        p_r = np.array(
            [1.8724230164209e-57, 0.10133307019429, 9.3407316166739e-17, 0.30643985383466]
        )

        _assert_matches(
            res, TERMS, self.BETA_R, se_r, lci_r, uci_r, t_r, p_r, 109.03941380745, 156
        )

    def test_canonical_link_is_default(self):
        """family alone resolves to 1/mu^2, and the label is not "Inversegaussian"."""
        res = make_sample(api_strat, weight=WEIGHT_COL).glm.fit(
            y=self.Y, x=X_COLS, family=self.FAMILY, tol=TOL_TIGHT
        )

        assert res.fitted.link == "inverse_squared"
        assert res.fitted.family == "InverseGaussian"
        np.testing.assert_allclose([c.est for c in res.coefs], self.BETA_R, rtol=RTOL, atol=ATOL)


# ===========================================================================
# Family / link compatibility
# ===========================================================================


class TestFamilyLinkCompatibility:
    """
    Every family admits exactly the links R's family constructors do, minus the
    two R has that svy does not implement (cauchit, sqrt).
    """

    # Read off `okLinks` in R: gaussian, binomial, poisson, Gamma and
    # inverse.gaussian, with R's "1/mu^2" spelled "inverse_squared".
    R_OK_LINKS = {
        "gaussian": {"inverse", "log", "identity"},
        "binomial": {"logit", "probit", "cloglog", "log", "identity"},
        "poisson": {"log", "identity"},
        "gamma": {"inverse", "log", "identity"},
        "inversegaussian": {"inverse", "log", "identity", "inverse_squared"},
    }

    def test_table_matches_r(self):
        assert {k: set(v) for k, v in FAMILY_LINKS.items()} == self.R_OK_LINKS

    def test_every_family_admits_its_default_link(self):
        for family, link in DEFAULT_LINKS.items():
            assert link in FAMILY_LINKS[family], f"{family} rejects its own default {link}"

    @pytest.mark.parametrize("family", sorted(R_OK_LINKS))
    @pytest.mark.parametrize(
        "link", ["identity", "logit", "probit", "cloglog", "log", "inverse", "inverse_squared"]
    )
    def test_resolve_link_follows_the_table(self, family, link):
        if link in self.R_OK_LINKS[family]:
            assert resolve_link(family, link) == link
        else:
            with pytest.raises(ValueError, match="does not admit"):
                resolve_link(family, link)

    @pytest.mark.parametrize(
        ("family", "link"),
        [("gaussian", "probit"), ("poisson", "cloglog"), ("binomial", "inverse_squared")],
    )
    def test_fit_rejects_unusable_pairing(self, family, link):
        """binomial + inverse_squared used to converge on a meaningless fit."""
        sample = make_sample(api_strat, weight=WEIGHT_COL)
        with pytest.raises(ValueError, match="does not admit"):
            sample.glm.fit(y="y_bin", x=X_COLS, family=family, link=link)


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
