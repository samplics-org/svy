# tests/svy/estimation/test_jackknife_fixture_jkn.py
"""
Proper stratified-JKn validation of the jackknife golden fixture.

The golden tests in test_replication.py run this fixture WITHOUT rscales
(the documented global (R-1)/R fallback). Here the same delete-one-PSU
weights are used with the JKn-correct per-replicate coefficients
rscales = (n_h-1)/n_h = 0.5 (4 strata x 2 PSUs), validated against
R survey 4.5 (2026-07-23):

  api <- read.csv("fake_survey_jackknife_25122025.csv")
  api <- api[complete.cases(api[, c("income","sex")]), ]
  api$low_income <- as.integer(api$income < 40000)
  api$hh_size <- (api$id %% 4) + 1
  rd <- svrepdesign(repweights=jk_1..8, weights=api$weight, type="JKn",
                    scale=1, rscales=rep(0.5, 8), combined.weights=TRUE)
  svymean(~income, rd)            # 54687.648909 (SE 1008.4721693)
  svytotal(~low_income, rd)       # 23.1428636   (SE 2.8326446)
  svyratio(~income, ~hh_size, rd) # 21945.594958 (SE 489.9267289)

The no-rscales SEs are exactly sqrt(7/4) larger (scale 7/8 vs 1/2).
"""

import numpy as np
import pytest

from svy import EstimationMethod, RepWeights, Sample
from svy.core.sample import Design


@pytest.fixture
def jkn_sample(load_survey_data):
    data = load_survey_data("fake_survey_jackknife_25122025.csv")
    rep_weights = RepWeights(
        method=EstimationMethod.JACKKNIFE,
        prefix="jk_",
        n_reps=8,
        df=7,
        rscales=(0.5,) * 8,
    )
    design = Design(
        row_index="id", wgt="weight", stratum="stratum", psu="psu", rep_wgts=rep_weights
    )
    return Sample(data, design)


def test_mean_matches_r_jkn(jkn_sample):
    est = jkn_sample.estimation.mean(y="income", method="replication", drop_nulls=True)
    res = est.estimates[0]
    np.testing.assert_allclose(res.est, 54687.648909, rtol=1e-8)
    np.testing.assert_allclose(res.se, 1008.4721693, rtol=1e-7)


def test_total_matches_r_jkn(jkn_sample):
    est = jkn_sample.estimation.total(y="low_income", method="replication", drop_nulls=True)
    res = est.estimates[0]
    np.testing.assert_allclose(res.est, 23.1428636, rtol=1e-7)
    np.testing.assert_allclose(res.se, 2.8326446, rtol=1e-7)


def test_ratio_matches_r_jkn(jkn_sample):
    est = jkn_sample.estimation.ratio(
        y="income", x="hh_size", method="replication", drop_nulls=True
    )
    res = est.estimates[0]
    np.testing.assert_allclose(res.est, 21945.594958, rtol=1e-8)
    np.testing.assert_allclose(res.se, 489.9267289, rtol=1e-7)


def test_jkn_se_is_sqrt_7_4_below_global_scale(load_survey_data, jkn_sample):
    """The documented no-rscales fallback is exactly sqrt(7/4) larger."""
    data = load_survey_data("fake_survey_jackknife_25122025.csv")
    rw_global = RepWeights(
        method=EstimationMethod.JACKKNIFE, prefix="jk_", n_reps=8, df=7
    )
    design = Design(
        row_index="id", wgt="weight", stratum="stratum", psu="psu", rep_wgts=rw_global
    )
    s_global = Sample(data, design)
    se_g = s_global.estimation.mean(y="income", method="replication", drop_nulls=True)
    se_j = jkn_sample.estimation.mean(y="income", method="replication", drop_nulls=True)
    ratio = se_g.estimates[0].se / se_j.estimates[0].se
    np.testing.assert_allclose(ratio, np.sqrt(7.0 / 4.0), rtol=1e-10)
