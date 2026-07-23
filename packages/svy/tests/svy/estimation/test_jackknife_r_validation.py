# tests/svy/estimation/test_jackknife_r_validation.py
"""
End-to-end validation of svy-GENERATED stratified jackknife (JKn) weights
against R's survey package.

The pre-existing golden replication tests use user-supplied replicate
weights and therefore never exercised create_jk_wgts' own scale/df
conventions — which is how a global (R-1)/R scale (exactly 7/6 off vs R
on this design), a total-PSU df, and an ignored variance_center survived.

Reference values from R survey 4.5 on domain_nulls_20260722.csv
(2 strata x 4 clusters, income null for non-employed + 2 MAR nulls):

    options(digits = 15)
    library(survey)
    d <- read.csv("tests/test_data/domain_nulls_20260722.csv")
    des <- svydesign(id = ~cluster, strata = ~stratum, weights = ~wgt, data = d)
    rdes     <- as.svrepdesign(des, type = "JKn")
    rdes_mse <- as.svrepdesign(des, type = "JKn", mse = TRUE)
    degf(rdes)                                # 6
    svymean(~income, rdes, na.rm = TRUE)      # 42949.3221906951, SE 1718.17426740466
    svymean(~income, rdes_mse, na.rm = TRUE)  # SE 1720.80068178043
    svytotal(~income, rdes, na.rm = TRUE)     # 114554002.6538, SE 18830996.1797964
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from svy import Design, Sample


BASE_DIR = Path(__file__).parents[2]

REL = 1e-9


@pytest.fixture
def jk_sample() -> Sample:
    df = pl.read_csv(BASE_DIR / "test_data" / "domain_nulls_20260722.csv")
    s = Sample(df, Design(stratum="stratum", psu="cluster", wgt="wgt"))
    return s.weighting.create_jk_wgts()


class TestGeneratedJknMatchesR:
    def test_df_and_rscales(self, jk_sample):
        rw = jk_sample.design.rep_wgts
        assert rw.n_reps == 8
        # Stratified jackknife df = #PSUs - #strata (R degf; was total PSUs)
        assert rw.df == 6
        # (n_h - 1)/n_h per deleted-PSU replicate; 4 clusters per stratum
        assert rw.rscales == (0.75,) * 8

    def test_mean_matches_r(self, jk_sample):
        r = jk_sample.estimation.mean("income", method="replication", drop_nulls=True)
        e = r.estimates[0]
        assert e.est == pytest.approx(42949.3221906951, rel=REL)
        # Global (R-1)/R scale gave 1855.84 here — exactly sqrt(7/6) off
        assert e.se == pytest.approx(1718.17426740466, rel=REL)

    def test_mse_centering_matches_r(self, jk_sample):
        """variance_center='estimate' == R's mse=TRUE. The jackknife branch
        previously ignored the parameter and always centered on the
        replicate mean."""
        r = jk_sample.estimation.mean(
            "income", method="replication", drop_nulls=True, variance_center="estimate"
        )
        assert r.estimates[0].se == pytest.approx(1720.80068178043, rel=REL)

    def test_total_matches_r(self, jk_sample):
        t = jk_sample.estimation.total("income", method="replication", drop_nulls=True)
        assert t.estimates[0].est == pytest.approx(114554002.6538, rel=1e-9)
        assert t.estimates[0].se == pytest.approx(18830996.1797964, rel=REL)

    def test_user_rscales_override(self, jk_sample):
        """User-supplied rscales flow through: doubling them scales the
        variance by 2 (SE by sqrt 2)."""
        import math

        base = jk_sample.estimation.mean("income", method="replication", drop_nulls=True)
        rw = jk_sample.design.rep_wgts
        boosted = Sample(
            jk_sample.data,
            jk_sample.design.update_rep_weights(
                method=rw.method,
                prefix=rw.prefix,
                n_reps=rw.n_reps,
                df=rw.df,
                rscales=tuple(2.0 * r for r in rw.rscales),
            ),
        )
        r2 = boosted.estimation.mean("income", method="replication", drop_nulls=True)
        assert r2.estimates[0].se == pytest.approx(
            base.estimates[0].se * math.sqrt(2.0), rel=1e-12
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
