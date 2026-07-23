# tests/svy/estimation/test_domain_nulls.py
"""
Domain estimation with missing y: drop_nulls=True must zero weights, not
drop rows.

Dataset: domain_nulls_20260722.csv (see tests/test_data/domain_nulls_test_data.py).
Two strata x 4 clusters x 6 units; income is null for every non-employed
respondent (standard skip pattern) plus two missing-at-random nulls among
the employed. Cluster 104 (stratum A) is entirely non-employed, so
physically dropping null-income rows deletes that whole PSU — the classic
subpopulation-estimation error. Weight-zeroing keeps the design intact.

Reference values from R survey 4.5:

    options(digits = 15)
    library(survey)
    d <- read.csv("tests/test_data/domain_nulls_20260722.csv")
    des <- svydesign(id = ~cluster, strata = ~stratum, weights = ~wgt, data = d)

    # Taylor, domain = employed (2 MAR nulls handled by na.rm)
    sub <- subset(des, employed == 1)
    svymean(~income, sub, na.rm = TRUE)   # 42949.3221906951, SE 1715.36393099045
    svytotal(~income, sub, na.rm = TRUE)  # 114554002.6538, SE 18830996.1797964
    svyby(~income, ~stratum, sub, svymean, na.rm = TRUE)
    #   A 37126.1943347756 (SE 1918.28005228556)
    #   B 48400.8401814090 (SE 2161.10028916013)

    # na.rm over the full design is the same domain (all nulls out)
    svymean(~income, des, na.rm = TRUE)   # identical to the subset result

    # JK1 replicate weights (as.svrepdesign(type="JK1") on the unstratified
    # cluster design; exported as rep1..rep8 in domain_nulls_jk1_20260722.csv)
    des1  <- svydesign(id = ~cluster, weights = ~wgt, data = d)
    rdes1 <- as.svrepdesign(des1, type = "JK1")
    svymean(~income, subset(rdes1, employed == 1), na.rm = TRUE)
    # 42949.3221906951, SE 2746.52964802544

    # What physically dropping null-income rows produces (the old bug):
    dcc <- d[!is.na(d$income), ]
    descc <- svydesign(id = ~cluster, strata = ~stratum, weights = ~wgt, data = dcc)
    svymean(~income, subset(descc, employed == 1))  # SE 1494.07160271379 (15% understated)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from svy import Design, RepWeights, Sample


BASE_DIR = Path(__file__).parents[2]

REL = 1e-9


@pytest.fixture
def data() -> pl.DataFrame:
    return pl.read_csv(BASE_DIR / "test_data" / "domain_nulls_20260722.csv")


@pytest.fixture
def sample(data) -> Sample:
    return Sample(data, Design(stratum="stratum", psu="cluster", wgt="wgt"))


class TestTaylorDomainNulls:
    def test_domain_mean_keeps_design(self, sample):
        r = sample.estimation.mean("income", where=pl.col("employed") == 1, drop_nulls=True)
        e = r.estimates[0]
        assert e.est == pytest.approx(42949.3221906951, rel=REL)
        assert e.se == pytest.approx(1715.36393099045, rel=REL)
        # The old row-dropping behavior produced SE 1494.07160271379
        assert e.se != pytest.approx(1494.07160271379, rel=1e-3)

    def test_narm_mean_equals_domain(self, sample):
        """All nulls are out-of-domain, so plain drop_nulls matches R na.rm."""
        r = sample.estimation.mean("income", drop_nulls=True)
        e = r.estimates[0]
        assert e.est == pytest.approx(42949.3221906951, rel=REL)
        assert e.se == pytest.approx(1715.36393099045, rel=REL)

    def test_domain_total(self, sample):
        r = sample.estimation.total("income", where=pl.col("employed") == 1, drop_nulls=True)
        e = r.estimates[0]
        assert e.est == pytest.approx(114554002.6538, rel=1e-9)
        assert e.se == pytest.approx(18830996.1797964, rel=REL)

    def test_domain_mean_by_stratum(self, sample):
        r = sample.estimation.mean(
            "income", by="stratum", where=pl.col("employed") == 1, drop_nulls=True
        )
        by_levels = {str(e.by_level[0]): e for e in r.estimates}
        assert by_levels["A"].est == pytest.approx(37126.1943347756, rel=REL)
        assert by_levels["A"].se == pytest.approx(1918.28005228556, rel=REL)
        assert by_levels["B"].est == pytest.approx(48400.8401814090, rel=REL)
        assert by_levels["B"].se == pytest.approx(2161.10028916013, rel=REL)


class TestReplicateDomainNulls:
    """Replicate weights must be zeroed alongside the main weight."""

    @pytest.fixture
    def jk1_sample(self) -> Sample:
        df = pl.read_csv(BASE_DIR / "test_data" / "domain_nulls_jk1_20260722.csv")
        rw = RepWeights(method="jackknife", prefix="rep", n_reps=8)
        return Sample(df, Design(psu="cluster", wgt="wgt", rep_wgts=rw))

    def test_jk1_domain_mean(self, jk1_sample):
        r = jk1_sample.estimation.mean(
            "income", where=pl.col("employed") == 1, method="replication", drop_nulls=True
        )
        e = r.estimates[0]
        assert e.est == pytest.approx(42949.3221906951, rel=REL)
        assert e.se == pytest.approx(2746.52964802544, rel=REL)

    def test_jk1_narm_equals_domain(self, jk1_sample):
        r = jk1_sample.estimation.mean("income", method="replication", drop_nulls=True)
        e = r.estimates[0]
        assert e.est == pytest.approx(42949.3221906951, rel=REL)
        assert e.se == pytest.approx(2746.52964802544, rel=REL)


class TestDesignIntegrity:
    def test_point_estimate_unchanged(self, sample, data):
        """Weight-zeroing and row-dropping agree on the point estimate."""
        r = sample.estimation.mean("income", where=pl.col("employed") == 1, drop_nulls=True)
        cc = data.filter(pl.col("income").is_not_null() & (pl.col("employed") == 1))
        manual = (cc["income"] * cc["wgt"]).sum() / cc["wgt"].sum()
        assert r.estimates[0].est == pytest.approx(manual, rel=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
