# tests/svy/estimation/test_sdr_validation.py
"""
SDR (Successive Difference Replication) validation tests.
Results should match R's survey package with type="ACS".

The estimation path is:
    sample.estimation.mean/total/prop/ratio(...)
    → _replicate_*_polars(...)
    → rs.replicate_*(...)  [Rust backend]
"""

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.enumerations import EstimationMethod


@pytest.fixture(scope="module")
def sdr_data():
    """Load SDR test data."""
    return pl.read_csv("tests/test_data/sdr_test_data.csv")


@pytest.fixture(scope="module")
def sdr_sample(sdr_data):
    """Create svy Sample with SDR design."""
    rep_weights = svy.RepWeights(
        method=EstimationMethod.SDR,
        prefix="repwtp",
        n_reps=80,
        df=79,  # ACS standard: n_reps - 1
    )
    design = svy.Design(wgt="pwgtp", rep_wgts=rep_weights)
    return svy.Sample(data=sdr_data, design=design)


class TestSDRMean:
    """Test SDR mean estimation."""

    def test_mean_income(self, sdr_sample):
        """Mean income estimation."""
        result = sdr_sample.estimation.mean("income")

        assert result is not None
        assert len(result.estimates) == 1

        est = result.estimates[0]
        assert est.est > 0
        assert est.se > 0
        assert est.lci < est.est < est.uci

    def test_mean_age(self, sdr_sample):
        """Mean age estimation."""
        result = sdr_sample.estimation.mean("age")

        est = result.estimates[0]
        assert 30 < est.est < 60  # Reasonable age range
        assert est.se > 0


class TestSDRTotal:
    """Test SDR total estimation."""

    def test_total_income(self, sdr_sample):
        """Total income estimation."""
        result = sdr_sample.estimation.total("income")

        est = result.estimates[0]
        assert est.est > 0
        assert est.se > 0


class TestSDRProportion:
    """Test SDR proportion estimation."""

    def test_prop_employed(self, sdr_sample):
        """Proportion employed estimation."""
        result = sdr_sample.estimation.prop("employed")

        # Should have 2 categories (0 and 1)
        assert len(result.estimates) == 2

        for est in result.estimates:
            assert 0 <= est.est <= 1
            assert est.se >= 0


class TestSDRRatio:
    """Test SDR ratio estimation."""

    def test_ratio_health_to_income(self, sdr_sample):
        """Ratio of health expenditure to income."""
        result = sdr_sample.estimation.ratio(y="health_exp", x="income")

        est = result.estimates[0]
        assert 0 < est.est < 1  # Health exp should be fraction of income
        assert est.se > 0


class TestSDRDomain:
    """Test SDR domain estimation."""

    def test_mean_income_by_region(self, sdr_sample):
        """Mean income by region."""
        result = sdr_sample.estimation.mean("income", by="region")

        # Should have 4 regions
        assert len(result.estimates) == 4

        for est in result.estimates:
            assert est.est > 0
            assert est.se > 0
            assert est.by_level is not None


class TestSDRvsBootstrap:
    """Verify SDR differs from Bootstrap with same data."""

    def test_sdr_se_is_2x_bootstrap_se(self, sdr_data):
        """
        SDR and Bootstrap should give different SEs.
        SDR uses coefficients 4/R, Bootstrap uses 1/R,
        so SDR SE = 2× Bootstrap SE (sqrt(4) = 2).
        """
        # SDR design
        rep_sdr = svy.RepWeights(method=EstimationMethod.SDR, prefix="repwtp", n_reps=80)
        design_sdr = svy.Design(wgt="pwgtp", rep_wgts=rep_sdr)
        sample_sdr = svy.Sample(data=sdr_data, design=design_sdr)

        # Bootstrap design (same weights, different method)
        rep_bs = svy.RepWeights(method=EstimationMethod.BOOTSTRAP, prefix="repwtp", n_reps=80)
        design_bs = svy.Design(wgt="pwgtp", rep_wgts=rep_bs)
        sample_bs = svy.Sample(data=sdr_data, design=design_bs)

        result_sdr = sample_sdr.estimation.mean("income", method="replication")
        result_bs = sample_bs.estimation.mean("income", method="replication")

        # Point estimates should be the same
        assert result_sdr.estimates[0].est == pytest.approx(result_bs.estimates[0].est, rel=1e-10)

        # SEs should differ: SDR SE = 2× Bootstrap SE (sqrt(4/R) / sqrt(1/R) = 2)
        se_ratio = result_sdr.estimates[0].se / result_bs.estimates[0].se
        assert se_ratio == pytest.approx(2.0, rel=0.01)
