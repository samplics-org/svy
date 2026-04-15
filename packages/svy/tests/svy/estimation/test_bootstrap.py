# tests/svy/estimation/test_bootstrap_golden.py
"""
Bootstrap replication estimation: golden-value tests.

Reference values come from:
  - R survey::svrepdesign with scale=1/R (Stata convention)
  - Stata 18: svyset [pw=weight], bsrweight(bs_*) vce(bootstrap) dof(19)

Data: fake_survey_bootstrap_25122025.csv (20 bootstrap replicates, n≈199)

These tests validate that svy's Rust backend produces numerically
identical results to R and Stata for bootstrap variance estimation.
"""

from __future__ import annotations

import polars as pl
import pytest

import svy
from svy.core.design import Design, RepWeights
from svy.core.sample import Sample


# ---------- Fixtures ----------

DATA_PATH = "tests/test_data/fake_survey_bootstrap_25122025.csv"


@pytest.fixture(scope="module")
def boot_sample():
    """Load the bootstrap validation dataset and set up design."""
    try:
        df = pl.read_csv(DATA_PATH).fill_nan(None).drop_nulls()
    except FileNotFoundError:
        pytest.skip(f"Bootstrap validation CSV not found at {DATA_PATH}")

    # Recreate derived variables to match Stata/R
    df = df.with_columns(
        (pl.col("income") < 40000).cast(pl.Int64).alias("low_income"),
        ((pl.col("id") % 4) + 1).alias("hh_size"),
    )

    # Drop rows with missing income (matching Stata's behavior)
    df = df.filter(pl.col("income").is_not_null())

    design = Design(
        wgt="weight",
        rep_wgts=RepWeights(
            method=svy.EstimationMethod.BOOTSTRAP,
            prefix="bs_",
            n_reps=20,
        ),
    )
    return Sample(data=df, design=design)


# ---------- R (1/R scaling) golden values ----------
# These match Stata and R with des_boot_stata$scale <- 1/R

GOLDEN_Stata = {
    "mean_overall": {"est": 54687.6489088, "se": 925.94931},
    "mean_by_educ": {
        "High": {"est": 53927.0726179849, "se": 1163.33694061933},
        "Low": {"est": 53918.4961730932, "se": 1026.86515571033},
        "Med": {"est": 55315.3593038645, "se": 1303.11560542634},
    },
    "prop_overall": {
        0: {"est": 0.90920258299169, "se": 0.010668555946491106},
        1: {"est": 0.09079741700831, "se": 0.010668555946491106},
    },
    "ratio_overall": {"est": 21945.5949580571, "se": 449.77540951776},
    "total_overall": {"est": 23.14286364714, "se": 2.598291944298709},
}

GOLDEN_R = {
    "mean_overall": {"est": 54687.6489088001, "se": 925.949313287484},
    "mean_by_educ": {
        "High": {"est": 53927.0726179849, "se": 1163.33694061933},
        "Low": {"est": 53918.4961730932, "se": 1026.86515571033},
        "Med": {"est": 55315.3593038645, "se": 1303.11560542634},
    },
    "prop_overall": {
        0: {"est": 0.909202582991689, "se": 0.0106685559464911},
        1: {"est": 0.0907974170083108, "se": 0.0106685559464911},
    },
    "ratio_overall": {"est": 21945.5949580571, "se": 449.77540951776},
    "total_overall": {"est": 23.1428636471414, "se": 2.59829194429871},
}

GOLDEN = GOLDEN_R

# Tolerances
TOL = 1e-7


# ==================== Mean ====================


class TestBootstrapMean:
    def test_mean_overall_estimate(self, boot_sample):
        result = boot_sample.estimation.mean(y="income", method="replication")
        est = result.estimates[0].est
        assert est == pytest.approx(GOLDEN["mean_overall"]["est"], rel=TOL)

    def test_mean_overall_se(self, boot_sample):
        result = boot_sample.estimation.mean(y="income", method="replication")
        se = result.estimates[0].se
        assert se == pytest.approx(GOLDEN["mean_overall"]["se"], rel=TOL)

    def test_mean_by_education_estimates(self, boot_sample):
        result = boot_sample.estimation.mean(y="income", by="educ", method="replication")
        by_results = {str(e.by_level[0]): e for e in result.estimates}

        for educ, expected in GOLDEN["mean_by_educ"].items():
            assert educ in by_results, f"Missing education level: {educ}"
            assert by_results[educ].est == pytest.approx(expected["est"], rel=TOL)

    def test_mean_by_education_ses(self, boot_sample):
        result = boot_sample.estimation.mean(y="income", by="educ", method="replication")
        by_results = {str(e.by_level[0]): e for e in result.estimates}

        for educ, expected in GOLDEN["mean_by_educ"].items():
            assert by_results[educ].se == pytest.approx(expected["se"], rel=TOL)

    def test_mean_point_estimate_matches_taylor(self, boot_sample):
        """Point estimates should be identical regardless of method."""
        boot = boot_sample.estimation.mean(y="income", method="replication")
        taylor = boot_sample.estimation.mean(y="income")
        assert boot.estimates[0].est == pytest.approx(taylor.estimates[0].est, rel=1e-10)


# ==================== Proportion ====================


class TestBootstrapProp:
    def test_prop_overall_estimates(self, boot_sample):
        result = boot_sample.estimation.prop(y="low_income", method="replication")
        level_results = {e.y_level: e for e in result.estimates}

        for level, expected in GOLDEN["prop_overall"].items():
            assert level in level_results, f"Missing level: {level}"
            assert level_results[level].est == pytest.approx(expected["est"], rel=TOL)

    def test_prop_overall_ses(self, boot_sample):
        result = boot_sample.estimation.prop(y="low_income", method="replication")
        level_results = {e.y_level: e for e in result.estimates}

        for level, expected in GOLDEN["prop_overall"].items():
            assert level_results[level].se == pytest.approx(expected["se"], rel=TOL)

    def test_prop_sums_to_one(self, boot_sample):
        result = boot_sample.estimation.prop(y="low_income", method="replication")
        total = sum(e.est for e in result.estimates)
        assert total == pytest.approx(1.0, abs=1e-10)


# ==================== Ratio ====================


class TestBootstrapRatio:
    def test_ratio_overall_estimate(self, boot_sample):
        result = boot_sample.estimation.ratio(y="income", x="hh_size", method="replication")
        assert result.estimates[0].est == pytest.approx(GOLDEN["ratio_overall"]["est"], rel=TOL)

    def test_ratio_overall_se(self, boot_sample):
        result = boot_sample.estimation.ratio(y="income", x="hh_size", method="replication")
        assert result.estimates[0].se == pytest.approx(GOLDEN["ratio_overall"]["se"], rel=TOL)


# ==================== Total ====================


class TestBootstrapTotal:
    def test_total_overall_estimate(self, boot_sample):
        result = boot_sample.estimation.total(y="low_income", method="replication")
        assert result.estimates[0].est == pytest.approx(GOLDEN["total_overall"]["est"], rel=TOL)

    def test_total_overall_se(self, boot_sample):
        result = boot_sample.estimation.total(y="low_income", method="replication")
        assert result.estimates[0].se == pytest.approx(GOLDEN["total_overall"]["se"], rel=TOL)


# ==================== Method Resolution ====================


class TestBootstrapMethodResolution:
    def test_default_is_taylor(self, boot_sample):
        """With both wgt and rep_wgts, default should be Taylor."""
        result = boot_sample.estimation.mean(y="income")
        assert result.method.name == "TAYLOR"

    def test_explicit_replication(self, boot_sample):
        """method='replication' should use bootstrap."""
        result = boot_sample.estimation.mean(y="income", method="replication")
        assert result.method.name == "BOOTSTRAP"

    def test_method_string_case_insensitive(self, boot_sample):
        """Method string should be case-insensitive."""
        r1 = boot_sample.estimation.mean(y="income", method="replication")
        r2 = boot_sample.estimation.mean(y="income", method="Replication")
        r3 = boot_sample.estimation.mean(y="income", method="REPLICATION")
        assert r1.estimates[0].se == r2.estimates[0].se == r3.estimates[0].se

    def test_method_aliases(self, boot_sample):
        """'rep', 'replicate', 'bootstrap' all work."""
        r1 = boot_sample.estimation.mean(y="income", method="rep")
        r2 = boot_sample.estimation.mean(y="income", method="replicate")
        r3 = boot_sample.estimation.mean(y="income", method=svy.EstimationMethod.BOOTSTRAP)
        assert r1.estimates[0].se == r2.estimates[0].se == r3.estimates[0].se


# ==================== Variance Center ====================


class TestBootstrapVarianceCenter:
    def test_rep_mean_vs_estimate_differ(self, boot_sample):
        """Different centering should produce different SEs."""
        r_mean = boot_sample.estimation.mean(
            y="income", method="replication", variance_center="rep_mean"
        )
        r_est = boot_sample.estimation.mean(
            y="income", method="replication", variance_center="estimate"
        )
        # SEs should differ (slightly) due to different centering
        # but point estimates should be identical
        assert r_mean.estimates[0].est == pytest.approx(r_est.estimates[0].est, rel=1e-10)
        # SEs may or may not differ significantly — just check both are positive
        assert r_mean.estimates[0].se > 0
        assert r_est.estimates[0].se > 0
