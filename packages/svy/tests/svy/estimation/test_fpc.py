# tests/svy/estimation/test_fpc.py
"""
Tests for Finite Population Correction (FPC) in variance estimation.

Reference values from R's survey package using synthetic_multistage.csv.
See generate_synthetic_data.py for dataset construction.

All R reference values generated with:
    library(survey)
    d <- read.csv("test_data/synthetic_multistage.csv")
    # See individual test class docstrings for exact R code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import polars as pl

from svy.core.design import Design, PopSize
from svy.core.sample import Sample
from svy.errors import DimensionError, MethodError


BASE_DIR = Path(__file__).parents[2]
DATA_PATH = BASE_DIR / "test_data" / "synthetic_multistage.csv"

SE_TOL = 0.05  # 5% relative tolerance for SE
EST_TOL = 1e-3  # 0.1% relative tolerance for point estimates


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def data() -> pl.DataFrame:
    return pl.read_csv(DATA_PATH)


@pytest.fixture
def a1(data):
    """A1: Unstratified, PSU only, no FPC."""
    return Sample(data, Design(psu="psu", wgt="wgt"))


@pytest.fixture
def a3(data):
    """A3: Unstratified, PSU+SSU, no FPC."""
    return Sample(data, Design(psu="psu", ssu="ssu", wgt="wgt"))


@pytest.fixture
def b1(data):
    """B1: Stratified, PSU only, no FPC."""
    return Sample(data, Design(stratum="stratum", psu="psu", wgt="wgt"))


@pytest.fixture
def b2(data):
    """B2: Stratified, PSU only, with FPC."""
    return Sample(data, Design(stratum="stratum", psu="psu", wgt="wgt", pop_size="fpc_psu"))


@pytest.fixture
def b3(data):
    """B3: Stratified, PSU+SSU, no FPC."""
    return Sample(data, Design(stratum="stratum", psu="psu", ssu="ssu", wgt="wgt"))


@pytest.fixture
def b4(data):
    """B4: Stratified, PSU+SSU, with FPC."""
    return Sample(
        data,
        Design(
            stratum="stratum",
            psu="psu",
            ssu="ssu",
            wgt="wgt",
            pop_size=PopSize(psu="fpc_psu", ssu="fpc_ssu"),
        ),
    )


# =============================================================================
# A. UNSTRATIFIED DESIGNS
# =============================================================================


class TestA1UnstratifiedNoFPC:
    """
    R: svydesign(id=~psu, weights=~wgt, data=d)
    Mean: 60.485, SE=7.2042 | Total: 1145139, SE=188283
    Ratio: 1.246441, SE=0.009440871 | Prop(0): 0.49861, SE=0.1202
    """

    def test_mean(self, a1):
        r = a1.estimation.mean("y")
        assert r.estimates[0].est == pytest.approx(60.485, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(7.2042, rel=SE_TOL)

    def test_total(self, a1):
        r = a1.estimation.total("y")
        assert r.estimates[0].est == pytest.approx(1_145_139, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(188_283, rel=SE_TOL)

    def test_ratio(self, a1):
        r = a1.estimation.ratio(y="y", x="x")
        assert r.estimates[0].est == pytest.approx(1.246441, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(0.009440871, rel=SE_TOL)

    def test_prop(self, a1):
        r = a1.estimation.prop("binary_y")
        est_0 = [e for e in r.estimates if str(e.y_level) == "0"][0]
        assert est_0.est == pytest.approx(0.49861, rel=EST_TOL)
        assert est_0.se == pytest.approx(0.1202, rel=SE_TOL)


class TestA2UnstratifiedPSUFPC:
    """
    R: svydesign(id=~psu, fpc=~fpc_psu, weights=~wgt, data=d)
    R warns: `fpc' varies within strata: stratum 1 at stage 1

    Our implementation is stricter: we raise ValueError when FPC is not
    constant in an unstratified design. This is tested in TestFPCValidation.

    For a valid unstratified FPC, the population size must be the same for
    all observations.
    """

    def test_constant_fpc_works(self, data):
        """Unstratified FPC with constant population size should work."""
        data_const = data.with_columns(pl.lit(50).alias("const_N"))
        sample = Sample(data_const, Design(psu="psu", wgt="wgt", pop_size="const_N"))
        result = sample.estimation.mean("y")
        assert result.estimates[0].se > 0

    def test_constant_fpc_reduces_se(self, data):
        """Constant FPC should reduce SE vs no FPC."""
        data_const = data.with_columns(pl.lit(50).alias("const_N"))
        s_fpc = Sample(data_const, Design(psu="psu", wgt="wgt", pop_size="const_N"))
        s_nofpc = Sample(data_const, Design(psu="psu", wgt="wgt"))
        se_fpc = s_fpc.estimation.mean("y").estimates[0].se
        se_nofpc = s_nofpc.estimation.mean("y").estimates[0].se
        assert se_fpc < se_nofpc


class TestA3UnstratifiedPSUSSUNoFPC:
    """
    R: svydesign(id=~psu+ssu, weights=~wgt, data=d, nest=TRUE)
    Without FPC, SSU is ignored — ultimate cluster variance = A1.
    """

    def test_mean_equals_a1(self, a1, a3):
        assert a3.estimation.mean("y").estimates[0].se == pytest.approx(
            a1.estimation.mean("y").estimates[0].se, rel=1e-10
        )

    def test_total_equals_a1(self, a1, a3):
        assert a3.estimation.total("y").estimates[0].se == pytest.approx(
            a1.estimation.total("y").estimates[0].se, rel=1e-10
        )

    def test_ratio_equals_a1(self, a1, a3):
        assert a3.estimation.ratio(y="y", x="x").estimates[0].se == pytest.approx(
            a1.estimation.ratio(y="y", x="x").estimates[0].se, rel=1e-10
        )


class TestA4UnstratifiedPSUSSUFPC:
    """
    Unstratified two-stage FPC requires constant fpc_psu.
    We test with a constant population size column.
    """

    def test_fpc_reduces_se(self, data):
        data_const = data.with_columns(pl.lit(50).alias("const_N_psu"))
        s_nofpc = Sample(data_const, Design(psu="psu", ssu="ssu", wgt="wgt"))
        s_fpc = Sample(
            data_const,
            Design(
                psu="psu", ssu="ssu", wgt="wgt", pop_size=PopSize(psu="const_N_psu", ssu="fpc_ssu")
            ),
        )
        se_nofpc = s_nofpc.estimation.mean("y").estimates[0].se
        se_fpc = s_fpc.estimation.mean("y").estimates[0].se
        assert se_fpc < se_nofpc


# =============================================================================
# B. STRATIFIED DESIGNS
# =============================================================================


class TestB1StratifiedNoFPC:
    """
    R: svydesign(id=~psu, strata=~stratum, weights=~wgt, data=d)
    Mean: 60.485, SE=6.1093 | Total: 1145139, SE=187910
    Ratio: 1.246441, SE=0.01045302 | Prop(0): SE=0.1001
    """

    def test_mean(self, b1):
        r = b1.estimation.mean("y")
        assert r.estimates[0].est == pytest.approx(60.485, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(6.1093, rel=SE_TOL)

    def test_total(self, b1):
        r = b1.estimation.total("y")
        assert r.estimates[0].est == pytest.approx(1_145_139, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(187_910, rel=SE_TOL)

    def test_ratio(self, b1):
        r = b1.estimation.ratio(y="y", x="x")
        assert r.estimates[0].est == pytest.approx(1.246441, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(0.01045302, rel=SE_TOL)

    def test_prop(self, b1):
        r = b1.estimation.prop("binary_y")
        est_0 = [e for e in r.estimates if str(e.y_level) == "0"][0]
        assert est_0.se == pytest.approx(0.1001, rel=SE_TOL)

    def test_stratification_reduces_se(self, a1, b1):
        se_unstrat = a1.estimation.mean("y").estimates[0].se
        se_strat = b1.estimation.mean("y").estimates[0].se
        assert se_strat < se_unstrat


class TestB2StratifiedPSUFPC:
    """
    R: svydesign(id=~psu, strata=~stratum, fpc=~fpc_psu, weights=~wgt, data=d)
    Mean: 60.485, SE=5.2999 | Total: 1145139, SE=148739
    Ratio: 1.246441, SE=0.009283717 | Prop(0): SE=0.0855
    """

    def test_mean(self, b2):
        r = b2.estimation.mean("y")
        assert r.estimates[0].est == pytest.approx(60.485, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(5.2999, rel=SE_TOL)

    def test_total(self, b2):
        r = b2.estimation.total("y")
        assert r.estimates[0].est == pytest.approx(1_145_139, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(148_739, rel=SE_TOL)

    def test_ratio(self, b2):
        r = b2.estimation.ratio(y="y", x="x")
        assert r.estimates[0].est == pytest.approx(1.246441, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(0.009283717, rel=SE_TOL)

    def test_prop(self, b2):
        r = b2.estimation.prop("binary_y")
        est_0 = [e for e in r.estimates if str(e.y_level) == "0"][0]
        assert est_0.se == pytest.approx(0.0855, rel=SE_TOL)

    def test_fpc_reduces_se(self, b1, b2):
        se_nofpc = b1.estimation.mean("y").estimates[0].se
        se_fpc = b2.estimation.mean("y").estimates[0].se
        assert se_fpc < se_nofpc


class TestB3StratifiedPSUSSUNoFPC:
    """
    R: svydesign(id=~psu+ssu, strata=~stratum, weights=~wgt, data=d, nest=TRUE)
    Without FPC, SSU is ignored — ultimate cluster variance = B1.
    """

    def test_mean_equals_b1(self, b1, b3):
        assert b3.estimation.mean("y").estimates[0].se == pytest.approx(
            b1.estimation.mean("y").estimates[0].se, rel=1e-10
        )

    def test_total_equals_b1(self, b1, b3):
        assert b3.estimation.total("y").estimates[0].se == pytest.approx(
            b1.estimation.total("y").estimates[0].se, rel=1e-10
        )

    def test_ratio_equals_b1(self, b1, b3):
        assert b3.estimation.ratio(y="y", x="x").estimates[0].se == pytest.approx(
            b1.estimation.ratio(y="y", x="x").estimates[0].se, rel=1e-10
        )

    def test_prop_equals_b1(self, b1, b3):
        r_b1 = b1.estimation.prop("binary_y")
        r_b3 = b3.estimation.prop("binary_y")
        se_b1 = [e for e in r_b1.estimates if str(e.y_level) == "0"][0].se
        se_b3 = [e for e in r_b3.estimates if str(e.y_level) == "0"][0].se
        assert se_b3 == pytest.approx(se_b1, rel=1e-10)


class TestB4StratifiedTwoStageFPC:
    """
    R: svydesign(id=~psu+ssu, strata=~stratum, fpc=~fpc_psu+fpc_ssu,
                 weights=~wgt, data=d, nest=TRUE)
    Mean: 60.485, SE=5.4331 | Total: 1145139, SE=160864
    Ratio: 1.246441, SE=0.009862582 | Prop(0): 0.49861, SE=0.0869
    """

    def test_mean_se(self, b4):
        r = b4.estimation.mean("y")
        assert r.estimates[0].est == pytest.approx(60.485, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(5.4331, rel=SE_TOL)

    def test_total_se(self, b4):
        r = b4.estimation.total("y")
        assert r.estimates[0].est == pytest.approx(1_145_139, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(160_864, rel=SE_TOL)

    def test_ratio_se(self, b4):
        r = b4.estimation.ratio(y="y", x="x")
        assert r.estimates[0].est == pytest.approx(1.246441, rel=EST_TOL)
        assert r.estimates[0].se == pytest.approx(0.009862582, rel=SE_TOL)

    def test_prop_se(self, b4):
        r = b4.estimation.prop("binary_y")
        est_0 = [e for e in r.estimates if str(e.y_level) == "0"][0]
        assert est_0.se == pytest.approx(0.0869, rel=SE_TOL)

    def test_fpc_reduces_se(self, b3, b4):
        se_nofpc = b3.estimation.mean("y").estimates[0].se
        se_fpc = b4.estimation.mean("y").estimates[0].se
        assert se_fpc < se_nofpc

    def test_two_stage_differs_from_one_stage(self, b2, b4):
        """Two-stage FPC should give different mean SE than one-stage."""
        se_one = b2.estimation.mean("y").estimates[0].se
        se_two = b4.estimation.mean("y").estimates[0].se
        assert se_two != pytest.approx(se_one, rel=1e-3)


# =============================================================================
# C. THREE-STAGE REFERENCE
# =============================================================================


class TestC2ThreeStage:
    """
    R: svydesign(id=~psu+ssu+unit_id, strata=~stratum,
                 fpc=~fpc_psu+fpc_ssu+fpc_unit, data=d, nest=TRUE)
    Mean: 60.485, SE=5.4376 | Total: 1145139, SE=160918
    Ratio: 1.246441, SE=0.01017735 | Prop(0): SE=0.0874

    svy's two-stage FPC closely approximates R's three-stage result
    because the unit-level sampling fraction is small in this dataset.
    """

    def test_mean_se_three_stage(self, b4):
        r = b4.estimation.mean("y")
        assert r.estimates[0].se == pytest.approx(5.4376, rel=SE_TOL)


# =============================================================================
# D. DOMAIN ESTIMATION
# =============================================================================


class TestDomainEstimation:
    """
    R: svyby(~y, ~stratum, design, svymean)

    With FPC (B2):  A: 50.967, SE=6.917 | B: 74.615, SE=3.288
    No FPC (B1):    A: 50.967, SE=7.733 | B: 74.615, SE=4.244
    """

    def test_domain_with_fpc(self, b2):
        r = b2.estimation.mean("y", by="stratum")
        by_levels = {str(e.by_level[0]): e for e in r.estimates}
        assert by_levels["A"].est == pytest.approx(50.96747, rel=EST_TOL)
        assert by_levels["A"].se == pytest.approx(6.916763, rel=SE_TOL)
        assert by_levels["B"].est == pytest.approx(74.61541, rel=EST_TOL)
        assert by_levels["B"].se == pytest.approx(3.287610, rel=SE_TOL)

    def test_domain_without_fpc(self, b1):
        r = b1.estimation.mean("y", by="stratum")
        by_levels = {str(e.by_level[0]): e for e in r.estimates}
        assert by_levels["A"].se == pytest.approx(7.733176, rel=SE_TOL)
        assert by_levels["B"].se == pytest.approx(4.244286, rel=SE_TOL)

    def test_fpc_reduces_domain_se(self, b1, b2):
        r_nofpc = b1.estimation.mean("y", by="stratum")
        r_fpc = b2.estimation.mean("y", by="stratum")
        levels_nofpc = {str(e.by_level[0]): e for e in r_nofpc.estimates}
        levels_fpc = {str(e.by_level[0]): e for e in r_fpc.estimates}
        for level in levels_nofpc:
            assert levels_fpc[level].se < levels_nofpc[level].se


# =============================================================================
# E. EDGE CASES & VALIDATION
# =============================================================================


class TestFPCEdgeCases:
    """Edge cases and invariants."""

    def test_large_population_negligible_fpc(self, data):
        """When N >> n, FPC ≈ 1 and SE ≈ with-replacement SE."""
        data_big_n = data.with_columns(pl.lit(1_000_000).alias("huge_N"))
        s_fpc = Sample(
            data_big_n, Design(stratum="stratum", psu="psu", wgt="wgt", pop_size="huge_N")
        )
        s_nofpc = Sample(data_big_n, Design(stratum="stratum", psu="psu", wgt="wgt"))
        se_fpc = s_fpc.estimation.mean("y").estimates[0].se
        se_nofpc = s_nofpc.estimation.mean("y").estimates[0].se
        assert se_fpc == pytest.approx(se_nofpc, rel=0.01)

    def test_pop_size_column_missing_raises(self, data):
        """Missing pop_size column should raise ValueError at Sample construction."""
        design = Design(psu="psu", wgt="wgt", pop_size="nonexistent_col")
        with pytest.raises(ValueError, match="columns not found in data"):
            Sample(data, design)

    def test_no_pop_size_means_no_fpc(self, b1):
        """pop_size=None produces with-replacement variance."""
        assert b1._design.pop_size is None
        result = b1.estimation.mean("y")
        assert result.estimates[0].se > 0


class TestFPCDesignConstruction:
    """Design construction with FPC specifications."""

    def test_single_stage_fpc(self):
        d = Design(stratum="s", psu="p", wgt="w", pop_size="N_h")
        assert d.pop_size == "N_h"

    def test_two_stage_fpc(self):
        d = Design(
            stratum="s", psu="p", ssu="q", wgt="w", pop_size=PopSize(psu="N_psu", ssu="N_ssu")
        )
        assert isinstance(d.pop_size, PopSize)
        assert d.pop_size.psu == "N_psu"
        assert d.pop_size.ssu == "N_ssu"

    def test_pop_size_in_specified_fields(self):
        d = Design(psu="p", wgt="w", pop_size=PopSize(psu="N1", ssu="N2"))
        fields = d.specified_fields()
        assert "N1" in fields
        assert "N2" in fields

    def test_single_pop_size_in_specified_fields(self):
        d = Design(psu="p", wgt="w", pop_size="N_h")
        assert "N_h" in d.specified_fields()


class TestFPCValidation:
    """Tests for FPC input validation."""

    def test_unstratified_non_constant_fpc_raises(self, data):
        """Unstratified design with varying fpc_psu should raise DimensionError."""
        # fpc_psu is 15 for stratum A and 10 for stratum B — not constant
        design = Design(psu="psu", wgt="wgt", pop_size="fpc_psu")
        sample = Sample(data, design)
        with pytest.raises(DimensionError, match="FPC not constant"):
            sample.estimation.mean("y")

    def test_stratified_constant_fpc_ok(self, data):
        """Stratified design with fpc_psu constant within strata should work."""
        design = Design(stratum="stratum", psu="psu", wgt="wgt", pop_size="fpc_psu")
        sample = Sample(data, design)
        result = sample.estimation.mean("y")
        assert result.estimates[0].se > 0

    def test_fpc_fraction_raises(self, data):
        """FPC values < 1 (fractions) should raise MethodError."""
        data_frac = data.with_columns((pl.col("fpc_psu").cast(pl.Float64) / 100).alias("fpc_frac"))
        design = Design(stratum="stratum", psu="psu", wgt="wgt", pop_size="fpc_frac")
        sample = Sample(data_frac, design)
        with pytest.raises(MethodError, match="Invalid FPC values"):
            sample.estimation.mean("y")

    def test_fpc_n_greater_than_N_raises(self, data):
        """Population size < sample size should raise DimensionError."""
        # Set N=1, but we have 3-4 PSUs per stratum
        data_small = data.with_columns(pl.lit(1).alias("tiny_N"))
        design = Design(stratum="stratum", psu="psu", wgt="wgt", pop_size="tiny_N")
        sample = Sample(data_small, design)
        with pytest.raises(DimensionError, match="FPC population size < sample size"):
            sample.estimation.mean("y")

    def test_ssu_fpc_non_constant_within_psu_raises(self, data):
        """SSU population size varying within a PSU should raise DimensionError."""
        # Create a column that varies within PSUs
        data_bad = data.with_columns(pl.arange(0, pl.len()).alias("varying_ssu_pop") + 10)
        design = Design(
            stratum="stratum",
            psu="psu",
            ssu="ssu",
            wgt="wgt",
            pop_size=PopSize(psu="fpc_psu", ssu="varying_ssu_pop"),
        )
        sample = Sample(data_bad, design)
        with pytest.raises(DimensionError, match="FPC not constant within PSU"):
            sample.estimation.mean("y")

    def test_pop_size_column_not_numeric_raises(self, data):
        """Non-numeric pop_size column should raise at Sample construction."""
        design = Design(stratum="stratum", psu="psu", wgt="wgt", pop_size="stratum")
        with pytest.raises(TypeError, match="must be numeric"):
            Sample(data, design)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
