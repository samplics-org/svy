# tests/svy/estimation/test_weight_casting.py
"""
Tests for automatic Float64 casting of weight columns.

Ensures that integer weight columns are automatically cast to Float64
before being sent to the Rust backend, preventing dtype errors.
"""

import polars as pl
import pytest

import svy

from svy.core.enumerations import EstimationMethod


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def data_with_int_weights():
    """Sample data with integer weights (Int64)."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "stratum": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 4, 4, 5, 5, 6],
            "y": [10.0, 12.0, 15.0, 11.0, 14.0, 20.0, 22.0, 18.0, 25.0, 21.0],
            "x": [5.0, 6.0, 7.0, 5.5, 6.5, 10.0, 11.0, 9.0, 12.0, 10.5],
            "category": ["Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No", "Yes", "Yes"],
            "wgt": [100, 100, 150, 150, 120, 200, 200, 180, 180, 160],  # Int64
        }
    )


@pytest.fixture
def data_with_float_weights():
    """Sample data with float weights (Float64) - control group."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "stratum": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 4, 4, 5, 5, 6],
            "y": [10.0, 12.0, 15.0, 11.0, 14.0, 20.0, 22.0, 18.0, 25.0, 21.0],
            "x": [5.0, 6.0, 7.0, 5.5, 6.5, 10.0, 11.0, 9.0, 12.0, 10.5],
            "category": ["Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No", "Yes", "Yes"],
            "wgt": [100.0, 100.0, 150.0, 150.0, 120.0, 200.0, 200.0, 180.0, 180.0, 160.0],
        }
    )


@pytest.fixture
def data_with_int_rep_weights():
    """Sample data with integer replicate weights."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "stratum": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 4, 4, 5, 5, 6],
            "y": [10.0, 12.0, 15.0, 11.0, 14.0, 20.0, 22.0, 18.0, 25.0, 21.0],
            "wgt": [100, 100, 150, 150, 120, 200, 200, 180, 180, 160],  # Int64
            "repwgt1": [100, 0, 150, 150, 120, 200, 200, 180, 180, 160],  # Int64
            "repwgt2": [100, 100, 0, 150, 120, 200, 200, 180, 180, 160],  # Int64
            "repwgt3": [100, 100, 150, 0, 120, 200, 200, 180, 180, 160],  # Int64
            "repwgt4": [100, 100, 150, 150, 0, 200, 200, 180, 180, 160],  # Int64
            "repwgt5": [100, 100, 150, 150, 120, 0, 200, 180, 180, 160],  # Int64
            "repwgt6": [100, 100, 150, 150, 120, 200, 0, 180, 180, 160],  # Int64
        }
    )


# ============================================================================
# Test Main Weight Casting (Taylor Methods)
# ============================================================================


class TestMainWeightCastingTaylor:
    """Test automatic casting of main weight column for Taylor methods."""

    def test_weight_column_is_int64(self, data_with_int_weights):
        """Verify fixture has Int64 weights."""
        assert data_with_int_weights["wgt"].dtype == pl.Int64

    def test_mean_with_int_weights(self, data_with_int_weights):
        """Mean estimation should work with integer weights."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data_with_int_weights, design=design)

        # Should not raise an error
        result = sample.estimation.mean("y")

        assert result is not None
        assert len(result.estimates) == 1
        assert result.estimates[0].est > 0
        assert result.estimates[0].se > 0

    def test_total_with_int_weights(self, data_with_int_weights):
        """Total estimation should work with integer weights."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data_with_int_weights, design=design)

        result = sample.estimation.total("y")

        assert result is not None
        assert len(result.estimates) == 1
        assert result.estimates[0].est > 0
        assert result.estimates[0].se > 0

    def test_ratio_with_int_weights(self, data_with_int_weights):
        """Ratio estimation should work with integer weights."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data_with_int_weights, design=design)

        result = sample.estimation.ratio(y="y", x="x")

        assert result is not None
        assert len(result.estimates) == 1
        assert result.estimates[0].est > 0
        assert result.estimates[0].se > 0

    def test_prop_with_int_weights(self, data_with_int_weights):
        """Proportion estimation should work with integer weights."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data_with_int_weights, design=design)

        result = sample.estimation.prop("category")

        assert result is not None
        assert len(result.estimates) == 2  # Yes and No
        for est in result.estimates:
            assert 0 <= est.est <= 1
            assert est.se >= 0


# ============================================================================
# Test Int vs Float Weights Produce Same Results
# ============================================================================


class TestIntVsFloatWeightsEquivalence:
    """Verify that integer and float weights produce identical results."""

    def test_mean_int_vs_float_same_result(self, data_with_int_weights, data_with_float_weights):
        """Int and float weights should produce identical mean results."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")

        sample_int = svy.Sample(data=data_with_int_weights, design=design)
        sample_float = svy.Sample(data=data_with_float_weights, design=design)

        result_int = sample_int.estimation.mean("y")
        result_float = sample_float.estimation.mean("y")

        assert result_int.estimates[0].est == pytest.approx(
            result_float.estimates[0].est, rel=1e-10
        )
        assert result_int.estimates[0].se == pytest.approx(result_float.estimates[0].se, rel=1e-10)

    def test_total_int_vs_float_same_result(self, data_with_int_weights, data_with_float_weights):
        """Int and float weights should produce identical total results."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")

        sample_int = svy.Sample(data=data_with_int_weights, design=design)
        sample_float = svy.Sample(data=data_with_float_weights, design=design)

        result_int = sample_int.estimation.total("y")
        result_float = sample_float.estimation.total("y")

        assert result_int.estimates[0].est == pytest.approx(
            result_float.estimates[0].est, rel=1e-10
        )
        assert result_int.estimates[0].se == pytest.approx(result_float.estimates[0].se, rel=1e-10)

    def test_ratio_int_vs_float_same_result(self, data_with_int_weights, data_with_float_weights):
        """Int and float weights should produce identical ratio results."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")

        sample_int = svy.Sample(data=data_with_int_weights, design=design)
        sample_float = svy.Sample(data=data_with_float_weights, design=design)

        result_int = sample_int.estimation.ratio(y="y", x="x")
        result_float = sample_float.estimation.ratio(y="y", x="x")

        assert result_int.estimates[0].est == pytest.approx(
            result_float.estimates[0].est, rel=1e-10
        )
        assert result_int.estimates[0].se == pytest.approx(result_float.estimates[0].se, rel=1e-10)

    def test_prop_int_vs_float_same_result(self, data_with_int_weights, data_with_float_weights):
        """Int and float weights should produce identical proportion results."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")

        sample_int = svy.Sample(data=data_with_int_weights, design=design)
        sample_float = svy.Sample(data=data_with_float_weights, design=design)

        result_int = sample_int.estimation.prop("category")
        result_float = sample_float.estimation.prop("category")

        # Sort by y_level to ensure comparison is correct
        ests_int = sorted(result_int.estimates, key=lambda x: str(x.y_level))
        ests_float = sorted(result_float.estimates, key=lambda x: str(x.y_level))

        for est_i, est_f in zip(ests_int, ests_float):
            assert est_i.est == pytest.approx(est_f.est, rel=1e-10)
            assert est_i.se == pytest.approx(est_f.se, rel=1e-10)


# ============================================================================
# Test Domain Estimation with Int Weights
# ============================================================================


class TestDomainEstimationIntWeights:
    """Test domain estimation with integer weights."""

    def test_mean_by_domain_with_int_weights(self, data_with_int_weights):
        """Mean by domain should work with integer weights."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data_with_int_weights, design=design)

        result = sample.estimation.mean("y", by="stratum")

        assert result is not None
        assert len(result.estimates) == 2  # A and B
        for est in result.estimates:
            assert est.est > 0
            assert est.se > 0

    def test_total_by_domain_with_int_weights(self, data_with_int_weights):
        """Total by domain should work with integer weights."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data_with_int_weights, design=design)

        result = sample.estimation.total("y", by="stratum")

        assert result is not None
        assert len(result.estimates) == 2
        for est in result.estimates:
            assert est.est > 0
            assert est.se > 0


# ============================================================================
# Test Replicate Weight Casting
# ============================================================================


class TestReplicateWeightCasting:
    """Test automatic casting of replicate weight columns."""

    def test_rep_weights_are_int64(self, data_with_int_rep_weights):
        """Verify fixture has Int64 replicate weights."""
        assert data_with_int_rep_weights["repwgt1"].dtype == pl.Int64
        assert data_with_int_rep_weights["repwgt6"].dtype == pl.Int64

    def test_jackknife_mean_with_int_rep_weights(self, data_with_int_rep_weights):
        """Jackknife mean should work with integer replicate weights."""
        rep_wgts = svy.RepWeights(
            prefix="repwgt",
            method=EstimationMethod.JACKKNIFE,
            n_reps=6,  # We have repwgt1 through repwgt6
        )
        design = svy.Design(wgt="wgt", rep_wgts=rep_wgts)
        sample = svy.Sample(data=data_with_int_rep_weights, design=design)

        # Should not raise an error
        result = sample.estimation.mean("y")

        assert result is not None
        assert len(result.estimates) == 1
        assert result.estimates[0].est > 0
        assert result.estimates[0].se > 0

    def test_jackknife_total_with_int_rep_weights(self, data_with_int_rep_weights):
        """Jackknife total should work with integer replicate weights."""
        rep_wgts = svy.RepWeights(
            prefix="repwgt",
            method=EstimationMethod.JACKKNIFE,
            n_reps=6,  # We have repwgt1 through repwgt6
        )
        design = svy.Design(wgt="wgt", rep_wgts=rep_wgts)
        sample = svy.Sample(data=data_with_int_rep_weights, design=design)

        result = sample.estimation.total("y")

        assert result is not None
        assert len(result.estimates) == 1
        assert result.estimates[0].est > 0
        assert result.estimates[0].se > 0


# ============================================================================
# Test Various Integer Types
# ============================================================================


class TestVariousIntegerTypes:
    """Test that various integer types are handled correctly."""

    @pytest.mark.parametrize("int_dtype", [pl.Int8, pl.Int16, pl.Int32, pl.Int64])
    def test_signed_int_weights(self, int_dtype):
        """Various signed integer types should work."""
        # Use small values that fit in Int8 (max 127)
        data = pl.DataFrame(
            {
                "stratum": ["A", "A", "B", "B"],
                "psu": [1, 2, 3, 4],
                "y": [10.0, 20.0, 30.0, 40.0],
                "wgt": pl.Series([10, 15, 20, 18]).cast(int_dtype),
            }
        )

        assert data["wgt"].dtype == int_dtype

        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data, design=design)

        result = sample.estimation.mean("y")

        assert result is not None
        assert result.estimates[0].est > 0

    @pytest.mark.parametrize("uint_dtype", [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64])
    def test_unsigned_int_weights(self, uint_dtype):
        """Various unsigned integer types should work."""
        # Use small values that fit in UInt8 (max 255)
        data = pl.DataFrame(
            {
                "stratum": ["A", "A", "B", "B"],
                "psu": [1, 2, 3, 4],
                "y": [10.0, 20.0, 30.0, 40.0],
                "wgt": pl.Series([10, 15, 20, 18]).cast(uint_dtype),
            }
        )

        assert data["wgt"].dtype == uint_dtype

        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data, design=design)

        result = sample.estimation.mean("y")

        assert result is not None
        assert result.estimates[0].est > 0


# ============================================================================
# Test _ensure_float64 Helper Method
# ============================================================================


class TestEnsureFloat64Helper:
    """Test the _ensure_float64 helper method directly."""

    def test_ensure_float64_casts_int_columns(self, data_with_int_weights):
        """_ensure_float64 should cast int columns to float64."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data_with_int_weights, design=design)

        # Access the estimation object to get the helper
        estimation = sample.estimation

        # Original data has int weights
        assert data_with_int_weights["wgt"].dtype == pl.Int64

        # Cast using helper
        result = estimation._ensure_float64(data_with_int_weights, ["wgt"])

        # Result should have float weights
        assert result["wgt"].dtype == pl.Float64

    def test_ensure_float64_no_op_for_float_columns(self, data_with_float_weights):
        """_ensure_float64 should not modify already-float columns."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data_with_float_weights, design=design)

        estimation = sample.estimation

        # Original data has float weights
        assert data_with_float_weights["wgt"].dtype == pl.Float64

        # Cast using helper (should be no-op)
        result = estimation._ensure_float64(data_with_float_weights, ["wgt"])

        # Result should still have float weights
        assert result["wgt"].dtype == pl.Float64

    def test_ensure_float64_handles_missing_columns(self, data_with_int_weights):
        """_ensure_float64 should handle non-existent columns gracefully."""
        design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = svy.Sample(data=data_with_int_weights, design=design)

        estimation = sample.estimation

        # Should not raise an error for non-existent column
        result = estimation._ensure_float64(data_with_int_weights, ["nonexistent_col"])

        # Data should be unchanged
        assert result.shape == data_with_int_weights.shape

    def test_ensure_float64_multiple_columns(self, data_with_int_rep_weights):
        """_ensure_float64 should cast multiple columns at once."""
        design = svy.Design(wgt="wgt")
        sample = svy.Sample(data=data_with_int_rep_weights, design=design)

        estimation = sample.estimation

        cols_to_cast = ["wgt", "repwgt1", "repwgt2", "repwgt3"]

        result = estimation._ensure_float64(data_with_int_rep_weights, cols_to_cast)

        for col in cols_to_cast:
            assert result[col].dtype == pl.Float64
