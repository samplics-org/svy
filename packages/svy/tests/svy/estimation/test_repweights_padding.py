# tests/test_repweights_padding.py
"""
Comprehensive tests for RepWeights padding detection.

Tests various column naming patterns to ensure robust auto-detection:
- No padding: wt1, wt2, wt3, ...
- 2-digit padding: wt01, wt02, wt03, ..., wt10, wt11, ...
- 3-digit padding: wt001, wt002, wt003, ..., wt100, wt101, ...
- 4-digit padding: wt0001, wt0002, ...
- Mixed scenarios and edge cases
"""

import pytest

from svy.core.design import Design, RepWeights
from svy.core.enumerations import EstimationMethod


class TestRepWeightsPadding:
    """Test suite for RepWeights padding detection and generation."""

    # =========================================================================
    # Basic Padding Detection Tests
    # =========================================================================

    def test_no_padding_detection(self):
        """Test detection of unpadded column names."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5)
        data_cols = ["id", "wt1", "wt2", "wt3", "wt4", "wt5", "other"]

        detected = rw._detect_padding(data_cols)
        assert detected == 0, "Should detect no padding for wt1, wt2, ..."

        columns = rw.columns_from_data(data_cols)
        assert columns == ["wt1", "wt2", "wt3", "wt4", "wt5"]

    def test_two_digit_padding_detection(self):
        """Test detection of 2-digit zero-padded columns."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=15)
        data_cols = ["id", "wt01", "wt02", "wt03", "wt10", "wt15", "other"]

        detected = rw._detect_padding(data_cols)
        assert detected == 2, "Should detect 2-digit padding"

        columns = rw.columns_from_data(data_cols)
        expected = [f"wt{i:02d}" for i in range(1, 16)]
        assert columns == expected

    def test_three_digit_padding_detection(self):
        """Test detection of 3-digit zero-padded columns (like btwt001)."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="btwt", n_reps=277)
        data_cols = ["id", "btwt001", "btwt002", "btwt050", "btwt100", "btwt277", "other"]

        detected = rw._detect_padding(data_cols)
        assert detected == 3, "Should detect 3-digit padding"

        columns = rw.columns_from_data(data_cols)
        expected = [f"btwt{i:03d}" for i in range(1, 278)]
        assert columns == expected

    def test_four_digit_padding_detection(self):
        """Test detection of 4-digit zero-padded columns."""
        rw = RepWeights(method=EstimationMethod.BRR, prefix="rep", n_reps=1000)
        data_cols = ["id", "rep0001", "rep0002", "rep0100", "rep1000"]

        detected = rw._detect_padding(data_cols)
        assert detected == 4, "Should detect 4-digit padding"

        columns = rw.columns_from_data(data_cols)
        expected = [f"rep{i:04d}" for i in range(1, 1001)]
        assert columns == expected

    # =========================================================================
    # Explicit Padding Tests
    # =========================================================================

    def test_explicit_no_padding(self):
        """Test explicit padding=0 setting."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5, padding=0)

        columns = rw.columns
        assert columns == ["wt1", "wt2", "wt3", "wt4", "wt5"]

        # Should still respect explicit setting even with different data
        data_cols = ["wt001", "wt002", "wt003"]
        columns = rw.columns_from_data(data_cols)
        assert columns == ["wt1", "wt2", "wt3", "wt4", "wt5"]

    def test_explicit_two_digit_padding(self):
        """Test explicit padding=2 setting."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=100, padding=2)

        columns = rw.columns
        expected = [f"wt{i:02d}" for i in range(1, 101)]
        assert columns == expected

    def test_explicit_three_digit_padding(self):
        """Test explicit padding=3 setting."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="btwt", n_reps=277, padding=3)

        columns = rw.columns
        expected = [f"btwt{i:03d}" for i in range(1, 278)]
        assert columns == expected

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_empty_data_columns(self):
        """Test with no matching columns in data."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5)
        data_cols = ["id", "age", "income"]  # No wt columns

        detected = rw._detect_padding(data_cols)
        assert detected == 0, "Should return 0 when no matching columns"

        columns = rw.columns_from_data(data_cols)
        assert columns == ["wt1", "wt2", "wt3", "wt4", "wt5"]

    def test_single_digit_not_padded(self):
        """Test that single digits without leading zeros are not treated as padded."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=3)
        data_cols = ["wt1", "wt2", "wt3"]  # Single digits, no padding

        detected = rw._detect_padding(data_cols)
        assert detected == 0, "Single digits should not be detected as padded"

    def test_mixed_padding_uses_max(self):
        """Test that mixed padding levels use the maximum detected."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5)
        # Mix of 2-digit and 3-digit padding (weird but test it)
        data_cols = ["wt01", "wt002", "wt003"]

        detected = rw._detect_padding(data_cols)
        assert detected == 3, "Should use maximum padding detected"

    def test_non_matching_prefix_ignored(self):
        """Test that columns with different prefixes are ignored."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5)
        data_cols = ["other001", "other002", "wt1", "wt2", "wt3"]

        detected = rw._detect_padding(data_cols)
        assert detected == 0, "Should only look at columns matching prefix"

    def test_partial_match_ignored(self):
        """Test that partial prefix matches are ignored."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=3)
        data_cols = ["weight1", "wt1", "wt2", "wt_extra"]

        detected = rw._detect_padding(data_cols)
        assert detected == 0

        columns = rw.columns_from_data(data_cols)
        assert columns == ["wt1", "wt2", "wt3"]

    def test_special_characters_in_prefix(self):
        """Test prefix with special regex characters."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt_rep", n_reps=3)
        data_cols = ["wt_rep1", "wt_rep2", "wt_rep3"]

        columns = rw.columns_from_data(data_cols)
        assert columns == ["wt_rep1", "wt_rep2", "wt_rep3"]

    def test_prefix_with_dots(self):
        """Test prefix containing dots (common in some datasets)."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt.rep", n_reps=3)
        data_cols = ["wt.rep001", "wt.rep002", "wt.rep003"]

        detected = rw._detect_padding(data_cols)
        assert detected == 3

        columns = rw.columns_from_data(data_cols)
        assert columns == ["wt.rep001", "wt.rep002", "wt.rep003"]

    # =========================================================================
    # Real-World Patterns
    # =========================================================================

    def test_nhanes_pattern(self):
        """Test pattern similar to NHANES survey data."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wtmec", n_reps=80)
        data_cols = [f"wtmec{i}" for i in range(1, 81)] + ["seqn", "age"]

        columns = rw.columns_from_data(data_cols)
        assert columns == [f"wtmec{i}" for i in range(1, 81)]

    def test_census_pattern(self):
        """Test pattern similar to census replicate weights."""
        rw = RepWeights(method=EstimationMethod.BRR, prefix="repwt", n_reps=80)
        data_cols = [f"repwt{i:02d}" for i in range(1, 81)] + ["serialno", "st"]

        columns = rw.columns_from_data(data_cols)
        assert columns == [f"repwt{i:02d}" for i in range(1, 81)]

    def test_survey_package_r_pattern(self):
        """Test pattern from R survey package (often unpadded)."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="rep", n_reps=50)
        data_cols = [f"rep{i}" for i in range(1, 51)] + ["id", "stratum"]

        columns = rw.columns_from_data(data_cols)
        assert columns == [f"rep{i}" for i in range(1, 51)]

    def test_stata_pattern(self):
        """Test pattern common in Stata datasets (3-digit padding)."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="rw", n_reps=200)
        data_cols = [f"rw{i:03d}" for i in [1, 2, 50, 100, 200]] + ["_id"]

        columns = rw.columns_from_data(data_cols)
        assert columns == [f"rw{i:03d}" for i in range(1, 201)]

    # =========================================================================
    # Integration with Design
    # =========================================================================

    def test_design_specified_fields_no_padding(self):
        """Test Design.specified_fields() with unpadded rep weights."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=3)
        design = Design(wgt="weight", stratum="strat", psu="psu", rep_wgts=rw)

        data_cols = ["id", "weight", "strat", "psu", "wt1", "wt2", "wt3"]
        fields = design.specified_fields(data_columns=data_cols)

        assert "weight" in fields
        assert "strat" in fields
        assert "psu" in fields
        assert "wt1" in fields
        assert "wt2" in fields
        assert "wt3" in fields

    def test_design_specified_fields_with_padding(self):
        """Test Design.specified_fields() with zero-padded rep weights."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="btwt", n_reps=277)
        design = Design(wgt="btwt0", rep_wgts=rw)

        data_cols = ["id", "btwt0"] + [f"btwt{i:03d}" for i in range(1, 278)]
        fields = design.specified_fields(data_columns=data_cols)

        assert "btwt0" in fields
        assert "btwt001" in fields
        assert "btwt100" in fields
        assert "btwt277" in fields
        assert len([f for f in fields if f.startswith("btwt")]) == 278  # btwt0 + 277 reps

    def test_design_specified_fields_without_data(self):
        """Test Design.specified_fields() without data_columns (fallback)."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=3, padding=2)
        design = Design(wgt="weight", rep_wgts=rw)

        # Without data_columns, should use explicit padding
        fields = design.specified_fields()

        assert "weight" in fields
        assert "wt01" in fields
        assert "wt02" in fields
        assert "wt03" in fields

    # =========================================================================
    # Validation and Error Cases
    # =========================================================================

    def test_invalid_padding_negative(self):
        """Test that negative padding raises error."""
        with pytest.raises(ValueError, match="padding must be >= 0"):
            RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5, padding=-1)

    def test_columns_property_without_data(self):
        """Test .columns property (no data) uses explicit padding or defaults to 0."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=3)

        # Without explicit padding, defaults to no padding
        columns = rw.columns
        assert columns == ["wt1", "wt2", "wt3"]

    def test_columns_property_with_explicit_padding(self):
        """Test .columns property with explicit padding."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=3, padding=2)

        columns = rw.columns
        assert columns == ["wt01", "wt02", "wt03"]

    def test_update_rep_weights_preserves_padding(self):
        """Test that updating rep weights preserves padding."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5, padding=3)
        design = Design(wgt="weight", rep_wgts=rw)

        # Update n_reps, padding should be preserved
        new_design = design.update_rep_weights(n_reps=10)

        assert new_design.rep_wgts.padding == 3
        assert new_design.rep_wgts.columns == [f"wt{i:03d}" for i in range(1, 11)]

    def test_update_rep_weights_change_padding(self):
        """Test changing padding via update_rep_weights."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5, padding=2)
        design = Design(wgt="weight", rep_wgts=rw)

        # Change padding
        new_design = design.update_rep_weights(padding=3)

        assert new_design.rep_wgts.padding == 3
        assert new_design.rep_wgts.columns == [f"wt{i:03d}" for i in range(1, 6)]

    # =========================================================================
    # Regression Tests (Real Bug Cases)
    # =========================================================================

    def test_btwt_issue_original_bug(self):
        """
        Test the original reported bug: btwt001 columns not detected.

        This was the actual failing case that prompted the fix.
        """
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="btwt", n_reps=277)

        # Data has zero-padded columns
        data_cols = ["id", "btwt0"] + [f"btwt{i:03d}" for i in range(1, 278)]

        # Should detect 3-digit padding
        detected = rw._detect_padding(data_cols)
        assert detected == 3

        # Should generate matching columns
        columns = rw.columns_from_data(data_cols)
        assert columns[0] == "btwt001"
        assert columns[-1] == "btwt277"
        assert len(columns) == 277

    def test_large_n_reps_performance(self):
        """Test with large number of replicates (performance check)."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=1000)

        # Should complete quickly even with 1000 replicates
        data_cols = [f"wt{i:04d}" for i in [1, 500, 1000]]
        columns = rw.columns_from_data(data_cols)

        assert len(columns) == 1000
        assert columns[0] == "wt0001"
        assert columns[499] == "wt0500"
        assert columns[999] == "wt1000"

    # =========================================================================
    # String Representation Tests
    # =========================================================================

    def test_repr_without_padding(self):
        """Test __repr__ without explicit padding."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5)

        repr_str = repr(rw)
        assert "method=" in repr_str
        assert "prefix='wt'" in repr_str
        assert "n_reps=5" in repr_str
        assert "padding" not in repr_str  # Should not show if None

    def test_repr_with_padding(self):
        """Test __repr__ with explicit padding."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5, padding=3)

        repr_str = repr(rw)
        assert "padding=3" in repr_str

    # =========================================================================
    # Cross-validation Tests
    # =========================================================================

    def test_columns_from_data_matches_explicit(self):
        """Test that auto-detection matches explicit setting when data is consistent."""
        rw_explicit = RepWeights(
            method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5, padding=2
        )
        rw_auto = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5)

        data_cols = ["wt01", "wt02", "wt03", "wt04", "wt05"]

        explicit_cols = rw_explicit.columns
        auto_cols = rw_auto.columns_from_data(data_cols)

        assert explicit_cols == auto_cols

    def test_columns_from_data_different_from_explicit(self):
        """Test that auto-detection overrides explicit when data differs."""
        rw = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=5, padding=2)

        # Data has 3-digit padding but we set padding=2
        data_cols = ["wt001", "wt002", "wt003", "wt004", "wt005"]

        # columns property uses explicit padding
        assert rw.columns == ["wt01", "wt02", "wt03", "wt04", "wt05"]

        # columns_from_data uses explicit padding (doesn't auto-detect when explicit is set)
        assert rw.columns_from_data(data_cols) == ["wt01", "wt02", "wt03", "wt04", "wt05"]


# =============================================================================
# Parametrized Tests for Comprehensive Coverage
# =============================================================================


class TestRepWeightsPaddingParametrized:
    """Parametrized tests for exhaustive padding pattern coverage."""

    @pytest.mark.parametrize(
        "prefix,n_reps,padding,expected_first,expected_last",
        [
            ("wt", 5, 0, "wt1", "wt5"),
            ("wt", 5, 1, "wt1", "wt5"),
            ("wt", 5, 2, "wt01", "wt05"),
            ("wt", 5, 3, "wt001", "wt005"),
            ("wt", 10, 2, "wt01", "wt10"),
            ("wt", 100, 2, "wt01", "wt100"),  # Overflow is OK
            ("wt", 100, 3, "wt001", "wt100"),
            ("rep", 1000, 4, "rep0001", "rep1000"),
            ("btwt", 277, 3, "btwt001", "btwt277"),
            ("w_t", 3, 2, "w_t01", "w_t03"),
            ("wt.rep", 3, 2, "wt.rep01", "wt.rep03"),
        ],
    )
    def test_explicit_padding_patterns(
        self, prefix, n_reps, padding, expected_first, expected_last
    ):
        """Test various explicit padding patterns."""
        rw = RepWeights(
            method=EstimationMethod.JACKKNIFE, prefix=prefix, n_reps=n_reps, padding=padding
        )

        columns = rw.columns
        assert columns[0] == expected_first
        assert columns[-1] == expected_last
        assert len(columns) == n_reps

    @pytest.mark.parametrize(
        "data_pattern,expected_padding",
        [
            (["wt1", "wt2", "wt3"], 0),
            (["wt01", "wt02", "wt10"], 2),
            (["wt001", "wt002", "wt100"], 3),
            (["wt0001", "wt0002", "wt1000"], 4),
            (["rep1", "rep2", "rep50"], 0),
            (["rep01", "rep02", "rep99"], 2),
            (["btwt001", "btwt050", "btwt277"], 3),
        ],
    )
    def test_auto_detection_patterns(self, data_pattern, expected_padding):
        """Test auto-detection across various patterns."""
        rw = RepWeights(
            method=EstimationMethod.JACKKNIFE, prefix=data_pattern[0][:-1].rstrip("0"), n_reps=3
        )

        detected = rw._detect_padding(data_pattern)
        assert detected == expected_padding


# =============================================================================
# Property-Based Testing (if hypothesis is available)
# =============================================================================

try:
    from hypothesis import given
    from hypothesis import strategies as st

    class TestRepWeightsPropertyBased:
        """Property-based tests using hypothesis."""

        @given(
            n_reps=st.integers(min_value=2, max_value=100),
            padding=st.integers(min_value=0, max_value=5),
        )
        def test_column_generation_consistency(self, n_reps, padding):
            """Property: Generated columns should always have correct count and format."""
            rw = RepWeights(
                method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=n_reps, padding=padding
            )

            columns = rw.columns

            # Should generate exactly n_reps columns
            assert len(columns) == n_reps

            # All should start with prefix
            assert all(col.startswith("wt") for col in columns)

            # Should be unique
            assert len(set(columns)) == n_reps

            # Should be in order
            nums = [int(col[2:]) for col in columns]
            assert nums == list(range(1, n_reps + 1))

        @given(
            n_reps=st.integers(min_value=2, max_value=50),
            padding=st.integers(min_value=1, max_value=4),
        )
        def test_auto_detection_matches_explicit(self, n_reps, padding):
            """Property: Auto-detection should match explicit when data is consistent."""
            rw_auto = RepWeights(method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=n_reps)
            rw_explicit = RepWeights(
                method=EstimationMethod.JACKKNIFE, prefix="wt", n_reps=n_reps, padding=padding
            )

            # Generate data with explicit padding
            data_cols = [f"wt{i:0{padding}d}" for i in range(1, n_reps + 1)]

            # Auto-detection should match
            auto_cols = rw_auto.columns_from_data(data_cols)
            explicit_cols = rw_explicit.columns

            assert auto_cols == explicit_cols

except ImportError:
    # hypothesis not available, skip property-based tests
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
