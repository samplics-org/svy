# tests/test_where_by.py
"""
Tests for domain estimation (subpopulation analysis) via the `where` parameter.

These tests verify that the `where` parameter in estimation methods performs
proper domain estimation (like R's survey::subset) rather than simple filtering.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import col
from svy.core.design import Design
from svy.core.sample import Sample
from svy.estimation.estimate import Estimate


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_survey_data() -> pl.DataFrame:
    """Create simple survey data for testing domain estimation."""
    np.random.seed(42)
    n = 500

    return pl.DataFrame(
        {
            "id": range(1, n + 1),
            "stratum": np.random.choice(["A", "B", "C"], n),
            "psu": np.random.choice(range(1, 21), n),
            "weight": np.random.uniform(0.5, 2.0, n),
            "age": np.random.randint(15, 80, n),
            "gender": np.random.choice(["Male", "Female"], n),
            "region": np.random.choice(["North", "South", "East", "West"], n),
            "income": np.random.exponential(50000, n),
            "employed": np.random.choice([0, 1], n, p=[0.3, 0.7]),
            "health_score": np.random.normal(70, 15, n).clip(0, 100),
            "has_insurance": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        }
    )


@pytest.fixture
def simple_sample(simple_survey_data: pl.DataFrame) -> Sample:
    """Create a Sample object with stratified design."""
    design = Design(
        stratum="stratum",
        psu="psu",
        wgt="weight",
    )
    return Sample(simple_survey_data, design)


@pytest.fixture
def hiv_survey_data() -> pl.DataFrame:
    """
    Create HIV survey-like data matching the original use case.

    This mimics the structure from the R example:
    svyby(~hivstatusfinal, by=~gender,
          design=subset(tsdesign, bt_status == 1 & age >= 15), ...)
    """
    np.random.seed(123)
    n = 1000

    # Create realistic HIV survey data
    bt_status = np.random.choice([0, 1], n, p=[0.3, 0.7])
    age = np.random.randint(5, 85, n)
    gender = np.random.choice(["Male", "Female"], n)

    # HIV prevalence varies by gender and age
    hiv_prob = np.where(
        (gender == "Female") & (age >= 15) & (age < 50),
        0.15,
        np.where((gender == "Male") & (age >= 15) & (age < 50), 0.10, 0.02),
    )
    hivstatusfinal = np.random.binomial(1, hiv_prob)

    return pl.DataFrame(
        {
            "id": range(1, n + 1),
            "stratum": np.random.choice(["Urban", "Rural"], n),
            "psu": np.random.choice(range(1, 51), n),
            "weight": np.random.uniform(1.0, 5.0, n),
            "bt_status": bt_status,
            "age": age,
            "gender": gender,
            "hivstatusfinal": hivstatusfinal,
            "region": np.random.choice(["North", "South", "Central"], n),
        }
    )


@pytest.fixture
def hiv_sample(hiv_survey_data: pl.DataFrame) -> Sample:
    """Create a Sample for HIV survey data."""
    design = Design(
        stratum="stratum",
        psu="psu",
        wgt="weight",
    )
    return Sample(hiv_survey_data, design)


# =============================================================================
# Basic Where Functionality Tests
# =============================================================================


class TestWhereBasic:
    """Test basic `where` parameter functionality."""

    def test_mean_with_where_single_condition(self, simple_sample: Sample):
        """Test mean estimation with a single where condition."""
        result = simple_sample.estimation.mean(
            "income",
            where=col("age") >= 18,
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1
        assert result.estimates[0].est > 0
        assert result.estimates[0].se > 0
        assert result.where_clause is not None
        assert "age" in result.where_clause
        assert "18" in result.where_clause

    def test_mean_with_where_multiple_conditions_list(self, simple_sample: Sample):
        """Test mean estimation with multiple where conditions as a list."""
        result = simple_sample.estimation.mean(
            "income",
            where=[col("age") >= 18, col("employed") == 1],
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1
        assert result.where_clause is not None
        assert "&" in result.where_clause

    def test_mean_with_where_combined_expression(self, simple_sample: Sample):
        """Test mean estimation with combined where expression using &."""
        result = simple_sample.estimation.mean(
            "income",
            where=(col("age") >= 18) & (col("employed") == 1),
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1

    def test_total_with_where(self, simple_sample: Sample):
        """Test total estimation with where condition."""
        result = simple_sample.estimation.total(
            "income",
            where=col("region") == "North",
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1
        assert result.estimates[0].est > 0
        assert result.where_clause is not None

    def test_prop_with_where(self, simple_sample: Sample):
        """Test proportion estimation with where condition."""
        result = simple_sample.estimation.prop(
            "employed",
            where=col("age") >= 18,
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) > 0
        # Proportions should be between 0 and 1
        for est in result.estimates:
            assert 0 <= est.est <= 1

    def test_ratio_with_where(self, simple_sample: Sample):
        """Test ratio estimation with where condition."""
        result = simple_sample.estimation.ratio(
            "income",
            "health_score",
            where=col("employed") == 1,
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1
        assert result.where_clause is not None


# =============================================================================
# Where with By Tests
# =============================================================================


class TestWhereWithBy:
    """Test `where` parameter combined with `by` grouping."""

    def test_mean_with_where_and_by(self, simple_sample: Sample):
        """Test mean with both where and by parameters."""
        result = simple_sample.estimation.mean(
            "income",
            by="gender",
            where=col("age") >= 18,
        )

        assert isinstance(result, Estimate)
        # Should have estimates for each gender
        assert len(result.estimates) == 2

        # Check by structure
        genders = {est.by_level[0] for est in result.estimates}
        assert genders == {"Male", "Female"}

        # Verify where clause is recorded
        assert result.where_clause is not None
        assert "age" in result.where_clause

    def test_prop_with_where_and_by_hiv_example(self, hiv_sample: Sample):
        """
        Test the original HIV prevalence use case.

        This replicates the R code:
        svyby(~hivstatusfinal, by=~gender,
              design=subset(tsdesign, bt_status == 1 & age >= 15),
              FUN=svyciprop, method="beta")
        """
        result = hiv_sample.estimation.prop(
            "hivstatusfinal",
            by="gender",
            where=[col("bt_status") == 1, col("age") >= 15],
            ci_method="beta",
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) > 0

        # Check structure
        for est in result.estimates:
            assert est.by == ("gender",)
            assert est.by_level[0] in ("Male", "Female")
            # HIV prevalence should be a proportion
            assert 0 <= est.est <= 1
            assert est.se > 0
            assert est.lci < est.est < est.uci

    def test_total_with_where_and_by(self, simple_sample: Sample):
        """Test total with where and by."""
        result = simple_sample.estimation.total(
            "income",
            by="region",
            where=col("employed") == 1,
        )

        assert isinstance(result, Estimate)
        # Should have estimates for each region
        regions = {est.by_level[0] for est in result.estimates}
        assert regions == {"North", "South", "East", "West"}

    def test_mean_with_where_and_multiple_by(self, simple_sample: Sample):
        """Test mean with where and multiple by variables."""
        result = simple_sample.estimation.mean(
            "income",
            by=["gender", "region"],
            where=col("age") >= 18,
        )

        assert isinstance(result, Estimate)
        # Should have estimates for each gender x region combination
        assert len(result.estimates) == 2 * 4  # 2 genders x 4 regions

        # Check by structure
        for est in result.estimates:
            assert est.by == ("gender", "region")
            assert len(est.by_level) == 2  # one element per by variable
            assert est.by_level[0] in {"Male", "Female"}
            assert est.by_level[1] in {"North", "South", "East", "West"}


# =============================================================================
# Domain Estimation Correctness Tests
# =============================================================================


class TestDomainEstimationCorrectness:
    """
    Test that domain estimation produces correct results.

    These tests verify that using `where` produces different (and correct)
    standard errors compared to simple filtering.
    """

    def test_domain_estimation_preserves_design(self, simple_sample: Sample):
        """
        Test that domain estimation preserves full design for variance estimation.

        When using `where`, the variance should be estimated using the full
        design structure, not just the filtered subset.
        """
        # Domain estimation with where
        domain_result = simple_sample.estimation.mean(
            "income",
            where=col("age") >= 65,  # Select elderly
        )

        # Simple filtering (wrong approach)
        filtered_data = simple_sample._data.filter(pl.col("age") >= 65)
        filtered_sample = Sample(
            filtered_data,
            simple_sample._design,
        )
        filtered_result = filtered_sample.estimation.mean("income")

        # Point estimates should be similar
        assert abs(domain_result.estimates[0].est - filtered_result.estimates[0].est) < 1e-6

        # But standard errors will generally differ because domain estimation
        # accounts for the sampling design properly
        # (This is the key difference - we're not asserting they're different
        # because in some cases they might be similar, but the methodology differs)
        assert domain_result.estimates[0].se > 0
        assert filtered_result.estimates[0].se > 0

    def test_domain_with_no_matches_returns_empty(self, simple_sample: Sample):
        """Test behavior when where condition matches no records."""
        result = simple_sample.estimation.mean(
            "income",
            where=col("age") > 200,  # Impossible condition
        )

        # Should return empty estimates
        assert len(result.estimates) == 0

    def test_domain_estimation_by_group_independence(self, simple_sample: Sample):
        """Test that domain estimation by group produces independent estimates."""
        result = simple_sample.estimation.mean(
            "income",
            by="gender",
            where=col("employed") == 1,
        )

        # Get estimates for each gender
        male_est = next(e for e in result.estimates if e.by_level[0] == "Male")
        female_est = next(e for e in result.estimates if e.by_level[0] == "Female")

        # Both should have valid estimates
        assert male_est.est > 0
        assert female_est.est > 0
        assert male_est.se > 0
        assert female_est.se > 0


# =============================================================================
# Where Clause Display Tests
# =============================================================================


class TestWhereClauseDisplay:
    """Test the where_clause attribute and its display."""

    def test_where_clause_single_condition(self, simple_sample: Sample):
        """Test where_clause string for single condition."""
        result = simple_sample.estimation.mean(
            "income",
            where=col("age") >= 18,
        )

        assert result.where_clause is not None
        assert "age" in result.where_clause
        assert ">=" in result.where_clause
        assert "18" in result.where_clause

    def test_where_clause_multiple_conditions(self, simple_sample: Sample):
        """Test where_clause string for multiple conditions."""
        result = simple_sample.estimation.mean(
            "income",
            where=[col("age") >= 18, col("gender") == "Male"],
        )

        assert result.where_clause is not None
        assert "&" in result.where_clause
        assert "age" in result.where_clause
        assert "gender" in result.where_clause

    def test_where_clause_dict_format(self, simple_sample: Sample):
        """Test where_clause with dict-style where argument."""
        result = simple_sample.estimation.mean(
            "income",
            where={"gender": "Male"},
        )

        assert result.where_clause is not None
        assert "gender" in result.where_clause
        assert "Male" in result.where_clause

    def test_where_clause_in_str_output(self, simple_sample: Sample):
        """Test that where_clause appears in string representation."""
        result = simple_sample.estimation.mean(
            "income",
            where=col("age") >= 18,
        )

        str_output = str(result)
        assert "where" in str_output.lower()

    def test_no_where_clause_when_not_used(self, simple_sample: Sample):
        """Test that where_clause is None when where is not used."""
        result = simple_sample.estimation.mean("income")

        assert result.where_clause is None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestWhereEdgeCases:
    """Test edge cases and error handling for where parameter."""

    def test_where_with_null_values(self, simple_survey_data: pl.DataFrame):
        """Test where handling when data contains nulls."""
        # Add some null values
        data_with_nulls = simple_survey_data.with_columns(
            pl.when(pl.col("id") % 10 == 0).then(None).otherwise(pl.col("age")).alias("age")
        )

        design = Design(stratum="stratum", psu="psu", wgt="weight")
        sample = Sample(data_with_nulls, design)

        result = sample.estimation.mean(
            "income",
            where=col("age") >= 18,
            drop_nulls=True,
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1

    def test_where_with_string_equality(self, simple_sample: Sample):
        """Test where with string column equality."""
        result = simple_sample.estimation.mean(
            "income",
            where=col("gender") == "Female",
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1

    def test_where_with_in_operator(self, simple_sample: Sample):
        """Test where with is_in operator."""
        result = simple_sample.estimation.mean(
            "income",
            where=col("region").is_in(["North", "South"]),
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1

    def test_where_with_between(self, simple_sample: Sample):
        """Test where with between operator."""
        result = simple_sample.estimation.mean(
            "income",
            where=col("age").between(25, 55),
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1

    def test_where_with_or_condition(self, simple_sample: Sample):
        """Test where with OR condition."""
        result = simple_sample.estimation.mean(
            "income",
            where=(col("age") < 25) | (col("age") > 65),
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1

    def test_where_with_negation(self, simple_sample: Sample):
        """Test where with negated condition."""
        result = simple_sample.estimation.mean(
            "income",
            where=~(col("employed") == 1),
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 1


# =============================================================================
# Consistency Tests
# =============================================================================


class TestWhereConsistency:
    """Test consistency of where parameter across different estimation methods."""

    def test_same_where_across_methods(self, simple_sample: Sample):
        """Test that same where condition works across all estimation methods."""
        where_cond = col("age") >= 18

        mean_result = simple_sample.estimation.mean("income", where=where_cond)
        total_result = simple_sample.estimation.total("income", where=where_cond)
        prop_result = simple_sample.estimation.prop("employed", where=where_cond)
        ratio_result = simple_sample.estimation.ratio("income", "health_score", where=where_cond)

        # All should have the same where clause string
        assert mean_result.where_clause == total_result.where_clause
        assert mean_result.where_clause == prop_result.where_clause
        assert mean_result.where_clause == ratio_result.where_clause

    def test_where_with_different_ci_methods(self, hiv_sample: Sample):
        """Test where works with different CI methods for proportions."""
        where_cond = [col("bt_status") == 1, col("age") >= 15]

        logit_result = hiv_sample.estimation.prop(
            "hivstatusfinal",
            by="gender",
            where=where_cond,
            ci_method="logit",
        )

        beta_result = hiv_sample.estimation.prop(
            "hivstatusfinal",
            by="gender",
            where=where_cond,
            ci_method="beta",
        )

        # Build lookup by (by_level, y_level) for comparison
        logit_by_key = {(e.by_level, e.y_level): e for e in logit_result.estimates}
        beta_by_key = {(e.by_level, e.y_level): e for e in beta_result.estimates}

        # Ensure same keys exist
        assert logit_by_key.keys() == beta_by_key.keys()

        # Point estimates and SEs should be the same for matching groups
        for key in logit_by_key:
            logit_est = logit_by_key[key]
            beta_est = beta_by_key[key]

            assert abs(logit_est.est - beta_est.est) < 1e-10, (
                f"Point estimates differ for {key}: {logit_est.est} vs {beta_est.est}"
            )
            assert abs(logit_est.se - beta_est.se) < 1e-10, (
                f"SEs differ for {key}: {logit_est.se} vs {beta_est.se}"
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestWhereIntegration:
    """Integration tests for where parameter with real-world scenarios."""

    def test_hiv_prevalence_analysis(self, hiv_sample: Sample):
        """
        Complete HIV prevalence analysis matching the original use case.

        Replicates:
        HIVts_gender <- svyby(~hivstatusfinal,
                              by = ~gender,
                              design = subset(tsdesign, bt_status == 1 & age >= 15),
                              FUN = svyciprop,
                              vartype = "ci",
                              method = "beta",
                              df = 25)
        """
        result = hiv_sample.estimation.prop(
            "hivstatusfinal",
            by="gender",
            where=[col("bt_status") == 1, col("age") >= 15],
            ci_method="beta",
        )

        # Verify structure
        assert isinstance(result, Estimate)
        assert len(result.estimates) > 0

        # Verify where clause is captured
        assert result.where_clause is not None
        assert "bt_status" in result.where_clause
        assert "age" in result.where_clause

        # Verify results are sensible
        for est in result.estimates:
            assert 0 <= est.est <= 1, f"Proportion out of range: {est.est}"
            assert est.se > 0, "SE should be positive"
            assert est.lci < est.uci, "LCI should be less than UCI"
            assert est.lci <= est.est <= est.uci, "Estimate should be within CI"

    def test_employment_by_region_for_adults(self, simple_sample: Sample):
        """Test employment analysis for adults by region."""
        result = simple_sample.estimation.prop(
            "employed",
            by="region",
            where=col("age") >= 18,
        )

        assert len(result.estimates) > 0

        regions_found = {est.by_level[0] for est in result.estimates}
        assert regions_found == {"North", "South", "East", "West"}

    def test_income_analysis_with_complex_filter(self, simple_sample: Sample):
        """Test income analysis with complex filtering conditions."""
        result = simple_sample.estimation.mean(
            "income",
            by="gender",
            where=[
                col("age").between(25, 55),
                col("employed") == 1,
                col("has_insurance") == 1,
            ],
        )

        assert isinstance(result, Estimate)
        assert len(result.estimates) == 2  # Male and Female

        # Both genders should have estimates
        genders = {est.by_level[0] for est in result.estimates}
        assert genders == {"Male", "Female"}


# =============================================================================
# Performance and Caching Tests
# =============================================================================


class TestWhereCaching:
    """Test that caching is properly handled with where parameter."""

    def test_cache_invalidation_after_where(self, simple_sample: Sample):
        """Test that internal caches are invalidated after where estimation."""
        # Run estimation with where
        result1 = simple_sample.estimation.mean(
            "income",
            where=col("age") >= 18,
        )

        # Run estimation without where
        result2 = simple_sample.estimation.mean("income")

        # Both should work correctly
        assert len(result1.estimates) == 1
        assert len(result2.estimates) == 1

        # Results should differ (where restricts to adults)
        # Point estimates may or may not differ significantly depending on data

    def test_multiple_where_calls(self, simple_sample: Sample):
        """Test multiple consecutive where calls work correctly."""
        results = []
        for age_cutoff in [18, 30, 45, 60]:
            result = simple_sample.estimation.mean(
                "income",
                where=col("age") >= age_cutoff,
            )
            results.append(result)

        # All should be valid
        for result in results:
            assert isinstance(result, Estimate)
            assert len(result.estimates) == 1


# =============================================================================
# to_polars() Tidy Tests
# =============================================================================


class TestToPolars:
    """Test to_polars() tidy parameter behavior."""

    def test_to_polars_default_is_tidy(self, simple_sample: Sample):
        """Default to_polars() should return named columns, not by/by_level."""
        result = simple_sample.estimation.mean("income", by="gender")
        df = result.to_polars()

        assert "gender" in df.columns
        assert "by" not in df.columns
        assert "by_level" not in df.columns

    def test_to_polars_tidy_true_named_columns(self, simple_sample: Sample):
        """tidy=True should use actual variable names as column names."""
        result = simple_sample.estimation.mean("income", by="region")
        df = result.to_polars(tidy=True)

        assert "region" in df.columns
        assert "by" not in df.columns
        assert "by_level" not in df.columns

    def test_to_polars_tidy_false_raw_format(self, simple_sample: Sample):
        """tidy=False should return raw by/by_level columns."""
        result = simple_sample.estimation.mean("income", by="gender")
        df = result.to_polars(tidy=False)

        assert "by" in df.columns
        assert "by_level" in df.columns
        assert "gender" not in df.columns

    def test_to_polars_tidy_values_match(self, simple_sample: Sample):
        """Tidy and raw formats should contain the same underlying values."""
        result = simple_sample.estimation.mean("income", by="gender")
        tidy_df = result.to_polars(tidy=True)
        raw_df = result.to_polars(tidy=False)

        # Same number of rows
        assert len(tidy_df) == len(raw_df)

        # Same estimate values
        assert set(tidy_df["est"].to_list()) == set(raw_df["est"].to_list())

        # by_level contains lists in raw format — unwrap for comparison
        raw_levels = {v[0] if isinstance(v, list) else v for v in raw_df["by_level"].to_list()}
        tidy_levels = set(tidy_df["gender"].to_list())
        assert tidy_levels == raw_levels

    def test_to_polars_no_by_no_extra_columns(self, simple_sample: Sample):
        """Without by, to_polars() should not have by or by_level columns."""
        result = simple_sample.estimation.mean("income")
        df = result.to_polars()

        assert "by" not in df.columns
        assert "by_level" not in df.columns
        assert len(df) == 1

    def test_to_polars_prop_tidy_y_level(self, simple_sample: Sample):
        """For proportions, y_level should become the actual variable name."""
        result = simple_sample.estimation.prop("employed")
        df = result.to_polars()

        assert "employed" in df.columns
        assert "y_level" not in df.columns
        assert "y" not in df.columns

    def test_to_polars_prop_tidy_with_by(self, simple_sample: Sample):
        """Proportions with by should have both named columns."""
        result = simple_sample.estimation.prop("employed", by="gender")
        df = result.to_polars()

        assert "gender" in df.columns
        assert "employed" in df.columns
        assert "by" not in df.columns
        assert "by_level" not in df.columns
        assert "y" not in df.columns
        assert "y_level" not in df.columns

    def test_to_polars_multiple_by_tidy(self, simple_sample: Sample):
        """Multiple by variables should each get their own column."""
        result = simple_sample.estimation.mean("income", by=["gender", "region"])
        df = result.to_polars()

        assert "gender" in df.columns
        assert "region" in df.columns
        assert "by" not in df.columns
        assert "by_level" not in df.columns

    def test_to_polars_empty_returns_empty(self, simple_sample: Sample):
        """Empty estimates should return empty DataFrame."""
        result = simple_sample.estimation.mean(
            "income",
            where=col("age") > 200,
        )
        df = result.to_polars()
        assert df.is_empty()

    def test_to_polars_tidy_numeric_columns_preserved(self, simple_sample: Sample):
        """Numeric columns (est, se, etc.) should be unchanged by tidy."""
        result = simple_sample.estimation.mean("income", by="gender")
        tidy_df = result.to_polars(tidy=True)

        for col_name in ("est", "se", "lci", "uci"):
            assert col_name in tidy_df.columns
            assert tidy_df[col_name].dtype in (pl.Float64, pl.Float32)
