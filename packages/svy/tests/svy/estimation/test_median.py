"""
Tests for median estimation in svy.

These tests use a simple hardcoded dataset to verify median estimation works correctly.
R equivalent code is provided in comments for validation.

The median estimation path is:
    sample.estimation.median(...)
    → _taylor_median_polars(...)
    → rs.taylor_median(...)  [Rust backend]
    → median_variance_woodruff() in taylor.rs

R Setup Code:
-------------
```r
library(survey)

test_data <- data.frame(
    id = 1:20,
    stratum = rep(c("A", "B"), each = 10),
    psu = rep(1:4, each = 5),
    weight = c(1.5, 1.5, 1.5, 1.5, 1.5,
               2.0, 2.0, 2.0, 2.0, 2.0,
               1.8, 1.8, 1.8, 1.8, 1.8,
               2.2, 2.2, 2.2, 2.2, 2.2),
    income = c(25000, 30000, 35000, 40000, 45000,
               50000, 55000, 60000, 65000, 70000,
               28000, 33000, 38000, 43000, 48000,
               52000, 57000, 62000, 67000, 72000),
    region = c("North", "North", "South", "South", "North",
               "South", "North", "South", "North", "South",
               "North", "South", "North", "South", "North",
               "South", "North", "South", "North", "South"),
    age_group = c("Young", "Young", "Old", "Old", "Young",
                  "Old", "Young", "Old", "Young", "Old",
                  "Young", "Old", "Young", "Old", "Young",
                  "Old", "Young", "Old", "Young", "Old")
)

design <- svydesign(id = ~psu, strata = ~stratum, weights = ~weight, data = test_data)

# Simple median
svyquantile(~income, design, quantiles = 0.5, ci = TRUE)

# By region
svyby(~income, ~region, design, svyquantile, quantiles = 0.5, ci = TRUE)

# By age_group
svyby(~income, ~age_group, design, svyquantile, quantiles = 0.5, ci = TRUE)

# Domain (stratum A only)
svyquantile(~income, subset(design, stratum == "A"), quantiles = 0.5, ci = TRUE)
```
"""

import numpy as np
import polars as pl
import pytest


# ============================================================================
# Test Dataset
# ============================================================================


def create_test_data() -> pl.DataFrame:
    """Create a simple test dataset for median estimation tests."""
    return pl.DataFrame(
        {
            "id": list(range(1, 21)),
            "stratum": ["A"] * 10 + ["B"] * 10,
            "psu": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
            "weight": [1.5] * 5 + [2.0] * 5 + [1.8] * 5 + [2.2] * 5,
            "income": [
                25000,
                30000,
                35000,
                40000,
                45000,
                50000,
                55000,
                60000,
                65000,
                70000,
                28000,
                33000,
                38000,
                43000,
                48000,
                52000,
                57000,
                62000,
                67000,
                72000,
            ],
            "region": [
                "North",
                "North",
                "South",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
                "North",
                "South",
            ],
            "age_group": [
                "Young",
                "Young",
                "Old",
                "Old",
                "Young",
                "Old",
                "Young",
                "Old",
                "Young",
                "Old",
                "Young",
                "Old",
                "Young",
                "Old",
                "Young",
                "Old",
                "Young",
                "Old",
                "Young",
                "Old",
            ],
        }
    )


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_data() -> pl.DataFrame:
    return create_test_data()


@pytest.fixture
def sample(test_data):
    import svy

    design = svy.Design(stratum="stratum", psu="psu", wgt="weight")
    return svy.Sample(data=test_data, design=design)


# ============================================================================
# Basic Median Tests
# ============================================================================


class TestMedianBasic:
    """Basic tests for median estimation (Rust backend via rs.taylor_median)."""

    def test_median_simple(self, sample):
        """
        Test simple median estimation without grouping.

        R: svyquantile(~income, design, quantiles = 0.5, ci = TRUE)
        """
        result = sample.estimation.median("income")

        assert result is not None
        assert len(result.estimates) == 1

        est = result.estimates[0]

        assert est.y == "income"
        assert est.est is not None
        assert est.se is not None
        assert est.se >= 0
        assert est.lci is not None
        assert est.uci is not None
        assert est.lci <= est.est <= est.uci

        # Weighted median should be around 48000-52000 for this dataset
        assert 40000 <= est.est <= 60000, f"Median {est.est} outside expected range"

    def test_median_quantile_methods(self, sample):
        """
        Test median with different quantile interpolation methods.

        R equivalents:
        - LOWER:  svyquantile(..., method="constant", f=0)
        - HIGHER: svyquantile(..., method="constant", f=1)
        - LINEAR: svyquantile(..., method="linear")
        """
        from svy.core.enumerations import QuantileMethod

        results = {}
        for method in [QuantileMethod.LOWER, QuantileMethod.HIGHER, QuantileMethod.LINEAR]:
            result = sample.estimation.median("income", q_method=method)
            results[method.name] = result.estimates[0].est

        # LOWER should be <= HIGHER
        assert results["LOWER"] <= results["HIGHER"], (
            f"LOWER ({results['LOWER']}) should be <= HIGHER ({results['HIGHER']})"
        )

    def test_median_by_single_group(self, sample):
        """
        Test median estimation with a single grouping variable.

        R: svyby(~income, ~region, design, svyquantile, quantiles = 0.5, ci = TRUE)
        """
        result = sample.estimation.median("income", by="region")

        assert len(result.estimates) == 2

        regions = {est.by_level[0] for est in result.estimates}
        assert regions == {"North", "South"}

        for est in result.estimates:
            assert est.y == "income"
            assert est.est is not None
            assert est.se is not None
            assert est.se >= 0

    def test_median_by_age_group(self, sample):
        """
        Test median estimation with age_group as grouping variable.

        R: svyby(~income, ~age_group, design, svyquantile, quantiles = 0.5, ci = TRUE)
        """
        result = sample.estimation.median("income", by="age_group")

        assert len(result.estimates) == 2

        age_groups = {est.by_level[0] for est in result.estimates}
        assert age_groups == {"Young", "Old"}

        for est in result.estimates:
            assert est.y == "income"
            assert est.est is not None
            assert est.se is not None


# ============================================================================
# Median with Domain (Where) Tests
# ============================================================================


class TestMedianDomain:
    """Tests for median estimation with domain/subpopulation restrictions."""

    def test_median_with_where_clause(self, sample):
        """
        Test median estimation with a where clause (domain estimation).

        R:
            design_subset <- subset(design, stratum == "A")
            svyquantile(~income, design_subset, quantiles = 0.5, ci = TRUE)
        """
        import svy

        result = sample.estimation.median("income", where=svy.col("stratum") == "A")

        assert result is not None
        assert len(result.estimates) == 1

        est = result.estimates[0]
        assert est.est is not None
        assert est.se is not None

        # For stratum A only, income ranges from 25000-70000
        assert 35000 <= est.est <= 60000, f"Median {est.est} outside expected range for stratum A"
        assert result.where_clause is not None

    def test_median_with_where_and_by(self, sample):
        """
        Test median estimation with both where and by.

        R:
            design_subset <- subset(design, stratum == "A")
            svyby(~income, ~region, design_subset, svyquantile, quantiles = 0.5, ci = TRUE)
        """
        import svy

        result = sample.estimation.median("income", by="region", where=svy.col("stratum") == "A")

        assert result is not None
        assert len(result.estimates) >= 1

        for est in result.estimates:
            assert est.est is not None
            assert est.se is not None


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestMedianEdgeCases:
    """Tests for edge cases and error handling in median estimation."""

    def test_median_with_nulls_raises(self, test_data):
        """Test that median raises error when nulls present and drop_nulls=False."""
        import svy

        data_with_nulls = test_data.with_columns(
            pl.when(pl.col("id") <= 2).then(None).otherwise(pl.col("income")).alias("income")
        )

        design = svy.Design(stratum="stratum", psu="psu", wgt="weight")
        sample = svy.Sample(data=data_with_nulls, design=design)

        with pytest.raises(ValueError, match="Missing or invalid values"):
            sample.estimation.median("income", drop_nulls=False)

    def test_median_with_nulls_drop(self, test_data):
        """Test that median works when drop_nulls=True."""
        import svy

        data_with_nulls = test_data.with_columns(
            pl.when(pl.col("id") <= 2).then(None).otherwise(pl.col("income")).alias("income")
        )

        design = svy.Design(stratum="stratum", psu="psu", wgt="weight")
        sample = svy.Sample(data=data_with_nulls, design=design)

        result = sample.estimation.median("income", drop_nulls=True)
        assert result is not None
        assert len(result.estimates) == 1
        assert result.estimates[0].est is not None

    def test_median_nonexistent_column(self, sample):
        """Test that median raises error for nonexistent column."""
        with pytest.raises((KeyError, pl.exceptions.ColumnNotFoundError)):
            sample.estimation.median("nonexistent_column")


# ============================================================================
# Design Effect Tests
# ============================================================================


class TestMedianDeff:
    """Tests for design effect calculation in median estimation."""

    @pytest.mark.skip(reason="deff not yet supported for median/quantile estimation")
    def test_median_with_deff(self, sample):
        """Test median estimation with design effect calculation."""
        result = sample.estimation.median("income", deff=True)

        assert result is not None
        assert len(result.estimates) == 1


# ============================================================================
# R Comparison Tests
# ============================================================================


class TestMedianRComparison:
    """Tests that output values for manual comparison with R."""

    def test_output_for_r_comparison(self, sample):
        """
        Output all test results for manual R comparison.

        Run in R:
        ```r
        library(survey)
        test_data <- data.frame(
            id = 1:20,
            stratum = rep(c("A", "B"), each = 10),
            psu = rep(1:4, each = 5),
            weight = c(1.5, 1.5, 1.5, 1.5, 1.5,
                       2.0, 2.0, 2.0, 2.0, 2.0,
                       1.8, 1.8, 1.8, 1.8, 1.8,
                       2.2, 2.2, 2.2, 2.2, 2.2),
            income = c(25000, 30000, 35000, 40000, 45000,
                       50000, 55000, 60000, 65000, 70000,
                       28000, 33000, 38000, 43000, 48000,
                       52000, 57000, 62000, 67000, 72000),
            region = c("North", "North", "South", "South", "North",
                       "South", "North", "South", "North", "South",
                       "North", "South", "North", "South", "North",
                       "South", "North", "South", "North", "South")
        )
        design <- svydesign(id = ~psu, strata = ~stratum, weights = ~weight,
                            data = test_data)
        print(svyquantile(~income, design, quantiles = 0.5, ci = TRUE))
        print(svyby(~income, ~region, design, svyquantile, quantiles = 0.5,
                     ci = TRUE))
        ```
        """
        print("\n" + "=" * 60)
        print("SVY PYTHON RESULTS - Compare with R")
        print("=" * 60)

        # Simple median
        result1 = sample.estimation.median("income")
        est1 = result1.estimates[0]
        print(f"\n1. Simple Median:")
        print(f"   Estimate: {est1.est}")
        print(f"   SE:       {est1.se}")
        print(f"   95% CI:   ({est1.lci}, {est1.uci})")

        # By region
        result2 = sample.estimation.median("income", by="region")
        print(f"\n2. Median by Region:")
        for est in result2.estimates:
            print(f"   {est.by_level[0]}: est={est.est}, se={est.se}")

        print("\n" + "=" * 60)
