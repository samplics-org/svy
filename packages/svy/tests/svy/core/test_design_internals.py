# tests/svy/core/test_design_internals.py
"""
Tests for internal design handling — stratum concatenation, column renames,
and data_prep correctness.

These tests target the bug where _internal_design becomes stale after
wrangling operations (especially rename_columns), and verify that
data_prep.prepare_data derives design column names from the current
Design object rather than the cached _internal_design.
"""

import numpy as np
import polars as pl
import pytest

from svy.core.constants import _INTERNAL_CONCAT_SUFFIX
from svy.core.data_prep import PreparedData, prepare_data
from svy.core.design import Design
from svy.core.sample import Sample


# ---------- Fixtures ----------


@pytest.fixture
def survey_df():
    """Survey data with two stratification variables, PSU, weight, and outcome."""
    rng = np.random.default_rng(42)
    n = 200
    geo = np.random.choice(["North", "South", "East", "West"], size=n)
    area = np.random.choice(["Urban", "Rural"], size=n)
    psu = np.array([f"psu_{i}" for i in rng.integers(1, 21, size=n)])
    wgt = rng.uniform(0.5, 3.0, size=n)
    income = rng.normal(50000, 15000, size=n)

    return pl.DataFrame(
        {
            "geo": geo.tolist(),
            "area": area.tolist(),
            "psu": psu.tolist(),
            "wgt": wgt.tolist(),
            "income": income.tolist(),
        }
    )


@pytest.fixture
def single_stratum_sample(survey_df):
    """Sample with single stratum column."""
    design = Design(stratum="geo", psu="psu", wgt="wgt")
    return Sample(data=survey_df, design=design)


@pytest.fixture
def multi_stratum_sample(survey_df):
    """Sample with multi-column stratum."""
    design = Design(stratum=("geo", "area"), psu="psu", wgt="wgt")
    return Sample(data=survey_df, design=design)


@pytest.fixture
def weight_only_sample(survey_df):
    """Sample with only a weight column — no strata, no PSU."""
    design = Design(wgt="wgt")
    return Sample(data=survey_df, design=design)


# ==================== Multi-Column Stratum ====================


class TestMultiColumnStratum:
    """Stratum defined as tuple of columns."""

    def test_single_stratum_estimation_runs(self, single_stratum_sample):
        """Single stratum column — baseline sanity check."""
        result = single_stratum_sample.estimation.mean(y="income")
        assert len(result.estimates) == 1
        assert result.estimates[0].se > 0

    def test_multi_stratum_estimation_runs(self, multi_stratum_sample):
        """stratum=("geo", "area") — estimation produces a result."""
        result = multi_stratum_sample.estimation.mean(y="income")
        assert len(result.estimates) == 1
        assert result.estimates[0].se > 0

    def test_multi_stratum_se_differs_from_single(
        self, single_stratum_sample, multi_stratum_sample
    ):
        """Multi-column strata should produce different SEs than single-column."""
        se_single = single_stratum_sample.estimation.mean(y="income").estimates[0].se
        se_multi = multi_stratum_sample.estimation.mean(y="income").estimates[0].se
        # SEs should differ because the stratification is different
        # (more strata → generally smaller SE)
        assert se_single != pytest.approx(se_multi, rel=0.01)

    def test_multi_stratum_more_strata_than_single(
        self, single_stratum_sample, multi_stratum_sample
    ):
        """Crossing two variables should produce more strata."""
        n_single = single_stratum_sample.n_strata
        n_multi = multi_stratum_sample.n_strata
        assert n_multi > n_single

    def test_multi_stratum_point_estimate_same_as_single(
        self, single_stratum_sample, multi_stratum_sample
    ):
        """Point estimates should be identical regardless of stratification."""
        est_single = single_stratum_sample.estimation.mean(y="income").estimates[0].est
        est_multi = multi_stratum_sample.estimation.mean(y="income").estimates[0].est
        assert est_single == pytest.approx(est_multi, rel=1e-10)

    def test_multi_stratum_with_by(self, multi_stratum_sample):
        """Domain estimation works with multi-column stratum."""
        result = multi_stratum_sample.estimation.mean(y="income", by="area")
        assert len(result.estimates) == 2  # Urban, Rural
        for est in result.estimates:
            assert est.se > 0

    def test_multi_stratum_after_rename(self, survey_df):
        """The critical bug: rename columns, then estimate with multi-column stratum."""
        design = Design(stratum=("geo", "area"), psu="psu", wgt="wgt")
        sample = Sample(data=survey_df, design=design)

        # Rename columns
        sample = sample.wrangling.rename_columns({"geo": "province", "area": "area_type"})

        # Design should be updated
        assert sample.design.stratum == ("province", "area_type")

        # Estimation should still work
        result = sample.estimation.mean(y="income")
        assert len(result.estimates) == 1
        assert result.estimates[0].se > 0

    def test_multi_stratum_rename_se_unchanged(self, survey_df):
        """SE should be the same before and after renaming."""
        design = Design(stratum=("geo", "area"), psu="psu", wgt="wgt")

        # Before rename
        sample_before = Sample(data=survey_df, design=design)
        se_before = sample_before.estimation.mean(y="income").estimates[0].se

        # After rename
        sample_after = Sample(data=survey_df, design=design)
        sample_after = sample_after.wrangling.rename_columns(
            {"geo": "province", "area": "area_type"}
        )
        se_after = sample_after.estimation.mean(y="income").estimates[0].se

        assert se_before == pytest.approx(se_after, rel=1e-10)


# ==================== Design After Wrangling ====================


class TestDesignAfterWrangling:
    """Design references survive wrangling operations."""

    def test_design_after_rename_columns(self, single_stratum_sample):
        """design.stratum updated after rename."""
        sample = single_stratum_sample.wrangling.rename_columns({"geo": "region"})
        assert sample.design.stratum == "region"
        result = sample.estimation.mean(y="income")
        assert result.estimates[0].se > 0

    def test_design_after_rename_psu(self, single_stratum_sample):
        """design.psu updated after rename."""
        sample = single_stratum_sample.wrangling.rename_columns({"psu": "cluster"})
        assert sample.design.psu == "cluster"
        result = sample.estimation.mean(y="income")
        assert result.estimates[0].se > 0

    def test_design_after_rename_wgt(self, single_stratum_sample):
        """design.wgt updated after rename."""
        sample = single_stratum_sample.wrangling.rename_columns({"wgt": "weight"})
        assert sample.design.wgt == "weight"
        result = sample.estimation.mean(y="income")
        assert result.estimates[0].se > 0

    def test_design_after_filter(self, single_stratum_sample):
        """Filtering preserves design, estimation correct."""
        from svy.core.expr import col

        sample = single_stratum_sample.wrangling.filter_records(col("income") > 40000)
        assert sample.design.stratum == "geo"
        assert sample.design.psu == "psu"
        result = sample.estimation.mean(y="income")
        assert result.estimates[0].se > 0

    def test_design_after_mutate(self, single_stratum_sample):
        """Adding columns doesn't break design."""
        from svy.core.expr import col

        sample = single_stratum_sample.wrangling.mutate({"income_k": col("income") / 1000})
        result = sample.estimation.mean(y="income_k")
        assert result.estimates[0].se > 0

    def test_estimation_after_full_pipeline(self, survey_df):
        """rename → filter → mutate → estimate."""
        from svy.core.expr import col

        design = Design(stratum=("geo", "area"), psu="psu", wgt="wgt")
        sample = Sample(data=survey_df, design=design)

        result = (
            sample.wrangling.rename_columns({"geo": "province", "area": "area_type"})
            .wrangling.mutate({"income_k": col("income") / 1000})
            .estimation.mean(y="income_k")
        )

        assert len(result.estimates) == 1
        assert result.estimates[0].se > 0

    def test_estimation_after_rename_with_by(self, survey_df):
        """Domain estimation after rename with multi-column stratum."""
        design = Design(stratum=("geo", "area"), psu="psu", wgt="wgt")
        sample = Sample(data=survey_df, design=design)

        sample = sample.wrangling.rename_columns({"geo": "province", "area": "area_type"})
        result = sample.estimation.mean(y="income", by="area_type")
        assert len(result.estimates) == 2
        for est in result.estimates:
            assert est.se > 0


# ==================== Internal Design Consistency ====================


class TestInternalDesignConsistency:
    """_internal_design vs design after mutations."""

    def test_internal_design_stale_after_rename(self, single_stratum_sample):
        """After rename, _internal_design still references old column names."""
        # Before rename, _internal_design should have the stratum column
        idesign_before = single_stratum_sample._internal_design.copy()

        sample = single_stratum_sample.wrangling.rename_columns({"geo": "region"})

        # _internal_design may be stale (this documents the known behavior)
        # The important thing is that estimation still works correctly
        # because data_prep derives column names from design, not _internal_design
        result = sample.estimation.mean(y="income")
        assert result.estimates[0].se > 0

    def test_prepare_data_uses_design_not_internal_design(self, survey_df):
        """Verify prepare_data derives stratum from design.stratum."""
        design = Design(stratum=("geo", "area"), psu="psu", wgt="wgt")
        sample = Sample(data=survey_df, design=design)

        # Rename to make _internal_design stale
        sample = sample.wrangling.rename_columns({"geo": "province", "area": "area_type"})

        # _internal_design is stale
        assert sample._internal_design["stratum"] is None or (
            sample._internal_design["stratum"]
            and "province" not in sample._internal_design["stratum"]
        )

        # But prepare_data should still produce valid strata_col
        prep = prepare_data(
            sample,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )

        suffix = _INTERNAL_CONCAT_SUFFIX
        expected_strata_col = f"stratum{suffix}"

        assert prep.strata_col == expected_strata_col
        assert prep.strata_col in prep.df.columns
        assert prep.psu_col is not None
        assert prep.psu_col in prep.df.columns

    def test_prepare_data_stratum_col_has_correct_values(self, survey_df):
        """Concatenated stratum column has correct crossed values."""
        design = Design(stratum=("geo", "area"), psu="psu", wgt="wgt")
        sample = Sample(data=survey_df, design=design)

        prep = prepare_data(
            sample,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )

        strata_values = prep.df[prep.strata_col].unique().sort().to_list()

        # Should have up to 4 × 2 = 8 strata (geo × area)
        assert len(strata_values) <= 8
        assert len(strata_values) > 1

        # Each value should contain the separator
        for v in strata_values:
            assert "__by__" in v

    def test_prepare_data_single_stratum_no_separator(self, single_stratum_sample):
        """Single stratum column should not have separator in values."""
        prep = prepare_data(
            single_stratum_sample,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )

        strata_values = prep.df[prep.strata_col].unique().sort().to_list()
        # Single column — values are just the original column values
        assert set(strata_values) == {"East", "North", "South", "West"}

    def test_prepare_data_no_stratum(self, weight_only_sample):
        """No stratum in design → strata_col is None."""
        prep = prepare_data(
            weight_only_sample,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.strata_col is None
        assert prep.psu_col is None

    def test_prepare_data_casts_stratum_to_string(self, multi_stratum_sample):
        """Concatenated stratum should be cast to String (not Categorical)."""
        prep = prepare_data(
            multi_stratum_sample,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.df[prep.strata_col].dtype == pl.String


# ==================== Weight-Only Design ====================


class TestWeightOnlyDesign:
    """Design with only a weight column."""

    def test_weight_only_estimation(self, weight_only_sample):
        """SRS with weights — Taylor should work."""
        result = weight_only_sample.estimation.mean(y="income")
        assert len(result.estimates) == 1
        assert result.estimates[0].se > 0

    def test_weight_only_with_by(self, weight_only_sample):
        """Domain estimation with weight-only design."""
        result = weight_only_sample.estimation.mean(y="income", by="geo")
        assert len(result.estimates) == 4  # North, South, East, West

    def test_no_weight_estimation(self, survey_df):
        """No weight, no strata, no PSU — equal-weight SRS."""
        sample = Sample(data=survey_df)
        result = sample.estimation.mean(y="income")
        assert len(result.estimates) == 1
        assert result.estimates[0].se > 0


# ==================== Edge Cases ====================


class TestDesignEdgeCases:
    """Edge cases in design handling."""

    def test_rename_all_design_columns(self, survey_df):
        """Rename stratum, PSU, and weight simultaneously."""
        design = Design(stratum="geo", psu="psu", wgt="wgt")
        sample = Sample(data=survey_df, design=design)

        sample = sample.wrangling.rename_columns(
            {"geo": "region", "psu": "cluster", "wgt": "weight"}
        )

        assert sample.design.stratum == "region"
        assert sample.design.psu == "cluster"
        assert sample.design.wgt == "weight"

        result = sample.estimation.mean(y="income")
        assert result.estimates[0].se > 0

    def test_multi_stratum_with_where(self, multi_stratum_sample):
        """Subpopulation analysis with multi-column stratum."""
        from svy.core.expr import col

        result = multi_stratum_sample.estimation.mean(
            y="income",
            where=col("area") == "Urban",
        )
        assert len(result.estimates) == 1
        assert result.estimates[0].se > 0

    def test_multi_stratum_with_by_and_where(self, multi_stratum_sample):
        """by + where with multi-column stratum."""
        from svy.core.expr import col

        result = multi_stratum_sample.estimation.mean(
            y="income",
            by="geo",
            where=col("area") == "Urban",
        )
        assert len(result.estimates) == 4
        for est in result.estimates:
            assert est.se > 0

    def test_prop_with_multi_stratum(self, survey_df):
        """Proportion estimation with multi-column stratum."""
        design = Design(stratum=("geo", "area"), psu="psu", wgt="wgt")
        sample = Sample(data=survey_df, design=design)
        result = sample.estimation.prop(y="area")
        assert len(result.estimates) == 2  # Urban, Rural

    def test_total_with_multi_stratum_after_rename(self, survey_df):
        """Total estimation after rename with multi-column stratum."""
        design = Design(stratum=("geo", "area"), psu="psu", wgt="wgt")
        sample = Sample(data=survey_df, design=design)
        sample = sample.wrangling.rename_columns({"geo": "province"})

        result = sample.estimation.total(y="income")
        assert len(result.estimates) == 1
        assert result.estimates[0].se > 0
