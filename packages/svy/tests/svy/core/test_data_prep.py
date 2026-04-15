# tests/svy/core/test_data_prep.py
"""
Tests for svy.core.data_prep.prepare_data.

These tests verify the critical behaviors of the unified data preparation
function, particularly:

1. drop_nulls only checks analysis-relevant columns, not all columns
2. select_columns is an optimization flag that does not change results
3. Design column resolution works after renames
4. Weight column handling (missing weight, zero weights, casting)
5. Where clause and domain column creation
6. By column resolution and concatenation
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy.core.constants import _INTERNAL_CONCAT_SUFFIX
from svy.core.data_prep import PreparedData, extract_where_cols, prepare_data
from svy.core.design import Design
from svy.core.sample import Sample


# ---------- Fixtures ----------


@pytest.fixture
def base_df():
    """DataFrame with analysis columns and some irrelevant columns with nulls."""
    n = 100
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "y_var": rng.normal(100, 10, n).tolist(),
            "x_var": rng.normal(50, 5, n).tolist(),
            "group_var": (["A"] * 50 + ["B"] * 50),
            "stratum": (["S1"] * 25 + ["S2"] * 25 + ["S3"] * 25 + ["S4"] * 25),
            "psu": [f"psu_{i % 10}" for i in range(n)],
            "wgt": rng.uniform(1.0, 5.0, n).tolist(),
            # Irrelevant columns with nulls — should NOT cause row drops
            "irrelevant_with_nulls": [None] * n,
            "rep_wgt1": [None if i % 3 == 0 else 1.0 for i in range(n)],
            "rep_wgt2": [None if i % 5 == 0 else 2.0 for i in range(n)],
            "notes": [None if i % 2 == 0 else "ok" for i in range(n)],
        }
    )


@pytest.fixture
def sample_with_nulls(base_df):
    """Sample with irrelevant null columns — should not affect estimation."""
    design = Design(stratum="stratum", psu="psu", wgt="wgt")
    return Sample(data=base_df, design=design)


@pytest.fixture
def sample_clean():
    """Sample with no nulls anywhere."""
    n = 50
    rng = np.random.default_rng(99)
    df = pl.DataFrame(
        {
            "income": rng.normal(50000, 10000, n).tolist(),
            "age": rng.integers(20, 70, n).tolist(),
            "region": (["North"] * 25 + ["South"] * 25),
            "cluster": [f"c_{i % 5}" for i in range(n)],
            "weight": rng.uniform(1.0, 3.0, n).tolist(),
        }
    )
    design = Design(stratum="region", psu="cluster", wgt="weight")
    return Sample(data=df, design=design)


@pytest.fixture
def sample_no_design():
    """Sample with no design columns."""
    n = 30
    rng = np.random.default_rng(77)
    df = pl.DataFrame(
        {
            "y": rng.normal(10, 2, n).tolist(),
            "x": rng.normal(5, 1, n).tolist(),
        }
    )
    return Sample(data=df)


# ==================== Null Handling ====================


class TestDropNullsScope:
    """drop_nulls should only check analysis-relevant columns."""

    def test_irrelevant_nulls_do_not_drop_rows(self, sample_with_nulls):
        """Nulls in irrelevant columns should NOT cause rows to be dropped."""
        prep = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        # All 100 rows should survive — nulls are only in irrelevant columns
        assert prep.df.height == 100

    def test_relevant_nulls_do_drop_rows(self, base_df):
        """Nulls in the y column should cause those rows to be dropped."""
        # Inject nulls into y_var
        df = base_df.with_columns(
            pl.when(pl.col("y_var") > 110).then(None).otherwise(pl.col("y_var")).alias("y_var")
        )
        design = Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = Sample(data=df, design=design)

        n_nulls = df["y_var"].null_count()
        assert n_nulls > 0  # sanity check

        prep = prepare_data(
            sample,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep.df.height == 100 - n_nulls

    def test_nulls_in_x_column_drop_rows(self, base_df):
        """Nulls in extra_cols (x variables) should cause drops."""
        df = base_df.with_columns(
            pl.when(pl.col("x_var") > 55).then(None).otherwise(pl.col("x_var")).alias("x_var")
        )
        design = Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = Sample(data=df, design=design)

        n_nulls = df["x_var"].null_count()
        assert n_nulls > 0

        prep = prepare_data(
            sample,
            y="y_var",
            extra_cols=["x_var"],
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep.df.height == 100 - n_nulls

    def test_nulls_in_by_column_drop_rows(self, base_df):
        """Nulls in by column should cause drops."""
        df = base_df.with_columns(
            pl.when(pl.arange(0, pl.len()) < 5)
            .then(None)
            .otherwise(pl.col("group_var"))
            .alias("group_var")
        )
        design = Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = Sample(data=df, design=design)

        prep = prepare_data(
            sample,
            y="y_var",
            by="group_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep.df.height == 95

    def test_nulls_in_weight_column_drop_rows(self, base_df):
        """Nulls in the weight column should cause drops."""
        df = base_df.with_columns(
            pl.when(pl.arange(0, pl.len()) < 3).then(None).otherwise(pl.col("wgt")).alias("wgt")
        )
        design = Design(stratum="stratum", psu="psu", wgt="wgt")
        sample = Sample(data=df, design=design)

        prep = prepare_data(
            sample,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep.df.height == 97


# ==================== select_columns Invariance ====================


class TestSelectColumnsInvariance:
    """select_columns should not change analytical results."""

    def test_same_row_count(self, sample_with_nulls):
        """Same number of rows regardless of select_columns."""
        prep_true = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        prep_false = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep_true.df.height == prep_false.df.height

    def test_same_y_values(self, sample_with_nulls):
        """Same y column values regardless of select_columns."""
        prep_true = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        prep_false = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep_true.df["y_var"].to_list() == prep_false.df["y_var"].to_list()

    def test_same_weight_values(self, sample_with_nulls):
        """Same weight column values regardless of select_columns."""
        prep_true = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        prep_false = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep_true.df[prep_true.weight_col].to_list() == pytest.approx(
            prep_false.df[prep_false.weight_col].to_list()
        )

    def test_same_strata_col(self, sample_with_nulls):
        """Same strata_col resolution regardless of select_columns."""
        prep_true = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        prep_false = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep_true.strata_col == prep_false.strata_col
        assert prep_true.psu_col == prep_false.psu_col

    def test_select_true_has_fewer_columns(self, sample_with_nulls):
        """select_columns=True should produce fewer columns."""
        prep_true = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        prep_false = prepare_data(
            sample_with_nulls,
            y="y_var",
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert len(prep_true.df.columns) < len(prep_false.df.columns)

    def test_with_extra_cols(self, sample_with_nulls):
        """extra_cols are preserved with both settings."""
        prep_true = prepare_data(
            sample_with_nulls,
            y="y_var",
            extra_cols=["x_var", "group_var"],
            drop_nulls=True,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        prep_false = prepare_data(
            sample_with_nulls,
            y="y_var",
            extra_cols=["x_var", "group_var"],
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert "x_var" in prep_true.df.columns
        assert "group_var" in prep_true.df.columns
        assert prep_true.df.height == prep_false.df.height


# ==================== Design Column Resolution ====================


class TestDesignColumnResolution:
    """Design columns are resolved from the current Design, not _internal_design."""

    def test_single_stratum(self, sample_clean):
        """Single stratum produces correct strata_col."""
        prep = prepare_data(
            sample_clean,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        suffix = _INTERNAL_CONCAT_SUFFIX
        assert prep.strata_col == f"stratum{suffix}"
        assert prep.psu_col == f"psu{suffix}"

    def test_no_stratum(self, sample_no_design):
        """No stratum in design → strata_col is None."""
        prep = prepare_data(
            sample_no_design,
            y="y",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.strata_col is None
        assert prep.psu_col is None

    def test_strata_col_cast_to_string(self, sample_clean):
        """Strata column is cast to String for Rust backend."""
        prep = prepare_data(
            sample_clean,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.df[prep.strata_col].dtype == pl.String

    def test_psu_col_cast_to_string(self, sample_clean):
        """PSU column is cast to String for Rust backend."""
        prep = prepare_data(
            sample_clean,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.df[prep.psu_col].dtype == pl.String


# ==================== Weight Handling ====================


class TestWeightHandling:
    def test_weight_column_resolved(self, sample_clean):
        """Weight column name comes from design."""
        prep = prepare_data(
            sample_clean,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.weight_col == "weight"

    def test_no_weight_creates_ones(self, sample_no_design):
        """No weight in design → creates __svy_ones__ column."""
        prep = prepare_data(
            sample_no_design,
            y="y",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.weight_col == "__svy_ones__"
        assert prep.df[prep.weight_col].to_list() == [1.0] * 30

    def test_weight_cast_to_float64(self):
        """Integer weights are cast to Float64."""
        df = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "w": [1, 2, 3],  # integer weights
            }
        )
        sample = Sample(data=df, design=Design(wgt="w"))
        prep = prepare_data(
            sample,
            y="y",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.df[prep.weight_col].dtype == pl.Float64


# ==================== Y Column Handling ====================


class TestYColumnHandling:
    def test_y_cast_to_float64(self, sample_clean):
        """y column is cast to Float64 when cast_y_float=True."""
        prep = prepare_data(
            sample_clean,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.df["income"].dtype == pl.Float64

    def test_y_not_cast_when_false(self):
        """y column is NOT cast when cast_y_float=False."""
        df = pl.DataFrame(
            {
                "category": ["A", "B", "C", "A", "B"],
                "w": [1.0, 2.0, 3.0, 1.0, 2.0],
            }
        )
        sample = Sample(data=df, design=Design(wgt="w"))
        prep = prepare_data(
            sample,
            y="category",
            drop_nulls=False,
            cast_y_float=False,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.df["category"].dtype == pl.String


# ==================== By Column Resolution ====================


class TestByColumnResolution:
    def test_single_by(self, sample_clean):
        """Single by column produces correct by_col name."""
        prep = prepare_data(
            sample_clean,
            y="income",
            by="region",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.by_col == f"by{_INTERNAL_CONCAT_SUFFIX}"
        assert prep.by_cols == ["region"]

    def test_no_by(self, sample_clean):
        """No by → by_col is None."""
        prep = prepare_data(
            sample_clean,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.by_col is None
        assert prep.by_cols == []

    def test_multi_by(self):
        """Multiple by columns produce concatenated by_col."""
        df = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "g1": ["A", "A", "B", "B"],
                "g2": ["X", "Y", "X", "Y"],
                "w": [1.0, 1.0, 1.0, 1.0],
            }
        )
        sample = Sample(data=df, design=Design(wgt="w"))
        prep = prepare_data(
            sample,
            y="y",
            by=("g1", "g2"),
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.by_col == f"by{_INTERNAL_CONCAT_SUFFIX}"
        assert prep.by_cols == ["g1", "g2"]


# ==================== Where Clause ====================


class TestWhereClause:
    def test_where_creates_domain_col(self, sample_clean):
        """Where clause creates __svy_domain__ column."""
        from svy.core.expr import col

        prep = prepare_data(
            sample_clean,
            y="income",
            where=col("region") == "North",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep.domain_col == "__svy_domain__"
        assert prep.domain_val == "true"

    def test_where_zeros_excluded_weights(self, sample_clean):
        """Where clause sets excluded observations' weights to zero."""
        from svy.core.expr import col

        prep = prepare_data(
            sample_clean,
            y="income",
            where=col("region") == "North",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        # South observations should have zero weight
        weights = prep.df[prep.weight_col].to_numpy()
        domain = prep.df["__svy_domain__"].to_list()

        n_zero = (weights == 0.0).sum()
        n_excluded = sum(1 for d in domain if d != "true")
        assert n_zero == n_excluded
        assert n_zero == 25  # 25 South observations

    def test_where_preserves_all_rows(self, sample_clean):
        """Where clause preserves all rows (design-based subpopulation)."""
        from svy.core.expr import col

        prep = prepare_data(
            sample_clean,
            y="income",
            where=col("region") == "North",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep.df.height == 50  # all rows preserved

    def test_no_where(self, sample_clean):
        """No where clause → no domain column."""
        prep = prepare_data(
            sample_clean,
            y="income",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        assert prep.domain_col is None
        assert prep.domain_val is None


# ==================== extract_where_cols ====================


class TestExtractWhereCols:
    def test_none(self):
        assert extract_where_cols(None) == []

    def test_dict(self):
        cols = extract_where_cols({"sex": 2, "region": "North"})
        assert set(cols) == {"sex", "region"}

    def test_polars_expr(self):
        cols = extract_where_cols(pl.col("age") > 30)
        assert cols == ["age"]

    def test_sequence_of_exprs(self):
        cols = extract_where_cols([pl.col("age") > 30, pl.col("sex") == "M"])
        assert set(cols) == {"age", "sex"}

    def test_empty_dict(self):
        assert extract_where_cols({}) == []


# ==================== Paired Difference ====================


class TestPairedDifference:
    def test_y_pair_creates_diff_column(self):
        """y_pair creates a difference column."""
        df = pl.DataFrame(
            {
                "before": [10.0, 20.0, 30.0],
                "after": [12.0, 18.0, 35.0],
                "w": [1.0, 1.0, 1.0],
            }
        )
        sample = Sample(data=df, design=Design(wgt="w"))
        prep = prepare_data(
            sample,
            y="before",
            y_pair="after",
            drop_nulls=False,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep.y_col.startswith("__svy_before_minus_after")
        expected_diff = [-2.0, 2.0, -5.0]
        actual_diff = prep.df[prep.y_col].to_list()
        assert actual_diff == pytest.approx(expected_diff)


# ==================== GLM Regression Scenario ====================


class TestGLMScenario:
    """Simulate the GLM use case that was failing."""

    def test_many_null_columns_do_not_break_glm(self):
        """GLM-like scenario: many irrelevant columns with nulls."""
        n = 200
        rng = np.random.default_rng(42)

        data = {
            "y": rng.normal(100, 10, n).tolist(),
            "x1": rng.normal(50, 5, n).tolist(),
            "x2": rng.normal(25, 3, n).tolist(),
            "wgt": rng.uniform(1.0, 5.0, n).tolist(),
        }
        # Add 500 "replicate weight" columns with nulls
        for i in range(1, 501):
            data[f"rep_wgt{i}"] = [None if j % 7 == 0 else 1.0 for j in range(n)]

        df = pl.DataFrame(data)
        sample = Sample(data=df, design=Design(wgt="wgt"))

        # select_columns=False (GLM default) should still work
        prep = prepare_data(
            sample,
            y="y",
            extra_cols=["x1", "x2"],
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep.df.height == n  # no rows dropped

    def test_many_null_columns_select_true_same_result(self):
        """select_columns=True and False give same row count with null columns."""
        n = 100
        rng = np.random.default_rng(42)

        data = {
            "y": rng.normal(100, 10, n).tolist(),
            "x1": rng.normal(50, 5, n).tolist(),
            "wgt": rng.uniform(1.0, 5.0, n).tolist(),
        }
        for i in range(1, 101):
            data[f"junk_{i}"] = [None] * n  # entirely null columns

        df = pl.DataFrame(data)
        sample = Sample(data=df, design=Design(wgt="wgt"))

        prep_true = prepare_data(
            sample,
            y="y",
            extra_cols=["x1"],
            drop_nulls=True,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        prep_false = prepare_data(
            sample,
            y="y",
            extra_cols=["x1"],
            drop_nulls=True,
            cast_y_float=True,
            select_columns=False,
            apply_singleton_filter=False,
        )
        assert prep_true.df.height == prep_false.df.height == n
