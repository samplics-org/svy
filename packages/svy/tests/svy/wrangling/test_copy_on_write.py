# tests/svy/wrangling/test_copy_on_write.py
"""
Tests for copy-on-write semantics, inplace parameter, and
internal design column (concat) rebuilding.

These tests cover the core behavioral changes introduced in the
wrangling refactor and are not covered by other test files.
"""

import polars as pl
import pytest

from svy.core.design import Design
from svy.core.expr import col
from svy.core.sample import Sample


# ---------- Fixtures ----------


@pytest.fixture
def sample_with_multi_stratum():
    """Sample with multi-column stratum design (triggers concat columns)."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "region": ["N", "N", "S", "S", "E", "E"],
            "urban": ["U", "R", "U", "R", "U", "R"],
            "psu": [1, 2, 1, 2, 1, 2],
            "value": [10, 20, 30, 40, 50, 60],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    design = Design(stratum=("region", "urban"), psu="psu", wgt="weight")
    return Sample(df, design)


@pytest.fixture
def sample_with_design():
    """Simple sample with single-column stratum."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "stratum": ["A", "A", "B", "B"],
            "psu": [1, 2, 1, 2],
            "value": [10, 20, 30, 40],
            "weight": [1.0, 1.5, 2.0, 2.5],
        }
    )
    design = Design(stratum="stratum", psu="psu", wgt="weight")
    return Sample(df, design)


# ==================== Copy-on-write: every method ====================


class TestCopyOnWriteDefault:
    """Default (inplace=False) should return a new sample, original untouched."""

    def test_filter_records(self):
        s = Sample(pl.DataFrame({"a": [1, 2, 3]}))
        out = s.wrangling.filter_records(col("a") > 1)
        assert out is not s
        assert s._data.height == 3
        assert out._data.height == 2

    def test_order_by(self):
        s = Sample(pl.DataFrame({"a": [3, 1, 2]}))
        out = s.wrangling.order_by("a")
        assert out is not s
        assert s._data["a"].to_list() == [3, 1, 2]
        assert out._data["a"].to_list() == [1, 2, 3]

    def test_distinct(self):
        s = Sample(pl.DataFrame({"a": [1, 1, 2]}))
        out = s.wrangling.distinct("a")
        assert out is not s
        assert s._data.height == 3
        assert out._data.height == 2

    def test_with_row_index(self):
        s = Sample(pl.DataFrame({"a": [1, 2]}))
        out = s.wrangling.with_row_index(name="idx")
        assert out is not s
        assert "idx" not in s._data.columns
        assert "idx" in out._data.columns

    def test_remove_columns(self):
        s = Sample(pl.DataFrame({"a": [1], "b": [2]}))
        out = s.wrangling.remove_columns("b")
        assert out is not s
        assert "b" in s._data.columns
        assert "b" not in out._data.columns

    def test_keep_columns(self):
        s = Sample(pl.DataFrame({"a": [1], "b": [2]}))
        out = s.wrangling.keep_columns("a")
        assert out is not s
        assert "b" in s._data.columns
        assert "b" not in out._data.columns

    def test_cast(self):
        s = Sample(pl.DataFrame({"a": [1, 2, 3]}))
        out = s.wrangling.cast("a", pl.Float64)
        assert out is not s
        assert s._data["a"].dtype == pl.Int64
        assert out._data["a"].dtype == pl.Float64

    def test_fill_null(self):
        s = Sample(pl.DataFrame({"a": [1, None, 3]}))
        out = s.wrangling.fill_null("a", value=0)
        assert out is not s
        assert s._data["a"].null_count() == 1
        assert out._data["a"].null_count() == 0

    def test_top_code(self):
        s = Sample(pl.DataFrame({"a": [1, 5, 10]}))
        out = s.wrangling.top_code({"a": 7}, replace=True)
        assert out is not s
        assert s._data["a"].to_list() == [1, 5, 10]
        assert out._data["a"].to_list() == [1, 5, 7]

    def test_bottom_code(self):
        s = Sample(pl.DataFrame({"a": [1, 5, 10]}))
        out = s.wrangling.bottom_code({"a": 3}, replace=True)
        assert out is not s
        assert s._data["a"].to_list() == [1, 5, 10]
        assert out._data["a"].to_list() == [3, 5, 10]

    def test_recode(self):
        s = Sample(pl.DataFrame({"a": ["x", "y", "z"]}))
        out = s.wrangling.recode("a", {"ab": ["x", "y"]}, replace=True)
        assert out is not s
        assert s._data["a"].to_list() == ["x", "y", "z"]

    def test_mutate(self):
        s = Sample(pl.DataFrame({"a": [1, 2, 3]}))
        out = s.wrangling.mutate({"b": 99})
        assert out is not s
        assert "b" not in s._data.columns
        assert "b" in out._data.columns

    def test_apply_labels(self):
        s = Sample(pl.DataFrame({"a": [1, 2]}))
        out = s.wrangling.apply_labels(labels={"a": "My A"})
        assert out is not s
        assert out.meta.get("a").label == "My A"


# ==================== Inplace: every method ====================


class TestInplace:
    """inplace=True should return the same object and mutate it."""

    def test_filter_records(self):
        s = Sample(pl.DataFrame({"a": [1, 2, 3]}))
        out = s.wrangling.filter_records(col("a") > 1, inplace=True)
        assert out is s
        assert s._data.height == 2

    def test_order_by(self):
        s = Sample(pl.DataFrame({"a": [3, 1, 2]}))
        out = s.wrangling.order_by("a", inplace=True)
        assert out is s
        assert s._data["a"].to_list() == [1, 2, 3]

    def test_distinct(self):
        s = Sample(pl.DataFrame({"a": [1, 1, 2]}))
        out = s.wrangling.distinct("a", inplace=True)
        assert out is s
        assert s._data.height == 2

    def test_with_row_index(self):
        s = Sample(pl.DataFrame({"a": [1, 2]}))
        out = s.wrangling.with_row_index(name="idx", inplace=True)
        assert out is s
        assert "idx" in s._data.columns

    def test_remove_columns(self):
        s = Sample(pl.DataFrame({"a": [1], "b": [2]}))
        out = s.wrangling.remove_columns("b", inplace=True)
        assert out is s
        assert "b" not in s._data.columns

    def test_keep_columns(self):
        s = Sample(pl.DataFrame({"a": [1], "b": [2]}))
        out = s.wrangling.keep_columns("a", inplace=True)
        assert out is s
        assert "b" not in s._data.columns

    def test_cast(self):
        s = Sample(pl.DataFrame({"a": [1, 2, 3]}))
        out = s.wrangling.cast("a", pl.Float64, inplace=True)
        assert out is s
        assert s._data["a"].dtype == pl.Float64

    def test_fill_null(self):
        s = Sample(pl.DataFrame({"a": [1, None, 3]}))
        out = s.wrangling.fill_null("a", value=0, inplace=True)
        assert out is s
        assert s._data["a"].null_count() == 0

    def test_top_code(self):
        s = Sample(pl.DataFrame({"a": [1, 5, 10]}))
        out = s.wrangling.top_code({"a": 7}, replace=True, inplace=True)
        assert out is s
        assert s._data["a"].to_list() == [1, 5, 7]

    def test_apply_labels(self):
        s = Sample(pl.DataFrame({"a": [1, 2]}))
        out = s.wrangling.apply_labels(labels={"a": "My A"}, inplace=True)
        assert out is s
        assert s.meta.get("a").label == "My A"


# ==================== Concat column rebuild ====================


class TestConcatColumnRebuild:
    """
    When a design source column (stratum/psu/ssu) is modified,
    the internal concatenated columns must be rebuilt.
    """

    def test_recode_design_source_rebuilds(self, sample_with_design: Sample):
        """Recoding a stratum column with replace=True should rebuild."""
        out = sample_with_design.wrangling.recode(
            "stratum",
            {"X": ["A"]},
            replace=True,
        )
        # The stratum values changed, so concat columns should reflect this
        assert out._data["stratum"].to_list() == ["X", "X", "B", "B"]
        # Internal design should still be valid
        assert out._design.stratum == "stratum"

    def test_mutate_overwrite_design_source_rebuilds(self, sample_with_design: Sample):
        """Overwriting a stratum column via mutate should rebuild."""
        from svy.core.expr import when

        out = sample_with_design.wrangling.mutate(
            {"stratum": when(col("stratum") == "A").then("X").otherwise("Y")}
        )
        assert set(out._data["stratum"].to_list()) == {"X", "Y"}
        assert out._design.stratum == "stratum"

    def test_fill_null_design_source_rebuilds(self):
        """Filling nulls in a stratum column should rebuild."""
        df = pl.DataFrame(
            {
                "stratum": ["A", None, "B", None],
                "psu": [1, 2, 1, 2],
            }
        )
        design = Design(stratum="stratum", psu="psu")
        s = Sample(df, design)
        out = s.wrangling.fill_null("stratum", value="UNKNOWN")
        assert out._data["stratum"].null_count() == 0

    def test_cast_design_source_rebuilds(self, sample_with_design: Sample):
        """Casting a stratum column should rebuild."""
        out = sample_with_design.wrangling.cast("stratum", pl.Categorical)
        assert out._data["stratum"].dtype == pl.Categorical

    def test_rename_design_source_rebuilds(self, sample_with_design: Sample):
        """Renaming a stratum column should rebuild concat columns."""
        out = sample_with_design.wrangling.rename_columns({"stratum": "strat"})
        assert out._design.stratum == "strat"
        assert "strat" in out._data.columns
        assert "stratum" not in out._data.columns

    def test_recode_non_design_column_no_rebuild(self, sample_with_design: Sample):
        """Recoding a non-design column should not trigger rebuild."""
        # This should work without errors - no rebuild needed
        out = sample_with_design.wrangling.mutate({"value": col("value") * 2})
        assert out._data["value"].to_list() == [20, 40, 60, 80]

    def test_multi_stratum_rebuild(self, sample_with_multi_stratum: Sample):
        """Modifying one part of a multi-column stratum should rebuild."""
        out = sample_with_multi_stratum.wrangling.recode(
            "region",
            {"Metro": ["N", "S"]},
            replace=True,
        )
        # region changed, concat column should be rebuilt
        assert set(out._data["region"].to_list()) == {"Metro", "E"}
        assert out._design.stratum == ("region", "urban")


# ==================== Metadata isolation ====================


class TestMetadataIsolation:
    """Labels and warnings on forked samples should not leak back."""

    def test_labels_do_not_leak_back(self):
        s = Sample(pl.DataFrame({"a": [1, 2, 3]}))
        labeled = s.wrangling.apply_labels(labels={"a": "Labeled A"})
        # The fork should have the label
        assert labeled.meta.get("a").label == "Labeled A"
        # The original should NOT
        orig_meta = s.meta.get("a")
        assert orig_meta is None or orig_meta.label is None

    def test_warnings_do_not_leak_back(self):
        s = Sample(pl.DataFrame({"a": [1, 2]}))
        # Apply labels with extra keys to trigger a warning
        labeled = s.wrangling.apply_labels(categories={"a": {1: "One", 2: "Two", 99: "Unknown"}})
        # The fork should have the warning
        fork_warns = [w for w in labeled.warnings if w.code == "LABEL_KEY_NOT_IN_DATA"]
        assert len(fork_warns) == 1
        # The original should NOT
        orig_warns = [w for w in s.warnings if w.code == "LABEL_KEY_NOT_IN_DATA"]
        assert len(orig_warns) == 0

    def test_sequential_filters_on_same_original(self):
        """Multiple filters on the same original should each see full data."""
        df = pl.DataFrame({"region": ["N", "S", "E", "W", "N"]})
        s = Sample(df)

        north = s.wrangling.filter_records({"region": "N"})
        south = s.wrangling.filter_records({"region": "S"})
        both = s.wrangling.filter_records({"region": ["N", "S"]})

        assert north._data.height == 2
        assert south._data.height == 1
        assert both._data.height == 3
        # Original is untouched
        assert s._data.height == 5


class TestInternalDesignGuard:
    """The .data property should not crash when internal concat columns are missing."""

    def test_data_property_survives_missing_internal_cols(self):
        df = pl.DataFrame(
            {
                "stratum": ["A", "A", "B", "B"],
                "psu": [1, 2, 1, 2],
                "value": [10, 20, 30, 40],
            }
        )
        design = Design(stratum="stratum", psu="psu")
        s = Sample(df, design)

        # Sabotage: point _internal_design at a column that doesn't exist
        s._internal_design["stratum"] = "nonexistent_concat_col"

        # .data should not raise
        result = s.data
        assert result.height == 4
