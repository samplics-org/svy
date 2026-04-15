# tests/svy/wrangling/test_wrangling_integration.py
"""Integration tests for wrangling operations - method chaining and interactions."""

import polars as pl
import pytest

from svy.core.design import Design
from svy.core.expr import col
from svy.core.sample import Sample


# ---------- Fixtures ----------


@pytest.fixture
def sample_complex():
    """Complex sample with various column types."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "First Name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
            "Age (Years)": [25, 35, None, 45, 55, 30],
            "Income": [50000, 75000, 100000, 125000, 60000, 80000],
            "Region": ["N", "S", "E", "W", "N", "S"],
            "Active?": [True, False, True, True, False, True],
        }
    )
    return Sample(df)


@pytest.fixture
def sample_with_design():
    """Sample with survey design."""
    df = pl.DataFrame(
        {
            "id": list(range(1, 13)),
            "value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            "stratum": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "psu": [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            "weight": [1.0] * 12,
        }
    )
    design = Design(stratum="stratum", psu="psu", wgt="weight")
    return Sample(df, design)


# ==================== Method Chaining ====================


def test_chain_clean_names_then_filter(sample_complex: Sample):
    out = sample_complex.wrangling.clean_names().wrangling.filter_records(col("age_years") > 30)

    assert "first_name" in out._data.columns
    assert "First Name" not in out._data.columns

    ages = out._data["age_years"].to_list()
    assert all(a is None or a > 30 for a in ages)


def test_chain_mutate_then_filter(sample_complex: Sample):
    out = (
        sample_complex.wrangling.clean_names()
        .wrangling.mutate({"income_k": col("income") / 1000})
        .wrangling.filter_records(col("income_k") > 70)
    )

    assert "income_k" in out._data.columns
    assert all(v > 70 for v in out._data["income_k"].to_list())


def test_chain_filter_then_mutate():
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    s = Sample(df)

    out = s.wrangling.filter_records(col("a") > 2).wrangling.mutate({"c": col("a") + col("b")})

    assert out._data.height == 3
    assert "c" in out._data.columns


def test_chain_multiple_mutates():
    df = pl.DataFrame({"x": [1, 2, 3]})
    s = Sample(df)

    out = s.wrangling.mutate({"y": col("x") * 2}).wrangling.mutate({"z": col("y") + col("x")})

    assert out._data["y"].to_list() == [2, 4, 6]
    assert out._data["z"].to_list() == [3, 6, 9]


def test_chain_sort_then_distinct():
    df = pl.DataFrame(
        {
            "group": ["A", "B", "A", "B", "A"],
            "value": [3, 1, 2, 4, 1],
        }
    )
    s = Sample(df)

    out = s.wrangling.order_by("value").wrangling.distinct("group", keep="first")

    assert out._data.height == 2


def test_chain_categorize_then_recode():
    df = pl.DataFrame({"value": [5, 15, 25, 35, 45]})
    s = Sample(df)

    out = s.wrangling.categorize("value", bins=[0, 20, 40, 60]).wrangling.recode(
        "svy_value_categorized", {"Low": ["(0, 20]"], "Mid": ["(20, 40]"], "High": ["(40, 60]"]}
    )

    recoded = out._data["svy_svy_value_categorized_recoded"].to_list()
    assert recoded == ["Low", "Low", "Mid", "Mid", "High"]


def test_chain_top_code_then_bottom_code():
    df = pl.DataFrame({"income": [100, 500, 1000, 2000, 5000]})
    s = Sample(df)

    out = s.wrangling.top_code({"income": 3000}, replace=True).wrangling.bottom_code(
        {"income": 300}, replace=True
    )

    assert out._data["income"].to_list() == [300, 500, 1000, 2000, 3000]


# ==================== Design Preservation ====================


def test_design_preserved_through_chain(sample_with_design: Sample):
    out = (
        sample_with_design.wrangling.mutate({"value_squared": col("value") ** 2})
        .wrangling.filter_records(col("value") > 30)
        .wrangling.order_by("value", descending=True)
    )

    assert out._design.stratum == "stratum"
    assert out._design.psu == "psu"
    assert out._design.wgt == "weight"


def test_design_updated_on_rename(sample_with_design: Sample):
    out = sample_with_design.wrangling.rename_columns(
        {"stratum": "strat", "psu": "cluster", "weight": "wt"}
    )

    assert out._design.stratum == "strat"
    assert out._design.psu == "cluster"
    assert out._design.wgt == "wt"


def test_design_cleaned_on_column_removal(sample_with_design: Sample):
    out = sample_with_design.wrangling.remove_columns(["stratum", "psu"], force=True)

    assert out._design.stratum is None
    assert out._design.psu is None
    assert out._design.wgt == "weight"


# ==================== Labels Integration ====================


def test_labels_preserved_through_filter():
    df = pl.DataFrame({"sex": [1, 2, 1, 2, 1], "age": [20, 25, 30, 35, 40]})
    s = Sample(df)

    labeled = s.wrangling.apply_labels(
        labels={"sex": "Gender"},
        categories={"sex": {1: "Male", 2: "Female"}},
    )
    filtered = labeled.wrangling.filter_records(col("age") > 25)

    meta = filtered.meta.get("sex")
    assert meta is not None
    assert meta.value_labels == {1: "Male", 2: "Female"}


def test_labels_updated_on_rename():
    df = pl.DataFrame({"old_name": [1, 2, 3]})
    s = Sample(df)

    labeled = s.wrangling.apply_labels(
        labels={"old_name": "Original Label"},
        categories={"old_name": {1: "A", 2: "B", 3: "C"}},
    )
    renamed = labeled.wrangling.rename_columns({"old_name": "new_name"})

    new_meta = renamed.meta.get("new_name")
    old_meta = renamed.meta.get("old_name")
    assert new_meta is not None
    assert old_meta is None
    assert new_meta.label == "Original Label"


# ==================== Complex Workflows ====================


def test_full_data_cleaning_workflow(sample_complex: Sample):
    """Simulate a realistic data cleaning pipeline."""
    out = (
        sample_complex.wrangling.clean_names()
        .wrangling.fill_null("age_years", strategy="mean")
        .wrangling.top_code({"income": 100000}, replace=True)
        .wrangling.categorize(
            "age_years", bins=[0, 30, 50, 100], labels=["Young", "Middle", "Senior"]
        )
        .wrangling.filter_records(col("active") == True)
        .wrangling.apply_labels(
            labels={"region": "Geographic Region"},
            categories={"region": {"N": "North", "S": "South", "E": "East", "W": "West"}},
        )
        .wrangling.order_by("income", descending=True)
    )

    assert "age_years" in out._data.columns
    assert out._data["age_years"].null_count() == 0
    assert max(out._data["income"].to_list()) <= 100000
    assert "svy_age_years_categorized" in out._data.columns
    assert all(out._data["active"].to_list())
    region_meta = out.meta.get("region")
    assert region_meta is not None
    assert region_meta.label == "Geographic Region"


def test_survey_subpopulation_analysis(sample_with_design: Sample):
    """Simulate subpopulation analysis workflow."""
    out = (
        sample_with_design.wrangling.filter_records({"stratum": ["A", "B"]})
        .wrangling.mutate({"log_value": col("value").log()})
        .wrangling.categorize("value", bins=[0, 30, 60, 200], labels=["Low", "Med", "High"])
    )

    assert out._design.stratum == "stratum"
    assert out._design.psu == "psu"
    assert set(out._data["stratum"].to_list()) <= {"A", "B"}


# ==================== Edge Cases ====================


def test_empty_dataframe_operations():
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Utf8)})
    s = Sample(df)

    out = s.wrangling.mutate({"c": col("a") * 2}).wrangling.order_by("a")

    assert out._data.height == 0
    assert "c" in out._data.columns


def test_single_row_operations():
    df = pl.DataFrame({"a": [42], "b": ["only"]})
    s = Sample(df)

    out = s.wrangling.mutate({"c": col("a") + 1}).wrangling.filter_records(col("a") > 0)

    assert out._data.height == 1
    assert out._data["c"].to_list() == [43]


def test_operations_preserve_dtypes():
    df = pl.DataFrame(
        {
            "int_col": pl.Series([1, 2, 3], dtype=pl.Int64),
            "float_col": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            "str_col": pl.Series(["a", "b", "c"], dtype=pl.Utf8),
        }
    )
    s = Sample(df)

    out = s.wrangling.filter_records(col("int_col") > 1)

    assert out._data["int_col"].dtype == pl.Int64
    assert out._data["float_col"].dtype == pl.Float64
    assert out._data["str_col"].dtype == pl.Utf8


# ==================== Return Value Consistency ====================


def test_all_methods_return_sample():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    s = Sample(df)

    assert isinstance(s.wrangling.mutate({"c": 1}), Sample)
    assert isinstance(s.wrangling.filter_records(col("a") > 0), Sample)
    assert isinstance(s.wrangling.order_by("a"), Sample)
    assert isinstance(s.wrangling.distinct(), Sample)
    assert isinstance(s.wrangling.rename_columns({"a": "x"}), Sample)

    s2 = Sample(pl.DataFrame({"a": [1, 2, 3]}))
    assert isinstance(s2.wrangling.cast("a", pl.Float64), Sample)

    s3 = Sample(pl.DataFrame({"a": [1, None, 3]}))
    assert isinstance(s3.wrangling.fill_null("a", value=0), Sample)


def test_method_chaining_returns_sample():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)

    # mutate returns new instance by default (inplace=False)
    result = s.wrangling.mutate({"b": 1})
    assert isinstance(result, Sample)
    assert "b" in result._data.columns
    assert result is not s

    # With inplace=True, returns same instance
    s2 = Sample(pl.DataFrame({"x": [1, 2, 3]}))
    result2 = s2.wrangling.mutate({"y": 1}, inplace=True)
    assert result2 is s2

    # filter_records returns new instance by default
    s3 = Sample(pl.DataFrame({"a": [1, 2, 3]}))
    result3 = s3.wrangling.filter_records(col("a") > 0)
    assert result3 is not s3
    assert isinstance(result3, Sample)

    # filter_records with inplace returns same instance
    s4 = Sample(pl.DataFrame({"a": [1, 2, 3]}))
    result4 = s4.wrangling.filter_records(col("a") > 0, inplace=True)
    assert result4 is s4


# ==================== Copy-on-write semantics ====================


def test_copy_on_write_filter_preserves_original():
    """Filtering should not modify the original sample."""
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    s = Sample(df)
    out = s.wrangling.filter_records(col("a") > 3)
    assert out._data.height == 2
    assert s._data.height == 5  # original unchanged


def test_copy_on_write_mutate_preserves_original():
    """Mutating should not modify the original sample."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.mutate({"b": col("a") * 10})
    assert "b" in out._data.columns
    assert "b" not in s._data.columns  # original unchanged


def test_copy_on_write_labels_isolated():
    """Labels applied to a fork should not affect the original."""
    df = pl.DataFrame({"x": [1, 2, 3]})
    s = Sample(df)
    labeled = s.wrangling.apply_labels(labels={"x": "My X"})
    assert labeled.meta.get("x").label == "My X"
    # Original should not have the label
    orig_meta = s.meta.get("x")
    assert orig_meta is None or orig_meta.label is None


def test_copy_on_write_rename_preserves_original():
    """Renaming should not modify the original sample."""
    df = pl.DataFrame({"old": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.rename_columns({"old": "new"})
    assert "new" in out._data.columns
    assert "old" in s._data.columns  # original unchanged
    assert "new" not in s._data.columns
