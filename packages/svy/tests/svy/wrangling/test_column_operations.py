# tests/svy/wrangling/test_column_operations.py
"""Tests for column manipulation operations: remove_columns, keep_columns, cast, fill_null."""

import polars as pl
import pytest

from svy.core.design import Design
from svy.core.sample import Sample
from svy.errors import MethodError


# ---------- Fixtures ----------


@pytest.fixture
def sample_basic():
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": ["x", "y", "z"],
        }
    )
    return Sample(df)


@pytest.fixture
def sample_with_design():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "value": [100, 200, 300, 400],
            "stratum": ["A", "A", "B", "B"],
            "psu": [1, 2, 1, 2],
            "weight": [1.0, 1.5, 2.0, 2.5],
        }
    )
    design = Design(stratum="stratum", psu="psu", wgt="weight")
    return Sample(df, design)


@pytest.fixture
def sample_with_nulls():
    df = pl.DataFrame(
        {
            "x": [1, None, 3, None, 5],
            "y": [None, 2.0, None, 4.0, None],
            "z": ["a", None, "c", None, "e"],
        }
    )
    return Sample(df)


# ==================== remove_columns / drop ====================


def test_remove_columns_single_column(sample_basic: Sample):
    out = sample_basic.wrangling.remove_columns("c")
    assert "c" not in out._data.columns
    assert "a" in out._data.columns
    assert "b" in out._data.columns


def test_remove_columns_multiple_columns(sample_basic: Sample):
    out = sample_basic.wrangling.remove_columns(["a", "c"])
    assert "a" not in out._data.columns
    assert "c" not in out._data.columns
    assert "b" in out._data.columns


def test_remove_columns_returns_new_instance(sample_basic: Sample):
    out = sample_basic.wrangling.remove_columns("c")
    assert out is not sample_basic
    # Original unchanged
    assert "c" in sample_basic._data.columns


def test_remove_columns_inplace(sample_basic: Sample):
    out = sample_basic.wrangling.remove_columns("c", inplace=True)
    assert out is sample_basic
    assert "c" not in sample_basic._data.columns


def test_remove_columns_design_column_raises_without_force(sample_with_design: Sample):
    with pytest.raises(MethodError) as exc_info:
        sample_with_design.wrangling.remove_columns("stratum")
    assert exc_info.value.code == "DROP_PROTECTED_COLUMNS"


def test_remove_columns_design_column_allowed_with_force(sample_with_design: Sample):
    out = sample_with_design.wrangling.remove_columns("stratum", force=True)
    assert "stratum" not in out._data.columns
    assert out._design.stratum is None


def test_remove_columns_multiple_design_columns_with_force(sample_with_design: Sample):
    out = sample_with_design.wrangling.remove_columns(["stratum", "psu", "weight"], force=True)
    assert "stratum" not in out._data.columns
    assert "psu" not in out._data.columns
    assert "weight" not in out._data.columns
    assert out._design.stratum is None
    assert out._design.psu is None
    assert out._design.wgt is None


def test_drop_is_alias_for_remove_columns(sample_basic: Sample):
    out = sample_basic.wrangling.drop("c")
    assert "c" not in out._data.columns


# ==================== keep_columns / select ====================


def test_keep_columns_single_column(sample_basic: Sample):
    out = sample_basic.wrangling.keep_columns("a")
    assert "a" in out._data.columns
    assert "b" not in out._data.columns
    assert "c" not in out._data.columns


def test_keep_columns_multiple_columns(sample_basic: Sample):
    out = sample_basic.wrangling.keep_columns(["a", "c"])
    assert "a" in out._data.columns
    assert "c" in out._data.columns
    assert "b" not in out._data.columns


def test_keep_columns_returns_new_instance(sample_basic: Sample):
    out = sample_basic.wrangling.keep_columns(["a", "b"])
    assert out is not sample_basic
    # Original unchanged
    assert "c" in sample_basic._data.columns


def test_keep_columns_inplace(sample_basic: Sample):
    out = sample_basic.wrangling.keep_columns(["a", "b"], inplace=True)
    assert out is sample_basic
    assert "c" not in sample_basic._data.columns


def test_keep_columns_preserves_data(sample_basic: Sample):
    out = sample_basic.wrangling.keep_columns(["a", "b"])
    assert out._data["a"].to_list() == [1, 2, 3]
    assert out._data["b"].to_list() == [10, 20, 30]


def test_keep_columns_raises_if_drops_design_column(sample_with_design: Sample):
    with pytest.raises(MethodError) as exc_info:
        sample_with_design.wrangling.keep_columns(["id", "value"])
    assert exc_info.value.code == "KEEP_DROPS_PROTECTED"


def test_keep_columns_with_force_cleans_design(sample_with_design: Sample):
    out = sample_with_design.wrangling.keep_columns(["id", "value"], force=True)
    assert "id" in out._data.columns
    assert "value" in out._data.columns
    assert "stratum" not in out._data.columns
    assert "psu" not in out._data.columns
    assert "weight" not in out._data.columns
    assert out._design.stratum is None
    assert out._design.psu is None
    assert out._design.wgt is None


def test_keep_columns_including_design_columns_no_error(sample_with_design: Sample):
    out = sample_with_design.wrangling.keep_columns(["id", "stratum", "psu", "weight"])
    assert "stratum" in out._data.columns
    assert "psu" in out._data.columns
    assert "weight" in out._data.columns


def test_keep_columns_subset_excluding_non_design(sample_with_design: Sample):
    out = sample_with_design.wrangling.keep_columns(["id", "stratum", "psu", "weight"])
    assert "value" not in out._data.columns
    assert "stratum" in out._data.columns
    assert "psu" in out._data.columns


def test_select_is_alias_for_keep_columns(sample_basic: Sample):
    out = sample_basic.wrangling.select(["a", "b"])
    assert "a" in out._data.columns
    assert "b" in out._data.columns
    assert "c" not in out._data.columns


# ==================== cast ====================


def test_cast_single_column_with_dtype(sample_basic: Sample):
    out = sample_basic.wrangling.cast("a", pl.Float64)
    assert out._data["a"].dtype == pl.Float64
    assert out._data["a"].to_list() == [1.0, 2.0, 3.0]


def test_cast_multiple_columns_same_dtype(sample_basic: Sample):
    out = sample_basic.wrangling.cast(["a", "b"], pl.Float32)
    assert out._data["a"].dtype == pl.Float32
    assert out._data["b"].dtype == pl.Float32


def test_cast_with_mapping(sample_basic: Sample):
    out = sample_basic.wrangling.cast({"a": pl.Float64, "b": pl.Utf8})
    assert out._data["a"].dtype == pl.Float64
    assert out._data["b"].dtype == pl.Utf8
    assert out._data["b"].to_list() == ["10", "20", "30"]


def test_cast_returns_new_instance(sample_basic: Sample):
    out = sample_basic.wrangling.cast("a", pl.Float64)
    assert out is not sample_basic
    # Original unchanged
    assert sample_basic._data["a"].dtype != pl.Float64


def test_cast_inplace(sample_basic: Sample):
    out = sample_basic.wrangling.cast("a", pl.Float64, inplace=True)
    assert out is sample_basic
    assert sample_basic._data["a"].dtype == pl.Float64


def test_cast_requires_dtype_when_not_mapping(sample_basic: Sample):
    with pytest.raises(MethodError) as exc_info:
        sample_basic.wrangling.cast("a")
    assert exc_info.value.code == "CAST_DTYPE_REQUIRED"


def test_cast_strict_mode_raises_on_invalid():
    df = pl.DataFrame({"x": ["a", "b", "c"]})
    s = Sample(df)
    with pytest.raises(Exception):
        s.wrangling.cast("x", pl.Int64, strict=True)


def test_cast_non_strict_returns_null_on_invalid():
    df = pl.DataFrame({"x": ["1", "two", "3"]})
    s = Sample(df)
    out = s.wrangling.cast("x", pl.Int64, strict=False)
    assert out._data["x"].to_list() == [1, None, 3]


def test_cast_int_to_categorical():
    df = pl.DataFrame({"code": [1, 2, 1, 3]})
    s = Sample(df)
    out = s.wrangling.cast("code", pl.Categorical)
    assert out._data["code"].dtype == pl.Categorical


# ==================== fill_null ====================


def test_fill_null_with_value(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.fill_null("x", value=0)
    assert out._data["x"].to_list() == [1, 0, 3, 0, 5]


def test_fill_null_with_string_value(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.fill_null("z", value="MISSING")
    assert out._data["z"].to_list() == ["a", "MISSING", "c", "MISSING", "e"]


def test_fill_null_multiple_columns(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.fill_null(["x", "y"], value=999)
    assert None not in out._data["x"].to_list()
    assert None not in out._data["y"].to_list()


def test_fill_null_strategy_forward():
    df = pl.DataFrame({"v": [1, None, None, 4, None]})
    s = Sample(df)
    out = s.wrangling.fill_null("v", strategy="forward")
    assert out._data["v"].to_list() == [1, 1, 1, 4, 4]


def test_fill_null_strategy_backward():
    df = pl.DataFrame({"v": [None, None, 3, None, 5]})
    s = Sample(df)
    out = s.wrangling.fill_null("v", strategy="backward")
    assert out._data["v"].to_list() == [3, 3, 3, 5, 5]


def test_fill_null_strategy_mean():
    df = pl.DataFrame({"v": [1.0, None, 3.0, None, 5.0]})
    s = Sample(df)
    out = s.wrangling.fill_null("v", strategy="mean")
    assert out._data["v"].to_list() == [1.0, 3.0, 3.0, 3.0, 5.0]


def test_fill_null_strategy_min():
    df = pl.DataFrame({"v": [10, None, 30, None, 50]})
    s = Sample(df)
    out = s.wrangling.fill_null("v", strategy="min")
    assert out._data["v"].to_list() == [10, 10, 30, 10, 50]


def test_fill_null_strategy_max():
    df = pl.DataFrame({"v": [10, None, 30, None, 50]})
    s = Sample(df)
    out = s.wrangling.fill_null("v", strategy="max")
    assert out._data["v"].to_list() == [10, 50, 30, 50, 50]


def test_fill_null_strategy_zero():
    df = pl.DataFrame({"v": [1, None, 3, None, 5]})
    s = Sample(df)
    out = s.wrangling.fill_null("v", strategy="zero")
    assert out._data["v"].to_list() == [1, 0, 3, 0, 5]


def test_fill_null_strategy_one():
    df = pl.DataFrame({"v": [0, None, 2, None, 4]})
    s = Sample(df)
    out = s.wrangling.fill_null("v", strategy="one")
    assert out._data["v"].to_list() == [0, 1, 2, 1, 4]


def test_fill_null_returns_new_instance(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.fill_null("x", value=0)
    assert out is not sample_with_nulls


def test_fill_null_inplace(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.fill_null("x", value=0, inplace=True)
    assert out is sample_with_nulls
    assert out._data["x"].to_list() == [1, 0, 3, 0, 5]


def test_fill_null_no_nulls_unchanged():
    df = pl.DataFrame({"v": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.fill_null("v", value=999)
    assert out._data["v"].to_list() == [1, 2, 3]
