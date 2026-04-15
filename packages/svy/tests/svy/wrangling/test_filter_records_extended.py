# tests/svy/wrangling/test_filter_records_extended.py
"""Extended tests for filter_records functionality."""

import polars as pl
import pytest

from svy.core.design import Design
from svy.core.expr import col
from svy.core.sample import Sample
from svy.errors import MethodError


# ---------- Fixtures ----------


@pytest.fixture
def sample_basic():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [25, 30, 35, 40, 45],
            "region": ["N", "S", "E", "W", "N"],
            "active": [True, False, True, False, True],
        }
    )
    return Sample(df)


@pytest.fixture
def sample_with_nulls():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10, None, 30, None, 50],
            "category": ["A", "B", None, "A", None],
        }
    )
    return Sample(df)


# ==================== Expression-based filtering ====================


def test_filter_greater_than(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("age") > 35)
    assert out._data["age"].to_list() == [40, 45]


def test_filter_less_than(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("age") < 35)
    assert out._data["age"].to_list() == [25, 30]


def test_filter_greater_than_or_equal(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("age") >= 35)
    assert out._data["age"].to_list() == [35, 40, 45]


def test_filter_less_than_or_equal(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("age") <= 35)
    assert out._data["age"].to_list() == [25, 30, 35]


def test_filter_equality(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("age") == 35)
    assert out._data["age"].to_list() == [35]


def test_filter_inequality(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("age") != 35)
    assert out._data["age"].to_list() == [25, 30, 40, 45]


def test_filter_boolean_column(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("active"))
    assert out._data.height == 3
    assert all(out._data["active"].to_list())


def test_filter_negated_boolean_column(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(~col("active"))
    assert out._data.height == 2
    assert not any(out._data["active"].to_list())


# ==================== Combined expressions (AND) ====================


def test_filter_combined_and_explicit(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records((col("age") > 30) & (col("region") == "N"))
    assert out._data.height == 1
    assert out._data["id"].to_list() == [5]


def test_filter_list_of_expressions_and(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(
        [
            col("age") >= 30,
            col("active") == True,
        ]
    )
    assert out._data.height == 2
    assert set(out._data["id"].to_list()) == {3, 5}


# ==================== Dict-based filtering ====================


def test_filter_dict_single_value(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records({"region": "N"})
    assert out._data["region"].to_list() == ["N", "N"]


def test_filter_dict_list_membership(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records({"region": ["N", "S"]})
    assert set(out._data["region"].to_list()) == {"N", "S"}


def test_filter_dict_set_membership(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records({"region": {"N", "E"}})
    regions = set(out._data["region"].to_list())
    assert regions <= {"N", "E"}


def test_filter_dict_tuple_membership(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records({"region": ("W", "S")})
    regions = set(out._data["region"].to_list())
    assert regions <= {"W", "S"}


def test_filter_dict_multiple_conditions(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records({"region": "N", "active": True})
    assert out._data.height == 2
    assert all(r == "N" for r in out._data["region"].to_list())
    assert all(out._data["active"].to_list())


def test_filter_dict_boolean_value(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records({"active": False})
    assert out._data.height == 2
    assert not any(out._data["active"].to_list())


# ==================== Negate parameter ====================


def test_filter_negate_expr(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("region") == "N", negate=True)
    assert "N" not in out._data["region"].to_list()


def test_filter_negate_dict(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records({"active": True}, negate=True)
    assert not any(out._data["active"].to_list())


def test_filter_negate_list_membership(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records({"region": ["N", "S"]}, negate=True)
    assert set(out._data["region"].to_list()) == {"E", "W"}


# ==================== Null handling ====================


def test_filter_keeps_nulls_with_gt(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.filter_records(col("value") > 20)
    assert out._data["value"].to_list() == [30, 50]


def test_filter_is_null(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.filter_records(col("value").is_null())
    assert out._data.height == 2
    assert out._data["value"].null_count() == 2


def test_filter_is_not_null(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.filter_records(col("value").is_not_null())
    assert out._data.height == 3
    assert out._data["value"].null_count() == 0


# ==================== Empty and edge cases ====================


def test_filter_none_is_noop(sample_basic: Sample):
    original_height = sample_basic._data.height
    out = sample_basic.wrangling.filter_records(None)
    assert out._data.height == original_height


def test_filter_empty_dict_is_noop(sample_basic: Sample):
    original_height = sample_basic._data.height
    out = sample_basic.wrangling.filter_records({})
    assert out._data.height == original_height


def test_filter_empty_list_is_noop(sample_basic: Sample):
    original_height = sample_basic._data.height
    out = sample_basic.wrangling.filter_records([])
    assert out._data.height == original_height


def test_filter_to_empty_result(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("age") > 100)
    assert out._data.height == 0


def test_filter_all_rows_match(sample_basic: Sample):
    original_height = sample_basic._data.height
    out = sample_basic.wrangling.filter_records(col("age") > 0)
    assert out._data.height == original_height


# ==================== Error handling ====================


def test_filter_missing_column_raises(sample_basic: Sample):
    with pytest.raises(MethodError):
        sample_basic.wrangling.filter_records({"nonexistent": 1})


def test_filter_missing_column_in_expr_raises(sample_basic: Sample):
    with pytest.raises(Exception):
        sample_basic.wrangling.filter_records(col("nonexistent") > 0)


# ==================== Design and singletons ====================


def test_filter_with_design_preserves_design():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "value": [10, 20, 30, 40],
            "stratum": ["A", "A", "B", "B"],
            "psu": [1, 2, 1, 2],
        }
    )
    design = Design(stratum="stratum", psu="psu")
    s = Sample(df, design)
    out = s.wrangling.filter_records(col("value") > 15)
    assert out._design.stratum == "stratum"
    assert out._design.psu == "psu"


def test_filter_check_singletons_flag():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "stratum": ["A", "A", "B", "B"],
            "psu": [1, 2, 1, 2],
        }
    )
    design = Design(stratum="stratum", psu="psu")
    s = Sample(df, design)
    out = s.wrangling.filter_records({"psu": [1]}, check_singletons=True)
    assert hasattr(out, "_singletons")


# ==================== String comparisons ====================


def test_filter_string_equality(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("region") == "N")
    assert all(r == "N" for r in out._data["region"].to_list())


def test_filter_string_in_list():
    df = pl.DataFrame({"name": ["Alice", "Bob", "Charlie", "David"]})
    s = Sample(df)
    out = s.wrangling.filter_records(col("name").is_in(["Alice", "Charlie"]))
    assert set(out._data["name"].to_list()) == {"Alice", "Charlie"}


# ==================== Numeric edge cases ====================


def test_filter_float_comparison():
    df = pl.DataFrame({"value": [1.1, 2.2, 3.3, 4.4]})
    s = Sample(df)
    out = s.wrangling.filter_records(col("value") > 2.5)
    assert out._data["value"].to_list() == [3.3, 4.4]


def test_filter_negative_values():
    df = pl.DataFrame({"value": [-10, -5, 0, 5, 10]})
    s = Sample(df)
    out = s.wrangling.filter_records(col("value") < 0)
    assert out._data["value"].to_list() == [-10, -5]


# ==================== Copy-on-write ====================


def test_filter_does_not_modify_original(sample_basic: Sample):
    original_height = sample_basic._data.height
    out = sample_basic.wrangling.filter_records(col("age") > 35)
    assert out._data.height == 2
    assert sample_basic._data.height == original_height


def test_filter_inplace(sample_basic: Sample):
    out = sample_basic.wrangling.filter_records(col("age") > 35, inplace=True)
    assert out is sample_basic
    assert sample_basic._data.height == 2
