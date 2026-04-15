# tests/svy/wrangling/test_row_operations.py
"""Tests for row manipulation operations: sort, distinct, with_row_index."""

import polars as pl
import pytest

from svy.core.sample import Sample


# ---------- Fixtures ----------


@pytest.fixture
def sample_unsorted():
    df = pl.DataFrame(
        {
            "name": ["Charlie", "Alice", "Bob", "Diana"],
            "age": [30, 25, 35, 25],
            "score": [85.0, 90.0, 75.0, 95.0],
        }
    )
    return Sample(df)


@pytest.fixture
def sample_with_nulls():
    df = pl.DataFrame(
        {
            "name": ["A", "B", "C", "D", "E"],
            "value": [3, None, 1, None, 2],
        }
    )
    return Sample(df)


@pytest.fixture
def sample_with_duplicates():
    df = pl.DataFrame(
        {
            "id": [1, 2, 1, 3, 2, 1],
            "category": ["A", "B", "A", "C", "B", "A"],
            "value": [10, 20, 11, 30, 21, 12],
        }
    )
    return Sample(df)


# ==================== sort ====================


def test_sort_single_column_ascending(sample_unsorted: Sample):
    out = sample_unsorted.wrangling.order_by("name")
    assert out._data["name"].to_list() == ["Alice", "Bob", "Charlie", "Diana"]


def test_sort_single_column_descending(sample_unsorted: Sample):
    out = sample_unsorted.wrangling.order_by("age", descending=True)
    ages = out._data["age"].to_list()
    assert ages == [35, 30, 25, 25]


def test_sort_multiple_columns(sample_unsorted: Sample):
    out = sample_unsorted.wrangling.order_by(["age", "name"])
    names = out._data["name"].to_list()
    assert names == ["Alice", "Diana", "Charlie", "Bob"]


def test_sort_multiple_columns_mixed_order(sample_unsorted: Sample):
    out = sample_unsorted.wrangling.order_by(["age", "score"], descending=[True, False])
    names = out._data["name"].to_list()
    assert names[0] == "Bob"
    assert names[1] == "Charlie"
    assert names[2] == "Alice"
    assert names[3] == "Diana"


def test_sort_nulls_last_default(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.order_by("value")
    values = out._data["value"].to_list()
    assert values == [1, 2, 3, None, None]


def test_sort_nulls_first(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.order_by("value", nulls_last=False)
    values = out._data["value"].to_list()
    assert values == [None, None, 1, 2, 3]


def test_sort_returns_new_instance(sample_unsorted: Sample):
    out = sample_unsorted.wrangling.order_by("name")
    assert out is not sample_unsorted


def test_sort_inplace(sample_unsorted: Sample):
    out = sample_unsorted.wrangling.order_by("name", inplace=True)
    assert out is sample_unsorted
    assert out._data["name"].to_list() == ["Alice", "Bob", "Charlie", "Diana"]


def test_sort_preserves_all_data(sample_unsorted: Sample):
    out = sample_unsorted.wrangling.order_by("name")
    assert "name" in out._data.columns
    assert "age" in out._data.columns
    assert "score" in out._data.columns
    assert out._data.height == 4


def test_sort_stable_ordering():
    df = pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B"],
            "order": [1, 2, 3, 1, 2],
            "value": [10, 10, 10, 20, 20],
        }
    )
    s = Sample(df)
    out = s.wrangling.order_by("value")
    orders_for_10 = out._data.filter(pl.col("value") == 10)["order"].to_list()
    assert orders_for_10 == [1, 2, 3]


# ==================== distinct ====================


def test_distinct_all_columns(sample_with_duplicates: Sample):
    out = sample_with_duplicates.wrangling.distinct()
    assert out._data.height == 6


def test_distinct_single_column():
    df = pl.DataFrame(
        {
            "id": [1, 2, 1, 3, 2],
            "value": [10, 20, 30, 40, 50],
        }
    )
    s = Sample(df)
    out = s.wrangling.distinct("id")
    assert out._data.height == 3
    ids = set(out._data["id"].to_list())
    assert ids == {1, 2, 3}


def test_distinct_multiple_columns(sample_with_duplicates: Sample):
    out = sample_with_duplicates.wrangling.distinct(["id", "category"])
    assert out._data.height == 3


def test_distinct_keep_first():
    df = pl.DataFrame({"id": [1, 1, 1], "seq": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.distinct("id", keep="first")
    assert out._data["seq"].to_list() == [1]


def test_distinct_keep_last():
    df = pl.DataFrame({"id": [1, 1, 1], "seq": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.distinct("id", keep="last")
    assert out._data["seq"].to_list() == [3]


def test_distinct_keep_any():
    df = pl.DataFrame({"id": [1, 1, 2, 2], "seq": [1, 2, 3, 4]})
    s = Sample(df)
    out = s.wrangling.distinct("id", keep="any")
    assert out._data.height == 2


def test_distinct_keep_none():
    df = pl.DataFrame({"id": [1, 1, 2, 3, 3], "value": [10, 11, 20, 30, 31]})
    s = Sample(df)
    out = s.wrangling.distinct("id", keep="none")
    assert out._data.height == 1
    assert out._data["id"].to_list() == [2]


def test_distinct_maintain_order_true():
    df = pl.DataFrame({"id": [3, 1, 2, 1, 3], "order": [1, 2, 3, 4, 5]})
    s = Sample(df)
    out = s.wrangling.distinct("id", maintain_order=True)
    ids = out._data["id"].to_list()
    assert ids == [3, 1, 2]


def test_distinct_maintain_order_false():
    df = pl.DataFrame({"id": [3, 1, 2, 1, 3], "order": [1, 2, 3, 4, 5]})
    s = Sample(df)
    out = s.wrangling.distinct("id", maintain_order=False)
    assert out._data.height == 3
    assert set(out._data["id"].to_list()) == {1, 2, 3}


def test_distinct_returns_new_instance(sample_with_duplicates: Sample):
    out = sample_with_duplicates.wrangling.distinct("id")
    assert out is not sample_with_duplicates


def test_distinct_inplace(sample_with_duplicates: Sample):
    out = sample_with_duplicates.wrangling.distinct("id", inplace=True)
    assert out is sample_with_duplicates
    assert out._data.height == 3


def test_distinct_with_nulls():
    df = pl.DataFrame({"id": [1, None, 2, None, 1], "value": [10, 20, 30, 40, 50]})
    s = Sample(df)
    out = s.wrangling.distinct("id")
    assert out._data.height == 3


# ==================== with_row_index ====================


def test_with_row_index_default_name():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.with_row_index()
    assert "row_index" in out._data.columns
    assert out._data["row_index"].to_list() == [0, 1, 2]


def test_with_row_index_custom_name():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.with_row_index(name="my_id")
    assert "my_id" in out._data.columns
    assert out._data["my_id"].to_list() == [0, 1, 2]


def test_with_row_index_custom_offset():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.with_row_index(offset=1)
    assert out._data["row_index"].to_list() == [1, 2, 3]


def test_with_row_index_custom_name_and_offset():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.with_row_index(name="record_id", offset=100)
    assert "record_id" in out._data.columns
    assert out._data["record_id"].to_list() == [100, 101, 102]


def test_with_row_index_returns_new_instance():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.with_row_index()
    assert out is not s


def test_with_row_index_inplace():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.with_row_index(inplace=True)
    assert out is s
    assert "row_index" in s._data.columns


def test_with_row_index_preserves_data():
    df = pl.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"]})
    s = Sample(df)
    out = s.wrangling.with_row_index()
    assert out._data["a"].to_list() == [10, 20, 30]
    assert out._data["b"].to_list() == ["x", "y", "z"]


def test_with_row_index_after_filter():
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    s = Sample(df)
    from svy.core.expr import col

    filtered = s.wrangling.filter_records(col("a") > 2)
    out = filtered.wrangling.with_row_index(name="new_idx")
    assert out._data["new_idx"].to_list() == [0, 1, 2]


def test_with_row_index_large_offset():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.with_row_index(offset=1000000)
    assert out._data["row_index"].to_list() == [1000000, 1000001, 1000002]
