# tests/svy/wrangling/test_distinct_bookkeeping.py
"""distinct() without cols compares the user's columns only.

svy's row index differs on every row, so including it meant no row was ever a
duplicate.
"""

from __future__ import annotations

import polars as pl
import pytest

from svy import Design, Sample
from svy.core.constants import _INTERNAL_CONCAT_SUFFIX, SVY_ROW_INDEX


def _dups() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 1, 2, 1, 3, 2],
            "b": ["x", "x", "y", "x", "z", "y"],
        }
    )


def _user(s: Sample) -> list[tuple]:
    return s._data.select("a", "b").rows()


def test_removes_duplicate_rows():
    out = Sample(_dups()).wrangling.distinct()
    assert _user(out) == [(1, "x"), (2, "y"), (3, "z")]


@pytest.mark.parametrize(
    "keep,expected",
    [
        ("first", [0, 2, 4]),
        ("last", [3, 4, 5]),
        ("none", [4]),
    ],
)
def test_keep_selects_rows(keep, expected):
    out = Sample(_dups()).wrangling.distinct(keep=keep)
    assert out._data[SVY_ROW_INDEX].to_list() == expected


def test_keep_any_removes_duplicates():
    out = Sample(_dups()).wrangling.distinct(keep="any")
    assert sorted(_user(out)) == [(1, "x"), (2, "y"), (3, "z")]


def test_maintain_order_false_still_dedups():
    out = Sample(_dups()).wrangling.distinct(maintain_order=False)
    assert sorted(_user(out)) == [(1, "x"), (2, "y"), (3, "z")]


def test_no_duplicates_is_unchanged():
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = Sample(df).wrangling.distinct()
    assert out._data.equals(Sample(df)._data)


def test_all_rows_identical():
    df = pl.DataFrame({"a": [7] * 4, "b": ["q"] * 4})
    out = Sample(df).wrangling.distinct()
    assert out._data.height == 1
    assert out._data[SVY_ROW_INDEX].to_list() == [0]


def test_empty_sample():
    df = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Int64, "b": pl.Utf8})
    out = Sample(df).wrangling.distinct()
    assert out._data.height == 0


def test_nulls_are_equal():
    df = pl.DataFrame({"a": [None, None, 1], "b": ["x", "x", None]})
    out = Sample(df).wrangling.distinct()
    assert out._data.height == 2


def test_row_index_kept_from_surviving_rows():
    out = Sample(_dups()).wrangling.distinct()
    assert SVY_ROW_INDEX in out._data.columns
    assert out._data[SVY_ROW_INDEX].to_list() == [0, 2, 4]


def test_concat_design_columns_do_not_block():
    df = _dups().with_columns(pl.lit("s").alias("s1"), pl.col("a").alias("p1"))
    s = Sample(df, Design(stratum=("s1", "b"), psu=("p1", "b")))
    assert any(_INTERNAL_CONCAT_SUFFIX in c for c in s._data.columns)
    out = s.wrangling.distinct()
    assert _user(out) == [(1, "x"), (2, "y"), (3, "z")]


def test_explicit_cols_unchanged():
    out = Sample(_dups()).wrangling.distinct("a", keep="last")
    assert out._data[SVY_ROW_INDEX].to_list() == [3, 4, 5]
    out = Sample(_dups()).wrangling.distinct(["a", "b"])
    assert _user(out) == [(1, "x"), (2, "y"), (3, "z")]


def test_inplace():
    s = Sample(_dups())
    out = s.wrangling.distinct(inplace=True)
    assert out is s
    assert s._data.height == 3


def test_not_inplace_leaves_original():
    s = Sample(_dups())
    s.wrangling.distinct()
    assert s._data.height == 6


def test_estimation_after_distinct():
    df = _dups().with_columns(pl.lit(1.0).alias("w"))
    out = Sample(df, Design(wgt="w")).wrangling.distinct()
    res = out.estimation.total("a").to_polars()
    assert res["est"][0] == pytest.approx(6.0)
