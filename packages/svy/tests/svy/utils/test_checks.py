from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars.testing import assert_frame_equal

from svy.core.types import Category
from svy.utils.checks import (
    any_missing,
    drop_missing,
    to_stringnumber,
)


# -----------------------
# to_Categoryber
# -----------------------


@pytest.mark.parametrize(
    "inp, expected",
    [
        (True, True),  # bool preserved
        (False, False),
        (3, 3),  # int preserved
        (3.0, 3),  # float integral -> int
        (1.5, 1.5),  # float non-integral
        (np.int64(7), 7),  # numpy integer -> int
        (np.float64(2.0), 2),  # numpy float integral -> int
        (np.float64(2.25), 2.25),  # numpy float non-integral -> float
        ("42", 42),  # str int -> int
        ("3.0", 3),  # str float integral -> int
        ("3.14", 3.14),  # str float -> float
        (b"8", 8),  # bytes int -> int
        (bytearray(b"9.0"), 9),  # bytearray float integral -> int
        ("A", "A"),  # non-numeric str -> str
        ("nan", "nan"),  # str "nan" stays string (not finite)
    ],
)
def test_to_stringnumber_conversions(inp: object, expected: Category):
    out = to_stringnumber(token=inp)
    # for floats, exact match is fine here; converters only produce finite floats
    assert out == expected
    assert isinstance(out, (bool, int, float, str))


def test_to_Categoryber_other_object_becomes_str():
    class X:
        def __str__(self) -> str:
            return "X-obj"

    out = to_stringnumber(token=X())
    assert isinstance(out, str)
    assert out == "X-obj"


# -----------------------
# remove_missing_values
# -----------------------


def test_remove_missing_values_nulls_and_nans():
    df = pl.DataFrame(
        {
            "a": [1, 2, None, 4],  # Int64 (nullable)
            "b": [0.0, float("nan"), 3.0, 4.0],  # Float64
            "c": ["x", "y", "z", "w"],  # Utf8
        }
    )
    # Drop rows with nulls in a or nans in b
    cleaned = drop_missing(df=df, cols=["a", "b"])
    # Expect rows 0 and 3 only
    expected = pl.DataFrame({"a": [1, 4], "b": [0.0, 4.0], "c": ["x", "w"]})
    assert_frame_equal(cleaned, expected)


def test_remove_missing_values_infinities_option():
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [0.0, float("inf"), float("-inf"), 5.0],
            "c": ["x", "y", "z", "w"],
        }
    )
    # Keep ±∞ when treat_infinite_as_missing=False
    keep_inf = drop_missing(df=df, cols=["a", "b"])
    assert_frame_equal(keep_inf, df)

    # Drop ±∞ when True
    drop_inf = drop_missing(df=df, cols=["a", "b"], treat_infinite_as_missing=True)
    expected = pl.DataFrame({"a": [1, 4], "b": [0.0, 5.0], "c": ["x", "w"]})
    assert_frame_equal(drop_inf, expected)


def test_remove_missing_values_missing_column_raises():
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    with pytest.raises(KeyError, match="Columns not found"):
        _ = drop_missing(df=df, cols=["a", "c"])


def test_remove_missing_values_empty_cols_returns_input():
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    out = drop_missing(df=df, cols=[])
    assert_frame_equal(out, df)


def test_remove_missing_values_streaming_same_result():
    df = pl.DataFrame(
        {
            "a": [1, None, 3, 4],
            "b": [0.0, float("nan"), 2.0, 3.0],
        }
    )
    cols = ["a", "b"]
    eager_like = drop_missing(df=df, cols=cols, streaming=False)
    streamed = drop_missing(df=df, cols=cols, streaming=True)
    assert_frame_equal(eager_like, streamed)


# -----------------------
# any_missing
# -----------------------


@pytest.mark.parametrize(
    "values, infinite, expected",
    [
        ([1.0, 2.0], True, False),
        ([1.0, None], False, True),
        ([1.0, float("nan")], False, True),
        ([1.0, float("inf")], False, False),
        ([1.0, float("-inf")], True, True),
    ],
)
def test_any_missing_floats(values, infinite, expected):
    df = pl.DataFrame({"a": values}, schema={"a": pl.Float64})
    assert any_missing(df, ["a"], infinite=infinite) is expected


def test_any_missing_non_float_reads_nulls_only():
    df = pl.DataFrame({"i": [1, 2], "s": ["x", None]})
    assert not any_missing(df, ["i"], infinite=True)
    assert any_missing(df, ["i", "s"], infinite=True)


def test_drop_missing_returns_the_frame_when_nothing_is_missing():
    df = pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert drop_missing(df=df, cols=["a", "b"], treat_infinite_as_missing=True) is df
