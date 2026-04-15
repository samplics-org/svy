from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from polars.testing import assert_frame_equal

from svy.core.types import Category
from svy.utils.checks import (
    as_1d,
    as_float64_1d,
    check_same_length,
    check_weights_finite_positive,
    drop_missing,
    to_stringnumber,
    validate_xyw,
)


# -----------------------
# as_1d / as_float64_1d
# -----------------------


def test_as_1d_ok():
    a = np.array([1, 2, 3])
    out = as_1d(a=a, name="a")
    assert out.ndim == 1
    assert out.shape == (3,)


def test_as_1d_raises_on_2d():
    a = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="a must be 1-D"):
        as_1d(a=a, name="a")


def test_as_float64_1d_casts_and_is_1d():
    a = np.array([1, 2, 3], dtype=np.int32)
    out = as_float64_1d(a=a, name="a")
    assert out.ndim == 1
    assert out.dtype == float
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0]))


# -----------------------
# check_same_length / weights
# -----------------------


def test_check_same_length_ok():
    a = np.arange(5)
    b = np.ones(5)
    # Should not raise
    check_same_length((a, "a"), (b, "b"))


def test_check_same_length_mismatch_raises():
    a = np.arange(5)
    b = np.ones(4)
    with pytest.raises(ValueError):
        check_same_length((a, "a"), (b, "b"))


def test_check_weights_finite_positive_ok():
    w = np.array([0.5, 0.5, 1.0])
    check_weights_finite_positive(w=w)  # no raise


@pytest.mark.parametrize(
    "w",
    [
        np.array([0.0, 0.0]),  # sum == 0
        np.array([1.0, math.nan]),  # NaN present
        np.array([1.0, math.inf, -math.inf]),  # non-finite
    ],
)
def test_check_weights_finite_positive_raises(w):
    with pytest.raises(ValueError, match="Weights must be finite and sum to a positive value"):
        check_weights_finite_positive(w=w)


# -----------------------
# validate_xyw
# -----------------------


def test_validate_xyw_basic_no_x():
    y = np.array([1, 2, 3], dtype=int)
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    yy, ww, xx = validate_xyw(y=y, w=w, x=None, require_x=False)
    assert yy.dtype == float and ww.dtype == float
    assert xx is None
    np.testing.assert_allclose(yy, y.astype(float))
    np.testing.assert_allclose(ww, w)


def test_validate_xyw_requires_x_raises():
    y = np.array([1, 2, 3])
    w = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="Parameter x must be provided"):
        validate_xyw(y=y, w=w, x=None, require_x=True)


def test_validate_xyw_with_x_ok():
    y = np.array([1, 2, 3])
    w = np.array([1.0, 2.0, 3.0])
    x = np.array([4.0, 5.0, 6.0])
    yy, ww, xx = validate_xyw(y=y, w=w, x=x, require_x=True)
    assert xx is not None and xx.dtype == float
    np.testing.assert_allclose(xx, x)


def test_validate_xyw_x_length_mismatch_raises():
    y = np.array([1, 2, 3])
    w = np.array([1.0, 2.0, 3.0])
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        validate_xyw(y=y, w=w, x=x, require_x=True)


def test_validate_xyw_weights_invalid_raises():
    y = np.array([1, 2, 3])
    w = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Weights must be finite"):
        validate_xyw(y=y, w=w, x=None, require_x=False)


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
