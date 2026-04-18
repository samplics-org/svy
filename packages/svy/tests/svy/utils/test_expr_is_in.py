# tests/core/test_expr_is_in.py
import polars as pl
import pytest
import warnings
import svy


def test_is_in_with_list_no_warning():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = df.filter(svy.col("x").is_in([1, 3])._e)
    assert result["x"].to_list() == [1, 3]


def test_is_in_with_series_no_warning():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    s = pl.Series([2, 4])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = df.filter(svy.col("x").is_in(s)._e)
    assert result["x"].to_list() == [2, 4]


def test_is_in_with_svy_expr_no_warning():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [2, 3, 5]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = df.filter(svy.col("x").is_in(svy.col("y"))._e)
    # Each row: is x[i] in the set {2, 3, 5}? 1→No, 2→Yes, 3→Yes
    assert result["x"].to_list() == [2, 3]


def test_is_not_in_with_series_no_warning():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    s = pl.Series([2, 4])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = df.filter(svy.col("x").is_not_in(s)._e)
    assert result["x"].to_list() == [1, 3]
