# tests/svy/wrangling/test_recode_dtype.py
"""recode must handle dtype mismatches between the column and the lookup values.

polars 2.0 dropped lossy coercion in `is_in`, so passing int literals
(e.g. [7, 9]) against a Float64 column raises InvalidOperationError
unless the lookup values are cast to the column's dtype.
"""

import polars as pl

from svy.core.sample import Sample


def test_recode_float64_column_with_int_literals():
    """A Float64 column (e.g. NHANES data with None) recoded with int values."""
    df = pl.DataFrame({"x": [1.0, 7.0, 9.0, 2.0, None]})
    s = Sample(df)

    out = s.wrangling.recode("x", {None: [7, 9]}, replace=True)

    result = out._data["x"].to_list()
    assert result == [1.0, None, None, 2.0, None]


def test_recode_int64_column_with_int_literals():
    """An Int64 column recoded with int values (no dtype mismatch)."""
    df = pl.DataFrame({"x": [1, 7, 9, 2, 7]})
    s = Sample(df)

    out = s.wrangling.recode("x", {None: [7, 9]}, replace=True)

    result = out._data["x"].to_list()
    assert result == [1, None, None, 2, None]


def test_recode_float64_column_with_float_literals():
    """A Float64 column recoded with float values (already matching)."""
    df = pl.DataFrame({"x": [1.0, 7.0, 9.0, 2.0, None]})
    s = Sample(df)

    out = s.wrangling.recode("x", {None: [7.0, 9.0]}, replace=True)

    result = out._data["x"].to_list()
    assert result == [1.0, None, None, 2.0, None]


def test_recode_string_column_with_string_literals():
    """A String column recoded with string values (no regression)."""
    df = pl.DataFrame({"cat": ["a", "b", "c", "a", "b"]})
    s = Sample(df)

    out = s.wrangling.recode("cat", {"X": ["a", "b"]}, replace=True)

    result = out._data["cat"].to_list()
    assert result == ["X", "X", "c", "X", "X"]


def test_recode_float64_column_multiple_new_values():
    """Multiple new-value groups on a Float64 column with int literals.

    The output column becomes String (polars promotes to a common type when
    float values and string labels coexist).
    """
    df = pl.DataFrame({"x": [1.0, 7.0, 9.0, 2.0, 8.0]})
    s = Sample(df)

    out = s.wrangling.recode("x", {"refused": [7], "unknown": [9]}, replace=True)

    result = out._data["x"].to_list()
    assert result == ["1.0", "refused", "unknown", "2.0", "8.0"]


def test_recode_into_new_column_preserves_original():
    """recode into a new column leaves the original Float64 column intact."""
    df = pl.DataFrame({"x": [1.0, 7.0, 9.0, 2.0]})
    s = Sample(df)

    out = s.wrangling.recode("x", {None: [7, 9]}, into="x_recoded")

    assert out._data["x"].to_list() == [1.0, 7.0, 9.0, 2.0]
    assert out._data["x_recoded"].to_list() == [1.0, None, None, 2.0]


def test_recode_float64_column_only_none_mapping():
    """Recode mapping None to None on a Float64 column with actual nulls."""
    df = pl.DataFrame({"x": [1.0, None, 9.0, 2.0, None]})
    s = Sample(df)

    out = s.wrangling.recode("x", {-1: [None]}, replace=True)

    result = out._data["x"].to_list()
    assert result == [1.0, -1, 9.0, 2.0, -1]
