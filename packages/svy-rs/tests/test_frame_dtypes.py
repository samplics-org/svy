"""
The kernel imports a frame whatever integer dtypes its columns have.

Without polars' dtype-i8/i16/u8/u16 features, pyo3-polars rejected the whole
frame (`cannot create series from Int8`) when any column had one of those
dtypes, even a column the kernel never reads.
"""

import polars as pl
import pytest
import svy_rs as ps
from polars.testing import assert_frame_equal

SMALL_INTS = [pl.Int8, pl.Int16, pl.UInt8, pl.UInt16]


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "value": [1.0, 4.0, 2.0, 8.0, 5.0, 3.0],
            "weight": [1.0, 2.0, 1.5, 1.0, 3.0, 2.0],
            "strata": ["a", "a", "a", "b", "b", "b"],
            "psu": ["1", "2", "3", "4", "5", "6"],
            "passenger": [0, 1, 0, 1, 1, 0],
        }
    )


@pytest.mark.parametrize("dtype", SMALL_INTS, ids=str)
@pytest.mark.parametrize("fn", [ps.taylor_mean, ps.taylor_total], ids=lambda f: f.__name__)
def test_small_int_column_in_frame(df, fn, dtype):
    kwargs = {
        "value_col": "value",
        "weight_col": "weight",
        "strata_col": "strata",
        "psu_col": "psu",
    }
    got, got_cov = fn(df.with_columns(pl.col("passenger").cast(dtype)), **kwargs)
    expected, expected_cov = fn(df, **kwargs)
    assert_frame_equal(got, expected)
    assert got_cov == expected_cov
