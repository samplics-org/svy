"""Writers store infinity and NaN as missing, as haven does: none of the formats holds them."""

import math

import polars as pl
import pytest

from svy_io import read_dta, read_sav, read_xpt, write_dta, write_sav, write_xpt


FORMATS = [
    ("sav", write_sav, read_sav),
    ("dta", write_dta, read_dta),
    ("xpt", write_xpt, read_xpt),
]


@pytest.mark.parametrize(("ext", "write", "read"), FORMATS, ids=[f[0] for f in FORMATS])
def test_non_finite_values_are_written_as_missing(tmp_path, ext, write, read):
    df = pl.DataFrame({"x": [1.5, math.inf, -math.inf, math.nan, None, 2.0]})
    path = tmp_path / f"nonfinite.{ext}"
    write(df, path)
    back, _ = read(path)
    assert back["x"].to_list() == [1.5, None, None, None, None, 2.0]
