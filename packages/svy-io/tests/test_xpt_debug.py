# tests/test_xpt_debug.py


import polars as pl

from svy_io import read_xpt, write_xpt


def test_write_xpt_basic(tmp_path):
    df = pl.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "score": [95.5, 87.3, 91.2]}
    )

    path = tmp_path / "test.xpt"
    write_xpt(df, str(path))
    assert path.exists()
    assert path.stat().st_size > 0

    # Optional: round-trip check
    df2, _meta = read_xpt(str(path))
    assert df2.shape == df.shape
