# tests/test_sas_arrow_extras.py
from pathlib import Path

import pyarrow as pa

from svy_io import read_sas_arrow


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/sas"


def tpath(rel: str) -> str:
    """Return absolute path inside tests/sas/."""
    return str((DATA / rel).resolve())


def test_arrow_zero_rows_path():
    tbl, meta = read_sas_arrow(tpath("hadley.sas7bdat"), n_max=0)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 0
    assert meta["n_rows"] == 0
