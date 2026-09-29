# tests/test_sas_arrow_extras.py
import shutil

from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

from svy_io import read_sas_arrow, read_xpt


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


# ---- XPT --------------------------------------------------------------------


def _ssp(tmp_path):
    path = tmp_path / "hadley.ssp"
    shutil.copy(DATA / "hadley.xpt", path)
    return path


def test_arrow_reads_xpt_by_content(tmp_path):
    tbl, meta = read_sas_arrow(_ssp(tmp_path))
    expected, meta_expected = read_xpt(DATA / "hadley.xpt", coerce_temporals=False)

    assert pl.from_arrow(tbl).equals(expected)
    assert meta["vars"] == meta_expected["vars"]


def test_arrow_xpt_keeps_field_metadata_and_catalog_labels(tmp_path):
    tbl, meta = read_sas_arrow(_ssp(tmp_path), catalog_path=tpath("formats.sas7bcat"))

    workshop = tbl.schema.field("workshop").metadata
    assert workshop[b"format"] == b"WORKSHOP5"
    assert workshop[b"label_set"] == b"WORKSHOP"
    assert {vl["set_name"] for vl in meta["value_labels"]} == {"WORKSHOP", "$GENDER"}


def test_arrow_xpt_forwards_row_and_column_selection(tmp_path):
    tbl, _ = read_sas_arrow(_ssp(tmp_path), rows_skip=1, n_max=2, cols_skip=["gender"])

    assert "gender" not in tbl.column_names
    assert tbl.column("id").to_pylist() == [2.0, 3.0]


def test_arrow_xpt_zero_rows(tmp_path):
    tbl, meta = read_sas_arrow(_ssp(tmp_path), n_max=0)

    assert tbl.num_rows == 0
    assert tbl.num_columns == 7
    assert meta["n_rows"] == 0


def test_arrow_refuses_cport(tmp_path):
    path = tmp_path / "h206e.ssp"
    path.write_bytes(b"**COMPRESSED** " * 4 + b"\x00" * 200)

    with pytest.raises(RuntimeError, match="SAS CPORT file"):
        read_sas_arrow(path)
