"""SAS Transport (XPT) files under conventional names, and CPORT refusal.

Uses the real svy-io reader: the behaviour under test is its content-based
choice between the sas7bdat, XPORT and CPORT readers.
"""

from __future__ import annotations

import logging

import polars as pl
import pytest
import svy_io

import svy

from svy.errors.io_errors import IoError


CPORT_HEAD = b"**COMPRESSED** " * 4 + b"**COMPRESSED********LIB CONTROL X64_10PR"


@pytest.fixture
def ssp(tmp_path):
    path = tmp_path / "h188b.ssp"
    df = pl.DataFrame({"dupersid": [1.0, 2.0, 3.0], "perwt": [100.5, 200.0, 50.25]})
    svy_io.write_xpt(df, path, version=5)
    return path


def test_read_sas_reads_xpt_named_ssp(ssp, caplog):
    with caplog.at_level(logging.WARNING, logger="svy.io.base"):
        df = svy.read_sas(ssp)

    assert df.columns == ["dupersid", "perwt"]
    assert df["perwt"].to_list() == [100.5, 200.0, 50.25]
    assert "may not match format" not in caplog.text


def test_create_from_sas_builds_a_sample_from_ssp(ssp):
    sample = svy.create_from_sas(ssp)

    assert isinstance(sample, svy.Sample)
    assert sample.data["dupersid"].to_list() == [1.0, 2.0, 3.0]


def test_read_sas_columns_on_xpt(ssp):
    df = svy.read_sas(ssp, columns=["perwt"])

    assert df.columns == ["perwt"]


def test_cport_raises_a_guiding_error(tmp_path):
    path = tmp_path / "h206e.ssp"
    path.write_bytes(CPORT_HEAD + b"\x00" * 200)

    with pytest.raises(IoError) as exc:
        svy.read_sas(path)

    err = exc.value
    assert err.code == "READSTAT_PARSE_FAILED"
    assert "SAS CPORT file" in err.detail
    assert "PROC CIMPORT" in err.hint


def test_catalog_with_xpt_is_refused(ssp):
    with pytest.raises(IoError, match="catalog_path applies value labels to .sas7bdat"):
        svy.read_sas(ssp, catalog_path="formats.sas7bcat")


@pytest.mark.parametrize(
    "alias, target",
    [
        ("read_xpt", "read_sas"),
        ("read_xpt_with_labels", "read_sas_with_labels"),
        ("create_from_xpt", "create_from_sas"),
        ("write_xpt", "write_sas"),
    ],
)
def test_xpt_aliases_are_the_sas_functions(alias, target):
    assert getattr(svy.io, alias) is getattr(svy.io, target)
    assert alias in svy.io.__all__


@pytest.mark.parametrize("name", ["read_xpt", "create_from_xpt", "write_xpt"])
def test_xpt_aliases_are_exported_at_top_level(name):
    assert getattr(svy, name) is getattr(svy.io, name)
    assert name in svy.__all__


def test_write_xpt_then_read_xpt_round_trips(ssp, tmp_path):
    sample = svy.create_from_xpt(ssp)
    out = tmp_path / "out.xpt"

    svy.write_xpt(sample, out)

    assert svy.read_xpt(out).equals(svy.read_sas(ssp))
