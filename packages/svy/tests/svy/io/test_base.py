# tests/svy/core/test_io.py
from __future__ import annotations

import polars as pl
import pytest

from svy.core.sample import Sample
from svy.io import (
    create_from_csv,
    create_from_parquet,
    create_from_sas,
    create_from_sav,
    create_from_spss,
    create_from_stata,
    read_csv,
    read_parquet,
    read_sas,
    read_sav,
    read_spss,
    read_stata,
    scan_csv,
    write_sav,
    write_spss,
    write_stata,
)
from svy.metadata import VariableMeta


def test_read_spss_attaches_labels(dummy_svyio, tmp_path):
    # Create a real placeholder file so _preflight_read passes
    p = tmp_path / "fake.sav"
    p.touch()

    # Use the creator which returns a Sample and attaches labels
    sample = create_from_spss(p)
    assert isinstance(sample, Sample)
    assert isinstance(sample.data, pl.DataFrame)

    # data content (Sample may add an internal row-index column; don't assert exact width)
    df = sample.data
    assert df.height == 3
    assert {"id", "sex", "wgt"}.issubset(set(df.columns))
    assert df.select("sex").to_series().to_list() == [1, 2, 9]

    # labels are attached (your Label(label, categories))
    meta = sample.meta
    assert isinstance(meta.get("sex"), VariableMeta)
    assert meta.get("sex").label == "Sex of respondent"
    cats = meta.get("sex").value_labels or {}
    assert cats[1] == "Male" and cats[2] == "Female"


@pytest.mark.parametrize(
    "reader,ext",
    [
        (read_stata, ".dta"),
        (read_sas, ".sas7bdat"),
    ],
)
def test_read_other_formats(dummy_svyio, reader, ext, tmp_path):
    p = tmp_path / f"fake{ext}"
    p.touch()

    # Request specific columns and encoding
    df = reader(p, columns=["id", "sex"], encoding="latin1")
    assert isinstance(df, pl.DataFrame)

    # 1. Check Result: The wrapper should have filtered the columns for us
    assert df.columns == ["id", "sex"]

    # 2. Check Implementation Details (Backend calls)
    calls = [c for c in dummy_svyio.calls if c.fn.startswith("read_")]
    last_call = calls[-1]

    if "dta" in ext:
        # Stata Engine: explicit columns/encoding are removed to prevent crashes
        assert "columns" not in last_call.kwargs
        assert "cols_skip" not in last_call.kwargs
        # We explicitly verify encoding is NOT passed
        assert "encoding" not in last_call.kwargs
    elif "sas" in ext:
        # SAS Engine: encoding is supported and passed through
        assert last_call.kwargs.get("encoding") == "latin1"


def test_create_from_stata_and_sas_return_sample(dummy_svyio, tmp_path):
    p_dta = tmp_path / "fake.dta"
    p_sas = tmp_path / "fake.sas7bdat"
    p_dta.touch()
    p_sas.touch()

    s1 = create_from_stata(p_dta)
    s2 = create_from_sas(p_sas)

    assert isinstance(s1, Sample)
    assert isinstance(s2, Sample)


def test_write_spss_renders_metadata_and_calls_backend(tmp_path, dummy_svyio):
    # Seed labels via create_from_spss
    p = tmp_path / "fake.sav"
    p.touch()
    s = create_from_spss(p)

    out = tmp_path / "out.sav"
    write_spss(s, out)

    # last call at svy_io layer should be write_spss with expected metadata
    last = [c for c in dummy_svyio.calls if c.fn == "write_spss"][-1]
    meta = last.kwargs["metadata"]
    assert "variables" in meta and "sex" in meta["variables"]
    v = meta["variables"]["sex"]
    assert v["label"] == "Sex of respondent"
    assert v["values"][1] == "Male" and v["values"][2] == "Female"


def test_write_stata_splits_metadata_correctly(tmp_path, dummy_svyio):
    """
    Stata requires 'var_labels' and 'value_labels' as separate args,
    not a single 'metadata' dict.
    """
    p = tmp_path / "fake.sav"
    p.touch()
    s = create_from_spss(p)  # Creates sample with labels

    out = tmp_path / "out.dta"
    write_stata(s, out, version=118)

    # Find the write_stata call
    call = [c for c in dummy_svyio.calls if c.fn == "write_stata"][-1]

    # Assert args are split
    assert "metadata" not in call.kwargs

    # Check Variable Labels
    assert "var_labels" in call.kwargs
    assert call.kwargs["var_labels"]["sex"] == "Sex of respondent"

    # Check Value Labels
    assert "value_labels" in call.kwargs
    assert call.kwargs["value_labels"]["sex"][1] == "Male"


def test_read_standard_formats(tmp_path):
    """
    Test CSV/Parquet support (relies on Polars native IO, not dummy_svyio).
    """
    # Create a real CSV
    p_csv = tmp_path / "data.csv"
    p_csv.write_text("id,val\n1,10\n2,20")

    # Test Eager Read
    df = read_csv(p_csv)
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (2, 2)

    # Test Lazy Scan
    lz = scan_csv(p_csv)
    assert isinstance(lz, pl.LazyFrame)
    assert lz.collect().shape == (2, 2)

    # Test Factory
    s = create_from_csv(p_csv)
    assert isinstance(s, Sample)
    assert s.data.height == 2
    assert {"id", "val"}.issubset(set(s.data.columns))

    # Create Parquet via Polars
    p_pqt = tmp_path / "data.parquet"
    df.write_parquet(p_pqt)

    # Test Parquet Read
    df_pqt = read_parquet(p_pqt)
    assert df_pqt.shape == (2, 2)

    # Test Factory
    s_pqt = create_from_parquet(p_pqt)
    assert isinstance(s_pqt, Sample)


def test_aliases_point_to_spss_variants():
    assert read_sav is read_spss
    assert write_sav is write_spss
    assert create_from_sav is create_from_spss
