# tests/svy/io/test_input_errors.py
"""Bad input to svy.io raises an svy error that is also the builtin it replaces."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import svy

from svy.errors import Code, IoError, SvyError, UsageError


@pytest.fixture
def files(tmp_path) -> dict[str, Path]:
    df = pl.DataFrame({"y": [1.0, 2.0], "g": ["a", "b"], "w": [1.0, 1.0]})
    out = {"csv": tmp_path / "t.csv", "parquet": tmp_path / "t.parquet", "dir": tmp_path}
    df.write_csv(out["csv"])
    df.write_parquet(out["parquet"])
    return out


@pytest.fixture
def sample() -> svy.Sample:
    return svy.Sample(
        pl.DataFrame({"y": [1.0, 2.0], "g": [1, 2], "w": [1.0, 1.0]}), svy.Design(wgt="w")
    )


def _raises(area, kind, code, call, *, param=None, says=None):
    with pytest.raises(area) as err:
        call()
    e = err.value
    assert type(e) is getattr(area, kind.__name__)
    assert isinstance(e, kind) and isinstance(e, SvyError)
    assert e.code == code
    if param is not None:
        assert e.param == param
    if says is not None:
        assert says in e.detail
    return e


# ---- arguments the reader or writer does not take -----------------------------------


@pytest.mark.parametrize(
    "call, where",
    [
        (lambda f, s: svy.io.read_csv(f["csv"], nonsense=1), "io.read_csv"),
        (lambda f, s: svy.io.scan_csv(f["csv"], nonsense=1), "io.scan_csv"),
        (lambda f, s: svy.io.read_parquet(f["parquet"], nonsense=1), "io.read_parquet"),
        (lambda f, s: svy.io.scan_parquet(f["parquet"], nonsense=1), "io.scan_parquet"),
        (lambda f, s: svy.io.write_csv(s, f["dir"] / "o.csv", nonsense=1), "io.write_csv"),
        (
            lambda f, s: svy.io.write_parquet(s, f["dir"] / "o.parquet", nonsense=1),
            "io.write_parquet",
        ),
        (lambda f, s: svy.io.write_stata(s, f["dir"] / "o.dta", nonsense=1), "io.write_stata"),
        (lambda f, s: svy.io.write_spss(s, f["dir"] / "o.sav", nonsense=1), "io.write_spss"),
        (lambda f, s: svy.io.create_from_csv(f["csv"], nonsense=1), "io.read_csv"),
    ],
    ids=[
        "read_csv",
        "scan_csv",
        "read_parquet",
        "scan_parquet",
        "write_csv",
        "write_parquet",
        "write_stata",
        "write_spss",
        "create_from_csv",
    ],
)
def test_unknown_argument(files, sample, call, where):
    e = _raises(
        UsageError,
        TypeError,
        Code.UNKNOWN_ARGUMENT,
        lambda: call(files, sample),
        says=f"{where} got unexpected keyword argument(s): nonsense",
    )
    assert e.got == ["nonsense"]
    assert not {"source", "file", "path", "df", "store"} & set(e.expected)


def test_misspelt_argument_suggests_the_right_one(files):
    e = _raises(
        UsageError,
        TypeError,
        Code.UNKNOWN_ARGUMENT,
        lambda: svy.io.read_csv(files["csv"], seperator=";"),
    )
    assert "did you mean separator=?" in e.detail


def test_known_arguments_still_reach_the_engine(files):
    df = svy.io.read_csv(files["csv"], columns=["g"], n_rows=1)
    assert df.to_dict(as_series=False) == {"g": ["a"]}


# ---- what the engine rejects keeps its kind ----------------------------------------


@pytest.mark.parametrize(
    "call, says",
    [
        (
            lambda s, d: svy.io.write_spss(s, d / "o.sav", compress="nope"),
            "compress must be 'byte', 'none', or 'zsav'",
        ),
        (
            lambda s, d: svy.io.write_stata(s, d / "o.dta", version=99),
            "Unsupported Stata version 99",
        ),
    ],
    ids=["spss compress", "stata version"],
)
def test_invalid_option_is_a_value_error(sample, tmp_path, call, says):
    _raises(IoError, ValueError, Code.IO_WRITE_FAILED, lambda: call(sample, tmp_path), says=says)


def test_file_that_is_not_the_format_stays_an_os_error(files):
    _raises(
        IoError,
        OSError,
        Code.IO_READ_FAILED,
        lambda: svy.io.read_parquet(files["csv"]),
        says="PAR1",
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda f: svy.io.read_csv(f["csv"], columns=["y", "nope"]),
        lambda f: svy.io.read_parquet(f["parquet"], columns=["nope"]),
    ],
    ids=["csv", "parquet"],
)
def test_missing_column(files, call):
    _raises(
        IoError,
        KeyError,
        Code.MISSING_COLUMNS,
        lambda: call(files),
        param="columns",
        says='"nope"',
    )


# ---- path and sample of the wrong type ------------------------------------------------


@pytest.mark.parametrize("path", [3, None, b"t.csv"], ids=["int", "None", "bytes"])
def test_path_of_wrong_type(path):
    _raises(IoError, TypeError, Code.INVALID_TYPE, lambda: svy.io.read_csv(path), param="path")


@pytest.mark.parametrize(
    "write, frames",
    [
        (svy.io.write_csv, True),
        (svy.io.write_parquet, True),
        (svy.io.write_spss, False),
        (svy.io.write_stata, False),
        (svy.io.write_sas, False),
    ],
    ids=["csv", "parquet", "spss", "stata", "sas"],
)
def test_writing_something_that_is_not_a_sample(tmp_path, write, frames):
    e = _raises(
        IoError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: write({"y": [1]}, tmp_path / "o"),
        param="sample",
    )
    assert ("DataFrame" in e.expected) is frames


@pytest.mark.parametrize(
    "write, read, name",
    [
        (svy.io.write_csv, svy.io.read_csv, "o.csv"),
        (svy.io.write_parquet, svy.io.read_parquet, "o.parquet"),
    ],
    ids=["csv", "parquet"],
)
def test_text_writers_take_a_polars_frame(tmp_path, write, read, name):
    write(pl.DataFrame({"y": [1, 2]}), tmp_path / name)
    assert read(tmp_path / name)["y"].to_list() == [1, 2]


def test_scan_still_takes_a_glob(tmp_path):
    for i in range(2):
        pl.DataFrame({"y": [i]}).write_parquet(tmp_path / f"p{i}.parquet")
    lazy = svy.io.scan_parquet(tmp_path / "*.parquet")
    assert sorted(lazy.collect()["y"].to_list()) == [0, 1]
