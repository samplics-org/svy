"""Zip member choice, warning location, cleanup and file-object inputs."""

from __future__ import annotations

import io
import tempfile
import zipfile

from pathlib import Path

import pytest
import svy_io.spss as spss_mod

from svy_io import read_dta, read_por, read_sas, read_sav, read_spss
from svy_io.stata import read_stata_arrow


DATA = Path(__file__).parent / "data"
SAV = DATA / "spss/labelled-num-na.sav"
OTHER_SAV = DATA / "spss/variable-label.sav"
DTA = DATA / "stata/types.dta"
OTHER_DTA = DATA / "stata/notes.dta"
SAS7BDAT = DATA / "sas/hadley.sas7bdat"


def _zip(tmp_path, members: dict[str, Path | bytes], name: str = "archive.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        for arcname, src in members.items():
            if isinstance(src, Path):
                z.write(src, arcname)
            else:
                z.writestr(arcname, src)
    return path


@pytest.fixture
def mkdtemp_calls(monkeypatch):
    made: list[str] = []
    real = tempfile.mkdtemp

    def spy(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(tempfile, "mkdtemp", spy)
    return made


# ---------------- several members ----------------


@pytest.mark.parametrize(
    "reader, first, second, ext",
    [
        (read_sav, SAV, OTHER_SAV, ".sav"),
        (read_spss, SAV, OTHER_SAV, ".sav"),
        (read_dta, DTA, OTHER_DTA, ".dta"),
        (read_stata_arrow, DTA, OTHER_DTA, ".dta"),
        (read_sas, SAS7BDAT, SAS7BDAT, ".sas7bdat"),
    ],
)
def test_several_members_warn_at_the_callers_line(tmp_path, reader, first, second, ext):
    path = _zip(tmp_path, {f"a{ext}": first, f"b{ext}": second})

    with pytest.warns(UserWarning, match=f"Using the first one: a{ext}") as rec:
        got, _ = reader(path)

    assert rec[0].filename == __file__
    assert got.equals(reader(first)[0])


def test_several_por_members_warn_at_the_callers_line(tmp_path, monkeypatch):
    def fake_native(path, *args):
        raise RuntimeError("stop after the member is chosen")

    monkeypatch.setattr(spss_mod.native, "df_parse_por_file", fake_native)
    path = _zip(tmp_path, {"a.por": b"a", "b.por": b"b"})

    with pytest.warns(UserWarning, match="Using the first one: a.por") as rec:
        with pytest.raises(RuntimeError, match="stop after"):
            read_por(path)

    assert rec[0].filename == __file__


def test_read_spss_prefers_a_sav_member_over_a_por(tmp_path, monkeypatch):
    def por_must_not_run(*args, **kwargs):
        raise AssertionError("the .por member must not be read when a .sav is present")

    monkeypatch.setattr(spss_mod.native, "df_parse_por_file", por_must_not_run)
    path = _zip(tmp_path, {"b.por": b"por bytes", "a.sav": SAV})

    df, _ = read_spss(path)

    assert df.equals(read_sav(SAV)[0])


# ---------------- .por members ----------------


@pytest.mark.parametrize("reader_name", ["read_spss", "read_por"])
def test_por_member_goes_to_the_por_parser(tmp_path, monkeypatch, reader_name):
    seen: dict[str, object] = {}

    def fake_native(path, *args):
        seen["path"] = path
        seen["bytes"] = Path(path).read_bytes()
        raise RuntimeError("stop after dispatch")

    def sav_must_not_run(*args, **kwargs):
        raise AssertionError("a .por member must not reach the .sav parser")

    monkeypatch.setattr(spss_mod.native, "df_parse_por_file", fake_native)
    monkeypatch.setattr(spss_mod.native, "df_parse_sav_file", sav_must_not_run)
    path = _zip(tmp_path, {"survey.por": b"por bytes"})

    with pytest.raises(RuntimeError, match="stop after dispatch"):
        getattr(spss_mod, reader_name)(path)

    assert seen["path"].endswith("survey.por")
    assert seen["bytes"] == b"por bytes"
    assert not Path(seen["path"]).exists()


# ---------------- cleanup ----------------


@pytest.mark.parametrize("reader, member", [(read_dta, "bad.dta"), (read_sav, "bad.sav")])
def test_extraction_dir_is_removed_when_the_parse_fails(tmp_path, mkdtemp_calls, reader, member):
    path = _zip(tmp_path, {member: b"not a data file"})

    with pytest.raises(RuntimeError):
        reader(path)

    assert len(mkdtemp_calls) == 1
    assert not Path(mkdtemp_calls[0]).exists()


def test_extraction_dir_is_removed_for_arrow_and_por(tmp_path, mkdtemp_calls, monkeypatch):
    def fake_native(path, *args):
        raise RuntimeError("stop after dispatch")

    monkeypatch.setattr(spss_mod.native, "df_parse_por_file", fake_native)

    read_stata_arrow(_zip(tmp_path, {"survey.dta": DTA}))
    with pytest.raises(RuntimeError):
        read_por(_zip(tmp_path, {"survey.por": b"por"}, name="por.zip"))

    assert len(mkdtemp_calls) == 2
    assert not any(Path(d).exists() for d in mkdtemp_calls)


# ---------------- file objects are unaffected ----------------


def test_sav_file_object_is_read(mkdtemp_calls):
    df, _ = read_sav(io.BytesIO(SAV.read_bytes()))

    assert df.equals(read_sav(SAV)[0])
    assert mkdtemp_calls == []


def test_dta_file_object_is_read(mkdtemp_calls):
    with open(DTA, "rb") as fh:
        df, _ = read_dta(fh)

    assert df.equals(read_dta(DTA)[0])
    assert mkdtemp_calls == []


def test_dta_file_object_is_read_as_arrow(mkdtemp_calls):
    table, _ = read_stata_arrow(io.BytesIO(DTA.read_bytes()))

    assert table.equals(read_stata_arrow(DTA)[0])
    assert mkdtemp_calls == []
