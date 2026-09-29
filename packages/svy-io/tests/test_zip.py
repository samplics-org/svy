"""SPSS and Stata readers on zip archives."""

from __future__ import annotations

import zipfile

from pathlib import Path

import pytest

from svy_io import read_dta, read_por, read_sav, read_spss
from svy_io.stata import read_stata_arrow


DATA = Path(__file__).parent / "data"
SAV = DATA / "spss/labelled-num-na.sav"
DTA = DATA / "stata/types.dta"


def _zip(tmp_path, members: dict[str, Path | str], name: str = "archive.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        for arcname, src in members.items():
            if isinstance(src, Path):
                z.write(src, arcname)
            else:
                z.writestr(arcname, src)
    return path


@pytest.mark.parametrize("reader", [read_sav, read_spss])
def test_sav_member_is_read(tmp_path, reader):
    path = _zip(tmp_path, {"readme.txt": "notes", "data/survey.sav": SAV})

    df, meta = reader(path)
    plain, plain_meta = read_sav(SAV)

    assert df.equals(plain)
    assert meta["vars"] == plain_meta["vars"]


def test_read_spss_forwards_its_arguments_for_a_zip(tmp_path):
    path = _zip(tmp_path, {"survey.sav": SAV})

    df, _ = read_spss(path, n_max=1)

    assert df.height == 1


def test_uppercase_member_and_archive_names(tmp_path):
    path = _zip(tmp_path, {"SURVEY.SAV": SAV}, name="ARCHIVE.ZIP")

    df, _ = read_spss(path)

    assert df.equals(read_sav(SAV)[0])


def test_first_sav_member_is_used_with_a_warning(tmp_path):
    path = _zip(tmp_path, {"a.sav": SAV, "b.sav": SAV})

    with pytest.warns(UserWarning, match="Using the first one: a.sav"):
        read_sav(path)


@pytest.mark.parametrize(
    "reader, exts",
    [
        (read_sav, ".sav/.zsav"),
        (read_por, ".por"),
        (read_spss, ".sav/.zsav/.por"),
        (read_dta, ".dta"),
        (read_stata_arrow, ".dta"),
    ],
)
def test_zip_without_a_matching_member_lists_its_files(tmp_path, reader, exts):
    path = _zip(tmp_path, {"readme.txt": "notes", "codes.csv": "a,b\n"})

    with pytest.raises(FileNotFoundError) as exc:
        reader(path)

    msg = str(exc.value)
    assert f"contains no {exts} files" in msg
    assert "readme.txt" in msg and "codes.csv" in msg


def test_read_sav_ignores_a_dta_member(tmp_path):
    path = _zip(tmp_path, {"survey.dta": DTA})

    with pytest.raises(FileNotFoundError, match="contains no .sav/.zsav files"):
        read_sav(path)


def test_dta_member_is_read(tmp_path):
    path = _zip(tmp_path, {"readme.txt": "notes", "survey.dta": DTA})

    df, meta = read_dta(path)
    plain, plain_meta = read_dta(DTA)

    assert df.equals(plain)
    assert meta["vars"] == plain_meta["vars"]


def test_read_stata_arrow_reads_a_dta_member(tmp_path):
    path = _zip(tmp_path, {"survey.dta": DTA})

    table, _ = read_stata_arrow(path)

    assert table.equals(read_stata_arrow(DTA)[0])


def test_invalid_zip_is_refused(tmp_path):
    path = tmp_path / "broken.zip"
    path.write_bytes(b"not a zip")

    with pytest.raises(ValueError, match="not a valid zip archive"):
        read_dta(path)


def test_extraction_dir_is_removed(tmp_path, monkeypatch):
    import tempfile

    made = []
    real = tempfile.mkdtemp

    def spy(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(tempfile, "mkdtemp", spy)
    read_dta(_zip(tmp_path, {"survey.dta": DTA}))
    read_spss(_zip(tmp_path, {"survey.sav": SAV}, name="spss.zip"))

    assert len(made) == 2
    assert not any(Path(d).exists() for d in made)
