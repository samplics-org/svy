"""Reading from zip archives, and the errors when no member matches.

Uses the real svy-io reader. Only read_sas extracts from a zip; read_spss and
read_stata take the archive as their own format and fail to parse it.
"""

from __future__ import annotations

import zipfile

import polars as pl
import pytest
import svy_io

import svy

from svy.errors.io_errors import IoError, map_os_error


@pytest.fixture
def zip_without_data(tmp_path):
    path = tmp_path / "survey.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("readme.txt", "codebook to follow")
        z.writestr("codes.csv", "code,label\n1,yes\n")
    return path


def test_read_sas_zip_without_member_names_the_members(zip_without_data):
    with pytest.raises(IoError) as exc:
        svy.read_sas(zip_without_data)

    err = exc.value
    assert err.code == "ARCHIVE_MEMBER_NOT_FOUND"
    assert err.where == "io.read_sas"
    assert "readme.txt" in err.detail
    assert "codes.csv" in err.detail
    assert ".sas7bdat" in err.detail
    assert err.extra["path"] == str(zip_without_data)
    assert err.hint
    assert isinstance(err.__cause__, FileNotFoundError)


def test_create_from_sas_zip_without_member(zip_without_data):
    with pytest.raises(IoError) as exc:
        svy.create_from_sas(zip_without_data)

    assert exc.value.code == "ARCHIVE_MEMBER_NOT_FOUND"


def test_read_sas_zip_with_xpt_member_reads(tmp_path):
    xpt = tmp_path / "h188b.xpt"
    svy_io.write_xpt(pl.DataFrame({"perwt": [100.5, 200.0]}), xpt, version=5)
    path = tmp_path / "h188b.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.write(xpt, "h188b.xpt")
        z.writestr("readme.txt", "notes")

    df = svy.read_sas(path)

    assert df["perwt"].to_list() == [100.5, 200.0]


@pytest.mark.parametrize("reader", [svy.read_spss, svy.read_stata])
def test_spss_stata_zip_is_not_reported_missing(reader, zip_without_data):
    with pytest.raises(IoError) as exc:
        reader(zip_without_data)

    assert exc.value.code not in {"FILE_NOT_FOUND", "ARCHIVE_MEMBER_NOT_FOUND"}


@pytest.mark.parametrize("reader", [svy.read_sas, svy.read_spss, svy.read_stata])
def test_missing_zip_is_file_not_found(reader, tmp_path):
    path = tmp_path / "absent.zip"

    with pytest.raises(IoError) as exc:
        reader(path)

    assert exc.value.code == "FILE_NOT_FOUND"
    assert exc.value.extra["path"] == str(path)


def test_map_os_error_names_the_missing_companion(tmp_path):
    data = tmp_path / "h188b.sas7bdat"
    data.write_bytes(b"")
    e = FileNotFoundError(2, "No such file or directory", "formats.sas7bcat")

    err = map_os_error(e, where="io.read_sas", path=data)

    assert err.code == "FILE_NOT_FOUND"
    assert err.extra["path"] == "formats.sas7bcat"


def test_map_os_error_existing_path_without_filename(tmp_path):
    data = tmp_path / "h188b.sas7bdat"
    data.write_bytes(b"")

    err = map_os_error(FileNotFoundError("lookup failed"), where="io.read_sas", path=data)

    assert err.code == "IO_READ_FAILED"
    assert err.detail == "lookup failed"
