"""`read_sas` picks the SAS reader from the file's content, not its name.

SAS Transport files are routinely published under conventional extensions
(`.ssp`, `.dat`) rather than `.xpt`, and a CPORT file can carry the very same
`.ssp` name while being an unreadable, different format.
"""

import io
import shutil
import zipfile

from pathlib import Path

import polars as pl
import pytest

from svy_io import read_sas, read_xpt, write_xpt
from svy_io.sas import _sniff_sas_format


HERE = Path(__file__).resolve().parent
SAS7BDAT = HERE / "data/sas/hadley.sas7bdat"

# The first bytes of a PROC CPORT file.
CPORT_HEAD = b"**COMPRESSED** " * 4 + b"**COMPRESSED********LIB CONTROL X64_10PR"


@pytest.fixture
def xpt(tmp_path):
    df = pl.DataFrame(
        {
            "id": [1.0, 2.0, 3.0, 4.0],
            "wgt": [10.5, 20.0, 30.25, 40.0],
            "grp": ["a", "b", "a", "c"],
        }
    )
    path = tmp_path / "data.xpt"
    write_xpt(df, path, version=5)
    return path


@pytest.fixture
def cport(tmp_path):
    path = tmp_path / "h206e.ssp"
    path.write_bytes(CPORT_HEAD + b"\x00" * 200)
    return path


def _renamed(src: Path, name: str) -> Path:
    dst = src.with_name(name)
    shutil.copy(src, dst)
    return dst


# ---- sniffing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "head, kind",
    [
        (b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000", "xport"),
        (b"HEADER RECORD*******LIBV8   HEADER RECORD!!!!!!!000000", "xport"),
        (CPORT_HEAD, "cport"),
        (b"\x00" * 64, None),
        (b"", None),
    ],
)
def test_sniff_recognises_sas_formats(tmp_path, head, kind):
    path = tmp_path / "f.bin"
    path.write_bytes(head)
    assert _sniff_sas_format(str(path)) == kind


def test_sniff_leaves_sas7bdat_to_the_native_reader():
    assert _sniff_sas_format(str(SAS7BDAT)) is None


def test_sniff_of_a_missing_file_defers_to_the_reader(tmp_path):
    assert _sniff_sas_format(str(tmp_path / "nope.sas7bdat")) is None


# ---- XPORT under any name ---------------------------------------------------


@pytest.mark.parametrize("name", ["data.ssp", "data.SSP", "data.dat", "data"])
def test_xpt_is_read_whatever_its_extension(xpt, name):
    expected, meta_expected = read_xpt(xpt, coerce_temporals=False)
    df, meta = read_sas(str(_renamed(xpt, name)))

    assert df.equals(expected)
    assert meta["vars"] == meta_expected["vars"]


def test_xpt_is_read_from_a_file_object(xpt):
    expected, _ = read_xpt(xpt, coerce_temporals=False)
    df, _ = read_sas(io.BytesIO(xpt.read_bytes()))

    assert df.equals(expected)


def test_xpt_is_read_from_a_zip(xpt, tmp_path):
    archive = tmp_path / "h188bssp.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(_renamed(xpt, "h188b.ssp"), "h188b.ssp")

    expected, _ = read_xpt(xpt, coerce_temporals=False)
    df, _ = read_sas(str(archive))

    assert df.equals(expected)


def test_zip_prefers_sas7bdat_over_xpt(xpt, tmp_path):
    archive = tmp_path / "both.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(xpt, "data.xpt")
        z.write(SAS7BDAT, "hadley.sas7bdat")

    df, _ = read_sas(str(archive))

    assert df.equals(read_sas(str(SAS7BDAT))[0])


def test_zip_without_sas_data_lists_its_contents(tmp_path):
    archive = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("readme.txt", "no data here")

    with pytest.raises(FileNotFoundError, match="readme.txt"):
        read_sas(str(archive))


def test_dispatch_forwards_row_and_column_selection(xpt):
    path = str(_renamed(xpt, "data.ssp"))

    df, _ = read_sas(path, rows_skip=1, n_max=2, cols_skip=["wgt"])

    assert df.columns == ["id", "grp"]
    assert df["id"].to_list() == [2.0, 3.0]


def test_dispatch_forwards_post_processing(xpt):
    path = str(_renamed(xpt, "data.ssp"))

    df, _ = read_sas(path, zap_empty_str=True, coerce_temporals=True)

    assert df.equals(read_xpt(xpt, zap_empty_str=True, coerce_temporals=True)[0])


def test_catalog_with_xpt_is_refused(xpt):
    with pytest.raises(ValueError, match="catalog_path applies value labels to .sas7bdat"):
        read_sas(str(_renamed(xpt, "data.ssp")), catalog_path="formats.sas7bcat")


def test_unrecognised_content_named_xpt_still_goes_to_the_xpt_reader(tmp_path):
    bogus = tmp_path / "bogus.xpt"
    bogus.write_bytes(b"not a sas file at all" * 10)

    with pytest.raises(RuntimeError, match="Failed to parse XPT"):
        read_sas(str(bogus))


# ---- encoding ---------------------------------------------------------------


def test_xpt_encoding_is_honoured(tmp_path):
    path = tmp_path / "accents.xpt"
    write_xpt(pl.DataFrame({"s": ["café"]}), path, version=5)

    as_utf8, _ = read_xpt(path)
    as_latin1, _ = read_xpt(path, encoding="latin1")
    via_sas, _ = read_sas(str(path), encoding="latin1")

    assert as_utf8["s"].to_list() == ["café"]
    assert as_latin1["s"].to_list() == ["cafÃ©"]
    assert via_sas["s"].to_list() == ["cafÃ©"]


# ---- CPORT ------------------------------------------------------------------


def test_cport_is_refused_with_a_hint(cport):
    with pytest.raises(RuntimeError, match="SAS CPORT file") as exc:
        read_sas(str(cport))

    msg = str(exc.value)
    assert str(cport) in msg
    assert ". Hint: " in msg and "PROC CIMPORT" in msg


def test_cport_is_refused_by_read_xpt_too(cport):
    with pytest.raises(RuntimeError, match="SAS CPORT file.*Hint: .*PROC CIMPORT"):
        read_xpt(cport)


def test_cport_in_a_zip_names_the_archive(cport, tmp_path):
    archive = tmp_path / "h206essp.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(cport, "h206e.ssp")

    with pytest.raises(RuntimeError, match="h206essp.zip is a SAS CPORT file"):
        read_sas(str(archive))
