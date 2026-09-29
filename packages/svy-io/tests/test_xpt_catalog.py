"""A .sas7bcat format catalog labels XPT files as it labels .sas7bdat files.

hadley.xpt holds hadley.sas7bdat's data with the same format names, except
that `workshop` carries a width (`WORKSHOP5.`) so the label set must be
matched without it.
"""

import io
import shutil
import zipfile

from pathlib import Path

import pytest

from svy_io import read_sas, read_xpt


HERE = Path(__file__).resolve().parent
SAS = HERE / "data/sas"
XPT = SAS / "hadley.xpt"
SAS7BDAT = SAS / "hadley.sas7bdat"
CATALOG = SAS / "formats.sas7bcat"


def _labels(meta):
    sets = {vl["set_name"]: vl["mapping"] for vl in meta["value_labels"]}
    return {v["name"]: sets.get(v["label_set"]) for v in meta["vars"] if v["label_set"]}


@pytest.mark.parametrize("levels", ["default", "labels", "values", "both"])
def test_xpt_with_catalog_matches_sas7bdat_with_catalog(levels):
    df, meta = read_xpt(XPT, catalog_path=CATALOG, factorize=True, levels=levels)
    expected, meta_expected = read_sas(
        str(SAS7BDAT), catalog_path=str(CATALOG), factorize=True, levels=levels
    )

    assert df.equals(expected)
    assert _labels(meta) == _labels(meta_expected)


def test_label_set_matches_the_catalog_despite_the_width():
    _, meta = read_xpt(XPT, catalog_path=CATALOG)
    workshop = next(v for v in meta["vars"] if v["name"] == "workshop")

    assert workshop["label_set"] == "WORKSHOP"
    assert workshop["fmt"] == "WORKSHOP5"
    assert _labels(meta) == {
        "workshop": {"1": "R", "2": "SAS"},
        "gender": {"f": "Female", "m": "Male"},
    }


def test_without_a_catalog_there_are_no_value_labels():
    _, meta = read_xpt(XPT)

    assert meta["value_labels"] == []


def test_read_sas_forwards_the_catalog_to_the_xpt_reader(tmp_path):
    ssp = tmp_path / "hadley.ssp"
    shutil.copy(XPT, ssp)

    df, meta = read_sas(str(ssp), catalog_path=str(CATALOG), factorize=True)
    expected, meta_expected = read_xpt(XPT, catalog_path=CATALOG, factorize=True)

    assert df.equals(expected)
    assert _labels(meta) == _labels(meta_expected)


def test_catalog_in_a_zip_labels_the_xpt(tmp_path):
    archive = tmp_path / "hadley.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(XPT, "hadley.xpt")
        z.write(CATALOG, "formats.sas7bcat")

    _, meta = read_sas(str(archive))

    assert _labels(meta) == _labels(read_xpt(XPT, catalog_path=CATALOG)[1])


def test_catalog_as_a_file_object():
    _, meta = read_xpt(XPT, catalog_path=io.BytesIO(CATALOG.read_bytes()))

    assert _labels(meta)["workshop"] == {"1": "R", "2": "SAS"}


def test_zero_rows_keeps_the_labels():
    df, meta = read_xpt(XPT, n_max=0, catalog_path=CATALOG)

    assert df.height == 0
    assert _labels(meta)["gender"] == {"f": "Female", "m": "Male"}
