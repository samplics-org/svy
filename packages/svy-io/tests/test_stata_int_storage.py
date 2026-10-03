# tests/test_stata_int_storage.py
"""Integer columns are written as Stata byte/int/long and read back as Int64."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from svy_io import read_sas, read_sav, read_xpt, write_sav, write_xpt
from svy_io.stata import read_dta, write_dta
from svy_io.tagged_na import TaggedNA, na_tag


DATA = Path(__file__).resolve().parent / "data"


def _roundtrip(tmp_path, df: pl.DataFrame, **kw):
    out = tmp_path / "ints.dta"
    write_dta(df, out, version=118, **kw)
    return read_dta(str(out))


def _kinds(meta) -> dict[str, str]:
    return {v["name"]: v["kind"] for v in meta["vars"]}


@pytest.mark.parametrize(
    "dtype,values,kind",
    [
        (pl.Int8, [-127, 0, 100], "int8"),
        (pl.Int8, [-128, 0], "int16"),
        (pl.Int8, [101], "int16"),
        (pl.Int16, [-32767, 32740], "int16"),
        (pl.Int16, [-32768], "int32"),
        (pl.Int16, [32741], "int32"),
        (pl.Int32, [-2147483647, 2147483620], "int32"),
        (pl.Int32, [-2147483648], "double"),
        (pl.Int32, [2147483621], "double"),
        (pl.Int64, [1, 2, 3], "int8"),
        (pl.Int64, [-1000, 70000], "int32"),
        (pl.Int64, [2**40, -(2**40)], "double"),
        (pl.UInt8, [0, 255], "int16"),
        (pl.UInt16, [65535], "int32"),
        (pl.UInt32, [4294967295], "double"),
        (pl.UInt64, [0, 7], "int8"),
    ],
)
def test_integer_column_gets_smallest_stata_storage(tmp_path, dtype, values, kind):
    df = pl.DataFrame({"x": pl.Series(values + [None], dtype=dtype)})

    got, meta = _roundtrip(tmp_path, df)

    assert _kinds(meta) == {"x": kind}
    assert got.schema["x"] == (pl.Float64 if kind == "double" else pl.Int64)
    assert got["x"].to_list() == values + [None]


def test_all_null_integer_column_is_byte(tmp_path):
    df = pl.DataFrame({"x": pl.Series([None, None], dtype=pl.Int32)})

    got, meta = _roundtrip(tmp_path, df)

    assert _kinds(meta) == {"x": "int8"}
    assert got.schema["x"] == pl.Int64
    assert got["x"].to_list() == [None, None]


def test_storage_is_chosen_per_column(tmp_path):
    df = pl.DataFrame(
        {
            "b": pl.Series([1, None, 3], dtype=pl.Int64),
            "i": pl.Series([1, 300, None], dtype=pl.Int64),
            "l": pl.Series([None, 1, 100_000], dtype=pl.Int64),
            "d": pl.Series([1, 2, 3_000_000_000], dtype=pl.Int64),
            "f": [0.5, None, 2.0],
        }
    )

    got, meta = _roundtrip(tmp_path, df)

    assert _kinds(meta) == {"b": "int8", "i": "int16", "l": "int32", "d": "double", "f": "double"}
    assert got.schema == pl.Schema(
        {"b": pl.Int64, "i": pl.Int64, "l": pl.Int64, "d": pl.Float64, "f": pl.Float64}
    )
    assert got.to_dict(as_series=False) == {
        "b": [1, None, 3],
        "i": [1, 300, None],
        "l": [None, 1, 100_000],
        "d": [1.0, 2.0, 3_000_000_000.0],
        "f": [0.5, None, 2.0],
    }


@pytest.mark.parametrize("version", [113, 114, 115, 117, 118, 119])
def test_integer_storage_on_every_supported_format(tmp_path, version):
    df = pl.DataFrame({"b": [1, 2], "i": [1, 32740], "l": [1, 2147483620]})
    out = tmp_path / f"ints{version}.dta"

    write_dta(df, out, version=version)
    got, meta = read_dta(str(out))

    assert _kinds(meta) == {"b": "int8", "i": "int16", "l": "int32"}
    assert got.equals(df)


def test_value_labels_match_integer_codes(tmp_path):
    df = pl.DataFrame({"v106": [0, 1, 2, 3, None], "big": [1, 2, 70_000, 1, 2]})
    value_labels = {
        "v106": {0: "None", 1: "Primary", 2: "Secondary", 3: "Higher"},
        "big": {1: "one", 70_000: "many"},
    }

    got, meta = _roundtrip(tmp_path, df, value_labels=value_labels)

    assert _kinds(meta) == {"v106": "int8", "big": "int32"}
    assert got.schema["v106"] == pl.Int64
    assert {v["name"]: v["label_set"] for v in meta["vars"]} == {"v106": "v106", "big": "big"}
    mappings = {vl["set_name"]: vl["mapping"] for vl in meta["value_labels"]}
    # A code read from the data spells the same as its label key.
    assert {str(v) for v in got["v106"].drop_nulls().to_list()} == set(mappings["v106"])
    assert mappings["big"][str(got["big"][2])] == "many"

    labelled, _ = read_dta(str(tmp_path / "ints.dta"), factorize=True)
    assert labelled["v106"].to_list() == ["None", "Primary", "Secondary", "Higher", None]
    assert labelled["big"].to_list() == ["one", "2", "many", "one", "2"]


def test_tagged_missing_on_integer_column_is_written_as_double(tmp_path):
    # Tagged values send the column through the Float64 path; their tags are
    # not written yet, so they come back as plain missings.
    df = pl.DataFrame({"x": pl.Series([1, TaggedNA("a"), 3, None], dtype=pl.Object)})

    got, meta = _roundtrip(tmp_path, df)

    assert _kinds(meta) == {"x": "double"}
    assert got["x"].to_list() == [1.0, None, 3.0, None]


def test_stata_integer_storage_reads_as_int64():
    df, meta = read_dta(str(DATA / "stata/types.dta"))

    assert {k: v for k, v in _kinds(meta).items() if k != "vstr"} == {
        "vfloat": "float",
        "vdouble": "double",
        "vlong": "int32",
        "vint": "int16",
        "vbyte": "int8",
        "vdate": "int32",
        "vdatetime": "double",
    }
    for col in ("vlong", "vint", "vbyte", "vdate"):
        assert df.schema[col] == pl.Int64


def test_tagged_missings_on_integer_storage_are_hydrated():
    df, meta = read_dta(str(DATA / "stata/tagged-na-int.dta"))

    assert _kinds(meta) == {"x": "int32"}
    assert meta["tagged_missings"] == [{"col": "x", "rows": [5, 6, 7], "tags": ["a", "h", "z"]}]
    x = df["x"].to_list()
    assert x[:5] == [1, 2, 3, 4, 5]
    assert all(type(v) is int for v in x[:5])
    assert [na_tag(v) for v in x[5:]] == ["a", "h", "z"]


def test_sas_and_spss_numerics_stay_double(tmp_path):
    frames = [
        read_sas(str(DATA / "sas/hadley.sas7bdat")),
        read_xpt(str(DATA / "sas/hadley.xpt")),
        *(read_sav(str(p)) for p in sorted((DATA / "spss").glob("*.sav"))),
    ]

    ints = pl.DataFrame({"a": pl.Series([1, 2, None], dtype=pl.Int8), "b": [1, 2**20, 3]})
    write_sav(ints, tmp_path / "ints.sav")
    write_xpt(ints, tmp_path / "ints.xpt")
    frames += [read_sav(str(tmp_path / "ints.sav")), read_xpt(str(tmp_path / "ints.xpt"))]

    for df, meta in frames:
        assert set(_kinds(meta).values()) <= {"double", "string"}
        assert not any(dt.is_integer() for dt in df.schema.values())
