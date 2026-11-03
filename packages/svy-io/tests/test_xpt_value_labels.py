"""write_xpt value labels: format names on the data, a PROC FORMAT CNTLIN dataset beside it."""

import shutil
import subprocess

from datetime import date

import polars as pl
import pytest

from svy_io import read_sas_arrow, write_xpt


pd = pytest.importorskip("pandas")


def _cntlin_rows(path):
    """The formats file read by pandas' pure-Python XPT reader, independent of ReadStat."""
    with pd.read_sas(path, format="xport", iterator=True, encoding="utf-8") as reader:
        assert [f["name"] for f in reader.fields] == [
            b"FMTNAME",
            b"START",
            b"END",
            b"LABEL",
            b"TYPE",
        ]
        # pandas raises StopIteration on a dataset with no rows.
        if reader.nobs == 0:
            return []
        return [tuple(row) for row in reader.read().itertuples(index=False)]


def _v5_formats(path):
    with pd.read_sas(path, format="xport", iterator=True) as reader:
        return {f["name"].decode().strip(): f["nform"].decode().strip() for f in reader.fields}


def _formats(path):
    table, _ = read_sas_arrow(path)
    return {f.name: (f.metadata or {}).get(b"format", b"").decode() for f in table.schema}


@pytest.fixture
def survey():
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "sex": [1, 2, 1],
            "region": ["N", "S", "N"],
            "score": [1.5, 2.5, 3.0],
        }
    )


SURVEY_LABELS = {
    "sex": {1: "Male", 2: "Female"},
    "region": {"N": "North", "S": "South"},
    "score": {2.5: "Half"},
}


@pytest.mark.parametrize("version", [5, 8])
def test_cntlin_table_holds_one_row_per_code(tmp_path, survey, version):
    fmts = tmp_path / "fmts.xpt"
    write_xpt(
        survey, tmp_path / "d.xpt", version=version, value_labels=SURVEY_LABELS, formats_path=fmts
    )
    assert _cntlin_rows(fmts) == [
        ("SEXF", "1", "1", "Male", "N"),
        ("SEXF", "2", "2", "Female", "N"),
        ("REGIONF", "N", "N", "North", "C"),
        ("REGIONF", "S", "S", "South", "C"),
        ("SCOREF", "2.5", "2.5", "Half", "N"),
    ]


def test_formats_file_is_a_v5_member_named_formats(tmp_path, survey):
    fmts = tmp_path / "fmts.xpt"
    write_xpt(survey, tmp_path / "d.xpt", value_labels=SURVEY_LABELS, formats_path=fmts)
    assert fmts.read_bytes()[:80].startswith(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!")
    with pd.read_sas(fmts, format="xport", iterator=True) as reader:
        assert reader.member_info["set_name"] == "FORMATS"


@pytest.mark.parametrize("version", [5, 8])
def test_labelled_columns_carry_their_format_name(tmp_path, survey, version):
    data = tmp_path / "d.xpt"
    write_xpt(
        survey,
        data,
        version=version,
        value_labels=SURVEY_LABELS,
        formats_path=tmp_path / "fmts.xpt",
    )
    read = _v5_formats(data) if version == 5 else _formats(data)
    assert read == {"id": "", "sex": "SEXF", "region": "$REGIONF", "score": "SCOREF"}


def test_v5_names_truncate_to_eight_and_stay_unique(tmp_path):
    df = pl.DataFrame(
        {
            "income_1": [1],
            "income_2": [1],
            "region_a": ["a"],
            "x": [1],
            "q1": [1],
            "abcdefgh": ["a"],
            "abcdef": [1],
        }
    )
    labels = {
        col: ({"a": "A"} if df.schema[col] == pl.String else {1: "One"}) for col in df.columns
    }
    data, fmts = tmp_path / "d.xpt", tmp_path / "fmts.xpt"
    write_xpt(df, data, version=5, value_labels=labels, formats_path=fmts)
    expected = {
        "income_1": "INCOME_F",
        "income_2": "INCOME2F",
        "region_a": "$REGIONF",
        "x": "X_F",
        "q1": "Q1F",
        "abcdefgh": "$ABCDEFF",
        "abcdef": "ABCDEF2F",
    }
    assert _v5_formats(data) == expected
    assert [r[0] for r in _cntlin_rows(fmts)] == [n.lstrip("$") for n in expected.values()]


def test_v8_names_keep_up_to_thirty_one_characters(tmp_path):
    num, text = "a" * 32, "b" * 32
    df = pl.DataFrame({num: [1], text: ["z"]})
    data, fmts = tmp_path / "d.xpt", tmp_path / "fmts.xpt"
    write_xpt(df, data, value_labels={num: {1: "One"}, text: {"z": "Zed"}}, formats_path=fmts)
    assert _formats(data) == {num: "A" * 30 + "F", text: "$" + "B" * 29 + "F"}
    assert [r[0] for r in _cntlin_rows(fmts)] == ["A" * 30 + "F", "B" * 29 + "F"]


def test_codes_are_written_in_shortest_round_trip_form(tmp_path):
    df = pl.DataFrame({"v": [0.1]})
    labels = {"v": {True: "true", "7": "seven", 0.1: "tenth", -3.0: "neg", 1e-05: "tiny"}}
    fmts = tmp_path / "fmts.xpt"
    write_xpt(df, tmp_path / "d.xpt", value_labels=labels, formats_path=fmts)
    assert [r[1] for r in _cntlin_rows(fmts)] == ["1", "7", "0.1", "-3", "1e-05"]


def test_an_empty_label_set_attaches_no_format(tmp_path, survey):
    data, fmts = tmp_path / "d.xpt", tmp_path / "fmts.xpt"
    write_xpt(survey, data, version=5, value_labels={"sex": {}}, formats_path=fmts)
    assert _v5_formats(data)["sex"] == ""
    assert _cntlin_rows(fmts) == []


def test_empty_label_sets_need_no_formats_path(tmp_path, survey):
    data = tmp_path / "d.xpt"
    write_xpt(survey, data, version=5, value_labels={"sex": {}})
    assert _v5_formats(data)["sex"] == ""


def test_formats_path_without_labels_writes_an_empty_table(tmp_path, survey):
    fmts = tmp_path / "fmts.xpt"
    write_xpt(survey, tmp_path / "d.xpt", formats_path=fmts)
    assert _cntlin_rows(fmts) == []


def test_reader_list_form_is_accepted(tmp_path, survey):
    fmts = tmp_path / "fmts.xpt"
    labels = [{"col": "sex", "labels": {"1": "Male"}}]
    write_xpt(survey, tmp_path / "d.xpt", value_labels=labels, formats_path=fmts)
    assert _cntlin_rows(fmts) == [("SEXF", "1", "1", "Male", "N")]


@pytest.mark.parametrize(
    ("frame", "labels", "match"),
    [
        (
            {"x": [1]},
            {"y": {1: "a"}},
            r"value_labels name columns that are not in the DataFrame: \['y'\]",
        ),
        (
            {"x": [date(2020, 1, 1)]},
            {"x": {1: "a"}},
            r"numeric and text columns only; these are not: \['x'\]",
        ),
        ({"x": [1]}, {"x": {"one": "a"}}, r"must be finite numbers: \{'x': \['one'\]\}"),
        ({"x": [1.0]}, {"x": {float("nan"): "a"}}, r"must be finite numbers: \{'x': \[nan\]\}"),
        ({"x": [1]}, {"x": {1: "a\x00b"}}, r"value labels cannot contain NUL \(\\x00\): \['x'\]"),
        (
            {"s": ["a"]},
            {"s": {"a\x00": "a"}},
            r"value labels cannot contain NUL \(\\x00\): \['s'\]",
        ),
        ({"x": [1]}, {"x": {1: "é" * 101}}, r"at most 200 bytes in XPT: \['x'\]"),
        ({"s": ["a"]}, {"s": {"a" * 201: "a"}}, r"at most 200 bytes in XPT: \['s'\]"),
    ],
)
def test_bad_value_labels_are_rejected_before_writing(tmp_path, frame, labels, match):
    data, fmts = tmp_path / "d.xpt", tmp_path / "fmts.xpt"
    with pytest.raises(ValueError, match=match):
        write_xpt(pl.DataFrame(frame), data, value_labels=labels, formats_path=fmts)
    assert not data.exists() and not fmts.exists()


def test_labels_without_formats_path_are_rejected(tmp_path, survey):
    data = tmp_path / "d.xpt"
    with pytest.raises(ValueError, match="XPT files hold no value labels; pass formats_path="):
        write_xpt(survey, data, value_labels=SURVEY_LABELS)
    assert not data.exists()


def test_formats_path_must_differ_from_path(tmp_path, survey):
    data = tmp_path / "d.xpt"
    with pytest.raises(ValueError, match="formats_path must differ from path"):
        write_xpt(survey, data, value_labels=SURVEY_LABELS, formats_path=data)
    assert not data.exists()


def _r_has(*packages):
    if shutil.which("Rscript") is None:
        return False
    check = "quit(status = !all(sapply(commandArgs(TRUE), requireNamespace, quietly = TRUE)))"
    return subprocess.run(["Rscript", "-e", check, *packages], capture_output=True).returncode == 0


def _rscript(code, *args):
    out = subprocess.run(["Rscript", "-e", code, *map(str, args)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return [line[4:] for line in out.stdout.splitlines() if line.startswith("OUT ")]


@pytest.mark.skipif(not _r_has("foreign", "Hmisc"), reason="needs R with foreign and Hmisc")
def test_hmisc_applies_the_formats_to_the_data(tmp_path, survey):
    data, fmts = tmp_path / "d.xpt", tmp_path / "fmts.xpt"
    write_xpt(survey, data, version=5, value_labels=SURVEY_LABELS, formats_path=fmts)
    code = """
    a <- commandArgs(TRUE)
    d <- suppressWarnings(Hmisc::sasxport.get(a[1], formats = foreign::read.xport(a[2])))
    for (v in c("sex", "region", "score")) cat("OUT", v, as.character(d[[v]]), "\n")
    """
    assert _rscript(code, data, fmts) == [
        "sex Male Female Male ",
        "region North South North ",
        "score NA Half NA ",
    ]


@pytest.mark.skipif(not _r_has("haven"), reason="needs R with haven")
def test_haven_reads_v8_format_names(tmp_path):
    long = "c" * 32
    df = pl.DataFrame({"sex": [1], long: ["z"]})
    data = tmp_path / "d.xpt"
    labels = {"sex": {1: "Male"}, long: {"z": "Zed"}}
    write_xpt(df, data, value_labels=labels, formats_path=tmp_path / "fmts.xpt")
    code = """
    d <- haven::read_xpt(commandArgs(TRUE)[1])
    for (v in names(d)) cat("OUT", attr(d[[v]], "format.sas"), "\n")
    """
    assert _rscript(code, data) == ["SEXF ", "$" + "C" * 29 + "F "]
