"""
Edge cases through the real svy-io writers and readers, for SPSS, Stata and SAS Transport.

Each test pins what a format keeps, what it cannot hold, and how svy says so.
Known gaps are strict xfails, so fixing one turns its test red until the mark goes.
"""

from __future__ import annotations

import math
import warnings

from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import pytest

import svy

from svy.errors._io import IoError


FORMATS = {
    "sav": (svy.io.write_spss, svy.io.create_from_spss),
    "dta": (svy.io.write_stata, svy.io.create_from_stata),
    "xpt": (svy.io.write_sas, svy.io.create_from_xpt),
}
LABELLED = ["sav", "dta"]  # XPT carries variable labels only

FIXTURES = Path(__file__).resolve().parents[4] / "svy-io/tests/data"


def _roundtrip(tmp_path, sample, fmt):
    write, read = FORMATS[fmt]
    path = tmp_path / f"t.{fmt}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write(sample, path)
    return read(path), [str(w.message) for w in caught]


def _labelled(df, var, labels):
    return svy.Sample(df).set_value_labels(var, labels)


# ---- values and missing values ----


@pytest.mark.parametrize("fmt", FORMATS)
def test_numeric_nulls_survive(tmp_path, fmt):
    df = pl.DataFrame({"x": [1.0, None, 3.0], "n": [1, None, 3]})
    back, _ = _roundtrip(tmp_path, svy.Sample(df), fmt)
    assert back.data["x"].to_list() == [1.0, None, 3.0]
    assert back.data["n"].to_list() == [1, None, 3]


@pytest.mark.parametrize("fmt", FORMATS)
def test_all_null_numeric_column_survives(tmp_path, fmt):
    df = pl.DataFrame(
        {"x": [None, None], "k": [1.0, 2.0]}, schema={"x": pl.Float64, "k": pl.Float64}
    )
    back, _ = _roundtrip(tmp_path, svy.Sample(df), fmt)
    assert back.data["x"].to_list() == [None, None]


@pytest.mark.parametrize("fmt", FORMATS)
def test_string_null_becomes_empty_string(tmp_path, fmt):
    # None of the three formats has a missing value for text.
    df = pl.DataFrame({"s": ["a", None, ""], "k": [1.0, 2.0, 3.0]})
    back, _ = _roundtrip(tmp_path, svy.Sample(df), fmt)
    assert back.data["s"].to_list() == ["a", "", ""]


@pytest.mark.parametrize("fmt", ["sav", "dta"])
def test_nan_and_infinity_become_missing(tmp_path, fmt):
    df = pl.DataFrame({"x": [math.nan, math.inf, -math.inf, 1.0]})
    back, _ = _roundtrip(tmp_path, svy.Sample(df), fmt)
    assert back.data["x"].to_list() == [None, None, None, 1.0]


def test_xpt_keeps_infinity_and_drops_nan(tmp_path):
    df = pl.DataFrame({"x": [math.nan, math.inf, -math.inf, 1.0]})
    back, _ = _roundtrip(tmp_path, svy.Sample(df), "xpt")
    assert back.data["x"].to_list() == [None, math.inf, -math.inf, 1.0]


@pytest.mark.parametrize("fmt", FORMATS)
def test_integers_beyond_int32_survive(tmp_path, fmt):
    df = pl.DataFrame({"x": [2**31, 2**53, -(2**40)]})
    back, _ = _roundtrip(tmp_path, svy.Sample(df), fmt)
    assert back.data["x"].to_list() == [2**31, 2**53, -(2**40)]


@pytest.mark.parametrize("fmt", FORMATS)
def test_zero_rows_roundtrip(tmp_path, fmt):
    df = pl.DataFrame({"x": [], "s": []}, schema={"x": pl.Float64, "s": pl.String})
    back, _ = _roundtrip(tmp_path, svy.Sample(df), fmt)
    assert back.n_records == 0
    assert back.data.columns == ["x", "s"]


@pytest.mark.parametrize("width", [1, 100])
def test_xpt_drops_trailing_blank_rows_and_says_so(tmp_path, width):
    df = pl.DataFrame({"s": ["a" * width, "", None, "b", " ", None]})
    back, warned = _roundtrip(tmp_path, svy.Sample(df), "xpt")
    assert back.n_records == 4
    assert any("last 2 row(s)" in w for w in warned)


def test_xpt_keeps_trailing_rows_with_a_number(tmp_path):
    df = pl.DataFrame({"s": ["a", ""], "x": [1.0, None]})
    back, warned = _roundtrip(tmp_path, svy.Sample(df), "xpt")
    assert back.n_records == 2
    assert not any("row(s)" in w for w in warned)


# ---- dates ----


DATED = pl.DataFrame(
    {"d": [date(2024, 1, 31), None], "t": [datetime(2024, 1, 31, 12, 30, 5), None]}
)


@pytest.mark.parametrize("fmt", FORMATS)
def test_dates_roundtrip_as_dates(tmp_path, fmt):
    back, _ = _roundtrip(tmp_path, svy.Sample(DATED), fmt)
    assert back.data["d"].to_list() == [date(2024, 1, 31), None]
    assert back.data["t"].to_list() == [datetime(2024, 1, 31, 12, 30, 5), None]


def test_stata_writes_times_as_milliseconds_and_says_so(tmp_path):
    df = pl.DataFrame({"t": [time(12, 30, 5)], "dur": [timedelta(seconds=90)]})
    back, warned = _roundtrip(tmp_path, svy.Sample(df), "dta")
    assert back.data["t"].to_list() == [45_005_000.0]
    assert back.data["dur"].to_list() == [90_000.0]
    assert any("t, dur" in w for w in warned)


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="svy-io test data not available")
@pytest.mark.parametrize(
    "source, read, column, first",
    [
        ("spss/datetime.sav", svy.io.read_spss, "date", date(2014, 9, 22)),
        ("stata/datetime-d.dta", svy.io.read_stata, "date", date(2015, 11, 2)),
        ("sas/datetime.sas7bdat", svy.io.read_sas, "VAR2", date(2015, 2, 2)),
    ],
)
def test_dates_in_files_read_as_dates(source, read, column, first):
    assert read(FIXTURES / source)[column][0] == first


# ---- names ----


@pytest.mark.parametrize("fmt", FORMATS)
def test_mixed_case_names_survive(tmp_path, fmt):
    df = pl.DataFrame({"HH1": [1.0], "Q1a": [2.0]})
    back, _ = _roundtrip(tmp_path, svy.Sample(df), fmt)
    assert back.data.columns == ["HH1", "Q1a"]


@pytest.mark.parametrize("fmt", FORMATS)
def test_invalid_names_fail_with_a_readable_message(tmp_path, fmt):
    df = pl.DataFrame({"q1.a": [1.0]})
    with pytest.raises(IoError, match=r"(?i)name") as err:
        _roundtrip(tmp_path, svy.Sample(df), fmt)
    assert "rc=" not in str(err.value)


def test_xpt_rejects_names_longer_than_32(tmp_path):
    with pytest.raises(IoError, match="at most 32"):
        _roundtrip(tmp_path, svy.Sample(pl.DataFrame({"v" * 40: [1.0]})), "xpt")


def test_xpt_rejects_non_ascii_names(tmp_path):
    with pytest.raises(IoError, match="ASCII"):
        _roundtrip(tmp_path, svy.Sample(pl.DataFrame({"région": [1.0]})), "xpt")


def test_write_failure_is_reported_as_a_write(tmp_path):
    with pytest.raises(IoError) as err:
        _roundtrip(tmp_path, svy.Sample(pl.DataFrame({"q1.a": [1.0]})), "dta")
    assert err.value.code == "IO_WRITE_FAILED"


# ---- value labels ----


@pytest.mark.parametrize("fmt", LABELLED)
@pytest.mark.parametrize(
    "labels",
    [
        {-1: "inapplicable", 1: "yes", 2: "no"},  # negative reserve codes
        {1: "uncodeable", 2: "uncodeable"},  # shared text
        {1: "yes", 2: "no", 9: "don't know"},  # a code with no data
        {1: "Côte d'Ivoire", 2: "São Tomé"},  # non-ASCII text
    ],
    ids=["negative", "shared-text", "unused-code", "non-ascii"],
)
def test_value_labels_survive(tmp_path, fmt, labels):
    df = pl.DataFrame({"x": [-1.0, 1.0, 2.0]})
    back, _ = _roundtrip(tmp_path, _labelled(df, "x", labels), fmt)
    assert back.meta.value_labels["x"] == {str(k): v for k, v in labels.items()}


@pytest.mark.parametrize("fmt", LABELLED)
@pytest.mark.parametrize("key", [1, 1.0, "1"], ids=["int", "float", "text"])
def test_label_codes_of_any_spelling_are_written(tmp_path, fmt, key):
    df = pl.DataFrame({"x": [1.0, 2.0]})
    back, _ = _roundtrip(tmp_path, _labelled(df, "x", {key: "yes"}), fmt)
    assert back.meta.value_labels["x"] == {"1": "yes"}


def test_spss_keeps_fractional_and_text_codes(tmp_path):
    df = pl.DataFrame({"x": [1.5, 2.0], "s": ["a", "b"]})
    sample = _labelled(df, "x", {1.5: "half", 2: "two"}).set_value_labels("s", {"a": "Alpha"})
    back, _ = _roundtrip(tmp_path, sample, "sav")
    assert back.meta.value_labels["x"] == {"1.5": "half", "2": "two"}
    assert back.meta.value_labels["s"] == {"a": "Alpha"}


def test_stata_skips_label_sets_it_cannot_hold_and_says_so(tmp_path):
    df = pl.DataFrame({"x": [1.5, 2.0], "s": ["a", "b"], "ok": [1.0, 2.0]})
    sample = (
        _labelled(df, "x", {1.5: "half"})
        .set_value_labels("s", {"a": "Alpha"})
        .set_value_labels("ok", {1: "yes"})
    )
    back, warned = _roundtrip(tmp_path, sample, "dta")
    assert back.meta.value_labels["ok"] == {"1": "yes"}
    assert not back.meta.value_labels.get("x") and not back.meta.value_labels.get("s")
    assert any("s, x" in w for w in warned)


def test_xpt_says_value_labels_are_not_written(tmp_path):
    sample = _labelled(pl.DataFrame({"x": [1.0]}), "x", {1: "yes"})
    _, warned = _roundtrip(tmp_path, sample, "xpt")
    assert any("not written" in w for w in warned)


@pytest.mark.parametrize("fmt", FORMATS)
def test_variable_labels_survive(tmp_path, fmt):
    sample = svy.Sample(pl.DataFrame({"x": [1.0]})).set_var_label("x", "Région (déclarée)")
    back, _ = _roundtrip(tmp_path, sample, fmt)
    assert back.meta.var_labels["x"] == "Région (déclarée)"


# ---- across formats, from files other software wrote ----


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="svy-io test data not available")
@pytest.mark.parametrize("target", LABELLED)
@pytest.mark.parametrize(
    "source, reader",
    [
        ("spss/labelled-num.sav", svy.io.create_from_spss),
        ("spss/labelled-str.sav", svy.io.create_from_spss),
        ("spss/labelled-num-na.sav", svy.io.create_from_spss),
        ("stata/types.dta", svy.io.create_from_stata),
        ("stata/tagged-na-int.dta", svy.io.create_from_stata),
        ("sas/hadley.xpt", svy.io.create_from_xpt),
    ],
)
def test_read_then_write_keeps_data_and_writable_labels(tmp_path, source, reader, target):
    original = reader(FIXTURES / source)
    back, warned = _roundtrip(tmp_path, original, target)
    assert back.n_records == original.n_records
    assert [c.lower() for c in back.data.columns] == [c.lower() for c in original.data.columns]
    skipped = " ".join(warned)
    for var, labels in original.meta.value_labels.items():
        if labels and var not in skipped:
            kept = back.meta.value_labels.get(var) or back.meta.value_labels.get(var.lower())
            assert kept == labels, var


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="svy-io test data not available")
def test_xpt_ibm_denormal_zero_reads_as_zero():
    # Written by SAS: NHANES stores a zero phlebotomy weight as an IBM denormal.
    w = svy.io.read_xpt(FIXTURES / "sas/nhanes_ghb_l.xpt")["WTPH2YR"]
    assert (w == 0).sum() == 449
    assert w.min() == 0


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="svy-io test data not available")
@pytest.mark.parametrize(
    "source, read",
    [
        ("stata/tagged-na-double.dta", svy.io.read_stata),
        ("stata/tagged-na-int.dta", svy.io.read_stata),
        ("sas/tagged-na.sas7bdat", svy.io.read_sas),
    ],
)
def test_tagged_missings_read_as_nulls(source, read):
    x = read(FIXTURES / source)["x"]
    assert x.dtype == pl.Float64
    assert x.null_count() > 0
