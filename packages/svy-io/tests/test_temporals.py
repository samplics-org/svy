# tests/test_temporals.py
"""Unit tests for svy_io.temporals (no native reader required)."""

from datetime import date, datetime, timedelta

import polars as pl

from svy_io.temporals import coerce_sas_temporals, coerce_spss_temporals


# ─────────────────────────── SAS ───────────────────────────


def test_sas_datetime_format_is_not_captured_by_date_branch():
    """DATETIME formats must convert as seconds, preserving time-of-day."""
    # 1738507332 s since 1960-01-01 == 2015-02-02 14:42:12
    df = pl.DataFrame({"dt": [1738507332.0, None]})
    meta = {"vars": [{"name": "dt", "fmt": "DATETIME20."}]}

    out = coerce_sas_temporals(df, meta)

    assert isinstance(out.schema["dt"], pl.Datetime)
    assert out["dt"][0] == datetime(2015, 2, 2, 14, 42, 12)
    assert out["dt"][1] is None


def test_sas_date_time_and_weekdate_formats():
    df = pl.DataFrame(
        {
            "d1": [20121.0],  # DATE: days since 1960-01-01
            "d2": [20121.0],  # WEEKDATE
            "t": [52932.0],  # TIME: seconds since midnight
        }
    )
    meta = {
        "vars": [
            {"name": "d1", "fmt": "DATE9."},
            {"name": "d2", "fmt": "WEEKDATE"},
            {"name": "t", "fmt": "TIME8."},
        ]
    }

    out = coerce_sas_temporals(df, meta)

    assert out.schema["d1"] == pl.Date
    assert out["d1"][0] == date(2015, 2, 2)
    assert out.schema["d2"] == pl.Date
    assert out["d2"][0] == date(2015, 2, 2)
    assert out.schema["t"] == pl.Duration
    assert out["t"][0] == timedelta(seconds=52932)


# ─────────────────────────── SPSS ───────────────────────────


def test_spss_null_format_does_not_crash():
    """Variables with fmt=None in metadata must be treated as format-less."""
    df = pl.DataFrame({"x": [1.0, 2.0]})
    meta = {"vars": [{"name": "x", "fmt": None, "label": None}]}

    out = coerce_spss_temporals(df, meta)

    assert out.schema["x"] == pl.Float64
    assert out["x"].to_list() == [1.0, 2.0]


def test_spss_no_heuristic_conversion_by_default():
    """Format-less numeric columns (e.g. 9-digit IDs) must stay numeric."""
    df = pl.DataFrame(
        {
            "respondent_id": [123456789.0, 987654321.0],  # magnitude-sniffing bait
            "update_count": [3.0, 10.0],  # name contains "date"
        }
    )
    meta = {"vars": []}

    out = coerce_spss_temporals(df, meta)

    assert out.schema["respondent_id"] == pl.Float64
    assert out["respondent_id"].to_list() == [123456789.0, 987654321.0]
    assert out.schema["update_count"] == pl.Float64
    assert out["update_count"].to_list() == [3.0, 10.0]


def test_spss_heuristics_are_opt_in():
    """With infer_formats=True, name-based inference converts *_date columns."""
    spss_days = (date(2015, 2, 2) - date(1582, 10, 14)).days
    df = pl.DataFrame({"birth_date": [float(spss_days * 86_400)]})  # seconds
    meta = {"vars": []}

    out = coerce_spss_temporals(df, meta, infer_formats=True)

    assert out.schema["birth_date"] == pl.Date
    assert out["birth_date"][0] == date(2015, 2, 2)


def test_spss_explicit_formats_still_convert():
    """Metadata-declared formats convert regardless of infer_formats."""
    spss_secs = (datetime(2015, 2, 2, 14, 42, 12) - datetime(1582, 10, 14)).total_seconds()
    df = pl.DataFrame({"when": [spss_secs], "day": [spss_secs]})
    meta = {
        "vars": [
            {"name": "when", "fmt": "DATETIME20"},
            {"name": "day", "fmt": "DATE11"},
        ]
    }

    out = coerce_spss_temporals(df, meta)

    assert isinstance(out.schema["when"], pl.Datetime)
    assert out["when"][0] == datetime(2015, 2, 2, 14, 42, 12)
    assert out.schema["day"] == pl.Date
    assert out["day"][0] == date(2015, 2, 2)
