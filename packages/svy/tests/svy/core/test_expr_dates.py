# tests/svy/core/test_expr_dates.py
from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

import svy

from svy.errors import MethodError


# StataNow 19.5: datediff(a, b, "day" / "month" / "year")
STATA_DATEDIFF = [
    (date(2020, 1, 31), date(2020, 2, 29), 29, 0, 0),
    (date(2020, 1, 31), date(2020, 3, 1), 30, 1, 0),
    (date(2020, 1, 15), date(2020, 2, 14), 30, 0, 0),
    (date(2020, 1, 15), date(2020, 2, 15), 31, 1, 0),
    (date(2020, 2, 29), date(2021, 2, 28), 365, 11, 0),
    (date(2020, 2, 29), date(2021, 3, 1), 366, 12, 1),
    (date(2020, 3, 15), date(2020, 1, 16), -59, -1, 0),
    (date(2020, 3, 15), date(2020, 1, 15), -60, -2, 0),
    (date(2020, 3, 15), date(2020, 1, 14), -61, -2, 0),
    (date(2019, 12, 31), date(2020, 1, 1), 1, 0, 0),
]


def _eval(df: pl.DataFrame, e) -> list:
    return df.select(e._e.alias("out"))["out"].to_list()


@pytest.fixture
def stata_frame() -> pl.DataFrame:
    a, b, *_ = zip(*STATA_DATEDIFF)
    return pl.DataFrame({"a": list(a), "b": list(b)})


def test_days_between_matches_stata(stata_frame):
    got = _eval(stata_frame, svy.col("a").days_between(svy.col("b")))
    assert got == [r[2] for r in STATA_DATEDIFF]


def test_months_between_matches_stata(stata_frame):
    got = _eval(stata_frame, svy.col("a").months_between(svy.col("b")))
    assert got == [r[3] for r in STATA_DATEDIFF]


def test_years_between_matches_stata(stata_frame):
    got = _eval(stata_frame, svy.col("a").years_between(svy.col("b")))
    assert got == [r[4] for r in STATA_DATEDIFF]


def test_between_accepts_a_literal_date_and_nulls():
    df = pl.DataFrame({"a": [date(2020, 1, 1), None]})
    assert _eval(df, svy.col("a").days_between(date(2020, 1, 11))) == [10, None]
    assert _eval(df, svy.col("a").years_between(svy.lit(date(2030, 1, 1)))) == [10, None]


def test_between_truncates_datetimes():
    df = pl.DataFrame({"a": [datetime(2020, 1, 1, 23, 59)], "b": [datetime(2020, 1, 2, 0, 1)]})
    assert _eval(df, svy.col("a").days_between(svy.col("b"))) == [1]


def test_between_returns_int64(stata_frame):
    for e in (
        svy.col("a").days_between(svy.col("b")),
        svy.col("a").months_between(svy.col("b")),
        svy.col("a").years_between(svy.col("b")),
    ):
        assert stata_frame.select(e._e).dtypes == [pl.Int64]


def test_str_to_date_with_padded_day_month_year_numbers():
    df = pl.DataFrame({"d": [28072025, 8082025, None]})
    e = svy.col("d").to_str().pad_left(8, "0").str_to_date("%d%m%Y")
    assert _eval(df, e) == [date(2025, 7, 28), date(2025, 8, 8), None]


def test_str_to_date_strict_raises_and_lenient_gives_null():
    df = pl.DataFrame({"d": ["28072025", "99999999"]})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        _eval(df, svy.col("d").str_to_date("%d%m%Y"))
    assert _eval(df, svy.col("d").str_to_date("%d%m%Y", strict=False)) == [
        date(2025, 7, 28),
        None,
    ]


def test_str_to_date_infers_iso_format():
    df = pl.DataFrame({"d": ["2025-07-28"]})
    assert _eval(df, svy.col("d").str_to_date()) == [date(2025, 7, 28)]


def test_date_from_column_names_expressions_and_ints():
    df = pl.DataFrame({"y": [2025, 2024], "m": [7, 2], "d": [28, 29]})
    assert _eval(df, svy.date("y", "m", "d")) == [date(2025, 7, 28), date(2024, 2, 29)]
    assert _eval(df, svy.date(svy.col("y") - 1, svy.col("m"), 1)) == [
        date(2024, 7, 1),
        date(2023, 2, 1),
    ]
    assert _eval(df, svy.date(2000, "m", "d")) == [date(2000, 7, 28), date(2000, 2, 29)]


def test_date_strict_raises_on_impossible_parts():
    df = pl.DataFrame({"y": [2025], "m": [2], "d": [30]})
    with pytest.raises(pl.exceptions.ComputeError):
        _eval(df, svy.date("y", "m", "d"))


def test_date_lenient_gives_null_for_codes_and_impossible_parts():
    df = pl.DataFrame(
        {
            "y": [2025, 2025, 2025, None, 2023],
            "m": [7, 2, 98, 1, 2],
            "d": [8, 30, 1, 1, 29],
        }
    )
    assert _eval(df, svy.date("y", "m", "d", strict=False)) == [
        date(2025, 7, 8),
        None,
        None,
        None,
        None,
    ]


def test_dates_in_a_mutate_step():
    df = pl.DataFrame(
        {
            "dob": [1012024, 15062024, 30062024],
            "dose_y": [2024, 2024, 2024],
            "dose_m": [3, 6, 9],
            "dose_d": [1, 20, 30],
            "w": [1.0, 2.0, 1.5],
        }
    )
    s = svy.Sample(df, svy.Design(wgt="w"))
    birth = svy.col("dob").to_str().pad_left(8, "0").str_to_date("%d%m%Y")
    dose = svy.date("dose_y", "dose_m", "dose_d")
    s2 = s.wrangling.mutate(
        {
            "age_days": birth.days_between(dose),
            "age_months": birth.months_between(dose),
            "after_birth": dose > birth,
        }
    )
    out = s2.data
    assert out["age_days"].to_list() == [60, 5, 92]
    assert out["age_months"].to_list() == [2, 0, 3]
    assert out["after_birth"].to_list() == [True, True, True]


def test_mutate_reports_an_unparseable_date():
    df = pl.DataFrame({"d": ["28072025", "99999999"], "w": [1.0, 1.0]})
    s = svy.Sample(df, svy.Design(wgt="w"))
    with pytest.raises(MethodError, match="strict=False"):
        s.wrangling.mutate({"dt": svy.col("d").str_to_date("%d%m%Y")})
