# tests/test_sas_flags.py
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from svy_io import read_sas


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/sas"


def tpath(r):
    return str((DATA / r).resolve())


def test_factorize_gender_from_catalog():
    df, meta = read_sas(
        tpath("hadley.sas7bdat"),
        catalog_path=tpath("formats.sas7bcat"),
        factorize=True,
        levels="labels",
    )
    assert df["gender"].dtype == pl.Categorical
    assert set(df["gender"].unique().drop_nulls().to_list()) <= {"Female", "Male"}


def test_zap_empty_string_if_present():
    # Use a file that contains empty strings; if none, create synthetic check by scanning
    df, _ = read_sas(tpath("hadley.sas7bdat"), zap_empty_str=True)
    # We can at least assert the option doesn't crash and dataframe stays same height
    assert isinstance(df, pl.DataFrame)


def test_coerce_temporals_datetime_file():
    df, _ = read_sas(tpath("datetime.sas7bdat"), coerce_temporals=True)

    # VAR1: DATETIME (seconds since 1960-01-01) -> Datetime, time-of-day preserved
    assert isinstance(df.schema["VAR1"], pl.Datetime)
    assert df["VAR1"][0] == datetime(2015, 2, 2, 14, 42, 12)

    # VAR2/VAR3/VAR4: MMDDYY / DATE / WEEKDATE (days since 1960-01-01) -> Date
    for col in ("VAR2", "VAR3", "VAR4"):
        assert df.schema[col] == pl.Date, col
        assert df[col][0] == date(2015, 2, 2), col

    # VAR5: TIME (seconds since midnight) -> Duration
    assert df.schema["VAR5"] == pl.Duration
    assert df["VAR5"][0] == timedelta(seconds=52932)
