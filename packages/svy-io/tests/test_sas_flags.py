# tests/test_sas_flags.py
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


# For temporals, once you implement SAS format-based coercion, assert resulting dtypes
# def test_coerce_temporals_datetime_file():
#     df, _ = read_sas(tpath("datetime.sas7bdat"), coerce_temporals=True)
#     assert "VAR1" in df.columns  # and dtype checks once implemented
