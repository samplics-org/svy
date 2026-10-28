"""String user-missing codes in a .sav file are read as haven reads them.

Expected values are haven 2.5.5's reading of data/spss/labelled-str-missing.sav,
which data/spss/labelled-str-missing.R builds.
"""

from pathlib import Path

from svy_io.spss import read_sav
from svy_io.zap import zap_missing


FIXTURE = Path(__file__).parent / "data" / "spss" / "labelled-str-missing.sav"
CODES = ["x", "99", "zz"]


def test_string_codes_become_null_by_default():
    df, _ = read_sav(FIXTURE)
    assert df["s"].to_list() == ["a", None, None, "b", None]
    assert df["n"].to_list() == [1.0, None, None, 2.0, 3.0]


def test_string_codes_are_strings_in_the_variable_metadata():
    _, meta = read_sav(FIXTURE)
    by_name = {v["name"]: v.get("user_missing") for v in meta["vars"]}
    assert by_name["s"] == {"values": CODES}
    assert by_name["n"] == {"values": [8.0], "range": [9.0, 10.0]}


def test_numeric_looking_string_code_stays_a_string():
    _, meta = read_sav(FIXTURE)
    rules = {r["col"]: r for r in meta["user_missing"]}
    assert rules["s"]["na_values"] == CODES
    assert all(isinstance(v, str) for v in rules["s"]["na_values"])
    assert rules["s"]["na_range"] is None
    assert rules["n"]["na_values"] == [8.0]
    assert rules["n"]["na_range"] == (9.0, 10.0)


def test_user_na_keeps_the_codes_and_records_them():
    df, meta = read_sav(FIXTURE, user_na=True)
    assert df["s"].to_list() == ["a", "x", "99", "b", "zz"]
    labelled = meta["labelled_columns"]["s"]
    assert labelled.na_values == CODES
    assert labelled.na_range is None


def test_zap_missing_nulls_the_string_codes():
    df, meta = read_sav(FIXTURE, user_na=True)
    zapped = zap_missing(df, meta)
    assert zapped["s"].to_list() == ["a", None, None, "b", None]
    assert zapped["n"].to_list() == [1.0, None, None, 2.0, 3.0]
