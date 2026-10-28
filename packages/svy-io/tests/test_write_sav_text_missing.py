"""write_sav takes text user-missing codes for string columns, as haven's write_sav does.

haven 2.5.5 reads the written file back with na_values "x", "99", "zz" and NA in those cells.
"""

import polars as pl
import pytest

from svy_io import read_sav, write_sav


DF = pl.DataFrame({"s": ["a", "x", "99", "b", "zz"], "n": [1.0, 8.0, 9.0, 2.0, 3.0]})
MISSING = {"s": {"na_values": ["x", "99", "zz"]}, "n": {"na_values": [8], "na_range": (9, 10)}}


def test_text_codes_round_trip(tmp_path):
    path = tmp_path / "text.sav"
    write_sav(DF, path, user_missing=MISSING)
    df, meta = read_sav(path)
    assert df["s"].to_list() == ["a", None, None, "b", None]
    assert df["n"].to_list() == [1.0, None, None, 2.0, 3.0]
    rules = {r["col"]: r for r in meta["user_missing"]}
    assert rules["s"]["na_values"] == ["x", "99", "zz"]
    assert rules["n"]["na_values"] == [8.0]
    assert rules["n"]["na_range"] == (9.0, 10.0)


def test_text_codes_are_kept_with_user_na(tmp_path):
    path = tmp_path / "text.sav"
    write_sav(DF, path, user_missing=MISSING)
    df, meta = read_sav(path, user_na=True)
    assert df["s"].to_list() == ["a", "x", "99", "b", "zz"]
    assert meta["labelled_columns"]["s"].na_values == ["x", "99", "zz"]


@pytest.mark.parametrize(
    ("user_missing", "message"),
    [
        ({"n": {"na_values": ["x"]}}, r"numeric columns must be numbers; columns: \['n'\]"),
        ({"s": {"na_values": ["x"], "na_range": ("a", "b")}}, r"not na_range; columns: \['s'\]"),
        (
            {"s": {"na_values": ["toolongcode"]}},
            "'toolongcode' for column 's' is longer than the 8 bytes",
        ),
    ],
    ids=["text code on numeric column", "range on string column", "code over 8 bytes"],
)
def test_invalid_codes_are_refused(tmp_path, user_missing, message):
    with pytest.raises(ValueError, match=message):
        write_sav(DF, tmp_path / "bad.sav", user_missing=user_missing)
