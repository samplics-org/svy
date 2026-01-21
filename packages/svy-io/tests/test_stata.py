# tests/test_stata.py
from __future__ import annotations

import math
import random

from datetime import date as _date
from datetime import datetime as _datetime
from pathlib import Path

import polars as pl
import pytest


# When you wire these up, import from svy_io.readers (or __init__):
# from svy_io import read_dta, read_stata, write_dta
# For now we xfail/skip where not implemented.
try:
    from svy_io.stata import read_dta as _read_dta

    HAVE_READ_DTA = True
except Exception:
    HAVE_READ_DTA = False

HERE = Path(__file__).resolve().parent
DATA = HERE / "data/stata"


def tpath(rel: str) -> str:
    return str((DATA / rel).resolve())


# ─────────────────────────────────────────────────────────────────────────────
# read_stata / read_dta
# ─────────────────────────────────────────────────────────────────────────────


def test_stata_data_types_read_into_expected_types_45():
    if not HAVE_READ_DTA:
        pytest.xfail("stub pending")
    df, _meta = _read_dta(tpath("types.dta"))
    # vapply(typeof) in R ~ polars dtype mapping here
    got = {k: str(v) for k, v in df.schema.items()}
    # Expect numeric columns parsed as floats (double in R), strings as Utf8
    assert got == {
        "vfloat": "Float64",
        "vdouble": "Float64",
        "vlong": "Float64",
        "vint": "Float64",
        "vbyte": "Float64",
        "vstr": "String",
        "vdate": "Float64",  # pre-conversion raw numeric is ok if you don’t convert yet
        "vdatetime": "Float64",  # same
    }


def test_stata_td_tc_read_into_expected_classes():
    if not HAVE_READ_DTA:
        pytest.xfail("stub pending")
    df, _meta = _read_dta(tpath("types.dta"), coerce_temporals=True)
    # In haven: %td -> Date, %tc -> POSIXct
    # Here, assert post-conversion (when implemented). For now, just stub.
    assert df.schema["vdate"] == pl.Date
    assert isinstance(df.schema["vdatetime"], pl.Datetime)


def test_old_d_format_read_into_date_class():
    if not HAVE_READ_DTA:
        pytest.xfail("stub pending")
    df, _ = _read_dta(tpath("datetime-d.dta"), coerce_temporals=True)
    # expect a Date "2015-11-02"
    assert df.select(pl.col("date").dt.strftime("%Y-%m-%d")).to_series()[0] == "2015-11-02"


def test_tagged_double_missings_read_correctly():
    if not HAVE_READ_DTA:
        pytest.xfail("stub pending")
    df, meta = _read_dta(tpath("tagged-na-double.dta"))
    x = df["x"].to_list()
    # Once you emit TaggedNA on ingest, assert tags a/h/z on the last three.
    from svy_io.tagged_na import na_tag

    assert [na_tag(v) for v in x[5:]] == ["a", "h", "z"]
    # And labels’ tags
    # (You’ll need to expose per-column value_labels in meta similar to SAS)


def test_tagged_integer_missings_read_correctly():
    if not HAVE_READ_DTA:
        pytest.xfail("stub pending")
    df, meta = _read_dta(tpath("tagged-na-int.dta"))
    x = df["x"].to_list()
    from svy_io.tagged_na import na_tag

    assert [na_tag(v) for v in x[5:]] == ["a", "h", "z"]


def test_file_label_and_notes_stored_as_attributes():
    if not HAVE_READ_DTA:
        pytest.xfail("stub pending")
    df, meta = _read_dta(tpath("notes.dta"))
    # haven stores dataset label at top-level; we mirror as meta["file_label"]
    assert meta.get("file_label") == "This is a test dataset."
    # notes: if you choose to expose, maybe meta["notes"] = [...]
    assert isinstance(meta.get("notes"), list) and len(meta["notes"]) == 2


def test_only_selected_columns_are_read():
    # In our Python API we typically support cols_skip, not tidyselect includes.
    # You can implement an include arg to reach parity later.
    pass

    if not HAVE_READ_DTA:
        pytest.xfail("stub pending")

    def rows_after_skipping(n: int) -> int:
        df, _ = _read_dta(tpath("notes.dta"), rows_skip=n)
        return df.height

    n0 = rows_after_skipping(0)
    assert rows_after_skipping(1) == max(n0 - 1, 0)
    assert rows_after_skipping(n0 - 1) == (1 if n0 > 0 else 0)
    assert rows_after_skipping(n0 + 0) == 0
    assert rows_after_skipping(n0 + 1) == 0


def test_can_limit_the_number_of_rows_to_read_stata():
    if not HAVE_READ_DTA:
        pytest.xfail("stub pending")

    def rows_with_limit(n_max) -> int:
        df, _ = _read_dta(tpath("notes.dta"), n_max=n_max)
        return df.height

    total = rows_with_limit(None)
    assert rows_with_limit(0) == 0
    assert rows_with_limit(1) == min(1, total)
    assert rows_with_limit(total) == total
    assert rows_with_limit(total + 1) == total
    # Accept None/NA/-1 as unlimited as you prefer; adjust when implemented
    assert rows_with_limit(None) == total


# ─────────────────────────────────────────────────────────────────────────────
# write_dta (roundtrips, metadata)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from svy_io.stata import write_dta as _write_dta

    HAVE_WRITE_DTA = True
except Exception:
    HAVE_WRITE_DTA = False


def _roundtrip(tmp_path, df: pl.DataFrame, **write_kw):
    """
    Helper: write via _write_dta then read back via _read_dta.
    Returns (df_read, meta, df_returned_from_write).
    """
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("read_dta/write_dta not wired yet")
    out = tmp_path / "rt.dta"
    ret = _write_dta(df, out, **write_kw)
    df2, meta = _read_dta(str(out))
    return df2, meta, ret


def _roundtrip_var(tmp_path, values, *, dtype: pl.DataType | None = None, **write_kw):
    """
    Mirror Haven's roundtrip_var(x, 'dta'): we write a single column 'x' and read back.
    Returns the read-back Series (pl.Series).
    """
    s = pl.Series("x", values, dtype=dtype) if dtype is not None else pl.Series("x", values)
    df = pl.DataFrame({"x": s})
    df2, _meta, _ret = _roundtrip(tmp_path, df, **write_kw)
    return df2["x"]


def _is_missing(v) -> bool:
    return (v is None) or (isinstance(v, float) and math.isnan(v))


def _as_float_list(seq):
    out = []
    for v in seq:
        if v is None:
            out.append(None)
        else:
            out.append(float(v))
    return out


def test_can_roundtrip_basic_types(tmp_path):
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    # doubles
    x = [0.1, 2.5, None, -3.0]
    got = _roundtrip_var(tmp_path, x, dtype=pl.Float64, version=118, na_policy="nan")
    assert got.dtype == pl.Float64
    assert got.to_list() == x

    # integers → Stata read path usually yields Float64; compare values numerically
    xi = list(range(1, 11)) + [None]
    goti = _roundtrip_var(tmp_path, xi, dtype=pl.Int64, version=118, na_policy="nan")
    assert goti.dtype == pl.Float64
    assert goti.to_list() == _as_float_list(xi)

    # logicals → 1/0 on read
    xb = [True, False, True, None]
    gotb = _roundtrip_var(tmp_path, xb, dtype=pl.Boolean, version=118, na_policy="nan")
    # Boolean column will come back numeric (float) from Stata; check values
    vals = gotb.to_list()
    assert vals[0] == 1.0 and vals[1] == 0.0 and vals[2] == 1.0 and _is_missing(vals[3])

    # strings
    xs = list("abcdef") + [None]
    gots = _roundtrip_var(tmp_path, xs, dtype=pl.Utf8, version=118)
    want = [c for c in "abcdef"] + [""]
    got_list = gots.to_list()
    assert got_list[:-1] == want[:-1]
    assert got_list[-1] in ("", None)


def test_can_roundtrip_missing_values_as_much_as_possible(tmp_path):
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    # Scalar NA (we'll put it in a one-element column)
    g1 = _roundtrip_var(tmp_path, [None], dtype=pl.Int64, version=118)
    assert _is_missing(g1.to_list()[0])

    g2 = _roundtrip_var(tmp_path, [None], dtype=pl.Float64, version=118)
    assert _is_missing(g2.to_list()[0])

    # For strings, Stata uses "" as missing; accept "" or None
    g3 = _roundtrip_var(tmp_path, [None], dtype=pl.Utf8, version=118)
    assert g3.to_list()[0] in ("", None)


def test_can_roundtrip_date_times_and_label_preservation(tmp_path):
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    # Dates and datetimes: we currently rely on numeric encodings on read.
    d = [_date(2010, 1, 1), None, _date(2010, 1, 3)]
    ts = [_datetime(2010, 1, 1, 9, 0, 0), None, _datetime(2010, 1, 2, 12, 0, 0)]
    df = pl.DataFrame(
        {
            "d": pl.Series("d", d, dtype=pl.Date),
            "ts": pl.Series("ts", ts, dtype=pl.Datetime),
            "x": [1, 2, 3],
        }
    )

    df2, meta, _ = _roundtrip(tmp_path, df, version=118, file_label="abc", na_policy="nan")
    assert df2.height == df.height
    # When reader exposes file_label, assert equality (kept as xfail until then)
    assert meta.get("file_label") == "abc"


def test_can_roundtrip_tagged_NAs(tmp_path):
    pytest.xfail("tagged NAs on write/read not implemented yet")
    # Once implemented: create numeric column with user-missing tags .a/.b etc,
    # and assert tags persist both in data and in value label domains.


def test_infinity_gets_converted_to_NA_on_write(tmp_path):
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    s = [float("inf"), 0.0, -float("inf"), None]
    got = _roundtrip_var(tmp_path, s, dtype=pl.Float64, version=118, na_policy="nan")
    vals = got.to_list()
    assert _is_missing(vals[0])
    assert vals[1] == 0.0
    assert _is_missing(vals[2])
    assert _is_missing(vals[3])


@pytest.mark.xfail(reason="Categorical/factor support not implemented yet")
def test_factors_become_labelleds_on_write(tmp_path):
    """Polars Categorical should become integers with value labels in Stata"""
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    df = pl.DataFrame(
        {"category": pl.Series(["Low", "Medium", "High", "Low", "High"]).cast(pl.Categorical)}
    )

    df2, meta, _ = _roundtrip(tmp_path, df, version=118)

    # Should come back as numeric with value labels
    assert df2.schema["category"] == pl.Float64  # Stata reads as numeric


def test_labels_are_preserved(tmp_path):
    """Variable labels should survive roundtrip"""
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    var_labels = {"x": "X Variable Label", "y": "Y Variable Label"}

    df2, meta, _ = _roundtrip(tmp_path, df, version=118, var_labels=var_labels)

    # Extract variable labels from meta
    actual_labels = {v["name"]: v.get("label") for v in meta.get("vars", [])}

    assert actual_labels == var_labels


@pytest.mark.xfail(reason="Value labels on write not implemented yet")
def test_labelleds_are_round_tripped(tmp_path):
    """Value labels (integer -> string mappings) should survive roundtrip"""
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    df = pl.DataFrame({"status": [1, 2, 3, 1, 2]})
    value_labels = {"status": {1: "Active", 2: "Inactive", 3: "Pending"}}

    # This should work once value_labels_json is implemented in the writer
    df2, meta, _ = _roundtrip(tmp_path, df, version=118, value_labels=value_labels)

    # Check that value labels are in meta
    status_labels = None
    for vl in meta.get("value_labels", []):
        if vl["set_name"] == "status":  # Adjust based on actual structure
            status_labels = vl["mapping"]
            break

    assert status_labels == {"1": "Active", "2": "Inactive", "3": "Pending"}


def test_can_write_labelled_with_null_labels(tmp_path):
    # TBD with labelled support semantics.
    pass


def test_labels_are_converted_to_utf8(tmp_path):
    # When meta exposes label names, assert they are UTF-8 normalized.
    pass


def test_supports_stata_version_15(tmp_path):
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")
    df = pl.DataFrame({"x": list("abc"), "y": [0.1, 0.2, 0.3]})
    df2, _meta, _ = _roundtrip(tmp_path, df, version=118)  # 118 ~ Stata 15
    assert df2.height == df.height
    assert df2.schema["x"] == pl.String


def test_can_roundtrip_file_labels(tmp_path):
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")
    df = pl.DataFrame({"x": [1]})
    # no label
    df2, meta, _ = _roundtrip(tmp_path, df, version=118)
    assert meta.get("file_label") is None
    # with label
    df2, meta, _ = _roundtrip(tmp_path, df, version=118, file_label="abcd")
    assert meta.get("file_label") == "abcd"


def test_file_label_validation(tmp_path):
    """File labels longer than 80 characters should raise an error"""
    if not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    df = pl.DataFrame({"x": [1]})
    long_label = "a" * 100  # 100 characters, exceeds 80-char limit

    with pytest.raises(ValueError, match="file_label must be 80 characters or fewer"):
        _write_dta(df, tmp_path / "test.dta", version=118, file_label=long_label)


def test_variable_label_roundtrip_with_special_characters(tmp_path):
    """Variable labels with UTF-8 characters should work"""
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    df = pl.DataFrame({"x": [1, 2, 3]})
    var_labels = {"x": "Temperature (°C) — μ±σ"}

    df2, meta, _ = _roundtrip(tmp_path, df, version=118, var_labels=var_labels)

    actual_labels = {v["name"]: v.get("label") for v in meta.get("vars", [])}
    assert actual_labels == var_labels


def test_invalid_files_generate_informative_errors(tmp_path):
    # Too-long file label (>80)
    df = pl.DataFrame({"x": [1]})
    long = "a" * 100
    with pytest.raises(Exception):
        _write_dta(df, tmp_path / "x1.dta", version=118, file_label=long)
    # Invalid variable names / lengths enforcement belongs in write_dta as well.


def test_cant_write_non_integer_labels_401(tmp_path):
    df = pl.DataFrame({"x": [1.0, 2.5, 3.0]})
    # value_labels use non-integer keys -> should raise
    with pytest.raises(Exception):
        _write_dta(df, tmp_path / "x2.dta", version=118, value_labels={"x": {1.5: "b"}})


def _long_string(n, m):
    # produce n strings of length ~m by sampling uppercase letters
    out = []
    for _ in range(n):
        out.append("".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=m)))
    return out


@pytest.mark.xfail(
    reason="strL support blocked by ReadStat v1.1.9 bug - see github.com/WizardMac/ReadStat"
)
def test_can_roundtrip_long_strings_strL(tmp_path):
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    # Below and above Stata str# limit (~2045) should both work with v117+ (strL)
    for m in (400, 1000, 3000):
        x = _long_string(10, m)
        got = _roundtrip_var(tmp_path, x, dtype=pl.Utf8, version=118)
        assert got.to_list() == x


def test_write_dta_returns_input_unaltered_invisibly(tmp_path):
    if not HAVE_READ_DTA or not HAVE_WRITE_DTA:
        pytest.xfail("stub pending")

    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "dt": [
                _datetime(2022, 1, 1, 12, 0, 0),
                _datetime(2022, 1, 2, 12, 0, 0),
                _datetime(2022, 1, 3, 12, 0, 0),
                _datetime(2022, 1, 4, 12, 0, 0),
                _datetime(2022, 1, 5, 12, 0, 0),
            ],
        }
    )

    out = tmp_path / "inv.dta"
    df_returned = _write_dta(df, out, version=118)
    assert df_returned is df
