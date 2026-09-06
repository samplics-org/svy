# tests/test_spss_encoding.py
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from svy_io.sas import read_sas
from svy_io.spss import read_sav, write_sav


CAFE_CP1252 = b"caf\xe9"  # "café" in Windows-1252, invalid as UTF-8
HERE = Path(__file__).resolve().parent


def _sav_with_bytes(tmp_path: Path, value: bytes, label: bytes | None = None) -> Path:
    """Write a UTF-8 .sav with ASCII placeholders, then splice raw bytes into it."""
    out = tmp_path / "enc.sav"
    df = pl.DataFrame({"x": ["pre VALUE post", "plain"], "y": [1.0, 2.0]})
    write_sav(df, out, var_labels={"x": "LABEL here"})
    raw = out.read_bytes()
    for old, new in [(b"VALUE", value), (b"LABEL", label)]:
        if new is None:
            continue
        assert raw.count(old) == 1 and len(old) == len(new)
        raw = raw.replace(old, new)
    out.write_bytes(raw)
    return out


def _label(meta: dict) -> str:
    return next(v["label"] for v in meta["vars"] if v["name"] == "x")


def test_sav_invalid_utf8_raises_and_names_the_option(tmp_path: Path):
    path = _sav_with_bytes(tmp_path, CAFE_CP1252 + b" ")
    with pytest.raises(RuntimeError, match=r"invalid byte sequence.*\(rc=17\).*utf8-lossy"):
        read_sav(path)


def test_sav_explicit_encoding_decodes_the_bytes(tmp_path: Path):
    path = _sav_with_bytes(tmp_path, CAFE_CP1252 + b" ", label=CAFE_CP1252 + b" ")
    df, meta = read_sav(path, encoding="windows-1252")
    assert df["x"][0] == "pre café  post"
    assert _label(meta) == "café  here"
    assert meta["had_invalid_utf8"] is False


def test_sav_utf8_lossy_replaces_invalid_bytes(tmp_path: Path):
    path = _sav_with_bytes(tmp_path, CAFE_CP1252 + b" ", label=CAFE_CP1252 + b" ")
    df, meta = read_sav(path, encoding="utf8-lossy")
    assert df["x"].to_list() == ["pre caf�  post", "plain"]
    assert _label(meta) == "caf�  here"
    assert meta["had_invalid_utf8"] is True


def test_sav_valid_utf8_is_unchanged_under_lossy(tmp_path: Path):
    path = _sav_with_bytes(tmp_path, "b\U0001f3cd".encode())
    for enc in (None, "utf8-lossy"):
        df, meta = read_sav(path, encoding=enc)
        assert df["x"][0] == "pre b\U0001f3cd post"
        assert meta["had_invalid_utf8"] is False


def test_sav_unknown_encoding_raises(tmp_path: Path):
    path = _sav_with_bytes(tmp_path, b"VALUE")
    with pytest.raises(RuntimeError, match=r"unsupported character set \(rc=7\)"):
        read_sav(path, encoding="no-such-codec")


# No sas7bdat writer exists, so SAS is checked on the shipped fixtures: lossy
# mode must round-trip clean data, labels included, and the error path must
# carry ReadStat's text.
def test_sas_utf8_lossy_round_trips_clean_data():
    data = str(HERE / "data/sas/hadley.sas7bdat")
    catalog = str(HERE / "data/sas/formats.sas7bcat")
    strict = read_sas(data, catalog_path=catalog)
    lossy = read_sas(data, catalog_path=catalog, encoding="utf8-lossy")
    assert lossy[0].equals(strict[0])
    # value_labels is emitted in hash order; compare as a mapping
    by_set = lambda meta: {vl["set_name"]: vl["mapping"] for vl in meta["value_labels"]}  # noqa: E731
    assert by_set(lossy[1]) == by_set(strict[1])
    assert lossy[1]["had_invalid_utf8"] is False


def test_sas_unknown_encoding_raises():
    with pytest.raises(RuntimeError, match=r"unsupported character set \(rc=7\)"):
        read_sas(str(HERE / "data/sas/hadley.sas7bdat"), encoding="no-such-codec")
