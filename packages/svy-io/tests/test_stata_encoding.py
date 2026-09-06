# tests/test_stata_encoding.py
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from svy_io.stata import read_dta, write_dta


CAFE_CP1252 = b"caf\xe9"  # "café" in Windows-1252, invalid as UTF-8
MOTORBIKE_UTF8 = "\U0001f3cd".encode()  # F0 9F 8F 8D: 8D and 8F are undefined in CP1252


def _dta_with_bytes(
    tmp_path: Path, version: int, value: bytes, label: bytes | None = None
) -> Path:
    """Write a .dta with ASCII placeholders, then splice raw bytes into it."""
    out = tmp_path / f"enc{version}.dta"
    df = pl.DataFrame({"x": ["pre VALUE post", "plain"], "y": [1.0, 2.0]})
    write_dta(df, out, version=version, var_labels={"x": "LABEL here"})
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


def test_stata13_decodes_windows1252_by_default(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 117, b"caf\xe9\x20", label=b"caf\xe9\x20")
    for enc in (None, "windows-1252", "latin1"):
        df, meta = read_dta(path, encoding=enc)
        assert df["x"][0] == "pre café  post"
        assert _label(meta) == "café  here"
        assert meta["had_invalid_utf8"] is False


def test_stata13_with_utf8_text_raises_and_names_the_option(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 117, b"b" + MOTORBIKE_UTF8)
    with pytest.raises(RuntimeError, match=r"\(rc=17\).*encoding='utf-8'.*utf8-lossy"):
        read_dta(path)


@pytest.mark.parametrize("enc", ["utf-8", "utf8-lossy", "UTF8-LOSSY"])
def test_stata13_with_utf8_text_reads_with_encoding(tmp_path: Path, enc: str):
    path = _dta_with_bytes(tmp_path, 117, b"b" + MOTORBIKE_UTF8, label=b"L" + MOTORBIKE_UTF8)
    df, meta = read_dta(path, encoding=enc)
    assert df["x"].to_list() == ["pre b\U0001f3cd post", "plain"]
    assert _label(meta) == "L\U0001f3cd here"
    assert meta["had_invalid_utf8"] is False


def test_strict_utf8_rejects_invalid_bytes_and_lossy_replaces_them(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 117, CAFE_CP1252 + b" ", label=CAFE_CP1252 + b" ")
    with pytest.raises(RuntimeError, match=r"\(rc=17\)"):
        read_dta(path, encoding="utf-8")

    df, meta = read_dta(path, encoding="utf8-lossy")
    assert df["x"][0] == "pre caf�  post"
    assert _label(meta) == "caf�  here"
    assert meta["had_invalid_utf8"] is True


def test_stata14_invalid_utf8_is_lossy_by_default(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 118, CAFE_CP1252 + b" ")
    df, meta = read_dta(path)
    assert df["x"][0] == "pre caf�  post"
    assert meta["had_invalid_utf8"] is True

    with pytest.raises(RuntimeError, match=r"\(rc=17\)"):
        read_dta(path, encoding="utf-8")


def test_unknown_encoding_raises(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 117, b"VALUE")
    with pytest.raises(RuntimeError, match="unsupported character set"):
        read_dta(path, encoding="no-such-codec")
