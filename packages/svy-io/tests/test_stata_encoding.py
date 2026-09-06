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


# ---- detection on formats <= 117 (no declared encoding) ----


def test_stata13_windows1252_bytes_fall_back_to_cp1252(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 117, CAFE_CP1252 + b" ", label=CAFE_CP1252 + b" ")
    df, meta = read_dta(path)
    assert df["x"][0] == "pre café  post"
    assert _label(meta) == "café  here"
    assert meta["encoding"] == "windows-1252"
    assert meta["had_invalid_utf8"] is False


def test_stata13_utf8_text_is_detected(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 117, b"b" + MOTORBIKE_UTF8, label=b"L" + MOTORBIKE_UTF8)
    df, meta = read_dta(path)
    assert df["x"].to_list() == ["pre b\U0001f3cd post", "plain"]
    assert _label(meta) == "L\U0001f3cd here"
    assert meta["encoding"] == "utf-8"
    assert meta["had_invalid_utf8"] is False


def test_stata13_ascii_is_unaffected(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 117, b"VALUE")
    df, meta = read_dta(path)
    assert df["x"].to_list() == ["pre VALUE post", "plain"]
    assert _label(meta) == "LABEL here"
    assert meta["encoding"] == "utf-8"
    assert meta["had_invalid_utf8"] is False


@pytest.mark.parametrize("version", [113, 114, 115])
def test_detection_covers_older_binary_headers(tmp_path: Path, version: int):
    path = _dta_with_bytes(tmp_path, version, b"b" + MOTORBIKE_UTF8)
    df, meta = read_dta(path)
    assert df["x"][0] == "pre b\U0001f3cd post"
    assert meta["encoding"] == "utf-8"

    path = _dta_with_bytes(tmp_path, version, CAFE_CP1252 + b" ")
    df, meta = read_dta(path)
    assert df["x"][0] == "pre café  post"
    assert meta["encoding"] == "windows-1252"


def test_neither_utf8_nor_cp1252_raises_with_hint(tmp_path: Path):
    # E9 8D: invalid UTF-8 (8D must be followed by a continuation byte), and
    # 8D is undefined in Windows-1252.
    path = _dta_with_bytes(tmp_path, 117, b"\xe9\x8d   ")
    with pytest.raises(RuntimeError, match=r"\(rc=17\).*encoding=.*utf8-lossy"):
        read_dta(path)


# ---- explicit encoding skips detection ----


@pytest.mark.parametrize("enc", ["windows-1252", "latin1"])
def test_explicit_code_page_is_used_and_reported(tmp_path: Path, enc: str):
    path = _dta_with_bytes(tmp_path, 117, CAFE_CP1252 + b" ")
    df, meta = read_dta(path, encoding=enc)
    assert df["x"][0] == "pre café  post"
    assert meta["encoding"] == enc


def test_explicit_utf8_is_strict(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 117, CAFE_CP1252 + b" ")
    with pytest.raises(RuntimeError, match=r"\(rc=17\)"):
        read_dta(path, encoding="utf-8")


@pytest.mark.parametrize("enc", ["utf-8", "utf8-lossy", "UTF8-LOSSY"])
def test_stata13_with_utf8_text_reads_with_encoding(tmp_path: Path, enc: str):
    path = _dta_with_bytes(tmp_path, 117, b"b" + MOTORBIKE_UTF8, label=b"L" + MOTORBIKE_UTF8)
    df, meta = read_dta(path, encoding=enc)
    assert df["x"].to_list() == ["pre b\U0001f3cd post", "plain"]
    assert _label(meta) == "L\U0001f3cd here"
    assert meta["encoding"] == "utf-8"
    assert meta["had_invalid_utf8"] is False


def test_utf8_lossy_replaces_invalid_bytes(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 117, CAFE_CP1252 + b" ", label=CAFE_CP1252 + b" ")
    df, meta = read_dta(path, encoding="utf8-lossy")
    assert df["x"][0] == "pre caf�  post"
    assert _label(meta) == "caf�  here"
    assert meta["encoding"] == "utf-8"
    assert meta["had_invalid_utf8"] is True


# ---- format 118+ is UTF-8 by specification ----


def test_stata14_invalid_utf8_is_lossy_by_default(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 118, CAFE_CP1252 + b" ")
    df, meta = read_dta(path)
    assert df["x"][0] == "pre caf�  post"
    assert meta["encoding"] == "utf-8"
    assert meta["had_invalid_utf8"] is True

    with pytest.raises(RuntimeError, match=r"\(rc=17\)"):
        read_dta(path, encoding="utf-8")


def test_unknown_encoding_raises(tmp_path: Path):
    path = _dta_with_bytes(tmp_path, 117, b"VALUE")
    with pytest.raises(RuntimeError, match="unsupported character set"):
        read_dta(path, encoding="no-such-codec")
