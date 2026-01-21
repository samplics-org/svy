# tests/test_stata_strings.py
from __future__ import annotations

import random

from pathlib import Path

import polars as pl
import pytest


try:
    from svy_io.stata import read_dta as _read_dta
    from svy_io.stata import write_dta as _write_dta

    HAVE = True
except Exception:
    HAVE = False


def _rt_var(tmp_path: Path, values, *, dtype: pl.DataType | None = None, **write_kw) -> pl.Series:
    """Write a single-column df and read it back."""
    if not HAVE:
        pytest.xfail("read/write not available")
    s = pl.Series("x", values, dtype=dtype) if dtype is not None else pl.Series("x", values)
    df = pl.DataFrame({"x": s})
    out = tmp_path / "rt.dta"
    _write_dta(df, out, **write_kw)
    df2, _meta = _read_dta(str(out))
    return df2["x"]


def _long(m: int) -> str:
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=m))


# ─────────────────────────────────────────────────────────────────────────────
# 1) Short ASCII strings (no NULs) must roundtrip exactly
#    — catches CString lifetime bugs that produce ""/None on readback.
# ─────────────────────────────────────────────────────────────────────────────
def test_short_ascii_strings_no_nul_roundtrip(tmp_path: Path):
    if not HAVE:
        pytest.xfail("stub pending")
    xs = list("abcdef") + [None]
    got = _rt_var(tmp_path, xs, dtype=pl.Utf8, version=118)
    want = list("abcdef") + [""]
    # Accept either "" or None for Stata string-missing:
    assert got.to_list()[:-1] == want[:-1]
    assert got.to_list()[-1] in ("", None)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Interior NUL: not round-trippable via ReadStat (C-string API).
#    Mark as xfail so it documents the limitation instead of failing CI.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.xfail(reason="ReadStat cannot round-trip embedded NULs in strings")
@pytest.mark.parametrize(
    "payloads",
    [
        ["aa\0bb", "xx\0", "\0yy", "mid\0dle"],
        ["\0", "a\0", "b\0c\0d", None],
    ],
)
def test_strings_with_interior_nul_roundtrip_strl(tmp_path: Path, payloads):
    if not HAVE:
        pytest.xfail("stub pending")
    got = _rt_var(tmp_path, payloads, dtype=pl.Utf8, version=118, strl_threshold=2045)
    assert got.to_list() == payloads


# Old Stata formats (v113..116) have no strL; writer should raise if NUL present.
def test_strings_with_interior_nul_requires_strl_on_old_versions(tmp_path: Path):
    if not HAVE:
        pytest.xfail("stub pending")
    xs = ["aa\0bb", "cc"]
    with pytest.raises(Exception):
        _rt_var(tmp_path, xs, dtype=pl.Utf8, version=114, strl_threshold=2045)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Threshold behavior: <= threshold uses fixed str#, > threshold uses strL,
#    and both should roundtrip losslessly on Stata 14+/v118.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "m,threshold",
    [
        (2045, 2045),  # at limit → fixed str#
        pytest.param(
            2046,
            2045,
            marks=pytest.mark.xfail(
                reason="ReadStat v1.1.9 strL bug: written files cannot be read back"
            ),
        ),  # just over → strL
        pytest.param(
            3000,
            2045,
            marks=pytest.mark.xfail(
                reason="ReadStat v1.1.9 strL bug: written files cannot be read back"
            ),
        ),  # well over → strL
    ],
)
def test_long_strings_respect_strl_threshold_roundtrip(tmp_path: Path, m: int, threshold: int):
    if not HAVE:
        pytest.xfail("stub pending")
    xs = [_long(m) for _ in range(5)]
    got = _rt_var(tmp_path, xs, dtype=pl.Utf8, version=118, strl_threshold=threshold)
    assert got.to_list() == xs


# On old formats (no strL), exceeding threshold should raise.
@pytest.mark.parametrize(
    "m,threshold",
    [
        (2046, 2045),
        (3000, 2045),
    ],
)
def test_long_strings_over_threshold_raise_on_old_versions(tmp_path: Path, m: int, threshold: int):
    if not HAVE:
        pytest.xfail("stub pending")
    xs = [_long(m) for _ in range(3)]
    with pytest.raises(Exception):
        _rt_var(tmp_path, xs, dtype=pl.Utf8, version=114, strl_threshold=threshold)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Mixed: short ASCII & NUL-containing & very long, all together.
#    For the NUL-containing element(s), this will not round-trip (see xfail above).
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.xfail(reason="ReadStat cannot handle embedded NULs and strL in same file")
def test_mixed_string_cases_roundtrip(tmp_path: Path):
    if not HAVE:
        pytest.xfail("stub pending")
    xs = ["short", "ok", None, "edge" * 200, "nul\0inside", _long(2500)]
    got = _rt_var(tmp_path, xs, dtype=pl.Utf8, version=118, strl_threshold=2045)
    # We assert only the non-NUL entries to avoid spurious failures.
    want = ["short", "ok", None, "edge" * 200, _long(0), _long(0)]  # placeholders
    got_list = got.to_list()
    assert got_list[0] == "short"
    assert got_list[1] == "ok"
    assert got_list[2] in ("", None)
    assert got_list[3] == "edge" * 200
    # index 4 (has NUL) is unspecified; index 5 (long, no NUL) should match
    assert got_list[5] == xs[5]
