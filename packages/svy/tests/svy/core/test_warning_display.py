# tests/svy/core/test_warning_display.py
"""Findings print like errors, with their own marker; a store prints its findings."""

import io

import polars as pl
import pytest

from rich.console import Console

import svy

from svy.core.warnings import Severity, SvyWarning, WarningStore
from svy.errors.base_errors import SvyError
from svy.selection import Stage


def _rich(obj) -> str:
    buf = io.StringIO()
    Console(file=buf, width=100, color_system=None).print(obj)
    return buf.getvalue()


def _finding(level: Severity, code: str = "SOME_CODE") -> SvyWarning:
    return SvyWarning(code=code, title="A title", detail="What happened.", level=level, param="n")


@pytest.mark.parametrize(
    "level, mark, rich_mark",
    [(Severity.INFO, "ℹ️", "ℹ"), (Severity.WARNING, "⚠️", "▲"), (Severity.ERROR, "❌", "✗")],
)
def test_finding_prints_like_an_error_with_its_marker(level, mark, rich_mark):
    w = _finding(level)
    text = str(w)
    assert f"{mark} A title [SOME_CODE]" in text
    assert "What happened." in text and "- param: n" in text
    assert f"{rich_mark} A title [SOME_CODE]" in _rich(w)


def test_markdown_and_html_carry_the_marker():
    w = _finding(Severity.INFO)
    assert w.markdown().startswith("**ℹ️ A title**")
    assert "ℹ️" in w.html()


def test_errors_keep_their_marker():
    e = SvyError(title="Bad", detail="d", code="BAD")
    assert "❌ Bad [BAD]" in str(e)
    assert "✗ Bad [BAD]" in _rich(e)


def test_empty_store():
    store = WarningStore()
    assert str(store) == repr(store) == "No findings."


def test_store_lists_its_findings_with_counts():
    store = WarningStore()
    store.add(_finding(Severity.WARNING, "FIRST"))
    store.add(_finding(Severity.INFO, "SECOND"))
    shown = str(store)
    assert repr(store) == shown
    assert "Findings (1 warning, 1 info)" in shown
    assert "FIRST" in shown and "SECOND" in shown
    plain = store.__plain_str__()
    assert plain.startswith("Findings (1 warning, 1 info)")
    assert "WARNING" in plain and "SECOND" in plain


_LONG = (
    "Stage 2 (ea) asks for more units than 2 parent(s) hold, so all their units are taken, "
    "with probability 1: geo1=geo_02, geo2=geo_02_02; geo1=geo_03, geo2=geo_03_05"
)


def _widths(text: str) -> list[int]:
    return [len(line.rstrip()) for line in text.splitlines()]


def _long_store() -> WarningStore:
    store = WarningStore()
    store.add(SvyWarning(code="LONG", title="t", detail=_LONG, level=Severity.INFO))
    store.add(_finding(Severity.WARNING, "SHORT"))
    return store


def test_store_stacks_labelled_blocks():
    plain = _long_store().__plain_str__()
    assert "  Level   INFO\n  Code    LONG\n  Detail  Stage 2" in plain
    assert "  Level   WARNING\n  Code    SHORT\n  Detail  What happened." in plain
    assert max(_widths(plain)) <= 88
    # Continuation lines sit under the detail column, not under the labels.
    detail = plain.split("Detail  Stage 2")[1].split("\n\n")[0]
    assert all(line.startswith(" " * 10) for line in detail.splitlines()[1:])


def test_store_prints_within_text_width():
    store = _long_store()
    shown = _rich(store)
    assert max(_widths(shown)) <= 88
    assert "Level" in shown and "Code" in shown and "Detail" in shown
    assert "geo2=geo_03_05" in shown


def test_error_panel_capped_at_text_width():
    e = SvyError(title="Bad", detail=_LONG, code="BAD", hint=_LONG)
    shown = _rich(e)
    assert max(_widths(shown)) <= 88
    assert "geo2=geo_03_05" in shown


def test_print_width_env_overrides_text_width(monkeypatch):
    monkeypatch.setenv("SVY_PRINT_WIDTH", "60")
    assert max(_widths(_rich(SvyError(title="Bad", detail=_LONG, code="BAD")))) <= 60
    assert max(_widths(_long_store().__plain_str__())) <= 60


def test_error_message_is_not_wrapped():
    assert _LONG in str(SvyError(title="Bad", detail=_LONG, code="BAD"))


def test_sample_warnings_print():
    frame = pl.DataFrame(
        {
            "region": ["a"] * 6 + ["b"] * 4,
            "district": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2],
            "ea": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    s = svy.Sample(frame, svy.Design(stratum="region")).sampling.multistage(
        [Stage("district", n=1), Stage("ea", n=5)], rstate=1
    )
    (w,) = list(s.warnings)
    assert w.code == "STAGE_TAKES_ALL"
    assert "Stage 2 (ea)" in w.detail
    assert "region=" in w.detail and "district=" in w.detail
    assert all(set(p) == {"region", "district"} for p in w.got)
    assert "STAGE_TAKES_ALL" in str(s.warnings)


def test_unstratified_parent_is_named():
    frame = pl.DataFrame({"district": [1, 1, 2, 2], "ea": [1, 2, 1, 2]})
    s = svy.Sample(frame, svy.Design()).sampling.multistage([Stage("district", n=5)], rstate=1)
    (w,) = list(s.warnings)
    assert "the whole frame" in w.detail
