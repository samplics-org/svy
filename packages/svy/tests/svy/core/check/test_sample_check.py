"""Sample.check(): the data against its design, one section per declared part."""

from __future__ import annotations

import msgspec
import polars as pl

import svy

from svy.core.check import SampleCheck


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "reg": ["N", "N", "S", "S", "S", "W"],
            "psu": [1, 2, 1, 2, 2, 1],
            "id": [1, 2, 3, 4, 5, 6],
            "w": [1.0, 2.0, 0.0, 1.5, 3.0, 4.0],
        }
    )


def test_every_declared_part_has_a_section():
    s = svy.Sample(_frame(), svy.Design(stratum="reg", psu="psu", wgt="w", case_id="id"))
    r = s.check()
    assert (r.weights.n, r.weights.n_zero, r.weights.n_positive) == (6, 1, 5)
    assert (
        r.weights.deff == svy.Sample(_frame().filter(pl.col("w") > 0), svy.Design(wgt="w")).deff_w
    )
    assert r.case_id.columns == ["id"] and r.case_id.n_duplicated == 0
    assert r.nesting.n_strata == 3 and r.nesting.n_psus_across_strata == 2
    assert (r.singletons.n_singletons, r.singletons.n_unhandled) == (1, 1)
    assert r.singletons.examples == ["W"]


def test_undeclared_parts_are_none():
    r = svy.Sample(_frame()).check()
    assert r == SampleCheck()
    assert repr(r) == "Sample check: nothing declared to check"
    r = svy.Sample(_frame(), svy.Design(wgt="w")).check()
    assert r.weights is not None
    assert (r.case_id, r.nesting, r.singletons) == (None, None, None)


def test_case_id_on_a_panel_is_unique_within_wave():
    df = pl.DataFrame({"id": [1, 2, 1, 2], "wave": [1, 1, 2, 2], "w": [1.0] * 4})
    r = svy.Sample(df, svy.Design(case_id="id", wave="wave", wgt="w")).check()
    assert r.case_id.columns == ["id", "wave"]
    assert r.case_id.n_duplicated == 0


def test_case_id_on_several_columns():
    df = pl.DataFrame(
        {"clu": [1, 1, 2, 2], "hh": [1, 2, 1, 1], "ln": [1, 1, 1, 2], "w": [1.0] * 4}
    )
    r = svy.Sample(df, svy.Design(case_id=["clu", "hh", "ln"], wgt="w")).check()
    assert r.case_id.columns == ["clu", "hh", "ln"]
    assert r.case_id.n_duplicated == 0


def test_singletons_handled_by_the_rule():
    s = svy.Sample(_frame(), svy.Design(stratum="reg", psu="psu", singleton="center"))
    r = s.check()
    assert (r.singletons.n_singletons, r.singletons.n_unhandled) == (1, 0)


def test_limit_caps_examples():
    s = svy.Sample(_frame(), svy.Design(stratum="reg", psu="psu"))
    assert len(s.check(limit=1).nesting.examples) == 1


def test_report_serializes_and_prints():
    s = svy.Sample(_frame(), svy.Design(stratum="reg", psu="psu", wgt="w", case_id="id"))
    r = s.check()
    assert msgspec.json.decode(msgspec.json.encode(r), type=SampleCheck) == r
    text = repr(r)
    for title in (
        "Weight check: w",
        "Key check: id",
        "Nesting check: psu in reg",
        "Singleton check",
    ):
        assert title in text
    assert "Weight check" in str(r)
