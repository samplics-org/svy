# tests/svy/errors/test_result_findings.py
"""A result keeps the warnings its call raised, prints them, and saves them."""

from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import Code, DesignWarning, ModelWarning, Severity, SvyWarning
from svy.errors._findings import collecting_findings, emit_finding, merge_findings
from svy.serialize import from_json, to_json


N = 60


def _reps_without_design() -> svy.Sample:
    """Replicate weights but no stratum or psu: every Taylor call warns TAYLOR_WITHOUT_DESIGN."""
    rng = np.random.default_rng(20261027)
    df = pl.DataFrame(
        {
            "y": rng.normal(10, 2, N),
            "x": rng.uniform(1, 2, N),
            "g": np.tile(["a", "b"], N // 2),
            "c": np.tile(["p", "q", "r"], N // 3),
            "w": np.ones(N),
            "r1": rng.uniform(0.5, 1.5, N),
            "r2": rng.uniform(0.5, 1.5, N),
        }
    )
    return svy.Sample(df, svy.Design(wgt="w", rep_wgts=svy.BrrWgts(prefix="r", n_reps=2)))


T = {"method": "taylor"}
CALLS = {
    "mean": lambda s: s.estimation.mean("y", **T),
    "total": lambda s: s.estimation.total("y", **T),
    "prop": lambda s: s.estimation.prop("g", **T),
    "ratio": lambda s: s.estimation.ratio("y", "x", **T),
    "median": lambda s: s.estimation.median("y", **T),
    "quantile": lambda s: s.estimation.quantile("y", p=0.25, **T),
    "corr": lambda s: s.estimation.corr(("y", "x"), **T),
    "cov": lambda s: s.estimation.cov(("y", "x"), **T),
    "mean of two": lambda s: s.estimation.mean(["y", "x"], **T),
    "tabulate": lambda s: s.categorical.tabulate("c", "g", **T),
    "ttest": lambda s: s.categorical.ttest("y", group="g", **T),
    "ttest by": lambda s: s.categorical.ttest("y", mean_h0=10, by="g", **T),
    "ranktest": lambda s: s.categorical.ranktest("y", group="g", **T),
    "glm": lambda s: s.glm.fit("y", x=["x"], **T).fitted,
}


def _quiet(call, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SvyWarning)
        return call(*args)


def _codes(result) -> list[str]:
    return [f.code for f in result.findings]


def _members(result) -> list:
    if hasattr(result, "results"):
        return list(result.results)
    if hasattr(result, "_members"):
        return list(result._members())
    return []


# ---- every analysis keeps what its call raised ---------------------------------


@pytest.mark.parametrize("name", list(CALLS))
def test_result_keeps_the_warning_its_call_raised(name):
    s = _reps_without_design()
    with pytest.warns(DesignWarning, match=r"\[TAYLOR_WITHOUT_DESIGN\]"):
        result = CALLS[name](s)
    holders = _members(result) if name == "mean of two" else [result, *_members(result)]
    for held in holders:
        assert _codes(held) == [Code.TAYLOR_WITHOUT_DESIGN], type(held).__name__
        assert held.findings[0] is s.warnings.list(code=Code.TAYLOR_WITHOUT_DESIGN)[0]


@pytest.mark.parametrize("name", list(CALLS))
def test_a_repeat_the_store_suppresses_still_reaches_the_result(name):
    s = _reps_without_design()
    _quiet(CALLS[name], s)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        again = CALLS[name](s)
    assert not [r for r in rec if "TAYLOR_WITHOUT_DESIGN" in str(r.message)]
    assert len(s.warnings.list(code=Code.TAYLOR_WITHOUT_DESIGN)) == 1
    held = _members(again)[0] if name == "mean of two" else again
    assert _codes(held) == [Code.TAYLOR_WITHOUT_DESIGN]


@pytest.mark.parametrize("name", list(CALLS))
def test_result_prints_its_warning_as_a_note(name):
    result = _quiet(CALLS[name], _reps_without_design())
    note = "note: Taylor variance on a design with no stratum or psu [TAYLOR_WITHOUT_DESIGN]"
    assert note in str(result)
    assert note in result.__plain_str__()
    assert str(result).count(note) == 1


def test_a_call_that_raises_nothing_keeps_nothing():
    s = _reps_without_design()
    _quiet(CALLS["mean"], s)
    clean = s.estimation.mean("y", method="replication")
    assert clean.findings == []
    assert "note:" not in str(clean)


@pytest.mark.parametrize(
    "max_iter, code", [(100, Code.GLM_SEPARATION), (1, Code.GLM_NOT_CONVERGED)]
)
def test_glm_keeps_its_own_warnings(max_iter, code):
    rng = np.random.default_rng(20261027)
    n = 120
    g = np.tile(["a", "b", "c"], n // 3)
    x = rng.normal(size=n)
    y = np.where(g == "c", 0, (x + rng.normal(size=n) > 0).astype(np.int64))
    s = svy.Sample(pl.DataFrame({"x": x, "g": g, "y": y, "w": np.ones(n)}), svy.Design(wgt="w"))
    fit = _quiet(
        lambda: s.glm.fit("y", x=["x", svy.Cat("g")], family="binomial", max_iter=max_iter)
    ).fitted
    assert _codes(fit) == [code]
    assert type(fit.findings[0]) is ModelWarning
    assert f"[{code}]" in str(fit)
    assert json.loads(to_json(fit))["findings"][0]["code"] == code


# ---- the collector -------------------------------------------------------------


def _warning(code: str, level: Severity = Severity.WARNING, detail: str = "d") -> SvyWarning:
    return SvyWarning(code=code, title="t", detail=detail, level=level)


def test_collector_keeps_warnings_not_info_and_nests():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SvyWarning)
        with collecting_findings() as outer:
            emit_finding(_warning("A"))
            with collecting_findings() as inner:
                emit_finding(_warning("B"))
                emit_finding(_warning("C", Severity.INFO))
        emit_finding(_warning("D"))
    assert [f.code for f in outer] == ["A", "B"]
    assert [f.code for f in inner] == ["B"]


def test_merge_keeps_order_and_drops_repeats():
    a, b = _warning("A"), _warning("B")
    same_as_a = _warning("A")
    other_detail = _warning("A", detail="other")
    merged = merge_findings([a], [b, same_as_a, other_detail, b])
    assert merged == (a, b, other_detail)
    assert merge_findings([], []) == ()


# ---- saved results -------------------------------------------------------------

SAVED = ["mean", "prop", "tabulate", "ttest", "ttest by", "glm"]


@pytest.mark.parametrize("name", SAVED)
def test_saved_result_keeps_its_warnings(name):
    result = _quiet(CALLS[name], _reps_without_design())
    payload = json.loads(to_json(result))
    (saved,) = payload["findings"]
    assert saved == {
        "code": "TAYLOR_WITHOUT_DESIGN",
        "level": "WARNING",
        "title": "Taylor variance on a design with no stratum or psu",
        "detail": result.findings[0].detail,
        "hint": result.findings[0].hint,
        "var": None,
        "got": saved["got"],
    }
    back = from_json(to_json(result))
    assert [f.code for f in back.findings] == ["TAYLOR_WITHOUT_DESIGN"]


def test_saved_result_leaves_info_out_and_none_when_empty():
    s = _reps_without_design()
    est = _quiet(CALLS["mean"], s)
    est.findings.append(_warning("AUDIT", Severity.INFO))
    assert [f["code"] for f in json.loads(to_json(est))["findings"]] == ["TAYLOR_WITHOUT_DESIGN"]
    clean = s.estimation.mean("y", method="replication")
    assert json.loads(to_json(clean))["findings"] is None


def test_older_payload_without_findings_still_decodes():
    est = _quiet(CALLS["mean"], _reps_without_design())
    payload = json.loads(to_json(est))
    del payload["findings"]
    payload["schema_version"] = "svy-result/0.10"
    assert from_json(json.dumps(payload).encode()).findings is None
