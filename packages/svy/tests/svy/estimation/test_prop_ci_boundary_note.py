# tests/svy/estimation/test_prop_ci_boundary_note.py
"""
A proportion at 0 or 1 has no logit/beta/wilson interval. That is a finding about
the returned estimate, handled like the domain-singleton one: kept in
``sample.warnings`` at INFO, on the estimate's ``findings`` at WARNING, and
printed as one note under its table, never raised as a Python warning.
korn-graubard has a one-sided interval there, so it is no finding.

Data: 2 strata x 4 PSUs x 10 persons; female == 1 is persons 1-5 of every PSU.
``y_zero`` and ``y_one`` are constant among women; ``y3`` lacks level "c" there
and ``y_many`` is "a" there, never among men.
"""

from __future__ import annotations

import math
import re
import warnings

import polars as pl
import pytest

from svy import Design, Sample, Singleton, col
from svy.core.warnings import Severity, WarnCode
from svy.estimation.base import prop_ci_boundary_note


NAN_AT_BOUNDARY = ["logit", "beta", "wilson"]
NOTE_2 = "note: CI undefined at p = 0 or 1 for 2 rows (ci_method='{}')"


def _frame() -> pl.DataFrame:
    rows = []
    for h in (1, 2):
        for k in range(4):
            for j in range(10):
                fem = j < 5
                rows.append(
                    {
                        "stratum": h,
                        "psu": f"{h}-{k + 1}",
                        "female": int(fem),
                        "wgt": 10.0 + k if h == 1 else 20.0 - k,
                        "y_zero": 0 if fem else int(j < 8 - k),
                        "y_one": 1 if fem else int(j < 6 + k % 2),
                        "y_mid": int(j < 1 + (k + h) % 3) if fem else int(j < 7 - h),
                        "y3": ("a" if j < 2 + k % 2 else "b") if fem else "abc"[(j + k) % 3],
                        "y_many": "a" if fem else f"l{(j + 5 * k) % 12}",
                    }
                )
    return pl.DataFrame(rows)


def _sample() -> Sample:
    return Sample(_frame(), Design(stratum="stratum", psu="psu", wgt="wgt"))


def _no_warning(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn()


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _boundary(sample: Sample) -> list:
    return sample.warnings.list(code=WarnCode.PROP_CI_BOUNDARY)


DOMAINS = {
    "by": dict(by="female"),
    "where": dict(where=col("female") == 1),
}


def _estimate(method, **kwargs):
    s = _sample()
    if method == "replication":
        s = s.weighting.create_jk_wgts()
    kwargs = dict(kwargs, method=method)
    return s, kwargs


# =============================================================================
# 1. The finding is on the estimate and the sample, and nothing is raised
# =============================================================================


@pytest.mark.parametrize("method", ["taylor", "replication"])
@pytest.mark.parametrize("ci_method", NAN_AT_BOUNDARY)
@pytest.mark.parametrize("domain", list(DOMAINS))
def test_prop_records_note_without_warning(method, ci_method, domain):
    s, kw = _estimate(method, **DOMAINS[domain])
    r = _no_warning(lambda: s.estimation.prop("y_zero", ci_method=ci_method, **kw))

    nan_rows = [e for e in r.estimates if math.isnan(e.lci)]
    assert len(nan_rows) == 2
    (f,) = r.findings
    assert f.code == WarnCode.PROP_CI_BOUNDARY
    assert f.level == Severity.WARNING
    assert f.where == "estimation.prop"
    assert f.param == "ci_method" and f.got == ci_method
    assert f.var == "y_zero"
    assert prop_ci_boundary_note([f]) == NOTE_2.format(ci_method)
    assert f.extra["n_rows"] == 2
    assert "korn-graubard" in f.hint
    (kept,) = _boundary(s)
    assert kept.level == Severity.INFO
    assert kept.detail == f.detail


@pytest.mark.parametrize("method", ["taylor", "replication"])
def test_mean_as_factor_records_note_without_warning(method):
    s, kw = _estimate(method, by="female")
    r = _no_warning(lambda: s.estimation.mean("y_one", as_factor=True, **kw))
    (f,) = r.findings
    assert f.where == "estimation.mean"
    assert f.got == "logit"
    assert prop_ci_boundary_note([f]) == NOTE_2.format("logit")
    # mean() has no ci_method, so the hint names prop().
    assert "prop(..., ci_method='korn-graubard')" in f.hint
    assert len(_boundary(s)) == 1


@pytest.mark.parametrize("method", ["taylor", "replication"])
def test_ungrouped_prop_on_a_constant_column(method):
    s = Sample(
        _frame().filter(pl.col("female") == 1),
        Design(stratum="stratum", psu="psu", wgt="wgt"),
    )
    if method == "replication":
        s = s.weighting.create_jk_wgts()
    r = _no_warning(lambda: s.estimation.prop("y_zero", method=method))
    (f,) = r.findings
    # Only the level present is a row.
    assert f.extra["cells"] == [{"by": None, "level": 0, "p": 1}]
    assert (
        prop_ci_boundary_note([f])
        == "note: CI undefined at p = 0 or 1 for 1 row (ci_method='logit')"
    )


@pytest.mark.parametrize("method", ["taylor", "replication"])
def test_korn_graubard_is_no_finding(method):
    s, kw = _estimate(method, by="female")
    r = _no_warning(lambda: s.estimation.prop("y_zero", ci_method="korn-graubard", **kw))
    assert r.findings == []
    assert _boundary(s) == []
    assert "note:" not in str(r)


def test_rows_without_df_are_no_finding():
    # One PSU per female x psu domain: df = 0 makes every interval NaN, which is
    # not about p being 0 or 1.
    s = _sample()
    r = s.estimation.prop("y_zero", by=["female", "psu"])
    assert all(e.df == 0 for e in r.estimates)
    assert [str(f.code) for f in r.findings] == ["DOMAIN_SINGLETON_PSU"]
    assert _boundary(s) == []


def test_interior_proportions_are_no_finding():
    s = _sample()
    r = s.estimation.prop("y_mid", by="female")
    assert r.findings == []
    assert "note:" not in str(r) and "note:" not in r.__plain_str__()


def test_other_estimates_carry_no_findings():
    s = _sample()
    assert s.estimation.mean("y_zero", by="female").findings == []
    assert s.estimation.total("y_zero", by="female", as_factor=True).findings == []


# =============================================================================
# 2. What the finding says
# =============================================================================


def test_cells_use_native_levels_in_row_order():
    s = _sample()
    r = s.estimation.prop("y_zero", by=["female", "stratum"])
    (f,) = r.findings
    assert f.extra["n_rows"] == 4
    assert f.extra["cells"] == [
        {"by": {"female": 1, "stratum": 1}, "level": 0, "p": 1},
        {"by": {"female": 1, "stratum": 1}, "level": 1, "p": 0},
        {"by": {"female": 1, "stratum": 2}, "level": 0, "p": 1},
        {"by": {"female": 1, "stratum": 2}, "level": 1, "p": 0},
    ]
    assert "y_zero=0 in female=1, stratum=1 (p=1)" in f.detail
    assert "__svy_" not in f.detail


def test_one_row_is_singular():
    s = _sample()
    r = s.estimation.prop("y3", by="female")
    (f,) = r.findings
    assert f.extra["cells"] == [{"by": {"female": 1}, "level": "c", "p": 0}]
    assert (
        prop_ci_boundary_note([f])
        == "note: CI undefined at p = 0 or 1 for 1 row (ci_method='logit')"
    )


def test_detail_lists_ten_cells_then_counts():
    s = _sample()
    r = s.estimation.prop("y_many", by="female")
    (f,) = r.findings
    # Men lack "a"; women have only "a".
    assert f.extra["n_rows"] == 14
    assert [(c["by"]["female"], c["level"]) for c in f.extra["cells"]] == [(0, "a"), (1, "a")] + [
        (1, f"l{i}") for i in range(12)
    ]
    assert f.detail.endswith("y_many=l7 in female=1 (p=0); and 4 more.")


# =============================================================================
# 3. Every result shows its own note
# =============================================================================


def test_repeat_on_unchanged_sample_keeps_the_note():
    s = _sample()
    a = s.estimation.prop("y_zero", by="female")
    b = s.estimation.prop("y_zero", by="female")
    assert a.findings and b.findings
    assert b.findings[0].detail == a.findings[0].detail
    # The store records it once per sample state.
    assert len(_boundary(s)) == 1


def test_to_polars_is_unchanged_by_the_finding():
    s = _sample()
    r = s.estimation.prop("y_zero", by="female")
    with_finding = r.to_polars()
    r.findings = []
    assert with_finding.equals(r.to_polars())
    fresh = s.estimation.prop("y_zero", by="female", ci_method="korn-graubard")
    assert fresh.to_polars().columns == with_finding.columns


# =============================================================================
# 4. Printing
# =============================================================================


def test_note_prints_under_the_table():
    s = _sample()
    r = s.estimation.prop("y_zero", by="female", ci_method="beta")
    note = NOTE_2.format("beta")
    for text in (_plain(str(r)), r.__plain_str__()):
        assert text.count("note:") == 1
        assert note in text
        assert text.index("uci") < text.index("note:")
    assert r.__plain_str__().endswith("\n" + note)


def test_list_prints_one_note_per_code():
    s = _sample()
    r = s.estimation.prop(["y_zero", "y_mid"], where=col("female") == 1)
    assert [len(m.findings) for m in r] == [1, 0]
    for text in (_plain(str(r)), r.__plain_str__()):
        assert text.count("note:") == 1
        assert NOTE_2.format("logit") in text


def test_list_note_splits_counts_by_variable():
    s = _sample()
    r = s.estimation.prop(["y_zero", "y_one", "y3"], by="female")
    assert [m.findings[0].extra["n_rows"] for m in r] == [2, 2, 1]
    expected = (
        "note: CI undefined at p = 0 or 1 for 5 rows "
        "(y_zero: 2, y_one: 2, y3: 1; ci_method='logit')"
    )
    for text in (_plain(str(r)), r.__plain_str__()):
        assert text.count("note:") == 1
        assert expected in text


# =============================================================================
# 5. Next to the domain-singleton note
# =============================================================================


def test_prints_with_the_domain_singleton_note():
    # Men of stratum 1 sit in one PSU (1-1) of the four: a domain singleton.
    f = _frame().filter(
        ~((pl.col("stratum") == 1) & (pl.col("female") == 0)) | (pl.col("psu") == "1-1")
    )
    rule = Singleton("center", on_domain_singletons="warn")
    s = Sample(f, Design(stratum="stratum", psu="psu", wgt="wgt", singleton=rule))
    r = _no_warning(lambda: s.estimation.prop("y_zero", by="female"))
    assert [str(x.code) for x in r.findings] == ["DOMAIN_SINGLETON_PSU", "PROP_CI_BOUNDARY"]
    lines = [ln for ln in r.__plain_str__().splitlines() if ln.startswith("note:")]
    assert len(lines) == 2
    assert lines[1] == NOTE_2.format("logit")
