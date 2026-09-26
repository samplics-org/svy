# tests/svy/weighting/test_wgt_findings_recorded.py
"""Weighting findings are recorded in ``sample.warnings`` and raised once as a
``SvyUserWarning``; nothing goes to stdout.

Rows a null cell leaves unadjusted, non-convergence under ``strict=False``,
standardize's partial domains and panel adjust's skipped rule are recorded
once, with structured counts, on the sample the call returns. A call that
fails leaves the caller's sample as it was, inplace or not.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl
import pytest

import svy

from svy import Cat, Design, Sample, SvyUserWarning, TrimConfig, col
from svy.core.warnings import WarnCode
from svy.errors import WeightingError


NULL = WarnCode.CELLS_NULL_UNADJUSTED
MAXIT = WarnCode.MAX_ITER_REACHED


def _codes(s: Sample, code) -> list:
    return s.warnings.list(code=code)


def _one(s: Sample, code):
    found = _codes(s, code)
    assert len(found) == 1, [w.detail for w in found]
    json.dumps(found[0].to_dict())
    return found[0]


@pytest.fixture
def s() -> Sample:
    df = pl.DataFrame(
        {
            "a": ["x", "y", None, "x", "y", None, "x", "y"],
            "b": [1, 2, 1, None, 2, 1, None, 2],
            "g": ["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"],
            "st": ["rr", "nr", "rr", "rr", "nr", "rr", "rr", "rr"],
            "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    return Sample(df, Design(wgt="w"))


NULL_A = np.array([False, False, True, False, False, True, False, False])
NULL_B = np.array([False, False, False, True, False, False, True, False])


# ---------------------------------------------------------------------------
# Null cells: kept unadjusted, recorded once, with counts per column
# ---------------------------------------------------------------------------


CALLS = {
    "poststratify": (
        lambda s, **k: s.weighting.poststratify({"x": 10, "y": 20}, cells="a", **k),
        "ps_wgt",
    ),
    "poststratify_shares": (
        lambda s, **k: s.weighting.poststratify(shares={"x": 1, "y": 1}, cells="a", **k),
        "ps_wgt",
    ),
    "normalize": (lambda s, **k: s.weighting.normalize(cells="a", **k), "norm_wgt"),
    "standardize": (
        lambda s, **k: s.weighting.standardize("a", shares={"x": 1, "y": 3}, by="g", **k),
        "std_wgt",
    ),
    "adjust": (
        lambda s, **k: s.weighting.adjust("st", cells="a", respondents_only=False, **k),
        "nr_wgt",
    ),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_null_cells_recorded_and_left_unadjusted(s, name):
    call, wgt = CALLS[name]
    out = call(s)
    w = _one(out, NULL)
    assert w.got == {"a": 2}
    assert w.param == "cells"
    assert w.extra["n_rows"] == 2 and w.extra["wgt_name"] == wgt
    assert w.where == f"Sample.weighting.{name.split('_')[0]}"
    assert "'a'" in w.detail and repr(wgt) in w.detail
    np.testing.assert_allclose(out.data[wgt].to_numpy()[NULL_A], s.data["w"].to_numpy()[NULL_A])
    # the caller's sample is untouched by the default fork
    assert not _codes(s, NULL)


@pytest.mark.parametrize("name", list(CALLS))
def test_null_cells_recorded_inplace(s, name):
    call, _ = CALLS[name]
    out = call(s, inplace=True)
    assert out is s
    assert _one(s, NULL).got == {"a": 2}


def test_standardize_null_in_by(s):
    df = s.data.with_columns(
        pl.Series("g", ["g1", None, "g1", "g1", "g2", "g2", "g2", None]),
        pl.Series("a", ["x", "y"] * 4),
    )
    out = Sample(df, Design(wgt="w")).weighting.standardize("a", shares={"x": 1, "y": 1}, by="g")
    assert _one(out, NULL).got == {"g": 2}


def test_several_null_columns_at_once(s):
    out = s.weighting.poststratify({("x", 1): 1, ("y", 2): 2, ("x", 2): 0}, cells=["a", "b"])
    w = _one(out, NULL)
    assert w.got == {"a": 2, "b": 2}
    assert w.extra["n_rows"] == 4
    left = NULL_A | NULL_B
    np.testing.assert_allclose(out.data["ps_wgt"].to_numpy()[left], s.data["w"].to_numpy()[left])


def test_null_in_both_columns_counts_the_row_once():
    df = pl.DataFrame({"a": ["x", None, "y"], "b": [1, None, 2], "w": [1.0, 2.0, 3.0]})
    out = Sample(df, Design(wgt="w")).weighting.poststratify(
        {("x", 1): 5, ("y", 2): 5}, cells=["a", "b"]
    )
    w = _one(out, NULL)
    assert w.got == {"a": 1, "b": 1}
    assert w.extra["n_rows"] == 1


def test_all_rows_null_is_an_error():
    df = pl.DataFrame({"a": [None, None], "w": [1.0, 2.0]}, schema={"a": pl.Utf8, "w": pl.Float64})
    with pytest.raises(WeightingError) as ei:
        Sample(df, Design(wgt="w")).weighting.poststratify({"x": 1}, cells="a")
    assert ei.value.code == "NO_ROWS_IN_SCOPE"
    assert ei.value.got == {"a": 2}
    json.dumps(ei.value.to_dict())


def test_nulls_only_outside_where_are_not_a_finding(s):
    out = s.weighting.poststratify({"x": 10, "y": 20}, cells="a", where=col("a").is_not_null())
    assert not _codes(out, NULL)
    out = s.weighting.poststratify({"x": 10, "y": 20}, cells="a", where=col("g") == "g1")
    assert _one(out, NULL).got == {"a": 1}


def test_no_findings_when_nothing_is_wrong(s, capsys):
    clean = Sample(s.data.drop_nulls(), Design(wgt="w"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = clean.weighting.poststratify({"x": 3, "y": 3}, cells="a")
        out = out.weighting.rake(controls={"a": {"x": 3, "y": 3}}, wgt_name="rk")
        out = out.weighting.standardize("a", shares={"x": 1, "y": 1})
        out = out.weighting.normalize(cells="b")
    assert capsys.readouterr().out == ""
    assert not [w for w in out.warnings if str(w.code) in {NULL, MAXIT}]


def test_repeated_calls_and_forks_do_not_duplicate(s):
    a = s.weighting.poststratify({"x": 10, "y": 20}, cells="a")
    b = s.weighting.poststratify({"x": 10, "y": 20}, cells="a")
    assert len(_codes(a, NULL)) == 1 and len(_codes(b, NULL)) == 1
    chained = a.weighting.normalize(cells="a", wgt_name="n2")
    found = _codes(chained, NULL)
    assert [w.extra["wgt_name"] for w in found] == ["ps_wgt", "n2"]
    assert len(_codes(a, NULL)) == 1
    s.weighting.poststratify({"x": 10, "y": 20}, cells="a", inplace=True)
    s.weighting.poststratify({"x": 10, "y": 20}, cells="a", wgt_name="ps2", inplace=True)
    assert [w.extra["wgt_name"] for w in _codes(s, NULL)] == ["ps_wgt", "ps2"]


def test_trim_by_null_domain():
    df = pl.DataFrame({"g": ["a"] * 12 + [None] * 3, "w": [1.0] * 11 + [50.0, 60.0, 1.0, 1.0]})
    s = Sample(df, Design(wgt="w"))
    out = s.weighting.trim(upper=5.0, by="g", min_cell_size=1)
    w = _one(out, NULL)
    assert w.got == {"g": 3} and w.param == "by"
    assert out.data["trim_wgt"].to_list()[-3:] == [60.0, 1.0, 1.0]
    assert max(out.data["trim_wgt"].to_list()[:12]) <= 5.0 + 1e-9


def test_trim_by_numeric_null_domain_no_longer_silent():
    df = pl.DataFrame({"g": [1] * 12 + [None] * 2, "w": [1.0] * 11 + [50.0, 70.0, 1.0]})
    out = Sample(df, Design(wgt="w")).weighting.trim(upper=5.0, by="g", min_cell_size=1)
    assert _one(out, NULL).got == {"g": 2}


def test_calibrate_trimming_by_null():
    df = pl.DataFrame(
        {"c": ["a", "b"] * 10, "t": ["u"] * 18 + [None] * 2, "w": [1.0] * 19 + [5.0]}
    )
    cfg = TrimConfig(upper=Cap_q(), by="t", min_cell_size=1)
    out = Sample(df, Design(wgt="w")).weighting.calibrate(
        controls={Cat("c"): {"a": 12, "b": 12}}, trimming=cfg, strict=False
    )
    w = _one(out, NULL)
    assert w.got == {"t": 2} and w.param == "trimming.by"


def Cap_q():  # noqa: N802
    return svy.Threshold.quantile(0.95)


def test_panel_adjust_records_skipped_rule():
    w1 = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "g": ["A", "A", "B", "B"],
            "w": [1.0, 2.0, 3.0, 4.0],
            "resp": ["rr"] * 4,
            "wave": [1] * 4,
        }
    )
    w2 = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "g": ["A", "A", "B"],
            "w": [1.0, 2.0, 3.0],
            "resp": ["rr", "nr", "rr"],
            "wave": [2] * 3,
        }
    )
    s = Sample(pl.concat([w1, w2]), Design(case_id="id", wave="wave", wgt="w"))
    with pytest.warns(SvyUserWarning, match=r"\[PANEL_SCOPE_NOT_WAVES\]") as rec:
        out = s.weighting.adjust(
            "resp",
            cells="g",
            where=(col("wave") == 2) & (col("id") != 3),
            respondents_only=False,
        )
    assert _svy(rec) and all(r.filename == __file__ for r in _svy(rec))
    w = _one(out, WarnCode.PANEL_SCOPE_NOT_WAVES)
    assert "missing-in-scope rule was skipped" in w.detail
    assert "where=col('wave') == 2" in w.hint
    ok = s.weighting.adjust("resp", cells="g", where=col("wave") == 2, respondents_only=False)
    assert not _codes(ok, WarnCode.PANEL_SCOPE_NOT_WAVES)


def test_panel_adjust_null_cells_count_real_rows_only():
    w1 = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "g": ["A", None, "B", "B"],
            "w": [1.0, 2.0, 3.0, 4.0],
            "resp": ["rr"] * 4,
            "wave": [1] * 4,
        }
    )
    w2 = pl.DataFrame(
        {"id": [1, 3], "g": ["A", "B"], "w": [1.0, 3.0], "resp": ["rr", "rr"], "wave": [2] * 2}
    )
    s = Sample(pl.concat([w1, w2]), Design(case_id="id", wave="wave", wgt="w"))
    out = s.weighting.adjust("resp", cells="g", respondents_only=False)
    assert _one(out, NULL).got == {"g": 1}


# ---------------------------------------------------------------------------
# Standardize partial domains
# ---------------------------------------------------------------------------


def test_standardize_partial_domain_recorded_not_warned():
    df = pl.DataFrame(
        {"a": ["x", "y", "x", "x"], "g": ["g1", "g1", "g2", "g2"], "w": [1.0, 2.0, 3.0, 4.0]}
    )
    with pytest.warns(SvyUserWarning, match=r"\[DOMAIN_LEVELS_PARTIAL\]") as rec:
        out = Sample(df, Design(wgt="w")).weighting.standardize(
            "a", shares={"x": 1, "y": 1}, by="g"
        )
    assert _svy(rec) and all(r.filename == __file__ for r in _svy(rec))
    w = _one(out, WarnCode.DOMAIN_LEVELS_PARTIAL)
    assert w.got == ["g2"]
    assert "do not observe every level" in w.detail


# ---------------------------------------------------------------------------
# Non-convergence: strict raises (sample untouched), strict=False records
# ---------------------------------------------------------------------------


@pytest.fixture
def rk() -> Sample:
    df = pl.DataFrame(
        {
            "age": ["18-34", "35-54", "55+", "18-34", "35-54", "55+", "18-34", "35-54"],
            "region": ["North"] * 4 + ["South"] * 4,
            "w": [10.0] * 8,
        }
    )
    return Sample(df, Design(wgt="w"))


RK = {
    "age": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
    "region": {"North": 50.0, "South": 30.0},
}


def test_rake_strict_false_records_max_iter(rk, capsys):
    with pytest.warns(SvyUserWarning, match=r"\[MAX_ITER_REACHED\]") as rec:
        out = rk.weighting.rake(controls=RK, max_iter=3, tol=1e-20, strict=False)
    assert _svy(rec) and all(r.filename == __file__ for r in _svy(rec))
    assert capsys.readouterr().out == ""
    w = _one(out, MAXIT)
    assert w.got["max_iter"] == 3 and w.got["max_margin_error"] > 0
    assert w.expected == {"tol": 1e-20}
    assert "rk_wgt" in out.data.columns
    assert not _codes(rk, MAXIT)


def test_rake_strict_raises_and_leaves_sample_untouched(rk, capsys):
    before_cols, before_design = rk.data.columns, rk.design
    for inplace in (False, True):
        with pytest.raises(WeightingError) as ei:
            rk.weighting.rake(controls=RK, max_iter=3, tol=1e-20, inplace=inplace)
        assert ei.value.code == "CONVERGENCE_FAILED"
    assert rk.data.columns == before_cols and rk.design == before_design
    assert not _codes(rk, MAXIT)
    assert capsys.readouterr().out == ""


def test_rake_display_iter_still_prints(rk, capsys):
    rk.weighting.rake(controls=RK, max_iter=3, tol=1e-20, strict=False, display_iter=True)
    out = capsys.readouterr().out
    assert "Raking: max margin error =" in out and "[not converged]" in out
    rk.weighting.rake(
        controls=RK, trimming=TrimConfig(upper=12.0, max_iter=2), strict=False, display_iter=True
    )
    assert "Cycle   1 |" in capsys.readouterr().out


@pytest.fixture
def tight() -> Sample:
    df = pl.DataFrame({"c": ["a"] * 10 + ["b"] * 2, "w": [1.0] * 10 + [30.0, 1.0]})
    return Sample(df, Design(wgt="w"))


CFG = TrimConfig(upper=2.0, max_iter=2)
CYCLES = {
    "poststratify": lambda s, **k: s.weighting.poststratify(
        {"a": 10, "b": 30}, cells="c", trimming=CFG, **k
    ),
    "calibrate": lambda s, **k: s.weighting.calibrate(
        controls={Cat("c"): {"a": 10, "b": 30}}, trimming=CFG, **k
    ),
    "rake": lambda s, **k: s.weighting.rake(controls={"c": {"a": 10, "b": 30}}, trimming=CFG, **k),
}


@pytest.mark.parametrize("name", list(CYCLES))
def test_trim_cycle_strict_false_records(tight, name, capsys):
    with pytest.warns(SvyUserWarning, match=r"\[MAX_ITER_REACHED\]") as rec:
        out = CYCLES[name](tight, strict=False)
    assert _svy(rec) and all(r.filename == __file__ for r in _svy(rec))
    w = _one(out, MAXIT)
    assert "did not converge" in w.title
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name", list(CYCLES))
@pytest.mark.parametrize("inplace", [False, True])
def test_trim_cycle_strict_failure_leaves_sample_untouched(tight, name, inplace):
    data, design = tight.data, tight.design
    history = tight.design_history if hasattr(tight, "design_history") else None
    with pytest.raises(WeightingError) as ei:
        CYCLES[name](tight, inplace=inplace)
    assert ei.value.code == "CONVERGENCE_FAILED"
    assert tight.data.equals(data)
    assert tight.design == design
    if history is not None:
        assert tight.design_history == history


def test_standardize_trim_cycle_recorded():
    df = pl.DataFrame(
        {"c": ["a"] * 10 + ["b"] * 2, "g": ["g"] * 12, "w": [1.0] * 10 + [30.0, 1.0]}
    )
    with pytest.warns(SvyUserWarning, match=r"\[MAX_ITER_REACHED\]") as rec:
        out = Sample(df, Design(wgt="w")).weighting.standardize(
            "c", shares={"a": 1, "b": 3}, by="g", trimming=CFG
        )
    assert _svy(rec) and all(r.filename == __file__ for r in _svy(rec))
    assert "Trim-standardize" in _one(out, MAXIT).title


def test_failed_inplace_call_leaves_data_and_design(s):
    data, design = s.data, s.design
    with pytest.raises(WeightingError):
        s.weighting.adjust(
            "st", cells="a", trimming=TrimConfig(upper=3.0, by="nope"), inplace=True
        )
    assert s.data.equals(data) and s.design == design


def test_calibrate_where_keeps_scoped_findings():
    df = pl.DataFrame(
        {
            "c": ["a"] * 10 + ["b"] * 2 + ["a"],
            "k": [1] * 12 + [0],
            "w": [1.0] * 10 + [30.0, 1.0, 1.0],
        }
    )
    out = Sample(df, Design(wgt="w")).weighting.calibrate(
        controls={Cat("c"): {"a": 10, "b": 30}}, where=col("k") == 1, trimming=CFG, strict=False
    )
    assert "Trim-calibrate" in _one(out, MAXIT).title


# ---------------------------------------------------------------------------
# calibrate: the controls are checked on every path
# ---------------------------------------------------------------------------

NOTMET = "CALIBRATION_NOT_MET"


@pytest.fixture
def cal() -> Sample:
    df = pl.DataFrame(
        {
            "c": ["a", "b"] * 6,
            "d": ["d1"] * 6 + ["d2"] * 6,
            "one": [1.0] * 12,
            "x": [float(i % 5) for i in range(12)],
            "w": [1.0 + (i % 3) for i in range(12)],
        }
    )
    return Sample(df, Design(wgt="w"))


def test_calibrate_met_has_no_finding(cal):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = cal.weighting.calibrate(controls={Cat("c"): {"a": 20, "b": 30}, "x": 40})
        out = out.weighting.calibrate(
            controls={"d1": {"x": 20}, "d2": {"x": 25}}, by="d", wgt_name="c2"
        )
    assert not _codes(out, NOTMET)


def test_calibrate_inconsistent_controls_strict(cal):
    # "one" is the sum of the Cat indicators, so 50 vs 20 + 30 = 50 is fine and
    # 60 is not: no weights reproduce both.
    data, design = cal.data, cal.design
    for inplace in (False, True):
        with pytest.raises(WeightingError) as ei:
            cal.weighting.calibrate(
                controls={Cat("c"): {"a": 20, "b": 30}, "one": 60}, inplace=inplace
            )
        err = ei.value
        assert err.code == NOTMET
        assert err.expected == [20.0, 30.0, 60.0]
        assert len(err.got) == 3 and err.got != err.expected
        assert err.extra["max_rel_error"] > 1e-4 and err.extra["tol"] == 1e-4
        json.dumps(err.to_dict())
    assert cal.data.equals(data) and cal.design == design


def test_calibrate_inconsistent_controls_not_strict(cal):
    with pytest.warns(SvyUserWarning, match=r"\[CALIBRATION_NOT_MET\]") as rec:
        out = cal.weighting.calibrate(
            controls={Cat("c"): {"a": 20, "b": 30}, "one": 60}, strict=False
        )
    assert len(_svy(rec)) == 1 and rec[0].filename == __file__
    w = _one(out, NOTMET)
    assert w.expected == [20.0, 30.0, 60.0]
    assert "calib_wgt" in out.data.columns and "'calib_wgt' holds" in w.detail


def test_calibrate_matrix_not_met_names_the_domain(cal):
    X = np.column_stack([np.ones(12), np.ones(12)])
    ctl = {"d1": [10.0, 10.0], "d2": [10.0, 20.0]}
    with pytest.raises(WeightingError) as ei:
        cal.weighting.calibrate_matrix(aux_vars=X, control=ctl, by="d")
    err = ei.value
    assert err.code == NOTMET
    assert list(err.expected) == ["d2"] and err.expected["d2"] == [10.0, 20.0]
    assert list(err.got) == ["d2"]
    assert err.extra["domains"] == ["d2"]
    with pytest.warns(SvyUserWarning, match=r"\[CALIBRATION_NOT_MET\]"):
        out = cal.weighting.calibrate_matrix(aux_vars=X, control=ctl, by="d", strict=False)
    assert list(_one(out, NOTMET).got) == ["d2"]


def test_calibrate_matrix_weights_only_keeps_its_check(cal):
    X = np.column_stack([np.ones(12), np.ones(12)])
    with pytest.raises(WeightingError) as ei:
        cal.weighting.calibrate_matrix(aux_vars=X, control=[10.0, 20.0], weights_only=True)
    assert ei.value.code == NOTMET and ei.value.expected == [10.0, 20.0]
    w = cal.weighting.calibrate_matrix(
        aux_vars=X, control=[10.0, 20.0], weights_only=True, strict=False
    )
    assert isinstance(w, np.ndarray)


def test_calibrate_where_carries_the_finding(cal):
    with pytest.warns(SvyUserWarning, match=r"\[CALIBRATION_NOT_MET\]"):
        out = cal.weighting.calibrate(
            controls={Cat("c"): {"a": 10, "b": 15}, "one": 40},
            where=col("d") == "d1",
            strict=False,
        )
    assert _one(out, NOTMET)


def test_calibrate_bounded_still_refused(cal):
    with pytest.raises(NotImplementedError):
        cal.weighting.calibrate(controls={"x": 40}, bounded=True)


def test_calibrate_trimming_cycle_owns_the_fit_check(cal):
    cfg = TrimConfig(upper=2.5, max_iter=3)
    with pytest.raises(WeightingError) as ei:
        cal.weighting.calibrate(controls={Cat("c"): {"a": 20, "b": 30}, "one": 60}, trimming=cfg)
    assert ei.value.code == "CONVERGENCE_FAILED"
    with pytest.warns(SvyUserWarning, match=r"\[MAX_ITER_REACHED\]"):
        out = cal.weighting.calibrate(
            controls={Cat("c"): {"a": 20, "b": 30}, "one": 60}, trimming=cfg, strict=False
        )
    assert not _codes(out, NOTMET)


# ---------------------------------------------------------------------------
# adjust: rows in no adjustment class stay, with their weight
# ---------------------------------------------------------------------------


def _nr() -> Sample:
    df = pl.DataFrame(
        {
            "c": ["a", "a", None, None, "b", "b", "b", None],
            "st": ["rr", "nr", "nr", "uk", "rr", "nr", "rr", "in"],
            "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    return Sample(df, Design(wgt="w"))


def test_adjust_keeps_null_cell_rows_with_their_weight():
    s = _nr()
    with pytest.warns(SvyUserWarning, match=r"\[CELLS_NULL_UNADJUSTED\]") as rec:
        out = s.weighting.adjust("st", cells="c", unknown_to_inelig=False)
    assert len(_svy(rec)) == 1 and rec[0].filename == __file__
    d = out.data
    kept = d.filter(pl.col("c").is_null())
    assert kept["st"].to_list() == ["nr", "uk", "in"]
    assert kept["nr_wgt"].to_list() == [3.0, 4.0, 8.0]
    # in-class nonrespondents leave; their weight went to the respondents of their class
    assert d.filter(pl.col("c").is_not_null())["st"].to_list() == ["rr", "rr", "rr"]
    assert d["nr_wgt"].sum() == pytest.approx(s.data["w"].sum())
    w = _one(out, NULL)
    assert w.got == {"c": 3}
    assert w.extra["nonrespondents_kept"] == 3
    assert "stay in the sample" in w.detail


def test_adjust_keeps_rows_outside_where():
    s = Sample(_nr().data.drop_nulls(), Design(wgt="w"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = s.weighting.adjust("st", cells="c", where=col("c") == "a")
    d = out.data
    b = d.filter(pl.col("c") == "b")
    assert b["st"].to_list() == ["rr", "nr", "rr"]
    assert b["nr_wgt"].to_list() == b["w"].to_list()
    assert d.filter(pl.col("c") == "a")["nr_wgt"].to_list() == [3.0]
    assert d["nr_wgt"].sum() == pytest.approx(s.data["w"].sum())
    assert not _codes(out, NULL)


def test_adjust_respondents_only_false_unchanged():
    s = _nr()
    with pytest.warns(SvyUserWarning):
        out = s.weighting.adjust("st", cells="c", respondents_only=False)
    assert out.data.height == s.data.height
    ref = s.data["w"].to_numpy()
    got = out.data["nr_wgt"].to_numpy()
    nulls = s.data["c"].is_null().to_numpy()
    np.testing.assert_allclose(got[nulls], ref[nulls])


def test_adjust_no_nulls_no_scope_drops_all_nonrespondents():
    s = Sample(_nr().data.drop_nulls(), Design(wgt="w"))
    out = s.weighting.adjust("st", cells="c")
    assert set(out.data["st"].to_list()) == {"rr"}
    assert out.data["nr_wgt"].sum() == pytest.approx(s.data["w"].sum())


def test_adjust_panel_parity_null_cell_row_stays():
    w1 = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "g": ["A", "A", "B", "B"],
            "w": [1.0, 2.0, 3.0, 4.0],
            "resp": ["rr"] * 4,
            "wave": [1] * 4,
        }
    )
    w2 = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "g": ["A", None, "B", "B"],
            "w": [1.0, 2.0, 3.0, 4.0],
            "resp": ["rr", "nr", "rr", "nr"],
            "wave": [2] * 4,
        }
    )
    s = Sample(pl.concat([w1, w2]), Design(case_id="id", wave="wave", wgt="w"))
    with pytest.warns(SvyUserWarning, match=r"\[CELLS_NULL_UNADJUSTED\]"):
        out = s.weighting.adjust("resp", cells="g", where=col("wave") == 2)
    w2_out = out.data.filter(pl.col("wave") == 2).sort("id")
    # case 2 has a null cell at wave 2: kept, weight unchanged; case 4 left through its class
    assert w2_out["id"].to_list() == [1, 2, 3]
    assert w2_out.filter(pl.col("id") == 2)["nr_wgt"].to_list() == [2.0]
    assert _one(out, NULL).extra["nonrespondents_kept"] == 1


def _svy(rec) -> list:
    return [r for r in rec if issubclass(r.category, SvyUserWarning)]
