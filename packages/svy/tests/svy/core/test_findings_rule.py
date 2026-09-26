# tests/svy/core/test_findings_rule.py
"""One rule for findings: recorded in ``sample.warnings`` and raised once as a
``svy.SvyUserWarning`` at the caller's line; INFO entries are recorded only.
Without a sample the finding is only raised, with the same category and line."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

import numpy as np
import polars as pl
import pytest

import svy

from svy import Design, Sample, SvyUserWarning
from svy.core.warnings import Severity, WarnCode, WarningStore
from svy.selection import allocate


def _null_cell_sample() -> Sample:
    df = pl.DataFrame({"c": ["a", None, "b", "a"], "w": [1.0, 2.0, 3.0, 4.0]})
    return Sample(df, Design(wgt="w"))


def _ps(s: Sample, **k) -> Sample:
    return s.weighting.poststratify({"a": 5.0, "b": 5.0}, cells="c", **k)


def _svy_records(rec) -> list:
    return [r for r in rec if issubclass(r.category, SvyUserWarning)]


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_recorded_and_raised_once_at_the_callers_line():
    s = _null_cell_sample()
    with pytest.warns(SvyUserWarning) as rec:
        out = _ps(s)
    rec = _svy_records(rec)
    assert len(rec) == 1
    assert rec[0].filename == __file__
    assert str(rec[0].message).startswith("[CELLS_NULL_UNADJUSTED] Rows with a null left")
    assert issubclass(rec[0].category, UserWarning)
    assert len(out.warnings.list(code=WarnCode.CELLS_NULL_UNADJUSTED)) == 1


def test_category_filter_silences_but_still_records():
    s = _null_cell_sample()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=SvyUserWarning)
        out = _ps(s)
    assert not _svy_records(rec)
    assert out.warnings.list(code=WarnCode.CELLS_NULL_UNADJUSTED)


def test_warnings_as_errors_turn_a_finding_into_an_exception():
    s = _null_cell_sample()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SvyUserWarning)
        with pytest.raises(SvyUserWarning, match=r"\[CELLS_NULL_UNADJUSTED\]"):
            _ps(s)
    # the failed default-mode call left the caller's sample alone
    assert not s.warnings.list(code=WarnCode.CELLS_NULL_UNADJUSTED)


def test_info_entries_are_recorded_not_raised():
    df = pl.DataFrame({"w": [float(i) for i in range(1, 21)]})
    s = Sample(df, Design(wgt="w"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = s.weighting.trim(upper=15.0)
    assert out.warnings.list(code=WarnCode.WEIGHT_ADJ_AUDIT)
    assert not _svy_records(rec)


def test_a_repeat_the_store_suppresses_is_not_raised():
    s = _null_cell_sample()
    kw = dict(code="X_TEST", title="t", detail="same", where="test")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        s.warn(**kw)
        s.warn(**kw)
        s.warn(**{**kw, "detail": "other"})
    assert [str(r.message) for r in _svy_records(rec)] == ["[X_TEST] t: same", "[X_TEST] t: other"]
    assert len(s.warnings.list(code="X_TEST")) == 2


def test_a_capped_code_is_not_raised():
    s = _null_cell_sample()
    s._warnings = WarningStore(max_per_code=1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for i in range(3):
            s.warn(code="X_CAP", title="t", detail=str(i), where="test")
    assert len(_svy_records(rec)) == 1
    assert len(s.warnings.list(code="X_CAP")) == 1


def test_forks_do_not_raise_their_parents_findings_again():
    s = _null_cell_sample()
    with pytest.warns(SvyUserWarning):
        a = _ps(s)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        b = a.weighting.normalize(wgt_name="n2")
        c = a.wrangling.mutate({"z": pl.lit(1)})
    assert not _svy_records(rec)
    assert b.warnings.list(code=WarnCode.CELLS_NULL_UNADJUSTED)
    assert c.warnings.list(code=WarnCode.CELLS_NULL_UNADJUSTED)


def test_error_level_entries_are_raised_too():
    s = _null_cell_sample()
    with pytest.warns(SvyUserWarning, match="ERR_TEST"):
        s.warn(code="ERR_TEST", title="t", detail="d", where="test", level=Severity.ERROR)


def test_record_is_json_ready():
    s = _null_cell_sample()
    with pytest.warns(SvyUserWarning):
        out = _ps(s)
    json.dumps(out.warnings.to_dicts())


_SCRIPT = """
import polars as pl, svy
s = svy.Sample(pl.DataFrame({"c": ["a", None], "w": [1.0, 2.0]}), svy.Design(wgt="w"))
s.weighting.poststratify({"a": 1.0}, cells="c")  # the user's line
print("done")
"""


def _run(*flags: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.run(
        [sys.executable, *flags, "-c", _SCRIPT], capture_output=True, text=True, env=env
    )


def test_command_line_filters():
    default = _run()
    assert default.returncode == 0
    assert "SvyUserWarning: [CELLS_NULL_UNADJUSTED]" in default.stderr
    assert "<string>:4" in default.stderr
    assert _run("-W", "ignore").stderr == ""
    assert _run("-W", "ignore::UserWarning").stderr == ""
    failing = _run("-W", "error::UserWarning")
    assert failing.returncode != 0 and "done" not in failing.stdout


# ---------------------------------------------------------------------------
# Audited sites
# ---------------------------------------------------------------------------


def _cycle(strat, wgt_scale=1.0, **extra):
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "strat": strat,
            "psu": [1, 2, 1, 2, 3, 3],
            "w": [x * wgt_scale for x in [10.0, 12.0, 8.0, 9.0, 11.0, 7.0]],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            **extra,
        }
    )
    return Sample(df, Design(stratum="strat", psu="psu", wgt="w"))


def test_combine_samples_records_on_the_combined_sample():
    s1 = _cycle([1, 1, 2, 2, 1, 2], only=list("abcdef"))
    s2 = _cycle([3, 3, 4, 4, 3, 4], 2.0)
    with pytest.warns(SvyUserWarning) as rec:
        c = svy.combine_samples([s1, s2], wave_labels=["2", "1"])
    msgs = [str(r.message) for r in _svy_records(rec)]
    assert any(m.startswith("[COMBINE_COLUMNS_NULL_FILLED]") for m in msgs)
    assert any(m.startswith("[WAVE_LABELS_UNORDERED]") for m in msgs)
    assert all(r.filename == __file__ for r in _svy_records(rec))
    assert c.warnings.list(code="COMBINE_COLUMNS_NULL_FILLED")[0].got == ["only"]
    assert c.warnings.list(code="WAVE_LABELS_UNORDERED")
    assert not s1.warnings.list(code="COMBINE_COLUMNS_NULL_FILLED")


def test_rep_reset_recorded_via_update_design_and_raised_via_bare_design():
    df = pl.DataFrame({"w": [1.0, 2.0], "v": [1.0, 1.0], "r1": [1.0, 2.0], "r2": [1.0, 2.0]})
    rw = svy.BrrWgts(prefix="r", n_reps=2)
    s = Sample(df, Design(wgt="w", rep_wgts=rw))
    with pytest.warns(SvyUserWarning, match=r"\[REP_WGTS_RESET\]") as rec:
        s.update_design(wgt="v")
    assert rec[0].filename == __file__
    assert s.warnings.list(code="REP_WGTS_RESET")
    with pytest.warns(SvyUserWarning, match=r"\[REP_WGTS_RESET\]") as rec:
        Design(wgt="w", rep_wgts=rw).update(wgt="v")
    assert rec[0].filename == __file__
    with pytest.warns(SvyUserWarning, match=r"\[REP_WGTS_RESET\]"):
        t = Sample(df, Design(wgt="w", rep_wgts=rw)).use_weight("v")
    assert t.warnings.list(code="REP_WGTS_RESET")


def test_design_fields_removed_recorded():
    s = _cycle([1, 1, 2, 2, 1, 2])
    with pytest.warns(SvyUserWarning, match=r"\[DESIGN_FIELDS_REMOVED\]") as rec:
        out = s.wrangling.remove_columns("psu", force=True)
    assert rec[0].filename == __file__
    assert out.warnings.list(code="DESIGN_FIELDS_REMOVED")[0].got


def test_adjustment_not_credited_recorded():
    s = _cycle([1, 1, 2, 2, 1, 2]).weighting.poststratify(
        {1: 30.0, 2: 30.0}, cells="strat", wgt_name="ps"
    )
    cells = s.design.wgt_adjustment.cells[0]
    stripped = s._replace_data(s.data.drop(cells))
    with pytest.warns(SvyUserWarning, match=r"\[ADJUSTMENT_NOT_CREDITED\]") as rec:
        stripped.estimation.mean("x")
    assert rec[0].filename == __file__
    assert stripped.warnings.list(code="ADJUSTMENT_NOT_CREDITED")


def test_with_replacement_repeats_are_recorded_not_raised():
    """Drawing more units than a group holds is what wr=True means."""
    frame = Sample(pl.DataFrame({"id": list(range(5)), "g": ["a"] * 3 + ["b"] * 2}), Design())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = frame.sampling.srs(4, by="g", wr=True, rstate=1)
    found = out.warnings.list(code="SELECTION_N_EXCEEDS_GROUP")
    assert [f.got["group"] for f in found] == ["a", "b"]  # both groups hold fewer than 4
    assert all(f.level == Severity.INFO for f in found)


def test_without_replacement_too_large_n_is_still_an_error():
    frame = Sample(pl.DataFrame({"id": list(range(5)), "g": ["a"] * 3 + ["b"] * 2}), Design())
    with pytest.raises(ValueError, match="exceeds the available population"):
        frame.sampling.srs(4, by="g", rstate=1)


def test_allocation_without_a_sample_is_raised_only():
    with pytest.warns(SvyUserWarning, match="exceeds the total frame") as rec:
        allocate({"a": 3, "b": 2}, method="proportional", n_total=10)
    assert rec[0].filename == __file__


def test_glm_finding_recorded():
    rng = np.random.default_rng(1)
    df = pl.DataFrame(
        {"y": rng.normal(size=30), "x": rng.normal(size=30), "c": ["k"] * 30, "w": [1.0] * 30}
    )
    s = Sample(df, Design(wgt="w"))
    with pytest.warns(SvyUserWarning, match=r"\[GLM_TERM_DROPPED\]") as rec:
        s.glm.fit(y="y", x=["x", svy.Cat("c")])
    assert rec[0].filename == __file__
    assert s.warnings.list(code="GLM_TERM_DROPPED")


def test_accessor_overwrite_is_raised_only():
    @svy.register_sample_accessor("_svy_test_acc")
    class A:
        def __init__(self, s):
            self.s = s

    with pytest.warns(SvyUserWarning, match="already defined") as rec:

        @svy.register_sample_accessor("_svy_test_acc")
        class B:
            def __init__(self, s):
                self.s = s

    assert rec[0].filename == __file__
    delattr(Sample, "_svy_test_acc")


# ---------------------------------------------------------------------------
# De-duplication per sample state
# ---------------------------------------------------------------------------

_KW = dict(code="X_STATE", title="t", detail="same", where="test")


def _raised(fn) -> list[str]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return [str(r.message) for r in _svy_records(rec)]


def test_same_finding_on_an_unchanged_sample_once():
    s = _null_cell_sample()
    assert len(_raised(lambda: [s.warn(**_KW) for _ in range(3)])) == 1
    assert len(s.warnings.list(code="X_STATE")) == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.set_design(s.design),
        lambda s: s.update_design(wgt="w"),
        lambda s: s.wrangling.mutate({"z": pl.lit(1)}, inplace=True),
    ],
    ids=["set_design", "update_design", "data"],
)
def test_a_new_state_is_a_new_event(change):
    s = _null_cell_sample()
    assert len(_raised(lambda: s.warn(**_KW))) == 1
    change(s)
    assert len(_raised(lambda: s.warn(**_KW))) == 1
    assert len(_raised(lambda: s.warn(**_KW))) == 0
    found = s.warnings.list(code="X_STATE")
    assert len(found) == 2 and found[0].state != found[1].state


def test_forks_keep_their_own_state():
    s = _null_cell_sample()
    _raised(lambda: s.warn(**_KW))
    fork = s.wrangling.mutate({"z": pl.lit(1)})
    assert len(_raised(lambda: fork.warn(**_KW))) == 1
    assert len(_raised(lambda: s.warn(**_KW))) == 0
    assert len(s.warnings.list(code="X_STATE")) == 1
    assert len(fork.warnings.list(code="X_STATE")) == 2


def test_per_code_cap_spans_states():
    s = _null_cell_sample()
    s._warnings = WarningStore(max_per_code=2)
    n = 0
    for _ in range(4):
        n += len(_raised(lambda: s.warn(**_KW)))
        s.set_design(s.design)
    assert n == 2 and len(s.warnings.list(code="X_STATE")) == 2


def test_estimation_finding_once_per_state():
    df = pl.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "w": [1.0] * 4, "r1": [1.0] * 4, "r2": [2.0] * 4}
    )
    s = Sample(df, Design(wgt="w", rep_wgts=svy.BrrWgts(prefix="r", n_reps=2)))
    msgs = _raised(lambda: [s.estimation.mean("y", method="taylor") for _ in range(3)])
    assert [m.split("]")[0] for m in msgs] == ["[TAYLOR_WITHOUT_DESIGN"]
    s.set_design(s.design)
    assert len(_raised(lambda: s.estimation.mean("y", method="taylor"))) == 1


# ---------------------------------------------------------------------------
# Chosen behaviour is recorded, not raised
# ---------------------------------------------------------------------------


def test_trim_without_redistribution_records_the_sum_change_quietly():
    df = pl.DataFrame({"w": [1.0] * 19 + [100.0]})
    s = Sample(df, Design(wgt="w"))
    msgs = _raised(lambda: s.weighting.trim(upper=5.0, redistribute=False, inplace=True))
    assert not msgs
    (w,) = s.warnings.list(code=WarnCode.WEIGHT_SUM_CHANGED)
    assert w.level == Severity.INFO
