"""
Trim-and-readjust cycles (``trimming=`` on poststratify, standardize, rake,
calibrate, calibrate_matrix).

- The cycle iterates: the bounds are checked on the readjusted weights, so a
  feasible cap that needs several cycles converges given enough of them.
- A failed cycle whose controls are out of reach of the bounds is reported as
  TRIM_INFEASIBLE instead of CONVERGENCE_FAILED / MAX_ITER_REACHED, following
  on_nonconvergence.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl
import pytest

from svy import Cat, Cross, Design, Sample, SvyUserWarning, Threshold, TrimConfig, col
from svy.core.warnings import Severity, WarnCode
from svy.errors import WeightingError


INFEASIBLE = "TRIM_INFEASIBLE"


@pytest.fixture
def tight() -> Sample:
    # Cell b: two units, weights 30 and 1.
    df = pl.DataFrame({"c": ["a"] * 10 + ["b"] * 2, "w": [1.0] * 10 + [30.0, 1.0]})
    return Sample(df, Design(wgt="w"))


def _aux(s: Sample) -> np.ndarray:
    c = s.data["c"].to_numpy()
    return np.column_stack([c == "a", c == "b"]).astype(float)


def _cycles(a: float, b: float, cfg: TrimConfig) -> dict:
    return {
        "poststratify": lambda s, **k: s.weighting.poststratify(
            {"a": a, "b": b}, cells="c", trimming=cfg, **k
        ),
        "rake": lambda s, **k: s.weighting.rake(
            controls={"c": {"a": a, "b": b}}, trimming=cfg, **k
        ),
        "calibrate": lambda s, **k: s.weighting.calibrate(
            controls={Cat("c"): {"a": a, "b": b}}, trimming=cfg, **k
        ),
        "calibrate_where": lambda s, **k: s.weighting.calibrate(
            controls={Cat("c"): {"a": a, "b": b}}, where=col("w") > 0, trimming=cfg, **k
        ),
        "calibrate_matrix": lambda s, **k: s.weighting.calibrate_matrix(
            aux_vars=_aux(s), controls=[a, b], trimming=cfg, **k
        ),
    }


_WGT = {
    "poststratify": "ps_wgt",
    "rake": "rk_wgt",
    "calibrate": "calib_wgt",
    "calibrate_where": "calib_wgt",
    "calibrate_matrix": "calib_wgt",
}
_WHAT = {
    "poststratify": "Trim-poststratify cycle",
    "rake": "Trim-rake cycle",
    "calibrate": "Trim-calibrate cycle",
    "calibrate_where": "Trim-calibrate cycle",
    "calibrate_matrix": "Trim-calibrate cycle",
}

# Feasible (cell b can be [4, 2]) but needs about 20 cycles.
SLOW = _cycles(10.0, 6.0, TrimConfig(upper=4.0, max_iter=100))
# Out of reach: cell b needs 30 from two units capped at 2.
OUT = _cycles(10.0, 30.0, TrimConfig(upper=2.0, max_iter=5))


# ---------------------------------------------------------------------------
# The cycle iterates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SLOW))
def test_feasible_cap_needing_many_cycles_converges(tight, name):
    out = SLOW[name](tight)
    w = out.data[_WGT[name]].to_numpy()
    c = out.data["c"].to_numpy()
    assert w.max() <= 4.0 * (1 + 1e-4)
    assert w[c == "a"].sum() == pytest.approx(10.0, rel=1e-6)
    assert w[c == "b"].sum() == pytest.approx(6.0, rel=1e-6)


def test_feasible_standardize_converges():
    df = pl.DataFrame({"c": ["a"] * 10 + ["b"] * 2, "w": [1.0] * 10 + [30.0, 1.0]})
    out = Sample(df, Design(wgt="w")).weighting.standardize(
        "c", shares={"a": 5, "b": 1}, trimming=TrimConfig(upper=4.0, max_iter=100)
    )
    w = out.data["std_wgt"].to_numpy()
    assert w.max() <= 4.0 * (1 + 1e-4)
    assert w.sum() == pytest.approx(41.0)


@pytest.mark.parametrize("name", list(SLOW))
def test_too_few_cycles_is_still_non_convergence(tight, name):
    call = _cycles(10.0, 6.0, TrimConfig(upper=4.0, max_iter=2))[name]
    with pytest.raises(WeightingError) as ei:
        call(tight)
    assert ei.value.code == "CONVERGENCE_FAILED"
    assert "max_iter" in ei.value.hint


def test_bounds_already_met_leave_the_readjustment_unchanged(tight):
    plain = tight.weighting.poststratify({"a": 10.0, "b": 6.0}, cells="c")
    capped = tight.weighting.poststratify(
        {"a": 10.0, "b": 6.0}, cells="c", trimming=TrimConfig(upper=100.0)
    )
    np.testing.assert_array_equal(
        capped.data["ps_wgt"].to_numpy(), plain.data["ps_wgt"].to_numpy()
    )


# ---------------------------------------------------------------------------
# TRIM_INFEASIBLE under on_nonconvergence="error"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(OUT))
def test_out_of_reach_raises_trim_infeasible(tight, name):
    with pytest.raises(WeightingError) as ei:
        OUT[name](tight)
    err = ei.value
    assert err.code == INFEASIBLE and err.param == "trimming"
    assert err.detail.startswith(f"{_WHAT[name]} cannot converge")
    assert "More cycles cannot help" in err.detail and "NOT been modified" in err.detail
    assert "max_iter" not in err.hint
    json.dumps(err.to_dict())


def test_poststratify_error_names_the_cell_and_the_bound(tight):
    with pytest.raises(WeightingError) as ei:
        OUT["poststratify"](tight)
    err = ei.value
    assert err.got == {
        "controls": [
            {
                "control": "c='b'",
                "total": 30.0,
                "reachable": [0.0, 4.0],
                "n": 2,
                "mean_needed": 15.0,
            }
        ]
    }
    assert "the control for c='b' is 30, but its 2 unit(s) reach at most 4" in err.detail
    assert "(upper=2)" in err.detail
    assert "c='b' needs an upper bound of at least 15" in err.hint


@pytest.mark.parametrize("name", list(OUT))
@pytest.mark.parametrize("inplace", [False, True])
def test_out_of_reach_leaves_the_sample(tight, name, inplace):
    data, design, history = tight.data, tight.design, tight.design_history
    with pytest.raises(WeightingError):
        OUT[name](tight, inplace=inplace)
    assert tight.data.equals(data) and tight.design == design
    assert tight.design_history == history


def test_standardize_out_of_reach():
    # The total (41) over 12 units averages 3.4, above any cap of 2.
    df = pl.DataFrame({"c": ["a"] * 10 + ["b"] * 2, "w": [1.0] * 10 + [30.0, 1.0]})
    with pytest.raises(WeightingError) as ei:
        Sample(df, Design(wgt="w")).weighting.standardize(
            "c", shares={"a": 1, "b": 3}, trimming=TrimConfig(upper=2.0)
        )
    assert ei.value.code == INFEASIBLE
    assert ei.value.detail.startswith("Trim-standardize cycle cannot converge")


def test_lower_bound_out_of_reach(tight):
    with pytest.raises(WeightingError) as ei:
        tight.weighting.poststratify(
            {"a": 10.0, "b": 0.5}, cells="c", trimming=TrimConfig(lower=1.0, max_iter=5)
        )
    err = ei.value
    assert err.code == INFEASIBLE
    assert "reach at least 2" in err.detail and "(lower=1)" in err.detail
    assert "c='b' needs a lower bound of at most 0.25" in err.hint


def test_several_unreachable_controls_are_all_listed():
    df = pl.DataFrame({"c": ["a"] * 2 + ["b"] * 2 + ["z"] * 8, "w": [1.0] * 12})
    with pytest.raises(WeightingError) as ei:
        Sample(df, Design(wgt="w")).weighting.poststratify(
            {"a": 20.0, "b": 20.0, "z": 8.0}, cells="c", trimming=TrimConfig(upper=3.0)
        )
    err = ei.value
    assert [c["control"] for c in err.got["controls"]] == ["c='a'", "c='b'"]
    assert "1 other control(s) cannot be met either" in err.detail


def test_zero_weights_carry_nothing():
    # A zero-weight unit stays zero, so cell b has one unit to carry its total.
    df = pl.DataFrame({"c": ["a"] * 10 + ["b"] * 2, "w": [1.0] * 10 + [5.0, 0.0]})
    with pytest.raises(WeightingError) as ei:
        Sample(df, Design(wgt="w")).weighting.poststratify(
            {"a": 10.0, "b": 6.0}, cells="c", trimming=TrimConfig(upper=4.0, max_iter=5)
        )
    assert ei.value.code == INFEASIBLE
    assert ei.value.got["controls"][0]["n"] == 1


# ---------------------------------------------------------------------------
# rake: only fixed bounds are diagnosed
# ---------------------------------------------------------------------------


def test_rake_absolute_threshold_is_diagnosed(tight):
    with pytest.raises(WeightingError) as ei:
        tight.weighting.rake(
            controls={"c": {"a": 10, "b": 30}},
            trimming=TrimConfig(upper=Threshold.absolute(2.0), max_iter=5),
        )
    assert ei.value.code == INFEASIBLE


def test_rake_relative_threshold_is_not_diagnosed(tight):
    # A relative bound is resolved again on every cycle's raked weights, so the
    # last cycle's bound says nothing about the next.
    with pytest.raises(WeightingError) as ei:
        tight.weighting.rake(
            controls={"c": {"a": 10, "b": 30}},
            trimming=TrimConfig(upper=Threshold("median", 1.5), max_iter=3),
        )
    assert ei.value.code == "CONVERGENCE_FAILED"


def test_rake_names_the_margin_level():
    df = pl.DataFrame(
        {
            "sex": ["m", "f"] * 6,
            "age": ["y"] * 10 + ["o"] * 2,
            "w": [1.0] * 12,
        }
    )
    with pytest.raises(WeightingError) as ei:
        Sample(df, Design(wgt="w")).weighting.rake(
            controls={"sex": {"m": 20, "f": 20}, "age": {"y": 20, "o": 20}},
            trimming=TrimConfig(upper=5.0),
        )
    assert [c["control"] for c in ei.value.got["controls"]] == ["age='o'"]


# ---------------------------------------------------------------------------
# calibrate: control names, domains, continuous auxiliaries, untrimmed units
# ---------------------------------------------------------------------------


def test_calibrate_matrix_unlabeled_columns(tight):
    with pytest.raises(WeightingError) as ei:
        OUT["calibrate_matrix"](tight)
    assert ei.value.got["controls"][0]["control"] == "aux column 1"


def test_calibrate_matrix_labels_name_the_columns(tight):
    with pytest.raises(WeightingError) as ei:
        tight.weighting.calibrate_matrix(
            aux_vars=_aux(tight),
            controls={"a": 10.0, "b": 30.0},
            labels=["a", "b"],
            trimming=TrimConfig(upper=2.0),
        )
    assert ei.value.got["controls"][0]["control"] == "'b'"


def test_calibrate_cross_term_is_named_by_its_columns():
    df = pl.DataFrame(
        {"c": ["a"] * 10 + ["b"] * 2, "d": ["x"] * 12, "w": [1.0] * 10 + [30.0, 1.0]}
    )
    with pytest.raises(WeightingError) as ei:
        Sample(df, Design(wgt="w")).weighting.calibrate(
            controls={Cross(Cat("c"), Cat("d")): {("a", "x"): 10, ("b", "x"): 30}},
            trimming=TrimConfig(upper=2.0),
        )
    assert ei.value.got["controls"][0]["control"] == "(c, d)=('b', 'x')"


def test_calibrate_by_names_the_domain():
    df = pl.DataFrame(
        {
            "g": ["g1"] * 12 + ["g2"] * 12,
            "c": (["a"] * 10 + ["b"] * 2) * 2,
            "w": [1.0] * 24,
        }
    )
    with pytest.raises(WeightingError) as ei:
        Sample(df, Design(wgt="w")).weighting.calibrate(
            controls={
                "g1": {Cat("c"): {"a": 10, "b": 2}},
                "g2": {Cat("c"): {"a": 10, "b": 30}},
            },
            by="g",
            trimming=TrimConfig(upper=4.0),
        )
    [entry] = ei.value.got["controls"]
    assert entry["domain"] == "g2" and entry["control"] == "c='b'"
    assert "in domain 'g2'" in ei.value.detail


def test_calibrate_continuous_auxiliary():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 3, "w": [1.0] * 12})
    with pytest.raises(WeightingError) as ei:
        Sample(df, Design(wgt="w")).weighting.calibrate(
            controls={"x": 500.0}, trimming=TrimConfig(upper=5.0)
        )
    [entry] = ei.value.got["controls"]
    assert entry["control"] == "x" and entry["reachable"] == [0.0, 150.0]
    assert entry["mean_needed"] is None
    assert "each control's units can carry its total" in ei.value.hint


def test_calibrate_untrimmed_units_are_unbounded():
    # Group b is below min_cell_size, so it is calibrated but never trimmed and
    # can carry any total: not out of reach, whatever b's control.
    df = pl.DataFrame({"grp": ["a"] * 10 + ["b"] * 2, "w": [1.0] * 12})
    with pytest.warns(SvyUserWarning, match=r"\[DOMAIN_SKIPPED\]"):
        out = Sample(df, Design(wgt="w")).weighting.calibrate(
            controls={Cat("grp"): {"a": 10.0, "b": 100.0}},
            trimming=TrimConfig(upper=2.0, by="grp", min_cell_size=5),
        )
    w = out.data["calib_wgt"].to_numpy()
    assert w[10:].sum() == pytest.approx(100.0)
    assert out.warnings.list(code=WarnCode.DOMAIN_SKIPPED)


# ---------------------------------------------------------------------------
# on_nonconvergence="warn" / "ignore"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(OUT))
def test_warn_records_trim_infeasible_once(tight, name):
    with pytest.warns(SvyUserWarning, match=r"\[TRIM_INFEASIBLE\]") as rec:
        out = OUT[name](tight, on_nonconvergence="warn")
    assert len([r for r in rec if issubclass(r.category, SvyUserWarning)]) == 1
    [found] = out.warnings.list(code=INFEASIBLE)
    assert found.level == Severity.WARNING
    assert f"{_WGT[name]!r} holds the last cycle's weights" in found.detail
    assert "NOT been modified" not in found.detail
    assert not out.warnings.list(code=WarnCode.MAX_ITER_REACHED)
    assert _WGT[name] in out.data.columns
    json.dumps(found.to_dict())


@pytest.mark.parametrize("name", list(OUT))
def test_ignore_records_at_info_without_raising(tight, name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", SvyUserWarning)
        out = OUT[name](tight, on_nonconvergence="ignore")
    [found] = out.warnings.list(code=INFEASIBLE)
    assert found.level == Severity.INFO


@pytest.mark.parametrize("mode", ["warn", "ignore"])
@pytest.mark.filterwarnings(r"ignore:\[TRIM_INFEASIBLE\]:svy.SvyUserWarning")
def test_warn_and_ignore_keep_the_same_weights(tight, mode):
    kept = OUT["poststratify"](tight, on_nonconvergence=mode)
    ref = OUT["poststratify"](tight, on_nonconvergence="ignore")
    np.testing.assert_array_equal(kept.data["ps_wgt"].to_numpy(), ref.data["ps_wgt"].to_numpy())
