# tests/svy/weighting/test_wgt_errors_as_data.py
"""Weighting errors are data: a specific code, structured expected/got in the
data's own values, and a hint with the caller's names.

A caller that only runs the call and reports ``SvyError.to_dict()`` (no
preflight) must be able to act on it.
"""

from __future__ import annotations

import datetime as dt
import json
import warnings

import numpy as np
import polars as pl
import pytest

from svy import Cat, Design, Sample, TrimConfig
from svy.errors import DimensionError, MethodError, SvyError, WeightingError


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "zone": [1, 1, 2, 2, 3, 3, 1, 2],
            "sex": ["M", "F"] * 4,
            "b": [True, False] * 4,
            "st": ["rr", "nr", "rr", "Refused", "dk", "rr", "dk", "rr"],
            "day": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)] * 4,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4],
            "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )


@pytest.fixture
def s() -> Sample:
    return Sample(_df(), Design(wgt="w"))


def _no_numpy(x) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            _no_numpy(k)
            _no_numpy(v)
    elif isinstance(x, (list, tuple, set)):
        for v in x:
            _no_numpy(v)
    else:
        assert type(x).__module__ != "numpy", f"numpy value {x!r} in an error field"


def check(err: SvyError, code: str) -> dict:
    """The contract every weighting error keeps; returns the JSON payload."""
    assert err.code == code
    _no_numpy(err.expected)
    _no_numpy(err.got)
    for text in (err.detail, err.hint or "", str(err.expected), str(err.got)):
        assert "np." not in text and "numpy" not in text
    payload = json.loads(json.dumps(err.to_dict()))["error"]
    assert payload["code"] == code
    return payload


def raises(fn, code: str, cls=WeightingError) -> SvyError:
    with pytest.raises(cls) as ei:
        fn()
    check(ei.value, code)
    return ei.value


# ---------------------------------------------------------------------------
# to_dict keeps structure
# ---------------------------------------------------------------------------


def test_to_dict_keeps_structured_values():
    err = SvyError(
        title="t",
        detail="d",
        expected=[1, dt.date(2024, 1, 1), ("a", True)],
        got={"missing": [np.int64(3)], ("A", 1): np.float64(2.5), dt.date(2024, 1, 2): 1},
        extra={"n": np.int32(4)},
    )
    e = json.loads(json.dumps(err.to_dict()))["error"]
    assert e["expected"] == [1, "2024-01-01", ["a", True]]
    assert e["got"] == {"missing": [3], "A_&_1": 2.5, "2024-01-02": 1}
    assert e["extra"] == {"n": 4}


def test_to_dict_still_shortens_prose():
    e = SvyError(title="t", detail="d", got="z" * 300).to_dict()["error"]
    assert e["got"].endswith("…")


# ---------------------------------------------------------------------------
# The three cases the brief verified
# ---------------------------------------------------------------------------


def test_poststratify_missing_and_extra_cells(s):
    err = raises(
        lambda: s.weighting.poststratify({1: 10, 2: 10, 9: 10}, cells="zone"),
        "CONTROLS_KEYS_MISMATCH",
    )
    assert err.expected == [1, 2, 3]
    assert err.got == {"missing": [3], "extra": [9]}
    assert err.param == "controls"
    assert "'zone'" in err.hint and "{1: ..., 2: ..., 3: ...}" in err.hint
    p = check(err, "CONTROLS_KEYS_MISMATCH")
    assert p["got"] == {"missing": [3], "extra": [9]}


def test_rake_margins_disagree(s):
    err = raises(
        lambda: s.weighting.rake(controls={"b": {True: 3, False: 3}, "sex": {"M": 3, "F": 4}}),
        "MARGINS_DISAGREE",
    )
    assert err.got == {"b": 6.0, "sex": 7.0}
    assert "'b'" in err.hint and "shares=" in err.hint
    assert "Margins disagree" in err.detail


def test_adjust_unknown_statuses(s):
    err = raises(lambda: s.weighting.adjust("st"), "RESP_STATUS_UNKNOWN")
    assert err.got == {"Refused": 1, "dk": 2}
    assert err.expected == ["rr", "nr", "in", "uk"]
    assert "'Refused'" in err.hint and "'dk'" in err.hint and "resp_mapping=" in err.hint
    p = check(err, "RESP_STATUS_UNKNOWN")
    assert p["got"] == {"Refused": 1, "dk": 2}


def test_adjust_unknown_statuses_with_a_mapping(s):
    err = raises(
        lambda: s.weighting.adjust("st", resp_mapping={"rr": "rr", "nr": ["nr"]}),
        "RESP_STATUS_UNKNOWN",
    )
    assert err.param == "resp_mapping"
    assert err.expected == ["rr", "nr"]
    assert "'nr': ['nr', 'Refused', 'dk']" in err.hint


def test_adjust_unknown_int_statuses_are_native(s):
    df = _df().with_columns(pl.Series("code", [1, 2, 1, 7, 7, 1, 2, 1]))
    err = raises(
        lambda: Sample(df, Design(wgt="w")).weighting.adjust(
            "code", resp_mapping={"rr": 1, "nr": "2"}
        ),
        "RESP_STATUS_UNKNOWN",
    )
    assert err.got == {7: 2}
    assert json.loads(json.dumps(err.to_dict()))["error"]["got"] == {"7": 2}


def test_adjust_null_status(s):
    df = _df().with_columns(pl.Series("st2", ["rr", None, "nr", "rr"] * 2))
    err = raises(
        lambda: Sample(df, Design(wgt="w")).weighting.adjust("st2"), "RESP_STATUS_UNKNOWN"
    )
    assert err.got == {None: 2}
    json.dumps(err.to_dict())
    out = Sample(df, Design(wgt="w")).weighting.adjust(
        "st2", resp_mapping={"rr": "rr", "nr": "nr", "uk": None}
    )
    assert "nr_wgt" in out.data.columns


# ---------------------------------------------------------------------------
# Every other audited failure, one code each
# ---------------------------------------------------------------------------


def test_resp_mapping_key_invalid(s):
    err = raises(
        lambda: s.weighting.adjust("st", resp_mapping={"ok": "rr"}), "RESP_MAPPING_KEY_INVALID"
    )
    assert err.got == "ok" and err.expected == ["rr", "nr", "in", "uk"]


def test_resp_mapping_conflict(s):
    err = raises(
        lambda: s.weighting.adjust(
            "st", resp_mapping={"rr": ["rr", "dk"], "nr": ["nr", "Refused", "DK"]}
        ),
        "RESP_MAPPING_CONFLICT",
    )
    assert err.got == {"dk": ["rr", "nr"]}


@pytest.mark.parametrize(
    "ctl, bad",
    [
        ({1: -1, 2: 1, 3: 1}, {1: -1.0}),
        ({1: "x", 2: 1, 3: 1}, {1: "x"}),
        ({"1": float("nan"), 2: 1, 3: 1}, None),
    ],
)
def test_controls_value_invalid(s, ctl, bad):
    err = raises(lambda: s.weighting.poststratify(ctl, cells="zone"), "CONTROLS_VALUE_INVALID")
    if bad is not None:
        assert err.got == bad
    assert list(err.got) == [1]


def test_controls_value_invalid_rake_and_calibrate(s):
    raises(lambda: s.weighting.rake(controls={"sex": {"M": -2, "F": 1}}), "CONTROLS_VALUE_INVALID")
    err = raises(
        lambda: s.weighting.calibrate(controls={"x": float("inf")}), "CONTROLS_VALUE_INVALID"
    )
    # Calibration totals may be negative (a continuous auxiliary).
    s.weighting.calibrate(
        controls={"x": -1.0, Cat("sex"): {"M": 5, "F": 5}}, on_nonconvergence="warn"
    )
    assert err.param == "controls['x']"


def test_controls_all_zero(s):
    raises(
        lambda: s.weighting.poststratify(shares={"M": 0, "F": 0}, cells="sex"), "CONTROLS_ALL_ZERO"
    )
    raises(lambda: s.weighting.rake(controls={"sex": {"M": 0, "F": 0}}), "CONTROLS_ALL_ZERO")
    raises(lambda: s.weighting.standardize("sex", shares={"M": 0, "F": 0}), "CONTROLS_ALL_ZERO")


def test_controls_missing_and_conflict(s):
    raises(lambda: s.weighting.poststratify(cells="sex"), "CONTROLS_MISSING")
    raises(lambda: s.weighting.rake(), "CONTROLS_MISSING")
    raises(
        lambda: s.weighting.poststratify({"M": 1, "F": 1}, shares={"M": 1, "F": 1}, cells="sex"),
        "CONTROLS_CONFLICT",
    )
    raises(
        lambda: s.weighting.rake(
            controls={"sex": {"M": 1, "F": 1}}, shares={"sex": {"M": 1, "F": 1}}
        ),
        "CONTROLS_CONFLICT",
    )


def test_controls_type_invalid(s):
    raises(lambda: s.weighting.poststratify([1, 2], cells="sex"), "CONTROLS_TYPE_INVALID")
    err = raises(lambda: s.weighting.poststratify(shares=1, cells="sex"), "CONTROLS_TYPE_INVALID")
    assert "dict.fromkeys" in err.hint
    raises(lambda: s.weighting.rake(controls=[("sex", 1)]), "CONTROLS_TYPE_INVALID")
    raises(lambda: s.weighting.calibrate(controls=[1]), "CONTROLS_TYPE_INVALID")
    raises(lambda: s.weighting.calibrate(controls={"a": 1}, by="sex"), "CONTROLS_TYPE_INVALID")
    X = np.ones((8, 1))
    raises(
        lambda: s.weighting.calibrate_matrix(aux_vars=X, controls=[8.0], by="sex"),
        "CONTROLS_TYPE_INVALID",
    )


def test_controls_scalar_for_cells(s):
    err = raises(lambda: s.weighting.poststratify(100, cells="sex"), "CONTROLS_SCALAR_FOR_CELLS")
    assert "{'F': 100, 'M': 100}" in err.hint or "'M': 100" in err.hint
    raises(lambda: s.weighting.calibrate(controls={Cat("sex"): 10}), "CONTROLS_SCALAR_FOR_CELLS")


def test_cells_required(s):
    raises(lambda: s.weighting.poststratify(shares={"M": 1, "F": 1}), "CELLS_REQUIRED")
    raises(lambda: s.weighting.normalize({"M": 1, "F": 1}), "CELLS_REQUIRED")
    raises(lambda: s.weighting.standardize(None, shares={"M": 1}), "CELLS_REQUIRED")


@pytest.mark.parametrize(
    "call, param",
    [
        (lambda s: s.weighting.poststratify({"a": 1}, cells="nope"), "cells"),
        (lambda s: s.weighting.normalize(cells=["sex", "nope"]), "cells"),
        (lambda s: s.weighting.rake(controls={"nope": {"a": 1}}), "controls keys"),
        (lambda s: s.weighting.adjust("nope"), "resp_status"),
        (lambda s: s.weighting.adjust("st", cells="nope"), "cells"),
        (lambda s: s.weighting.trim(upper=5.0, by="nope"), "by"),
        (lambda s: s.weighting.calibrate(controls={"nope": 1.0}), "controls"),
        (lambda s: s.weighting.calibrate(controls={Cat("nope"): {"a": 1}}), "controls"),
        (
            lambda s: s.weighting.calibrate_matrix(
                aux_vars=np.ones((8, 1)), controls={"a": [1]}, by="nope"
            ),
            "by",
        ),
        (lambda s: s.weighting.control_aux_template(x=["x"], by="nope"), "by"),
        (lambda s: s.weighting.create_jk_wgts(psu="nope"), "psu"),
        (lambda s: s.weighting.standardize("nope", shares={"a": 1}), "cells"),
    ],
)
def test_missing_columns(s, call, param):
    err = raises(lambda: call(s), "MISSING_COLUMNS")
    assert err.param == param
    assert err.got == ["nope"]
    assert "zone" in err.expected
    assert not any(c.startswith("__svy") or c == "svy_row_index" for c in err.expected)
    assert "not found in data" in err.detail


def test_missing_column_hint_suggests_the_close_name(s):
    err = raises(lambda: s.weighting.poststratify({"a": 1}, cells="zones"), "MISSING_COLUMNS")
    assert "Did you mean 'zones' -> 'zone'?" in err.hint


def test_wgt_name_exists(s):
    err = raises(lambda: s.weighting.poststratify(10.0, wgt_name="x"), "WGT_NAME_EXISTS")
    assert err.got == "x" and "wgt_name='x_2'" in err.hint
    raises(
        lambda: s.weighting.rake(controls={"sex": {"M": 1, "F": 1}}, wgt_name="w"),
        "WGT_NAME_EXISTS",
    )
    raises(lambda: s.weighting.trim(upper=5.0, wgt_name="x"), "WGT_NAME_EXISTS")
    raises(lambda: s.weighting.calibrate(controls={"x": 30.0}, wgt_name="x"), "WGT_NAME_EXISTS")


def test_refused_trim_leaves_no_finding(s):
    # 8 units, below the default min_cell_size: trimming would report the skip.
    data, design, n = s.data, s.design, len(s.warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for kw in ({"wgt_name": "x"}, {"wgt_name": "x", "by": "zone", "min_cell_size": 1}):
            raises(lambda: s.weighting.trim(upper=5.0, **kw), "WGT_NAME_EXISTS")
        raises(lambda: s.weighting.trim(upper=5.0, wgt_name="x", inplace=True), "WGT_NAME_EXISTS")
    assert s.data.equals(data) and s.design == design and len(s.warnings) == n


def test_calibrate_checks_wgt_name_before_solving():
    # These controls cannot be met; the taken name is refused first.
    df = pl.DataFrame(
        {"c": ["a", "b"] * 6, "one": [1.0] * 12, "w": [1.0 + (i % 3) for i in range(12)]}
    )
    s = Sample(df, Design(wgt="w"))
    controls = {Cat("c"): {"a": 20, "b": 30}, "one": 60}
    raises(lambda: s.weighting.calibrate(controls=controls, wgt_name="one"), "WGT_NAME_EXISTS")
    raises(lambda: s.weighting.calibrate(controls=controls), "CALIBRATION_NOT_MET")


def test_no_weight():
    s = Sample(_df(), Design())
    for call in (
        lambda: s.weighting.poststratify(10.0),
        lambda: s.weighting.normalize(),
        lambda: s.weighting.rake(controls={"sex": {"M": 1, "F": 1}}),
        lambda: s.weighting.adjust("st"),
        lambda: s.weighting.standardize("sex", shares={"M": 1, "F": 1}),
        lambda: s.weighting.trim(upper=3.0),
        lambda: s.weighting.calibrate(controls={"x": 3.0}),
    ):
        err = raises(call, "WGT_MISSING")
        assert "Sample weight is None" in err.detail


def test_no_rows_in_scope_and_where_invalid(s):
    import svy

    nothing = svy.col("zone") > 100
    raises(lambda: s.weighting.poststratify(10.0, where=nothing), "NO_ROWS_IN_SCOPE")
    raises(
        lambda: s.weighting.rake(controls={"sex": {"M": 1, "F": 1}}, where=nothing),
        "NO_ROWS_IN_SCOPE",
    )
    raises(lambda: s.weighting.poststratify(10.0, where=svy.col("nope") > 1), "WHERE_INVALID")


def test_columns_empty(s):
    raises(lambda: s.weighting.poststratify(10.0, cells=[]), "COLUMNS_EMPTY")
    raises(lambda: s.weighting.control_aux_template(x=["x"], by=[]), "COLUMNS_EMPTY")


def test_param_renamed(s):
    err = raises(lambda: s.weighting.poststratify(10.0, by="sex"), "PARAM_RENAMED")
    assert (err.expected, err.got) == ("cells", "by")
    assert "`by=` was renamed to `cells=`" in err.detail


def test_convergence_failed(s):
    err = raises(
        lambda: s.weighting.rake(
            controls={"zone": {1: 10, 2: 20, 3: 10}, "sex": {"M": 25, "F": 15}},
            max_iter=1,
            tol=1e-15,
        ),
        "CONVERGENCE_FAILED",
    )
    assert err.got["max_iter"] == 1 and err.got["max_margin_error"] > 0
    assert err.expected == {"tol": 1e-15}
    assert "max_iter (now 1)" in err.hint


def test_trim_cycles_not_converged():
    df = pl.DataFrame({"c": ["a"] * 10 + ["b"] * 2, "w": [1.0] * 10 + [30.0, 1.0]})
    t = Sample(df, Design(wgt="w"))
    cfg = TrimConfig(upper=2.0, max_iter=2)
    raises(
        lambda: t.weighting.poststratify({"a": 10, "b": 30}, cells="c", trimming=cfg),
        "CONVERGENCE_FAILED",
    )
    raises(
        lambda: t.weighting.calibrate(controls={Cat("c"): {"a": 10, "b": 30}}, trimming=cfg),
        "CONVERGENCE_FAILED",
    )


def test_bounds_exceeded(s):
    err = raises(
        lambda: s.weighting.rake(controls={"sex": {"M": 50.0, "F": 30.0}}, bounds=(None, 1.2)),
        "BOUNDS_EXCEEDED",
    )
    assert err.param == "bounds"
    assert err.expected == {"bounds": [None, 1.2]}


def test_calibration_not_met(s):
    X = np.column_stack([np.ones(8), np.ones(8)])
    raises(
        lambda: s.weighting.calibrate_matrix(aux_vars=X, controls=[10.0, 20.0], weights_only=True),
        "CALIBRATION_NOT_MET",
    )


def test_cells_by_overlap_and_factor_conflict(s):
    err = raises(
        lambda: s.weighting.standardize("sex", shares={"M": 1, "F": 1}, by="sex"),
        "CELLS_BY_OVERLAP",
    )
    assert err.got == ["sex"]
    err = raises(lambda: s.weighting.normalize(factor=2.0, cells="sex"), "FACTOR_CONFLICT")
    assert err.got == ["cells"]


def test_calibration_terms(s):
    err = raises(
        lambda: s.weighting.calibrate(controls={Cat("zone", ref=9): {1: 1, 2: 1}}),
        "REF_LEVEL_UNKNOWN",
    )
    assert err.got == 9 and err.expected == [1, 2, 3]
    raises(lambda: s.weighting.control_aux_template(x=[]), "AUX_TERMS_EMPTY")
    raises(lambda: s.weighting.calibrate(controls={3.5: 1.0}), "TERM_INVALID")


def test_trim_inputs(s):
    err = raises(lambda: s.weighting.trim(), "TRIM_BOUNDS_MISSING")
    assert isinstance(err, ValueError)
    err = raises(lambda: s.weighting.trim(upper=-1.0, min_cell_size=1), "THRESHOLD_INVALID")
    assert isinstance(err, ValueError) and err.got == -1.0
    err = raises(lambda: s.weighting.trim(upper=True, min_cell_size=1), "THRESHOLD_INVALID")
    assert isinstance(err, TypeError)
    err = raises(lambda: TrimConfig(upper=2.0, min_cell_size=0), "INVALID_RANGE")
    assert isinstance(err, ValueError) and err.param == "min_cell_size"


def test_psu_required(s):
    for call in (
        s.weighting.create_jk_wgts,
        s.weighting.create_brr_wgts,
        lambda: s.weighting.create_bs_wgts(10),
    ):
        err = raises(call, "PSU_REQUIRED")
        assert "psu=" in err.hint


def test_data_nulls_are_structured():
    df = _df().with_columns(pl.Series("sex", ["M", None] * 4))
    t = Sample(df, Design(wgt="w"))
    err = raises(lambda: t.weighting.rake(controls={"sex": {"M": 1}}), "MARGIN_NA", DimensionError)
    assert err.got == {"sex": 4}
    err = raises(
        lambda: t.weighting.calibrate(controls={"M": {"x": 1}}, by="sex"), "BY_NA", DimensionError
    )
    assert err.got == {"sex": 4}
    raises(
        lambda: t.weighting.calibrate_matrix(
            aux_vars=np.ones((8, 1)), controls={"M": [1]}, by="sex"
        ),
        "BY_NA",
        DimensionError,
    )


def test_shape_mismatch(s):
    err = raises(
        lambda: s.weighting.calibrate_matrix(aux_vars=np.ones((5, 1)), controls=[1.0]),
        "SHAPE_MISMATCH",
        DimensionError,
    )
    assert err.expected == {"rows": 8, "cols": 1} and err.got == {"rows": 5, "cols": 1}
    err = raises(
        lambda: s.weighting.calibrate_matrix(aux_vars=np.ones((8, 1)), controls=[1.0, 2]),
        "SHAPE_MISMATCH",
        DimensionError,
    )
    assert err.got == {"totals": 2}


def test_replicate_psu_counts_are_structured():
    df = pl.DataFrame({"stratum": [1, 2, 2, 2, 2], "psu": [1, 2, 2, 3, 3], "wgt": [1.0] * 5})
    t = Sample(df, Design(wgt="wgt", stratum="stratum", psu="psu"))
    err = raises(t.weighting.create_brr_wgts, "ODD_PSU_COUNT", DimensionError)
    assert err.got == {1: 1}
    assert "1=1" in err.detail
    err = raises(
        lambda: t.weighting.create_jk_wgts(paired=True), "INSUFFICIENT_PSU", DimensionError
    )
    assert err.got == {1: 1}


def test_codes_are_stable_on_method_error(s):
    # WeightingError is a MethodError, so existing handlers still catch it.
    with pytest.raises(MethodError):
        s.weighting.poststratify({"x": 1}, cells="sex")
