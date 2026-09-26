# tests/svy/weighting/test_wgt_api_consistency.py
"""One shape across the weighting methods.

``bounds=(lo, hi)`` on the factor g, ``controls=`` everywhere,
``on_nonconvergence="error"|"warn"|"ignore"`` in place of ``strict``, renamed
parameters refused with PARAM_RENAMED, and every parameter documented.
"""

from __future__ import annotations

import inspect
import json
import re
import warnings

import numpy as np
import polars as pl
import pytest

from svy import Cat, Design, Sample, SvyUserWarning, TrimConfig, col
from svy.core.warnings import Severity, WarnCode
from svy.errors import MethodError, WeightingError
from svy.weighting.base import Weighting


MAXIT = WarnCode.MAX_ITER_REACHED
NOTMET = "CALIBRATION_NOT_MET"


def _svy(rec) -> list:
    return [r for r in rec if issubclass(r.category, SvyUserWarning)]


def _found(s: Sample, code) -> list:
    return s.warnings.list(code=code)


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
REGION = {"region": {"North": 50.0, "South": 30.0}}


@pytest.fixture
def tight() -> Sample:
    df = pl.DataFrame({"c": ["a"] * 10 + ["b"] * 2, "w": [1.0] * 10 + [30.0, 1.0]})
    return Sample(df, Design(wgt="w"))


@pytest.fixture
def tight_rr() -> Sample:
    df = pl.DataFrame(
        {"c": ["a"] * 10 + ["b"] * 2, "st": ["rr"] * 12, "w": [1.0] * 10 + [30.0, 1.0]}
    )
    return Sample(df, Design(wgt="w"))


@pytest.fixture
def two_domains() -> Sample:
    # Domain "A" cannot converge in one iteration; domain "B" has nothing to trim.
    df = pl.DataFrame({"dom": ["A"] * 12 + ["B"] * 10, "w": [1.0] * 10 + [30.0, 1.0] + [1.0] * 10})
    return Sample(df, Design(wgt="w"))


ONE_PASS = TrimConfig(upper=2.0, max_iter=1, tol=1e-20, min_cell_size=1)


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


# ---------------------------------------------------------------------------
# rake(bounds=)
# ---------------------------------------------------------------------------


def _g(out: Sample, before: Sample) -> np.ndarray:
    return out.data["rk_wgt"].to_numpy() / before.data["w"].to_numpy()


@pytest.mark.parametrize("bounds", [None, (None, None), [None, None]])
def test_rake_no_bounds_forms_agree(rk, bounds):
    ref = rk.weighting.rake(controls=REGION)
    out = rk.weighting.rake(controls=REGION, bounds=bounds)
    np.testing.assert_array_equal(out.data["rk_wgt"].to_numpy(), ref.data["rk_wgt"].to_numpy())


@pytest.mark.parametrize(
    "bounds",
    [
        (0.5, 2.0),
        [0.5, 2.0],
        (0.5, None),
        (None, 2.0),
        (np.float64(0.5), np.int64(2)),
        (0, 10),
        (0.75, 1.25),
    ],
)
def test_rake_bounds_that_hold_change_nothing(rk, bounds):
    # g is 50/40 = 1.25 in the North and 30/40 = 0.75 in the South
    ref = rk.weighting.rake(controls=REGION)
    out = rk.weighting.rake(controls=REGION, bounds=bounds)
    np.testing.assert_allclose(out.data["rk_wgt"].to_numpy(), ref.data["rk_wgt"].to_numpy())
    g = _g(out, rk)
    lo, hi = bounds
    assert lo is None or g.min() >= float(lo) - 1e-12
    assert hi is None or g.max() <= float(hi) + 1e-12


@pytest.mark.parametrize("bounds", [(None, 1.2), (0.8, None), (0.8, 1.2), (1.3, 2.0)])
@pytest.mark.parametrize("inplace", [False, True])
def test_rake_violated_bounds_raise_and_leave_sample(rk, bounds, inplace):
    data, design = rk.data, rk.design
    with pytest.raises(WeightingError) as ei:
        rk.weighting.rake(controls=REGION, bounds=bounds, inplace=inplace)
    err = ei.value
    assert err.code == "BOUNDS_EXCEEDED"
    assert err.param == "bounds"
    assert err.expected == {"bounds": [bounds[0], bounds[1]]}
    assert "checked after raking" in err.detail
    json.dumps(err.to_dict())
    assert rk.data.equals(data) and rk.design == design


def test_rake_bounds_checked_on_replicates_too(rk):
    from svy.core.repwgts import BootstrapWgts

    rng = np.random.default_rng(1)
    reps = {f"r{i}": 10.0 * rng.uniform(0.2, 1.8, 8) for i in range(1, 5)}
    s = Sample(
        rk.data.with_columns(**reps),
        Design(wgt="w", rep_wgts=BootstrapWgts(prefix="r", n_reps=4)),
    )
    # the main weight needs g in [0.75, 1.25]; the replicates need more
    s.weighting.rake(controls=REGION, bounds=(0.75, 1.25), ignore_reps=True)
    with pytest.raises(WeightingError) as ei:
        s.weighting.rake(controls=REGION, bounds=(0.75, 1.25))
    assert ei.value.code == "BOUNDS_EXCEEDED"


@pytest.mark.parametrize(
    "bounds, code, exc",
    [
        (1.2, "INVALID_TYPE", TypeError),
        ("0.5,2", "INVALID_TYPE", TypeError),
        ({"lo": 0.5}, "INVALID_TYPE", TypeError),
        ((), "INVALID_TYPE", TypeError),
        ((1.2,), "INVALID_TYPE", TypeError),
        ((0.5, 1.0, 2.0), "INVALID_TYPE", TypeError),
        (("a", 2.0), "INVALID_TYPE", TypeError),
        ((0.5, "2"), "INVALID_TYPE", TypeError),
        ((True, 2.0), "INVALID_TYPE", TypeError),
        ((0.5, False), "INVALID_TYPE", TypeError),
        ((2.0, 1.0), "INVALID_RANGE", ValueError),
        ((float("nan"), 2.0), "INVALID_RANGE", ValueError),
        ((0.5, float("inf")), "INVALID_RANGE", ValueError),
        ((float("-inf"), 2.0), "INVALID_RANGE", ValueError),
    ],
)
def test_rake_bounds_refused_with_guidance(rk, bounds, code, exc):
    with pytest.raises(WeightingError) as ei:
        rk.weighting.rake(controls=RK, bounds=bounds)
    err = ei.value
    assert isinstance(err, exc)
    assert err.code == code and err.param == "bounds"
    assert "bounds=(0.5, None)" in err.hint
    json.dumps(err.to_dict())


def test_rake_bounds_lo_above_hi_names_both(rk):
    with pytest.raises(WeightingError) as ei:
        rk.weighting.rake(controls=RK, bounds=(2, 1))
    assert "lo=2 above hi=1" in ei.value.detail


# ---------------------------------------------------------------------------
# calibrate(bounds=) / calibrate_matrix(bounds=): validated, then refused
# ---------------------------------------------------------------------------


def _cal_matrix(s, **k):
    X = np.column_stack([np.ones(12), s.data["x"].to_numpy()])
    return s.weighting.calibrate_matrix(aux_vars=X, controls=[40.0, 70.0], **k)


CAL_CALLS = {
    "calibrate": lambda s, **k: s.weighting.calibrate(controls={"x": 70.0}, **k),
    "calibrate_matrix": _cal_matrix,
}


@pytest.mark.parametrize("name", list(CAL_CALLS))
@pytest.mark.parametrize("bounds", [None, (None, None)])
def test_calibrate_unset_bounds_run(cal, name, bounds):
    out = CAL_CALLS[name](cal, bounds=bounds)
    assert "calib_wgt" in out.data.columns


@pytest.mark.parametrize("name", list(CAL_CALLS))
@pytest.mark.parametrize("bounds", [(0.5, 2.0), (0.5, None), (None, 2.0), [0.3, 2.3]])
@pytest.mark.parametrize("inplace", [False, True])
def test_calibrate_bounds_not_supported_yet(cal, name, bounds, inplace):
    data, design = cal.data, cal.design
    with pytest.raises(WeightingError) as ei:
        CAL_CALLS[name](cal, bounds=bounds, inplace=inplace)
    err = ei.value
    assert isinstance(err, NotImplementedError)
    assert err.code == "NOT_SUPPORTED" and err.param == "bounds"
    assert "trimming=" in err.hint
    json.dumps(err.to_dict())
    assert cal.data.equals(data) and cal.design == design


@pytest.mark.parametrize("name", list(CAL_CALLS))
@pytest.mark.parametrize("bounds", [1.0, (1.0,), (2.0, 1.0), ("a", None)])
def test_calibrate_bad_bounds_validated_before_refusal(cal, name, bounds):
    with pytest.raises(WeightingError) as ei:
        CAL_CALLS[name](cal, bounds=bounds)
    assert ei.value.code in ("INVALID_TYPE", "INVALID_RANGE")


def test_calibrate_where_bounds_refused(cal):
    with pytest.raises(WeightingError) as ei:
        cal.weighting.calibrate(controls={"x": 30.0}, where=col("d") == "d1", bounds=(0.5, 2))
    assert ei.value.code == "NOT_SUPPORTED"


def test_calibrate_matrix_controls_required(cal):
    X = np.ones((12, 1))
    with pytest.raises(WeightingError) as ei:
        cal.weighting.calibrate_matrix(aux_vars=X)
    assert ei.value.code == "CONTROLS_MISSING" and ei.value.param == "controls"


def test_calibrate_matrix_controls_by_domain(cal):
    X = np.ones((12, 1))
    out = cal.weighting.calibrate_matrix(aux_vars=X, controls={"d1": [10.0], "d2": [20.0]}, by="d")
    w = out.data.group_by("d").agg(pl.col("calib_wgt").sum()).sort("d")
    assert w["calib_wgt"].to_list() == pytest.approx([10.0, 20.0])


# ---------------------------------------------------------------------------
# Renamed parameters: refused, naming the replacement
# ---------------------------------------------------------------------------


def _renamed(call) -> WeightingError:
    with pytest.raises(WeightingError) as ei:
        call()
    err = ei.value
    assert err.code == "PARAM_RENAMED"
    json.dumps(err.to_dict())
    return err


@pytest.mark.parametrize(
    "kwargs, old, shown",
    [
        ({"ll_bound": 0.5}, "ll_bound", "bounds=(0.5, None)"),
        ({"up_bound": 2}, "up_bound", "bounds=(None, 2)"),
        ({"ll_bound": 0.5, "up_bound": 2.0}, "ll_bound", "bounds=(0.5, 2.0)"),
        ({"up_bound": None}, "up_bound", "bounds=(None, None)"),
    ],
)
def test_rake_old_bounds_refused(rk, kwargs, old, shown):
    err = _renamed(lambda: rk.weighting.rake(controls=RK, **kwargs))
    assert err.param == old and err.expected == "bounds"
    assert shown in err.hint


@pytest.mark.parametrize("name", list(CAL_CALLS))
@pytest.mark.parametrize("value", [True, False])
def test_calibrate_bounded_refused(cal, name, value):
    err = _renamed(lambda: CAL_CALLS[name](cal, bounded=value))
    assert err.param == "bounded" and err.expected == "bounds"
    assert "not supported yet" in err.hint


def test_calibrate_matrix_control_refused(cal):
    X = np.ones((12, 1))
    err = _renamed(lambda: cal.weighting.calibrate_matrix(aux_vars=X, control=[10.0]))
    assert err.param == "control" and err.expected == "controls"


STRICT_CALLS = {
    "poststratify": lambda s, **k: s.weighting.poststratify({"a": 10, "b": 30}, cells="c", **k),
    "rake": lambda s, **k: s.weighting.rake(controls={"c": {"a": 10, "b": 30}}, **k),
    "calibrate": lambda s, **k: s.weighting.calibrate(
        controls={Cat("c"): {"a": 10, "b": 30}}, **k
    ),
    "calibrate_matrix": lambda s, **k: s.weighting.calibrate_matrix(
        aux_vars=np.ones((12, 1)), controls=[40.0], **k
    ),
}


@pytest.mark.parametrize("name", list(STRICT_CALLS))
@pytest.mark.parametrize("value, mode", [(True, "error"), (False, "warn")])
def test_strict_refused_with_mapping(tight, name, value, mode):
    err = _renamed(lambda: STRICT_CALLS[name](tight, strict=value))
    assert err.param == "strict" and err.expected == "on_nonconvergence"
    assert f'on_nonconvergence="{mode}"' in err.hint


@pytest.mark.parametrize("name", list(STRICT_CALLS))
def test_unknown_kwarg_still_a_type_error(tight, name):
    with pytest.raises(TypeError, match="nope"):
        STRICT_CALLS[name](tight, nope=1)


def test_rename_refused_leaves_sample(tight):
    data, design = tight.data, tight.design
    with pytest.raises(WeightingError):
        STRICT_CALLS["rake"](tight, strict=False, inplace=True)
    assert tight.data.equals(data) and tight.design == design


# ---------------------------------------------------------------------------
# on_nonconvergence: error / warn / ignore on every method
# ---------------------------------------------------------------------------

CFG = TrimConfig(upper=2.0, max_iter=2)
X2 = np.column_stack([np.ones(12), np.ones(12)])

# name -> (fixture, call, error code, finding code, weight column)
NONCONV = {
    "rake": (
        "rk",
        lambda s, **k: s.weighting.rake(controls=RK, max_iter=3, tol=1e-20, **k),
        "CONVERGENCE_FAILED",
        MAXIT,
        "rk_wgt",
    ),
    "rake_trim_cycle": (
        "tight",
        lambda s, **k: s.weighting.rake(controls={"c": {"a": 10, "b": 30}}, trimming=CFG, **k),
        "CONVERGENCE_FAILED",
        MAXIT,
        "rk_wgt",
    ),
    "poststratify_trim_cycle": (
        "tight",
        lambda s, **k: s.weighting.poststratify({"a": 10, "b": 30}, cells="c", trimming=CFG, **k),
        "CONVERGENCE_FAILED",
        MAXIT,
        "ps_wgt",
    ),
    "calibrate_trim_cycle": (
        "tight",
        lambda s, **k: s.weighting.calibrate(
            controls={Cat("c"): {"a": 10, "b": 30}}, trimming=CFG, **k
        ),
        "CONVERGENCE_FAILED",
        MAXIT,
        "calib_wgt",
    ),
    "calibrate_where_trim_cycle": (
        "tight",
        lambda s, **k: s.weighting.calibrate(
            controls={Cat("c"): {"a": 10, "b": 30}}, where=col("w") > 0, trimming=CFG, **k
        ),
        "CONVERGENCE_FAILED",
        MAXIT,
        "calib_wgt",
    ),
    "calibrate_matrix_trim_cycle": (
        "tight",
        lambda s, **k: s.weighting.calibrate_matrix(
            aux_vars=np.column_stack(
                [(s.data["c"] == "a").to_numpy(), (s.data["c"] == "b").to_numpy()]
            ).astype(float),
            controls=[10.0, 30.0],
            trimming=CFG,
            **k,
        ),
        "CONVERGENCE_FAILED",
        MAXIT,
        "calib_wgt",
    ),
    "calibrate_not_met": (
        "cal",
        lambda s, **k: s.weighting.calibrate(
            controls={Cat("c"): {"a": 20, "b": 30}, "one": 60}, **k
        ),
        NOTMET,
        NOTMET,
        "calib_wgt",
    ),
    "calibrate_where_not_met": (
        "cal",
        lambda s, **k: s.weighting.calibrate(
            controls={Cat("c"): {"a": 10, "b": 15}, "one": 40}, where=col("d") == "d1", **k
        ),
        NOTMET,
        NOTMET,
        "calib_wgt",
    ),
    "calibrate_matrix_not_met": (
        "cal",
        lambda s, **k: s.weighting.calibrate_matrix(
            aux_vars=X2, controls={"d1": [10.0, 10.0], "d2": [10.0, 20.0]}, by="d", **k
        ),
        NOTMET,
        NOTMET,
        "calib_wgt",
    ),
    "standardize_trim_cycle": (
        "tight",
        lambda s, **k: s.weighting.standardize("c", shares={"a": 1, "b": 3}, trimming=CFG, **k),
        "CONVERGENCE_FAILED",
        MAXIT,
        "std_wgt",
    ),
    "adjust_trimming": (
        "tight_rr",
        lambda s, **k: s.weighting.adjust("st", trimming=ONE_PASS, **k),
        "CONVERGENCE_FAILED",
        MAXIT,
        "nr_wgt",
    ),
    "trim": (
        "tight",
        lambda s, **k: s.weighting.trim(upper=2.0, max_iter=1, tol=1e-20, min_cell_size=1, **k),
        "CONVERGENCE_FAILED",
        MAXIT,
        "trim_wgt",
    ),
    "trim_where": (
        "tight",
        lambda s, **k: s.weighting.trim(
            upper=2.0, max_iter=1, tol=1e-20, min_cell_size=1, where=col("w") > 0, **k
        ),
        "CONVERGENCE_FAILED",
        MAXIT,
        "trim_wgt",
    ),
    "trim_by": (
        "two_domains",
        lambda s, **k: s.weighting.trim(
            upper=2.0, by="dom", max_iter=1, tol=1e-20, min_cell_size=1, **k
        ),
        "CONVERGENCE_FAILED",
        MAXIT,
        "trim_wgt",
    ),
}


def _case(request, name):
    fixture, call, err_code, code, wgt = NONCONV[name]
    return request.getfixturevalue(fixture), call, err_code, code, wgt


@pytest.mark.parametrize("name", list(NONCONV))
@pytest.mark.parametrize("inplace", [False, True])
def test_on_nonconvergence_error(request, name, inplace):
    s, call, err_code, code, _ = _case(request, name)
    data, design, history = s.data, s.design, s.design_history
    with pytest.raises(WeightingError) as ei:
        call(s, inplace=inplace)
    assert ei.value.code == err_code
    assert 'on_nonconvergence="warn"' in ei.value.detail
    assert s.data.equals(data) and s.design == design
    assert s.design_history == history
    assert not _found(s, code)


@pytest.mark.parametrize("name", list(NONCONV))
def test_on_nonconvergence_error_is_the_default(request, name):
    s, call, err_code, _, _ = _case(request, name)
    with pytest.raises(WeightingError) as ei:
        call(s)
    assert ei.value.code == err_code


@pytest.mark.parametrize("name", list(NONCONV))
@pytest.mark.parametrize("inplace", [False, True])
def test_on_nonconvergence_warn(request, name, inplace):
    s, call, _, code, wgt = _case(request, name)
    with pytest.warns(SvyUserWarning, match=rf"\[{code}\]") as rec:
        out = call(s, on_nonconvergence="warn", inplace=inplace)
    raised = [r for r in _svy(rec) if f"[{code}]" in str(r.message)]
    assert len(raised) == 1 and raised[0].filename == __file__
    found = _found(out, code)
    assert len(found) == 1 and found[0].level == Severity.WARNING
    assert "on_nonconvergence" not in found[0].detail
    assert wgt in out.data.columns
    assert (out is s) is inplace
    if not inplace:
        assert not _found(s, code) and wgt not in s.data.columns


@pytest.mark.parametrize("name", list(NONCONV))
@pytest.mark.parametrize("inplace", [False, True])
def test_on_nonconvergence_ignore(request, name, inplace):
    s, call, _, code, wgt = _case(request, name)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = call(s, on_nonconvergence="ignore", inplace=inplace)
    assert not [r for r in _svy(rec) if f"[{code}]" in str(r.message)]
    found = _found(out, code)
    assert len(found) == 1 and found[0].level == Severity.INFO
    json.dumps(found[0].to_dict())
    assert wgt in out.data.columns
    assert (out is s) is inplace


@pytest.mark.parametrize("name", list(NONCONV))
def test_warn_and_ignore_keep_the_same_weights(request, name):
    s, call, _, code, wgt = _case(request, name)
    with pytest.warns(SvyUserWarning):
        warned = call(s, on_nonconvergence="warn")
    ignored = call(s, on_nonconvergence="ignore")
    np.testing.assert_array_equal(warned.data[wgt].to_numpy(), ignored.data[wgt].to_numpy())


@pytest.mark.parametrize("name", list(NONCONV))
@pytest.mark.parametrize("value", ["raise", "Error", "", None, True, 1])
def test_on_nonconvergence_bad_value(request, name, value):
    s, call, _, _, _ = _case(request, name)
    data = s.data
    with pytest.raises(MethodError) as ei:
        call(s, on_nonconvergence=value)
    err = ei.value
    assert err.code == "INVALID_CHOICE" and err.param == "on_nonconvergence"
    assert err.expected == ["error", "warn", "ignore"]
    assert s.data.equals(data)


def test_converged_run_records_nothing_in_any_mode(rk):
    for mode in ("error", "warn", "ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = rk.weighting.rake(controls=RK, on_nonconvergence=mode)
        assert not _found(out, MAXIT)


def test_weights_only_ignore_returns_array_quietly(cal):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w = cal.weighting.calibrate_matrix(
            aux_vars=X2, controls=[10.0, 20.0], weights_only=True, on_nonconvergence="ignore"
        )
    assert isinstance(w, np.ndarray) and w.shape == (12,)
    with pytest.raises(WeightingError) as ei:
        cal.weighting.calibrate_matrix(aux_vars=X2, controls=[10.0, 20.0], weights_only=True)
    assert ei.value.code == NOTMET


def test_ignored_finding_does_not_block_a_later_warn(rk):
    # INFO and WARNING records of one finding are separate entries
    out = rk.weighting.rake(controls=RK, max_iter=3, tol=1e-20, on_nonconvergence="ignore")
    with pytest.warns(SvyUserWarning, match=r"\[MAX_ITER_REACHED\]"):
        out = out.weighting.rake(
            controls=RK, max_iter=3, tol=1e-20, on_nonconvergence="warn", wgt_name="rk2"
        )
    levels = sorted(w.level for w in _found(out, MAXIT))
    assert levels == [Severity.INFO, Severity.WARNING]


# ---------------------------------------------------------------------------
# Docstrings: every public weighting method documents every parameter
# ---------------------------------------------------------------------------

DOCUMENTED = [
    "create_brr_wgts",
    "create_jk_wgts",
    "create_bs_wgts",
    "create_sdr_wgts",
    "adjust",
    "normalize",
    "poststratify",
    "standardize",
    "controls_margins_template",
    "rake",
    "control_aux_template",
    "build_aux_matrix",
    "calibrate",
    "calibrate_matrix",
    "trim",
]

CELLS_BY_RULE = (
    "``cells=`` are the classes the adjustment is computed over; ``by=`` runs the whole "
    "method separately within each domain."
)
ON_TRIAD = (
    '"error" raises and leaves the sample as it was; "warn" records the finding in '
    '``sample.warnings`` and raises it once as a ``SvyUserWarning``; "ignore" records it '
    "at INFO level without raising."
)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _documented_params(doc: str) -> set[str]:
    section = inspect.cleandoc(doc).split("Parameters\n----------\n", 1)[1]
    section = re.split(r"\n[A-Z][a-z]+\n-{3,}", section)[0]
    names: set[str] = set()
    for line in section.splitlines():
        m = re.match(r"^([A-Za-z_][\w, ]*?) :", line)
        if m:
            names.update(n.strip() for n in m.group(1).split(","))
    return names


def test_public_weighting_methods_are_all_checked():
    public = {
        n for n, v in vars(Weighting).items() if not n.startswith("_") and inspect.isfunction(v)
    }
    assert public == set(DOCUMENTED)


@pytest.mark.parametrize("name", DOCUMENTED)
def test_every_parameter_documented(name):
    fn = getattr(Weighting, name)
    doc = inspect.getdoc(fn)
    assert doc and "Parameters\n----------\n" in doc, name
    params = {
        p.name
        for p in inspect.signature(fn).parameters.values()
        if p.name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    }
    documented = _documented_params(doc)
    assert params - documented == set(), f"{name}: undocumented {params - documented}"
    assert documented - params == set(), f"{name}: documents unknown {documented - params}"


@pytest.mark.parametrize("name", DOCUMENTED)
def test_cells_by_rule_stated_where_either_is_taken(name):
    fn = getattr(Weighting, name)
    params = inspect.signature(fn).parameters
    if "cells" in params or "by" in params:
        assert CELLS_BY_RULE in _flat(fn.__doc__), name


@pytest.mark.parametrize("name", DOCUMENTED)
def test_on_triad_documented_identically(name):
    fn = getattr(Weighting, name)
    if "on_nonconvergence" in inspect.signature(fn).parameters:
        assert ON_TRIAD in _flat(fn.__doc__), name


@pytest.mark.parametrize(
    "name",
    ["poststratify", "rake", "calibrate", "calibrate_matrix", "standardize", "adjust", "trim"],
)
def test_no_strict_left(name):
    params = inspect.signature(getattr(Weighting, name)).parameters
    assert "strict" not in params and "on_nonconvergence" in params
    assert params["on_nonconvergence"].default == "error"


def test_rake_bounds_docstring_says_checked_not_enforced():
    doc = _flat(Weighting.rake.__doc__)
    assert "CHECKED after raking, not enforced" in doc
    assert "g = new weight / old weight" in doc


# ---------------------------------------------------------------------------
# trim / adjust / standardize specifics
# ---------------------------------------------------------------------------


def test_trim_error_names_only_the_failing_domain(two_domains):
    with pytest.raises(WeightingError) as ei:
        two_domains.weighting.trim(upper=2.0, by="dom", max_iter=1, tol=1e-20, min_cell_size=1)
    err = ei.value
    assert err.got["domains"] == ["A"] and err.got["max_iter"] == 1
    assert err.expected == {"tol": 1e-20}
    assert "max_iter (now 1)" in err.hint
    json.dumps(err.to_dict())


def test_trim_warn_records_one_finding_per_failing_domain(two_domains):
    with pytest.warns(SvyUserWarning, match=r"\[MAX_ITER_REACHED\]"):
        out = two_domains.weighting.trim(
            upper=2.0, by="dom", max_iter=1, tol=1e-20, min_cell_size=1, on_nonconvergence="warn"
        )
    found = _found(out, MAXIT)
    assert [w.got["domain"] for w in found] == ["A"]
    assert found[0].param == "max_iter"
    audits = _found(out, WarnCode.WEIGHT_ADJ_AUDIT)
    assert sorted(a.extra["domain"] for a in audits) == ["A", "B"]


def test_trim_converged_records_nothing_in_any_mode(tight):
    for mode in ("error", "warn", "ignore"):
        out = tight.weighting.trim(
            upper=2.0, max_iter=100, min_cell_size=1, on_nonconvergence=mode
        )
        assert not _found(out, MAXIT)


def test_trim_in_place_column_error_leaves_weight(tight):
    before = tight.data["w"].to_numpy().copy()
    with pytest.raises(WeightingError):
        tight.weighting.trim(
            upper=2.0, max_iter=1, tol=1e-20, min_cell_size=1, wgt_name=None, inplace=True
        )
    np.testing.assert_array_equal(tight.data["w"].to_numpy(), before)


def test_adjust_trimming_finding_names_the_config(tight_rr):
    with pytest.warns(SvyUserWarning, match=r"\[MAX_ITER_REACHED\]"):
        out = tight_rr.weighting.adjust("st", trimming=ONE_PASS, on_nonconvergence="warn")
    assert _found(out, MAXIT)[0].param == "trimming.max_iter"
    with pytest.raises(WeightingError) as ei:
        tight_rr.weighting.adjust("st", trimming=ONE_PASS)
    assert "trimming.max_iter (now 1)" in ei.value.hint


@pytest.mark.parametrize("update", [True, False])
def test_adjust_trimming_error_either_design_mode(tight_rr, update):
    data, design = tight_rr.data, tight_rr.design
    with pytest.raises(WeightingError) as ei:
        tight_rr.weighting.adjust("st", trimming=ONE_PASS, update_design_wgts=update, inplace=True)
    assert ei.value.code == "CONVERGENCE_FAILED"
    assert tight_rr.data.equals(data) and tight_rr.design == design


@pytest.mark.parametrize("name", ["standardize", "adjust", "trim"])
def test_new_on_nonconvergence_refuses_bad_value(tight_rr, name):
    calls = {
        "standardize": lambda **k: tight_rr.weighting.standardize(
            "c", shares={"a": 1, "b": 1}, **k
        ),
        "adjust": lambda **k: tight_rr.weighting.adjust("st", **k),
        "trim": lambda **k: tight_rr.weighting.trim(upper=2.0, **k),
    }
    with pytest.raises(MethodError) as ei:
        calls[name](on_nonconvergence="raise")
    assert ei.value.code == "INVALID_CHOICE"


# ---------------------------------------------------------------------------
# rake: trim-rake cycles are capped by trimming.max_iter
# ---------------------------------------------------------------------------

RAKE_TIGHT = {"c": {"a": 10, "b": 30}}


@pytest.mark.parametrize("cycles", [1, 2, 5])
@pytest.mark.parametrize("max_iter", [1, 2, 3])
def test_rake_cycles_follow_trimming_max_iter(rk, capsys, cycles, max_iter):
    # Two margins with a few IPF sweeps never meet tol=1e-20, so every allowed
    # cycle runs, however many sweeps each raking pass gets.
    rk.weighting.rake(
        controls=RK,
        max_iter=max_iter,
        tol=1e-20,
        trimming=TrimConfig(upper=12.0, max_iter=cycles),
        on_nonconvergence="ignore",
        display_iter=True,
    )
    out = capsys.readouterr().out
    assert len(re.findall(r"^Cycle +\d+ \|", out, flags=re.M)) == cycles


def test_rake_trim_cycle_error_reports_trimming_cap(tight):
    with pytest.raises(WeightingError) as ei:
        tight.weighting.rake(
            controls=RAKE_TIGHT, max_iter=50, trimming=TrimConfig(upper=2.0, max_iter=3)
        )
    err = ei.value
    assert err.code == "CONVERGENCE_FAILED"
    assert err.got["max_iter"] == 3 and err.got["trim_bounds_met"] is False
    assert "after 3 iterations" in err.detail
    assert "trimming.max_iter (now 3)" in err.hint


def test_rake_trim_cycle_finding_names_trimming_cap(tight):
    with pytest.warns(SvyUserWarning, match=r"Trim-rake cycle did not converge after 3 cycles"):
        out = tight.weighting.rake(
            controls=RAKE_TIGHT,
            max_iter=50,
            trimming=TrimConfig(upper=2.0, max_iter=3),
            on_nonconvergence="warn",
        )
    w = _found(out, MAXIT)[0]
    assert w.param == "trimming.max_iter" and w.got["max_iter"] == 3


def test_rake_without_trimming_keeps_its_own_cap(rk):
    with pytest.raises(WeightingError) as ei:
        rk.weighting.rake(controls=RK, max_iter=3, tol=1e-20)
    assert ei.value.got["max_iter"] == 3
    assert "Increase max_iter (now 3)" in ei.value.hint


def test_rake_small_max_iter_does_not_cut_trim_cycles(tight):
    # One IPF sweep per pass fits a single margin exactly, so a converging
    # trim-rake cycle is not limited by rake's max_iter.
    cfg = TrimConfig(upper=40.0, max_iter=10)
    a = tight.weighting.rake(controls=RAKE_TIGHT, max_iter=1, trimming=cfg)
    b = tight.weighting.rake(controls=RAKE_TIGHT, max_iter=100, trimming=cfg)
    np.testing.assert_allclose(a.data["rk_wgt"].to_numpy(), b.data["rk_wgt"].to_numpy())
