# tests/svy/size/test_input_errors.py
"""Bad input to SampleSize raises an svy error that is also the builtin it replaces."""

from __future__ import annotations

import math

import pytest

import svy

from svy.errors import Code, SelectionError, SvyError, UsageError


S = svy.SampleSize


def _raises(area, kind, code, call, *, param=None, says=None):
    with pytest.raises(area) as err:
        call()
    e = err.value
    assert type(e) is getattr(area, kind.__name__)
    assert isinstance(e, kind) and isinstance(e, SvyError)
    assert e.code == code
    if param is not None:
        assert e.param == param
    if says is not None:
        assert says in e.detail
    return e


# ---- inputs that are not numbers -----------------------------------------------------


@pytest.mark.parametrize(
    "call, param",
    [
        (lambda: S().estimate_prop(p="a", moe=0.05), "p"),
        (lambda: S().estimate_prop(p=None, moe=0.05), "p"),
        (lambda: S().estimate_prop(p=0.5, moe="0.05"), "moe"),
        (lambda: S().estimate_mean(sigma="x", moe=1), "sigma"),
        (lambda: S().compare_props(p1="x", p2=0.4), "p1"),
        (lambda: S().compare_props(p1=0.5, p2=0.4, power=[0.8]), "power"),
        (lambda: S().compare_means(mu1="x", mu2=2, sigma1=1), "mu1"),
        (lambda: S().compare_means(mu1=1, mu2=2, sigma1=1, sigma2="y"), "sigma2"),
    ],
    ids=[
        "prop p str",
        "prop p None",
        "prop moe str",
        "mean sigma",
        "props p1",
        "props power list",
        "means mu1",
        "means sigma2",
    ],
)
def test_input_that_is_not_a_number(call, param):
    _raises(SelectionError, TypeError, Code.INVALID_TYPE, call, param=param)


def test_sigma2_may_be_left_out():
    # Equal SDs: n per group = 2 * (1.96 + 0.8416)^2 * 1 / 1^2 = 15.7, rounded up.
    assert S().compare_means(mu1=1, mu2=2, sigma1=1).n == (16.0, 16.0)


# ---- method names ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "call, param, allowed",
    [
        (
            lambda: S().estimate_prop(p=0.5, moe=0.05, method="wilson"),
            "method",
            ["wald", "fleiss"],
        ),
        (
            lambda: S().compare_props(p1=0.5, p2=0.4, var_mode="pooled"),
            "var_mode",
            ["alt-props", "pooled-prop", "max-var"],
        ),
    ],
    ids=["method", "var_mode"],
)
def test_unknown_option(call, param, allowed):
    e = _raises(SelectionError, ValueError, Code.INVALID_CHOICE, call, param=param)
    assert e.expected == allowed


@pytest.mark.parametrize(
    "call, param",
    [
        (lambda: S().estimate_prop(p=0.5, moe=0.05, method=3), "method"),
        (lambda: S().compare_props(p1=0.5, p2=0.4, var_mode=3), "var_mode"),
    ],
    ids=["method", "var_mode"],
)
def test_option_of_wrong_type(call, param):
    _raises(SelectionError, TypeError, Code.INVALID_TYPE, call, param=param)


def test_options_ignore_case_and_spaces():
    assert S().estimate_prop(p=0.5, moe=0.1, method=" WALD ").n == 97.0
    assert S().compare_props(p1=0.5, p2=0.4, var_mode="Max-Var").n is not None


# ---- Fleiss near its bound ------------------------------------------------------------


@pytest.mark.parametrize(
    "p, moe",
    [(0.99, 0.5), (0.5, 0.5), (0.99, 0.6), ({"a": 0.99, "b": 0.5}, {"a": 0.5, "b": 0.1})],
    ids=["near 1", "middle", "past bound", "one stratum"],
)
def test_fleiss_needs_moe_below_one_half(p, moe):
    e = _raises(
        SelectionError,
        ValueError,
        Code.INVALID_RANGE,
        lambda: S().estimate_prop(p=p, moe=moe, method="fleiss"),
        param="moe",
    )
    assert "moe < 0.5" in e.hint


def test_fleiss_just_below_the_bound():
    # p=0.9 and d=0.4999 sit in the first branch: f = 8d(1 - 2d).
    d, z = 0.4999, 1.959963984540054
    f = 8 * d * (1 - 2 * d)
    expected = math.ceil(f * z * z / (4 * d * d) + 1 / d - 2 * z * z + (z + 2) / f)
    assert S().estimate_prop(p=0.9, moe=d, method="fleiss").n == expected


# ---- per-stratum mappings -------------------------------------------------------------


def test_mappings_with_different_strata():
    e = _raises(
        UsageError,
        ValueError,
        Code.MAPPING_KEYS_MISMATCH,
        lambda: S().estimate_prop(p={"a": 0.5, "b": 0.4}, moe={"a": 0.05, "c": 0.05}),
        param="moe",
        says="missing: ['b']; extra: ['c']",
    )
    assert e.expected == ["a", "b"] and e.got == ["a", "c"]


def test_mapping_missing_a_stratum():
    _raises(
        UsageError,
        ValueError,
        Code.MAPPING_KEYS_MISMATCH,
        lambda: S().estimate_mean(sigma={"a": 1, "b": 2}, moe=0.5, deff={"a": 1.5}),
        param="deff",
        says="missing: ['b']",
    )


def test_empty_mapping():
    _raises(
        UsageError,
        ValueError,
        Code.MAPPING_KEYS_MISMATCH,
        lambda: S().estimate_prop(p={}, moe=0.05),
        param="p",
        says="'p' is an empty mapping",
    )


# ---- values that used to give a wrong size silently ----------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: S().estimate_prop(p=0.5, moe=0.05, pop_size=0),
        lambda: S().estimate_prop(p=0.5, moe=0.05, pop_size=-5),
        lambda: S().estimate_mean(sigma={"a": 1, "b": 1}, moe=0.5, pop_size={"a": 100, "b": 0}),
        lambda: S().compare_props(p1=0.5, p2=0.4, pop_size=(1000, 0)),
        lambda: S().compare_means(mu1=1, mu2=2, sigma1=1, pop_size=-1),
    ],
    ids=["prop zero", "prop negative", "mean one stratum", "props one group", "means"],
)
def test_population_size_must_be_positive(call):
    e = _raises(SelectionError, ValueError, Code.INVALID_RANGE, call, param="pop_size")
    assert "None skips the finite population correction" in e.hint


def test_population_size_still_corrects():
    # n0 = ceil(1.96^2 * 0.25 / 0.05^2) = 385; FPC: ceil(1000 * 385 / (1000 + 385 - 1)) = 279.
    assert S().estimate_prop(p=0.5, moe=0.05, pop_size=1000).n == 279.0


def test_a_stratum_without_population_size_is_not_corrected():
    out = S().estimate_prop(p={"a": 0.5, "b": 0.5}, moe=0.05, pop_size={"a": 1000, "b": None})
    assert out.n == {"a": 279.0, "b": 385.0}


@pytest.mark.parametrize("alloc_ratio", [0, -1, {"a": 1.0, "b": 0.0}], ids=["zero", "neg", "map"])
@pytest.mark.parametrize("goal", ["compare_props", "compare_means"])
def test_allocation_ratio_must_be_positive(goal, alloc_ratio):
    strat = isinstance(alloc_ratio, dict)
    args = {"p1": 0.5, "p2": 0.4} if goal == "compare_props" else {"mu1": 1, "mu2": 2, "sigma1": 1}
    if strat:
        args = {k: {"a": v, "b": v} for k, v in args.items()}
    _raises(
        SelectionError,
        ValueError,
        Code.INVALID_RANGE,
        lambda: getattr(S(), goal)(**args, alloc_ratio=alloc_ratio),
        param="alloc_ratio",
    )


@pytest.mark.parametrize("two_sides", ["no", 1, None], ids=["str", "int", "None"])
@pytest.mark.parametrize("goal", ["compare_props", "compare_means"])
def test_two_sides_must_be_a_bool(goal, two_sides):
    args = {"p1": 0.5, "p2": 0.4} if goal == "compare_props" else {"mu1": 1, "mu2": 2, "sigma1": 1}
    _raises(
        SelectionError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: getattr(S(), goal)(**args, two_sides=two_sides),
        param="two_sides",
    )


def test_one_sided_test_needs_fewer_units():
    # z_alpha drops from 1.96 to 1.645: n = ceil((1.645 + 0.8416)^2 * 2 / 1) = 13.
    assert S().compare_means(mu1=1, mu2=2, sigma1=1, two_sides=False).n == (13.0, 13.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: S().compare_props(p1=0.65, p2=0.85, two_sides=False, delta=0.2001),
        lambda: S().compare_means(mu1=1, mu2=2, sigma1=1, two_sides=False, delta=1.5),
        lambda: S().compare_means(
            mu1={"x": 1, "y": 1},
            mu2={"x": 2, "y": 2},
            sigma1=1,
            two_sides=False,
            delta={"x": 0.5, "y": 1.5},
        ),
    ],
    ids=["props", "means", "means by stratum"],
)
def test_one_sided_margin_above_the_effect_cannot_be_powered(call):
    _raises(
        SelectionError,
        ValueError,
        Code.INFEASIBLE_DESIGN,
        call,
        param="delta",
        says="does not exceed",
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: S().compare_props(p1=0.25, p2=0.75, two_sides=False, delta=0.5),
        lambda: S().compare_means(mu1=1, mu2=2, sigma1=1, two_sides=False, delta=1.0),
    ],
    ids=["props", "means"],
)
def test_one_sided_margin_equal_to_the_effect_is_unbounded(call):
    _raises(SelectionError, ValueError, Code.INFEASIBLE_DESIGN, call, says="unbounded")


@pytest.mark.parametrize(
    "delta, expected",
    [
        # n = ceil(2 * ((1.6449 + 0.8416) / (1 - delta))^2)
        (0.5, (50.0, 50.0)),  # superiority: 2 * 4.9729^2 = 49.46
        (-0.5, (6.0, 6.0)),  # non-inferiority: 2 * 1.6576^2 = 5.50
    ],
    ids=["superiority", "non-inferiority"],
)
def test_one_sided_margin_below_the_effect(delta, expected):
    assert S().compare_means(mu1=0, mu2=1, sigma1=1, two_sides=False, delta=delta).n == expected


def test_one_sided_without_margin_tests_in_the_direction_of_the_effect():
    assert S().compare_means(mu1=2, mu2=1, sigma1=1, two_sides=False).n == (13.0, 13.0)


@pytest.mark.parametrize("labels", [["a", "b", "c"], ["a"], []], ids=["three", "one", "empty"])
@pytest.mark.parametrize("goal", ["compare_props", "compare_means"])
def test_two_group_labels(goal, labels):
    args = {"p1": 0.5, "p2": 0.4} if goal == "compare_props" else {"mu1": 1, "mu2": 2, "sigma1": 1}
    e = _raises(
        SelectionError,
        ValueError,
        Code.GROUP_LABELS_LENGTH,
        lambda: getattr(S(), goal)(**args, group_labels=labels),
        param="group_labels",
        says=f"{len(labels)} names for 2 groups",
    )
    assert e.expected == 2


@pytest.mark.parametrize("labels", ["ab", ["a", 2]], ids=["str", "non-str label"])
def test_group_labels_of_wrong_type(labels):
    _raises(
        SelectionError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: S().compare_props(p1=0.5, p2=0.4, group_labels=labels),
        param="group_labels",
    )


def test_group_labels_reach_the_table():
    table = S().compare_props(p1=0.5, p2=0.4, group_labels=("treated", "control")).to_polars()
    assert sorted(table["group"].to_list()) == ["control", "treated"]


def test_to_polars_checks_its_own_labels():
    _raises(
        SelectionError,
        ValueError,
        Code.GROUP_LABELS_LENGTH,
        lambda: S().compare_props(p1=0.5, p2=0.4).to_polars(group_labels=["x"]),
        param="group_labels",
    )


@pytest.mark.parametrize(
    "min_n, kind, code",
    [
        (-1, ValueError, Code.INVALID_RANGE),
        (1.5, TypeError, Code.INVALID_TYPE),
        (True, TypeError, Code.INVALID_TYPE),
    ],
    ids=["negative", "fraction", "bool"],
)
def test_min_n(min_n, kind, code):
    _raises(
        SelectionError,
        kind,
        code,
        lambda: S().allocate(10, pop_size={"a": 10, "b": 20}, min_n=min_n),
        param="min_n",
    )


def test_min_n_zero_sets_no_floor():
    # Proportional: 10 * 2/302 = 0.07 for "a"; with no floor it may get 0 units.
    out = S().allocate(10, pop_size={"a": 2, "b": 300}, min_n=0)
    assert out.n == {"a": 0, "b": 10}
