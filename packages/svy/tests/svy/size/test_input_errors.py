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
