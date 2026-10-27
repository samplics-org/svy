# tests/svy/estimation/test_input_errors.py
"""Bad input to estimation raises an svy error that is also the builtin it replaces."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import (
    Code,
    DataError,
    DesignError,
    EstimationError,
    SvyError,
    UsageError,
)


N = 40


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(20261027)
    return pl.DataFrame(
        {
            "y": rng.normal(10, 2, N),
            "x": rng.uniform(1, 2, N),
            "yn": [None, *rng.normal(10, 2, N - 1)],
            "g": np.tile(["a", "b"], N // 2),
            "flag": np.tile([True, False, True, True], N // 4),
            "s": np.repeat(["s1", "s2"], N // 2),
            "p": np.arange(N) // 2,
            "w": np.ones(N),
        }
    )


@pytest.fixture(scope="module")
def sample(df) -> svy.Sample:
    return svy.Sample(df, svy.Design(stratum="s", psu="p", wgt="w"))


def _raises(area, kind, code, call, *, param=None, says=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", svy.SvyWarning)
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


# ---- columns ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "call, param",
    [
        (lambda e: e.mean("nope"), "y"),
        (lambda e: e.total("nope"), "y"),
        (lambda e: e.prop("nope"), "y"),
        (lambda e: e.median("nope"), "y"),
        (lambda e: e.quantile("nope", p=0.5), "y"),
        (lambda e: e.ratio("y", "nope"), "x"),
        (lambda e: e.mean("y", by="nope"), "by"),
        (lambda e: e.mean("y", by=["g", "nope"]), "by"),
        (lambda e: e.mean("y", where=svy.col("nope") > 1), "where"),
        (lambda e: e.mean("y", where={"nope": 1}), "where"),
        (lambda e: e.corr(("y", "nope")), "cols"),
        (lambda e: e.cov(("nope", "y")), "cols"),
    ],
    ids=[
        "mean",
        "total",
        "prop",
        "median",
        "quantile",
        "ratio x",
        "by",
        "by list",
        "where expr",
        "where dict",
        "corr",
        "cov",
    ],
)
def test_missing_column_names_the_parameter(sample, call, param):
    e = _raises(
        DataError, KeyError, Code.MISSING_COLUMNS, lambda: call(sample.estimation), param=param
    )
    assert e.got == ["nope"]
    available = e.extra["available_preview"]
    assert "y" in available and not [c for c in available if c.startswith("__svy_")]


def test_missing_group_column_in_ttest(sample):
    _raises(
        DataError,
        KeyError,
        Code.MISSING_COLUMNS,
        lambda: sample.categorical.ttest("y", group="nope"),
        param="group",
    )


@pytest.mark.parametrize(
    "call, param",
    [(lambda e: e.mean("g"), "y"), (lambda e: e.ratio("y", "g"), "x")],
    ids=["mean of text", "ratio over text"],
)
def test_text_column_where_a_number_is_needed(sample, call, param):
    e = _raises(
        DataError,
        TypeError,
        Code.COLUMN_NOT_NUMERIC,
        lambda: call(sample.estimation),
        param=param,
    )
    assert e.got == "String" and "'g'" in e.detail


def test_boolean_column_is_numeric(sample):
    # 3 of every 4 rows are True, equal weights.
    assert sample.estimation.mean("flag").estimates[0].est == pytest.approx(0.75)


# ---- missing values ----------------------------------------------------------------


def test_missing_values_name_the_column(sample):
    e = _raises(DataError, ValueError, Code.MISSING_VALUES, lambda: sample.estimation.mean("yn"))
    assert e.got == ["yn (NULL)"]
    assert "drop_nulls=True" in e.hint


def test_missing_values_inside_a_domain_say_so(sample):
    e = _raises(
        DataError,
        ValueError,
        Code.MISSING_VALUES,
        lambda: sample.estimation.mean("yn", where=svy.col("g") == "a"),
    )
    assert "inside the `where` domain" in e.detail


# ---- method --------------------------------------------------------------------------


def test_method_of_wrong_type(sample):
    _raises(
        UsageError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: sample.estimation.mean("y", method=1),
        param="method",
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.estimation.mean("y", method="nope"),
        lambda s: s.categorical.tabulate("g", method="nope"),
    ],
    ids=["mean", "tabulate"],
)
def test_unknown_method(sample, call):
    e = _raises(UsageError, ValueError, Code.INVALID_CHOICE, lambda: call(sample), param="method")
    assert e.got == "nope" and e.expected == ["taylor", "replication"]


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.estimation.mean("y", method="replication"),
        lambda s: s.categorical.tabulate("g", method="replication"),
        lambda s: s.glm.fit("y", x=["x"], method="replication"),
    ],
    ids=["mean", "tabulate", "glm"],
)
def test_replication_without_replicate_weights(sample, call):
    _raises(DesignError, ValueError, Code.REP_WGTS_MISSING, lambda: call(sample), param="method")


# ---- estimation's own options --------------------------------------------------------


@pytest.mark.parametrize("y", [3, None, ["y", 3]], ids=["int", "None", "list with int"])
def test_y_that_is_not_a_name(sample, y):
    _raises(
        EstimationError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: sample.estimation.mean(y),
        param="y",
    )


def test_ratio_x_that_is_not_a_name(sample):
    _raises(
        EstimationError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: sample.estimation.ratio(["y"], 3),
        param="x",
    )


@pytest.mark.parametrize(
    "p, says",
    [
        (1.5, "got 1.5"),
        (0, "got 0.0"),
        (1, "got 1.0"),
        ([0.25, 1.0], "got 1.0"),
        ([], "at least one probability"),
        ([0.5, 0.5], "duplicate probabilities"),
    ],
)
def test_invalid_probabilities(sample, p, says):
    e = _raises(
        EstimationError,
        ValueError,
        Code.QUANTILE_P_INVALID,
        lambda: sample.estimation.quantile("y", p=p),
        param="p",
        says=says,
    )
    assert "strictly between 0 and 1" in e.hint


@pytest.mark.parametrize("p", ["a", [0.5, "a"], True, None], ids=["str", "mixed", "bool", "None"])
def test_probabilities_of_wrong_type(sample, p):
    _raises(
        EstimationError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: sample.estimation.quantile("y", p=p),
        param="p",
    )


def test_q_method_of_wrong_type(sample):
    _raises(
        EstimationError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: sample.estimation.quantile("y", p=0.5, q_method=3),
        param="q_method",
    )


def test_unknown_q_method(sample):
    e = _raises(
        EstimationError,
        ValueError,
        Code.INVALID_CHOICE,
        lambda: sample.estimation.quantile("y", p=0.5, q_method="nope"),
        param="q_method",
    )
    assert e.expected == ["higher", "lower", "nearest", "linear", "middle"]


def test_unknown_ci_method(sample):
    e = _raises(
        EstimationError,
        ValueError,
        Code.INVALID_CHOICE,
        lambda: sample.estimation.prop("g", ci_method="nope"),
        param="ci_method",
    )
    assert e.got == "nope" and "logit" in e.expected


def test_prop_of_a_continuous_variable(sample):
    _raises(
        EstimationError,
        TypeError,
        Code.PROP_NOT_CATEGORICAL,
        lambda: sample.estimation.prop("y"),
        param="y",
        says="'y'",
    )


# ---- contrasts -----------------------------------------------------------------------


def test_estd_without_a_key():
    _raises(EstimationError, TypeError, Code.ESTD_KEY_MISSING, svy.estd)


@pytest.mark.parametrize(
    "call, says",
    [
        (lambda: svy.estd("a") + "b", "add a contrast expression and str"),
        (lambda: "b" * svy.estd("a"), "and str"),
        (lambda: svy.estd("a") / None, "and NoneType"),
        (lambda: svy.estd("a") - True, "and bool"),
    ],
    ids=["add str", "str times", "divide None", "subtract bool"],
)
def test_contrast_with_a_non_number(call, says):
    _raises(EstimationError, TypeError, Code.CONTRAST_OPERAND_INVALID, call, says=says)


# ---- printing options on a result ----------------------------------------------------


@pytest.fixture
def estimate(sample):
    return sample.estimation.mean("y")


@pytest.mark.parametrize(
    "call, kind, code, param",
    [
        (lambda e: e.set_print_width("wide"), TypeError, Code.INVALID_TYPE, "width"),
        (lambda e: e.set_print_width(20), ValueError, Code.INVALID_RANGE, "width"),
        (lambda e: type(e).set_default_print_width(5), ValueError, Code.INVALID_RANGE, "width"),
        (
            lambda e: setattr(e, "decimals", {"bad": 1}),
            ValueError,
            Code.INVALID_CHOICE,
            "decimals",
        ),
        (lambda e: setattr(e, "decimals", 1.5), TypeError, Code.INVALID_TYPE, "decimals"),
        (lambda e: setattr(e, "layout", "wide"), ValueError, Code.INVALID_CHOICE, "layout"),
    ],
    ids=["width text", "width 20", "class width", "decimals key", "decimals float", "layout"],
)
def test_print_options(estimate, call, kind, code, param):
    _raises(EstimationError, kind, code, lambda: call(estimate), param=param)


def test_print_width_boundary(estimate):
    assert estimate.set_print_width(21).print_width == 21
    assert estimate.set_print_width(None).print_width is None
