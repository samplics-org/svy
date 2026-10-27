# tests/svy/regression/test_input_errors.py
"""Bad input to GLM fits, predictions and margins raises svy errors."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import Code, DataError, ModelError, SvyError
from svy.regression import GLM, GLMFit, GLMPred


N = 80


@pytest.fixture(scope="module")
def sample() -> svy.Sample:
    rng = np.random.default_rng(20261027)
    x = rng.normal(size=N)
    df = pl.DataFrame(
        {
            "x": x,
            "g": np.tile(["a", "b"], N // 2),
            "txt": ["t"] * N,
            "y": (x + rng.normal(size=N) > 0).astype(np.int64),
            "yc": 2 * x + rng.normal(size=N),
            "cnt": rng.poisson(3, N),
            "off": np.zeros(N),
            "w": np.ones(N),
            "s": np.repeat(["s1", "s2"], N // 2),
            "p": np.arange(N) // 2,
        }
    )
    return svy.Sample(df, svy.Design(stratum="s", psu="p", wgt="w"))


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


# ---- fit --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, param, kind, code",
    [
        ({"family": "logistic"}, "family", ValueError, Code.INVALID_CHOICE),
        ({"family": 3}, "family", TypeError, Code.INVALID_TYPE),
        ({"link": "logistic"}, "link", ValueError, Code.INVALID_CHOICE),
        ({"link": 3}, "link", TypeError, Code.INVALID_TYPE),
        ({"x": [3]}, "x", TypeError, Code.INVALID_TYPE),
        ({"x": ["x", None]}, "x", TypeError, Code.INVALID_TYPE),
        ({"max_iter": 0}, "max_iter", ValueError, Code.INVALID_RANGE),
        ({"max_iter": 1.5}, "max_iter", TypeError, Code.INVALID_TYPE),
        ({"tol": -1.0}, "tol", ValueError, Code.INVALID_RANGE),
        ({"tol": float("nan")}, "tol", ValueError, Code.INVALID_RANGE),
        ({"tol": "small"}, "tol", TypeError, Code.INVALID_TYPE),
    ],
    ids=[
        "family unknown",
        "family int",
        "link unknown",
        "link int",
        "x int",
        "x None",
        "max_iter 0",
        "max_iter float",
        "tol negative",
        "tol nan",
        "tol str",
    ],
)
def test_fit_option(sample, kw, param, kind, code):
    args = {"x": ["x"], "family": "binomial", **kw}
    _raises(ModelError, kind, code, lambda: sample.glm.fit("y", **args), param=param)


def test_family_names_keep_their_message(sample):
    _raises(
        ModelError,
        ValueError,
        Code.INVALID_CHOICE,
        lambda: sample.glm.fit("y", x=["x"], family="logistic"),
        says="Unknown family 'logistic'",
    )


def test_link_the_family_does_not_admit(sample):
    e = _raises(
        ModelError,
        ValueError,
        Code.LINK_NOT_ADMITTED,
        lambda: sample.glm.fit("y", x=["x"], family="binomial", link="inverse"),
        param="link",
        says="does not admit link 'inverse'",
    )
    assert "logit" in e.expected and "inverse" not in e.expected


def test_text_predictor_needs_cat(sample):
    e = _raises(
        DataError,
        TypeError,
        Code.COLUMN_NOT_NUMERIC,
        lambda: sample.glm.fit("y", x=["x", "g"], family="binomial"),
        param="x",
    )
    assert "svy.Cat('g')" in e.hint


def test_one_iteration_is_allowed(sample):
    with pytest.warns(svy.errors.ModelWarning, match="GLM_NOT_CONVERGED"):
        sample.glm.fit("y", x=["x"], family="binomial", max_iter=1)


@pytest.mark.parametrize(
    "offset, kind, code",
    [("nope", ValueError, Code.MISSING_COLUMNS), (3, TypeError, Code.INVALID_TYPE)],
    ids=["missing", "int"],
)
def test_offset(sample, offset, kind, code):
    _raises(
        ModelError,
        kind,
        code,
        lambda: sample.glm.fit("cnt", x=["x"], family="poisson", offset=offset),
        param="offset",
    )


# ---- predict ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(sample) -> GLM:
    return sample.glm.fit("y", x=["x", svy.Cat("g")], family="binomial")


@pytest.mark.parametrize("data", [{"x": [1.0]}, None], ids=["dict", "None"])
def test_predict_needs_a_frame(fitted, data):
    _raises(
        ModelError, TypeError, Code.INVALID_TYPE, lambda: fitted.predict(data), param="new_data"
    )


def test_predict_without_the_offset_column(sample):
    model = sample.glm.fit("cnt", x=["x"], family="poisson", offset="off")
    _raises(
        ModelError,
        ValueError,
        Code.PRED_COLUMN_MISSING,
        lambda: model.predict(pl.DataFrame({"x": [0.0, 1.0]})),
        param="offset",
        says="fitted with offset='off'",
    )


# ---- margins ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, param",
    [({"variables": ["nope"]}, "variables"), ({"at": {"nope": [1.0]}}, "at")],
    ids=["variables", "at"],
)
def test_margins_variable_not_in_the_model(fitted, kw, param):
    e = _raises(
        ModelError,
        ValueError,
        Code.MARGINS_UNKNOWN_VARIABLE,
        lambda: fitted.margins(**kw),
        param=param,
        says="Variable 'nope' not found in model",
    )
    assert "x" in e.expected


def test_margins_before_fitting(sample):
    _raises(
        ModelError,
        ValueError,
        Code.MODEL_NOT_FITTED,
        lambda: GLM(sample).margins(variables=["x"]),
    )


# ---- results ----------------------------------------------------------------------------


def test_exponentiate_needs_a_ratio_link(sample):
    linear = sample.glm.fit("yc", x=["x"]).fitted
    e = _raises(
        ModelError,
        ValueError,
        Code.EXPONENTIATE_NOT_RATIO,
        lambda: linear.to_polars(exponentiate=True),
        param="exponentiate",
        says="'identity' link",
    )
    assert e.expected == ["cloglog", "log", "logit"]


def test_term_test_needs_a_name(fitted):
    _raises(
        ModelError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: fitted.fitted.term_test(3),
        param="term",
    )


@pytest.mark.parametrize("cls", [GLM, GLMFit, GLMPred], ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "width, kind, code",
    [(20, ValueError, Code.INVALID_RANGE), ("wide", TypeError, Code.INVALID_TYPE)],
    ids=["too narrow", "text"],
)
def test_default_print_width(cls, width, kind, code):
    before = cls.PRINT_WIDTH
    _raises(ModelError, kind, code, lambda: cls.set_default_print_width(width), param="width")
    assert cls.PRINT_WIDTH == before
