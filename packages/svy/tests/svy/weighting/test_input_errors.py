# tests/svy/weighting/test_input_errors.py
"""Bad input to weighting raises svy errors that are also the builtins they replace."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import Code, DataError, SvyError, UsageError, WeightingError


N = 40


@pytest.fixture
def sample() -> svy.Sample:
    rng = np.random.default_rng(20261027)
    df = pl.DataFrame(
        {
            "g": np.tile(["a", "b"], N // 2),
            "x": rng.uniform(1, 2, N),
            "txt": ["t"] * N,
            "w": rng.uniform(1, 3, N),
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


@pytest.mark.parametrize(
    "method, kw",
    [
        ("normalize", {"facter": 2.0}),
        ("poststratify", {"cels": "g"}),
        ("rake", {"max_iters": 10}),
        ("adjust", {"resp_status": "g", "cels": "g"}),
    ],
)
def test_unknown_argument_suggests_the_right_one(sample, method, kw):
    typo = list(kw)[-1]
    e = _raises(
        UsageError,
        TypeError,
        Code.UNKNOWN_ARGUMENT,
        lambda: getattr(sample.weighting, method)(**kw),
        says=f"{method}() got an unexpected keyword argument {typo!r}",
    )
    assert e.got == [typo] and "self" not in e.expected
    assert e.hint is not None and e.hint.startswith("Did you mean ")


def test_unknown_argument_without_a_close_name(sample):
    e = _raises(
        UsageError, TypeError, Code.UNKNOWN_ARGUMENT, lambda: sample.weighting.normalize(zz=1)
    )
    assert e.hint is None


def test_text_auxiliary_needs_cat(sample):
    e = _raises(
        DataError,
        TypeError,
        Code.COLUMN_NOT_NUMERIC,
        lambda: sample.weighting.calibrate(controls={"txt": 10.0}),
        param="controls",
    )
    assert "svy.Cat('txt')" in e.hint


@pytest.mark.parametrize("scale", ["a", ["a"] * N, {"x": 1}], ids=["str", "list of str", "dict"])
def test_calibration_scale_of_wrong_type(sample, scale):
    _raises(
        WeightingError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: sample.weighting.calibrate(controls={"x": 60.0}, scale=scale),
        param="scale",
    )


@pytest.mark.parametrize(
    "scale",
    [0, -1.0, float("nan"), float("inf"), [1.0] * (N - 1) + [0.0]],
    ids=["zero", "negative", "nan", "inf", "one row zero"],
)
def test_calibration_scale_not_positive(sample, scale):
    _raises(
        WeightingError,
        ValueError,
        Code.INVALID_RANGE,
        lambda: sample.weighting.calibrate(controls={"x": 60.0}, scale=scale),
        param="scale",
        says="must be finite and greater than 0",
    )


@pytest.mark.parametrize(
    "n_reps, kind, code",
    [
        (1, ValueError, Code.INVALID_RANGE),
        (0, ValueError, Code.INVALID_RANGE),
        (-5, ValueError, Code.INVALID_RANGE),
        ("5", TypeError, Code.INVALID_TYPE),
        (2.0, TypeError, Code.INVALID_TYPE),
        (True, TypeError, Code.INVALID_TYPE),
    ],
    ids=["one", "zero", "negative", "str", "float", "bool"],
)
@pytest.mark.parametrize("bs_kind", ["rao-wu", "poisson"])
def test_bootstrap_replicate_count(sample, n_reps, kind, code, bs_kind):
    _raises(
        WeightingError,
        kind,
        code,
        lambda: sample.weighting.create_bs_wgts(n_reps=n_reps, kind=bs_kind),
        param="n_reps",
    )


def test_two_bootstrap_replicates_is_the_minimum(sample):
    out = sample.weighting.create_bs_wgts(n_reps=2, rstate=1)
    assert out.design.rep_wgts.n_reps == 2
