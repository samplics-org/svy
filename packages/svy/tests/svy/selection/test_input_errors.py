# tests/svy/selection/test_input_errors.py
"""Bad input to selection raises an svy error that is also the builtin it replaces."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import Code, SelectionError, SvyError, UsageError


N = 30


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": np.arange(N),
            "g": np.repeat(["a", "b", "c"], 10),
            "s": np.tile(["s1", "s2"], N // 2),
            "mos": np.arange(1, N + 1).astype(float),
        }
    )


def _sample(df, **design) -> svy.Sample:
    return svy.Sample(df, svy.Design(**design))


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


DRAWS = {
    "srs": lambda s, **kw: s.sampling.srs(**kw),
    "pps_sys": lambda s, **kw: s.sampling.pps_sys(**kw),
    "pps_wr": lambda s, **kw: s.sampling.pps_wr(**kw),
    "pps_brewer": lambda s, **kw: s.sampling.pps_brewer(**kw),
    "pps_rs": lambda s, **kw: s.sampling.pps_rs(**kw),
}


# ---- n ---------------------------------------------------------------------------


@pytest.mark.parametrize("draw", list(DRAWS))
@pytest.mark.parametrize(
    "n, says",
    [
        (-1, "got -1"),
        (1.5, "got 1.5"),
        (float("nan"), "got nan"),
        (float("inf"), "got inf"),
        ({"a": 2, "b": -1, "c": 2}, "got -1"),
        ({"a": 2, "b": 2.5, "c": 2}, "got 2.5"),
    ],
    ids=["negative", "fraction", "nan", "inf", "map negative", "map fraction"],
)
def test_n_that_is_not_a_whole_count(df, draw, n, says):
    s = _sample(df, mos="mos")
    by = "g" if isinstance(n, dict) else None
    _raises(
        SelectionError,
        ValueError,
        Code.SELECTION_N_INVALID,
        lambda: DRAWS[draw](s, n=n, by=by),
        param="n",
        says=says,
    )


@pytest.mark.parametrize(
    "n", ["3", None, True, {"a": "2", "b": 2, "c": 2}], ids=["str", "None", "bool", "map str"]
)
def test_n_of_wrong_type(df, n):
    by = "g" if isinstance(n, dict) else None
    _raises(
        SelectionError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: _sample(df).sampling.srs(n=n, by=by),
        param="n",
    )


@pytest.mark.parametrize("n", [0, 2, 2.0, np.int64(2), {"a": 1, "b": 0, "c": 2.0}])
def test_whole_counts_of_any_numeric_type_draw(df, n):
    out = _sample(df).sampling.srs(n=n, by="g" if isinstance(n, dict) else None, rstate=1)
    expected = 3 if isinstance(n, dict) else int(n)
    assert out.data["svy_prob_selection"].drop_nulls().len() == expected


@pytest.mark.parametrize(
    "design, by, n, group, available",
    [
        ({}, None, 31, "__all__", 30),
        ({}, "g", 11, "a", 10),
        ({"stratum": "s"}, None, 16, "s1", 15),
    ],
    ids=["frame", "by group", "stratum"],
)
def test_n_beyond_the_group_without_replacement(df, design, by, n, group, available):
    e = _raises(
        SelectionError,
        ValueError,
        Code.SELECTION_N_EXCEEDS_GROUP,
        lambda: _sample(df, **design).sampling.srs(n=n, by=by),
        param="n",
        says=f"requested n={n} exceeds the available population of {available}",
    )
    assert e.got == {"group": group, "n": n, "available": available}
    assert "wr=True" in e.hint


def test_n_equal_to_the_group_draws_every_unit(df):
    out = _sample(df).sampling.srs(n=10, by="g", rstate=1)
    assert out.data["svy_prob_selection"].to_list() == [1.0] * N


# ---- columns ---------------------------------------------------------------------


@pytest.mark.parametrize("draw", list(DRAWS))
@pytest.mark.parametrize(
    "kw, param",
    [
        ({"by": "nope"}, "by"),
        ({"by": ["g", "nope"]}, "by"),
        ({"where": svy.col("nope") > 1}, "where"),
        ({"where": {"nope": 1}}, "where"),
    ],
    ids=["by", "by list", "where expr", "where dict"],
)
def test_missing_column_names_the_parameter(df, draw, kw, param):
    e = _raises(
        SelectionError,
        KeyError,
        Code.MISSING_COLUMNS,
        lambda: DRAWS[draw](_sample(df, mos="mos"), n=2, **kw),
        param=param,
    )
    assert e.got == ["nope"]
    available = e.extra["available_preview"]
    assert "g" in available and not [c for c in available if c.startswith("__svy_")]


@pytest.mark.parametrize("draw", ["srs", "pps_sys"])
def test_missing_order_by_column(df, draw):
    _raises(
        SelectionError,
        KeyError,
        Code.MISSING_COLUMNS,
        lambda: DRAWS[draw](_sample(df, mos="mos"), n=2, order_by="nope"),
        param="order_by",
    )


# ---- order_type and rstate -----------------------------------------------------------


@pytest.mark.parametrize("draw", ["srs", "pps_sys"])
def test_unknown_order_type(df, draw):
    e = _raises(
        SelectionError,
        ValueError,
        Code.INVALID_CHOICE,
        lambda: DRAWS[draw](_sample(df, mos="mos"), n=2, order_by="id", order_type="up"),
        param="order_type",
    )
    assert e.got == "up" and e.expected == ["ascending", "descending", "random"]


@pytest.mark.parametrize("rstate", ["x", 1.5, [1, 2]], ids=["str", "float", "list"])
def test_rstate_of_wrong_type(df, rstate):
    _raises(
        UsageError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: _sample(df).sampling.srs(n=2, rstate=rstate),
        param="rstate",
    )


def test_negative_seed(df):
    _raises(
        UsageError,
        ValueError,
        Code.INVALID_RANGE,
        lambda: _sample(df).sampling.srs(n=2, rstate=-1),
        param="rstate",
    )


@pytest.mark.parametrize(
    "rstate",
    [0, np.int64(7), np.random.default_rng(7), np.random.SeedSequence(7)],
    ids=["zero", "numpy int", "Generator", "SeedSequence"],
)
def test_accepted_seeds_draw(df, rstate):
    out = _sample(df).sampling.srs(n=4, rstate=rstate)
    assert out.data["svy_prob_selection"].drop_nulls().len() == 4
