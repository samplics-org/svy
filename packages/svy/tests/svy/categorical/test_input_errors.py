# tests/svy/categorical/test_input_errors.py
"""Bad input to categorical analyses raises an svy error that is also the builtin it replaces."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.categorical import Table, TTestOneGroup, TTestTwoGroups
from svy.errors import Code, DataError, EstimationError, SvyError


N = 40


@pytest.fixture(scope="module")
def sample() -> svy.Sample:
    rng = np.random.default_rng(20261027)
    df = pl.DataFrame(
        {
            "y": rng.normal(size=N),
            "g": np.tile(["a", "b"], N // 2),
            "g4": np.tile(["a", "b", "c", "d"], N // 4),
            "c": np.tile(["p", "q"], N // 2),
            "s": np.repeat(["s1", "s2"], N // 2),
            "p": np.arange(N) // 2,
            "w": np.ones(N),
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


# ---- tabulate --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, param",
    [
        ({"rowvar": "nope"}, "rowvar"),
        ({"rowvar": "c", "colvar": "nope"}, "colvar"),
        ({"rowvar": "c", "where": svy.col("nope") > 1}, "where"),
        ({"rowvar": "c", "where": {"nope": 1}}, "where"),
    ],
    ids=["rowvar", "colvar", "where expr", "where dict"],
)
def test_tabulate_missing_column(sample, kw, param):
    e = _raises(
        DataError,
        KeyError,
        Code.MISSING_COLUMNS,
        lambda: sample.categorical.tabulate(**kw),
        param=param,
    )
    assert e.got == ["nope"]
    assert not [c for c in e.extra["available_preview"] if c.startswith("__svy_")]


@pytest.mark.parametrize(
    "kw, param",
    [({"rowvar": 3}, "rowvar"), ({"rowvar": "c", "colvar": ["g"]}, "colvar")],
    ids=["rowvar int", "colvar list"],
)
def test_tabulate_variable_that_is_not_a_name(sample, kw, param):
    _raises(
        EstimationError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: sample.categorical.tabulate(**kw),
        param=param,
    )


def test_unknown_units(sample):
    e = _raises(
        EstimationError,
        ValueError,
        Code.INVALID_CHOICE,
        lambda: sample.categorical.tabulate("c", units="share"),
        param="units",
    )
    assert e.expected == ["proportion", "percent", "count"]


def test_units_of_wrong_type(sample):
    _raises(
        EstimationError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: sample.categorical.tabulate("c", units=3),
        param="units",
    )


def test_units_aliases_still_work(sample):
    # 20 of 40 equally weighted rows are "p".
    table = sample.categorical.tabulate("c", units=" PCT ")
    assert table.estimates[0].est == pytest.approx(50.0)


# ---- t-test and rank test --------------------------------------------------------------


def test_ttest_group_with_more_than_two_levels(sample):
    e = _raises(
        EstimationError,
        RuntimeError,
        Code.TEST_GROUP_COUNT,
        lambda: sample.categorical.ttest("y", group="g4"),
        param="group",
        says="requires exactly 2 groups, found 4",
    )
    assert "where=" in e.hint


def test_ttest_domain_with_two_of_the_levels_runs(sample):
    r = sample.categorical.ttest("y", group="g4", where=svy.col("g4").is_in(["a", "b"]))
    assert 0.0 <= r.stats.p_value <= 1.0


@pytest.mark.parametrize("test", ["ttest", "ranktest"])
@pytest.mark.parametrize("alternative", ["nope", "Less", None], ids=["unknown", "case", "None"])
def test_unknown_alternative(sample, test, alternative):
    e = _raises(
        EstimationError,
        ValueError,
        Code.INVALID_CHOICE,
        lambda: getattr(sample.categorical, test)("y", group="g", alternative=alternative),
        param="alternative",
    )
    assert e.expected == ["two-sided", "less", "greater"]


def test_one_sided_p_values_still_split_the_two_sided_one(sample):
    two = sample.categorical.ttest("y", group="g").stats
    less = sample.categorical.ttest("y", group="g", alternative="less").stats
    greater = sample.categorical.ttest("y", group="g", alternative="greater").stats
    assert less.p_value + greater.p_value == pytest.approx(1.0)
    assert min(less.p_value, greater.p_value) == pytest.approx(two.p_value / 2)


def test_unknown_rank_score(sample):
    e = _raises(
        EstimationError,
        ValueError,
        Code.INVALID_CHOICE,
        lambda: sample.categorical.ranktest("y", group="g", score="wilcoxon"),
        param="score",
    )
    assert e.expected == ["kruskal-wallis", "vander-waerden", "median"]


def test_rank_score_of_wrong_type(sample):
    _raises(
        EstimationError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: sample.categorical.ranktest("y", group="g", score=3),
        param="score",
        says="'score'",
    )


# ---- results ---------------------------------------------------------------------------


@pytest.mark.parametrize("result", ["ttest one", "ttest two"], ids=["one group", "two groups"])
def test_unknown_frame_component(sample, result):
    r = {
        "ttest one": lambda: sample.categorical.ttest("y", mean_h0=0),
        "ttest two": lambda: sample.categorical.ttest("y", group="g"),
    }[result]()
    _raises(
        EstimationError,
        ValueError,
        Code.INVALID_CHOICE,
        lambda: r.to_polars(component="tests"),
        param="component",
    )


@pytest.mark.parametrize("cls", [Table, TTestOneGroup, TTestTwoGroups], ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "width, kind, code",
    [(20, ValueError, Code.INVALID_RANGE), ("wide", TypeError, Code.INVALID_TYPE)],
    ids=["too narrow", "text"],
)
def test_default_print_width(cls, width, kind, code):
    before = cls.PRINT_WIDTH
    _raises(EstimationError, kind, code, lambda: cls.set_default_print_width(width), param="width")
    assert cls.PRINT_WIDTH == before


@pytest.fixture
def table(sample) -> Table:
    return sample.categorical.tabulate("c", "g")


def test_table_print_width(table):
    _raises(
        EstimationError,
        ValueError,
        Code.INVALID_RANGE,
        lambda: setattr(table, "print_width", 5),
        param="width",
    )
    table.print_width = 21
    assert table.print_width == 21


@pytest.mark.parametrize(
    "call, kind, code",
    [
        (lambda t: setattr(t, "decimals", 1.5), TypeError, Code.INVALID_TYPE),
        (lambda t: t.set_decimals(2, bad=1), ValueError, Code.INVALID_CHOICE),
    ],
    ids=["decimals float", "unknown key"],
)
def test_table_decimals(table, call, kind, code):
    _raises(EstimationError, kind, code, lambda: call(table), param="decimals")


@pytest.mark.parametrize(
    "call, says",
    [
        (lambda t: setattr(t, "units", "count"), "cannot set 'units'"),
        (lambda t: delattr(t, "rowvar"), "cannot delete 'rowvar'"),
    ],
    ids=["set", "delete"],
)
def test_table_is_frozen(table, call, says):
    _raises(EstimationError, AttributeError, Code.TABLE_FROZEN, lambda: call(table), says=says)
    assert table.rowvar == "c"
