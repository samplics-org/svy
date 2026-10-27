import math

import polars as pl
import pytest

import svy


# Stratum "1" has no weight; stratum "2" is an ordinary domain.
_W = [0.0] * 6 + [1.0, 2.0, 1.5, 1.0, 2.0, 1.0]
_Y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]


def _sample(w=_W):
    df = pl.DataFrame(
        {
            "s": ["1"] * 6 + ["2"] * 6,
            "p": [str(i) for i in range(12)],
            "w": w,
            "y": _Y,
            "g": ["b", "a"] * 6,
            "r": ["a", "b"] * 6,
            "c": ["x", "x", "y", "y", "x", "y"] * 2,
        }
    )
    return svy.Sample(df, svy.Design(stratum="s", psu="p", wgt="w"))


def _row(frame, level):
    return frame.filter(pl.col("s") == level).row(0, named=True)


def test_tabulate_over_a_zero_weight_domain_gives_nan_tests():
    stats = _sample().categorical.tabulate(rowvar="r", colvar="c", where=svy.col("s") == "1").stats

    assert stats.chisq.df == 1
    assert math.isnan(stats.chisq.value) and math.isnan(stats.chisq.p_value)
    assert all(
        math.isnan(v) for v in (stats.f.value, stats.f.df_num, stats.f.df_den, stats.f.p_value)
    )


def test_tabulate_with_one_row_level_has_no_test():
    stats = _sample().categorical.tabulate(rowvar="r", colvar="c", where=svy.col("r") == "a").stats

    assert stats.chisq.df == 0
    assert math.isnan(stats.chisq.value) and math.isnan(stats.chisq.p_value)
    assert math.isnan(stats.f.value) and math.isnan(stats.f.p_value)


# R survey 4.5: d <- svydesign(ids=~p, strata=~s, weights=~w, data=df); sub <- subset(d, s == "2")
#   svyttest(y ~ 0, sub); svyttest(y ~ g, sub)
_R_ONE = {
    "diff": 4.8235294117647056,
    "t": 5.1968638444836097,
    "df": 4.0,
    "p_value": 0.0065301862359392927,
}
_R_TWO = {
    "diff": 0.13888888888888948,
    "t": 0.075635630802096981,
    "df": 4.0,
    "p_value": 0.94334078371936436,
}


@pytest.mark.parametrize(("group", "golden"), [(None, _R_ONE), ("g", _R_TWO)])
def test_ttest_zero_weight_by_group_is_a_nan_row(group, golden):
    out = _sample().categorical.ttest(y="y", group=group, by="s").to_polars()

    empty = _row(out, "1")
    assert all(math.isnan(empty[k]) for k in ("diff", "se", "t", "df", "p_value"))
    other = _row(out, "2")
    assert {k: other[k] for k in golden} == pytest.approx(golden, rel=1e-12)


def test_two_sample_nan_row_keeps_the_group_labels():
    out = str(_sample().categorical.ttest(y="y", group="g", by="s"))

    assert "['a' vs 'b']" in out


def test_ranktest_zero_weight_by_group_is_a_nan_row():
    s = _sample()
    by = s.categorical.ranktest(y="y", group="g", by="s")
    alone = s.categorical.ranktest(y="y", group="g", where=svy.col("s") == "2").to_polars()

    out = by.to_polars()
    assert all(math.isnan(_row(out, "1")[k]) for k in ("t", "df", "p_value"))
    assert {k: _row(out, "2")[k] for k in ("t", "df", "p_value")} == pytest.approx(
        alone.select("t", "df", "p_value").row(0, named=True), rel=1e-12
    )
    assert "['a' vs 'b']" in str(by)


@pytest.mark.parametrize(
    ("test", "message"),
    [
        ("ttest", "Two-sample t-test requires exactly 2 groups, found 1"),
        ("ranktest", "Rank test requires at least 2 groups with positive weight, found 1"),
    ],
)
def test_without_by_a_missing_group_still_raises(test, message):
    s = _sample(w=[0.0, 1.0] * 6)  # only group "a" carries weight

    with pytest.raises(RuntimeError, match=message):
        getattr(s.categorical, test)(y="y", group="g")
