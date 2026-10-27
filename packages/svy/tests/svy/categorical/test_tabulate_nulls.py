"""tabulate with missing table values: rows leave the domain, their PSUs stay in the variance."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scipy.stats import t as t_dist

from svy import Design, Sample
from svy.errors import Code, DataError, EstimationError


REL = 1e-9


def _data() -> pl.DataFrame:
    """3 strata x 4 PSUs x 5 rows; every `a` is missing in PSUs 2 and 7, every `b` in PSU 11."""
    rng = np.random.default_rng(20261027)
    n = 60
    psu = np.repeat(np.arange(1, 13), 5)
    a = rng.choice(["x", "y", "z"], n)
    b = rng.choice(["p", "q"], n)
    w = rng.uniform(20, 80, n).round(3)
    a_miss = (rng.random(n) < 0.15) | np.isin(psu, [2, 7])
    b_miss = (rng.random(n) < 0.10) | (psu == 11)
    return pl.DataFrame(
        {
            "stratum": np.repeat([1, 2, 3], 20),
            "psu": psu,
            "w": w,
            "a": [None if m else v for v, m in zip(a, a_miss)],
            "b": [None if m else v for v, m in zip(b, b_miss)],
        }
    )


@pytest.fixture
def sample() -> Sample:
    return Sample(_data(), Design(stratum="stratum", psu="psu", wgt="w"))


def _cells(tbl) -> dict:
    return {(e.rowvar, e.colvar): e for e in tbl.estimates}


# R: des <- svydesign(ids=~psu, strata=~stratum, weights=~w, data=d)
#    svymean(~a, des, na.rm=TRUE); svytotal(~a, des, na.rm=TRUE); degf(subset(des, !is.na(a)))
R_PROP_A = {
    "x": (0.399425452038072, 0.0305118381348768),
    "y": (0.258774910192768, 0.0564973756765384),
    "z": (0.341799637769160, 0.0585452982225685),
}
R_TOTAL_A = {
    "x": (878.176, 155.212280018904),
    "y": (568.942, 171.098067296702),
    "z": (751.480, 184.048531615984),
}
R_DF_A = 7
# R: sub <- subset(des, !is.na(a) & !is.na(b))
#    svymean(~interaction(a, b), sub); svychisq(~a + b, sub, statistic="F")
R_PROP_AB = {
    ("x", "p"): (0.215106764379157, 0.0734846740883252),
    ("y", "p"): (0.133604342442716, 0.0536673909962892),
    ("z", "p"): (0.144286427094477, 0.0451817690394217),
    ("x", "q"): (0.204568928167357, 0.0750542225027774),
    ("y", "q"): (0.109603660727669, 0.0461583762910282),
    ("z", "q"): (0.192829877188624, 0.0792831487766521),
}
R_F_AB = (0.130548082139221, 1.53102785695295, 9.18616714171767, 0.826474994157136)


def test_one_way_proportions_match_r(sample):
    cells = _cells(sample.categorical.tabulate("a", drop_nulls=True))
    assert {k[0] for k in cells} == set(R_PROP_A)
    for level, (est, se) in R_PROP_A.items():
        cell = next(c for k, c in cells.items() if k[0] == level)
        assert cell.est == pytest.approx(est, rel=REL)
        assert cell.se == pytest.approx(se, rel=REL)


def test_one_way_counts_and_df_match_r(sample):
    tbl = sample.categorical.tabulate("a", units="count", drop_nulls=True)
    t_crit = t_dist.ppf(0.975, R_DF_A)
    for cell in tbl.estimates:
        est, se = R_TOTAL_A[cell.rowvar]
        assert cell.est == pytest.approx(est, rel=REL)
        assert cell.se == pytest.approx(se, rel=REL)
        assert (cell.uci - cell.est) / cell.se == pytest.approx(t_crit, rel=REL)


def test_two_way_cells_and_rao_scott_f_match_r(sample):
    tbl = sample.categorical.tabulate("a", "b", drop_nulls=True)
    cells = _cells(tbl)
    assert set(cells) == set(R_PROP_AB)
    for key, (est, se) in R_PROP_AB.items():
        assert cells[key].est == pytest.approx(est, rel=REL)
        assert cells[key].se == pytest.approx(se, rel=REL)
    f = tbl.stats.f
    assert (f.value, f.df_num, f.df_den, f.p_value) == pytest.approx(R_F_AB, rel=REL)


@pytest.mark.parametrize("replicate", [False, True])
def test_agrees_with_where_and_prop(sample, replicate):
    s = sample.weighting.create_jk_wgts() if replicate else sample
    method = "replication" if replicate else None
    plain = _cells(s.categorical.tabulate("a", method=method, drop_nulls=True))
    scoped = _cells(
        s.categorical.tabulate("a", where=pl.col("w") > 0, method=method, drop_nulls=True)
    )
    prop = {p.y_level: p for p in s.estimation.prop("a", method=method, drop_nulls=True).estimates}
    assert set(plain) == set(scoped)
    for key, cell in plain.items():
        assert (cell.est, cell.se) == pytest.approx((scoped[key].est, scoped[key].se), rel=REL)
        assert (cell.est, cell.se) == pytest.approx((prop[key[0]].est, prop[key[0]].se), rel=REL)


def test_missing_table_value_without_drop_nulls_raises(sample):
    with pytest.raises(DataError.ValueError) as exc:
        sample.categorical.tabulate("a", "b")
    err = exc.value
    assert err.code == Code.MISSING_VALUES
    assert err.got == ["a (NULL)", "b (NULL)"]
    assert err.hint == (
        "Set drop_nulls=True to exclude those rows from the estimate; "
        "they keep their place in the variance."
    )


def test_missing_table_value_inside_where_domain_raises(sample):
    with pytest.raises(DataError.ValueError) as exc:
        sample.categorical.tabulate("a", where=pl.col("stratum") == 1)
    assert exc.value.got == ["a (NULL)"]
    assert "inside the `where` domain" in str(exc.value)


@pytest.mark.parametrize("col", ["stratum", "psu", "w"])
@pytest.mark.parametrize("drop_nulls", [True, False])
@pytest.mark.parametrize("where", [None, pl.col("stratum") == 3])
def test_missing_design_value_raises(col, drop_nulls, where):
    # Rows 0 and 5 sit in PSUs 1 and 2 of stratum 1: no singleton, and outside `where`.
    df = _data().with_columns(
        pl.when(pl.int_range(pl.len()).is_in([0, 5])).then(None).otherwise(pl.col(col)).alias(col)
    )
    s = Sample(df, Design(stratum="stratum", psu="psu", wgt="w"))
    with pytest.raises(DataError.ValueError) as exc:
        s.categorical.tabulate("b", where=where, drop_nulls=drop_nulls)
    err = exc.value
    assert err.code == Code.MISSING_VALUES
    assert err.got == [f"{col} (NULL)"]
    assert "no place in the design" in err.hint


def test_every_table_value_missing_raises(sample):
    s = Sample(
        _data().with_columns(a=pl.lit(None, dtype=pl.String)),
        Design(stratum="stratum", psu="psu", wgt="w"),
    )
    with pytest.raises(EstimationError) as exc:
        s.categorical.tabulate("a", drop_nulls=True)
    assert "Every row is missing a table value." in str(exc.value)


def test_nan_in_a_float_table_variable_leaves_the_domain(sample):
    codes = {"x": 1.0, "y": 2.0, "z": 3.0}
    df = _data().with_columns(
        a_num=pl.col("a").replace_strict(codes, default=float("nan"), return_dtype=pl.Float64)
    )
    s = Sample(df, Design(stratum="stratum", psu="psu", wgt="w"))
    cells = s.categorical.tabulate("a_num", drop_nulls=True).estimates
    assert len(cells) == 3
    for cell, level in zip(sorted(cells, key=lambda c: float(c.rowvar)), "xyz"):
        assert (cell.est, cell.se) == pytest.approx(R_PROP_A[level], rel=REL)
