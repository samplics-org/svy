# tests/svy/estimation/test_estimate_n.py
"""
Every estimate row carries ``n``: the records behind it.

``n`` counts the rows in the row's domain (``where=`` and ``by=`` level, or
t-test group) with a nonzero weight and every variable of the estimate present.
It is the kernel's domain count, the same one proportion CIs use (#175), so it
is computed once and never re-derived by callers. Zero-weight rows represent no
population units and are not counted; negative calibrated weights are.

The expected counts below are plain polars counts over the same rule.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl
import pytest

from polars.testing import assert_frame_equal

import svy

from svy.serialize import from_json, to_json, to_polars


def _data() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    n = 240
    w = rng.uniform(1, 5, n)
    w[[0, 5, 50, 100]] = 0.0
    y = rng.normal(10, 2, n).tolist()
    x = rng.uniform(1, 3, n).tolist()
    c = rng.choice(["p", "q", "t"], n).tolist()
    for i in (1, 2, 60, 61, 130):
        y[i] = None
    for i in (3, 70, 131):
        x[i] = None
    for i in (4, 71):
        c[i] = None
    return pl.DataFrame(
        {
            "st": np.repeat(["a", "b", "c"], n // 3),
            "psu": np.repeat(np.arange(24), 10),
            "w": w,
            "y": y,
            "x": x,
            "c": c,
            "g": rng.choice(["m", "f"], n),
            "r": rng.choice(["n", "s", "e"], n),
        }
    )


DATA = _data()


def _sample(kind: str) -> svy.Sample:
    s = svy.Sample(DATA, svy.Design(stratum="st", psu="psu", wgt="w"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "bootstrap":
            return s.weighting.create_bs_wgts(n_reps=20, rstate=1)
        if kind == "jackknife":
            return s.weighting.create_jk_wgts()
    return s


def _expected(variables, by=None, where=None) -> dict:
    d = DATA.filter(pl.col("w") != 0)
    for v in variables:
        d = d.filter(pl.col(v).is_not_null())
    if where is not None:
        d = d.filter(where)
    if by is None:
        return {None: d.height}
    return {k: v for k, v in d.group_by(by).len().iter_rows()}


def _rows(res):
    return res.estimates if hasattr(res, "estimates") else [p for m in res for p in m.estimates]


def _got(res) -> dict:
    got: dict = {}
    for p in _rows(res):
        got.setdefault(p.by_level[0] if p.by_level else None, set()).add(p.n)
    return got


ESTIMATORS = {
    "mean": (lambda E, **k: E.mean("y", drop_nulls=True, **k), ["y"]),
    "total": (lambda E, **k: E.total("y", drop_nulls=True, **k), ["y"]),
    "prop": (lambda E, **k: E.prop("c", drop_nulls=True, **k), ["c"]),
    "ratio": (lambda E, **k: E.ratio("y", "x", drop_nulls=True, **k), ["y", "x"]),
    "median": (lambda E, **k: E.median("y", drop_nulls=True, **k), ["y"]),
    "quantile": (lambda E, **k: E.quantile("y", p=[0.25, 0.75], drop_nulls=True, **k), ["y"]),
    "corr": (lambda E, **k: E.corr(["y", "x"], drop_nulls=True, **k), ["y", "x"]),
    "cov": (lambda E, **k: E.cov(["y", "x"], drop_nulls=True, **k), ["y", "x"]),
}

IN_M = pl.col("g") == "m"
DOMAINS = {
    "none": ({}, None, None),
    "by": ({"by": "r"}, "r", None),
    "where": ({"where": IN_M}, None, IN_M),
    "by_where": ({"by": "r", "where": IN_M}, "r", IN_M),
}


@pytest.mark.parametrize("kind", ["taylor", "bootstrap", "jackknife"])
@pytest.mark.parametrize("estimator", sorted(ESTIMATORS))
@pytest.mark.parametrize("domain", sorted(DOMAINS))
def test_n_counts_domain_records_with_weight_and_values(kind, estimator, domain):
    fn, variables = ESTIMATORS[estimator]
    kwargs, by, where = DOMAINS[domain]
    res = fn(_sample(kind).estimation, **kwargs)
    expected = _expected(variables, by, where)
    assert _got(res) == {k: {v} for k, v in expected.items()}


def test_prop_rows_carry_the_domain_count_not_the_category_count():
    res = _sample("taylor").estimation.prop("c", by="r", drop_nulls=True)
    expected = _expected(["c"], "r")
    per_category = (
        DATA.filter((pl.col("w") != 0) & pl.col("c").is_not_null()).group_by("r", "c").len()
    )
    assert per_category["len"].max() < min(expected.values())
    for p in res.estimates:
        assert p.n == expected[p.by_level[0]]


def test_each_variable_counts_its_own_missing_values():
    res = _sample("taylor").estimation.mean(["y", "x"], drop_nulls=True)
    got = {p.y: p.n for p in _rows(res)}
    assert got == {"y": _expected(["y"])[None], "x": _expected(["x"])[None]}
    assert got["y"] != got["x"]


def test_by_several_columns():
    res = _sample("taylor").estimation.mean("x", by=["r", "g"], drop_nulls=True)
    d = DATA.filter((pl.col("w") != 0) & pl.col("x").is_not_null())
    expected = {(r, g): k for r, g, k in d.group_by("r", "g").len().iter_rows()}
    rows = _rows(res)
    assert len(rows) == len(expected)
    assert sum(p.n for p in rows) == sum(expected.values())
    assert sorted(p.n for p in rows) == sorted(expected.values())


def test_where_and_drop_nulls_change_n():
    E = _sample("taylor").estimation
    everyone = E.mean("x", drop_nulls=True).estimates[0].n
    men = E.mean("x", where=IN_M, drop_nulls=True).estimates[0].n
    assert men < everyone
    assert men == _expected(["x"], where=IN_M)[None]
    # y has 5 missing values, x has 3.
    assert E.mean("y", drop_nulls=True).estimates[0].n == everyone - 2


def test_zero_weights_not_counted_negative_weights_counted():
    d = pl.DataFrame(
        {
            "psu": np.arange(12),
            "w": [1.0, 2.0, 0.0, 0.0, -0.5, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            "y": np.arange(12.0),
        }
    )
    s = svy.Sample(d, svy.Design(psu="psu", wgt="w"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert s.estimation.mean("y").estimates[0].n == 10
        assert s.estimation.total("y").estimates[0].n == 10


def test_n_follows_the_sample_after_wrangling():
    s = _sample("taylor")
    kept = s.wrangling.filter_records(svy.col("g") == "m")
    assert (
        kept.estimation.mean("x", drop_nulls=True).estimates[0].n
        == _expected(["x"], where=IN_M)[None]
    )
    blanked = s.wrangling.mutate(
        {"x": pl.when(pl.col("r") == "e").then(None).otherwise(pl.col("x"))}
    )
    expected = _expected(["x"], where=pl.col("r") != "e")[None]
    assert blanked.estimation.mean("x", drop_nulls=True).estimates[0].n == expected


# ---------------------------------------------------------------------------
# t-tests
# ---------------------------------------------------------------------------


def test_two_groups_each_report_their_own_count():
    rng = np.random.default_rng(0)
    d = pl.DataFrame(
        {
            "y": rng.normal(size=20),
            "w": np.ones(20),
            "g": ["a"] * 17 + ["b"] * 3,
            "psu": np.arange(20),
        }
    )
    res = svy.Sample(d, svy.Design(psu="psu", wgt="w")).categorical.ttest("y", group="g")
    assert [(e.group_level, e.n) for e in res.estimates] == [("a", 17), ("b", 3)]


def _ttest_got(res) -> dict:
    results = [res] if hasattr(res, "estimates") else list(res)
    got = {}
    for r in results:
        for e in r.estimates:
            key = tuple(v for v in (e.by_level, e.group_level) if v is not None)
            got[key[0] if len(key) == 1 else (key or None)] = e.n
    return got


def _ttest_expected(group, by=None, where=None):
    d = DATA.filter((pl.col("w") != 0) & pl.col("y").is_not_null())
    if where is not None:
        d = d.filter(where)
    keys = [k for k in (by, group) if k]
    if not keys:
        return {None: d.height}
    rows = d.group_by(keys).len().iter_rows()
    return {(r[0] if len(keys) == 1 else tuple(r[:-1])): r[-1] for r in rows}


@pytest.mark.parametrize("group", ["g", None])
@pytest.mark.parametrize("domain", ["none", "where", "by"])
def test_ttest_counts_zero_weights_and_nulls_out(group, domain):
    kwargs = {"none": {}, "where": {"where": pl.col("r") == "n"}, "by": {"by": "r"}}[domain]
    T = _sample("taylor").categorical
    if group:
        res = T.ttest("y", group=group, drop_nulls=True, **kwargs)
    else:
        res = T.ttest("y", mean_h0=10, drop_nulls=True, **kwargs)
    assert _ttest_got(res) == _ttest_expected(group, kwargs.get("by"), kwargs.get("where"))


# ---------------------------------------------------------------------------
# Tables and saved results
# ---------------------------------------------------------------------------


def test_to_polars_has_n_in_both_views():
    res = _sample("taylor").estimation.mean("y", by="r", drop_nulls=True)
    expected = _expected(["y"], "r")
    tidy = res.to_polars()
    assert dict(zip(tidy["r"], tidy["n"])) == expected
    assert tidy["n"].dtype.is_integer()
    assert "n" in res.to_polars(tidy=False).columns


def test_ttest_to_polars_has_n():
    res = _sample("taylor").categorical.ttest("y", group="g", drop_nulls=True)
    tidy = res.to_polars(component="estimates")
    assert dict(zip(tidy["g"], tidy["n"])) == _ttest_expected("g")
    assert "n" in res.to_polars(component="estimates", tidy=False).columns


def test_n_survives_a_json_round_trip():
    s = _sample("taylor")
    for res in (
        s.estimation.prop("c", by="r", drop_nulls=True),
        s.categorical.ttest("y", group="g", drop_nulls=True),
    ):
        back = from_json(to_json(res))
        assert [e.n for e in back.estimates] == [e.n for e in res.estimates]
        assert all(e.n is not None for e in back.estimates)


def test_saved_table_matches_the_live_table():
    res = _sample("bootstrap").estimation.ratio("y", "x", by="r", drop_nulls=True)
    assert_frame_equal(to_polars(from_json(to_json(res))), res.to_polars())


def test_payload_saved_without_n_still_loads():
    res = _sample("taylor").estimation.mean("y", by="r", drop_nulls=True)
    payload = json.loads(to_json(res))
    payload["schema_version"] = "svy-result/0.4"
    for row in payload["estimates"]:
        del row["n"]
    old = from_json(json.dumps(payload))
    assert all(e.n is None for e in old.estimates)
    assert "n" not in to_polars(old).columns


# ---------------------------------------------------------------------------
# Tables and GLM fits
# ---------------------------------------------------------------------------

TABLES = {
    "one_way": ({"rowvar": "c"}, ["c"], None),
    "two_way": ({"rowvar": "c", "colvar": "r"}, ["c", "r"], None),
    "one_way_where": ({"rowvar": "c", "where": IN_M}, ["c"], IN_M),
    "two_way_where": ({"rowvar": "c", "colvar": "r", "where": IN_M}, ["c", "r"], IN_M),
}


@pytest.mark.parametrize("kind", ["taylor", "bootstrap", "jackknife"])
@pytest.mark.parametrize("shape", sorted(TABLES))
def test_table_cells_carry_the_table_count(kind, shape):
    kwargs, variables, where = TABLES[shape]
    table = _sample(kind).categorical.tabulate(drop_nulls=True, **kwargs)
    expected = _expected(variables, where=where)[None]
    assert {c.n for c in table.estimates} == {expected}
    assert set(table.to_polars()["n"]) == {expected}


def test_table_n_survives_a_json_round_trip():
    table = _sample("taylor").categorical.tabulate(rowvar="c", colvar="r", drop_nulls=True)
    back = from_json(to_json(table))
    assert [c.n for c in back.estimates] == [c.n for c in table.estimates]
    assert_frame_equal(to_polars(back), table.to_polars())


def test_table_saved_without_n_still_loads():
    table = _sample("taylor").categorical.tabulate(rowvar="c", drop_nulls=True)
    payload = json.loads(to_json(table))
    for cell in payload["estimates"]:
        del cell["n"]
    old = from_json(json.dumps(payload))
    assert all(c.n is None for c in old.estimates)
    assert "n" not in to_polars(old).columns


@pytest.mark.parametrize("where", [None, IN_M])
def test_glm_n_follows_the_same_rule(where):
    fit = _sample("taylor").glm.fit(y="y", x=["x"], where=where, drop_nulls=True)
    assert fit.stats.n == _expected(["y", "x"], where=where)[None]
