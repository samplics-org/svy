"""T-tests and rank tests by several variables: one typed column per by variable."""

import itertools
import math

import numpy as np
import polars as pl
import pytest

import svy

from svy.serialize import from_json, to_json, to_polars


A, B, C = [1, 2], ["x", "z"], [False, True]


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(20261028)
    n = 800
    df = pl.DataFrame(
        {
            "str": rng.integers(1, 5, n),
            "psu": np.arange(n) // 10,
            "w": rng.uniform(1, 3, n),
            "y": rng.normal(size=n),
            "g": rng.choice([1, 2], n),
            "k": rng.choice([1, 2, 3], n),
            "a": rng.choice(A, n),
            "b": rng.choice(B, n),
            "c": rng.choice(C, n),
        }
    )
    s = svy.Sample(df, svy.Design(stratum="str", psu="psu", wgt="w"))
    s.set_value_labels("a", {1: "One", 2: "Two"})
    return s


CALLS = {
    "ttest_two": lambda s, by, **kw: s.categorical.ttest("y", group="g", by=by, **kw),
    "ttest_one": lambda s, by, **kw: s.categorical.ttest("y", mean_h0=0, by=by, **kw),
    "rank_two": lambda s, by, **kw: s.categorical.ranktest("y", group="g", by=by, **kw),
    "rank_k": lambda s, by, **kw: s.categorical.ranktest(
        "y", group="k", by=by, score="kruskal-wallis", **kw
    ),
}
BYS = {
    "two": (["a", "b"], {"a": pl.Int64, "b": pl.String}),
    "three": (["a", "b", "c"], {"a": pl.Int64, "b": pl.String, "c": pl.Boolean}),
}
LEVELS = {"a": A, "b": B, "c": C}


def _expected_levels(by):
    return set(itertools.product(*(LEVELS[v] for v in by)))


@pytest.mark.parametrize("call", list(CALLS))
@pytest.mark.parametrize("bys", list(BYS))
def test_test_frame_has_a_typed_column_per_by_variable(sample, call, bys):
    by, dtypes = BYS[bys]
    df = CALLS[call](sample, by).to_polars()
    assert {v: df.schema[v] for v in by} == dtypes
    assert set(df.select(by).rows()) == _expected_levels(by)


@pytest.mark.parametrize("call", ["ttest_two", "ttest_one"])
@pytest.mark.parametrize("bys", list(BYS))
def test_estimates_frame_has_a_typed_column_per_by_variable(sample, call, bys):
    by, dtypes = BYS[bys]
    df = CALLS[call](sample, by).to_polars("estimates")
    assert {v: df.schema[v] for v in by} == dtypes
    assert set(df.select(by).rows()) == _expected_levels(by)


@pytest.mark.parametrize("call", list(CALLS))
def test_untidy_frame_keys_the_levels_by_variable(sample, call):
    df = CALLS[call](sample, ["a", "b"]).to_polars(tidy=False)
    assert df["by"].to_list() == [["a", "b"]] * df.height
    assert {(r["a"], r["b"]) for r in df["by_level"].to_list()} == _expected_levels(["a", "b"])


@pytest.mark.parametrize("call", ["ttest_two", "ttest_one", "rank_two"])
def test_result_rows_carry_the_levels_as_a_tuple(sample, call):
    r = CALLS[call](sample, ["a", "b"])
    diffs = [d for m in r for d in m.diff]
    assert all(d.by == ("a", "b") for d in diffs)
    assert {d.by_level for d in diffs} == _expected_levels(["a", "b"])


def test_k_sample_result_keeps_the_levels_as_tuples(sample):
    r = CALLS["rank_k"](sample, ["a", "b"])
    assert set(r.by_levels) == _expected_levels(["a", "b"])


@pytest.mark.parametrize("tidy", [True, False])
def test_k_sample_frame_names_its_domain_with_one_by_variable(sample, tidy):
    df = CALLS["rank_k"](sample, "a").to_polars(tidy=tidy)
    if tidy:
        assert df.schema["a"] == pl.Int64
        assert sorted(df["a"].to_list()) == A
    else:
        assert df["by"].to_list() == [["a"]] * len(A)
        assert sorted(lv["a"] for lv in df["by_level"].to_list()) == A
    assert df.columns[0] == "y_var"


@pytest.mark.parametrize("call", ["ttest_two", "ttest_one", "rank_two"])
def test_each_by_cell_equals_its_where_subset(sample, call):
    df = CALLS[call](sample, ["a", "b"]).to_polars()
    for a, b in _expected_levels(["a", "b"]):
        sub = CALLS[call](sample, None, where=(svy.col("a") == a) & (svy.col("b") == b))
        got = df.filter((pl.col("a") == a) & (pl.col("b") == b)).row(0, named=True)
        assert math.isclose(got["est"], sub.diff[0].diff, rel_tol=1e-12)
        assert math.isclose(got["se"], sub.diff[0].se, rel_tol=1e-12)


@pytest.mark.parametrize("call", list(CALLS))
def test_print_names_each_by_variable_with_its_label(sample, call):
    out = CALLS[call](sample, ["a", "b"]).__plain_str__()
    assert "a = One, b = x" in out
    assert "a = Two, b = z" in out


@pytest.mark.parametrize("call", ["ttest_two", "ttest_one"])
@pytest.mark.parametrize(
    "opts",
    [{}, {"tidy": False}, {"component": "estimates"}, {"component": "estimates", "tidy": False}],
)
def test_saved_result_gives_the_live_frame(sample, call, opts):
    r = CALLS[call](sample, ["a", "b", "c"])
    back = to_polars(from_json(to_json(r)), **opts)
    assert back.equals(r.to_polars(**opts))
