# tests/svy/categorical/test_rank_native_levels.py
"""Rank-test and t-test levels keep the source column's type.

The kernel groups on strings; ``by`` and group levels are mapped back to the
column's own values, and printed with bools in lowercase.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample


N = 240


@pytest.fixture(scope="module")
def sample() -> Sample:
    rng = np.random.default_rng(20260925)
    df = pl.DataFrame(
        {
            "stratum": np.repeat(["s1", "s2"], N // 2),
            "psu": np.repeat(np.arange(24), N // 24),
            "w": rng.uniform(1, 3, N),
            "y": rng.normal(10, 2, N),
            "zone": rng.integers(1, 4, N),
            "b": rng.integers(0, 2, N).astype(bool),
            "g": rng.choice(["a", "b"], N),
        }
    )
    return Sample(df, Design(stratum="stratum", psu="psu", wgt="w"))


def _plain(result) -> str:
    return result.__plain_str__()


class TestRankTestLevels:
    def test_two_sample_groups(self, sample):
        r = sample.categorical.ranktest("y", group="b", method="kruskal-wallis")
        assert r.groups.levels == (False, True)
        assert all(type(v) is bool for v in r.groups.levels)
        assert "[false vs true]" in _plain(r)

    def test_k_sample_groups(self, sample):
        r = sample.categorical.ranktest("y", group="zone", method="kruskal-wallis")
        assert r.group_levels == [1, 2, 3]
        assert all(type(v) is int for v in r.group_levels)

    def test_two_sample_by(self, sample):
        r = sample.categorical.ranktest("y", group="b", by="zone", method="kruskal-wallis")
        assert sorted(r.by_levels) == [1, 2, 3]
        assert all(type(v) is int for v in r.by_levels)
        assert [x.diff[0].by_level for x in r] == r.by_levels
        assert all(x.groups.levels == (False, True) for x in r)
        assert r.to_polars()["zone"].dtype == pl.Int64

    def test_k_sample_by_bool(self, sample):
        r = sample.categorical.ranktest("y", group="zone", by="b", method="median")
        assert sorted(r.by_levels) == [False, True]
        assert all(x.group_levels == [1, 2, 3] for x in r)
        text = _plain(r)
        assert "b = false" in text and "b = true" in text

    def test_where_drops_level(self, sample):
        r = sample.categorical.ranktest(
            "y", group="zone", where=pl.col("zone") != 2, method="kruskal-wallis"
        )
        assert r.groups.levels == (1, 3)

    def test_custom_score(self, sample):
        r = sample.categorical.ranktest("y", group="b", by="zone", score_fn=lambda rk, n: rk / n)
        assert sorted(r.by_levels) == [1, 2, 3]
        assert all(x.groups.levels == (False, True) for x in r)
        assert all(type(x.diff[0].by_level) is int for x in r)

    def test_string_group_unchanged(self, sample):
        r = sample.categorical.ranktest("y", group="g", method="kruskal-wallis")
        assert r.groups.levels == ("a", "b")
        assert "['a' vs 'b']" in _plain(r)


class TestTtestPrinting:
    def test_false_group_level_is_printed(self, sample):
        text = _plain(sample.categorical.ttest("y", group="b"))
        assert "[false vs true]" in text
        first_cells = [line.split()[0] for line in text.splitlines() if line.strip()]
        assert "false" in first_cells and "true" in first_cells

    def test_by_title(self, sample):
        text = _plain(sample.categorical.ttest("y", by="b", mean_h0=10))
        assert "b = false" in text and "b = true" in text
