# tests/svy/categorical/test_group_order.py
"""Numeric test groups are ordered by value, as R's ``factor()`` orders them.

Groups 2 and 10 used to be ordered as text ("10" < "2"), which flipped the
reported difference. The same data coded "a"/"b" is the reference.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample


N = 240


@pytest.fixture(scope="module")
def sample() -> Sample:
    rng = np.random.default_rng(3)
    df = pl.DataFrame(
        {
            "stratum": np.repeat(["s1", "s2"], N // 2),
            "psu": np.repeat(np.arange(24), N // 24),
            "w": rng.uniform(1, 3, N),
            "g": rng.choice([2, 10], N),
            "e": rng.normal(10, 2, N),
        }
    ).with_columns(
        y=pl.col("g") * 0.3 + pl.col("e"),
        gs=pl.when(pl.col("g") == 2).then(pl.lit("a")).otherwise(pl.lit("b")),
    )
    return Sample(df, Design(stratum="stratum", psu="psu", wgt="w"))


def test_ranktest_two_sample(sample):
    r = sample.categorical.ranktest("y", group="g", method="kruskal-wallis")
    ref = sample.categorical.ranktest("y", group="gs", method="kruskal-wallis")
    assert r.groups.levels == (2, 10)
    assert r.diff[0].diff == pytest.approx(ref.diff[0].diff, rel=1e-12)
    assert r.diff[0].diff > 0


def test_ranktest_by(sample):
    r = sample.categorical.ranktest("y", group="g", by="stratum", method="kruskal-wallis")
    ref = sample.categorical.ranktest("y", group="gs", by="stratum", method="kruskal-wallis")
    for a, b in zip(r, ref):
        assert a.groups.levels == (2, 10)
        assert a.diff[0].diff == pytest.approx(b.diff[0].diff, rel=1e-12)


def test_ttest_two_sample(sample):
    t = sample.categorical.ttest("y", group="g")
    ref = sample.categorical.ttest("y", group="gs")
    assert t.groups.levels == (2, 10)
    assert t.diff[0].diff == pytest.approx(ref.diff[0].diff, rel=1e-12)
