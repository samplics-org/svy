# tests/svy/categorical/test_where_nulls_categorical.py
"""
ttest / ranktest / tabulate ignore nulls outside the `where` domain, as R's
subset() does. See tests/svy/estimation/test_where_nulls.py for the dataset.

Reference values from R survey 4.5:

    d <- read.csv("tests/test_data/where_nulls_20260929.csv")
    des <- svydesign(id = ~psu, strata = ~stratum, weights = ~wgt, data = d)
    s <- subset(des, dom == 1)
    svyttest(y ~ factor(grp), s)      # t 0.517492785935185, df 7
    svyttest(I(y - 30) ~ 0, s)        # t 5.15066797656281, df 7
    svyttest(I(y - y2) ~ 0, s)        # t 22.4489899631897, df 7
    svymean(~I(y - y2), s)            # 18.1039297404344, SE 0.80644740677064
    svyranktest(y ~ factor(grp), s)   # t 0.46720885536867, df 7
    svytotal(~interaction(cat, g), s) # table counts below
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from svy import Design, RankScoreMethod, Sample


BASE_DIR = Path(__file__).parents[2]
REL = 1e-9
W = pl.col("dom") == 1

# (count, SE)
COUNTS = {
    ("a", "g1"): (228.48, 81.7828246027245),
    ("b", "g1"): (323.00, 155.9695927630340),
    ("c", "g1"): (251.05, 108.8242048749572),
    ("a", "g2"): (295.94, 115.3251799622846),
    ("b", "g2"): (222.65, 129.9786617102977),
    ("c", "g2"): (284.64, 114.7622132934007),
}


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.read_csv(BASE_DIR / "test_data" / "where_nulls_20260929.csv")


@pytest.fixture(scope="module")
def sample(data) -> Sample:
    return Sample(data, Design(stratum="stratum", psu="psu", wgt="wgt"))


class TestTTest:
    def test_two_groups(self, sample):
        r = sample.categorical.ttest(y="y", group="grp", where=W)
        assert r.stats.t == pytest.approx(0.517492785935185, rel=REL)
        assert r.stats.df == 7
        assert r.stats.p_value == pytest.approx(0.620758810258244, rel=1e-7)
        assert r.diff[0].diff == pytest.approx(1.655248531197789, rel=REL)

    def test_one_sample(self, sample):
        r = sample.categorical.ttest(y="y", mean_h0=30, where=W)
        assert r.stats.t == pytest.approx(5.15066797656281405, rel=REL)
        assert r.stats.df == 7
        assert r.stats.p_value == pytest.approx(0.00132303049040465, rel=1e-7)
        assert r.diff[0].diff == pytest.approx(6.02331291102032651, rel=REL)

    def test_paired(self, sample):
        r = sample.categorical.ttest(y="y", y_pair="y2", where=W)
        assert r.stats.t == pytest.approx(22.4489899631897, rel=REL)
        assert r.diff[0].diff == pytest.approx(18.1039297404344, rel=REL)

    def test_by(self, sample):
        """by + where: same answers as the drop_nulls=True path."""
        a = sample.categorical.ttest(y="y", group="grp", by="g", where=W)
        b = sample.categorical.ttest(y="y", group="grp", by="g", where=W, drop_nulls=True)
        for x, y in zip(a, b):
            assert x.stats.t == pytest.approx(y.stats.t, rel=1e-12)
            assert x.stats.df == y.stats.df

    def test_null_group_inside_domain_raises(self, data):
        df = data.with_columns(
            pl.when(W & (pl.col("unit") == 1)).then(None).otherwise(pl.col("grp")).alias("grp")
        )
        s = Sample(df, Design(stratum="stratum", psu="psu", wgt="wgt"))
        with pytest.raises(ValueError, match="inside the `where` domain.*grp \\(NULL\\)"):
            s.categorical.ttest(y="y", group="grp", where=W)


class TestRankTest:
    def test_kruskal_wallis(self, sample):
        r = sample.categorical.ranktest(
            y="y", group="grp", method=RankScoreMethod.KRUSKAL_WALLIS, where=W
        )
        assert r.stats.value == pytest.approx(0.4672088553686697, rel=REL)
        assert r.stats.df == 7
        assert r.stats.p_value == pytest.approx(0.6545447939569580, rel=1e-7)

    def test_equals_drop_nulls_path(self, sample):
        kw = dict(y="y", group="grp", method="vander-waerden", where=W)
        a = sample.categorical.ranktest(**kw)
        b = sample.categorical.ranktest(**kw, drop_nulls=True)
        assert a.stats.value == pytest.approx(b.stats.value, rel=1e-12)


class TestTabulate:
    def test_counts(self, sample):
        t = sample.categorical.tabulate(rowvar="cat", colvar="g", units="count", where=W)
        got = {(c.rowvar, c.colvar): (c.est, c.se) for c in t.estimates}
        assert set(got) == set(COUNTS)
        for key, ref in COUNTS.items():
            assert got[key] == pytest.approx(ref, rel=REL)

    def test_one_way_equals_drop_nulls_path(self, sample):
        a = sample.categorical.tabulate(rowvar="cat", where=W)
        b = sample.categorical.tabulate(rowvar="cat", where=W, drop_nulls=True)
        assert [(c.rowvar, c.est, c.se) for c in a.estimates] == [
            (c.rowvar, c.est, c.se) for c in b.estimates
        ]
