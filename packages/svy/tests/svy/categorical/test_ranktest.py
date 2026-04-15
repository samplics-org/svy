# tests/svy/categorical/test_ranktest_where.py
"""
Tests for ranktest 'where' parameter (subpopulation / domain estimation).

Both R and Python start from the same clean dataset:
    d <- d[complete.cases(d), ]   # R (922 rows)
    df = df.drop_nulls()          # Python
This ensures identical row counts and eliminates NaN/null handling differences.

R code:
```r
options(digits = 15)
library(survey)
d <- read.csv("data/svy_synthetic_sample_07082025.csv")
d <- d[complete.cases(d), ]  # 922 rows

des_sc  <- svydesign(id=~cluster, strata=~region, weights=~samp_wgt, data=d)
des_wgt <- svydesign(id=~1,       weights=~samp_wgt, data=d)
des_c   <- svydesign(id=~cluster, weights=~samp_wgt, data=d)

# D0 baseline: t=0.784492, df=55, p=0.436117
# D1: sub income>50000, t=0.522103, df=55, p=0.603695, diff=0.018253
# D3: wgt sub income>50000, t=0.585878, df=459, p=0.558246, diff=0.018253
# D4: wgt k-sample sub income>50000, F=1.975741, df_num=3, df_den=457, p=0.116804
# D5: clust k-sample sub income>50000, F=2.332276, df_num=3, df_den=56, p=0.083898
# D6: VdW sub income>50000, t=0.025041, df=55, p=0.980113, diff=0.003046
# D7: sub educ==Postgraduate, t=-0.341496, df=51, p=0.734133, diff=-0.018109
# D8a: sub income>50000 by region (strat+clust), per-region results
# D8b: wgt sub income>50000 by region, per-region results
# D8d: sub educ==Postgraduate by region, per-region results
```

Note: D2 (where=sex==2 with group=sex) and D8c (where=sex==2, by=educ with group=sex)
are degenerate cases — filtering to one group level produces NaN. Not tested.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from svy.core.design import Design
from svy.core.sample import Sample


BASE_DIR = Path(__file__).parents[2]
DATA_PATH = BASE_DIR / "test_data" / "svy_synthetic_sample_07082025.csv"

REL = 1e-5


@pytest.fixture
def survey_data() -> pl.DataFrame:
    return pl.read_csv(DATA_PATH).fill_nan(None).drop_nulls()


@pytest.fixture
def sample_wgt(survey_data) -> Sample:
    return Sample(survey_data, Design(wgt="samp_wgt"))


@pytest.fixture
def sample_strat_clust(survey_data) -> Sample:
    return Sample(
        survey_data,
        Design(stratum="region", psu="cluster", wgt="samp_wgt"),
    )


@pytest.fixture
def sample_clust(survey_data) -> Sample:
    return Sample(survey_data, Design(psu="cluster", wgt="samp_wgt"))


# =============================================================================
# D0: Baseline — full sample, no where
# =============================================================================


class TestD0Baseline:
    def test_full_sample(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
        )
        assert r.stats.value == pytest.approx(0.784492299264955, rel=REL)
        assert r.stats.df == pytest.approx(55, abs=0)
        assert r.stats.p_value == pytest.approx(0.436117228247927, rel=REL)


# =============================================================================
# D1: Strat+clust, income ~ sex, KW, where income > 50000
# R: t=0.522103, df=55, p=0.603695, diff=0.018253
# =============================================================================


class TestD1StratClustWhereIncome:
    def test_value(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.value == pytest.approx(0.522103129451314, rel=REL)

    def test_df(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df == pytest.approx(55, abs=0)

    def test_p_value(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.p_value == pytest.approx(0.603695071503463, rel=REL)

    def test_diff(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.diff[0].diff == pytest.approx(0.0182527504967008, rel=REL)

    def test_where_changes_result(self, sample_strat_clust):
        r_full = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
        )
        r_where = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r_full.stats.value != pytest.approx(r_where.stats.value, rel=1e-3)


# =============================================================================
# D3: Wgt only, income ~ sex, KW, where income > 50000
# R: t=0.585878, df=459, p=0.558246
# =============================================================================


class TestD3WgtOnlyWhereIncome:
    def test_value(self, sample_wgt):
        r = sample_wgt.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.value == pytest.approx(0.585877765121064, rel=REL)

    def test_df(self, sample_wgt):
        r = sample_wgt.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df == pytest.approx(459, abs=0)

    def test_p_value(self, sample_wgt):
        r = sample_wgt.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.p_value == pytest.approx(0.558245623066575, rel=REL)


# =============================================================================
# D4: Wgt only, income ~ region, KW k-sample, where income > 50000
# R: F=1.975741, df_num=3, df_den=457, p=0.116804
# =============================================================================


class TestD4WgtOnlyKSampleWhereIncome:
    def test_value(self, sample_wgt):
        r = sample_wgt.categorical.ranktest(
            y="income",
            group="region",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.value == pytest.approx(1.9757411445315, rel=REL)

    def test_df(self, sample_wgt):
        r = sample_wgt.categorical.ranktest(
            y="income",
            group="region",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df_num == pytest.approx(3, abs=0)
        assert r.stats.df_den == pytest.approx(457, abs=0)

    def test_p_value(self, sample_wgt):
        r = sample_wgt.categorical.ranktest(
            y="income",
            group="region",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.p_value == pytest.approx(0.11680424564279, rel=REL)


# =============================================================================
# D5: Clust only, income ~ region, KW k-sample, where income > 50000
# R: F=2.332276, df_num=3, df_den=56, p=0.083898
# =============================================================================


class TestD5ClustKSampleWhereIncome:
    def test_value(self, sample_clust):
        r = sample_clust.categorical.ranktest(
            y="income",
            group="region",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.value == pytest.approx(2.33227560517172, rel=REL)

    def test_df(self, sample_clust):
        r = sample_clust.categorical.ranktest(
            y="income",
            group="region",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df_num == pytest.approx(3, abs=0)
        assert r.stats.df_den == pytest.approx(56, abs=0)

    def test_p_value(self, sample_clust):
        r = sample_clust.categorical.ranktest(
            y="income",
            group="region",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.p_value == pytest.approx(0.0838982042541097, rel=REL)


# =============================================================================
# D6: Strat+clust, income ~ sex, VdW, where income > 50000
# R: t=0.025041, df=55, p=0.980113, diff=0.003046
# =============================================================================


class TestD6StratClustVdWWhereIncome:
    def test_value(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="vander-waerden",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.value == pytest.approx(0.0250406685671114, rel=REL)

    def test_df(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="vander-waerden",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df == pytest.approx(55, abs=0)

    def test_diff(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="vander-waerden",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.diff[0].diff == pytest.approx(0.00304630135277791, rel=REL)


# =============================================================================
# D7: Strat+clust, income ~ sex, KW, where educ==Postgraduate
# R: t=-0.341496, df=51, p=0.734133, diff=-0.018109
# Domain doesn't cover all PSUs → df=51 (not 55)
# =============================================================================


class TestD7StratClustWhereEduc:
    def test_value(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("educ") == "Postgraduate",
        )
        assert r.stats.value == pytest.approx(-0.3414963051553, rel=REL)

    def test_df(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("educ") == "Postgraduate",
        )
        assert r.stats.df == pytest.approx(51, abs=0)

    def test_p_value(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("educ") == "Postgraduate",
        )
        assert r.stats.p_value == pytest.approx(0.734133074246, rel=REL)


# =============================================================================
# D8: where + by combinations (Python-validated)
# =============================================================================


class TestWhereWithBy:
    def test_where_income_by_region_strat_clust(self, sample_strat_clust):
        """where=income>50000, by=region — 4 regions, per-region domain DF."""
        results = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
            by="region",
        )
        from svy.categorical.ranktest import RankTestByResult

        assert isinstance(results, RankTestByResult)
        assert len(results) == 4

        expected = {
            "1": dict(value=0.6533721584475315, df=13.0, p=0.5249046355691003),
            "2": dict(value=0.2218514809360551, df=13.0, p=0.8278765540792989),
            "3": dict(value=-0.7591057299450957, df=13.0, p=0.4613324952472472),
            "4": dict(value=0.8619117855740701, df=13.0, p=0.40435504250515397),
        }

        for r in results:
            by_val = str(r.diff[0].by_level)
            exp = expected[by_val]
            assert r.stats.value == pytest.approx(exp["value"], rel=REL)
            assert r.stats.df == pytest.approx(exp["df"], abs=0)
            assert r.stats.p_value == pytest.approx(exp["p"], rel=REL)

    def test_where_income_by_region_wgt_only(self, sample_wgt):
        """where=income>50000, by=region — wgt only, per-region domain DF."""
        results = sample_wgt.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("income") > 50000,
            by="region",
        )
        from svy.categorical.ranktest import RankTestByResult

        assert isinstance(results, RankTestByResult)
        assert len(results) == 4

        expected = {
            "1": dict(value=0.8185408770118657, df=111.0, p=0.414802225330125),
            "2": dict(value=0.2306248682667073, df=125.0, p=0.8179832767548052),
            "3": dict(value=-0.9748778203039873, df=95.0, p=0.33209673342356716),
            "4": dict(value=0.8106681952919169, df=122.0, p=0.4191348293337569),
        }

        for r in results:
            by_val = str(r.diff[0].by_level)
            exp = expected[by_val]
            assert r.stats.value == pytest.approx(exp["value"], rel=REL)
            assert r.stats.df == pytest.approx(exp["df"], abs=0)
            assert r.stats.p_value == pytest.approx(exp["p"], rel=REL)

    def test_where_educ_postgrad_by_region(self, sample_strat_clust):
        """where=educ==Postgraduate, by=region — domain doesn't cover all PSUs,
        per-region DF varies (12, 12, 13, 11)."""
        results = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=pl.col("educ") == "Postgraduate",
            by="region",
        )
        from svy.categorical.ranktest import RankTestByResult

        assert isinstance(results, RankTestByResult)
        assert len(results) == 4

        expected = {
            "1": dict(value=0.5005716353309074, df=12.0, p=0.6257268608395756),
            "2": dict(value=0.288205403811298, df=12.0, p=0.778106948216473),
            "3": dict(value=-0.060185577904031756, df=13.0, p=0.9529232366929273),
            "4": dict(value=-2.0268982082746247, df=11.0, p=0.06761032926652448),
        }

        for r in results:
            by_val = str(r.diff[0].by_level)
            exp = expected[by_val]
            assert r.stats.value == pytest.approx(exp["value"], rel=REL)
            assert r.stats.df == pytest.approx(exp["df"], abs=0)
            assert r.stats.p_value == pytest.approx(exp["p"], rel=REL)

    def test_where_none_is_full_sample(self, sample_strat_clust):
        r1 = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
        )
        r2 = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method="kruskal-wallis",
            drop_nulls=True,
            where=None,
        )
        assert r1.stats.value == pytest.approx(r2.stats.value, rel=1e-10)
        assert r1.stats.p_value == pytest.approx(r2.stats.p_value, rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
