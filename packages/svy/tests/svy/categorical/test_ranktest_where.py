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
from svy.core.enumerations import RankScoreMethod
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
            method=RankScoreMethod.KRUSKAL_WALLIS,
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
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.value == pytest.approx(0.522103129451314, rel=REL)

    def test_df(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df == pytest.approx(55, abs=0)

    def test_p_value(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.p_value == pytest.approx(0.603695071503463, rel=REL)

    def test_diff(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.diff[0].diff == pytest.approx(0.0182527504967008, rel=REL)

    def test_where_changes_result(self, sample_strat_clust):
        r_full = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
        )
        r_where = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
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
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.value == pytest.approx(0.585877765121064, rel=REL)

    def test_df(self, sample_wgt):
        r = sample_wgt.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df == pytest.approx(459, abs=0)

    def test_p_value(self, sample_wgt):
        r = sample_wgt.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
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
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.value == pytest.approx(1.9757411445315, rel=REL)

    def test_df(self, sample_wgt):
        r = sample_wgt.categorical.ranktest(
            y="income",
            group="region",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df_num == pytest.approx(3, abs=0)
        assert r.stats.df_den == pytest.approx(457, abs=0)

    def test_p_value(self, sample_wgt):
        r = sample_wgt.categorical.ranktest(
            y="income",
            group="region",
            method=RankScoreMethod.KRUSKAL_WALLIS,
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
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.value == pytest.approx(2.33227560517172, rel=REL)

    def test_df(self, sample_clust):
        r = sample_clust.categorical.ranktest(
            y="income",
            group="region",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df_num == pytest.approx(3, abs=0)
        assert r.stats.df_den == pytest.approx(56, abs=0)

    def test_p_value(self, sample_clust):
        r = sample_clust.categorical.ranktest(
            y="income",
            group="region",
            method=RankScoreMethod.KRUSKAL_WALLIS,
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
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("educ") == "Postgraduate",
        )
        assert r.stats.value == pytest.approx(-0.3414963051553, rel=REL)

    def test_df(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("educ") == "Postgraduate",
        )
        assert r.stats.df == pytest.approx(51, abs=0)

    def test_p_value(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
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
            method=RankScoreMethod.KRUSKAL_WALLIS,
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
            method=RankScoreMethod.KRUSKAL_WALLIS,
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
            method=RankScoreMethod.KRUSKAL_WALLIS,
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
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
        )
        r2 = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=None,
        )
        assert r1.stats.value == pytest.approx(r2.stats.value, rel=1e-10)
        assert r1.stats.p_value == pytest.approx(r2.stats.p_value, rel=1e-10)


# =============================================================================
# D9: Group labels must reflect the levels actually tested
# =============================================================================


class TestD9GroupLabelsUnderWhere:
    """Regression tests: reported group levels must be the ones the Rust
    kernel actually tested (rows in-domain with positive weight), not all
    levels present in the data."""

    @pytest.fixture
    def three_group_data(self) -> pl.DataFrame:
        # Group A out-of-domain under `where`; B clearly below C so the
        # delta sign is unambiguous.
        return pl.DataFrame(
            {
                "grp": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
                "y": (
                    [50.0, 52.0, 48.0, 51.0, 49.0, 53.0, 47.0, 50.0]  # A
                    + [10.0, 12.0, 11.0, 9.0, 13.0, 10.0, 12.0, 11.0]  # B
                    + [90.0, 92.0, 91.0, 89.0, 93.0, 90.0, 92.0, 91.0]  # C
                ),
                "wgt": [1.5] * 24,
                "keep": [0] * 8 + [1] * 16,
            }
        )

    def test_labels_are_active_levels(self, three_group_data):
        sample = Sample(three_group_data, Design(wgt="wgt"))
        r = sample.categorical.ranktest(
            y="y",
            group="grp",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("keep") == 1,
        )
        # Domain leaves only B and C: a two-sample test on (B, C)
        assert r.groups.levels == ("B", "C")

    def test_delta_matches_filtered_data(self, three_group_data):
        """where= (weights zeroed) must agree with physically filtering the
        rows for an unclustered design: same labels, same delta sign."""
        s_where = Sample(three_group_data, Design(wgt="wgt"))
        r_where = s_where.categorical.ranktest(
            y="y",
            group="grp",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            where=pl.col("keep") == 1,
        )
        filtered = three_group_data.filter(pl.col("keep") == 1)
        r_filtered = Sample(filtered, Design(wgt="wgt")).categorical.ranktest(
            y="y",
            group="grp",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
        )
        assert r_where.groups.levels == r_filtered.groups.levels
        assert r_where.diff[0].diff * r_filtered.diff[0].diff > 0

    def test_by_levels_get_their_own_labels(self):
        """Different by-levels can have different active groups; each result
        must carry the labels of its own by-level."""
        data = pl.DataFrame(
            {
                "site": ["s1"] * 16 + ["s2"] * 16,
                "grp": ["A"] * 8 + ["B"] * 8 + ["B"] * 8 + ["C"] * 8,
                "y": [float(v) for v in range(1, 33)],
                "wgt": [1.0] * 32,
            }
        )
        sample = Sample(data, Design(wgt="wgt"))
        results = sample.categorical.ranktest(
            y="y",
            group="grp",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
            by="site",
        )
        by_map = {str(r.diff[0].by_level): r for r in results}
        assert by_map["s1"].groups.levels == ("A", "B")
        assert by_map["s2"].groups.levels == ("B", "C")


# =============================================================================
# D10: Custom score_fn must honor by= (one test per by-level, not pooled)
# =============================================================================


def _wilcoxon_score(rankhat, n_hat):
    # Same transformation as the Rust built-in Wilcoxon/Kruskal-Wallis score
    # (svy-rs categorical/ranktest.rs: g(r) = r / N), so results must match
    # the built-in method exactly.
    return rankhat / n_hat


class TestD10CustomScoreFnBy:
    """Regression tests: score_fn used to silently ignore by= and return a
    single pooled test presented as the per-group answer."""

    def test_custom_wilcoxon_matches_builtin(self, sample_strat_clust):
        r_custom = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            score_fn=_wilcoxon_score,
            drop_nulls=True,
        )
        r_builtin = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            drop_nulls=True,
        )
        assert r_custom.groups.levels == r_builtin.groups.levels
        assert r_custom.stats.value == pytest.approx(r_builtin.stats.value, rel=1e-10)
        assert r_custom.stats.df == pytest.approx(r_builtin.stats.df, rel=1e-10)
        assert r_custom.stats.p_value == pytest.approx(r_builtin.stats.p_value, rel=1e-10)
        assert r_custom.diff[0].diff == pytest.approx(r_builtin.diff[0].diff, rel=1e-10)

    def test_custom_score_by_returns_per_level_results(self, sample_strat_clust):
        from svy.categorical.ranktest import RankTestByResult

        r_custom = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            score_fn=_wilcoxon_score,
            by="region",
            drop_nulls=True,
        )
        r_builtin = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            by="region",
            drop_nulls=True,
        )
        assert isinstance(r_custom, RankTestByResult)
        assert len(r_custom) == len(r_builtin) == 4

        builtin_by = {str(r.diff[0].by_level): r for r in r_builtin}
        for rc in r_custom:
            rb = builtin_by[str(rc.diff[0].by_level)]
            assert rc.groups.levels == rb.groups.levels
            assert rc.stats.value == pytest.approx(rb.stats.value, rel=1e-10)
            assert rc.stats.df == pytest.approx(rb.stats.df, rel=1e-10)
            assert rc.stats.p_value == pytest.approx(rb.stats.p_value, rel=1e-10)
            assert rc.diff[0].diff == pytest.approx(rb.diff[0].diff, rel=1e-10)

    def test_custom_score_by_with_where(self, sample_strat_clust):
        """by= combined with where= — per-level results, not pooled."""
        r_custom = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            score_fn=_wilcoxon_score,
            by="region",
            where=pl.col("income") > 50000,
            drop_nulls=True,
        )
        r_builtin = sample_strat_clust.categorical.ranktest(
            y="income",
            group="sex",
            method=RankScoreMethod.KRUSKAL_WALLIS,
            by="region",
            where=pl.col("income") > 50000,
            drop_nulls=True,
        )
        assert len(r_custom) == len(r_builtin)
        builtin_by = {str(r.diff[0].by_level): r for r in r_builtin}
        for rc in r_custom:
            rb = builtin_by[str(rc.diff[0].by_level)]
            assert rc.stats.value == pytest.approx(rb.stats.value, rel=1e-10)
            assert rc.diff[0].diff == pytest.approx(rb.diff[0].diff, rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
