# tests/svy/categorical/test_ttest_where.py
"""
Tests for ttest 'where' parameter (subpopulation / domain estimation).

T1-T4: R-validated (svyttest with subset on des_sc and des_wgt).
T5-T8: R-validated for est/se/df, Python-validated for t/p.
T9: Python-validated (where + by combinations).

Domain estimation zeroes weights for non-domain observations. This means
degrees_of_freedom counts only PSUs/observations with positive weight
in the domain, matching R's subset() behavior.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import svy
from svy.core.design import Design
from svy.core.sample import Sample


BASE_DIR = Path(__file__).parents[2]
DATA_PATH = BASE_DIR / "test_data" / "svy_synthetic_sample_07082025.csv"

REL = 1e-5


@pytest.fixture
def survey_data() -> pl.DataFrame:
    return pl.read_csv(DATA_PATH).fill_nan(None).drop_nulls()


@pytest.fixture
def sample_strat_clust(survey_data) -> Sample:
    return Sample(
        survey_data,
        Design(stratum="region", psu="cluster", wgt="samp_wgt"),
    )


@pytest.fixture
def sample_wgt(survey_data) -> Sample:
    return Sample(survey_data, Design(wgt="samp_wgt"))


# =============================================================================
# T1: Strat+clust, one-sample, where income > 50000, H0=60000
# R: t=4.1599, df=55, p=0.000113
# =============================================================================


class TestT1OneSampleStratClust:
    def test_t_stat(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=60000,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.t == pytest.approx(4.159851352809592, rel=REL)

    def test_df(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=60000,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df == pytest.approx(55, abs=0)

    def test_p_value(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=60000,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.p_value == pytest.approx(0.00011262702044984001, rel=REL)

    def test_estimate(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=60000,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.diff[0].diff == pytest.approx(1742.8657572740922, rel=REL)


# =============================================================================
# T2: Wgt only, one-sample, where income > 50000, H0=60000
# R: t=3.7444, df=459, p=0.000204
# =============================================================================


class TestT2OneSampleWgtOnly:
    def test_t_stat(self, sample_wgt):
        r = sample_wgt.categorical.ttest(
            y="income",
            mean_h0=60000,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.t == pytest.approx(3.7444087389093976, rel=REL)

    def test_df(self, sample_wgt):
        r = sample_wgt.categorical.ttest(
            y="income",
            mean_h0=60000,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df == pytest.approx(459, abs=0)

    def test_p_value(self, sample_wgt):
        r = sample_wgt.categorical.ttest(
            y="income",
            mean_h0=60000,
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.p_value == pytest.approx(0.00020379941233771487, rel=REL)


# =============================================================================
# T3: Strat+clust, two-sample by sex, where income > 50000
# R: t=0.1931, df=55, p=0.8476
# =============================================================================


class TestT3TwoSampleStratClust:
    def test_t_stat(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            group="sex",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.t == pytest.approx(0.19311291393154498, rel=REL)

    def test_df(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            group="sex",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df == pytest.approx(55, abs=0)

    def test_p_value(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            group="sex",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.p_value == pytest.approx(0.8475819088741542, rel=REL)


# =============================================================================
# T4: Wgt only, two-sample by sex, where income > 50000
# R: t=0.2046, df=459, p=0.8379
# =============================================================================


class TestT4TwoSampleWgtOnly:
    def test_t_stat(self, sample_wgt):
        r = sample_wgt.categorical.ttest(
            y="income",
            group="sex",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.t == pytest.approx(0.20463328917965304, rel=REL)

    def test_df(self, sample_wgt):
        r = sample_wgt.categorical.ttest(
            y="income",
            group="sex",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.df == pytest.approx(459, abs=0)

    def test_p_value(self, sample_wgt):
        r = sample_wgt.categorical.ttest(
            y="income",
            group="sex",
            drop_nulls=True,
            where=pl.col("income") > 50000,
        )
        assert r.stats.p_value == pytest.approx(0.8379493523880314, rel=REL)


# =============================================================================
# T5: where references EXTERNAL column: sex==2, H0=50000
# df=55 (sex==2 exists in every PSU)
# =============================================================================


class TestWhereExternalColumn:
    """where references columns not in y, group, or by."""

    def test_where_pl_expr(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=pl.col("sex") == 2,
        )
        assert r.stats.df == pytest.approx(55, abs=0)
        assert r.estimates[0].est == pytest.approx(50511.94596511162, rel=REL)
        assert r.estimates[0].se == pytest.approx(789.8745198153688, rel=REL)
        assert r.stats.t == pytest.approx(0.6481358143205905, rel=REL)
        assert r.stats.p_value == pytest.approx(0.5195938767753234, rel=REL)

    def test_where_svy_col(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=svy.col("sex") == 2,
        )
        assert r.estimates[0].est == pytest.approx(50511.94596511162, rel=REL)
        assert r.stats.t == pytest.approx(0.6481358143205905, rel=REL)

    def test_where_dict(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where={"sex": 2},
        )
        assert r.estimates[0].est == pytest.approx(50511.94596511162, rel=REL)
        assert r.stats.t == pytest.approx(0.6481358143205905, rel=REL)

    def test_all_where_forms_agree(self, sample_strat_clust):
        r_pl = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=pl.col("sex") == 2,
        )
        r_svy = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=svy.col("sex") == 2,
        )
        r_dict = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where={"sex": 2},
        )
        assert r_pl.stats.t == pytest.approx(r_svy.stats.t, rel=1e-10)
        assert r_pl.stats.t == pytest.approx(r_dict.stats.t, rel=1e-10)
        assert r_pl.estimates[0].est == pytest.approx(r_svy.estimates[0].est, rel=1e-10)
        assert r_pl.estimates[0].est == pytest.approx(r_dict.estimates[0].est, rel=1e-10)


# =============================================================================
# T6: where references TWO external columns: sex==2 & educ=="Postgraduate"
# R: degf(subset)=46, ttest df=45
# =============================================================================


class TestWhereMultipleExternalCols:
    def test_compound_expr(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=(pl.col("sex") == 2) & (pl.col("educ") == "Postgraduate"),
        )
        assert r.stats.df == pytest.approx(45, abs=0)
        assert r.estimates[0].est == pytest.approx(52282.0724342884, rel=REL)
        assert r.estimates[0].se == pytest.approx(1774.9127842918328, rel=REL)
        assert r.stats.t == pytest.approx(1.2857377863774393, rel=REL)
        assert r.stats.p_value == pytest.approx(0.20511173775185076, rel=REL)

    def test_list_of_exprs(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=[pl.col("sex") == 2, pl.col("educ") == "Postgraduate"],
        )
        assert r.estimates[0].est == pytest.approx(52282.0724342884, rel=REL)
        assert r.stats.t == pytest.approx(1.2857377863774393, rel=REL)

    def test_compound_and_list_agree(self, sample_strat_clust):
        r_comp = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=(pl.col("sex") == 2) & (pl.col("educ") == "Postgraduate"),
        )
        r_list = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=[pl.col("sex") == 2, pl.col("educ") == "Postgraduate"],
        )
        assert r_comp.stats.t == pytest.approx(r_list.stats.t, rel=1e-10)
        assert r_comp.estimates[0].est == pytest.approx(r_list.estimates[0].est, rel=1e-10)


# =============================================================================
# T7: Two-sample by sex, where educ=="Postgraduate"
# R: t=-0.0596, df=51, p=0.9527
# =============================================================================


class TestWhereTwoSampleExternal:
    def test_two_sample_where_external(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            group="sex",
            drop_nulls=True,
            where=pl.col("educ") == "Postgraduate",
        )
        assert r.stats.t == pytest.approx(-0.05955232040889406, rel=REL)
        assert r.stats.df == pytest.approx(51, abs=0)
        assert r.stats.p_value == pytest.approx(0.9527449394628185, rel=REL)
        assert r.diff[0].diff == pytest.approx(-173.76758422795024, rel=REL)


# =============================================================================
# T8: Full sample (no where), H0=50000
# =============================================================================


class TestWhereVsFullSample:
    def test_full_sample_reference(self, sample_strat_clust):
        r = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
        )
        assert r.estimates[0].est == pytest.approx(50091.248116800365, rel=REL)
        assert r.estimates[0].se == pytest.approx(509.4338766925862, rel=REL)
        assert r.stats.t == pytest.approx(0.17911670380615077, rel=REL)
        assert r.stats.df == pytest.approx(55, abs=0)

    def test_where_changes_estimate(self, sample_strat_clust):
        r_full = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
        )
        r_sub = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=pl.col("sex") == 2,
        )
        assert r_full.estimates[0].est == pytest.approx(50091.248116800365, rel=REL)
        assert r_sub.estimates[0].est == pytest.approx(50511.94596511162, rel=REL)


# =============================================================================
# where=None equivalence
# =============================================================================


class TestWhereNone:
    def test_where_none_is_full_sample(self, sample_strat_clust):
        r1 = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=49000,
            drop_nulls=True,
        )
        r2 = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=49000,
            drop_nulls=True,
            where=None,
        )
        assert r1.stats.t == pytest.approx(r2.stats.t, rel=1e-10)
        assert r1.stats.p_value == pytest.approx(r2.stats.p_value, rel=1e-10)


# =============================================================================
# T9: where + by combinations
# =============================================================================


class TestWhereWithBy:
    def test_where_sex2_by_educ(self, sample_strat_clust):
        """where=pl.col('sex')==2, by='educ' — 6 education levels.
        sex==2 exists in every PSU, so df=55 for all levels."""
        results = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=pl.col("sex") == 2,
            by="educ",
        )
        from svy.categorical.ttest import TTestByResult

        assert isinstance(results, TTestByResult)
        assert len(results) == 6

        expected = {
            "High School": dict(
                est=49553.9882723692,
                se=1391.3448145383315,
                t=-0.320561605556272,
                df=55.0,
                p=0.749757436600945,
            ),
            "Less than High School": dict(
                est=48972.7575868501,
                se=1418.3898577534083,
                t=-0.724231358208491,
                df=55.0,
                p=0.47199327915832623,
            ),
            "None": dict(
                est=53517.47066393566,
                se=5407.050676453174,
                t=0.6505340664280569,
                df=55.0,
                p=0.5180557410834917,
            ),
            "Other Training": dict(
                est=46383.162559082535,
                se=3800.928826674592,
                t=-0.951566736934091,
                df=55.0,
                p=0.3454798975733486,
            ),
            "Postgraduate": dict(
                est=52282.0724342884,
                se=1774.9127842918328,
                t=1.2857377863774393,
                df=55.0,
                p=0.2039217118549561,
            ),
            "Undergraduate": dict(
                est=51829.75451909497,
                se=1387.5521132104707,
                t=1.318692466880645,
                df=55.0,
                p=0.19273672209312898,
            ),
        }

        for r in results:
            by_val = r.diff[0].by_level
            exp = expected[by_val]
            assert r.estimates[0].est == pytest.approx(exp["est"], rel=REL)
            assert r.estimates[0].se == pytest.approx(exp["se"], rel=REL)
            assert r.stats.t == pytest.approx(exp["t"], rel=REL)
            assert r.stats.df == pytest.approx(exp["df"], abs=0)
            assert r.stats.p_value == pytest.approx(exp["p"], rel=REL)

    def test_where_educ_postgrad_by_sex(self, sample_strat_clust):
        """where=educ=='Postgraduate', by='sex' — 2 sex levels.
        educ=='Postgraduate' doesn't exist in all PSUs, so df=51."""
        results = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=pl.col("educ") == "Postgraduate",
            by="sex",
        )
        from svy.categorical.ttest import TTestByResult

        assert isinstance(results, TTestByResult)
        assert len(results) == 2

        expected = {
            "1": dict(
                est=52455.84001851617,
                se=2322.673036856926,
                t=1.0573335030570858,
                df=51.0,
                p=0.29534210185639087,
            ),
            "2": dict(
                est=52282.0724342884,
                se=1774.9127842918328,
                t=1.2857377863774393,
                df=51.0,
                p=0.20434201714097233,
            ),
        }

        for r in results:
            by_val = r.diff[0].by_level
            exp = expected[by_val]
            assert r.estimates[0].est == pytest.approx(exp["est"], rel=REL)
            assert r.estimates[0].se == pytest.approx(exp["se"], rel=REL)
            assert r.stats.t == pytest.approx(exp["t"], rel=REL)
            assert r.stats.df == pytest.approx(exp["df"], abs=0)
            assert r.stats.p_value == pytest.approx(exp["p"], rel=REL)

    def test_where_dict_with_by(self, sample_strat_clust):
        """where={"sex": 2}, by='educ' — dict form matches pl.Expr form."""
        r_dict = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where={"sex": 2},
            by="educ",
        )
        r_expr = sample_strat_clust.categorical.ttest(
            y="income",
            mean_h0=50000,
            drop_nulls=True,
            where=pl.col("sex") == 2,
            by="educ",
        )
        assert len(r_dict) == len(r_expr)
        for rd, re in zip(r_dict, r_expr):
            assert rd.estimates[0].est == pytest.approx(re.estimates[0].est, rel=1e-10)
            assert rd.stats.t == pytest.approx(re.stats.t, rel=1e-10)


# =============================================================================
# _extract_where_cols unit tests
# =============================================================================


class TestExtractWhereCols:
    def _extract(self, where):
        from svy.core.data_prep import extract_where_cols

        return extract_where_cols(where)

    def test_none(self):
        assert self._extract(None) == []

    def test_single_pl_expr(self):
        assert "sex" in self._extract(pl.col("sex") == 2)

    def test_compound_pl_expr(self):
        cols = self._extract((pl.col("sex") == 2) & (pl.col("region") == "North"))
        assert "sex" in cols
        assert "region" in cols

    def test_dict_form(self):
        assert set(self._extract({"sex": 2, "region": "North"})) == {"sex", "region"}

    def test_list_of_exprs(self):
        cols = self._extract([pl.col("sex") == 2, pl.col("age") > 30])
        assert "sex" in cols
        assert "age" in cols

    def test_svy_col_wrapper(self):
        assert "sex" in self._extract(svy.col("sex") == 2)

    def test_empty_dict(self):
        assert self._extract({}) == []

    def test_empty_list(self):
        assert self._extract([]) == []

    def test_duplicate_columns_ok(self):
        cols = self._extract([pl.col("sex") > 1, pl.col("sex") < 5])
        assert cols.count("sex") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
