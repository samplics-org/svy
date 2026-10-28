# tests/svy/categorical/test_ranktest_fpc.py
"""
ranktest applies the design's finite population correction, as R's svyranktest does.

R survey 4.5, on the test_data files:
```r
st <- read.csv("test_data/apistrat.csv"); c2 <- read.csv("test_data/apiclus2.csv")
st$m3 <- cut(st$meals, c(-1, 30, 60, 101), labels = c("lo", "mid", "hi"))
strat_fpc <- svydesign(id = ~1, strata = ~stype, weights = ~pw, fpc = ~fpc, data = st)
strat     <- svydesign(id = ~1, strata = ~stype, weights = ~pw, data = st)
clus2_fpc <- svydesign(id = ~dnum + snum, fpc = ~fpc1 + fpc2, data = c2)
clus2     <- svydesign(id = ~dnum + snum, weights = ~pw, data = c2)
svyranktest(api00 ~ awards, strat_fpc)   # two-sample; likewise strat
svyranktest(api00 ~ both, clus2_fpc)     # two-sample; likewise clus2
svyranktest(api00 ~ m3, strat_fpc)       # k-sample: F = Chisq / df; likewise strat
svyranktest(api00 ~ stype, clus2_fpc)    # k-sample; likewise clus2
```
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import svy


DATA = Path(__file__).parents[2] / "test_data"

# R prints these to 15 significant digits; svy agrees to about 1e-11.
REL = 1e-9


def _strat(fpc: bool) -> svy.Sample:
    df = pl.read_csv(DATA / "apistrat.csv", null_values="NA").with_columns(
        m3=pl.when(pl.col("meals") <= 30)
        .then(pl.lit("lo"))
        .when(pl.col("meals") <= 60)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("hi"))
    )
    return svy.Sample(df, svy.Design(stratum="stype", wgt="pw", pop_size="fpc" if fpc else None))


def _clus2(fpc: bool) -> svy.Sample:
    df = pl.read_csv(DATA / "apiclus2.csv", null_values="NA")
    pop = svy.PopSize(psu="fpc1", ssu="fpc2") if fpc else None
    return svy.Sample(df, svy.Design(psu="dnum", ssu="snum", wgt="pw", pop_size=pop))


SAMPLES = {
    "strat_fpc": lambda: _strat(True),
    "strat": lambda: _strat(False),
    "clus2_fpc": lambda: _clus2(True),
    "clus2": lambda: _clus2(False),
}

# (design, t, df, p)
R_TWO_SAMPLE = [
    ("strat_fpc", 2.11879850890955, 196, 0.0353667842843476),
    ("strat", 2.08978578762943, 196, 0.0379273567125392),
    ("clus2_fpc", 5.92130370080041, 38, 7.29949788105409e-07),
    ("clus2", 5.90838167606189, 38, 7.60367485073025e-07),
]
TWO_SAMPLE_GROUP = {"strat_fpc": "awards", "strat": "awards", "clus2_fpc": "both", "clus2": "both"}

# (design, F, df_num, df_den, p)
R_K_SAMPLE = [
    ("strat_fpc", 167.53824766091, 2, 195, 4.52237176634746e-43),
    ("strat", 162.483275268936, 2, 195, 2.9564632707546e-42),
    ("clus2_fpc", 8.87876301974333, 2, 37, 0.000708795996119318),
    ("clus2", 11.4868996347934, 2, 37, 0.000131656711392747),
]
K_SAMPLE_GROUP = {"strat_fpc": "m3", "strat": "m3", "clus2_fpc": "stype", "clus2": "stype"}


@pytest.mark.parametrize(
    ("design", "t", "df", "p"), R_TWO_SAMPLE, ids=[c[0] for c in R_TWO_SAMPLE]
)
def test_two_sample_matches_r(design, t, df, p):
    stats = SAMPLES[design]().categorical.ranktest("api00", group=TWO_SAMPLE_GROUP[design]).stats
    assert stats.value == pytest.approx(t, rel=REL)
    assert stats.df == df
    assert stats.p_value == pytest.approx(p, rel=REL)


@pytest.mark.parametrize(
    ("design", "f", "df_num", "df_den", "p"), R_K_SAMPLE, ids=[c[0] for c in R_K_SAMPLE]
)
def test_k_sample_matches_r(design, f, df_num, df_den, p):
    stats = SAMPLES[design]().categorical.ranktest("api00", group=K_SAMPLE_GROUP[design]).stats
    assert stats.value == pytest.approx(f, rel=REL)
    assert stats.df_num == df_num
    assert stats.df_den == df_den
    assert stats.p_value == pytest.approx(p, rel=REL)


@pytest.mark.parametrize(
    ("design", "t", "df", "p"), R_TWO_SAMPLE, ids=[c[0] for c in R_TWO_SAMPLE]
)
def test_custom_wilcoxon_score_matches_r(design, t, df, p):
    """``rk / n`` is R's default Wilcoxon score, so a custom score_fn gives the same test."""
    stats = (
        SAMPLES[design]()
        .categorical.ranktest(
            "api00", group=TWO_SAMPLE_GROUP[design], score_fn=lambda rk, n: rk / n
        )
        .stats
    )
    assert stats.value == pytest.approx(t, rel=REL)
    assert stats.df == df
    assert stats.p_value == pytest.approx(p, rel=REL)
