# tests/svy/categorical/test_ranktest_calibration.py
"""
ranktest credits a calibration in its variance, as R's svyranktest does through svyglm.

R survey 4.5, on test_data/apiclus1.csv:
```r
c1$m3 <- cut(c1$meals, c(-1, 30, 60, 101), labels = c("lo", "mid", "hi"))
d  <- svydesign(id = ~dnum, weights = ~pw, data = c1)
ps <- postStratify(d, ~stype, data.frame(stype = c("E", "H", "M"), Freq = c(4421, 755, 1018)))
rk <- rake(d, list(~stype, ~comp.imp),
           list(data.frame(stype = c("E", "H", "M"), Freq = c(4421, 755, 1018)),
                data.frame(comp.imp = c("No", "Yes"), Freq = c(2500, 3694))),
           control = list(maxit = 100, epsilon = 1e-12))
svyranktest(api00 ~ comp.imp, ps)   # two-sample; likewise d and rk
svyranktest(api00 ~ m3, ps)         # k-sample: F = Chisq / df; likewise d and rk
```
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import svy


DATA = Path(__file__).parents[2] / "test_data" / "apiclus1.csv"

# R prints 15 significant digits; svy agrees to about 1e-13.
REL = 1e-10

STYPE = {"E": 4421, "H": 755, "M": 1018}
COMP_IMP = {"No": 2500, "Yes": 3694}


def _sample(calibration: str) -> svy.Sample:
    df = pl.read_csv(DATA, null_values="NA").with_columns(
        m3=pl.when(pl.col("meals") <= 30)
        .then(pl.lit("lo"))
        .when(pl.col("meals") <= 60)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("hi"))
    )
    s = svy.Sample(df, svy.Design(psu="dnum", wgt="pw"))
    if calibration == "poststrat":
        return s.weighting.poststratify(controls=STYPE, cells="stype")
    if calibration == "raked":
        # R's default epsilon = 1 stops raking early; both sides converge fully here.
        return s.weighting.rake(
            controls={"stype": STYPE, "comp.imp": COMP_IMP}, tol=1e-12, max_iter=200
        )
    return s


# (calibration, t, df, p)
R_TWO_SAMPLE = [
    ("plain", 0.784784342328261, 13, 0.446646200769617),
    ("poststrat", 0.693516394922733, 13, 0.500189898952743),
    ("raked", 0.703907565261009, 13, 0.493906626740773),
]

# (calibration, F, df_num, df_den, p)
R_K_SAMPLE = [
    ("plain", 143.526548364849, 2, 12, 4.17443451537255e-09),
    ("poststrat", 121.47033388787, 2, 12, 1.08755883808575e-08),
    ("raked", 143.474406104374, 2, 12, 4.18317933140767e-09),
]


@pytest.mark.parametrize(
    ("calibration", "t", "df", "p"), R_TWO_SAMPLE, ids=[c[0] for c in R_TWO_SAMPLE]
)
def test_two_sample_matches_r(calibration, t, df, p):
    stats = _sample(calibration).categorical.ranktest("api00", group="comp.imp").stats
    assert stats.value == pytest.approx(t, rel=REL)
    assert stats.df == df
    assert stats.p_value == pytest.approx(p, rel=REL)


@pytest.mark.parametrize(
    ("calibration", "f", "df_num", "df_den", "p"), R_K_SAMPLE, ids=[c[0] for c in R_K_SAMPLE]
)
def test_k_sample_matches_r(calibration, f, df_num, df_den, p):
    stats = _sample(calibration).categorical.ranktest("api00", group="m3").stats
    assert stats.value == pytest.approx(f, rel=REL)
    assert stats.df_num == df_num
    assert stats.df_den == df_den
    assert stats.p_value == pytest.approx(p, rel=REL)


@pytest.mark.parametrize(
    ("calibration", "t", "df", "p"), R_TWO_SAMPLE, ids=[c[0] for c in R_TWO_SAMPLE]
)
def test_custom_wilcoxon_score_matches_r(calibration, t, df, p):
    """``rk / n`` is R's default Wilcoxon score, so a custom score_fn gives the same test."""
    stats = (
        _sample(calibration)
        .categorical.ranktest("api00", group="comp.imp", score_fn=lambda rk, n: rk / n)
        .stats
    )
    assert stats.value == pytest.approx(t, rel=REL)
    assert stats.df == df
    assert stats.p_value == pytest.approx(p, rel=REL)
