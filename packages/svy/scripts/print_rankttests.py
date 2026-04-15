from pathlib import Path

import polars as pl

import svy
from svy.core.enumerations import RankScoreMethod

BASE_DIR = Path("tests/test_data")
DATA_PATH = BASE_DIR / "svy_synthetic_sample_07082025.csv"

data = svy.read_csv(DATA_PATH).fill_nan(None).drop_nulls()
design = svy.Design(stratum="region", psu="cluster", wgt="samp_wgt")
sample = svy.Sample(data=data, design=design)

print(f"Rows: {data.height}")

# ── D0: Baseline — no where, no by ───────────────────────────────────────────
print("\n" + "=" * 70)
print("D0: baseline strat+clust, income ~ sex, KW, no where")
print("=" * 70)
r = sample.categorical.ranktest(
    y="income",
    group="sex",
    method=RankScoreMethod.KRUSKAL_WALLIS,
    drop_nulls=True,
)
print(r)

# ── D1: where income > 50000 ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("D1: strat+clust, income ~ sex, KW, where income > 50000")
print("=" * 70)
r = sample.categorical.ranktest(
    y="income",
    group="sex",
    method=RankScoreMethod.KRUSKAL_WALLIS,
    drop_nulls=True,
    where=svy.col("income") > 50000,
)
print(r)

# ── D3: wgt only, where income > 50000 ───────────────────────────────────────
print("\n" + "=" * 70)
print("D3: wgt only, income ~ sex, KW, where income > 50000")
print("=" * 70)
design_wgt = svy.Design(wgt="samp_wgt")
sample_wgt = svy.Sample(data=data, design=design_wgt)
r = sample_wgt.categorical.ranktest(
    y="income",
    group="sex",
    method=RankScoreMethod.KRUSKAL_WALLIS,
    drop_nulls=True,
    where=svy.col("income") > 50000,
)
print(r)

# ── D4: wgt only, k-sample, where income > 50000 ─────────────────────────────
print("\n" + "=" * 70)
print("D4: wgt only, income ~ region, KW k-sample, where income > 50000")
print("=" * 70)
r = sample_wgt.categorical.ranktest(
    y="income",
    group="region",
    method=RankScoreMethod.KRUSKAL_WALLIS,
    drop_nulls=True,
    where=svy.col("income") > 50000,
)
print(r)

# ── D5: clust only, k-sample, where income > 50000 ───────────────────────────
print("\n" + "=" * 70)
print("D5: clust only, income ~ region, KW k-sample, where income > 50000")
print("=" * 70)
design_clust = svy.Design(psu="cluster", wgt="samp_wgt")
sample_clust = svy.Sample(data=data, design=design_clust)
r = sample_clust.categorical.ranktest(
    y="income",
    group="region",
    method=RankScoreMethod.KRUSKAL_WALLIS,
    drop_nulls=True,
    where=svy.col("income") > 50000,
)
print(r)

# ── D6: VdW, where income > 50000 ────────────────────────────────────────────
print("\n" + "=" * 70)
print("D6: strat+clust, income ~ sex, VdW, where income > 50000")
print("=" * 70)
r = sample.categorical.ranktest(
    y="income",
    group="sex",
    method=RankScoreMethod.VANDER_WAERDEN,
    drop_nulls=True,
    where=svy.col("income") > 50000,
)
print(r)

# ── D7: where educ == Postgraduate ───────────────────────────────────────────
print("\n" + "=" * 70)
print("D7: strat+clust, income ~ sex, KW, where educ == Postgraduate")
print("=" * 70)
r = sample.categorical.ranktest(
    y="income",
    group="sex",
    method=RankScoreMethod.KRUSKAL_WALLIS,
    drop_nulls=True,
    where=svy.col("educ") == "Postgraduate",
)
print(r)

# ── D8: where + by combinations ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("D8: where + by combinations")
print("=" * 70)

print("\nD8a: strat+clust, income ~ sex, KW, where income>50000, by=region")
r = sample.categorical.ranktest(
    y="income",
    group="sex",
    method=RankScoreMethod.KRUSKAL_WALLIS,
    drop_nulls=True,
    where=svy.col("income") > 50000,
    by="region",
)
print(r)

print("\nD8b: wgt only, income ~ sex, KW, where income>50000, by=region")
r = sample_wgt.categorical.ranktest(
    y="income",
    group="sex",
    method=RankScoreMethod.KRUSKAL_WALLIS,
    drop_nulls=True,
    where=svy.col("income") > 50000,
    by="region",
)
print(r)

print("\nD8c: strat+clust, income ~ sex, KW, where sex==2, by=educ")
r = sample.categorical.ranktest(
    y="income",
    group="sex",
    method=RankScoreMethod.KRUSKAL_WALLIS,
    drop_nulls=True,
    where=svy.col("sex") == 2,
    by="educ",
)
print(r)

print("\nD8d: strat+clust, income ~ sex, KW, where educ==Postgraduate, by=region")
r = sample.categorical.ranktest(
    y="income",
    group="sex",
    method=RankScoreMethod.KRUSKAL_WALLIS,
    drop_nulls=True,
    where=svy.col("educ") == "Postgraduate",
    by="region",
)
print(r)
