"""
Print exact variance values for specific score pairs.
Run this, then add matching prints to Rust to find the discrepancy.
"""

import numpy as np
import polars as pl

from svy.engine.estimation.taylor import (
    _taylor_variance,
    _get_dummies_and_categories,
)

df = pl.read_csv("tests/test_data/svy_synthetic_sample_07082025.csv")
df = df.fill_nan(None).drop_nulls()
print(f"Rows: {df.height}")

# Clustered design
y_keys = (df["educ"].cast(pl.Utf8) + "__by__" + df["sex"].cast(pl.Utf8)).to_numpy()
wgt = df["samp_wgt"].to_numpy().astype(float)
psu = df["cluster"].to_numpy()
n = len(wgt)

# Create dummies and scores
Y, categories = _get_dummies_and_categories(y=y_keys, prop_positive=1, ensure_positive_first=True)
Y = Y.astype(float)
k = len(categories)
sum_w = wgt.sum()
props = (wgt[:, None] * Y).sum(axis=0) / sum_w
scores = (wgt[:, None] * (Y - props)) / sum_w

print(f"Categories: {categories}")
print(f"n_psus: {len(np.unique(psu))}")

# Print variances and cross-covariance for pair (0,1)
var_0 = float(_taylor_variance(y_score=scores[:, 0:1], wgt=wgt, stratum=None, psu=psu)[0, 0])
var_1 = float(_taylor_variance(y_score=scores[:, 1:2], wgt=wgt, stratum=None, psu=psu)[0, 0])

sum_01 = (scores[:, 0] + scores[:, 1])[:, None]
var_sum_01 = float(_taylor_variance(y_score=sum_01, wgt=wgt, stratum=None, psu=psu)[0, 0])
cov_01_polar = (var_sum_01 - var_0 - var_1) / 2.0

# Single-pass for reference
cov_full = _taylor_variance(y_score=scores[:, 0:2], wgt=wgt, stratum=None, psu=psu)
cov_01_single = float(cov_full[0, 1])

print(f"\nVar(s0):     {var_0:.18e}")
print(f"Var(s1):     {var_1:.18e}")
print(f"Var(s0+s1):  {var_sum_01:.18e}")
print(f"Cov(0,1) polar:  {cov_01_polar:.18e}")
print(f"Cov(0,1) single: {cov_01_single:.18e}")
print(f"Diff: {abs(cov_01_polar - cov_01_single):.2e}")

# Also print some PSU totals for verification
print(f"\n--- PSU total verification (first 5 PSUs) ---")
unique_psus = np.unique(psu)
for p in unique_psus[:5]:
    mask = psu == p
    t0 = scores[mask, 0].sum()
    t1 = scores[mask, 1].sum()
    ts = sum_01[mask, 0].sum()
    print(f"  PSU {p}: t0={t0:.15e}, t1={t1:.15e}, t_sum={ts:.15e}, t0+t1={t0 + t1:.15e}")

# Print score stats
print(f"\nScore[0] sum: {scores[:, 0].sum():.18e}")
print(f"Score[1] sum: {scores[:, 1].sum():.18e}")
print(f"Sum score sum: {sum_01[:, 0].sum():.18e}")
print(f"Score[0] first 5: {scores[:5, 0]}")
print(f"Score[1] first 5: {scores[:5, 1]}")
print(f"Sum first 5: {sum_01[:5, 0]}")
