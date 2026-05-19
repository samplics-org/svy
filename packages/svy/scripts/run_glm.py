"""
Diff R's intermediate quantities against the Python reconstruction.

Each "stage" represents one step in the sandwich variance computation.
The first stage where R and Python disagree at >1e-10 localizes the
1e-5 discrepancy we've been chasing.

Run from project root after the R script has written CSVs to /tmp/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# ----- Python reconstruction -----
DATA_PATH = Path("tests/test_data/apistrat.csv")
df = pl.read_csv(DATA_PATH)
df = df.with_columns((pl.col("api00") > 743).cast(pl.Float64).alias("y_bin"))

y = df["y_bin"].to_numpy()
X = np.column_stack(
    [
        np.ones(len(df)),
        df["ell"].to_numpy(),
        df["meals"].to_numpy(),
        df["mobility"].to_numpy(),
    ]
)
w_raw = df["pw"].to_numpy()
n = len(df)
w_prior_py = w_raw / np.mean(w_raw)  # R's rescale=TRUE

beta = np.array(
    [
        2.66575599278362,
        -0.03703249576700,
        -0.08468787757247,
        -0.00442859225861,
    ]
)

eta_py = X @ beta
mu_py = 1.0 / (1.0 + np.exp(-eta_py))
mu_eta_py = mu_py * (1.0 - mu_py)
w_irls_py = w_prior_py * mu_eta_py

Ainv_py = np.linalg.inv(X.T @ (w_irls_py[:, None] * X))
estfun_py = X * ((y - mu_py) * w_prior_py)[:, None]
infl_py = estfun_py @ Ainv_py
infl_centered_py = infl_py - infl_py.mean(axis=0)
vcov_py = (infl_centered_py.T @ infl_centered_py) * (n / (n - 1))


# ----- Load R intermediates -----
TMP = Path("/tmp")
r_rows = pd.read_csv(TMP / "r_per_row.csv")
r_estfun = pd.read_csv(TMP / "r_estfun.csv").values
r_infl = pd.read_csv(TMP / "r_infl.csv").values
r_Ainv = pd.read_csv(TMP / "r_Ainv.csv").values
r_vcov = pd.read_csv(TMP / "r_vcov.csv").values


def diff(label, A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != B.shape:
        print(f"{label:35s} SHAPE MISMATCH: R={A.shape}, py={B.shape}")
        return
    d = np.abs(A - B)
    max_abs = d.max()
    denom = np.maximum(np.abs(A), np.abs(B))
    nonzero = denom > 0
    max_rel = (d[nonzero] / denom[nonzero]).max() if nonzero.any() else 0.0
    marker = ""
    if max_rel > 1e-7:
        marker = "  <-- DIVERGENCE"
    print(f"{label:35s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}{marker}")


print("=" * 78)
print("Stage-by-stage diff: R vs Python reconstruction")
print("=" * 78)
print(f"R rows in CSV: {len(r_rows)}, Python rows: {n}")
print()

diff("eta (linear predictor)", r_rows["eta"].values, eta_py)
diff("mu (fitted)", r_rows["mu"].values, mu_py)
diff("mu_eta (link deriv)", r_rows["mu_eta"].values, mu_eta_py)
diff("w_prior (survey weights)", r_rows["w_prior"].values, w_prior_py)
diff("w_irls (IRLS weights)", r_rows["w_irls"].values, w_irls_py)
diff("y (response)", r_rows["y"].values, y)
print()
diff("Ainv (bread, 4x4)", r_Ainv, Ainv_py)
diff("estfun (n x k)", r_estfun, estfun_py)
diff("infl = estfun @ Ainv", r_infl, infl_py)
print()
diff("FINAL vcov", r_vcov, vcov_py)

# Also report SEs for context
print()
print("SE comparison:")
print(f"{'coef':<14}  {'R':>22}  {'Python recon':>22}")
se_R = np.sqrt(np.diag(r_vcov))
se_py = np.sqrt(np.diag(vcov_py))
for i, name in enumerate(["(Intercept)", "ell", "meals", "mobility"]):
    print(f"{name:<14}  {se_R[i]:>22.15g}  {se_py[i]:>22.15g}")
