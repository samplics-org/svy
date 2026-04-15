"""
Check Design printing across different configurations.
Run with:
    uv run python scripts/print_design.py
"""

from pathlib import Path
import polars as pl
import svy
from svy.core.design import Design, PopSize, RepWeights

BASE_DIR = Path("tests/test_data")
DATA_PATH = BASE_DIR / "svy_synthetic_sample_07082025.csv"
data = svy.read_csv(DATA_PATH).fill_nan(None).drop_nulls()

# ── D0: Simple weighted design ───────────────────────────────────────────────
print("=" * 60)
print("D0: simple strat+clust+wgt")
print("=" * 60)
d0 = Design(stratum="region", psu="cluster", wgt="samp_wgt")
print(d0)

# ── D1: Weight only ───────────────────────────────────────────────────────────
print("=" * 60)
print("D1: weight only")
print("=" * 60)
d1 = Design(wgt="samp_wgt")
print(d1)

# ── D2: Multi-column stratum and PSU (tuples) ─────────────────────────────────
print("=" * 60)
print("D2: tuple stratum + tuple psu")
print("=" * 60)
d2 = Design(
    stratum=("region", "educ"),
    psu=("cluster", "sex"),
    wgt="samp_wgt",
)
print(d2)

# ── D3: With row_index, wr, prob, hit, mos ────────────────────────────────────
print("=" * 60)
print("D3: full specification")
print("=" * 60)
d3 = Design(
    row_index="svy_row_index",
    stratum="region",
    psu="cluster",
    wgt="samp_wgt",
    prob="samp_wgt",
    hit="sex",
    mos="fam_size",
    wr=True,
)
print(d3)

# ── D4: With FPC (pop_size as string) ─────────────────────────────────────────
print("=" * 60)
print("D4: with pop_size string (FPC)")
print("=" * 60)
d4 = Design(
    stratum="region",
    psu="cluster",
    wgt="samp_wgt",
    pop_size="N_psu",
)
print(d4)

# ── D5: With FPC (PopSize namedtuple) ─────────────────────────────────────────
print("=" * 60)
print("D5: with PopSize(psu=, ssu=) (two-stage FPC)")
print("=" * 60)
d5 = Design(
    stratum="region",
    psu="cluster",
    ssu="sex",
    wgt="samp_wgt",
    pop_size=PopSize(psu="N_psu", ssu="N_ssu"),
)
print(d5)

# ── D6: With replicate weights (BRR) ─────────────────────────────────────────
print("=" * 60)
print("D6: with BRR replicate weights")
print("=" * 60)
d6 = Design(
    wgt="samp_wgt",
    rep_wgts=RepWeights(
        prefix="brr_",
        n_reps=8,
        method="BRR",
        df=7,
    ),
)
print(d6)

# ── D7: With replicate weights (Jackknife) ────────────────────────────────────
print("=" * 60)
print("D7: with Jackknife replicate weights")
print("=" * 60)
d7 = Design(
    stratum="region",
    psu="cluster",
    wgt="samp_wgt",
    rep_wgts=RepWeights(
        prefix="jk_",
        n_reps=56,
        method="Jackknife",
    ),
)
print(d7)

# ── D8: set_default_print_width ───────────────────────────────────────────────
print("=" * 60)
print("D8: narrow print width (50 chars)")
print("=" * 60)
Design.set_default_print_width(50)
print(d2)
Design.set_default_print_width(None)  # reset
