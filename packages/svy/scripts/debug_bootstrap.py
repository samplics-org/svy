"""
Minimal determinism test for svy.weighting.create_bs_wgts.

Builds a tiny deterministic sample and calls create_bs_wgts in a loop,
always with the same seed. Prints a fingerprint of the bootstrap weights
from each run.

If all fingerprints are identical, create_bs_wgts honors its seed.
If any differ, the function has a nondeterminism bug.
"""

import hashlib

import numpy as np
import polars as pl

import svy


# ----------------------------------------------------------------------------
# Tiny fixed dataset: 10 rows, 2 strata, 2 PSUs per stratum
# ----------------------------------------------------------------------------

data = pl.DataFrame(
    {
        "stratum": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "psu": ["a1", "a1", "a2", "a2", "a2", "b1", "b1", "b2", "b2", "b2"],
        "wgt": [1.3, 1.4, 8.0, 7.7, 1.0, 166.0, 12, 31.0, 12.0, 91.0],
    }
)

design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")


# ----------------------------------------------------------------------------
# Fingerprint helper
# ----------------------------------------------------------------------------


def fingerprint(df: pl.DataFrame, n_reps: int) -> tuple[str, float]:
    cols = [f"wgt{i}" for i in range(1, n_reps + 1)]
    m = df.select(cols).to_numpy()
    sha = hashlib.sha256(m.tobytes()).hexdigest()[:16]
    return sha, float(m.sum())


# ----------------------------------------------------------------------------
# Run create_bs_wgts 5 times with the SAME integer seed
# ----------------------------------------------------------------------------

print("=" * 60)
print("Test A: integer seed (rstate=147)")
print("=" * 60)

N_REPS = 20
SEED = 147

prints_A = []
for run in range(1, 6):
    sample = svy.Sample(data=data, design=design)
    sample = sample.weighting.create_bs_wgts(n_reps=N_REPS, rstate=SEED)
    sha, total = fingerprint(sample.data, N_REPS)
    prints_A.append(sha)
    print(f"  Run {run}: sha={sha}  sum={total:.6f}")

all_match_A = len(set(prints_A)) == 1
print(f"  All match: {all_match_A}")


# ----------------------------------------------------------------------------
# Run create_bs_wgts 5 times with the SAME Generator (recreated each time)
# ----------------------------------------------------------------------------

print()
print("=" * 60)
print("Test B: Generator from same seed (rstate=default_rng(147))")
print("=" * 60)

prints_B = []
for run in range(1, 6):
    sample = svy.Sample(data=data, design=design)
    rng = np.random.default_rng(147)  # fresh Generator each time
    sample = sample.weighting.create_bs_wgts(n_reps=N_REPS, rstate=rng)
    sha, total = fingerprint(sample.data, N_REPS)
    prints_B.append(sha)
    print(f"  Run {run}: sha={sha}  sum={total:.6f}")

all_match_B = len(set(prints_B)) == 1
print(f"  All match: {all_match_B}")


# ----------------------------------------------------------------------------
# Interpretation
# ----------------------------------------------------------------------------

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  Test A (int seed):        {'PASS' if all_match_A else 'FAIL'}")
print(f"  Test B (Generator seed):  {'PASS' if all_match_B else 'FAIL'}")

if all_match_A and all_match_B:
    print("  → create_bs_wgts IS deterministic.")
    print("    The bug must be elsewhere in replication.py.")
elif all_match_A and not all_match_B:
    print("  → create_bs_wgts is deterministic with int but not Generator.")
    print("    Use rstate=int in replication.py as a workaround.")
elif not all_match_A and not all_match_B:
    print("  → create_bs_wgts is NONdeterministic even with fixed seed.")
    print("    This is a real bug in svy that needs fixing.")
else:
    print("  → Unexpected pattern. Look at the raw fingerprints.")
