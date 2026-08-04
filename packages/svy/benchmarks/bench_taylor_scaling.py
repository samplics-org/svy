# benchmarks/bench_taylor_scaling.py
"""Check that Taylor estimation actually uses the cores it is given.

Taylor kernels have plenty of independent work — by-group cells, batched
variables, and (within one estimate) a design build that shares nothing with the
score pass. What limited them was not rayon width but redundant *serial* work
around the fan-out: the design was indexed twice per call, the factorized design
was rebuilt on every ``sample.estimation`` access, and per-estimate reporting
metadata ran ``np.unique`` over full-length design arrays once per variable.

Two things are measured here, because either alone is misleading:

* **cores used** — process CPU time (which sums rayon's threads) over wall time.
  Useful, but it *falls* when redundant parallel work is deleted, so it can only
  be read next to wall time.
* **thread scaling** — wall time at 1 thread over wall time at N. This is the
  property that actually breaks when a kernel silently goes serial (a missing
  ``Python::detach``, a reduction moved back onto one thread), and it is what
  this benchmark asserts on.

Scaling is measured in subprocesses with ``RAYON_NUM_THREADS`` and
``POLARS_MAX_THREADS`` pinned, since both pools are fixed at first use.

Usage:
    python benchmarks/bench_taylor_scaling.py
    python benchmarks/bench_taylor_scaling.py --rows 1000000
    python benchmarks/bench_taylor_scaling.py --skip-scaling   # utilisation only

Exits non-zero if any case scales worse than ``--min-speedup``, so it can be
wired into CI as a coarse guard.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import polars as pl

import svy


SEED = 20260804
N_STRATA = 50
PSU_PER_STRATUM = 40
N_Y_VARS = 8
Y_VARS = [f"y{j}" for j in range(N_Y_VARS)]
ROWS = 1_000_000
TIMING_REPS = 7
WARMUP = 2

# Floors, not targets. Measured on a 10-core M1 Max after the fan-out fixes:
# single-variable 1.48x, batched 2.93x, by-group 3.56x. The floors sit well
# under those so ordinary machine noise cannot trip them, while a kernel that
# regresses to serial (1.0x) fails loudly.
MIN_SPEEDUP = {
    # A single estimate overlaps two halves (design build vs score pass), so its
    # ceiling is ~2x by construction — not a sign of a missing fan-out.
    "mean_1": 1.20,
    "mean_8": 2.00,
    "mean_by": 2.50,
}


def make_sample(n_rows: int) -> svy.Sample:
    """A stratified two-stage sample, unequal weights, cluster random effect."""
    rng = np.random.default_rng(SEED)
    n_psu = N_STRATA * PSU_PER_STRATUM
    # Contiguous PSU blocks that always cover exactly n_rows.
    psu = (np.arange(n_rows, dtype=np.int64) * n_psu) // n_rows
    stratum = psu // PSU_PER_STRATUM
    psu_effect = rng.normal(0.0, 1.0, size=n_psu)[psu]

    x = rng.normal(10.0, 3.0, size=n_rows)
    data = {
        "stratum": stratum,
        "psu": psu,
        "wgt": np.exp(rng.normal(0.0, 0.4, size=n_rows)) * (10.0 + stratum % 7),
        "x": x,
        "grp": rng.integers(0, 12, size=n_rows),
    }
    for j in range(N_Y_VARS):
        data[f"y{j}"] = 2.0 * x + 5.0 * psu_effect + rng.normal(0.0, 4.0, size=n_rows) + j

    df = pl.DataFrame(data)
    return svy.Sample(df, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))


def cases(sample: svy.Sample) -> dict:
    return {
        "mean_1": ("Taylor mean, 1 variable", lambda: sample.estimation.mean("y0")),
        "total_1": ("Taylor total, 1 variable", lambda: sample.estimation.total("y0")),
        "mean_8": ("Taylor mean, 8 batched", lambda: sample.estimation.mean(Y_VARS)),
        "mean_by": ("Taylor mean by group", lambda: sample.estimation.mean("y0", by="grp")),
    }


def measure(fn, reps: int = TIMING_REPS) -> tuple[float, float]:
    """Median wall and CPU seconds. CPU sums every in-process thread, rayon's included."""
    for _ in range(WARMUP):
        fn()
    walls, cpus = [], []
    for _ in range(reps):
        w0, c0 = time.perf_counter(), time.process_time()
        fn()
        walls.append(time.perf_counter() - w0)
        cpus.append(time.process_time() - c0)
    return statistics.median(walls), statistics.median(cpus)


# ── child process: one case, one pinned thread count ─────────────────────────


def _run_pinned() -> None:
    """Entry point for the scaling subprocesses. Prints a single wall time."""
    n_rows = int(os.environ["BENCH_ROWS"])
    case = os.environ["BENCH_CASE"]
    sample = make_sample(n_rows)
    _, fn = cases(sample)[case]
    wall, _ = measure(fn)
    print(json.dumps({"wall_s": wall}))


def thread_scaling(n_rows: int, case: str, threads: list[int]) -> list[float]:
    """Wall time for ``case`` at each pinned thread count, via subprocesses."""
    walls = []
    for t in threads:
        env = dict(
            os.environ,
            BENCH_ROWS=str(n_rows),
            BENCH_CASE=case,
            BENCH_CHILD="1",
            RAYON_NUM_THREADS=str(t),
            POLARS_MAX_THREADS=str(t),
        )
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env=env,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"scaling child failed for {case} at {t} threads:\n{out.stderr[-2000:]}"
            )
        walls.append(json.loads(out.stdout.strip().splitlines()[-1])["wall_s"])
    return walls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument(
        "--min-speedup",
        type=float,
        default=None,
        help="override the per-case scaling floor (default: built-in per case)",
    )
    ap.add_argument("--skip-scaling", action="store_true")
    args = ap.parse_args()

    n_cores = os.cpu_count() or 1
    sample = make_sample(args.rows)

    print(
        f"n = {args.rows:,} rows, {N_STRATA} strata, "
        f"{N_STRATA * PSU_PER_STRATUM} PSUs, {n_cores} cores\n"
    )

    print("utilisation (cores = CPU time / wall time; read next to wall):")
    print(f"  {'case':<26} {'wall':>10} {'cpu':>10} {'cores':>7}")
    for key, (label, fn) in cases(sample).items():
        wall, cpu = measure(fn)
        print(f"  {label:<26} {wall * 1e3:9.1f}m {cpu * 1e3:9.1f}m {cpu / wall:7.2f}")

    if args.skip_scaling:
        return 0

    if n_cores < 4:
        print(f"\nskipping scaling: needs >= 4 cores, this machine has {n_cores}")
        return 0

    top = min(n_cores, 10)
    threads = [1, top]
    print(f"\nthread scaling (1 thread vs {top}):")
    print(f"  {'case':<26} {'1 thr':>10} {f'{top} thr':>10} {'speedup':>9} {'floor':>7}")

    failures = []
    for key in ("mean_1", "mean_8", "mean_by"):
        label = cases(sample)[key][0]
        walls = thread_scaling(args.rows, key, threads)
        speedup = walls[0] / walls[-1]
        floor = args.min_speedup if args.min_speedup is not None else MIN_SPEEDUP[key]
        ok = speedup >= floor
        print(
            f"  {label:<26} {walls[0] * 1e3:9.1f}m {walls[-1] * 1e3:9.1f}m "
            f"{speedup:8.2f}x {floor:6.2f}x {'' if ok else '  <-- REGRESSION'}"
        )
        if not ok:
            failures.append((label, speedup, floor))

    if failures:
        print("\nFAILED — these cases are not using the cores they were given:")
        for label, speedup, floor in failures:
            print(f"  {label}: {speedup:.2f}x vs floor {floor:.2f}x")
        print(
            "\nA case that drops to ~1.0x has gone serial. The usual causes are a "
            "PyO3 entry point that stopped releasing the GIL (`Python::detach`), or "
            "serial work added ahead of the fan-out — indexing the design twice, "
            "rebuilding the factorized design per call, or per-estimate work that "
            "scales with the variable count."
        )
        return 1

    print("\nOK — every case scales above its floor.")
    return 0


if __name__ == "__main__":
    if os.environ.get("BENCH_CHILD"):
        _run_pinned()
    else:
        sys.exit(main())
