"""Self-contained estimation benchmark harness (PERF_PLAN Phase A).

Two layers, one script:

  * end-to-end  — ms/call for mean/total/ratio/by/domain/replication/tabulate,
    at a small and a large row count, through the full Python + PyO3 + Rust
    stack. This is the number a user actually feels.
  * direct-kernel — calls `_internal.taylor_mean` straight, isolating the Rust
    kernel + marshalling from Python prep: no-design vs short-label vs
    long-label vs a deliberately fragmented (32-chunk) frame.

Unlike the older profile_estimation.py / bench_estimation.py, this generates
its own synthetic survey data (no dependency on the gitignored WB CSVs), so it
runs anywhere and in CI. The design shape matches the PERF_PLAN baseline:
~20 strata (geo1 x urbrur), ~50 PSUs/stratum.

Each case prints a machine-readable line so before/after diffs are scriptable:

    BENCH\t<label>\t<rows>\t<ms>

Usage:
    uv run python benchmarks/bench_kernel.py                 # 32k + 1M
    uv run python benchmarks/bench_kernel.py --rows 32000    # one size
    uv run python benchmarks/bench_kernel.py --reps 9 --reps-large 5
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import polars as pl

from svy_rs import _internal

import svy

from svy import Design, RepWeights, Sample
from svy.core.terms import Cat


N_STRATA = 20  # geo1 (10) x urbrur (2)
PSU_PER_STRATUM = 50


# ── Synthetic data ──────────────────────────────────────────────────────────


def gen_data(n_rows: int, *, n_reps: int = 0, seed: int = 42) -> pl.DataFrame:
    """A survey-shaped frame: 2-column stratum, clustered PSUs, weight, y, x,
    a low-cardinality `by` variable, a domain source, and optional replicate
    weights. Deterministic for a given (n_rows, seed)."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, N_STRATA, n_rows)  # stratum index
    p = rng.integers(0, PSU_PER_STRATUM, n_rows)  # PSU within stratum (reused labels)

    cols = {
        "_s": s,
        "_p": p,
        "hhweight": rng.uniform(0.5, 3.0, n_rows),
        "tot_exp": np.clip(rng.normal(1000.0, 300.0, n_rows), 0.0, None),
        "hhsize": rng.integers(1, 9, n_rows).astype(float),
        "sex": rng.integers(1, 3, n_rows),  # by: 2 levels
        "region": rng.integers(0, 5, n_rows),  # domain source: 5 levels
    }
    # Replicate weights: shape only (values need not be statistically valid to
    # exercise the replicate matrix pass).
    for k in range(n_reps):
        cols[f"rep_{k + 1}"] = cols["hhweight"] * rng.gamma(2.0, 0.5, n_rows)

    df = pl.DataFrame(cols).with_columns(
        ("geo" + (pl.col("_s") // 2).cast(pl.String)).alias("geo1"),  # 10 distinct
        (pl.col("_s") % 2).cast(pl.String).alias("urbrur"),  # 2 distinct
        ("ea" + pl.col("_p").cast(pl.String)).alias("ea"),  # 50, nested in stratum
        pl.col("sex").cast(pl.String),
    )
    return df.drop("_s", "_p").rechunk()  # single chunk: chunking is a controlled variable


# ── Timing ──────────────────────────────────────────────────────────────────


def best_ms(fn, reps: int) -> float:
    """Warm once, then return best-of-`reps` wall-clock in milliseconds."""
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def emit(label: str, rows: int, ms: float) -> None:
    print(f"BENCH\t{label}\t{rows}\t{ms:.3f}")


def emit_case(label: str, rows: int, fn, reps: int) -> None:
    """Time and emit one case, skipping it if this svy version lacks the API.

    Skipping rather than failing is what lets the same harness run against an
    older release to record a historical baseline: a version predating
    ``corr``/``cov`` simply omits those lines, and the comparison tool reports
    them as new rather than as a regression.
    """
    try:
        emit(label, rows, best_ms(fn, reps))
    except (AttributeError, TypeError, NotImplementedError) as exc:
        print(f"SKIP\t{label}\t{rows}\t{type(exc).__name__}: {exc}", file=sys.stderr)


# ── End-to-end cases (full Python + PyO3 + Rust) ────────────────────────────


def run_end_to_end(n_rows: int, reps: int, n_reps: int) -> None:
    df = gen_data(n_rows, n_reps=n_reps)

    design = Design(stratum=("geo1", "urbrur"), psu="ea", wgt="hhweight")
    sample = Sample(data=df, design=design)
    est = sample.estimation

    cases = {
        "mean/strat+cluster": lambda: est.mean(y="tot_exp"),
        "total/strat+cluster": lambda: est.total(y="tot_exp"),
        "ratio/strat+cluster": lambda: est.ratio(y="tot_exp", x="hhsize"),
        "mean/by=sex": lambda: est.mean(y="tot_exp", by="sex"),
        "mean/domain(region==0)": lambda: est.mean(y="tot_exp", where=svy.col("region") == 0),
        # Estimators beyond the original PERF_PLAN set. `prop` and the quantile
        # family have their own kernels; corr/cov share a moment kernel and are
        # absent before 0.23, so they skip cleanly on older releases.
        "prop/sex": lambda: est.prop(y="sex"),
        "median/tot_exp": lambda: est.median(y="tot_exp"),
        "quantile/tot_exp": lambda: est.quantile(y="tot_exp", p=(0.25, 0.5, 0.75)),
        "corr/tot_exp~hhsize": lambda: est.corr(("tot_exp", "hhsize")),
        "cov/tot_exp~hhsize": lambda: est.cov(("tot_exp", "hhsize")),
        # Batching shares one design build across variables; regressions here
        # show up as the batched call drifting toward the looped cost.
        "mean/batched(2 vars)": lambda: est.mean(y=["tot_exp", "hhsize"]),
        "mean/looped(2 vars)": lambda: [est.mean(y=v) for v in ("tot_exp", "hhsize")],
    }
    for label, fn in cases.items():
        emit_case(label, n_rows, fn, reps)

    # Replication path (matrix pass scales differently from Taylor).
    if n_reps > 0:
        rep_design = Design(
            stratum=("geo1", "urbrur"),
            psu="ea",
            wgt="hhweight",
            rep_wgts=RepWeights(method="Bootstrap", prefix="rep_", n_reps=n_reps),
        )
        rep_sample = Sample(data=df, design=rep_design)
        rep_est = rep_sample.estimation
        emit(
            f"mean/replication(R={n_reps})",
            n_rows,
            best_ms(lambda: rep_est.mean(y="tot_exp", method="replication"), reps),
        )

    # Calibrated designs: the score-centring sweep is variance-path work that
    # none of the cases above reach, because they all use an unadjusted design.
    # One case per sweep kind, since the three are different formulas -- cells
    # means, ten iterations over margins, and a WLS residual. The adjustment
    # itself is built outside the timed region; what is timed is estimation on
    # the adjusted design.
    try:
        w_tot = float(df["hhweight"].sum())
        sex_lv = sorted(df["sex"].unique().to_list())
        reg_lv = sorted(df["region"].unique().to_list())

        ps_sample = sample.weighting.poststratify(
            {v: w_tot / len(sex_lv) for v in sex_lv}, cells="sex"
        )
        rk_sample = sample.weighting.rake(
            shares={
                "sex": dict.fromkeys(sex_lv, 1.0),
                "region": dict.fromkeys(reg_lv, 1.0),
            }
        )
        cal_sample = sample.weighting.calibrate(
            controls={Cat("sex"): {v: w_tot / len(sex_lv) for v in sex_lv}}
        )
    except (AttributeError, TypeError, NotImplementedError) as exc:
        print(f"SKIP\tcalibrated/*\t{n_rows}\t{type(exc).__name__}: {exc}", file=sys.stderr)
    else:
        ps_est, rk_est, cal_est = (
            ps_sample.estimation,
            rk_sample.estimation,
            cal_sample.estimation,
        )
        calibrated_cases = {
            # cells sweep: one pass for the cell means, one to subtract them
            "mean/poststratified": lambda: ps_est.mean(y="tot_exp"),
            "total/poststratified": lambda: ps_est.total(y="tot_exp"),
            # grouped: the sweep is redone per group, restricted to its rows
            "mean/poststratified by=sex": lambda: ps_est.mean(y="tot_exp", by="sex"),
            # raking sweeps every margin, ten times over -- the costliest kind
            "mean/raked": lambda: rk_est.mean(y="tot_exp"),
            # GREG solves a k x k weighted least squares per score vector
            "mean/calibrated": lambda: cal_est.mean(y="tot_exp"),
        }
        for label, fn in calibrated_cases.items():
            emit_case(label, n_rows, fn, reps)

    # One tabulate case (categorical path).
    emit(
        "tabulate/sex",
        n_rows,
        best_ms(lambda: sample.categorical.tabulate("sex"), reps),
    )


# ── Direct-kernel probe (isolates Rust kernel + marshalling) ────────────────


def run_direct_kernel(n_rows: int, reps: int) -> None:
    rng = np.random.default_rng(7)
    strata_i = rng.integers(0, N_STRATA, n_rows)
    psu_i = rng.integers(0, PSU_PER_STRATUM, n_rows)
    y = rng.normal(100.0, 20.0, n_rows)
    w = rng.uniform(0.5, 2.0, n_rows)

    base = pl.DataFrame({"y": y, "w": w}).rechunk()
    df_short = base.with_columns(
        pl.Series("s", strata_i.astype("U3")),
        pl.Series("p", psu_i.astype("U3")),
    ).rechunk()
    df_long = base.with_columns(
        pl.Series("s", np.char.add("stratum_region_", strata_i.astype("U3"))),
        pl.Series("p", np.char.add("enumeration_area_", psu_i.astype("U4"))),
    ).rechunk()
    # Deliberately fragmented input (32 chunks) to measure the chunking penalty.
    n32 = n_rows // 32 or 1
    df_chunked = pl.concat([df_short.slice(i * n32, n32) for i in range(32)], rechunk=False)

    def call(df, s, p):
        return lambda: _internal.taylor_mean(df, "y", "w", s, p, None, None, None, None, None)

    emit("kernel/no-design", n_rows, best_ms(call(base, None, None), reps))
    emit("kernel/short-labels", n_rows, best_ms(call(df_short, "s", "p"), reps))
    emit("kernel/long-labels", n_rows, best_ms(call(df_long, "s", "p"), reps))
    emit("kernel/short-32chunks", n_rows, best_ms(call(df_chunked, "s", "p"), reps))


# ── Driver ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[32_000, 1_000_000],
        help="row counts to benchmark (default: 32000 1000000)",
    )
    ap.add_argument("--reps", type=int, default=9, help="best-of-N reps at small sizes")
    ap.add_argument("--reps-large", type=int, default=5, help="best-of-N reps at >=1M rows")
    ap.add_argument("--n-reps", type=int, default=40, help="replicate weights for the rep path")
    args = ap.parse_args()

    print(f"svy {svy.__version__}  |  columns tab-separated: BENCH<TAB>label<TAB>rows<TAB>ms\n")
    for n_rows in args.rows:
        reps = args.reps_large if n_rows >= 1_000_000 else args.reps
        print(f"# ── {n_rows:,} rows (best of {reps}) ──")
        run_end_to_end(n_rows, reps, args.n_reps)
        run_direct_kernel(n_rows, reps)
        print()


if __name__ == "__main__":
    main()
