"""Sample construction and wrangling cost across frame shapes, small to DHS-wide.

- Prints ``BENCH\t<size>\t<op>\t<ms>`` lines (best of k); ``--json`` saves them.
- ``--src <dir>`` imports svy from another checkout's ``src`` for before/after runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time


SIZES = {  # rows, columns, repeats
    "small": (1_000, 50, 20),
    "medium": (100_000, 800, 5),
    "large": (1_000_000, 200, 3),
    "wide": (20_000, 4_000, 3),
}


def _frame(n, p):
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(1)
    cols = {
        "hhid": np.arange(n),
        "stratum": rng.integers(0, 50, n),
        "psu": rng.integers(0, 1000, n),
        "wgt": rng.uniform(0.5, 3, n),
    }
    for j in range(p):
        cols[f"v{j}"] = rng.integers(0, 5, n) if j % 2 else rng.normal(size=n)
    hh = pl.DataFrame({"hhid": np.arange(n), "hhsize": rng.integers(1, 10, n)})
    return pl.DataFrame(cols), hh


def _best(f, k):
    times = []
    for _ in range(k):
        gc.collect()
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return min(times) * 1e3


def run(svy, size):
    n, p, k = SIZES[size]
    df, hh = _frame(n, p)
    design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
    res = {"Sample()": _best(lambda: svy.Sample(df, design), k)}
    labels = {f"v{j}": {c: f"lab{c}" for c in range(5)} for j in range(1, p, 4)}
    s = svy.Sample(df, design).wrangling.apply_labels(categories=labels)
    w = s.wrangling

    def chain():
        x = w.mutate({"z": svy.col("v0") * 2})
        x = x.wrangling.filter_records(svy.col("v1") > 1)
        x = x.wrangling.recode("v3", {9: [0, 1]})
        x = x.wrangling.join(hh, on="hhid")
        return x.wrangling.cast("v5", "Int32")

    ops = {
        "mutate": lambda: w.mutate({"z": svy.col("v0") * 2}),
        "cast": lambda: w.cast("v1", "Int32"),
        "recode": lambda: w.recode("v1", {9: [0, 1]}),
        "join m:1": lambda: w.join(hh, on="hhid"),
        "filter_records": lambda: w.filter_records(svy.col("v1") > 1),
        "order_by": lambda: w.order_by("v1"),
        "chain of 5": chain,
        # Untouched by wrangling: a control for machine noise.
        "estimation.mean": lambda: s.estimation.mean("v0"),
    }
    for name, f in ops.items():
        res[name] = _best(f, k)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="+", choices=list(SIZES), default=list(SIZES))
    ap.add_argument("--src", help="svy source dir to import instead of the installed one")
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()

    if args.src:
        sys.path.insert(0, args.src)
    import polars as pl

    import svy

    print(f"svy {svy.__version__} from {svy.__file__}, polars {pl.__version__}")
    out = {"svy": svy.__file__, "polars": pl.__version__, "results": {}}
    for size in args.sizes:
        res = run(svy, size)
        out["results"][size] = res
        for op, ms in res.items():
            print(f"BENCH\t{size}\t{op}\t{ms:.1f}", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
