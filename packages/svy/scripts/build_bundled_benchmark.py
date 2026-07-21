"""
Freeze a golden benchmark for the bundled datasets.

Run this ONCE whenever the bundled data is (re)generated.  It records, per
bundled dataset: the file sha256/size, shape, column list, a sum of every
numeric column, and — for the household sample — the design-based mean of
``pc_exp``.  ``tests/svy/datasets/test_bundled_benchmark.py`` asserts the live
files still reproduce these values, so any change to the shipped data (or a
regression in the estimation pipeline on this fixture) fails loudly.

    uv run python scripts/build_bundled_benchmark.py

The resulting JSON is committed; a diff to it must be reviewed.
"""

from __future__ import annotations

import hashlib
import json

from importlib.resources import as_file, files
from pathlib import Path

import polars as pl

import svy
import svy.datasets as d


OUT = Path(__file__).resolve().parents[1] / "tests" / "svy" / "datasets" / "bundled_benchmark.json"
_ROUND = 4


def _sha(slug: str, filename: str) -> tuple[str, int]:
    resource = files("svy.datasets") / "_bundled" / filename
    with as_file(resource) as p:
        raw = Path(p).read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _numeric_sums(df: pl.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype.is_numeric():
            val = df[col].sum()
            if val is not None:
                out[col] = round(float(val), _ROUND)
    return out


def main() -> None:
    benchmark: dict[str, dict] = {}
    for ds in d.catalog(source="bundled"):
        df = d.load(ds.slug, source="bundled")
        sha, size = _sha(ds.slug, f"{ds.slug}.parquet")
        entry = {
            "sha256": sha,
            "size_bytes": size,
            "n_rows": df.height,
            "n_cols": df.width,
            "columns": df.columns,
            "numeric_sums": _numeric_sums(df),
        }
        if ds.design:
            smp = svy.Sample(data=df, design=svy.Design(**ds.design))
            est = smp.estimation.mean("pc_exp")
            row = est.to_dicts()[0]
            entry["design_mean_pc_exp"] = {
                "est": round(float(row["est"]), _ROUND),
                "se": round(float(row["se"]), _ROUND),
                "n_strata": int(est.n_strata),
                "n_psus": int(est.n_psus),
            }
        benchmark[ds.slug] = entry

    OUT.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(benchmark)} datasets)")


if __name__ == "__main__":
    main()
