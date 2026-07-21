"""
Generate the small, self-consistent BUNDLED subsets of the WB synthetic
datasets that ship inside the svy wheel for offline use.

These bundled files are deliberately small (~0.5 MB total) so they can travel
with the package.  They are a REDUCED view of the full online datasets:

  * 19 strata (geo1 x urbrur) with an unequal EA (PSU) allocation
    (one stratum with 5 EAs, two with 3, the rest with 2) so the design is
    realistic and supports variance estimation with no singleton strata.
  * Bundled SAMPLE  = the sampled households in those EAs (+ their persons).
  * Bundled CENSUS  = a reduced census: within each retained EA we keep ALL
    sampled households, then top up with a random draw to a per-EA target that
    varies around ~80 households; a few EAs are left unrestricted (full census)
    as heavy-tail outliers.  So the bundled sample is always a subset of the
    bundled census.
  * ea_frame        = those EAs, with n_hlds_census recomputed to the reduced
    census counts so the frame reconciles exactly with the bundled census.

Everything is deterministic (fixed seeds) so re-running reproduces byte-identical
files.  Requires the full canonical parquet masters, which are NOT in the repo;
point SVY_WB_SRC at the directory that holds them.

Usage
-----
    SVY_WB_SRC=/path/to/wb/synthetic/v1.0.0 \
        uv run python scripts/build_bundled_datasets.py

Default SVY_WB_SRC is the local svylab data checkout.
"""

from __future__ import annotations

import hashlib
import json
import os

from pathlib import Path

import numpy as np
import polars as pl


# --- paths ---------------------------------------------------------------- #
SRC = Path(
    os.getenv(
        "SVY_WB_SRC",
        "/Users/msdiallo/DEV/samplics-org/svylab/data/datasets/examples/wb/synthetic/v1.0.0",
    )
)
OUT = Path(__file__).resolve().parents[1] / "src" / "svy" / "datasets" / "_bundled"
OUT.mkdir(parents=True, exist_ok=True)

# --- knobs (change here, then regenerate + refresh benchmark) ------------- #
TARGET_MEAN = 80  # avg census households per EA (incl. sampled)
TARGET_SD = 14  # spread around the mean
TARGET_FLOOR = 45  # never fewer than this (keeps census > sample per EA)
N_UNRESTRICTED = 3  # a few EAs kept as FULL census (heavy-tail outliers)
EA_ALLOC = {0: 5, 1: 3, 2: 3}  # stratum index -> #EAs; default 2 otherwise
EA_SEED = 12345
CENSUS_SEED = 999
VERSION = "1.0.0"

# --- static per-dataset metadata for the bundled registry ----------------- #
_CITATION = (
    "World Bank (2023). Synthetic Data for Household Surveys and Census "
    "(World Bank Synthetic Data 2023). https://doi.org/10.48529/78M1-AE09"
)
_SOURCE = "World Bank Microdata Library (catalog 5906)"
_LICENSE = "World Bank synthetic microdata terms"
_META = {
    "ea_frame_wb_2023": dict(
        title="WB Synthetic EA Frame 2023 (bundled subset)",
        description="Enumeration-area sampling frame: geography, urban/rural, and census household counts.",
        design=None,
        tags=("wb", "synthetic", "frame", "bundled"),
    ),
    "hld_sample_wb_2023": dict(
        title="WB Synthetic Household Sample 2023 (bundled subset)",
        description="Stratified two-stage household survey sample with design weights.",
        design={"stratum": ["geo1", "urbrur"], "psu": "ea", "wgt": "hhweight"},
        tags=("wb", "synthetic", "household", "sample", "bundled"),
    ),
    "ind_sample_wb_2023": dict(
        title="WB Synthetic Individual Sample 2023 (bundled subset)",
        description="Individual-level survey records; join to the household sample on 'hid' for design variables.",
        design=None,
        tags=("wb", "synthetic", "individual", "sample", "bundled"),
    ),
    "hld_pop_wb_2023": dict(
        title="WB Synthetic Household Census 2023 (bundled, reduced)",
        description="Reduced household census: full population of retained EAs subsampled around 80 HH/EA (sampled HHs always kept).",
        design=None,
        tags=("wb", "synthetic", "household", "census", "reduced", "bundled"),
    ),
    "ind_pop_wb_2023": dict(
        title="WB Synthetic Individual Census 2023 (bundled, reduced)",
        description="Individual records for the reduced household census; join on 'hid'.",
        design=None,
        tags=("wb", "synthetic", "individual", "census", "reduced", "bundled"),
    ),
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Canonical WB source not found: {SRC}\nSet SVY_WB_SRC.")

    hld_s = pl.read_parquet(SRC / "hld_sample_wb_2023.parquet")
    ind_s = pl.read_parquet(SRC / "ind_sample_wb_2023.parquet")
    ea = pl.read_parquet(SRC / "ea_frame_wb_2023.parquet")

    # 1. deterministic EA allocation across strata --------------------------
    sa_ea = hld_s.group_by("ea").agg(pl.col("geo1").first(), pl.col("urbrur").first())
    strata = sa_ea.select("geo1", "urbrur").unique().sort(["geo1", "urbrur"])

    rng = np.random.default_rng(EA_SEED)
    selected: list = []
    for i, (g, u) in enumerate(strata.iter_rows()):
        n = EA_ALLOC.get(i, 2)
        eas = sorted(sa_ea.filter((pl.col("geo1") == g) & (pl.col("urbrur") == u))["ea"].to_list())
        n = min(n, len(eas))
        idx = sorted(rng.choice(len(eas), size=n, replace=False))
        selected += [eas[k] for k in idx]
    selected = sorted(selected)
    sel_set = set(selected)

    # 2. bundled sample -----------------------------------------------------
    b_hld_s = hld_s.filter(pl.col("ea").is_in(sel_set)).sort(["ea", "hid"])
    samp_hids = set(b_hld_s["hid"].to_list())
    b_ind_s = ind_s.filter(pl.col("hid").is_in(samp_hids)).sort(list(ind_s.columns[:2]))

    # 3. reduced census (keep sampled HHs; random top-up to a varied target)
    hld_p_full = (
        pl.scan_parquet(SRC / "hld_pop_wb_2023.parquet")
        .filter(pl.col("ea").is_in(sel_set))
        .collect()
    )
    crng = np.random.default_rng(CENSUS_SEED)
    unrestricted = set(
        crng.choice(selected, size=min(N_UNRESTRICTED, len(selected)), replace=False).tolist()
    )
    keep_parts = []
    for (e,), sub in hld_p_full.group_by(["ea"], maintain_order=True):
        sub = sub.sort("hid")
        samp = sub.filter(pl.col("hid").is_in(samp_hids))
        rest = sub.filter(~pl.col("hid").is_in(samp_hids))
        if e in unrestricted:
            target = sub.height
        else:
            t = int(round(crng.normal(TARGET_MEAN, TARGET_SD)))
            target = max(TARGET_FLOOR, samp.height, min(t, sub.height))
        room = max(0, target - samp.height)
        if rest.height > room:
            idx = sorted(crng.choice(rest.height, size=room, replace=False))
            rest = rest[idx]
        keep_parts.append(pl.concat([samp, rest]))
    b_hld_p = pl.concat(keep_parts).sort(["ea", "hid"])
    cen_hids = set(b_hld_p["hid"].to_list())

    ind_cols = pl.read_parquet(SRC / "ind_pop_wb_2023.parquet", n_rows=1).columns
    b_ind_p = (
        pl.scan_parquet(SRC / "ind_pop_wb_2023.parquet")
        .filter(pl.col("hid").is_in(cen_hids))
        .collect()
        .sort(list(ind_cols[:2]))
    )

    # 4. ea_frame subset, reconcile n_hlds_census ---------------------------
    cen_counts = b_hld_p.group_by("ea").len().rename({"len": "n_hlds_census_reduced"})
    b_ea = (
        ea.filter(pl.col("ea").is_in(sel_set))
        .drop("n_hlds_census")
        .join(cen_counts, on="ea", how="left")
        .rename({"n_hlds_census_reduced": "n_hlds_census"})
        .sort("ea")
    )

    # 5. write + registry ---------------------------------------------------
    files = {
        "ea_frame_wb_2023": b_ea,
        "hld_sample_wb_2023": b_hld_s,
        "ind_sample_wb_2023": b_ind_s,
        "hld_pop_wb_2023": b_hld_p,
        "ind_pop_wb_2023": b_ind_p,
    }
    registry = []
    total = 0
    for slug, df in files.items():
        path = OUT / f"{slug}.parquet"
        df.write_parquet(path, compression="zstd")
        size = path.stat().st_size
        total += size
        meta = _META[slug]
        registry.append(
            {
                "slug": slug,
                "title": meta["title"],
                "description": meta["description"],
                "version": VERSION,
                "filename": f"{slug}.parquet",
                "sha256": sha256_of(path),
                "size_bytes": size,
                "n_rows": df.height,
                "n_cols": df.width,
                "license": _LICENSE,
                "citation": _CITATION,
                "source": _SOURCE,
                "design": meta["design"],
                "variables": {},
                "tags": list(meta["tags"]),
            }
        )
        print(f"  {slug:22} {df.height:>6} rows x {df.width:>2}  {size / 1024:7.1f} KB")

    (OUT / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print(f"  {'TOTAL':22} {'':>6}          {total / 1024:7.1f} KB  ->  {OUT}")


if __name__ == "__main__":
    main()
