"""
Generate the small, self-consistent BUNDLED subsets of the WB synthetic
datasets that ship inside the svy wheel for offline use.

The bundle is a coherent *toy country* carved from the full WB synthetic data —
small enough to travel with the package yet complete enough that the whole
workflow runs offline and reproducibly: **frame -> PPS-select EAs -> list
households (SRS) -> analyze**. Because the sample is drawn here (not reused from
the WB survey), its design weights inflate to *this* bundled population, so
weighted estimates match the bundled census truth.

Four files are produced (the individual census is intentionally NOT bundled —
it has no offline use and is the largest file; it stays online via
``source="remote"``):

  * ea_frame_wb_2023   -- sampling frame: N EAs/stratum (varying), with the
    census household count as the PPS measure of size.
  * hld_pop_wb_2023    -- the FULL household census over the frame's EAs (true
    population; a curated column subset).
  * hld_sample_wb_2023 -- a two-stage sample drawn here from the frame/census,
    with design weights (Sum of weights == census households).
  * ind_sample_wb_2023 -- the sample's individuals, enriched with the household
    design variables (geo1/geo2/ea/urbrur) + weight.

Everything is deterministic (fixed seeds). The sample seed is chosen so the
headline estimate lands close to the census truth (a representative — but still
genuine — random sample). Requires the full canonical parquet masters; point
SVY_WB_SRC at the directory holding them.

Usage
-----
    SVY_WB_SRC=/path/to/wb/synthetic/v1.0.0 \
        uv run python scripts/build_bundled_datasets.py
"""

from __future__ import annotations

import hashlib
import json
import os

from pathlib import Path

import numpy as np
import polars as pl

import svy


SRC = Path(
    os.getenv(
        "SVY_WB_SRC",
        "/Users/msdiallo/DEV/samplics-org/svylab/data/datasets/examples/wb/synthetic/v1.0.0",
    )
)
OUT = Path(__file__).resolve().parents[1] / "src" / "svy" / "datasets" / "_bundled"
OUT.mkdir(parents=True, exist_ok=True)

# --- knobs (change here, then regenerate + refresh benchmark) ------------- #
REGIONS_N = 4  # keep the first N geo1 regions -> fewer, non-redundant strata
N_MIN, N_MAX = 15, 20  # frame EAs per stratum (varying)
K_HH = 25  # households selected per EA (stage-2 SRS)
FRAME_SEED = 12345
SAMPLE_SEEDS = range(30)  # candidate sample seeds; the most representative is kept
VERSION = "1.0.0"

# Curated columns (design vars + the variables the tutorials use). The full
# schema stays available online via source="remote".
HH_COLS = [
    "hid",
    "geo1",
    "geo2",
    "ea",
    "urbrur",
    "hhsize",
    "rooms",
    "electricity",
    "water",
    "toilet",
    "tot_exp",
    "tot_food",
    "share_food",
    "pc_exp",
    "quint_nat",
]
IND_COLS = [
    "hid",
    "idno",
    "geo1",
    "geo2",
    "ea",
    "urbrur",
    "sex",
    "age",
    "marstat",
    "educ_attain",
    "yrs_school",
    "literacy",
    "act_status",
    "labor_force",
    "hhweight",
]

_CITATION = (
    "World Bank (2023). Synthetic Data for Household Surveys and Census "
    "(World Bank Synthetic Data 2023). https://doi.org/10.48529/78M1-AE09"
)
_SOURCE = "World Bank Microdata Library (catalog 5906)"
_LICENSE = "World Bank synthetic microdata terms"
_DESIGN_WGT = {"stratum": ["geo1", "urbrur"], "psu": "ea", "wgt": "hhweight"}
_DESIGN_FRAME = {"stratum": ["geo1", "urbrur"], "psu": "ea", "mos": "n_hlds_census"}

_NOTE_COMMON = (
    "Bundled offline subset derived from the full remote dataset: fewer regions "
    "and a curated column set. Load source='remote' for the complete data."
)
_META = {
    "ea_frame_wb_2023": dict(
        title="WB Synthetic EA Frame 2023 (bundled subset)",
        description="Enumeration-area sampling frame: geography, urban/rural, and the census household count used as the PPS measure of size.",
        design=_DESIGN_FRAME,
        tags=("wb", "synthetic", "frame", "bundled"),
        notes=_NOTE_COMMON,
    ),
    "hld_pop_wb_2023": dict(
        title="WB Synthetic Household Census 2023 (bundled)",
        description="The full household census (true population) over the frame's enumeration areas.",
        design=None,
        tags=("wb", "synthetic", "household", "census", "bundled"),
        notes=_NOTE_COMMON,
    ),
    "hld_sample_wb_2023": dict(
        title="WB Synthetic Household Sample 2023 (bundled subset)",
        description="Two-stage (PPS + SRS) household sample with design weights.",
        design=_DESIGN_WGT,
        tags=("wb", "synthetic", "household", "sample", "bundled"),
        notes=(
            _NOTE_COMMON + " The sample is re-drawn from the bundled census, so its weights "
            "match the bundled population (not the national one)."
        ),
    ),
    "ind_sample_wb_2023": dict(
        title="WB Synthetic Individual Sample 2023 (bundled subset)",
        description="Individuals of the sampled households, enriched with household design variables (geo1/geo2/ea/urbrur) and weights.",
        design=_DESIGN_WGT,
        tags=("wb", "synthetic", "individual", "sample", "bundled"),
        notes=(
            _NOTE_COMMON
            + " Re-drawn from the bundled census; weights match the bundled population."
        ),
    ),
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Canonical WB source not found: {SRC}\nSet SVY_WB_SRC.")

    ea = pl.read_parquet(SRC / "ea_frame_wb_2023.parquet")
    regions = sorted(ea["geo1"].unique().to_list())[:REGIONS_N]
    ea5 = ea.filter(pl.col("geo1").is_in(regions))
    strata = ea5.select("geo1", "urbrur").unique().sort(["geo1", "urbrur"])

    # 1. Frame: per stratum pick N in [N_MIN, N_MAX] EAs; M (2..5) scales with N.
    frng = np.random.default_rng(FRAME_SEED)
    frame_eas: list = []
    nm: dict = {}
    for g, u in strata.iter_rows():
        eas = sorted(ea5.filter((pl.col("geo1") == g) & (pl.col("urbrur") == u))["ea"].to_list())
        n = int(frng.integers(N_MIN, N_MAX + 1))
        m = int(np.clip(round(n / 4), 2, 5))
        idx = sorted(frng.choice(len(eas), size=min(n, len(eas)), replace=False))
        frame_eas += [eas[i] for i in idx]
        nm[(g, u)] = (n, m)
    frame_set = set(frame_eas)

    # 2. Full household census over the frame's EAs (curated columns).
    census = (
        pl.scan_parquet(SRC / "hld_pop_wb_2023.parquet")
        .filter(pl.col("ea").is_in(frame_set))
        .collect()
        .select(HH_COLS)
    )
    mos = census.group_by("ea").len().rename({"len": "n_hlds_census"})
    frame = (
        ea5.filter(pl.col("ea").is_in(frame_set))
        .drop("n_hlds_census")
        .join(mos, on="ea", how="left")
        .sort("ea")
    )

    # 3. Self-draw the sample (PPS EAs -> SRS households); keep the seed whose
    #    weighted mean pc_exp is closest to the census truth.
    design = svy.Design(stratum=("geo1", "urbrur"), psu="ea", mos="n_hlds_census")
    n_by = {stratum: m for stratum, (n, m) in nm.items()}
    truth = census["pc_exp"].mean()
    best = None
    for s in SAMPLE_SEEDS:
        # Rebuild the frame Sample each iteration (re-selecting on one Sample
        # accumulates svy_* columns).
        frame_smp = svy.Sample(frame, design)
        ea_sel = frame_smp.sampling.pps_sys(n=n_by, rstate=np.random.default_rng(s))
        sel_eas = set(ea_sel.data["ea"].to_list())
        hld_fr = census.filter(pl.col("ea").is_in(sel_eas))
        hh = ea_sel.sampling.add_stage(next_stage=hld_fr, prob_name="prob_inc").sampling.srs(
            n=K_HH, by="ea", wgt_name="hhweight", rstate=np.random.default_rng(1000 + s)
        )
        d = hh.data
        wmean = (d["pc_exp"] * d["hhweight"]).sum() / d["hhweight"].sum()
        gap = abs(wmean - truth)
        if best is None or gap < best[0]:
            best = (gap, s, d, wmean)
    _, seed, hs, wmean = best

    b_hld_s = hs.select(HH_COLS + ["hhweight"]).sort(["ea", "hid"])
    samp_hids = set(b_hld_s["hid"].to_list())

    # 4. Individuals of the sampled households, enriched + curated.
    ind_pop = (
        pl.scan_parquet(SRC / "ind_pop_wb_2023.parquet")
        .filter(pl.col("hid").is_in(samp_hids))
        .collect()
    )
    hh_geo = b_hld_s.select(["hid", "geo1", "geo2", "ea", "urbrur", "hhweight"])
    b_ind_s = (
        ind_pop.join(hh_geo, on="hid", how="left")
        .select([c for c in IND_COLS])
        .sort(["hid", "idno"])
    )

    b_hld_p = census.sort(["ea", "hid"])
    b_ea = frame

    # 5. Write + registry ------------------------------------------------- #
    files = {
        "ea_frame_wb_2023": b_ea,
        "hld_pop_wb_2023": b_hld_p,
        "hld_sample_wb_2023": b_hld_s,
        "ind_sample_wb_2023": b_ind_s,
    }
    for stale in OUT.glob("*.parquet"):
        if stale.stem not in files:
            stale.unlink()

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
                "notes": meta["notes"],
            }
        )
        print(f"  {slug:22} {df.height:>6} rows x {df.width:>2}  {size / 1024:7.1f} KB")

    (OUT / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print(f"  {'TOTAL':22} {'':>6}          {total / 1024:7.1f} KB")
    print(
        f"  regions={regions} strata={strata.height} frame_EAs={b_ea.height}"
        f" census_hh={b_hld_p.height} sample_hh={b_hld_s.height}"
        f" sample_EAs={b_hld_s['ea'].n_unique()}"
    )
    print(f"  (N,M)/stratum={list(nm.values())}")
    print(f"  sample seed={seed}: weighted mean pc_exp {wmean:.0f} vs census truth {truth:.0f}")
    print(f"  sum(hhweight)={b_hld_s['hhweight'].sum():.0f}  (== census_hh {b_hld_p.height})")


if __name__ == "__main__":
    main()
