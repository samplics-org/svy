"""
Generate the bundled synthetic panel ``panel_syn_2026``.

A three-wave household panel in LONG format (one row per case and wave),
small enough to ship in the wheel and complete enough for the panel tutorial
to run offline: stacking, level per wave, net change, transitions, and the
attrition-weighting chain.

Design
------
* 6 strata x 10 PSUs x 20 cases at wave 1 (1200 cases). ``psu`` is unique
  across strata; ``urban`` is a PSU attribute.
* Base weight ``w1`` = 1 / stratum inclusion probability, constant within a
  case across waves (the sampling design is the wave-1 design).
* Attrition ~15% per wave, monotone, logistic in the base-wave ``age_grp``
  and ``urban`` plus a PSU random effect, so adjusting within
  ``["stratum", "age_grp"]`` recovers most of the bias. Half of the
  attriters at a wave have a row with ``resp == "nr"`` and null outcomes
  (contacted, no interview); the other half have no row at all (lost), which
  is what the missing-in-scope rule of ``adjust`` is for.
* Producer longitudinal weights ``lw_12`` (waves 1-2) and ``lw_123`` (waves
  1-3) built by svy's own chain, ``adjust(cells=["stratum", "age_grp"])``
  scoped to the target wave; zero for cases not observed through that wave.
* ``emp`` (binary, for transitions) and ``inc`` (continuous, within-case
  correlation ~0.9, for change) with a small upward trend across waves.

Everything is deterministic (fixed seed). Usage:

    uv run python scripts/build_panel_dataset.py
"""

from __future__ import annotations

import hashlib
import json

from pathlib import Path

import numpy as np
import polars as pl

import svy

from svy import col


OUT = Path(__file__).resolve().parents[1] / "src" / "svy" / "datasets" / "_bundled"
SLUG = "panel_syn_2026"
VERSION = "1.0.0"
SEED = 20260906

N_STRATA, N_PSU, N_CASES = 6, 10, 20
INCL_PROB = [0.02, 0.03, 0.025, 0.015, 0.04, 0.02]
P_URBAN = [0.9, 0.7, 0.6, 0.3, 0.2, 0.5]


def _wave1(rng: np.random.Generator) -> pl.DataFrame:
    rows = []
    case = 0
    for h in range(N_STRATA):
        for j in range(N_PSU):
            psu = (h + 1) * 100 + j + 1
            urban = int(rng.random() < P_URBAN[h])
            for _ in range(N_CASES):
                case += 1
                rows.append((case, h + 1, psu, urban))
    df = pl.DataFrame(
        rows, schema=["case_id", "stratum", "psu", "urban"], orient="row"
    ).with_columns(
        pl.Series("age_grp", rng.choice([1, 2, 3, 4], size=case, p=[0.25, 0.35, 0.25, 0.15])),
        pl.Series("sex", rng.choice([1, 2], size=case)),
    )
    w1 = np.array([1.0 / INCL_PROB[h - 1] for h in df["stratum"].to_list()])
    return df.with_columns(pl.Series("w1", w1))


def _outcomes(base: pl.DataFrame, rng: np.random.Generator) -> list[pl.DataFrame]:
    n = base.height
    u = rng.normal(size=n)  # persistent case effect
    age = base["age_grp"].to_numpy()
    urban = base["urban"].to_numpy()
    waves = []
    for t in (1, 2, 3):
        e_inc = rng.normal(size=n)
        e_emp = rng.normal(size=n)
        inc = (2000 + 300 * age + 400 * urban + 800 * u + 250 * e_inc) * (1.03 ** (t - 1))
        latent = 0.3 + 1.2 * u + 0.6 * e_emp - 0.4 * (age == 4) + 0.1 * (t - 1)
        waves.append(
            base.with_columns(
                pl.lit(t, dtype=pl.Int64).alias("wave"),
                pl.Series("emp", (latent > 0).astype(np.int64)),
                pl.Series("inc", np.round(inc, 2)),
                pl.lit("rr").alias("resp"),
            )
        )
    return waves


def _attrite(waves: list[pl.DataFrame], rng: np.random.Generator) -> list[pl.DataFrame]:
    base = waves[0]
    psus = base["psu"].unique().sort().to_list()
    b_psu = dict(zip(psus, rng.normal(scale=0.3, size=len(psus))))
    age = base["age_grp"].to_numpy()
    urban = base["urban"].to_numpy()
    eta = -2.0 + 0.6 * (age == 1) + 0.5 * (1 - urban) + np.array([b_psu[p] for p in base["psu"]])
    p_attrite = 1 / (1 + np.exp(-eta))

    present = np.ones(base.height, dtype=bool)
    out = [waves[0]]
    for t in (2, 3):
        drop = present & (rng.random(base.height) < p_attrite)
        present = present & ~drop
        nr_row = drop & (rng.random(base.height) < 0.5)
        keep = present | nr_row
        wave = waves[t - 1].filter(pl.Series(keep))
        nr_mask = pl.Series(nr_row[keep])
        wave = wave.with_columns(
            pl.when(nr_mask).then(pl.lit("nr")).otherwise(pl.lit("rr")).alias("resp"),
            pl.when(nr_mask).then(None).otherwise(pl.col("emp")).alias("emp"),
            pl.when(nr_mask).then(None).otherwise(pl.col("inc")).alias("inc"),
        )
        out.append(wave)
    return out


def _producer_weights(long: pl.DataFrame) -> pl.DataFrame:
    design = svy.Design(case_id="case_id", wave="wave", stratum="stratum", psu="psu", wgt="w1")
    s = svy.Sample(long, design)
    s = s.weighting.adjust(
        "resp",
        cells=["stratum", "age_grp"],
        where=col("wave") == 2,
        wgt_name="lw_12",
        respondents_only=False,
    )
    s = s.weighting.adjust(
        "resp",
        cells=["stratum", "age_grp"],
        where=col("wave") == 3,
        wgt_name="lw_123",
        respondents_only=False,
    )
    return s.data.select("case_id", "wave", "lw_12", "lw_123")


def main() -> None:
    rng = np.random.default_rng(SEED)
    base = _wave1(rng)
    waves = _attrite(_outcomes(base, rng), rng)
    long = pl.concat(waves).sort("case_id", "wave")
    long = long.join(_producer_weights(long), on=["case_id", "wave"], how="left").select(
        "case_id",
        "wave",
        "stratum",
        "psu",
        "urban",
        "age_grp",
        "sex",
        "w1",
        "resp",
        "lw_12",
        "lw_123",
        "emp",
        "inc",
    )

    path = OUT / f"{SLUG}.parquet"
    long.write_parquet(path, compression="zstd")
    raw = path.read_bytes()
    entry = {
        "slug": SLUG,
        "title": "Synthetic three-wave household panel 2026 (bundled)",
        "description": (
            "Long-format panel: 1200 cases at wave 1 in 6 strata x 10 PSUs, ~15% "
            "attrition per wave, producer longitudinal weights lw_12 and lw_123, a "
            "binary outcome for transitions and a continuous one for change."
        ),
        "version": VERSION,
        "filename": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "n_rows": long.height,
        "n_cols": long.width,
        "license": "MIT (synthetic data generated by svy)",
        "citation": "svy synthetic panel, generated by scripts/build_panel_dataset.py",
        "source": "svy (synthetic)",
        "design": {
            "case_id": "case_id",
            "wave": "wave",
            "stratum": "stratum",
            "psu": "psu",
            "wgt": "w1",
        },
        "variables": {},
        "tags": ["synthetic", "panel", "longitudinal", "bundled"],
        "notes": (
            "resp is 'rr' on interviewed rows and 'nr' on contacted-but-not-interviewed "
            "rows (null outcomes); cases lost to follow-up have no row at later waves. "
            "lw_12 / lw_123 are zero for cases not observed through wave 2 / 3. "
            "Attrition is logistic in age_grp and urban with a PSU random effect."
        ),
    }
    reg_path = OUT / "registry.json"
    registry = [e for e in json.loads(reg_path.read_text()) if e["slug"] != SLUG]
    registry.append(entry)
    reg_path.write_text(json.dumps(registry, indent=2) + "\n")
    print(f"wrote {path} ({long.height} rows, {len(raw)} bytes)")


if __name__ == "__main__":
    main()
