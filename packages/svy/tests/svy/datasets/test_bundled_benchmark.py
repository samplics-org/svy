# tests/svy/datasets/test_bundled_benchmark.py
"""
Golden-benchmark + structural tests for the datasets bundled inside the wheel.

The bundled parquet files ship with the package (``svy/datasets/_bundled/``).
These tests pin them to a frozen benchmark (``bundled_benchmark.json``) so any
change to the shipped data is detected:

- **sha256 / size**  — the file is byte-identical to when it was frozen.
- **shape / columns** — schema is stable.
- **numeric_sums**    — every numeric column's total is unchanged (content).
- **design estimate** — the design-based mean of ``pc_exp`` on the household
  sample is unchanged; this doubles as a regression fixture for the estimation
  pipeline.

Because the benchmark is a committed file, regenerating the data forces a
reviewed diff to it (see ``scripts/build_bundled_benchmark.py``).  We do NOT
re-run the generator here — that needs the multi-GB canonical masters, which
are not in the repo.

The structural tests also assert the invariants the generator promises:
sample ⊂ census, reconciled EA sets, and a valid design (19 strata, no
singleton PSUs).
"""

from __future__ import annotations

import hashlib
import json

from importlib.resources import as_file, files
from pathlib import Path

import polars as pl
import pytest

import svy
import svy.datasets as d


BENCHMARK: dict[str, dict] = json.loads(
    (Path(__file__).parent / "bundled_benchmark.json").read_text()
)
SLUGS = sorted(BENCHMARK)
EXPECTED_SLUGS = {
    "ea_frame_wb_2023",
    "hld_sample_wb_2023",
    "ind_sample_wb_2023",
    "hld_pop_wb_2023",
    "ind_pop_wb_2023",
}


def _bundled_bytes(slug: str) -> bytes:
    resource = files("svy.datasets") / "_bundled" / f"{slug}.parquet"
    with as_file(resource) as p:
        return Path(p).read_bytes()


# --------------------------------------------------------------------------- #
# Catalog / packaging
# --------------------------------------------------------------------------- #


def test_bundled_catalog_has_expected_slugs():
    assert {ds.slug for ds in d.catalog(source="bundled")} == EXPECTED_SLUGS


def test_benchmark_covers_every_bundled_dataset():
    # Guards against adding a bundled file without freezing a benchmark for it.
    assert set(SLUGS) == EXPECTED_SLUGS


@pytest.mark.parametrize("slug", SLUGS)
def test_registry_and_files_present(slug):
    # registry.json + each parquet must be readable as package resources.
    assert (files("svy.datasets") / "_bundled" / f"{slug}.parquet").is_file()
    assert (files("svy.datasets") / "_bundled" / "registry.json").is_file()


# --------------------------------------------------------------------------- #
# Golden benchmark
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", SLUGS)
def test_file_sha256_and_size_match_benchmark(slug):
    raw = _bundled_bytes(slug)
    assert hashlib.sha256(raw).hexdigest() == BENCHMARK[slug]["sha256"]
    assert len(raw) == BENCHMARK[slug]["size_bytes"]


@pytest.mark.parametrize("slug", SLUGS)
def test_shape_and_columns_match_benchmark(slug):
    df = d.load(slug, source="bundled")
    bench = BENCHMARK[slug]
    assert df.height == bench["n_rows"]
    assert df.width == bench["n_cols"]
    assert df.columns == bench["columns"]


@pytest.mark.parametrize("slug", SLUGS)
def test_numeric_sums_match_benchmark(slug):
    df = d.load(slug, source="bundled")
    for col, expected in BENCHMARK[slug]["numeric_sums"].items():
        got = df[col].sum()
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-4), col


def test_design_based_estimate_matches_benchmark():
    slug = "hld_sample_wb_2023"
    bench = BENCHMARK[slug]["design_mean_pc_exp"]
    info = d.describe(slug, source="bundled")
    df = d.load(slug, source="bundled")
    est = svy.Sample(data=df, design=svy.Design(**info.design)).estimation.mean("pc_exp")
    row = est.to_dicts()[0]
    assert row["est"] == pytest.approx(bench["est"], rel=1e-6, abs=1e-4)
    assert row["se"] == pytest.approx(bench["se"], rel=1e-6, abs=1e-4)
    assert est.n_strata == bench["n_strata"]
    assert est.n_psus == bench["n_psus"]


# --------------------------------------------------------------------------- #
# Structural invariants promised by the generator
# --------------------------------------------------------------------------- #


def test_sample_is_subset_of_census():
    hh_s = d.load("hld_sample_wb_2023", source="bundled")
    hh_p = d.load("hld_pop_wb_2023", source="bundled")
    assert set(hh_s["hid"]).issubset(set(hh_p["hid"]))

    ind_s = d.load("ind_sample_wb_2023", source="bundled")
    ind_p = d.load("ind_pop_wb_2023", source="bundled")
    assert set(ind_s["hid"]).issubset(set(ind_p["hid"]))


def test_ea_sets_reconcile():
    frame = d.load("ea_frame_wb_2023", source="bundled")
    hh_s = d.load("hld_sample_wb_2023", source="bundled")
    hh_p = d.load("hld_pop_wb_2023", source="bundled")
    eas_frame = set(frame["ea"])
    assert eas_frame == set(hh_s["ea"]) == set(hh_p["ea"])
    # frame's n_hlds_census reconciles with the reduced census counts
    counts = hh_p.group_by("ea").len().sort("ea")
    frame_sorted = frame.sort("ea")
    assert frame_sorted["n_hlds_census"].to_list() == counts["len"].to_list()


def test_design_is_valid_no_singleton_strata():
    hh_s = d.load("hld_sample_wb_2023", source="bundled")
    psu = hh_s.group_by(["geo1", "urbrur"]).agg(pl.col("ea").n_unique().alias("n_psu"))
    assert psu.height == 19
    assert psu["n_psu"].min() >= 2  # no singleton strata → variance is defined
    # the deliberate unequal allocation: 16x2, 2x3, 1x5
    dist = dict(sorted(psu["n_psu"].value_counts().iter_rows()))
    assert dist == {2: 16, 3: 2, 5: 1}


def test_household_sample_has_design_metadata():
    info = d.describe("hld_sample_wb_2023", source="bundled")
    assert info.design == {"stratum": ["geo1", "urbrur"], "psu": "ea", "wgt": "hhweight"}
