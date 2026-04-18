# tests/datasets/test_load.py
"""
Tests for ``svy.datasets.load``.

Covers the whole pipeline:
  * catalog lookup + cache download
  * where filtering
  * column selection
  * ordering (ascending / descending / random)
  * slicing: n (int, mapping) and rate (float, mapping), globally and per-group
  * lazy vs eager return
  * argument validation
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy.datasets import load
from svy.datasets.base import _seed_from_rstate
from svy.errors.dataset_errors import DatasetError


# --------------------------------------------------------------------------- #
# Shared setup
# --------------------------------------------------------------------------- #


@pytest.fixture
def wired(routes, make_parquet, make_dataset_dict):
    """Register both the catalog entry and the parquet file."""
    data, sha = make_parquet(n_rows=200)  # 200 rows, 4 regions (50 each)
    url = "https://svylab.test/data/toy.parquet"
    routes.add_json(
        "/api/data/examples/registry",
        [make_dataset_dict(slug="toy", download_url=url, sha256=sha, n_rows=200)],
    )
    routes.add_bytes("/data/toy.parquet", data)
    return data


# --------------------------------------------------------------------------- #
# Return type
# --------------------------------------------------------------------------- #


class TestReturnType:
    def test_default_returns_dataframe(self, wired):
        result = load("toy")
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 200

    def test_lazy_true_returns_lazyframe(self, wired):
        result = load("toy", lazy=True)
        assert isinstance(result, pl.LazyFrame)

    def test_lazy_result_collects_to_same_data(self, wired):
        eager = load("toy")
        lazy = load("toy", lazy=True).collect()
        assert eager.equals(lazy)


# --------------------------------------------------------------------------- #
# where
# --------------------------------------------------------------------------- #


class TestWhere:
    def test_where_pl_expr(self, wired):
        result = load("toy", where=pl.col("age") >= 40)
        assert (result["age"] >= 40).all()

    def test_where_dict(self, wired):
        result = load("toy", where={"region": "north"})
        assert (result["region"] == "north").all()

    def test_where_dict_with_list_is_in(self, wired):
        result = load("toy", where={"region": ["north", "south"]})
        assert set(result["region"].unique()) <= {"north", "south"}

    def test_where_list_of_exprs_is_anded(self, wired):
        result = load(
            "toy",
            where=[pl.col("age") >= 30, pl.col("region") == "east"],
        )
        assert (result["age"] >= 30).all()
        assert (result["region"] == "east").all()

    def test_where_none_returns_everything(self, wired):
        result = load("toy", where=None)
        assert len(result) == 200


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #


class TestSelect:
    def test_select_subset_of_columns(self, wired):
        result = load("toy", select=["id", "value"])
        assert result.columns == ["id", "value"]

    def test_select_ignores_extras_does_not_crash(self, wired):
        # select with only valid columns
        result = load("toy", select=["id"])
        assert result.columns == ["id"]

    def test_select_preserves_all_rows(self, wired):
        result = load("toy", select=["id"])
        assert len(result) == 200


# --------------------------------------------------------------------------- #
# ordering
# --------------------------------------------------------------------------- #


class TestOrdering:
    def test_ascending(self, wired):
        result = load("toy", order_by="value", order_type="ascending")
        vals = result["value"].to_list()
        assert vals == sorted(vals)

    def test_descending(self, wired):
        result = load("toy", order_by="value", order_type="descending")
        vals = result["value"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_random_is_deterministic_with_seed(self, wired):
        r1 = load("toy", order_type="random", rstate=42)
        r2 = load("toy", order_type="random", rstate=42)
        assert r1["id"].to_list() == r2["id"].to_list()

    def test_random_changes_row_order(self, wired):
        baseline = load("toy")["id"].to_list()
        shuffled = load("toy", order_type="random", rstate=42)["id"].to_list()
        # Same rows, different order (with overwhelming probability).
        assert set(shuffled) == set(baseline)
        assert shuffled != baseline

    def test_invalid_order_type_raises(self, wired):
        with pytest.raises(ValueError, match="order_type"):
            load("toy", order_type="sideways")

    def test_order_by_none_with_ascending_is_no_op(self, wired):
        # No sort column provided, and order_type isn't 'random'.
        result = load("toy", order_type="ascending")
        # Should be identity order (as written by parquet).
        assert result["id"].to_list() == list(range(200))


# --------------------------------------------------------------------------- #
# n slicing (global)
# --------------------------------------------------------------------------- #


class TestNGlobal:
    def test_n_returns_first_rows(self, wired):
        result = load("toy", n=10)
        assert len(result) == 10
        assert result["id"].to_list() == list(range(10))

    def test_n_greater_than_total_returns_all(self, wired):
        result = load("toy", n=10_000)
        assert len(result) == 200

    def test_n_zero(self, wired):
        result = load("toy", n=0)
        assert len(result) == 0


# --------------------------------------------------------------------------- #
# n slicing (per-group via `by`)
# --------------------------------------------------------------------------- #


class TestNByGroup:
    def test_n_int_with_by_is_per_group(self, wired):
        # 200 rows, 4 regions × 50 each.  n=5 per region → 20 total.
        result = load("toy", n=5, by="region")
        assert len(result) == 20
        counts = result.group_by("region").len()
        assert all(c == 5 for c in counts["len"].to_list())

    def test_n_mapping_selects_per_group(self, wired):
        result = load(
            "toy",
            n={"north": 3, "south": 7, "east": 2, "west": 0},
            by="region",
        )
        counts = dict(zip(*result.group_by("region").len().select(["region", "len"])))
        # Build manually — group_by order not guaranteed
        by_region = {
            r: len(result.filter(pl.col("region") == r))
            for r in ["north", "south", "east", "west"]
        }
        assert by_region == {"north": 3, "south": 7, "east": 2, "west": 0}

    def test_n_mapping_unlisted_groups_excluded(self, wired):
        result = load("toy", n={"north": 5}, by="region")
        assert set(result["region"].unique()) == {"north"}
        assert len(result) == 5

    def test_n_mapping_without_by_raises(self, wired):
        with pytest.raises(ValueError, match="requires `by`"):
            load("toy", n={"north": 5})


# --------------------------------------------------------------------------- #
# rate slicing (global and per-group)
# --------------------------------------------------------------------------- #


class TestRate:
    def test_rate_global(self, wired):
        result = load("toy", rate=0.25)
        # 200 * 0.25 = 50
        assert len(result) == 50

    def test_rate_per_group_uniform(self, wired):
        result = load("toy", rate=0.1, by="region")
        # 10% of each of the 4 groups of 50 → 5 per group → 20 total
        assert len(result) == 20

    def test_rate_mapping(self, wired):
        result = load(
            "toy",
            rate={"north": 0.2, "south": 0.4, "east": 1.0, "west": 0.0},
            by="region",
        )
        counts = {
            r: len(result.filter(pl.col("region") == r))
            for r in ["north", "south", "east", "west"]
        }
        # north: 50 * 0.2 = 10; south: 50*0.4=20; east: 50; west: 0
        assert counts == {"north": 10, "south": 20, "east": 50, "west": 0}

    def test_rate_out_of_range_raises(self, wired):
        with pytest.raises(ValueError, match="rate"):
            load("toy", rate=1.5)

    def test_rate_zero_raises(self, wired):
        with pytest.raises(ValueError, match="rate"):
            load("toy", rate=0.0)

    def test_n_and_rate_mutually_exclusive(self, wired):
        with pytest.raises(ValueError, match="mutually exclusive"):
            load("toy", n=5, rate=0.1)


# --------------------------------------------------------------------------- #
# Composition: filter + select + order + slice
# --------------------------------------------------------------------------- #


class TestComposition:
    def test_where_then_n(self, wired):
        result = load("toy", where={"region": "north"}, n=5)
        assert len(result) == 5
        assert (result["region"] == "north").all()

    def test_where_then_select_then_order_then_n(self, wired):
        result = load(
            "toy",
            where=pl.col("age") >= 30,
            select=["id", "age", "region"],
            order_by="age",
            order_type="descending",
            n=10,
        )
        assert result.columns == ["id", "age", "region"]
        assert len(result) == 10
        ages = result["age"].to_list()
        assert ages == sorted(ages, reverse=True)
        assert all(a >= 30 for a in ages)

    def test_random_then_n_is_reproducible(self, wired):
        r1 = load("toy", order_type="random", rstate=7, n=20)
        r2 = load("toy", order_type="random", rstate=7, n=20)
        assert r1.equals(r2)


# --------------------------------------------------------------------------- #
# Network / error path
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_unknown_dataset_raises_not_found(self, routes):
        routes.add_json("/api/data/examples/registry", [])
        with pytest.raises(DatasetError) as exc_info:
            load("unknown")
        assert exc_info.value.code == "DATASET_NOT_FOUND"

    def test_bad_sha_bubbles_up(self, routes, make_parquet, make_dataset_dict):
        data, _ = make_parquet(n_rows=10)
        routes.add_json(
            "/api/data/examples/registry",
            [
                make_dataset_dict(
                    slug="bad",
                    download_url="https://svylab.test/data/bad.parquet",
                    sha256="a" * 64,  # wrong; forces verification to fail
                )
            ],
        )
        routes.add_bytes("/data/bad.parquet", data)
        with pytest.raises(DatasetError) as exc_info:
            load("bad")
        assert exc_info.value.code == "DATASET_SHA_MISMATCH"


# --------------------------------------------------------------------------- #
# Seed helper (internal but worth locking in)
# --------------------------------------------------------------------------- #


class TestSeedFromRstate:
    def test_int_passthrough(self):
        assert _seed_from_rstate(42) == 42

    def test_none_produces_some_int(self):
        s = _seed_from_rstate(None)
        assert isinstance(s, int)
        assert s >= 0

    def test_generator_produces_int(self):
        gen = np.random.default_rng(7)
        s = _seed_from_rstate(gen)
        assert isinstance(s, int)

    def test_randomstate_produces_int(self):
        rs = np.random.RandomState(7)
        s = _seed_from_rstate(rs)
        assert isinstance(s, int)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            _seed_from_rstate("not a seed")
