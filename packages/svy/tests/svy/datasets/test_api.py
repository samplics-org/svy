# tests/datasets/test_api.py
"""Tests for ``svy.datasets.api`` against the real backend contract.

The backend exposes a single registry endpoint:

    GET /api/data/examples/registry  -> list[BackendDataset]

``catalog()`` hits this endpoint once; ``describe(slug)`` reuses the same
result by filtering the cached registry locally.
"""

from __future__ import annotations

import time

import pytest

from svy.datasets import api
from svy.datasets.types import Dataset
from svy.errors.dataset_errors import DatasetError


REGISTRY_PATH = "/api/data/examples/registry"


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #


class TestCatalog:
    def test_returns_tuple_of_datasets(self, routes, make_backend_entry):
        payload = [
            make_backend_entry(slug="a"),
            make_backend_entry(slug="b"),
            make_backend_entry(slug="c"),
        ]
        routes.add_json(REGISTRY_PATH, payload)
        result = api.catalog()
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(d, Dataset) for d in result)
        assert [d.slug for d in result] == ["a", "b", "c"]

    def test_result_is_cached(self, routes, make_backend_entry):
        routes.add_json(REGISTRY_PATH, [make_backend_entry(slug="a")])
        api.catalog()
        hits_after_first = len(routes.hits)
        api.catalog()
        api.catalog()
        assert len(routes.hits) == hits_after_first

    def test_use_cache_false_forces_refetch(self, routes, make_backend_entry):
        routes.add_json(REGISTRY_PATH, [make_backend_entry(slug="a")])
        api.catalog()
        hits_after_first = len(routes.hits)
        api.catalog(use_cache=False)
        assert len(routes.hits) > hits_after_first

    def test_empty_registry(self, routes):
        routes.add_json(REGISTRY_PATH, [])
        assert api.catalog() == ()

    def test_server_error_raises(self, routes):
        routes.add_status(REGISTRY_PATH, 500)
        with pytest.raises(DatasetError) as exc_info:
            api.catalog()
        assert exc_info.value.code == "CATALOG_BAD_STATUS"

    def test_translates_backend_fields_into_dataset(self, routes, make_backend_entry):
        """``_from_backend`` should map backend shape to typed Dataset."""
        payload = [
            make_backend_entry(
                slug="toy",
                name="My Toy Dataset",
                version="v1.2.3",
                description="A small fixture.",
                download_url="https://svylab.test/data/toy.parquet",
                tags=("health", "demo"),
                design={"stratum": "region", "psu": "ea", "wgt": "wt"},
            )
        ]
        routes.add_json(REGISTRY_PATH, payload)

        (ds,) = api.catalog()
        assert ds.slug == "toy"
        assert ds.title == "My Toy Dataset"
        assert ds.version == "v1.2.3"
        assert ds.download_url == "https://svylab.test/data/toy.parquet"
        assert ds.tags == ("health", "demo")
        assert ds.design == {"stratum": "region", "psu": "ea", "wgt": "wt"}
        # Fields the backend doesn't provide fill in with safe defaults.
        assert ds.sha256 == ""
        assert ds.size_bytes == 0

    def test_sha256_passthrough_when_backend_provides(self, routes, make_backend_entry):
        """If a future backend includes sha256, the client should surface it."""
        payload = [
            make_backend_entry(slug="toy", sha256="abc123def"),
        ]
        routes.add_json(REGISTRY_PATH, payload)
        (ds,) = api.catalog()
        assert ds.sha256 == "abc123def"


# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #


class TestDescribe:
    def test_happy_path(self, routes, make_backend_entry):
        """describe() fetches the registry and filters locally."""
        routes.add_json(
            REGISTRY_PATH,
            [
                make_backend_entry(slug="a"),
                make_backend_entry(slug="toy"),
                make_backend_entry(slug="b"),
            ],
        )
        ds = api.describe("toy")
        assert isinstance(ds, Dataset)
        assert ds.slug == "toy"

    def test_not_found_raises_dataset_error(self, routes, make_backend_entry):
        routes.add_json(REGISTRY_PATH, [make_backend_entry(slug="something_else")])
        with pytest.raises(DatasetError) as exc_info:
            api.describe("missing")
        assert exc_info.value.code == "DATASET_NOT_FOUND"
        assert exc_info.value.got == "missing"

    def test_server_error_raises_catalog_bad_status(self, routes):
        routes.add_status(REGISTRY_PATH, 503)
        with pytest.raises(DatasetError) as exc_info:
            api.describe("toy")
        assert exc_info.value.code == "CATALOG_BAD_STATUS"

    def test_result_is_cached_in_memory(self, routes, make_backend_entry):
        """First describe triggers registry fetch; second is cached."""
        routes.add_json(REGISTRY_PATH, [make_backend_entry(slug="toy")])
        api.describe("toy")
        hits_after_first = len(routes.hits)
        api.describe("toy")
        api.describe("toy")
        assert len(routes.hits) == hits_after_first

    def test_catalog_warms_describe_cache(self, routes, make_backend_entry):
        """After catalog(), describe(slug) should be a cache hit."""
        routes.add_json(
            REGISTRY_PATH,
            [make_backend_entry(slug="a"), make_backend_entry(slug="b")],
        )
        api.catalog()
        hits_after_catalog = len(routes.hits)
        ds = api.describe("a")
        assert ds.slug == "a"
        assert len(routes.hits) == hits_after_catalog  # no extra request


# --------------------------------------------------------------------------- #
# clear_cache
# --------------------------------------------------------------------------- #


class TestClearCache:
    def test_clear_resets_both_caches(self, routes, make_backend_entry):
        routes.add_json(REGISTRY_PATH, [make_backend_entry(slug="a")])

        api.catalog()
        api.describe("a")
        hits_after_load = len(routes.hits)

        api.clear_cache()

        api.describe("a")
        # After clearing, describe() should hit the network again.
        assert len(routes.hits) > hits_after_load


# --------------------------------------------------------------------------- #
# TTL expiry
# --------------------------------------------------------------------------- #


class TestTTL:
    def test_catalog_cache_expires(self, routes, make_backend_entry, monkeypatch):
        routes.add_json(REGISTRY_PATH, [make_backend_entry(slug="a")])

        api.catalog()
        n_hits = len(routes.hits)

        base = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: base + api._LIST_TTL_SECONDS + 1)

        api.catalog()
        assert len(routes.hits) > n_hits

    def test_describe_cache_expires(self, routes, make_backend_entry, monkeypatch):
        routes.add_json(REGISTRY_PATH, [make_backend_entry(slug="a")])

        api.describe("a")
        n_hits = len(routes.hits)

        base = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: base + api._GET_TTL_SECONDS + 1)

        api.describe("a")
        assert len(routes.hits) > n_hits
