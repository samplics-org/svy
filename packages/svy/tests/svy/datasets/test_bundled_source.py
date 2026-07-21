# tests/svy/datasets/test_bundled_source.py
"""
Tests for the ``source`` policy on ``svy.datasets.load`` / ``catalog`` /
``describe`` and the offline fallback.

The autouse ``isolate_module_state`` fixture points the client at a mock
transport with no routes, so every catalog request 404s — i.e. this simulates
"catalog unreachable" without real network.
"""

from __future__ import annotations

import httpx
import pytest

import svy.datasets as d

from svy.errors.dataset_errors import DatasetError


SLUG = "hld_sample_wb_2023"
BUNDLED_ROWS = 1075


def _simulate_offline(routes) -> None:
    """Make the catalog endpoint raise a connection error (real 'internet down')."""

    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated offline", request=req)

    routes.add("/api/data/examples/registry", boom)


def test_source_bundled_reads_package_without_network(routes):
    df = d.load(SLUG, source="bundled")
    assert df.height == BUNDLED_ROWS
    assert routes.hits == []  # never touched the network


def test_source_remote_raises_when_catalog_unreachable(routes):
    _simulate_offline(routes)
    with pytest.raises(DatasetError) as ei:
        d.load(SLUG, source="remote")
    assert ei.value.code in {"CATALOG_UNREACHABLE", "CATALOG_BAD_STATUS"}


def test_source_auto_falls_back_to_bundled_with_warning(routes):
    _simulate_offline(routes)
    with pytest.warns(UserWarning, match="bundled subset"):
        df = d.load(SLUG, source="auto")
    assert df.height == BUNDLED_ROWS


def test_offline_env_makes_auto_use_bundled_without_network(monkeypatch, routes):
    monkeypatch.setenv("SVYLAB_OFFLINE", "1")
    df = d.load(SLUG)  # default source="auto"
    assert df.height == BUNDLED_ROWS
    assert routes.hits == []


def test_source_bundled_unknown_slug_raises_not_found():
    with pytest.raises(DatasetError) as ei:
        d.load("does_not_exist", source="bundled")
    assert ei.value.code == "DATASET_NOT_FOUND"


def test_catalog_bundled_lists_only_bundled(routes):
    slugs = {ds.slug for ds in d.catalog(source="bundled")}
    assert SLUG in slugs
    assert routes.hits == []


def test_describe_auto_falls_back_to_bundled_metadata(routes):
    _simulate_offline(routes)
    with pytest.warns(UserWarning):
        info = d.describe(SLUG, source="auto")
    assert info.slug == SLUG
    assert info.design == {"stratum": ["geo1", "urbrur"], "psu": "ea", "wgt": "hhweight"}


def test_load_bundled_applies_pipeline_args():
    # bundled data still supports the normal load() pipeline (select + n)
    df = d.load(SLUG, source="bundled", select=["ea", "geo1", "pc_exp"], n=10)
    assert df.columns == ["ea", "geo1", "pc_exp"]
    assert df.height == 10
