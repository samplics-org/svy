# tests/svy/datasets/test_remote_extra.py
"""
``httpx`` is optional (the ``remote`` extra), and ``source="auto"`` reads a
full copy already in the local cache before touching the network.

``sys.modules["httpx"] = None`` makes ``import httpx`` raise ImportError, which
is what a base install without the extra sees.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
import warnings

from pathlib import Path

import httpx
import pytest

import svy.datasets as d

from svy.datasets import _cache
from svy.datasets.types import Dataset
from svy.errors._dataset import DatasetError


SLUG = "hld_sample_wb_2023"
BUNDLED_ROWS = 825
NOT_BUNDLED = "ind_pop_wb_2023"


@pytest.fixture
def no_httpx(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)


def _entry(version: str, **fields) -> Dataset:
    return Dataset(
        slug=SLUG,
        title="cached",
        description="",
        version=version,
        download_url="https://svylab.test/data/x.parquet",
        sha256="",
        size_bytes=0,
        n_rows=0,
        n_cols=0,
        license="",
        citation="",
        source="svylab",
        **fields,
    )


def _seed_cache(
    make_parquet, *, version: str, n_rows: int, mtime: float | None = None, entry: bool = True
):
    data, sha = make_parquet(n_rows=n_rows)
    path = _cache.path_for(SLUG, version)
    path.write_bytes(data)
    if entry:
        _cache.write_entry(path, _entry(version))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path, sha


def _offline(routes) -> None:
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated offline", request=req)

    routes.add("/api/data/examples/registry", boom)


# --------------------------------------------------------------------------- #
# Base install: no httpx
# --------------------------------------------------------------------------- #


def test_import_svy_does_not_import_httpx():
    code = "import sys, svy; sys.exit('httpx' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], env=os.environ.copy())
    assert r.returncode == 0


def test_import_svy_works_without_httpx(tmp_path):
    code = (
        "import sys; sys.modules['httpx'] = None; import svy; "
        f"print(svy.datasets.load({SLUG!r}).n_records)"
    )
    r = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        env={**os.environ, "SVYLAB_CACHE_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(BUNDLED_ROWS)


def test_remote_without_httpx_raises_guiding_error(no_httpx, routes):
    with pytest.raises(DatasetError) as ei:
        d.load(SLUG, source="remote")
    assert ei.value.code == "REMOTE_UNAVAILABLE"
    assert "svy[remote]" in ei.value.hint
    assert 'source="bundled"' in ei.value.hint
    assert routes.hits == []


def test_auto_without_httpx_reads_bundled_silently(no_httpx, routes):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample = d.load(SLUG)
    assert sample.n_records == BUNDLED_ROWS
    assert routes.hits == []


def test_auto_without_httpx_not_bundled_raises(no_httpx):
    with pytest.raises(DatasetError) as ei:
        d.load(NOT_BUNDLED)
    assert ei.value.code == "REMOTE_UNAVAILABLE"


def test_auto_force_download_without_httpx_raises(no_httpx):
    with pytest.raises(DatasetError) as ei:
        d.load(SLUG, force_download=True)
    assert ei.value.code == "REMOTE_UNAVAILABLE"


def test_info_catalog_without_httpx(no_httpx):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        slugs = {ds.slug for ds in d.info()}
    assert SLUG in slugs
    with pytest.raises(DatasetError) as ei:
        d.info(source="remote")
    assert ei.value.code == "REMOTE_UNAVAILABLE"


def test_info_entry_without_httpx(no_httpx):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert d.info(SLUG).slug == SLUG
    with pytest.raises(DatasetError) as ei:
        d.info(NOT_BUNDLED)
    assert ei.value.code == "REMOTE_UNAVAILABLE"


def test_extras_declare_httpx_only_in_remote():
    project = tomllib.loads((Path(__file__).parents[3] / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    assert not any(r.startswith("httpx") for r in project["dependencies"])
    assert any(r.startswith("httpx") for r in extras["remote"])
    assert any(r.startswith("httpx") for r in extras["all"])
    assert not any(r.startswith("great-tables") for group in extras.values() for r in group)


# --------------------------------------------------------------------------- #
# auto: cache first
# --------------------------------------------------------------------------- #


def test_auto_reads_cached_copy_without_network(routes, make_parquet):
    _seed_cache(make_parquet, version="v1.0.0", n_rows=40)
    assert d.load(SLUG).n_records == 40
    assert routes.hits == []


def test_auto_reads_cached_copy_without_httpx(no_httpx, make_parquet):
    _seed_cache(make_parquet, version="v1.0.0", n_rows=40)
    assert d.load(SLUG).n_records == 40


def test_auto_prefers_most_recent_cached_version(routes, make_parquet):
    _seed_cache(make_parquet, version="v1.0.0", n_rows=30, mtime=1_000_000)
    _seed_cache(make_parquet, version="v2.0.0", n_rows=60, mtime=2_000_000)
    assert d.load(SLUG).n_records == 60


def test_auto_skips_cached_copy_failing_its_pin(no_httpx, make_parquet):
    path, _ = _seed_cache(make_parquet, version="v1.0.0", n_rows=40)
    _cache._write_pin(path, "0" * 64)
    assert d.load(SLUG).n_records == BUNDLED_ROWS


def test_auto_cached_copy_matching_pin_is_used(no_httpx, make_parquet):
    path, sha = _seed_cache(make_parquet, version="v1.0.0", n_rows=40)
    _cache._write_pin(path, sha)
    assert d.load(SLUG).n_records == 40


def test_force_download_bypasses_cache(routes, make_parquet, make_backend_entry):
    _seed_cache(make_parquet, version="v1.0.0", n_rows=40)
    data, sha = make_parquet(n_rows=70)
    entry = make_backend_entry(
        slug=SLUG, version="v1.0.0", download_url="https://svylab.test/data/x.parquet", sha256=sha
    )
    routes.add_json("/api/data/examples/registry", [entry])
    routes.add_bytes("/data/x.parquet", data)
    assert d.load(SLUG, force_download=True).n_records == 70
    assert routes.hits


def test_remote_does_not_use_cache_first(routes, make_parquet):
    _seed_cache(make_parquet, version="v1.0.0", n_rows=40)
    _offline(routes)
    with pytest.raises(DatasetError) as ei:
        d.load(SLUG, source="remote")
    assert ei.value.code == "CATALOG_UNREACHABLE"


def test_latest_cached_ignores_unusable_slugs():
    assert _cache.latest_cached("../etc") is None
    assert _cache.latest_cached("no_such_dataset") is None


def test_cached_copy_is_read_not_bundled(routes, make_parquet):
    _seed_cache(make_parquet, version="v1.0.0", n_rows=40)
    assert set(d.load(SLUG).data.columns) == {"id", "value", "region", "age"}


def test_cached_copy_without_its_entry_is_skipped(no_httpx, make_parquet):
    _seed_cache(make_parquet, version="v1.0.0", n_rows=40, entry=False)
    assert d.load(SLUG).n_records == BUNDLED_ROWS


def test_cached_copy_is_labelled_by_its_own_entry(no_httpx, make_parquet):
    # A text-era cached file must never pick up the bundled entry's labels.
    path, _ = _seed_cache(make_parquet, version="v1.0.0", n_rows=40)
    variables = {"region": {"label": "Region"}, "age": {"label": "Age in years"}}
    _cache.write_entry(path, _entry("v1.0.0", variables=variables))
    sample = d.load(SLUG)
    assert dict(sample.meta.var_labels) == {"region": "Region", "age": "Age in years"}
    assert sample.design.wgt is None


def test_clear_cache_removes_recorded_entries(make_parquet):
    path, _ = _seed_cache(make_parquet, version="v1.0.0", n_rows=40)
    assert d.clear_cache(SLUG) == 1
    assert list(path.parent.iterdir()) == []
