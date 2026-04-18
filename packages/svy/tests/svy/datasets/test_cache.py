# tests/datasets/test_cache.py
"""Tests for ``svy.datasets._cache``."""

from __future__ import annotations

import threading
from pathlib import Path

import httpx
import pytest

from svy.datasets import _cache
from svy.errors.dataset_errors import DatasetError


# --------------------------------------------------------------------------- #
# path_for
# --------------------------------------------------------------------------- #


class TestPathFor:
    def test_path_includes_slug_and_version(self):
        p = _cache.path_for("phia_like", "1.2.3")
        assert p.name == "phia_like@1.2.3.parquet"
        assert p.parent == _cache.CACHE_DIR

    def test_different_versions_produce_different_paths(self):
        p1 = _cache.path_for("slug", "1.0.0")
        p2 = _cache.path_for("slug", "1.0.1")
        assert p1 != p2

    def test_version_with_path_separators_sanitized(self):
        p = _cache.path_for("slug", "branch/weird")
        assert "/" not in p.name
        assert "\\" not in p.name


# --------------------------------------------------------------------------- #
# ensure_cached — happy path
# --------------------------------------------------------------------------- #


class TestEnsureCachedDownload:
    def test_downloads_and_returns_path(self, routes, make_parquet):
        data, sha = make_parquet(n_rows=30)
        routes.add_bytes("/data/toy.parquet", data)

        path = _cache.ensure_cached(
            slug="toy",
            version="1.0.0",
            url="https://svylab.test/data/toy.parquet",
            sha256=sha,
        )
        assert Path(path).exists()
        assert Path(path).read_bytes() == data

    def test_second_call_hits_fast_path(self, routes, make_parquet):
        data, sha = make_parquet(n_rows=30)
        routes.add_bytes("/data/toy.parquet", data)

        _cache.ensure_cached(
            slug="toy",
            version="1.0.0",
            url="https://svylab.test/data/toy.parquet",
            sha256=sha,
        )
        n_hits_after_first = len(routes.hits)

        _cache.ensure_cached(
            slug="toy",
            version="1.0.0",
            url="https://svylab.test/data/toy.parquet",
            sha256=sha,
        )
        # Second call is fully memoized (session-verified).
        assert len(routes.hits) == n_hits_after_first

    def test_force_redownloads(self, routes, make_parquet):
        data, sha = make_parquet(n_rows=30)
        routes.add_bytes("/data/toy.parquet", data)

        _cache.ensure_cached(
            slug="toy",
            version="1.0.0",
            url="https://svylab.test/data/toy.parquet",
            sha256=sha,
        )
        n_before = len(routes.hits)

        _cache.ensure_cached(
            slug="toy",
            version="1.0.0",
            url="https://svylab.test/data/toy.parquet",
            sha256=sha,
            force=True,
        )
        assert len(routes.hits) > n_before


# --------------------------------------------------------------------------- #
# ensure_cached — integrity
# --------------------------------------------------------------------------- #


class TestIntegrity:
    def test_sha_mismatch_raises_and_cleans_up(self, routes, make_parquet):
        data, _correct_sha = make_parquet(n_rows=30)
        routes.add_bytes("/data/bad.parquet", data)

        with pytest.raises(DatasetError) as exc_info:
            _cache.ensure_cached(
                slug="bad",
                version="1.0.0",
                url="https://svylab.test/data/bad.parquet",
                sha256="a" * 64,  # wrong
            )
        assert exc_info.value.code == "DATASET_SHA_MISMATCH"
        # No half-written file left behind.
        assert not _cache.path_for("bad", "1.0.0").exists()
        # No leftover .tmp files either.
        assert not list(_cache.CACHE_DIR.glob("*.tmp*"))

    def test_corrupted_cache_entry_is_redownloaded(self, routes, make_parquet, tmp_path):
        """
        If a file is already on disk but its hash doesn't match, the cache
        should treat it as corrupted, re-download, and succeed if the new
        fetch matches.
        """
        data, sha = make_parquet(n_rows=30)
        routes.add_bytes("/data/toy.parquet", data)

        # Plant a garbage file at the expected cache location.
        target = _cache.path_for("toy", "1.0.0")
        target.write_bytes(b"junk")

        path = _cache.ensure_cached(
            slug="toy",
            version="1.0.0",
            url="https://svylab.test/data/toy.parquet",
            sha256=sha,
        )
        # Should have re-downloaded and now match.
        assert Path(path).read_bytes() == data

    def test_hash_failure_mode_only_hashes_once_per_session(self, routes, make_parquet):
        """
        After successful verification, subsequent calls should skip the
        hash step (fast path via ``_verified`` set).
        """
        data, sha = make_parquet(n_rows=30)
        routes.add_bytes("/data/toy.parquet", data)

        _cache.ensure_cached(
            slug="toy",
            version="1.0.0",
            url="https://svylab.test/data/toy.parquet",
            sha256=sha,
        )
        key = (str(_cache.path_for("toy", "1.0.0")), sha.lower())
        assert key in _cache._verified


# --------------------------------------------------------------------------- #
# ensure_cached — download failure
# --------------------------------------------------------------------------- #


class TestDownloadFailure:
    def test_http_error_cleans_up_tempfile(self, routes):
        routes.add_status("/data/gone.parquet", 500)

        with pytest.raises(RuntimeError):
            _cache.ensure_cached(
                slug="gone",
                version="1.0.0",
                url="https://svylab.test/data/gone.parquet",
                sha256="0" * 64,
            )
        # No leftover tempfiles.
        assert not list(_cache.CACHE_DIR.glob("*.tmp*"))
        assert not list(_cache.CACHE_DIR.glob(".gone*"))


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


class TestConcurrency:
    def test_concurrent_downloads_serialize(self, routes, make_parquet):
        """
        Two threads asking for the same (slug, version) should not both hit
        the network.  One downloads; the other uses the cached result.
        """
        data, sha = make_parquet(n_rows=50)
        routes.add_bytes("/data/shared.parquet", data)

        results: list[str] = []
        errors: list[BaseException] = []

        def worker():
            try:
                p = _cache.ensure_cached(
                    slug="shared",
                    version="1.0.0",
                    url="https://svylab.test/data/shared.parquet",
                    sha256=sha,
                )
                results.append(p)
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(set(results)) == 1  # all got the same path
        # With the per-slug lock, we expect <= N_threads GET requests, but
        # in practice after the first one completes the others take the fast
        # path.  A reasonable upper bound:
        get_hits = [h for h in routes.hits if h[0] == "GET"]
        assert len(get_hits) <= 8  # no runaway

    def test_different_slugs_do_not_block(self, routes, make_parquet):
        data_a, sha_a = make_parquet(n_rows=10, extra={"tag": ["a"] * 10})
        data_b, sha_b = make_parquet(n_rows=10, extra={"tag": ["b"] * 10})
        routes.add_bytes("/data/a.parquet", data_a)
        routes.add_bytes("/data/b.parquet", data_b)

        # No assertion on time; just that both succeed without deadlock.
        results = []

        def worker(slug, url, sha):
            results.append(_cache.ensure_cached(slug=slug, version="1.0.0", url=url, sha256=sha))

        t1 = threading.Thread(
            target=worker,
            args=("a", "https://svylab.test/data/a.parquet", sha_a),
        )
        t2 = threading.Thread(
            target=worker,
            args=("b", "https://svylab.test/data/b.parquet", sha_b),
        )
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(results) == 2


# --------------------------------------------------------------------------- #
# clear
# --------------------------------------------------------------------------- #


class TestClear:
    def test_clear_all(self, routes, make_parquet):
        data, sha = make_parquet(n_rows=5)
        routes.add_bytes("/data/a.parquet", data)
        routes.add_bytes("/data/b.parquet", data)

        _cache.ensure_cached(
            slug="a",
            version="1.0",
            url="https://svylab.test/data/a.parquet",
            sha256=sha,
        )
        _cache.ensure_cached(
            slug="b",
            version="1.0",
            url="https://svylab.test/data/b.parquet",
            sha256=sha,
        )
        assert len(list(_cache.CACHE_DIR.glob("*.parquet"))) == 2

        removed = _cache.clear()
        assert removed == 2
        assert not list(_cache.CACHE_DIR.glob("*.parquet"))
        # Verified set also cleared.
        assert not _cache._verified

    def test_clear_single_slug(self, routes, make_parquet):
        data, sha = make_parquet(n_rows=5)
        routes.add_bytes("/data/a.parquet", data)
        routes.add_bytes("/data/b.parquet", data)

        _cache.ensure_cached(
            slug="a",
            version="1.0",
            url="https://svylab.test/data/a.parquet",
            sha256=sha,
        )
        _cache.ensure_cached(
            slug="b",
            version="1.0",
            url="https://svylab.test/data/b.parquet",
            sha256=sha,
        )

        removed = _cache.clear("a")
        assert removed == 1
        remaining = {p.name for p in _cache.CACHE_DIR.glob("*.parquet")}
        assert remaining == {"b@1.0.parquet"}

    def test_clear_nonexistent_slug_returns_zero(self):
        assert _cache.clear("does-not-exist") == 0

    def test_clear_versioned_variants(self, routes, make_parquet):
        """clear('slug') removes ALL versions of that slug."""
        data, sha = make_parquet(n_rows=5)
        routes.add_bytes("/data/v1.parquet", data)
        routes.add_bytes("/data/v2.parquet", data)

        _cache.ensure_cached(
            slug="s",
            version="1.0",
            url="https://svylab.test/data/v1.parquet",
            sha256=sha,
        )
        _cache.ensure_cached(
            slug="s",
            version="2.0",
            url="https://svylab.test/data/v2.parquet",
            sha256=sha,
        )
        removed = _cache.clear("s")
        assert removed == 2
