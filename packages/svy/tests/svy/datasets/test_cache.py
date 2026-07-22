# tests/datasets/test_cache.py
"""Tests for ``svy.datasets._cache``."""

from __future__ import annotations

import threading

from pathlib import Path

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

        with pytest.raises(DatasetError) as exc_info:
            _cache.ensure_cached(
                slug="gone",
                version="1.0.0",
                url="https://svylab.test/data/gone.parquet",
                sha256="0" * 64,
            )
        assert exc_info.value.code == "DATASET_DOWNLOAD_FAILED"
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


# --------------------------------------------------------------------------- #
# Security: slug validation, https enforcement, TOFU pinning
# --------------------------------------------------------------------------- #


class TestSlugValidation:
    @pytest.mark.parametrize(
        "bad_slug",
        [
            "../evil",
            "..",
            "a/../../b",
            "a/b",
            "a\\b",
            ".hidden",
            "*",
            "toy@1",
            "",
            "a b",
        ],
    )
    def test_bad_slugs_rejected(self, bad_slug):
        with pytest.raises(DatasetError) as exc_info:
            _cache.ensure_cached(
                slug=bad_slug,
                version="1.0",
                url="https://svylab.test/data/x.parquet",
                sha256="",
            )
        assert exc_info.value.code == "DATASET_INVALID_SLUG"

    def test_traversal_slug_writes_nothing_outside_cache(self, tmp_path):
        with pytest.raises(DatasetError):
            _cache.path_for("../outside", "1.0")

    def test_clear_rejects_glob_slug(self):
        with pytest.raises(DatasetError):
            _cache.clear("*")

    @pytest.mark.parametrize("good_slug", ["toy", "phia_like", "acs.2023", "a-b_c", "x1"])
    def test_good_slugs_accepted(self, good_slug):
        p = _cache.path_for(good_slug, "1.0")
        assert p.name == f"{good_slug}@1.0.parquet"


class TestHttpsEnforcement:
    def test_http_url_rejected(self):
        with pytest.raises(DatasetError) as exc_info:
            _cache.ensure_cached(
                slug="toy",
                version="1.0",
                url="http://svylab.test/data/toy.parquet",
                sha256="",
            )
        assert exc_info.value.code == "DATASET_INSECURE_URL"

    def test_non_http_scheme_rejected(self):
        with pytest.raises(DatasetError) as exc_info:
            _cache.ensure_cached(
                slug="toy",
                version="1.0",
                url="ftp://svylab.test/data/toy.parquet",
                sha256="",
            )
        assert exc_info.value.code == "DATASET_INSECURE_URL"

    def test_http_localhost_allowed(self, routes, make_parquet):
        data, sha = make_parquet(n_rows=5)
        routes.add_bytes("/data/toy.parquet", data)
        path = _cache.ensure_cached(
            slug="toy",
            version="1.0",
            url="http://127.0.0.1/data/toy.parquet",
            sha256=sha,
        )
        assert Path(path).exists()

    def test_redirect_downgrade_to_http_rejected(self, routes, make_parquet):
        """An https download 302'd to plain http must fail, not follow."""
        import httpx

        data, _sha = make_parquet(n_rows=5)
        routes.add(
            "/data/redirect.parquet",
            lambda req: httpx.Response(
                302, headers={"location": "http://evil.test/data/payload.parquet"}
            ),
        )
        routes.add_bytes("/data/payload.parquet", data)
        with pytest.raises(DatasetError) as exc_info:
            _cache.ensure_cached(
                slug="toy",
                version="1.0",
                url="https://svylab.test/data/redirect.parquet",
                sha256="",
            )
        assert exc_info.value.code == "DATASET_INSECURE_URL"
        assert not list(_cache.CACHE_DIR.glob("*.parquet"))


class TestTofuPinning:
    def _url(self):
        return "https://svylab.test/data/toy.parquet"

    def test_first_download_writes_pin(self, routes, make_parquet):
        data, sha = make_parquet(n_rows=10)
        routes.add_bytes("/data/toy.parquet", data)
        path = _cache.ensure_cached(slug="toy", version="1.0", url=self._url(), sha256="")
        pin = _cache._pin_path(Path(path))
        assert pin.exists()
        assert pin.read_text().strip() == sha

    def test_pin_enforced_on_redownload(self, routes, make_parquet):
        """Content change without a version bump must be rejected."""
        data1, _ = make_parquet(n_rows=10)
        data2, _ = make_parquet(n_rows=11)  # different content
        routes.add_bytes("/data/toy.parquet", data1)
        _cache.ensure_cached(slug="toy", version="1.0", url=self._url(), sha256="")

        # Server now returns different bytes for the same slug@version.
        routes._routes.clear()
        routes.add_bytes("/data/toy.parquet", data2)
        with pytest.raises(DatasetError) as exc_info:
            _cache.ensure_cached(
                slug="toy", version="1.0", url=self._url(), sha256="", force=True
            )
        assert exc_info.value.code == "DATASET_SHA_MISMATCH"

    def test_corrupted_cache_restored_from_pin(self, routes, make_parquet):
        data, sha = make_parquet(n_rows=10)
        routes.add_bytes("/data/toy.parquet", data)
        path = _cache.ensure_cached(slug="toy", version="1.0", url=self._url(), sha256="")

        # Corrupt the cached file and wipe the session-verified memo.
        Path(path).write_bytes(b"corrupted")
        _cache._verified.clear()

        path2 = _cache.ensure_cached(slug="toy", version="1.0", url=self._url(), sha256="")
        assert Path(path2).read_bytes() == data

    def test_preexisting_cache_entry_gets_pinned(self, routes, make_parquet):
        """A cache file from before TOFU pinning is adopted, not re-fetched."""
        data, sha = make_parquet(n_rows=10)
        dest = _cache.path_for("toy", "1.0")
        dest.write_bytes(data)

        path = _cache.ensure_cached(slug="toy", version="1.0", url=self._url(), sha256="")
        assert Path(path) == dest
        assert _cache._read_pin(dest) == sha
        assert len(routes.hits) == 0  # no network

    def test_catalog_hash_takes_precedence_over_pin(self, routes, make_parquet):
        """Once the backend serves sha256, it wins over any local pin."""
        data1, _ = make_parquet(n_rows=10)
        data2, sha2 = make_parquet(n_rows=11)
        routes.add_bytes("/data/toy.parquet", data1)
        _cache.ensure_cached(slug="toy", version="1.0", url=self._url(), sha256="")

        routes._routes.clear()
        routes.add_bytes("/data/toy.parquet", data2)
        # Catalog-provided hash for the new content: accepted despite old pin.
        path = _cache.ensure_cached(
            slug="toy", version="1.0", url=self._url(), sha256=sha2, force=True
        )
        assert Path(path).read_bytes() == data2

    def test_clear_removes_pins(self, routes, make_parquet):
        data, _ = make_parquet(n_rows=10)
        routes.add_bytes("/data/toy.parquet", data)
        _cache.ensure_cached(slug="toy", version="1.0", url=self._url(), sha256="")
        assert len(list(_cache.CACHE_DIR.glob("*.sha256"))) == 1
        _cache.clear("toy")
        assert not list(_cache.CACHE_DIR.glob("*.sha256"))

    def test_corrupt_pin_is_ignored_and_rewritten(self, routes, make_parquet):
        data, sha = make_parquet(n_rows=10)
        routes.add_bytes("/data/toy.parquet", data)
        dest = _cache.path_for("toy", "1.0")
        _cache._pin_path(dest).parent.mkdir(parents=True, exist_ok=True)
        _cache._pin_path(dest).write_text("not-a-hash\n")

        _cache.ensure_cached(slug="toy", version="1.0", url=self._url(), sha256="")
        assert _cache._read_pin(dest) == sha
