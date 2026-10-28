# tests/datasets/test_misc.py
"""
Additional tests to cover edge cases and lifecycle details not in the
main per-module test files.
"""

from __future__ import annotations

import httpx
import pytest

from svy.datasets import _cache, api
from svy.errors._dataset import DatasetError


# --------------------------------------------------------------------------- #
# api lifecycle
# --------------------------------------------------------------------------- #


class TestApiLifecycle:
    def test_get_client_is_lazy_and_reused(self, monkeypatch):
        """
        When no client has been created yet, the first call creates one;
        subsequent calls return the same instance.
        """
        real_client_cls = httpx.Client  # may already be patched globally; that's ok
        monkeypatch.setattr(api, "_client", None)
        # Bypass HTTP/2 by using a transport-only Client.
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *a, **kw: real_client_cls(
                transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"[]"))
            ),
        )
        c1 = api._get_client()
        c2 = api._get_client()
        assert c1 is c2

    def test_close_tears_down_client(self, monkeypatch):
        real_client_cls = httpx.Client
        monkeypatch.setattr(api, "_client", None)
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *a, **kw: real_client_cls(
                transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"[]"))
            ),
        )
        api._get_client()
        assert api._client is not None
        api.close()
        assert api._client is None

    def test_close_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(api, "_client", None)
        api.close()  # no client to close
        api.close()  # still no client


# --------------------------------------------------------------------------- #
# api: network-level failures (HTTPError that is not HTTPStatusError)
# --------------------------------------------------------------------------- #


class TestNetworkFailure:
    def test_catalog_connection_error_raises_unreachable(self, monkeypatch, make_dataset_dict):
        """
        An httpx.ConnectError (no server, DNS failure, etc.) should surface
        as CATALOG_UNREACHABLE, not CATALOG_BAD_STATUS.
        """

        def broken_get(*args, **kwargs):
            raise httpx.ConnectError("simulated connection refused")

        monkeypatch.setattr(api._get_client(), "get", broken_get)

        with pytest.raises(DatasetError) as exc_info:
            api.catalog()
        assert exc_info.value.code == "CATALOG_UNREACHABLE"
        assert "simulated" in exc_info.value.detail


# --------------------------------------------------------------------------- #
# _cache: re-download-after-corruption logs a warning
# --------------------------------------------------------------------------- #


class TestCorruptedCacheWarns:
    def test_warning_logged_on_corrupted_cache(self, routes, make_parquet, caplog):
        data, sha = make_parquet(n_rows=10)
        routes.add_bytes("/data/toy.parquet", data)

        target = _cache.path_for("toy", "1.0.0")
        target.write_bytes(b"corrupt")

        with caplog.at_level("WARNING"):
            _cache.ensure_cached(
                slug="toy",
                version="1.0.0",
                url="https://svylab.test/data/toy.parquet",
                sha256=sha,
            )
        assert any("failed hash check" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# _cache.clear: handles OSError gracefully
# --------------------------------------------------------------------------- #


class TestClearResilience:
    def test_clear_continues_past_unremovable_file(
        self, routes, make_parquet, monkeypatch, caplog
    ):
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

        # Make the first unlink raise; subsequent unlinks should still run.
        from pathlib import Path

        real_unlink = Path.unlink
        call_count = {"n": 0}

        def flaky_unlink(self, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated permission failure")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", flaky_unlink)

        with caplog.at_level("WARNING"):
            removed = _cache.clear()
        # One succeeded, one failed — total count reflects successes.
        assert removed == 1
        assert any("Could not remove" in rec.message for rec in caplog.records)
