# tests/datasets/conftest.py
"""
Shared fixtures for the datasets test suite.

Strategy
--------
- No real network: all HTTP goes through ``httpx.MockTransport``.
- No pollution of the user's ``~/.svy`` cache: each test gets a tmp
  cache directory via ``SVYLAB_CACHE_DIR`` env override + module patch.
- Small synthetic parquet files built on the fly with known shape and
  content so tests can assert exact values.
- Module-level state (shared httpx client, TTL caches, per-slug locks,
  session-verified set) is reset between tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import httpx
import msgspec
import polars as pl
import pytest


# --------------------------------------------------------------------------- #
# Parquet factory
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_parquet(tmp_path: Path) -> Callable[..., tuple[bytes, str]]:
    """
    Build a parquet file in memory and return ``(bytes, sha256_hex)``.

    Usage::

        data, sha = make_parquet(n_rows=100, extra={"region": [...]})
    """

    def _make(
        *,
        n_rows: int = 50,
        extra: dict[str, list[Any]] | None = None,
    ) -> tuple[bytes, str]:
        cols: dict[str, list[Any]] = {
            "id": list(range(n_rows)),
            "value": [float(i) * 1.5 for i in range(n_rows)],
            "region": [["north", "south", "east", "west"][i % 4] for i in range(n_rows)],
            "age": [18 + (i % 60) for i in range(n_rows)],
        }
        if extra:
            cols.update(extra)
        df = pl.DataFrame(cols)
        path = tmp_path / f"tmp_{n_rows}.parquet"
        df.write_parquet(path)
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        return data, sha

    return _make


# --------------------------------------------------------------------------- #
# Dataset metadata factory
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_backend_entry() -> Callable[..., dict[str, Any]]:
    """
    Build a backend-shape registry entry (as returned by
    ``GET /api/data/examples/registry``).

    Matches the real Litestar controller's ``Dataset`` TypedDict.
    """

    def _make(
        *,
        slug: str = "toy",
        name: str | None = None,
        version: str = "v1.0.0",
        description: str | None = None,
        download_url: str = "https://svylab.test/data/toy.parquet",
        tags: tuple[str, ...] = ("examples",),
        subpath: str = "misc",
        design: dict | None = None,
        sha256: str = "",  # backend may or may not provide
    ) -> dict[str, Any]:
        entry = {
            "name": name or f"Toy {slug}",
            "slug": slug,
            "version": version,
            "description": description or f"Fixture dataset {slug}.",
            "dataset": f"{slug}.parquet",
            "format": "parquet",
            "source": {
                "base_url": "",  # empty → use path as-is if absolute
                "path": download_url,
            },
            "tags": list(tags),
            "subpath": subpath,
        }
        # Optional richer fields (backend may add these later).
        if design is not None:
            entry["design"] = design
        if sha256:
            entry["sha256"] = sha256
        return entry

    return _make


@pytest.fixture
def make_dataset_dict(make_backend_entry):
    """Backward-compatible alias for make_backend_entry.

    Accepts a few legacy kwargs (``size_bytes``, ``n_rows``, ``n_cols``,
    ``variables``) and silently drops them so older tests still work without
    rewriting each call site.  ``sha256`` is passed through.
    """

    def _adapter(**kwargs):
        # Drop fields the backend doesn't speak.
        for legacy in ("size_bytes", "n_rows", "n_cols", "variables"):
            kwargs.pop(legacy, None)
        return make_backend_entry(**kwargs)

    return _adapter


# --------------------------------------------------------------------------- #
# HTTP transport mock
# --------------------------------------------------------------------------- #


class RouteRecorder:
    """Tiny route matcher that also records hits for assertion."""

    def __init__(self) -> None:
        self._routes: list[tuple[str, Callable[[httpx.Request], httpx.Response]]] = []
        self.hits: list[tuple[str, str]] = []  # (method, path)

    def add(self, path: str, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._routes.append((path, handler))

    def add_json(self, path: str, payload: Any, status: int = 200) -> None:
        body = msgspec.json.encode(payload)
        self.add(path, lambda req: httpx.Response(status, content=body))

    def add_bytes(self, path: str, data: bytes, status: int = 200) -> None:
        self.add(path, lambda req: httpx.Response(status, content=data))

    def add_status(self, path: str, status: int) -> None:
        self.add(path, lambda req: httpx.Response(status))

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.hits.append((request.method, request.url.path))
            for path, fn in self._routes:
                if request.url.path == path or request.url.path.endswith(path):
                    return fn(request)
            return httpx.Response(404, content=b'{"error": "no route"}')

        return httpx.MockTransport(handler)


@pytest.fixture
def routes() -> RouteRecorder:
    """A fresh route recorder per test."""
    return RouteRecorder()


# --------------------------------------------------------------------------- #
# Module state isolation
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def isolate_module_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, routes):
    """
    Before each test:
      * point CACHE_DIR at a tmp directory,
      * clear TTL caches and the session-verified set,
      * replace the shared httpx.Client with one using our mock transport.

    After each test:
      * tear the client down to avoid leaking across tests.
    """
    from svy.datasets import _cache, api

    # 1. Cache directory
    cache_dir = tmp_path / ".svy" / "datasets"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(_cache, "CACHE_DIR", cache_dir)

    # 2. Wipe api module-level caches + locks
    monkeypatch.setattr(api, "_list_cache", None)
    api._get_cache.clear()

    # 3. Wipe _cache module-level state
    _cache._download_locks.clear()
    _cache._verified.clear()

    # 4. Inject a mock httpx.Client that routes through our transport.
    #    api._client is lazy; set it explicitly so _get_client() returns ours.
    mock_client = httpx.Client(
        transport=routes.transport(),
        base_url="https://svylab.test",
        follow_redirects=True,
    )
    monkeypatch.setattr(api, "_client", mock_client)
    monkeypatch.setattr(api, "API_URL", "https://svylab.test")

    # 5. Also patch _cache's httpx.Client usage for downloads.
    #    ``monkeypatch.setattr(_cache.httpx, "Client", ...)`` would mutate the
    #    shared httpx module and bleed into api.py's client construction too.
    #    Instead, rebind _cache.httpx to a shim namespace with only Client
    #    overridden — api.py's ``httpx`` reference is unaffected.
    real_httpx_client = httpx.Client  # grab the real class BEFORE patching

    class _MockClientFactory:
        """Context-manager-compatible replacement for httpx.Client in _cache."""

        def __init__(self, *_args, **_kwargs):
            self._client = real_httpx_client(transport=routes.transport())

        def __enter__(self):
            return self._client

        def __exit__(self, *_):
            self._client.close()

    class _HttpxShim:
        """Namespace exposing only what _cache uses from httpx."""

        Client = _MockClientFactory
        Timeout = httpx.Timeout

    monkeypatch.setattr(_cache, "httpx", _HttpxShim)

    yield

    # Cleanup
    mock_client.close()
