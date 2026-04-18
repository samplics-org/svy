# src/svy/datasets/api.py
"""
Catalog API client for svylab datasets.

Responsibilities
----------------
- Fetch dataset metadata from the catalog service.
- Translate the backend's registry shape into typed ``Dataset`` objects.
- Reuse a single HTTP connection across calls.
- Cache list/describe results briefly to avoid redundant network hits.

Non-responsibilities
--------------------
- Downloading parquet files (that's ``_cache`` + ``base``).

Backend contract
----------------
This client targets svylab's ``/api/data/examples/*`` endpoints:

* ``GET /api/data/examples/registry`` → ``list[BackendDataset]``
* ``GET /api/data/examples/by-name/{slug}/download`` → parquet file

The backend's registry shape is translated into our typed ``Dataset`` via
``_from_backend``.  Fields the backend doesn't provide (sha256, size_bytes,
n_rows, n_cols, license, citation, source, design, variables) are filled
with safe defaults.  When ``sha256`` is empty, download integrity checks are
skipped — see ``_cache.ensure_cached``.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from typing import Any, Final

import httpx
import msgspec

from svy.datasets.types import Dataset
from svy.errors.dataset_errors import DatasetError


log = logging.getLogger(__name__)


# --- Configuration -------------------------------------------------------- #

API_URL: Final[str] = os.getenv("SVYLAB_API_URL", "https://svylab.com").rstrip("/")

# Endpoint paths (backend-specific).  Exposed at module level so tests can
# patch them without rebuilding the whole URL.
_REGISTRY_PATH: Final[str] = "/api/data/examples/registry"
_DOWNLOAD_PATH_TMPL: Final[str] = "/api/data/examples/by-name/{slug}/download"

# Separate, short timeouts for metadata (unlike parquet downloads which are long).
_API_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)

# Default TTL for in-process caches.
_LIST_TTL_SECONDS: Final[float] = 300.0  # 5 min
_GET_TTL_SECONDS: Final[float] = 300.0  # 5 min


# --- Shared client (lazy, thread-safe) ------------------------------------ #

_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _get_client() -> httpx.Client:
    """Return a process-wide shared ``httpx.Client``.

    Lazily initialized so importing this module doesn't open a connection.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    follow_redirects=True,
                    timeout=_API_TIMEOUT,
                    headers={"User-Agent": "svy-datasets-client"},
                )
    return _client


def close() -> None:
    """Close the shared HTTP client (e.g., at process shutdown)."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


# --- Internal fetch helper ------------------------------------------------ #


def _fetch(path: str) -> bytes:
    """Issue a GET against the configured catalog and return raw bytes."""
    url = f"{API_URL}/{path.lstrip('/')}"
    try:
        r = _get_client().get(url)
        r.raise_for_status()
        return r.content
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 404:
            # Re-raised by callers with dataset-specific context.
            raise
        raise DatasetError.catalog_bad_status(
            where="datasets.api._fetch", url=url, status=status
        ) from e
    except httpx.HTTPError as e:
        raise DatasetError.catalog_unreachable(
            where="datasets.api._fetch", url=url, reason=str(e)
        ) from e


# --- Backend-shape translation ------------------------------------------- #


def _download_url_for(slug: str) -> str:
    """Build the absolute download URL for a given slug."""
    return f"{API_URL}{_DOWNLOAD_PATH_TMPL.format(slug=slug)}"


def _from_backend(entry: dict[str, Any]) -> Dataset:
    """
    Translate one backend registry entry into a typed ``Dataset``.

    The backend's shape is minimal — missing fields are filled with safe
    defaults.  When/if the backend starts returning more metadata, this
    function is the single place to widen the translation.

    Backend shape (observed)::

        {
            "name": "...",
            "slug": "...",
            "version": "v1.0.0",
            "description": "...",
            "dataset": "file.parquet",
            "format": "parquet",
            "source": {"base_url": "...", "path": "..."},
            "tags": [...],
            "subpath": "..."
        }
    """
    slug = entry["slug"]

    # Prefer a URL the backend constructs; fall back to synthesizing one.
    src = entry.get("source") or {}
    base_url = (src.get("base_url") or API_URL).rstrip("/")
    path = src.get("path") or _DOWNLOAD_PATH_TMPL.format(slug=slug)
    if path.startswith(("http://", "https://")):
        download_url = path
    else:
        download_url = f"{base_url}{path}"

    return Dataset(
        slug=slug,
        title=entry.get("name", slug),
        description=entry.get("description", ""),
        version=entry.get("version", "0.0.0"),
        download_url=download_url,
        sha256=entry.get("sha256", ""),  # backend may start sending this
        size_bytes=entry.get("size_bytes", 0),
        n_rows=entry.get("n_rows", 0),
        n_cols=entry.get("n_cols", 0),
        license=entry.get("license", "unknown"),
        citation=entry.get("citation", ""),
        source=entry.get("source_citation", entry.get("name", "svylab")),
        design=entry.get("design"),
        variables=entry.get("variables", {}),
        tags=tuple(entry.get("tags", [])),
    )


# Cached decoder for the backend's raw registry list.  Stays a plain
# ``list[dict]`` because the backend schema is still in flux; we translate
# to typed ``Dataset`` after decoding.
_RAW_LIST_DECODER: Final = msgspec.json.Decoder(list[dict])


# --- TTL caches ----------------------------------------------------------- #


class _TTLEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: object, ttl: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl


_list_cache: _TTLEntry | None = None
_get_cache: dict[str, _TTLEntry] = {}
_cache_lock = threading.Lock()


def _cache_get_list() -> tuple[Dataset, ...] | None:
    with _cache_lock:
        entry = _list_cache
        if entry is None or entry.expires_at < time.monotonic():
            return None
        return entry.value  # type: ignore[return-value]


def _cache_put_list(value: tuple[Dataset, ...]) -> None:
    global _list_cache
    with _cache_lock:
        _list_cache = _TTLEntry(value, _LIST_TTL_SECONDS)


def _cache_get_one(slug: str) -> Dataset | None:
    with _cache_lock:
        entry = _get_cache.get(slug)
        if entry is None or entry.expires_at < time.monotonic():
            return None
        return entry.value  # type: ignore[return-value]


def _cache_put_one(slug: str, value: Dataset) -> None:
    with _cache_lock:
        _get_cache[slug] = _TTLEntry(value, _GET_TTL_SECONDS)


def clear_cache() -> None:
    """Drop all cached catalog responses.  Useful in tests or after updates."""
    global _list_cache
    with _cache_lock:
        _list_cache = None
        _get_cache.clear()


# --- Public API ----------------------------------------------------------- #


def catalog(*, use_cache: bool = True) -> tuple[Dataset, ...]:
    """
    Return all datasets known to the svylab catalog.

    Results are cached in-process for a few minutes.  Pass
    ``use_cache=False`` to force a fresh fetch.
    """
    if use_cache:
        cached = _cache_get_list()
        if cached is not None:
            return cached

    raw = _fetch(_REGISTRY_PATH)
    entries = _RAW_LIST_DECODER.decode(raw)
    datasets = tuple(_from_backend(e) for e in entries)
    _cache_put_list(datasets)

    # Populate per-slug cache too — saves a round-trip on the next describe().
    with _cache_lock:
        for ds in datasets:
            _get_cache[ds.slug] = _TTLEntry(ds, _GET_TTL_SECONDS)

    return datasets


def describe(slug: str, *, use_cache: bool = True) -> Dataset:
    """
    Return metadata for a single dataset by slug.

    The backend currently exposes only a registry list endpoint (not a
    per-slug detail endpoint), so we fetch the registry, cache it, and look
    up the slug locally.  This is efficient: the registry is small, the
    ``catalog()`` result populates the per-slug cache, and subsequent
    lookups are in-memory.

    Raises
    ------
    DatasetError
        With code ``DATASET_NOT_FOUND`` if the slug is unknown, or
        ``CATALOG_*`` codes if the catalog service is unreachable.
    """
    if use_cache:
        cached = _cache_get_one(slug)
        if cached is not None:
            return cached

    # Force a fresh registry fetch.  ``catalog()`` populates _get_cache, so
    # after this call the slug should be there — or it genuinely doesn't
    # exist in the registry.
    catalog(use_cache=use_cache)

    cached = _cache_get_one(slug)
    if cached is not None:
        return cached

    raise DatasetError.not_found(where="datasets.api.describe", slug=slug)
