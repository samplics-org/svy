# src/svy/datasets/_cache.py
"""
On-disk cache for dataset parquet files.

Design goals
------------
- **Versioned**: cache key is ``{slug}@{version}`` so updates don't collide.
- **Verified**: SHA-256 is checked after download and on every cache hit
  that hasn't been recently validated.
- **Atomic**: downloads write to a unique tempfile and rename on success.
- **Concurrent-safe**: unique tempfiles + atomic rename allow parallel
  loaders of the same slug without corruption.
- **Resumable (via streaming)**: 1 MB chunks keep memory bounded for
  large files while maximizing throughput.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading

from pathlib import Path
from typing import Final

import httpx

from svy.errors.dataset_errors import DatasetError


log = logging.getLogger(__name__)


# --- Configuration -------------------------------------------------------- #

CACHE_DIR: Final[Path] = Path(
    os.getenv("SVYLAB_CACHE_DIR", str(Path.home() / ".svy" / "datasets"))
)

_DOWNLOAD_CHUNK: Final[int] = 1024 * 1024  # 1 MB
_HASH_CHUNK: Final[int] = 1024 * 1024  # 1 MB

_DOWNLOAD_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(
    connect=10.0, read=300.0, write=30.0, pool=10.0
)

# Per-slug download locks: ensures two threads asking for the same dataset
# don't both download it.  One downloads; the other waits and reuses.
_download_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()

# In-process set of (path, sha256) pairs that we've already verified this
# session.  Avoids re-hashing large files on every load() call.
_verified: set[tuple[str, str]] = set()
_verified_lock = threading.Lock()


def _lock_for(slug_version: str) -> threading.Lock:
    with _locks_lock:
        lock = _download_locks.get(slug_version)
        if lock is None:
            lock = threading.Lock()
            _download_locks[slug_version] = lock
        return lock


# --- Public API ----------------------------------------------------------- #


def path_for(slug: str, version: str) -> Path:
    """Return the on-disk path for ``{slug}@{version}.parquet``."""
    # Version is part of the filename so old versions coexist until evicted.
    safe_version = version.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{slug}@{safe_version}.parquet"


def ensure_cached(
    *,
    slug: str,
    version: str,
    url: str,
    sha256: str,
    force: bool = False,
) -> str:
    """
    Ensure a dataset file is available on disk; return its path.

    If the file exists and its hash has already been verified this session,
    returns immediately.  Otherwise verifies (or downloads + verifies).

    Parameters
    ----------
    slug, version : str
        Cache key components.
    url : str
        Where to download if the file is missing.
    sha256 : str
        Expected hex digest.  Verified after download and on first use.
        If empty, integrity checks are skipped (a warning is logged).
    force : bool
        If True, re-download even if a valid cached copy exists.

    Raises
    ------
    DatasetError
        With code ``DATASET_SHA_MISMATCH`` if the hash check fails.
    RuntimeError
        On network or filesystem errors during download.
    """
    local_path = path_for(slug, version)

    # Empty sha256 → catalog doesn't expose one yet.  Skip verification.
    verify = bool(sha256)
    if not verify:
        log.warning("No sha256 provided for %r; integrity check skipped.", slug)

    cache_key = (str(local_path), sha256)

    # Fast path: already validated this session (only meaningful when we
    # actually verified something).
    if verify and not force and cache_key in _verified and local_path.exists():
        log.debug("Cache hit (verified): %s", local_path)
        return str(local_path)

    # Serialize concurrent downloads of the same slug@version.
    with _lock_for(f"{slug}@{version}"):
        # Re-check after acquiring the lock — another thread may have done it.
        if verify and not force and cache_key in _verified and local_path.exists():
            return str(local_path)

        if local_path.exists() and not force:
            if not verify:
                # No hash to check; trust the cache entry.
                log.debug("Cache hit (unverified): %s", local_path)
                return str(local_path)
            # File exists but hasn't been verified this session.  Verify
            # once, then cache the fact.  If hash mismatches, treat as a
            # corrupted cache entry: delete and re-download.
            if _sha256_of(local_path) == sha256.lower():
                _mark_verified(cache_key)
                log.debug("Cache hit (validated): %s", local_path)
                return str(local_path)
            log.warning("Cached file %s failed hash check; re-downloading.", local_path)
            local_path.unlink(missing_ok=True)

        _download(url=url, dest=local_path, sha256=sha256, slug=slug)
        if verify:
            _mark_verified(cache_key)
        return str(local_path)


def clear(slug: str | None = None) -> int:
    """
    Remove cached files.  If ``slug`` is given, only that dataset's files
    (all versions) are removed.  Returns the number of files deleted.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pattern = f"{slug}@*.parquet" if slug else "*.parquet"
    removed = 0
    for f in CACHE_DIR.glob(pattern):
        try:
            f.unlink()
            removed += 1
        except OSError:
            log.warning("Could not remove %s", f)
    with _verified_lock:
        if slug is None:
            _verified.clear()
        else:
            to_remove = {k for k in _verified if Path(k[0]).name.startswith(f"{slug}@")}
            _verified.difference_update(to_remove)
    return removed


# --- Internals ------------------------------------------------------------ #


def _mark_verified(key: tuple[str, str]) -> None:
    with _verified_lock:
        _verified.add(key)


def _sha256_of(path: Path) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _download(*, url: str, dest: Path, sha256: str, slug: str) -> None:
    """Stream ``url`` into ``dest`` atomically.

    Verifies the SHA-256 in-line when ``sha256`` is non-empty; otherwise
    downloads without integrity verification (a warning is logged by the
    caller).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset %r -> %s", slug, dest)

    # Unique temp file to avoid collision between concurrent callers.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{slug}.", suffix=".parquet.tmp", dir=str(CACHE_DIR)
    )
    tmp_path = Path(tmp_name)

    expected = sha256.lower()
    verify = bool(expected)
    h = hashlib.sha256() if verify else None

    try:
        with httpx.Client(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            with client.stream("GET", url) as r:
                r.raise_for_status()
                with os.fdopen(tmp_fd, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=_DOWNLOAD_CHUNK):
                        if h is not None:
                            h.update(chunk)
                        f.write(chunk)

        if verify:
            assert h is not None  # for type checkers
            actual = h.hexdigest()
            if actual != expected:
                tmp_path.unlink(missing_ok=True)
                raise DatasetError.sha_mismatch(
                    where="datasets._cache._download",
                    slug=slug,
                    expected=expected,
                    got=actual,
                )

        # Atomic rename within the same filesystem.
        shutil.move(str(tmp_path), str(dest))
        log.info("Download complete: %s", dest)

    except DatasetError:
        raise
    except Exception as ex:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {slug!r}: {ex}") from ex
