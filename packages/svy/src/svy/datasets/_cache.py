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
import re
import shutil
import tempfile
import threading

from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

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


# --- Validation ------------------------------------------------------------ #

# Slugs come from the (untrusted) backend registry and are interpolated into
# cache paths, glob patterns and tempfile prefixes.  Allowlist them strictly:
# a permissive slug like "../../foo" would write attacker-controlled bytes
# outside CACHE_DIR.
_SLUG_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Hosts for which plain http is acceptable (local development / tests).
_LOCAL_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})


def validate_slug(slug: str, *, where: str) -> str:
    """Return ``slug`` if it is safe to use in filesystem paths; raise otherwise."""
    if not _SLUG_RE.fullmatch(slug) or ".." in slug:
        raise DatasetError.invalid_slug(where=where, slug=slug)
    return slug


def _require_https(url: str, *, slug: str, where: str) -> None:
    """Reject non-https download URLs (plain http allowed for localhost only)."""
    parts = urlsplit(url)
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and (parts.hostname or "").lower() in _LOCAL_HOSTS:
        return
    raise DatasetError.insecure_url(where=where, slug=slug, url=url)


# --- Public API ----------------------------------------------------------- #


def path_for(slug: str, version: str) -> Path:
    """Return the on-disk path for ``{slug}@{version}.parquet``."""
    validate_slug(slug, where="datasets._cache.path_for")
    # Version is part of the filename so old versions coexist until evicted.
    safe_version = version.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{slug}@{safe_version}.parquet"


def _pin_path(local_path: Path) -> Path:
    """Sidecar file holding the first-seen (TOFU) sha256 for a cache entry."""
    return local_path.with_name(local_path.name + ".sha256")


def _read_pin(local_path: Path) -> str:
    """Return the pinned hex digest for ``local_path``, or '' if none."""
    try:
        pin = _pin_path(local_path).read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ""
    # A corrupt/truncated pin must not silently disable verification forever.
    return pin if re.fullmatch(r"[0-9a-f]{64}", pin) else ""


def _write_pin(local_path: Path, digest: str) -> None:
    _pin_path(local_path).write_text(digest + "\n", encoding="utf-8")


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
        Where to download if the file is missing.  Must be https (plain
        http is allowed for localhost only), including after redirects.
    sha256 : str
        Expected hex digest from the catalog.  Verified after download and
        on first use.  If empty, the digest of the first successful download
        is pinned on disk (trust-on-first-use) and enforced from then on.
    force : bool
        If True, re-download even if a valid cached copy exists.  The
        re-downloaded bytes must still match the catalog hash or TOFU pin.

    Raises
    ------
    DatasetError
        With code ``DATASET_SHA_MISMATCH`` if the hash check fails,
        ``DATASET_INVALID_SLUG`` / ``DATASET_INSECURE_URL`` on unsafe
        metadata, or ``DATASET_DOWNLOAD_FAILED`` on network/filesystem
        errors during download.
    """
    _WHERE = "datasets._cache.ensure_cached"
    validate_slug(slug, where=_WHERE)
    local_path = path_for(slug, version)

    # Effective expected hash: the catalog's, or the TOFU pin from an
    # earlier download.  Empty only before the very first download.
    expected = sha256.lower() if sha256 else _read_pin(local_path)
    tofu = not sha256
    verify = bool(expected)

    cache_key = (str(local_path), expected)

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
        # Another thread may have just created the first TOFU pin.
        if tofu and not expected:
            expected = _read_pin(local_path)
            verify = bool(expected)
            cache_key = (str(local_path), expected)

        if local_path.exists() and not force:
            if not verify:
                # Cached file predating TOFU pinning: adopt its digest as
                # the pin rather than trusting it silently forever.
                digest = _sha256_of(local_path)
                _write_pin(local_path, digest)
                _mark_verified((str(local_path), digest))
                log.info("Pinned sha256 for cached %r: %s", slug, digest)
                return str(local_path)
            # File exists but hasn't been verified this session.  Verify
            # once, then cache the fact.  If hash mismatches, treat as a
            # corrupted cache entry: delete and re-download (the fresh
            # download is verified against the same expected hash).
            if _sha256_of(local_path) == expected:
                _mark_verified(cache_key)
                log.debug("Cache hit (validated): %s", local_path)
                return str(local_path)
            log.warning("Cached file %s failed hash check; re-downloading.", local_path)
            local_path.unlink(missing_ok=True)

        _require_https(url, slug=slug, where=_WHERE)
        digest = _download(url=url, dest=local_path, sha256=expected, slug=slug)
        if tofu and not expected:
            # First-ever download without a catalog hash: pin what we got.
            _write_pin(local_path, digest)
            log.info(
                "No catalog sha256 for %r; pinned first-download hash %s "
                "(future downloads must match).",
                slug,
                digest,
            )
        _mark_verified((str(local_path), digest))
        return str(local_path)


def clear(slug: str | None = None) -> int:
    """
    Remove cached files (parquet + TOFU hash pins).  If ``slug`` is given,
    only that dataset's files (all versions) are removed.  Returns the
    number of parquet files deleted.
    """
    if slug is not None:
        validate_slug(slug, where="datasets._cache.clear")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pattern = f"{slug}@*.parquet" if slug else "*.parquet"
    removed = 0
    for f in CACHE_DIR.glob(pattern):
        try:
            f.unlink()
            removed += 1
        except OSError:
            log.warning("Could not remove %s", f)
        _pin_path(f).unlink(missing_ok=True)
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


def _download(*, url: str, dest: Path, sha256: str, slug: str) -> str:
    """Stream ``url`` into ``dest`` atomically; return the actual hex digest.

    Verifies the SHA-256 in-line when ``sha256`` is non-empty.  The digest
    is always computed and returned so the caller can TOFU-pin it.  The
    final URL after redirects must still satisfy the https policy — a
    compromised catalog must not be able to downgrade an https download to
    plain http via a redirect.
    """
    _WHERE = "datasets._cache._download"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset %r -> %s", slug, dest)

    # Unique temp file to avoid collision between concurrent callers.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{slug}.", suffix=".parquet.tmp", dir=str(CACHE_DIR)
    )
    tmp_path = Path(tmp_name)

    expected = sha256.lower()
    h = hashlib.sha256()

    try:
        with httpx.Client(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            with client.stream("GET", url) as r:
                r.raise_for_status()
                _require_https(str(r.url), slug=slug, where=_WHERE)
                with os.fdopen(tmp_fd, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=_DOWNLOAD_CHUNK):
                        h.update(chunk)
                        f.write(chunk)

        actual = h.hexdigest()
        if expected and actual != expected:
            tmp_path.unlink(missing_ok=True)
            raise DatasetError.sha_mismatch(
                where=_WHERE,
                slug=slug,
                expected=expected,
                got=actual,
            )

        # Atomic rename within the same filesystem.
        shutil.move(str(tmp_path), str(dest))
        log.info("Download complete: %s", dest)
        return actual

    except DatasetError:
        raise
    except Exception as ex:
        tmp_path.unlink(missing_ok=True)
        raise DatasetError.download_failed(
            where=_WHERE, slug=slug, url=url, reason=str(ex)
        ) from ex
