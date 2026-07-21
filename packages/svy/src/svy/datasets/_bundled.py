# src/svy/datasets/_bundled.py
"""
Access to the small dataset subsets bundled inside the wheel.

These files live in ``svy/datasets/_bundled/`` and ship with the package so
``svy.datasets`` works offline.  They are *reduced* views of the full online
datasets (fewer rows); see ``scripts/build_bundled_datasets.py`` for how they
are produced and ``_bundled/registry.json`` for their metadata.

This module is intentionally dependency-light and network-free.  It exposes:

- ``slugs()``            -- set of bundled dataset slugs
- ``has(slug)``          -- whether a slug is bundled
- ``describe(slug)``     -- bundled ``Dataset`` metadata (``download_url=""``)
- ``catalog()``          -- all bundled ``Dataset`` records
- ``read_lazy(slug)``    -- the bundled parquet as an in-memory ``LazyFrame``

``read_lazy`` reads the file eagerly and returns ``.lazy()`` so the result is
detached from the packaged resource — this keeps it correct even when the
package is imported from a zip (``importlib.resources.as_file`` may hand back a
temporary path that is cleaned up on context exit).
"""

from __future__ import annotations

import json

from functools import lru_cache
from importlib.resources import as_file, files
from typing import Any

import polars as pl

from svy.datasets.types import Dataset
from svy.errors.dataset_errors import DatasetError


_PACKAGE = "svy.datasets"
_DIRNAME = "_bundled"
_REGISTRY = "registry.json"


def _dir():
    """Traversable for the bundled data directory inside the package."""
    return files(_PACKAGE) / _DIRNAME


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[str, Any]]:
    """Parse ``registry.json`` once, keyed by slug.

    Degrades to an empty registry (i.e. "no bundled data") if the file is
    absent or unparseable, so a broken/partial install never breaks the
    remote path.  The build-time benchmark test guards registry validity.
    """
    try:
        raw = (_dir() / _REGISTRY).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return {}
    try:
        entries = json.loads(raw)
        return {e["slug"]: e for e in entries}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _to_dataset(entry: dict[str, Any]) -> Dataset:
    """Build a ``Dataset`` from a bundled registry entry.

    ``download_url`` is empty because bundled data is read from the package,
    never fetched.  All other fields come straight from the registry.
    """
    return Dataset(
        slug=entry["slug"],
        title=entry.get("title", entry["slug"]),
        description=entry.get("description", ""),
        version=entry.get("version", "0.0.0"),
        download_url="",
        sha256=entry.get("sha256", ""),
        size_bytes=entry.get("size_bytes", 0),
        n_rows=entry.get("n_rows", 0),
        n_cols=entry.get("n_cols", 0),
        license=entry.get("license", "unknown"),
        citation=entry.get("citation", ""),
        source=entry.get("source", "bundled"),
        design=entry.get("design"),
        variables=entry.get("variables", {}),
        tags=tuple(entry.get("tags", ())),
    )


def slugs() -> frozenset[str]:
    """Return the set of bundled dataset slugs."""
    return frozenset(_registry())


def has(slug: str) -> bool:
    """Whether ``slug`` has a bundled copy."""
    return slug in _registry()


def describe(slug: str) -> Dataset | None:
    """Return bundled metadata for ``slug``, or ``None`` if not bundled."""
    entry = _registry().get(slug)
    return _to_dataset(entry) if entry is not None else None


def catalog() -> tuple[Dataset, ...]:
    """Return all bundled datasets as ``Dataset`` records."""
    return tuple(_to_dataset(e) for e in _registry().values())


def read_lazy(slug: str) -> pl.LazyFrame:
    """Read the bundled parquet for ``slug`` as an in-memory ``LazyFrame``.

    Raises
    ------
    KeyError
        If ``slug`` is not bundled (callers should check ``has(slug)`` first).
    DatasetError
        With code ``BUNDLED_UNAVAILABLE`` if the packaged file is missing or
        unreadable.
    """
    entry = _registry().get(slug)
    if entry is None:
        raise KeyError(slug)
    resource = _dir() / entry["filename"]
    try:
        with as_file(resource) as path:
            df = pl.read_parquet(path)
    except DatasetError:
        raise
    except Exception as ex:  # missing file, unreadable parquet, extraction error
        raise DatasetError.bundled_unavailable(
            where="datasets._bundled.read_lazy", slug=slug, reason=str(ex)
        ) from ex
    return df.lazy()
