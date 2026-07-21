# src/svy/datasets/__init__.py
"""
Dataset loading and catalog access.

The public API lives under ``svy.datasets.*`` and uses short names because
the module namespace already carries the "dataset" context::

    import svy

    # Browse what's available
    for ds in svy.datasets.catalog():
        print(ds.summary())

    # Inspect one dataset
    info = svy.datasets.describe("phia_like")

    # Load data, optionally filtered/sliced
    data = svy.datasets.load("phia_like", n=500, where=pl.col("age") >= 18)

    # Cache management
    svy.datasets.clear_cache()               # all datasets
    svy.datasets.clear_cache("phia_like")    # one dataset, all versions

Public API
----------
load(name, ...)              -- load a dataset as DataFrame/LazyFrame
catalog()                    -- return all available datasets
describe(slug)               -- full metadata for one dataset
clear_cache(slug=None)       -- remove cached parquet files

Types
-----
Dataset                      -- metadata record (msgspec.Struct)

Errors
------
DatasetError                 -- all dataset/catalog/integrity failures;
                                differentiate via ``.code``
                                (CATALOG_UNREACHABLE, CATALOG_BAD_STATUS,
                                 DATASET_NOT_FOUND, DATASET_SHA_MISMATCH).

``DatasetError`` lives in ``svy.errors.dataset_errors`` alongside the rest
of the error taxonomy and inherits from ``SvyError``.
"""

from __future__ import annotations

import warnings

from svy.datasets import _bundled, api
from svy.datasets._cache import clear as _clear_cache
from svy.datasets.base import Source, _offline_env, load
from svy.datasets.types import Dataset, DatasetCatalog
from svy.errors.dataset_errors import DatasetError


_CATALOG_UNREACHABLE = {"CATALOG_UNREACHABLE", "CATALOG_BAD_STATUS"}


def catalog(*, use_cache: bool = True, source: Source = "auto") -> DatasetCatalog:
    """
    Return all available datasets.

    Parameters
    ----------
    use_cache : bool, default True
        Reuse the in-process catalog cache (remote only).
    source : {"auto", "remote", "bundled"}, default "auto"
        ``"remote"`` queries the online catalog; ``"bundled"`` lists only the
        packaged subsets (offline); ``"auto"`` tries remote and falls back to
        the bundled list if the catalog is unreachable (or immediately when
        ``SVYLAB_OFFLINE`` is set).
    """
    if source == "bundled" or (source == "auto" and _offline_env()):
        return _bundled.catalog()
    if source == "remote":
        return api.catalog(use_cache=use_cache)
    try:
        return api.catalog(use_cache=use_cache)
    except DatasetError as exc:
        if exc.code in _CATALOG_UNREACHABLE and _bundled.slugs():
            warnings.warn(
                "Dataset catalog unreachable; listing bundled subsets only.",
                stacklevel=2,
            )
            return _bundled.catalog()
        raise


def describe(slug: str, *, use_cache: bool = True, source: Source = "auto") -> Dataset:
    """
    Return metadata for a single dataset by slug.

    Honors the same ``source`` policy as :func:`catalog`.  Under ``"auto"``,
    bundled metadata is used only as a fallback when the catalog is
    unreachable (or immediately when ``SVYLAB_OFFLINE`` is set).
    """
    if source == "bundled" or (source == "auto" and _offline_env()):
        ds = _bundled.describe(slug)
        if ds is None:
            raise DatasetError.not_found(where="datasets.describe(source='bundled')", slug=slug)
        return ds
    if source == "remote":
        return api.describe(slug, use_cache=use_cache)
    try:
        return api.describe(slug, use_cache=use_cache)
    except DatasetError as exc:
        if exc.code in _CATALOG_UNREACHABLE:
            ds = _bundled.describe(slug)
            if ds is not None:
                warnings.warn(
                    f"Dataset catalog unreachable; using bundled metadata for {slug!r}.",
                    stacklevel=2,
                )
                return ds
        raise


def clear_cache(slug: str | None = None) -> int:
    """
    Remove cached parquet files.

    Parameters
    ----------
    slug : str, optional
        If given, only remove files for that dataset (all versions).
        Otherwise, clear the entire dataset cache.

    Returns
    -------
    int
        Number of files removed.
    """
    return _clear_cache(slug)


__all__ = [
    # Main functions
    "load",
    "catalog",
    "describe",
    "clear_cache",
    # Types
    "Dataset",
    "DatasetCatalog",
    # Errors (re-exported for convenience; canonical location is svy.errors)
    "DatasetError",
]
