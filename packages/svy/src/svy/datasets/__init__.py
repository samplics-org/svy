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

from svy.datasets._cache import clear as _clear_cache
from svy.datasets.api import catalog, describe
from svy.datasets.base import load
from svy.datasets.types import Dataset
from svy.errors.dataset_errors import DatasetError


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
    # Errors (re-exported for convenience; canonical location is svy.errors)
    "DatasetError",
]
