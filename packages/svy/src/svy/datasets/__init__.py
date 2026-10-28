# src/svy/datasets/__init__.py
"""
Example datasets: ``info`` tells you about them, ``load`` gives you one as a ``Sample``.

    import svy

    svy.datasets.info()                       # every dataset, printed as a table
    svy.datasets.info("hld_sample_wb_2023")   # one dataset's entry
    sample = svy.datasets.load("hld_sample_wb_2023")   # codes + design + labels
    svy.datasets.clear_cache()                # remove cached downloads

``DatasetError`` (``svy.errors``) covers every catalog, download and integrity
failure; tell them apart by ``.code``.  The online catalog needs the ``remote``
extra (``pip install "svy[remote]"``); without it, the bundled subsets work and
svy makes no network calls.
"""

from __future__ import annotations

from typing import overload

from svy.datasets import _bundled, api
from svy.datasets._cache import clear as _clear_cache
from svy.datasets.base import Source, _check_source, _offline_env, load
from svy.datasets.types import Dataset, DatasetCatalog
from svy.errors import Code, DatasetWarning
from svy.errors._dataset import DatasetError
from svy.errors._findings import emit_finding


_CATALOG_UNREACHABLE = {"CATALOG_UNREACHABLE", "CATALOG_BAD_STATUS"}


def _catalog(*, use_cache: bool, source: Source) -> DatasetCatalog:
    if source == "bundled" or (source == "auto" and _offline_env()):
        return _bundled.catalog()
    if source == "remote":
        return api.catalog(use_cache=use_cache)
    try:
        return api.catalog(use_cache=use_cache)
    except DatasetError as exc:
        if exc.code == "REMOTE_UNAVAILABLE" and _bundled.slugs():
            return _bundled.catalog()
        if exc.code in _CATALOG_UNREACHABLE and _bundled.slugs():
            emit_finding(
                DatasetWarning(
                    code=Code.DATASET_CATALOG_OFFLINE,
                    title="Dataset catalog unreachable",
                    detail="Dataset catalog unreachable; listing bundled subsets only.",
                    where="datasets.info",
                )
            )
            return _bundled.catalog()
        raise


def _describe(slug: str, *, use_cache: bool, source: Source) -> Dataset:
    if source == "bundled" or (source == "auto" and _offline_env()):
        ds = _bundled.describe(slug)
        if ds is None:
            raise DatasetError.not_bundled(
                where="datasets.info(source='bundled')",
                slug=slug,
                bundled=sorted(_bundled.slugs()),
            )
        return ds
    if source == "remote":
        return api.describe(slug, use_cache=use_cache)
    try:
        return api.describe(slug, use_cache=use_cache)
    except DatasetError as exc:
        if exc.code in {"DATASET_NOT_FOUND", "REMOTE_UNAVAILABLE"}:
            ds = _bundled.describe(slug)
            if ds is not None:
                return ds
        if exc.code in _CATALOG_UNREACHABLE:
            ds = _bundled.describe(slug)
            if ds is not None:
                emit_finding(
                    DatasetWarning(
                        code=Code.DATASET_CATALOG_OFFLINE,
                        title="Dataset catalog unreachable",
                        detail=f"Dataset catalog unreachable; using bundled metadata for {slug!r}.",
                        where="datasets.info",
                    )
                )
                return ds
        raise


@overload
def info(
    name: None = None, *, use_cache: bool = True, source: Source = "auto"
) -> DatasetCatalog: ...


@overload
def info(name: str, *, use_cache: bool = True, source: Source = "auto") -> Dataset: ...


def info(
    name: str | None = None, *, use_cache: bool = True, source: Source = "auto"
) -> DatasetCatalog | Dataset:
    """
    Every available dataset, or the entry of the one named.

    Parameters
    ----------
    name : str, optional
        Dataset slug.  Without it, the whole catalog.
    use_cache : bool, default True
        Reuse the in-process catalog cache (remote only).
    source : {"auto", "remote", "bundled"}, default "auto"
        ``"remote"`` queries the online catalog; ``"bundled"`` reads only the
        packaged subsets (offline); ``"auto"`` tries remote and falls back to
        the bundled entries if the catalog is unreachable (or immediately when
        ``SVYLAB_OFFLINE`` is set, or silently without the ``remote`` extra).

    Returns
    -------
    DatasetCatalog
        Without ``name``; prints as a table.
    Dataset
        With ``name``; ``.design`` and ``.variables`` are what ``load`` applies.
    """
    _check_source(source, where="datasets.info")
    if name is None:
        return _catalog(use_cache=use_cache, source=source)
    return _describe(name, use_cache=use_cache, source=source)


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


_MOVED = {"catalog": "svy.datasets.info()", "describe": "svy.datasets.info(name)"}


def __getattr__(name: str):
    if name in _MOVED:
        from svy.errors._api import UnknownNameError

        raise UnknownNameError.renamed(
            label="svy.datasets", name=name, new=_MOVED[name], since="0.36.0"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main functions
    "load",
    "info",
    "clear_cache",
    # Types
    "Dataset",
    "DatasetCatalog",
    # Errors (re-exported for convenience; canonical location is svy.errors)
    "DatasetError",
]
