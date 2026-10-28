# src/svy/datasets/base.py
"""
Dataset loader: resolve a slug to its file and registry entry, return a ``Sample``.

- The file and the entry always come from the same source, so the labels describe that file.
- The design and the variable and value labels come from ``Dataset.design`` and ``Dataset.variables``.
"""

from __future__ import annotations

import logging
import os

from typing import Any, Literal, Mapping, get_args

import polars as pl

from svy.core.design import Design
from svy.core.sample import Sample
from svy.datasets import _bundled, _cache, api
from svy.datasets.types import Dataset
from svy.engine.io.core import import_labels_from_svyio_meta
from svy.errors import Code, DatasetWarning
from svy.errors._dataset import DatasetError
from svy.errors._findings import emit_finding


log = logging.getLogger(__name__)

Source = Literal["auto", "remote", "bundled"]

# Env values that force offline (bundled-first) resolution in "auto" mode.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Remote-failure codes that "auto" recovers from by falling back to bundled.
# Note: DATASET_SHA_MISMATCH is deliberately excluded — a hash mismatch is an
# integrity failure that should surface, not be silently swapped for bundled.
_FALLBACK_CODES = frozenset(
    {"CATALOG_UNREACHABLE", "CATALOG_BAD_STATUS", "DATASET_DOWNLOAD_FAILED"}
)


def _offline_env() -> bool:
    """Whether SVYLAB_OFFLINE requests bundled-first resolution."""
    return os.getenv("SVYLAB_OFFLINE", "").strip().lower() in _TRUTHY


def _check_source(source: object, *, where: str) -> None:
    if source not in get_args(Source):
        raise DatasetError.invalid_choice(
            where=where, param="source", got=source, allowed=list(get_args(Source))
        )


def _from_bundled(name: str) -> tuple[pl.DataFrame, Dataset]:
    ds = _bundled.describe(name)
    if ds is None:
        raise DatasetError.not_bundled(
            where="datasets.load(source='bundled')",
            slug=name,
            bundled=sorted(_bundled.slugs()),
        )
    return _bundled.read(name), ds


def _from_remote(name: str, *, force_download: bool) -> tuple[pl.DataFrame, Dataset]:
    """Resolve a slug from the remote catalog, download+cache, and read it."""
    ds = api.describe(name)
    path = _cache.ensure_cached(
        slug=ds.slug,
        version=ds.version,
        url=ds.download_url,
        sha256=ds.sha256,
        force=force_download,
    )
    _cache.write_entry(path, ds)
    return pl.read_parquet(path), ds


def _resolve(name: str, *, source: Source, force_download: bool) -> tuple[pl.DataFrame, Dataset]:
    """Return the frame and registry entry for ``name`` per the ``source`` policy.

    - ``"bundled"``: read the packaged subset; never touch the network.
    - ``"remote"``: require the online catalog; error if unreachable.
    - ``"auto"``: bundled-first when ``SVYLAB_OFFLINE`` is set.  Otherwise a
      full copy already in the local cache is read without touching the
      network; failing that, try remote and, if the catalog/download is
      unreachable, fall back to the bundled subset (with a warning) when one
      exists.  Without ``httpx`` (no ``remote`` extra) the bundled subset is
      the expected result, so that fallback is silent.
    """
    if source == "auto" and _offline_env():
        source = "bundled"

    if source == "bundled":
        return _from_bundled(name)

    if source == "remote":
        return _from_remote(name, force_download=force_download)

    if not force_download:
        cached = _cache.latest_cached(name)
        if cached is not None:
            path, ds = cached
            return pl.read_parquet(path), ds

    # auto: remote, then bundled fallback on network/catalog failure. A slug
    # the catalog does not know but the wheel carries is bundled-only (the
    # synthetic panel): the bundled file IS the dataset, no warning.
    try:
        return _from_remote(name, force_download=force_download)
    except DatasetError as exc:
        if exc.code == "DATASET_NOT_FOUND" and _bundled.has(name):
            return _from_bundled(name)
        # force_download asks for a download, so it is not silently swapped.
        if exc.code == "REMOTE_UNAVAILABLE" and not force_download and _bundled.has(name):
            return _from_bundled(name)
        if exc.code in _FALLBACK_CODES and _bundled.has(name):
            df, ds = _from_bundled(name)
            emit_finding(
                DatasetWarning(
                    code=Code.DATASET_BUNDLED_SUBSET,
                    title="Using the bundled subset",
                    detail=f"Could not reach the dataset catalog; using the bundled subset of "
                    f"{name!r} ({ds.n_rows:,} rows). This is a reduced dataset — results "
                    f"will differ from the full online data. Use source='remote' once "
                    f"online, or set source='bundled' to silence this.",
                    where="datasets.load",
                )
            )
            return df, ds
        raise


def _typed_codes(values: Mapping[Any, str], dtype: pl.DataType) -> dict[Any, str]:
    # JSON keys are strings; a "11" key never matches the integer 11 in the data.
    if dtype.is_integer():
        return {int(code): label for code, label in values.items()}
    if dtype.is_float():
        return {float(code): label for code, label in values.items()}
    return dict(values)


def _to_sample(df: pl.DataFrame, ds: Dataset) -> Sample:
    sample = Sample(data=df, design=Design(**ds.design) if ds.design else None)
    variables = {
        var: {
            "label": spec.get("label"),
            "values": _typed_codes(spec.get("values") or {}, df.schema[var]),
        }
        for var, spec in ds.variables.items()
        if var in df.columns
    }
    if variables:
        import_labels_from_svyio_meta(sample.meta, {"variables": variables}, df)
    return sample


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def load(name: str, *, source: Source = "auto", force_download: bool = False) -> Sample:
    """
    Load an example dataset as a ready ``Sample``.

    The data holds the numeric codes; variable and value labels are in
    ``sample.meta`` and the design is the dataset's own (``info(name).design``).
    Subset with the ``Sample`` tools or ``where=`` at estimation; the raw frame
    is ``load(name).data``.

    Parameters
    ----------
    name : str
        Dataset slug, e.g. ``"hld_sample_wb_2023"``; ``info()`` lists them.
    source : {"auto", "remote", "bundled"}, default "auto"
        Where to load the data from.

        - ``"remote"``  — the full online dataset; errors if the catalog is
          unreachable. Needs the ``remote`` extra
          (``pip install "svy[remote]"``).
        - ``"bundled"`` — the small subset shipped inside the package; never
          touches the network. Deterministic and offline-safe (used by docs
          and CI). Note this is a *reduced* dataset, so results differ from
          the full data.
        - ``"auto"``    — a full copy already in the local cache if there is
          one (no network access); otherwise remote, falling back to the
          bundled subset (with a warning) if the catalog/download is
          unreachable. Without the ``remote`` extra it reads the bundled
          subset silently. Set the ``SVYLAB_OFFLINE`` environment variable
          to make ``"auto"`` prefer the bundled subset without any network
          attempt.
    force_download : bool, default False
        Re-download the parquet file even if cached.  Applies to remote data
        only; ignored when the bundled subset is used.

    Returns
    -------
    Sample
    """
    _check_source(source, where="datasets.load")
    df, ds = _resolve(name, source=source, force_download=force_download)
    return _to_sample(df, ds)
