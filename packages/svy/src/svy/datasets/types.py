# src/svy/datasets/types.py
"""
Type definitions for the datasets module.

Mirrors the convention of ``svy/core/types.py``: lightweight msgspec structs
and type aliases that callers across the module import.  Kept in one place
so the module surface area stays discoverable.

Design notes
------------
- ``Dataset.design`` is a plain dict, not a typed struct, so it stays
  decoupled from ``svy.Design``'s evolving signature.  Users spread it:
  ``svy.Design(**ds.design)``.
- ``Dataset.variables`` is a dict keyed by column name with free-form
  per-column metadata.  Phase 2 may tighten this to use
  ``MeasurementType`` / ``MissingKind`` from ``svy.core.enumerations``.
"""

from __future__ import annotations

from typing import Any, Mapping

import msgspec


class Dataset(msgspec.Struct, frozen=True, kw_only=True):
    """
    Metadata for a single example dataset in the svylab catalog.

    Attributes
    ----------
    slug : str
        Stable short identifier.  Used by ``svy.datasets.load(name=slug)``.
    title, description : str
        Human-facing name and prose description.
    version : str
        Semantic version of the dataset content.  Changing this invalidates
        client caches so users receive updates.
    download_url : str
        Fully qualified URL to the parquet file.  May point at the svylab
        API, a CDN, or an object store — the client does not care.
    sha256 : str
        Hex digest for integrity verification after download.
    size_bytes, n_rows, n_cols : int
        File and shape metadata.
    license, citation, source : str
        Attribution and reuse metadata.
    design : Mapping[str, Any] | None
        Survey design kwargs.  Spread directly into ``svy.Design(...)``::

            svy.Design(**ds.design)

        Typical keys: ``stratum``, ``psu``, ``ssu``, ``wgt``, ``fpc``,
        ``rep_weights``.  Whatever keys ``svy.Design`` accepts work here.
    variables : Mapping[str, Mapping[str, Any]]
        Column-level metadata keyed by column name.  Free-form per-column
        entries (``label``, ``unit``, ``categories``, etc.).
    tags : tuple[str, ...]
        Free-form tags for discovery.
    """

    slug: str
    title: str
    description: str
    version: str
    download_url: str
    sha256: str
    size_bytes: int
    n_rows: int
    n_cols: int
    license: str
    citation: str
    source: str
    design: Mapping[str, Any] | None = None
    variables: Mapping[str, Mapping[str, Any]] = {}
    tags: tuple[str, ...] = ()

    def summary(self) -> str:
        """One-line human-readable summary for list displays."""
        size_mb = self.size_bytes / (1024 * 1024)
        return f"{self.slug:<24} {self.title:<40} {self.n_rows:>10,} rows  {size_mb:>6.1f} MB"
