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

from typing import Any, ClassVar, Mapping

import msgspec


def _fmt_size(n_bytes: int) -> str:
    """Human-readable file size (KB under 1 MB, else MB)."""
    mb = n_bytes / (1024 * 1024)
    if mb >= 1:
        return f"{mb:.1f} MB"
    return f"{n_bytes / 1024:.0f} KB"


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

    # ---- display ---------------------------------------------------------
    # Class-level print-width override (ClassVar so msgspec treats it as a
    # class attribute, not a struct field).
    PRINT_WIDTH: ClassVar[int | None] = None

    def __rich_console__(self, console, options):
        from rich.table import Table as RTable

        from svy.ui.printing import make_panel

        t = RTable(
            show_header=False,
            box=None,
            show_edge=False,
            pad_edge=False,
            expand=False,
            padding=(0, 2),
        )
        t.add_column("Field", justify="left", no_wrap=True, style="bold")
        # Cap the value width so long descriptions wrap instead of stretching
        # the panel.
        t.add_column("Value", justify="left", overflow="fold", max_width=60)

        t.add_row("Title", self.title)
        t.add_row("Description", self.description or "—")
        t.add_row("Rows × Cols", f"{self.n_rows:,} × {self.n_cols}")
        t.add_row("Size", _fmt_size(self.size_bytes))
        t.add_row("Version", self.version)
        # One row per design key so long designs stay readable.
        if self.design:
            t.add_row("Design", "")
            for key, value in self.design.items():
                t.add_row("", f"{key} = {value!r}")
        else:
            t.add_row("Design", "—")
        t.add_row("Source", self.source or "—")
        t.add_row("License", self.license or "—")
        if self.tags:
            t.add_row("Tags", ", ".join(self.tags))
        yield make_panel([t], title=f"Dataset: {self.slug}", obj=self, kind="panel")

    def __plain_str__(self) -> str:
        lines = [
            f"Dataset: {self.slug}",
            f"  Title       : {self.title}",
            f"  Description : {self.description}",
            f"  Rows x Cols : {self.n_rows:,} x {self.n_cols}",
            f"  Size        : {_fmt_size(self.size_bytes)}",
            f"  Version     : {self.version}",
        ]
        if self.design:
            lines.append("  Design      :")
            for key, value in self.design.items():
                lines.append(f"      {key} = {value!r}")
        else:
            lines.append("  Design      : —")
        return "\n".join(lines)

    def __str__(self) -> str:
        from svy.ui.printing import render_rich_to_str, resolve_width

        try:
            return render_rich_to_str(self, width=resolve_width(self))
        except Exception:
            return self.__plain_str__()

    # Plain (no ANSI) repr — safe in logs and renders cleanly as a bare
    # expression in notebooks. Use print() / str() for the rich panel.
    def __repr__(self) -> str:
        return self.__plain_str__()


class DatasetCatalog(tuple):
    """
    An immutable, iterable collection of :class:`Dataset` records.

    Behaves exactly like a ``tuple`` (index it, iterate it, ``len()`` it) but
    prints as a compact table and can be exported to Polars.  Use
    :meth:`get` to drill into a single dataset's full metadata.
    """

    PRINT_WIDTH: int | None = None

    def get(self, slug: str) -> Dataset:
        """Return the :class:`Dataset` with ``slug``.

        Raises
        ------
        DatasetError
            With code ``DATASET_NOT_FOUND`` if no dataset has that slug.
        """
        for ds in self:
            if ds.slug == slug:
                return ds
        from svy.errors.dataset_errors import DatasetError

        raise DatasetError.not_found(where="DatasetCatalog.get", slug=slug)

    @property
    def slugs(self) -> tuple[str, ...]:
        """The slugs of every dataset in the catalog."""
        return tuple(ds.slug for ds in self)

    def to_polars(self):
        """Return the catalog as a Polars DataFrame (one row per dataset)."""
        import polars as pl

        return pl.DataFrame(
            {
                "slug": [d.slug for d in self],
                "title": [d.title for d in self],
                "rows": [d.n_rows for d in self],
                "cols": [d.n_cols for d in self],
                "size_mb": [round(d.size_bytes / (1024 * 1024), 2) for d in self],
            }
        )

    def __rich_console__(self, console, options):
        from rich import box
        from rich.table import Table
        from rich.text import Text

        from svy.ui.printing import make_panel

        # Header: summary stats (borderless key/value grid).
        header = Table(show_header=False, box=None, pad_edge=False, expand=False, padding=(0, 2))
        header.add_column(justify="left", style="bold")
        header.add_column(justify="left")
        header.add_row("Number of datasets", str(len(self)))
        if self:
            sizes = [d.size_bytes for d in self]
            header.add_row("Max size", _fmt_size(max(sizes)))
            header.add_row("Min size", _fmt_size(min(sizes)))

        # Body: one row per dataset, with a header rule (svy house style).
        # The title is cropped with an ellipsis when it exceeds the column.
        table = Table(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            expand=False,
            padding=(0, 2),
        )
        table.add_column("slug", justify="left", no_wrap=True)
        table.add_column("title", justify="left", no_wrap=True, overflow="ellipsis", max_width=40)
        table.add_column("rows", justify="right", no_wrap=True)
        table.add_column("cols", justify="right", no_wrap=True)
        for d in self:
            table.add_row(d.slug, d.title, f"{d.n_rows:,}", str(d.n_cols))

        yield make_panel([header, Text(""), table], title="Datasets", obj=self, kind="panel")

    def __plain_str__(self) -> str:
        lines = [f"Datasets: {len(self)}"]
        if self:
            sizes = [d.size_bytes for d in self]
            lines.append(f"Size: {_fmt_size(min(sizes))} – {_fmt_size(max(sizes))}")
        lines.append("")
        for d in self:
            lines.append(f"  {d.slug:<24} {d.n_rows:>10,} rows  {d.n_cols:>3} cols")
        return "\n".join(lines)

    def __str__(self) -> str:
        from svy.ui.printing import render_rich_to_str, resolve_width

        try:
            return render_rich_to_str(self, width=resolve_width(self))
        except Exception:
            return self.__plain_str__()

    # Plain (no ANSI) repr — safe in logs; use print() / str() for the rich
    # table.
    def __repr__(self) -> str:
        return self.__plain_str__()
