# src/svy/wrangling/_helpers.py
"""
Shared internal helpers for the wrangling namespace.

Provides the copy-on-write machinery (_fork, _resolve_target),
design-column introspection (_design_source_columns, _internal_columns,
_required_columns), concatenated-column rebuilding, and design
auto-cleaning.

These helpers operate on Sample objects and are used by every wrangling
module (columns, values, rows, mutate, labels).
"""

from __future__ import annotations

import copy

from typing import TYPE_CHECKING, cast

import polars as pl

from svy.core.constants import (
    _INTERNAL_CONCAT_SUFFIX,
    SVY_ROW_INDEX,
)
from svy.core.design import Design


if TYPE_CHECKING:
    from svy.core.sample import Sample


# -------------------------------------------------------------------
# LazyFrame materialisation helper
# -------------------------------------------------------------------


def _eager_df(sample: "Sample") -> pl.DataFrame:
    """Return sample._data as an eager DataFrame, collecting if LazyFrame."""
    data = sample._data
    return data if isinstance(data, pl.DataFrame) else cast(pl.DataFrame, data.collect())


# -------------------------------------------------------------------
# Copy-on-write primitives
# -------------------------------------------------------------------


def _fork(sample: "Sample", new_data: pl.DataFrame) -> "Sample":
    """
    Create a new Sample that shares the original's Design (immutable)
    but owns independent copies of all mutable state.

    Parameters
    ----------
    sample : Sample
        The original sample to fork from.
    new_data : pl.DataFrame
        The (possibly transformed) data for the new sample.

    Returns
    -------
    Sample
        A new Sample with its own ``_data``, ``_metadata``,
        ``_internal_design``, and ``_warnings``.
    """
    new = sample._replace_data(new_data)
    # _replace_data does copy.copy (shallow).  Design is frozen/immutable
    # so sharing it is fine, but mutable stores must be isolated.
    new._metadata = copy.deepcopy(sample._metadata)
    new._internal_design = copy.deepcopy(sample._internal_design)
    new._warnings = copy.deepcopy(sample._warnings)
    return new


def _resolve_target(sample: "Sample", new_data: pl.DataFrame, *, inplace: bool) -> "Sample":
    """
    Return either the original sample (mutated) or a fresh fork.

    Parameters
    ----------
    sample : Sample
        The original sample.
    new_data : pl.DataFrame
        Transformed data.
    inplace : bool
        If True, mutate the original sample's ``_data`` and return it.
        If False, return a new sample via :func:`_fork`.
    """
    if inplace:
        sample._data = new_data
        return sample
    return _fork(sample, new_data)


# -------------------------------------------------------------------
# Design-source column introspection
# -------------------------------------------------------------------


def _design_source_columns(sample: "Sample") -> set[str]:
    """
    Return the set of user-facing column names that the internal
    concatenated design columns (stratum/psu/ssu) are derived from.

    These are the columns whose *values* feed into the concatenated
    ``_svy_internal_`` columns.  If any of these values change, the
    concatenated columns must be rebuilt.
    """
    design: Design | None = getattr(sample, "_design", None)
    if design is None:
        return set()

    src: set[str] = set()

    def _add(field: str | tuple[str, ...] | None) -> None:
        if field is None:
            return
        if isinstance(field, str):
            src.add(field)
        elif isinstance(field, tuple):
            src.update(field)

    _add(design.stratum)
    _add(design.psu)
    _add(design.ssu)
    return src


def _internal_columns(sample: "Sample") -> set[str]:
    """
    Return set of internal svy columns that should be auto-preserved
    transparently.

    These columns are managed by svy internally and users should not
    need to think about them when using keep_columns/select or
    remove_columns/drop.
    """
    internal: set[str] = set()
    cols = set(sample._data.columns)

    if SVY_ROW_INDEX in cols:
        internal.add(SVY_ROW_INDEX)

    for c in cols:
        if "_svy_internal_" in c:
            internal.add(c)

    return internal


def _required_columns(sample: "Sample") -> set[str]:
    """
    Return set of user-specified design column names.

    This excludes internal svy columns (like svy_row_index) which are
    handled transparently and auto-preserved.
    """
    req: set[str] = set()
    design: Design | None = getattr(sample, "_design", None)
    internal_design = getattr(sample, "_internal_design", {}) or {}

    def add(x: str | tuple[str, ...] | None):
        if x is None:
            return
        if isinstance(x, str):
            req.add(x)
        elif isinstance(x, tuple):
            req.update(x)

    if design is not None:
        add(design.stratum)
        add(design.psu)
        add(design.ssu)
        add(design.wgt)
        add(design.prob)
        add(design.hit)
        add(design.mos)
        add(design.pop_size)
        if design.rep_wgts and design.rep_wgts.wgts:
            req.update(design.rep_wgts.wgts)

    for k in ("stratum", "psu", "ssu"):
        cname = internal_design.get(k)
        if isinstance(cname, str):
            req.add(cname)

    cols = set(sample._data.columns)
    internal = _internal_columns(sample)
    return {c for c in req if c in cols and c not in internal}


# -------------------------------------------------------------------
# Concatenated design-column rebuilding
# -------------------------------------------------------------------


def _rebuild_concat_columns(target: "Sample") -> None:
    """
    Rebuild the internal concatenated design columns on *target*.

    Called after any operation that may have changed the **values**
    of a design source column (stratum / psu / ssu) or renamed
    design source columns.

    The method drops stale concatenated columns, re-creates them via
    ``_create_concatenated_cols_from_lists``, and updates
    ``_internal_design`` on the target.
    """
    design: Design | None = getattr(target, "_design", None)
    if design is None:
        return

    idesign = target._internal_design or {}

    # Collect stale concat column names and drop them
    stale = [
        idesign[k]
        for k in ("stratum", "psu", "ssu")
        if idesign.get(k) and idesign[k] in target._data.columns
    ]
    if stale:
        target._data = target._data.drop(stale)

    # Re-create from current design + data
    new_data, (_, stratum_cols, psu_cols, ssu_cols) = target._create_concatenated_cols_from_lists(
        data=target._data,
        design=design,
        by=None,
        null_token="__Null__",
        suffix=_INTERNAL_CONCAT_SUFFIX,
    )
    target._data = new_data
    target._internal_design = {
        "stratum": f"stratum{_INTERNAL_CONCAT_SUFFIX}" if stratum_cols else None,
        "psu": f"psu{_INTERNAL_CONCAT_SUFFIX}" if psu_cols else None,
        "ssu": f"ssu{_INTERNAL_CONCAT_SUFFIX}" if ssu_cols else None,
        "suffix": _INTERNAL_CONCAT_SUFFIX,
    }


def _rebuild_concat_if_touched(
    sample: "Sample", target: "Sample", touched_columns: set[str]
) -> None:
    """
    Rebuild concatenated design columns only when necessary.

    Parameters
    ----------
    sample : Sample
        The original sample (used to look up design source columns).
    target : Sample
        The sample whose concat columns may need rebuilding.
    touched_columns : set[str]
        Column names whose *values* were modified by the operation.
    """
    if not touched_columns:
        return
    design_sources = _design_source_columns(sample)
    if touched_columns & design_sources:
        _rebuild_concat_columns(target)


# -------------------------------------------------------------------
# Design auto-cleaning (after column removal)
# -------------------------------------------------------------------


def _auto_clean_design(target: "Sample") -> None:
    """Remove references to columns that no longer exist in the data."""
    current_design: Design | None = getattr(target, "_design", None)
    if current_design is None:
        return

    cols = set(target._data.columns)

    def keep_name(x: str | None) -> str | None:
        return x if (x is None or x in cols) else None

    def keep_tuple(
        x: str | tuple[str, ...] | None,
    ) -> str | tuple[str, ...] | None:
        if x is None:
            return None
        if isinstance(x, str):
            return x if x in cols else None
        kept = tuple(c for c in x if c in cols)
        return kept or None

    updated_design = current_design.update(
        row_index=keep_name(current_design.row_index),
        stratum=keep_tuple(current_design.stratum),
        psu=keep_tuple(current_design.psu),
        ssu=keep_tuple(current_design.ssu),
        wgt=keep_name(current_design.wgt),
        prob=keep_name(current_design.prob),
        hit=keep_name(current_design.hit),
        mos=keep_name(current_design.mos),
    )

    if current_design.rep_wgts:
        new_wgts = tuple(c for c in current_design.rep_wgts.wgts if c in cols)
        new_rep = (
            current_design.rep_wgts.clone(wgts=new_wgts, n_reps=len(new_wgts))
            if new_wgts
            else None
        )
        updated_design = updated_design.update(rep_wgts=new_rep)

    internal_design = dict(getattr(target, "_internal_design", {}) or {})
    for k in ("stratum", "psu", "ssu"):
        cname = internal_design.get(k)
        if isinstance(cname, str) and cname not in cols:
            internal_design[k] = None

    target._design = updated_design
    target._internal_design = internal_design
