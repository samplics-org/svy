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

import warnings

from typing import TYPE_CHECKING, Iterable, cast

import polars as pl

from svy.core.constants import (
    _INTERNAL_CONCAT_SUFFIX,
    SVY_ROW_INDEX,
)
from svy.core.design import Design, PopSize, _user_stacklevel
from svy.errors import MethodError


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
    # _replace_data shares the frozen/immutable Design but deep-copies the
    # mutable stores (_metadata, _internal_design, _warnings) so they are
    # already isolated on the returned sample.
    return sample._replace_data(new_data)


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
        # Rebinding _data triggers Sample.__setattr__, which bumps the data
        # version and invalidates version-keyed caches. (The fork path is
        # versioned via _replace_data.)
        sample._data = new_data
        return sample
    return _fork(sample, new_data)


# -------------------------------------------------------------------
# Weight lineage: columns wrangling must not write into
# -------------------------------------------------------------------


def _weight_lineage_columns(sample: "Sample") -> dict[str, str]:
    """Weight columns the design or its history reads, with what reads them.

    The current and every earlier design's weight, replicate weights, and the
    weight-adjustment record's columns. Each value completes "'<col>' is ...";
    the current design's roles are listed first.
    """
    data_columns = sample._data.collect_schema().names()
    present = set(data_columns)
    roles: dict[str, str] = {}

    def put(col: str | None, why: str) -> None:
        if col is not None and col in present:
            roles.setdefault(col, why)

    history = getattr(sample, "_design_history", ())
    designs = [sample._design, *reversed(history)] if sample._design is not None else []
    for i, d in enumerate(designs):
        put(d.wgt, "the design's weight" if i == 0 else "the weight of an earlier design")
        for part, value in d._part_items():
            for col, why in part.lineage(value, data_columns):
                put(col, why)
    return roles


_SIGNED = (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
_UNSIGNED = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
# Integers a float holds exactly: 24-bit mantissa for Float32, 53 for Float64.
_EXACT_IN_FLOAT = {
    pl.Float32: (pl.Int8, pl.Int16, pl.UInt8, pl.UInt16),
    pl.Float64: (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32),
}


def _is_exact_widening(old: pl.DataType, new: pl.DataType) -> bool:
    """Whether every value of ``old`` is representable unchanged in ``new``."""
    if old == new:
        return True
    if old == pl.Float32 and new == pl.Float64:
        return True
    if new in _EXACT_IN_FLOAT:
        return any(old == t for t in _EXACT_IN_FLOAT[new])
    for family in (_SIGNED, _UNSIGNED):
        if old in family and new in family:
            return family.index(new) > family.index(old)
    if old in _UNSIGNED and new in _SIGNED:
        return _SIGNED.index(new) > _UNSIGNED.index(old)
    return False


def _unchanged(old: pl.Series, new: pl.Series) -> bool:
    if old.dtype == new.dtype and old.null_count() == new.null_count():
        try:
            # A column a step did not touch keeps its buffer.
            if old._get_buffer_info() == new._get_buffer_info():
                return True
        except Exception:
            pass
    if not _is_exact_widening(old.dtype, new.dtype):
        return False
    return old.cast(new.dtype).equals(new, check_names=False, null_equal=True)


def _guard_weight_writes(
    sample: "Sample",
    new_data: pl.DataFrame,
    *,
    where: str,
    targets: Iterable[str] = (),
    widening_only: bool = False,
) -> None:
    """Refuse a step that writes into a weight-lineage column.

    A weight overwritten under its own name is a new variable that the design,
    a later weight's record or the replicates would still read as the old one.
    ``targets`` are the existing columns the step writes by name: refused
    outright, except that a cast (``widening_only``) may widen exactly. Any
    other lineage column is refused if its values or type changed. Runs before
    any data is rebound, so a refused step leaves the sample untouched.
    """
    roles = _weight_lineage_columns(sample)
    if not roles:
        return
    old = _eager_df(sample)
    new = new_data if isinstance(new_data, pl.DataFrame) else new_data.collect()
    written = set(targets)

    def allowed(c: str) -> bool:
        if c not in new.columns:
            return True
        if c in written:
            return widening_only and _is_exact_widening(old.schema[c], new.schema[c])
        return _unchanged(old.get_column(c), new.get_column(c))

    bad = {c: why for c, why in roles.items() if not allowed(c)}
    if bad:
        raise MethodError.weight_overwrite(where=where, columns=bad)


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
    _add(design.variance_psu)
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

    Everything ``design.columns()`` names -- design fields, replicate weights
    and their units, and the weight-adjustment record's columns (including its
    ``__svy_cells_*``/``__svy_aux_*`` snapshots) -- plus the concatenated
    design columns. Internal svy columns (like svy_row_index) are handled
    transparently and auto-preserved, so they are excluded.
    """
    req: set[str] = set()
    design: Design | None = getattr(sample, "_design", None)
    internal_design = getattr(sample, "_internal_design", {}) or {}

    if design is not None:
        req.update(design.columns(data_columns=sample._data.columns))

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
    # The two ``target._data = ...`` rebinds above each trip Sample.__setattr__,
    # which bumps the data version — no explicit invalidation needed here.


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
    """Remove references to columns that no longer exist in the data.

    Everything that depended on a removed column goes with it: a design field
    is unset, a replicate set missing any column is dropped, a record missing
    any column is dropped, and replicates whose weight was removed lose that
    pairing. One warning lists what was removed from the design.
    """
    current_design: Design | None = getattr(target, "_design", None)
    if current_design is None:
        return

    cols = set(target._data.columns)
    removed: list[str] = []
    field_notes: dict[str, str] = {}

    def keep_name(field: str, x: str | None) -> str | None:
        if x is None or x in cols:
            return x
        field_notes[field] = f"{field}={x!r}"
        return None

    def keep_tuple(
        field: str,
        x: str | tuple[str, ...] | None,
    ) -> str | tuple[str, ...] | None:
        if x is None:
            return None
        if isinstance(x, str):
            return keep_name(field, x)
        kept = tuple(c for c in x if c in cols)
        if kept != x:
            field_notes[field] = f"{field}={tuple(c for c in x if c not in cols)!r}"
        return kept or None

    pop_size = current_design.pop_size
    if isinstance(pop_size, PopSize):
        if pop_size.psu not in cols:
            removed.append(f"pop_size={pop_size!r}")
            pop_size = None
        elif pop_size.ssu is not None and pop_size.ssu not in cols:
            removed.append(f"pop_size.ssu={pop_size.ssu!r}")
            pop_size = PopSize(psu=pop_size.psu, ssu=None)
    else:
        pop_size = keep_name("pop_size", pop_size)
        if "pop_size" in field_notes:
            removed.append(field_notes.pop("pop_size"))

    fields = {
        "case_id": keep_name("case_id", current_design.case_id),
        "wave": keep_name("wave", current_design.wave),
        "stratum": keep_tuple("stratum", current_design.stratum),
        "wgt": keep_name("wgt", current_design.wgt),
        "prob": keep_name("prob", current_design.prob),
        "hit": keep_name("hit", current_design.hit),
        "mos": keep_name("mos", current_design.mos),
        "psu": keep_tuple("psu", current_design.psu),
        "ssu": keep_tuple("ssu", current_design.ssu),
        "pop_size": pop_size,
        "wr": current_design.wr,
    }
    if "wgt" in field_notes:
        removed.append(field_notes.pop("wgt"))
    # Each part says what it keeps without the removed columns (replicates
    # missing a column or their weight go, a record missing a column goes,
    # singleton handling goes with its strata/PSU columns).
    parts: dict[str, object] = {}
    for part, value in current_design._part_items():
        parts[part.name], dropped = part.removed(value, current_design, fields, cols)
        removed.extend(dropped)
    removed.extend(field_notes.values())

    updated_design = Design(**fields, **parts)

    internal_design = dict(getattr(target, "_internal_design", {}) or {})
    for k in ("stratum", "psu", "ssu"):
        cname = internal_design.get(k)
        if isinstance(cname, str) and cname not in cols:
            internal_design[k] = None

    target._design = updated_design
    target._internal_design = internal_design
    if any(
        getattr(updated_design, f) != getattr(current_design, f)
        for f in ("stratum", "variance_psu", "ssu")
    ):
        # Stale concatenated columns and singleton flags would otherwise keep
        # describing the removed strata/PSUs.
        _rebuild_concat_columns(target)
        target._check_for_singletons()

    if removed:
        warnings.warn(
            "Removed from the design with the dropped columns: " + ", ".join(removed) + ".",
            UserWarning,
            stacklevel=_user_stacklevel(),
        )
