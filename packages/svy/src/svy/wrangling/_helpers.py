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

from msgspec.structs import replace as _struct_replace

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
        rec = d.wgt_adjustment
        if rec is not None:
            put(rec.prev_wgt, f"the weight {rec.new_wgt!r} was adjusted from")
            for c in rec.cells or ():
                put(c, f"a cell snapshot the {rec.kind} of {rec.new_wgt!r} reads")
            for c in rec.aux or ():
                put(c, f"an auxiliary column the {rec.kind} of {rec.new_wgt!r} reads")
        rw = d.rep_wgts
        if rw is not None:
            why = "a replicate weight" + ("" if rw.wgt is None else f" of {rw.wgt!r}")
            for c in rw.columns_from_data(data_columns):
                put(c, why)
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

    def keep_name(field: str, x: str | None) -> str | None:
        if x is None or x in cols:
            return x
        removed.append(f"{field}={x!r}")
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
            removed.append(f"{field}={tuple(c for c in x if c not in cols)!r}")
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

    new_wgt = keep_name("wgt", current_design.wgt)

    rep = current_design.rep_wgts
    if rep is not None:
        # RepWeights identifies its columns by prefix + n_reps.  If any of
        # the expected replicate columns was removed, the replicate design
        # is no longer valid and must be dropped as a whole (a partial set
        # of replicates cannot be represented and would give wrong variances).
        expected = rep.columns_from_data(sorted(cols))
        if not all(c in cols for c in expected):
            removed.append(f"rep_wgts ('{rep.prefix}', {rep.n_reps} replicates)")
            rep = None
        else:
            # The recorded units are ordinary column references and are dropped
            # the same way the design's are. Losing them costs provenance and,
            # for a declared JKn with no rep_coefs yet, the ability to derive --
            # which surfaces as the usual warning rather than silently reading a
            # column that is no longer there.
            rep_updates: dict[str, str | tuple[str, ...] | None] = {}
            for field in ("stratum", "psu"):
                cur = getattr(rep, field)
                if cur is None:
                    continue
                if isinstance(cur, str):
                    if cur not in cols:
                        rep_updates[field] = None
                        removed.append(f"rep_wgts.{field}={cur!r}")
                else:
                    # Keep whatever survives: a multi-column unit that loses one
                    # member is a coarser unit, not a missing one. Only an empty
                    # remainder clears the field.
                    kept = tuple(c for c in cur if c in cols)
                    if kept != cur:
                        rep_updates[field] = kept or None
                        removed.append(
                            f"rep_wgts.{field}={tuple(c for c in cur if c not in cols)!r}"
                        )
            if rep_updates:
                rep = _struct_replace(rep, **rep_updates)
            # Replicates go with their full-sample weight: removing it removes
            # them from the design (their columns stay). Left unpaired they
            # would attach silently to whatever weight is set next.
            if rep.wgt is not None and rep.wgt not in cols:
                removed.append(f"rep_wgts ('{rep.prefix}', {rep.n_reps} replicates)")
                rep = None

    rec = current_design.wgt_adjustment
    if rec is not None:
        rec_cols = (rec.new_wgt, rec.prev_wgt, *(rec.cells or ()), *(rec.aux or ()))
        if any(c not in cols for c in rec_cols):
            removed.append(f"wgt_adjustment ({rec.kind})")
            rec = None

    updated_design = Design(
        case_id=keep_name("case_id", current_design.case_id),
        wave=keep_name("wave", current_design.wave),
        stratum=keep_tuple("stratum", current_design.stratum),
        wgt=new_wgt,
        prob=keep_name("prob", current_design.prob),
        hit=keep_name("hit", current_design.hit),
        mos=keep_name("mos", current_design.mos),
        psu=keep_tuple("psu", current_design.psu),
        ssu=keep_tuple("ssu", current_design.ssu),
        pop_size=pop_size,
        wr=current_design.wr,
        rep_wgts=rep,
        wgt_adjustment=rec,
    )

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
