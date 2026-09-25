# src/svy/core/design_parts.py
"""Design parts: what a ``Design`` carries besides its column fields.

A part is a decision stored on the design that svy cannot recompute from the
data: the replicate weights, the weight-adjustment record, the singleton
handling. Each part is one object implementing the hooks below, registered in
order; ``Design``, ``Sample``, wrangling and ``svy.serialize`` loop over the
registry and never name a part, so a new part is one class and one
``register`` call.

Hooks (all receive the part's value; the defaults do nothing):

``check(value, fields)``
    ``Design(...)``: validate a value (``None`` included) against the design's
    column fields; return it, possibly normalized.
``after_update(old, value, passed, fields)``
    ``design.update``: the value on the new design. ``passed`` says whether the
    caller named it; if not, the part's rule decides (kept, dropped).
``columns(value, design, data_columns)``
    The data columns it reads (``design.columns()``, the ``Sample`` check,
    wrangling protection).
``renamed(value, renames, data_columns)``
    The value after a column rename.
``removed(value, old, fields, present)``
    After a forced column removal: ``(value, [labels of what was dropped])``.
``lineage(value, data_columns)``
    ``(column, why)`` pairs wrangling must not overwrite.
``has_lineage(value)``
    Whether the design depends on the rows staying the same.
``missing_note(value, missing)``
    An explanation appended to the missing-columns error.
``check_data(value, data)``
    Validate against the frame at ``Sample(...)`` and on every refresh.
``derive(sample)``
    Rebuild the state the sample derives from design + data. Called whenever
    the data or design changed since the last call (see ``Sample._sync_parts``).
``to_data(value)`` / ``from_data(data)``
    The saved form (``svy.serialize``).
``to_code(value)``
    Runnable ``svy.``-prefixed source.
``repr(value)``
    Its piece of ``repr(design)``, or None.

``follows_weight`` parts go with one weight column: ``update_design(wgt=X)``
restores them from the design in the history that produced X.
"""

from __future__ import annotations

import warnings

from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

import msgspec


if TYPE_CHECKING:
    import polars as pl

    from svy.core.design import Design, SingletonSpec
    from svy.core.sample import Sample


class DesignPart:
    """One optional part of a ``Design``. Subclasses override what they need."""

    name: str = ""
    follows_weight: bool = False
    #: Its saved struct, for parts without a field of their own on DesignData.
    data_type: Any = None

    def check(self, value: Any, fields: Mapping[str, Any]) -> Any:
        return value

    def after_update(
        self, old: Design, value: Any, passed: bool, fields: Mapping[str, Any]
    ) -> Any:
        return value if passed else old._parts.get(self.name)

    def columns(
        self, value: Any, design: Design, data_columns: Sequence[str] | None
    ) -> Iterable[str]:
        return ()

    def renamed(
        self, value: Any, renames: Mapping[str, str], data_columns: Sequence[str] | None
    ) -> Any:
        return value

    def removed(
        self, value: Any, old: Design, fields: Mapping[str, Any], present: set[str]
    ) -> tuple[Any, list[str]]:
        return value, []

    def lineage(self, value: Any, data_columns: Sequence[str]) -> Iterable[tuple[str, str]]:
        return ()

    def has_lineage(self, value: Any) -> bool:
        return False

    def missing_note(self, value: Any, missing: set[str]) -> str | None:
        return None

    def check_data(self, value: Any, data: pl.DataFrame) -> None:
        return None

    def derive(self, sample: Sample) -> None:
        return None

    def to_data(self, value: Any) -> Any:
        return value

    def from_data(self, data: Any) -> Any:
        return data

    def to_code(self, value: Any) -> str:
        return value._to_code()

    def repr(self, value: Any) -> str | None:
        return None if value is None else f"{self.name}={value!r}"


_REGISTRY: dict[str, DesignPart] = {}


def register(part: DesignPart) -> DesignPart:
    """Add a part; it is carried by every ``Design`` created afterwards."""
    from svy.core.design import _FIELDS, Design

    if not part.name or part.name in _REGISTRY or part.name in _FIELDS:
        raise ValueError(f"Design part name {part.name!r} is empty or taken.")
    if hasattr(Design, part.name):
        raise ValueError(f"Design part name {part.name!r} shadows a Design attribute.")
    _REGISTRY[part.name] = part
    return part


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def registered() -> tuple[DesignPart, ...]:
    return tuple(_REGISTRY.values())


def check_part_names(given: Mapping[str, Any], *, where: str) -> None:
    unknown = [k for k in given if k not in _REGISTRY]
    if unknown:
        raise TypeError(f"{where} got unexpected keyword argument(s): {', '.join(unknown)}")


def _unit_columns(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _as_list(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def _as_tuple(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v


# ---------------------------------------------------------------------------
# Replicate weights
# ---------------------------------------------------------------------------


class RepWgtsPart(DesignPart):
    name = "rep_wgts"
    follows_weight = True

    def check(self, value: Any, fields: Mapping[str, Any]) -> Any:
        from svy.core.repwgts import _RepWgtsBase

        if value is None:
            return None
        # Every variant inherits the base, so this covers the whole union.
        if not isinstance(value, _RepWgtsBase):
            raise TypeError("'rep_wgts' must be RepWgts | None")
        # Replicates describe one weight column; a design pointing elsewhere
        # would estimate with one and vary with the other.
        wgt = fields["wgt"]
        if value.wgt is None:
            return value if wgt is None else msgspec.structs.replace(value, wgt=wgt)
        if value.wgt != wgt:
            raise ValueError(
                f"rep_wgts go with weight {value.wgt!r} but the design's weight is "
                f"{wgt!r}. Pass replicate weights built from {wgt!r}, or leave "
                "rep_wgts.wgt unset to pair them with the design's weight."
            )
        return value

    def after_update(
        self, old: Design, value: Any, passed: bool, fields: Mapping[str, Any]
    ) -> Any:
        from svy.core.design import _warn_rep_wgts_reset

        # A sequence of column names is the legacy spelling, read as not passed.
        if passed and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            passed = False
        new_wgt = fields["wgt"]
        if not passed:
            # Kept while they describe the new weight, dropped otherwise.
            rep = old.rep_wgts
            if rep is not None and rep.wgt is not None and rep.wgt != new_wgt:
                _warn_rep_wgts_reset(rep, new_wgt)
                return None
            return rep
        # Passed replicates are paired with the new weight: the caller asserts
        # they match.
        if value is not None and value.wgt != new_wgt:
            return msgspec.structs.replace(value, wgt=new_wgt)
        return value

    def columns(
        self, value: Any, design: Design, data_columns: Sequence[str] | None
    ) -> Iterable[str]:
        cols = value.columns_from_data(data_columns) if data_columns is not None else value.columns
        return (*cols, *_unit_columns(value.stratum), *_unit_columns(value.psu))

    def renamed(
        self, value: Any, renames: Mapping[str, str], data_columns: Sequence[str] | None
    ) -> Any:
        return _rep_wgts_with_renames(value, dict(renames), data_columns)

    def removed(
        self, value: Any, old: Design, fields: Mapping[str, Any], present: set[str]
    ) -> tuple[Any, list[str]]:
        rep = value
        removed: list[str] = []
        # A partial set of replicates cannot be represented and would give wrong
        # variances, so a replicate set missing any column goes as a whole.
        expected = rep.columns_from_data(sorted(present))
        if not all(c in present for c in expected):
            return None, [f"rep_wgts ('{rep.prefix}', {rep.n_reps} replicates)"]
        # The recorded units are ordinary column references, dropped as the
        # design's are. A multi-column unit that loses one member is a coarser
        # unit, not a missing one; only an empty remainder clears the field.
        updates: dict[str, Any] = {}
        for field in ("stratum", "psu"):
            cur = getattr(rep, field)
            if cur is None:
                continue
            if isinstance(cur, str):
                if cur not in present:
                    updates[field] = None
                    removed.append(f"rep_wgts.{field}={cur!r}")
            else:
                kept = tuple(c for c in cur if c in present)
                if kept != cur:
                    updates[field] = kept or None
                    removed.append(
                        f"rep_wgts.{field}={tuple(c for c in cur if c not in present)!r}"
                    )
        if updates:
            rep = msgspec.structs.replace(rep, **updates)
        # Replicates go with their full-sample weight: removing it removes them
        # from the design (their columns stay). Left unpaired they would attach
        # silently to whatever weight is set next.
        if rep.wgt is not None and rep.wgt not in present:
            removed.append(f"rep_wgts ('{rep.prefix}', {rep.n_reps} replicates)")
            rep = None
        return rep, removed

    def lineage(self, value: Any, data_columns: Sequence[str]) -> Iterable[tuple[str, str]]:
        why = "a replicate weight" + ("" if value.wgt is None else f" of {value.wgt!r}")
        return ((c, why) for c in value.columns_from_data(data_columns))

    def has_lineage(self, value: Any) -> bool:
        return True

    def missing_note(self, value: Any, missing: set[str]) -> str | None:
        if any(c in missing for c in (*_unit_columns(value.stratum), *_unit_columns(value.psu))):
            return (
                "rep_wgts.stratum/psu name the units the replicates were built "
                "from, which is a separate question from the Design's stratum/psu."
            )
        return None

    def check_data(self, value: Any, data: pl.DataFrame) -> None:
        from svy.core.sample import FLOAT_DTYPES, INTEGER_DTYPES

        expected = value.columns_from_data(data.columns)
        missing = [c for c in expected if c not in data.columns]
        if missing:
            raise ValueError(
                f"Expected replicate weight columns not found in data: {missing[:10]}"
                + ("..." if len(missing) > 10 else "")
            )
        if value.n_reps > 0 and value.n_reps != len(expected):
            raise ValueError(
                f"RepWeights.n_reps ({value.n_reps}) does not match number of columns found ({len(expected)})."
            )
        schema = data.schema
        non_numeric = [
            c
            for c in expected
            if schema[c] not in INTEGER_DTYPES and schema[c] not in FLOAT_DTYPES
        ]
        if non_numeric:
            suffix = "..." if len(non_numeric) > 3 else ""
            raise TypeError(
                f"Replicate weight columns must be numeric; got non-numeric types for: {non_numeric[:3]}{suffix}"
            )

    def to_data(self, value: Any) -> Any:
        from svy.serialize.structs import (
            BootstrapWgtsData,
            BrrWgtsData,
            JackknifeWgtsData,
            SdrWgtsData,
        )

        cls = {
            "Bootstrap": BootstrapWgtsData,
            "Jackknife": JackknifeWgtsData,
            "BRR": BrrWgtsData,
            "SDR": SdrWgtsData,
        }[value.method]
        out = {f: _as_list(getattr(value, f)) for f in cls.__struct_fields__}
        for f in ("scale", "rep_coefs"):
            if out[f] is not None:
                out[f] = [float(x) for x in out[f]]
        if out["df"] is not None:
            out["df"] = float(out["df"])
        return cls(**out)

    def from_data(self, data: Any) -> Any:
        from svy.core.repwgts import resolve_rep_variant

        variant = resolve_rep_variant(type(data).__struct_config__.tag)
        return variant(**{f: _as_tuple(getattr(data, f)) for f in type(data).__struct_fields__})

    def repr(self, value: Any) -> str | None:
        if value is None:
            return "rep_wgts=None"
        return f"rep_wgts={value.method}(n_reps={value.n_reps}, prefix='{value.prefix}', df={value.df})"


def _rep_wgts_with_renames(
    rep_wgts: Any,
    renames: dict[str, str],
    data_columns: Sequence[str] | None = None,
) -> Any:
    """Return RepWeights updated for column renames.

    The replicate columns are the spec's own list, resolved against
    ``data_columns`` (the frame before the rename), so only those exact names
    count: a column that merely looks like one (``w2023``, ``W1``, ``w01`` next
    to unpadded ``w1``) is an ordinary column. A rename of the replicates can
    only be represented when every one is renamed to a common new prefix,
    keeping its number and padding. Otherwise raise.

    The recorded units (``stratum``/``psu``) are plain column references and are
    remapped independently of the prefix.
    """
    unit_updates: dict[str, str | tuple[str, ...]] = {}
    for field in ("stratum", "psu"):
        cur = getattr(rep_wgts, field)
        if cur is None:
            continue
        if isinstance(cur, str):
            if cur in renames:
                unit_updates[field] = renames[cur]
        else:
            # A multi-column unit is remapped element-wise; a rename touching
            # only some of its columns still has to move those.
            mapped = tuple(renames.get(c, c) for c in cur)
            if mapped != cur:
                unit_updates[field] = mapped

    rep_cols = (
        rep_wgts.columns_from_data(data_columns) if data_columns is not None else rep_wgts.columns
    )
    matched = [c for c in rep_cols if c in renames]
    if not matched:
        return msgspec.structs.replace(rep_wgts, **unit_updates) if unit_updates else rep_wgts
    new_prefixes: set[str] = set()
    for old in matched:
        new = renames[old]
        suffix = old[len(rep_wgts.prefix) :]
        if not new.endswith(suffix) or len(new) == len(suffix):
            raise ValueError(
                f"Cannot rename replicate weight column {old!r} to {new!r}: "
                f"the replicate number suffix {suffix!r} must be preserved."
            )
        new_prefixes.add(new[: len(new) - len(suffix)])
    if len(new_prefixes) > 1:
        raise ValueError(
            "Replicate weight columns must all be renamed with the same prefix; "
            f"got prefixes {sorted(new_prefixes)}."
        )
    # A partial rename cannot be represented: the spec names its columns as
    # prefix + number, so the new prefix would claim columns never renamed.
    not_renamed = [c for c in rep_cols if c not in renames]
    if not_renamed:
        raise ValueError(
            f"Partial replicate-weight rename: {len(not_renamed)} of "
            f"{len(rep_cols)} replicate columns were not renamed "
            f"(e.g. {not_renamed[:3]}). Rename all replicate columns together "
            "with a common new prefix, keeping the numeric suffixes."
        )
    return msgspec.structs.replace(rep_wgts, prefix=new_prefixes.pop(), **unit_updates)


# ---------------------------------------------------------------------------
# Weight-adjustment record
# ---------------------------------------------------------------------------


def _record_columns(rec: Any) -> tuple[str, ...]:
    return (rec.new_wgt, rec.prev_wgt, *(rec.cells or ()), *(rec.aux or ()))


class WgtAdjustmentPart(DesignPart):
    name = "wgt_adjustment"
    follows_weight = True

    def check(self, value: Any, fields: Mapping[str, Any]) -> Any:
        from svy.core.design import WgtAdjustment

        if value is None:
            return None
        if not isinstance(value, WgtAdjustment):
            raise TypeError("'wgt_adjustment' must be WgtAdjustment | None")
        if value.new_wgt != fields["wgt"]:
            raise ValueError(
                f"wgt_adjustment describes weight {value.new_wgt!r} but the "
                f"design's weight is {fields['wgt']!r}."
            )
        return value

    def after_update(
        self, old: Design, value: Any, passed: bool, fields: Mapping[str, Any]
    ) -> Any:
        if passed:
            return value
        # Kept while it describes the new weight; a record cannot be invented
        # for a column whose origin svy never saw.
        rec = old.wgt_adjustment
        return rec if rec is not None and rec.new_wgt == fields["wgt"] else None

    def columns(
        self, value: Any, design: Design, data_columns: Sequence[str] | None
    ) -> Iterable[str]:
        return _record_columns(value)

    def renamed(
        self, value: Any, renames: Mapping[str, str], data_columns: Sequence[str] | None
    ) -> Any:
        def names(x: tuple[str, ...] | None) -> tuple[str, ...] | None:
            return None if x is None else tuple(renames.get(c, c) for c in x)

        return msgspec.structs.replace(
            value,
            prev_wgt=renames.get(value.prev_wgt, value.prev_wgt),
            new_wgt=renames.get(value.new_wgt, value.new_wgt),
            cells=names(value.cells),
            aux=names(value.aux),
        )

    def removed(
        self, value: Any, old: Design, fields: Mapping[str, Any], present: set[str]
    ) -> tuple[Any, list[str]]:
        if any(c not in present for c in _record_columns(value)):
            return None, [f"wgt_adjustment ({value.kind})"]
        return value, []

    def lineage(self, value: Any, data_columns: Sequence[str]) -> Iterable[tuple[str, str]]:
        yield value.prev_wgt, f"the weight {value.new_wgt!r} was adjusted from"
        for c in value.cells or ():
            yield c, f"a cell snapshot the {value.kind} of {value.new_wgt!r} reads"
        for c in value.aux or ():
            yield c, f"an auxiliary column the {value.kind} of {value.new_wgt!r} reads"

    def has_lineage(self, value: Any) -> bool:
        return True

    def missing_note(self, value: Any, missing: set[str]) -> str | None:
        if any(c in missing for c in _record_columns(value)):
            return (
                f"wgt_adjustment ({value.kind}) reads its columns from the data; "
                "they are written by the weighting method and must travel with it."
            )
        return None

    def to_data(self, value: Any) -> Any:
        from svy.serialize.structs import WgtAdjustmentData

        return WgtAdjustmentData(
            kind=value.kind,
            prev_wgt=value.prev_wgt,
            new_wgt=value.new_wgt,
            cells=_as_list(value.cells),
            aux=_as_list(value.aux),
            pins_total=value.pins_total,
        )

    def from_data(self, data: Any) -> Any:
        from svy.core.design import WgtAdjustment

        return WgtAdjustment(
            kind=data.kind,
            prev_wgt=data.prev_wgt,
            new_wgt=data.new_wgt,
            cells=_as_tuple(data.cells),
            aux=_as_tuple(data.aux),
            pins_total=data.pins_total,
        )

    def repr(self, value: Any) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Singleton handling
# ---------------------------------------------------------------------------

#: Warnings from clearing a spec inside ``Sample.update_design`` wait here until
#: the new design is installed, so they can name the singletons of the data.
_PENDING: ContextVar[list[tuple[Any, str]] | None] = ContextVar("svy_pending_parts", default=None)


@contextmanager
def deferred_clear_warnings() -> Iterator[list[tuple[Any, str]]]:
    pending: list[tuple[Any, str]] = []
    token = _PENDING.set(pending)
    try:
        yield pending
    finally:
        _PENDING.reset(token)


def _variance_psu(fields: Mapping[str, Any]) -> Any:
    if fields.get("psu") is not None:
        return fields["psu"]
    return fields.get("case_id") if fields.get("wave") is not None else None


def singleton_cleared_message(spec: SingletonSpec, reason: str, now: Sequence[Any] | None) -> str:
    msg = f"singleton handling ({spec.method}) cleared: {reason}"
    if now is not None:
        msg += "; singletons now: " + (", ".join(repr(v) for v in now) if now else "none")
    return msg + ". Apply a singleton rule again (sample.singleton.*) if needed."


def warn_singleton_cleared(spec: SingletonSpec, reason: str, now: Sequence[Any] | None) -> None:
    from svy.core.design import _user_stacklevel

    warnings.warn(
        singleton_cleared_message(spec, reason, now), UserWarning, stacklevel=_user_stacklevel()
    )


class SingletonPart(DesignPart):
    name = "singleton"

    def check(self, value: Any, fields: Mapping[str, Any]) -> Any:
        from svy.core.design import SingletonSpec

        if value is None:
            return None
        if not isinstance(value, SingletonSpec):
            raise TypeError("'singleton' must be SingletonSpec | None")
        if fields["stratum"] is None:
            raise ValueError("singleton handling applies to strata; the design has no stratum.")
        return value

    def after_update(
        self, old: Design, value: Any, passed: bool, fields: Mapping[str, Any]
    ) -> Any:
        if passed:
            return value
        spec = old.singleton
        if spec is None:
            return None
        # Its strata no longer mean anything once the columns holding the
        # strata or PSUs change.
        before = {f: getattr(old, f) for f in ("stratum", "psu", "ssu")}
        if any(fields[f] != v for f, v in before.items()) or _variance_psu(fields) != (
            old.variance_psu
        ):
            reason = "the stratum/psu/ssu columns changed"
            pending = _PENDING.get()
            if pending is not None:
                pending.append((spec, reason))
            else:
                warn_singleton_cleared(spec, reason, None)
            return None
        return spec

    def columns(
        self, value: Any, design: Design, data_columns: Sequence[str] | None
    ) -> Iterable[str]:
        return (
            *_unit_columns(design.stratum),
            *_unit_columns(design.variance_psu),
            *_unit_columns(design.ssu),
        )

    def removed(
        self, value: Any, old: Design, fields: Mapping[str, Any], present: set[str]
    ) -> tuple[Any, list[str]]:
        if (
            any(fields[f] != getattr(old, f) for f in ("stratum", "psu", "ssu"))
            or fields["stratum"] is None
        ):
            return None, [f"singleton handling ({value.method})"]
        return value, []

    def derive(self, sample: Sample) -> None:
        from svy.core.singleton import _derive_singleton_state

        _derive_singleton_state(sample)

    def to_data(self, value: Any) -> Any:
        from svy.serialize.structs import SingletonSpecData

        return SingletonSpecData(
            method=value.method,
            strata=[_saved_value(v) for v in value.strata],
            mapping=[(_saved_value(a), _saved_value(b)) for a, b in value.mapping],
            name=value.name,
        )

    def from_data(self, data: Any) -> Any:
        from svy.core.design import SingletonSpec

        return SingletonSpec(
            method=data.method,
            strata=tuple(data.strata),
            mapping=tuple((a, b) for a, b in data.mapping),
            name=data.name,
        )


def _saved_value(value: Any) -> Any:
    """A stratum value in JSON-native form (dates as ISO strings)."""
    if isinstance(value, tuple):
        return [_saved_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else str(value)


# The built-in parts, registered directly: Design is not defined yet when this
# module is first imported (design.py imports it).
for _part in (RepWgtsPart(), WgtAdjustmentPart(), SingletonPart()):
    _REGISTRY[_part.name] = _part
