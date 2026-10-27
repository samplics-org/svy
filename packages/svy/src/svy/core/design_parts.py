# src/svy/core/design_parts.py
"""Design parts: what a ``Design`` carries besides its column fields.

A part is a decision stored on the design that svy cannot recompute from the
data: the replicate weights, the weight-adjustment record, the singleton
rule. Each part is one object implementing the hooks below, registered in
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

from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import msgspec
import numpy as np

from svy.errors import Code, DataError, DesignError, SingletonWarning


if TYPE_CHECKING:
    import polars as pl

    from svy.core.design import Design, Singleton
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
        raise DesignError.invalid_value(
            where="design_parts.register",
            param="name",
            got=part.name,
            detail=f"Design part name {part.name!r} is empty or taken.",
        )
    if hasattr(Design, part.name):
        raise DesignError.invalid_value(
            where="design_parts.register",
            param="name",
            got=part.name,
            detail=f"Design part name {part.name!r} shadows a Design attribute.",
        )
    _REGISTRY[part.name] = part
    return part


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def registered() -> tuple[DesignPart, ...]:
    return tuple(_REGISTRY.values())


def check_part_names(given: Mapping[str, Any], *, where: str, fields: Iterable[str] = ()) -> None:
    unknown = [k for k in given if k not in _REGISTRY]
    if unknown:
        from svy.errors._api import UsageError

        raise UsageError.unknown_arguments(
            where=where, unknown=unknown, known=[*fields, *_REGISTRY]
        )


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
            raise DesignError.invalid_type(
                where="Design",
                param="rep_wgts",
                got=value,
                expected="svy.BrrWgts, JackknifeWgts, BootstrapWgts, SdrWgts or None",
                detail="'rep_wgts' must be RepWgts | None",
            )
        # Replicates describe one weight column; a design pointing elsewhere
        # would estimate with one and vary with the other.
        wgt = fields["wgt"]
        if value.wgt is None:
            return value if wgt is None else msgspec.structs.replace(value, wgt=wgt)
        if value.wgt != wgt:
            raise DesignError.ValueError(
                title="Replicate weights for another weight",
                detail=f"rep_wgts go with weight {value.wgt!r} but the design's weight is {wgt!r}.",
                code=Code.REP_WGTS_WGT_MISMATCH,
                where="Design",
                param="rep_wgts",
                expected=wgt,
                got=value.wgt,
                hint=f"Pass replicate weights built from {wgt!r}, or leave "
                "rep_wgts.wgt unset to pair them with the design's weight.",
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
            raise DataError.ValueError(
                title="Column(s) not found",
                detail=f"Expected replicate weight columns not found in data: {missing[:10]}"
                + ("..." if len(missing) > 10 else ""),
                code=Code.MISSING_COLUMNS,
                where="Sample",
                param="rep_wgts",
                got=missing,
            )
        if value.n_reps > 0 and value.n_reps != len(expected):
            raise DesignError.ValueError(
                title="Replicate count does not match the columns",
                detail=f"RepWeights.n_reps ({value.n_reps}) does not match number of columns "
                f"found ({len(expected)}).",
                code=Code.REP_WGTS_COUNT_MISMATCH,
                where="Sample",
                param="n_reps",
                expected=len(expected),
                got=value.n_reps,
            )
        schema = data.schema
        non_numeric = [
            c
            for c in expected
            if schema[c] not in INTEGER_DTYPES and schema[c] not in FLOAT_DTYPES
        ]
        if non_numeric:
            suffix = "..." if len(non_numeric) > 3 else ""
            raise DataError.TypeError(
                title="Column is not numeric",
                detail="Replicate weight columns must be numeric; got non-numeric types for: "
                f"{non_numeric[:3]}{suffix}",
                code=Code.COLUMN_NOT_NUMERIC,
                where="Sample",
                param="rep_wgts",
                got=non_numeric,
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
        return f"rep_wgts={value!r}"


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
            raise DesignError.ValueError(
                title="Replicate-weight rename",
                detail=f"Cannot rename replicate weight column {old!r} to {new!r}: "
                f"the replicate number suffix {suffix!r} must be preserved.",
                code=Code.REP_WGTS_RENAME_INVALID,
                where="rename_columns",
                param="rep_wgts",
                hint="Rename all replicate columns together with a common new prefix, "
                "keeping the numeric suffixes.",
            )
        new_prefixes.add(new[: len(new) - len(suffix)])
    if len(new_prefixes) > 1:
        raise DesignError.ValueError(
            title="Replicate-weight rename",
            detail="Replicate weight columns must all be renamed with the same prefix; "
            f"got prefixes {sorted(new_prefixes)}.",
            code=Code.REP_WGTS_RENAME_INVALID,
            where="rename_columns",
            param="rep_wgts",
            hint="Rename all replicate columns together with a common new prefix, "
            "keeping the numeric suffixes.",
        )
    # A partial rename cannot be represented: the spec names its columns as
    # prefix + number, so the new prefix would claim columns never renamed.
    not_renamed = [c for c in rep_cols if c not in renames]
    if not_renamed:
        raise DesignError.ValueError(
            title="Replicate-weight rename",
            detail=f"Partial replicate-weight rename: {len(not_renamed)} of "
            f"{len(rep_cols)} replicate columns were not renamed (e.g. {not_renamed[:3]}).",
            code=Code.REP_WGTS_RENAME_INVALID,
            where="rename_columns",
            param="rep_wgts",
            hint="Rename all replicate columns together with a common new prefix, "
            "keeping the numeric suffixes.",
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
            raise DesignError.invalid_type(
                where="Design",
                param="wgt_adjustment",
                got=value,
                expected="WgtAdjustment | None",
                detail="'wgt_adjustment' must be WgtAdjustment | None",
            )
        if value.new_wgt != fields["wgt"]:
            raise DesignError.ValueError(
                title="Weight adjustment for another weight",
                detail=f"wgt_adjustment describes weight {value.new_wgt!r} but the "
                f"design's weight is {fields['wgt']!r}.",
                code=Code.WGT_ADJUSTMENT_MISMATCH,
                where="Design",
                param="wgt_adjustment",
                expected=fields["wgt"],
                got=value.new_wgt,
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
        return None if value is None else f"wgt_adjustment={value!r}"


# ---------------------------------------------------------------------------
# Singleton handling
# ---------------------------------------------------------------------------


def singleton_dropped_message(rule: Singleton, reason: str) -> str:
    return (
        f"singleton rule ({rule.method.value}) removed from the design: {reason}. "
        "Declare it again with sample.update_design(singleton=...) if needed."
    )


def warn_singleton_dropped(rule: Singleton, reason: str, *, sample: Any = None) -> None:
    from svy.errors._findings import emit_finding, findings_to

    with findings_to(sample) if sample is not None else nullcontext():
        emit_finding(
            SingletonWarning(
                code=Code.SINGLETON_RULE_CLEARED,
                title="Singleton rule removed",
                detail=singleton_dropped_message(rule, reason),
                where="Sample.update_design",
                param="singleton",
            )
        )


class SingletonPart(DesignPart):
    name = "singleton"

    def check(self, value: Any, fields: Mapping[str, Any]) -> Any:
        from svy.core.design import Singleton

        if value is None:
            return None
        if isinstance(value, str):
            value = Singleton(value)  # type: ignore[arg-type]
        if not isinstance(value, Singleton):
            raise DesignError.invalid_type(
                where="Design",
                param="singleton",
                got=value,
                expected="svy.Singleton | str | None",
                hint='e.g. singleton="center" or singleton=svy.Singleton("collapse", within="region").',
            )
        if fields["stratum"] is None:
            raise DesignError.ValueError(
                title="Singleton rule without strata",
                detail="singleton handling applies to strata; the design has no stratum.",
                code=Code.SINGLETON_WITHOUT_STRATUM,
                where="Design",
                param="singleton",
                hint="Declare stratum= with the singleton rule, or drop the rule.",
            )
        return value

    def after_update(
        self, old: Design, value: Any, passed: bool, fields: Mapping[str, Any]
    ) -> Any:
        if passed:
            return value
        # Declared intent: it applies to whatever strata the design has, so it
        # stays through stratum/PSU edits, unless there are no strata left.
        rule = old.singleton
        if rule is not None and fields["stratum"] is None:
            warn_singleton_dropped(rule, "the design has no stratum")
            return None
        return rule

    def columns(
        self, value: Any, design: Design, data_columns: Sequence[str] | None
    ) -> Iterable[str]:
        return (
            *_unit_columns(design.stratum),
            *_unit_columns(design.variance_psu),
            *_unit_columns(design.ssu),
            *(value.within or ()),
            *(value.order_by or ()),
        )

    def renamed(
        self, value: Any, renames: Mapping[str, str], data_columns: Sequence[str] | None
    ) -> Any:
        changes = {}
        for f in ("within", "order_by"):
            cols = getattr(value, f)
            if cols is not None and any(c in renames for c in cols):
                changes[f] = tuple(renames.get(c, c) for c in cols)
        return msgspec.structs.replace(value, **changes) if changes else value

    def removed(
        self, value: Any, old: Design, fields: Mapping[str, Any], present: set[str]
    ) -> tuple[Any, list[str]]:
        if fields["stratum"] is None:
            return None, [f"singleton rule ({value.method.value})"]
        gone = [c for c in (*(value.within or ()), *(value.order_by or ())) if c not in present]
        if gone:
            return None, [f"singleton rule ({value.method.value}; reads {', '.join(gone)})"]
        return value, []

    def missing_note(self, value: Any, missing: set[str]) -> str | None:
        cols = [c for c in (*(value.within or ()), *(value.order_by or ())) if c in missing]
        if cols:
            return f"the singleton rule's within/order_by read {', '.join(cols)}."
        return None

    def derive(self, sample: Sample) -> None:
        from svy.core.singleton import _derive_singleton_state

        _derive_singleton_state(sample)

    def to_data(self, value: Any) -> Any:
        from svy.serialize.structs import SingletonData

        if value.rstate is not None and not isinstance(value.rstate, int):
            from svy.errors import SerializationError

            raise SerializationError.ValueError(
                title="Cannot save the singleton rule",
                detail="a collapse rule with a Generator rstate cannot be saved.",
                code=Code.SINGLETON_RULE_NOT_SAVABLE,
                param="rstate",
                hint="Use an int seed, e.g. svy.Singleton('collapse', rstate=42).",
            )
        using: Any = msgspec.UNSET
        if isinstance(value.using, tuple):
            using = [(_saved_value(a), _saved_value(b)) for a, b in value.using]
        elif isinstance(value.using, str):
            using = msgspec.UNSET if value.using == "smallest" else value.using
        elif value.using is not None:
            from svy.errors import SerializationError

            raise SerializationError.ValueError(
                title="Cannot save the singleton rule",
                detail="a collapse rule chosen by a callable cannot be saved.",
                code=Code.SINGLETON_RULE_NOT_SAVABLE,
                param="using",
                hint="Save the mapping it produced, which the handled column of "
                "sample.singletons shows: svy.Singleton('collapse', using={singleton: target}).",
            )

        def unset(v: Any, default: Any) -> Any:
            return msgspec.UNSET if v == default else v

        return SingletonData(
            method=value.method.value,
            domains=unset(value.domains.value, "standard"),
            on_domain_singletons=unset(value.on_domain_singletons, "ignore"),
            using=using,
            within=msgspec.UNSET if value.within is None else list(value.within),
            order_by=msgspec.UNSET if value.order_by is None else list(value.order_by),
            descending=unset(value.descending, False),
            rstate=msgspec.UNSET if value.rstate is None else value.rstate,
            name=msgspec.UNSET if value.name in (None, "__pooled__") else value.name,
        )

    def from_data(self, data: Any) -> Any:
        from svy.core.design import Singleton

        kw: dict[str, Any] = {}
        for f in (
            "domains",
            "on_domain_singletons",
            "within",
            "order_by",
            "descending",
            "rstate",
            "name",
        ):
            v = getattr(data, f)
            if v is not msgspec.UNSET:
                kw[f] = v
        if data.using is not msgspec.UNSET:
            using = data.using
            kw["using"] = (
                using
                if isinstance(using, str)
                else {_as_stratum(a): _as_stratum(b) for a, b in using}
            )
        return Singleton(data.method, **kw)

    def repr(self, value: Any) -> str | None:
        return None if value is None else f"singleton={value!r}"


def _as_stratum(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _saved_value(value: Any) -> Any:
    """A stratum value for the saved struct: tuples as lists; dates kept, since
    ``to_json`` records their type under "temporal" and restores it."""
    if isinstance(value, tuple):
        return [_saved_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# The built-in parts, registered directly: Design is not defined yet when this
# module is first imported (design.py imports it).
for _part in (RepWgtsPart(), WgtAdjustmentPart(), SingletonPart()):
    _REGISTRY[_part.name] = _part
