# src/svy/core/design.py
from __future__ import annotations

import logging
import os
import sys
import warnings

from typing import (
    Any,
    Literal,
    Mapping,
    NamedTuple,
    Self,
    Sequence,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

import msgspec

from svy.core import design_parts as _dp
from svy.core.repwgts import (
    RepWeights,
    RepWgts,
    resolve_rep_variant,
)
from svy.ui.printing import make_panel, render_rich_to_str, resolve_width


log = logging.getLogger(__name__)


# =============================================================================
# Types & Sentinels
# =============================================================================


class _MissingType:
    pass


_MISSING = _MissingType()


def _is_MissingType(x: Any) -> TypeGuard[_MissingType]:
    return x is _MISSING


# =============================================================================
# Population Size (FPC)
# =============================================================================


class PopSize(NamedTuple):
    """
    Finite population correction (FPC) specification for multistage designs.

    Holds column names referencing population sizes at two stages:
    the PSU level and the SSU level (the second sampling stage).

    Parameters
    ----------
    psu : str
        Column name for the population size at the PSU level
        (e.g., total number of hospitals in a stratum).
    ssu : str
        Column name for the population size at the SSU level
        (e.g., total number of schools within a district, or total
        number of discharges within a hospital when there is no
        intermediate stage).

    Examples
    --------
    >>> PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
    PopSize(psu='TOTAL_HOSP', ssu='TOTAL_DISCHARGES')

    Notes
    -----
    For single-stage FPC, use a plain ``str`` for the ``pop_size`` field
    in :class:`Design` instead of ``PopSize``.

    In the standard two-stage variance approximation, the variance is
    computed from PSU-level score totals. The PSU FPC corrects for not
    sampling all PSUs; the SSU FPC corrects for not sampling all SSUs
    within each PSU. Unit-level variability is absorbed into the
    PSU-level residuals.
    """

    psu: str
    ssu: str | None = None


# =============================================================================
# Replicate Weights (Strict Configuration)
# =============================================================================


# =============================================================================
# Resolvers (Internal Helpers)
# =============================================================================

T = TypeVar("T")


@overload
def _pick(current: str, new: str | _MissingType) -> str: ...
@overload
def _pick(current: RepWgts | None, new: RepWgts | None | _MissingType) -> RepWgts | None: ...
@overload
def _pick(current: bool, new: bool | _MissingType) -> bool: ...
@overload
def _pick(current: T, new: T | _MissingType) -> T: ...
def _pick(current: T, new: T | _MissingType) -> T:
    """Overwrite with `new` unless `new` is the _MissingType sentinel."""
    return current if _is_MissingType(new) else cast(T, new)


def _pick_if_none(current: T | None, new: T | _MissingType) -> T | None:
    """
    Only uses `new` when current is None; otherwise keeps current.
    (Useful for "fill defaults" semantics.)
    """
    if current is not None:
        return current
    if _is_MissingType(new):
        return None
    return cast(T, new)


_SVY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _user_stacklevel() -> int:
    """Stack level of the first frame outside svy, relative to the caller.

    The same warning is reached from ``design.update`` and through
    ``sample.update_design``/``use_weight``, at different depths.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_SVY_DIR):
        frame = frame.f_back
        level += 1
    return level


def _warn_rep_wgts_reset(rep: RepWgts, new_wgt: str | None) -> None:
    warnings.warn(
        f"Replicate weights '{rep.prefix}' go with weight {rep.wgt!r}, not {new_wgt!r}, "
        "so they were removed from the design and variance falls back to Taylor "
        f"linearization. Pass rep_wgts= with the update to keep them with {new_wgt!r}.",
        UserWarning,
        stacklevel=_user_stacklevel(),
    )


def _norm_spec(
    name: str,
    value: str | Sequence[str] | None,
) -> str | tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "":
            raise ValueError(f"'{name}' must not be an empty string when provided")
        return value
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"'{name}' must be str | Sequence[str] | None")
    items = list(value)
    if not items:
        raise ValueError(f"'{name}' sequence must not be empty")
    for i, s in enumerate(items):
        if not isinstance(s, str):
            raise TypeError(f"'{name}' items must be str; got {type(s).__name__} at index {i}")
        if s == "":
            raise ValueError(f"'{name}' items must not contain empty strings")
    return tuple(items)


def _norm_pop_size(
    value: str | PopSize | None,
) -> str | PopSize | None:
    """Normalize and validate the pop_size argument."""
    if value is None:
        return None
    if isinstance(value, PopSize):
        if not value.psu or not isinstance(value.psu, str):
            raise ValueError("PopSize.psu must be a non-empty string")
        # ssu is optional: PopSize(psu=...) alone specifies a PSU-only FPC
        # (the type signature always allowed it; validation rejected it).
        if value.ssu is not None and (not value.ssu or not isinstance(value.ssu, str)):
            raise ValueError("PopSize.ssu must be a non-empty string or None")
        return value
    if isinstance(value, str):
        if not value:
            raise ValueError("'pop_size' must not be an empty string when provided")
        return value
    raise TypeError(f"'pop_size' must be str | PopSize | None, got {type(value).__name__}")


# =============================================================================
# Weight Adjustment Record
# =============================================================================


class WgtAdjustment(msgspec.Struct, frozen=True, kw_only=True):
    """How the current weights were last produced.

    One record, replaced by each weighting method: it describes the LAST
    adjustment, and the variance estimator accounts for that step only. Full
    lineage is not a weight log but the chain of Design snapshots on the Sample
    (``Sample.design_history``) -- every adjustment produces a new Design, and
    each carries its own record.

    ``kind`` names the technique, not the producing method, matching the
    ``RepWgts`` tags (``method="jackknife"`` from ``create_jk_wgts``). It is
    read in ``describe()``/repr, where the noun is the form that reads.

    Only what cannot be reconstituted is stored. Adjustment factors are
    ``new_wgt/prev_wgt`` (both columns persist, since no weighting method will
    overwrite an existing column) and achieved controls are ``sum(new_wgt)`` by
    cell, so neither is a field.

    Parameters
    ----------
    kind
        The technique that produced the current weights.
    prev_wgt, new_wgt
        Weight columns before and after. Variance uses ``prev_wgt`` for the
        cell means, matching R.
    cells
        Snapshotted class column(s), one per margin -- a tuple because raking
        sweeps each margin separately. Null marks a row outside the
        adjustment's scope: no centering, factor 1. Absent on provenance-only
        kinds.
    aux
        Resolved auxiliary columns for calibration, whose GREG model has no
        cell structure: its sweep is a WLS residual and needs the matrix.
    pins_total
        Whether the adjustment fixed the population total as well as the
        composition. ``controls`` pin both (k constraints); ``shares`` pin only
        composition (k-1), leaving the total's sampling variability intact.
    """

    kind: Literal[
        "nonresponse",
        "normalization",
        "poststratification",
        "raking",
        "calibration",
        "trimming",
        "standardization",
    ]
    prev_wgt: str
    new_wgt: str
    cells: tuple[str, ...] | None = None
    aux: tuple[str, ...] | None = None
    pins_total: bool = True

    #: Kinds the variance estimator can center for. The rest are provenance
    #: only: normalization's targets are conveniences rather than population
    #: facts, nonresponse has no variance record in R either, and trimming
    #: breaks the constraints a calibration asserted.
    VARIANCE_CONSUMED = frozenset(
        {"poststratification", "raking", "calibration", "standardization"}
    )

    @property
    def is_variance_consumed(self) -> bool:
        return self.kind in self.VARIANCE_CONSUMED

    def _to_code(self) -> str:
        return _struct_code(self, "svy.core.design.WgtAdjustment")


# =============================================================================
# Singleton handling
# =============================================================================

_SINGLETON_METHODS = ("certainty", "skip", "scale", "center", "collapse", "pool")
_POOLED = "__pooled__"


def _stratum_value(value: Any) -> Any:
    """A stratum as stored: its column's value, or a tuple of them for tuple strata."""
    if isinstance(value, list):
        value = tuple(value)
    for v in value if isinstance(value, tuple) else (value,):
        if isinstance(v, (list, tuple, dict, set)):
            raise TypeError(
                f"a stratum value must be a scalar or a tuple of scalars; got {value!r}"
            )
        hash(v)
    return value


def _typed(value: Any) -> Any:
    """A hashable identity that keeps 1, 1.0 and True apart."""
    if isinstance(value, tuple):
        return tuple(_typed(v) for v in value)
    return (type(value), value)


def _strata_arg(strata: Any) -> tuple[Any, ...]:
    if isinstance(strata, (str, bytes)) or not isinstance(strata, (Sequence, set, frozenset)):
        strata = [strata]
    return tuple(strata)


class SingletonSpec(msgspec.Struct, frozen=True, kw_only=True):
    """How the design's singleton strata are handled for variance estimation.

    A resolved decision, made with ``sample.singleton.certainty/skip/scale/
    center/collapse/pool`` or built with the constructor of each method.
    Strata are the stratum columns' own values, a tuple per stratum when the
    design has several stratum columns.

    The sample keeps it only while it describes the data: the singleton strata
    found in the data must be exactly the ones it handles (and a collapse
    target must still exist). Otherwise it is cleared with a warning.

    Parameters
    ----------
    method
        ``"certainty"``, ``"skip"``, ``"scale"``, ``"center"``, ``"collapse"``
        or ``"pool"``.
    strata
        The singleton strata handled (every method but ``collapse``).
    mapping
        ``collapse``: (singleton stratum, target stratum) pairs.
    name
        ``pool``: the pooled pseudo-stratum's name.
    """

    method: Literal["certainty", "skip", "scale", "center", "collapse", "pool"]
    strata: tuple[Any, ...] = ()
    mapping: tuple[tuple[Any, Any], ...] = ()
    name: str | None = None

    def __post_init__(self) -> None:
        if self.method not in _SINGLETON_METHODS:
            raise ValueError(
                f"Unknown singleton method {self.method!r}; use one of {_SINGLETON_METHODS}."
            )
        unique: dict[Any, Any] = {}
        for v in self.strata:
            v = _stratum_value(v)
            unique.setdefault(_typed(v), v)
        strata = tuple(unique.values())
        mapping = tuple((_stratum_value(a), _stratum_value(b)) for a, b in self.mapping)
        if self.method == "collapse":
            if not mapping or strata:
                raise ValueError("collapse takes a non-empty mapping and no strata.")
            sources = {_typed(a) for a, _ in mapping}
            if len(sources) != len(mapping):
                raise ValueError("collapse maps each singleton stratum once.")
            bad = [b for _, b in mapping if _typed(b) in sources]
            if bad:
                raise ValueError(f"collapse targets must not be singletons themselves: {bad!r}")
        elif not strata or mapping:
            raise ValueError(f"{self.method} takes a non-empty strata and no mapping.")
        name = self.name
        if self.method == "pool":
            name = _POOLED if name is None else name
            if not isinstance(name, str) or not name:
                raise ValueError("pool's name must be a non-empty string.")
        elif name is not None:
            raise ValueError(f"name applies to pool only, not {self.method}.")
        msgspec.structs.force_setattr(self, "strata", strata)
        msgspec.structs.force_setattr(self, "mapping", mapping)
        msgspec.structs.force_setattr(self, "name", name)

    @classmethod
    def certainty(cls, strata: Any) -> SingletonSpec:
        """Each singleton's PSU becomes a stratum, its SSUs (or rows) the PSUs."""
        return cls(method="certainty", strata=_strata_arg(strata))

    @classmethod
    def skip(cls, strata: Any) -> SingletonSpec:
        """The singleton strata contribute nothing to the variance."""
        return cls(method="skip", strata=_strata_arg(strata))

    @classmethod
    def scale(cls, strata: Any) -> SingletonSpec:
        """As ``skip``, with the variance scaled up by the singleton fraction."""
        return cls(method="scale", strata=_strata_arg(strata))

    @classmethod
    def center(cls, strata: Any) -> SingletonSpec:
        """The singletons' variance is taken around the grand mean."""
        return cls(method="center", strata=_strata_arg(strata))

    @classmethod
    def collapse(cls, mapping: Mapping[Any, Any]) -> SingletonSpec:
        """Each singleton stratum is merged into its target stratum."""
        return cls(method="collapse", mapping=tuple(dict(mapping).items()))

    @classmethod
    def pool(cls, strata: Any, name: str = _POOLED) -> SingletonSpec:
        """The singleton strata are pooled into one pseudo-stratum."""
        return cls(method="pool", strata=_strata_arg(strata), name=name)

    @property
    def handled(self) -> tuple[Any, ...]:
        """The singleton strata this spec was made for."""
        return tuple(a for a, _ in self.mapping) if self.method == "collapse" else self.strata

    @property
    def targets(self) -> tuple[Any, ...]:
        """The strata the singletons are collapsed into (collapse only)."""
        seen: dict[Any, Any] = {}
        for _, b in self.mapping:
            seen.setdefault(_typed(b), b)
        return tuple(seen.values())

    def __repr__(self) -> str:
        return self._code("", repr)

    def _to_code(self) -> str:
        return self._code("svy.", _value_code)

    def _code(self, prefix: str, fmt: Any) -> str:
        """The constructor call that rebuilds this spec."""
        if self.method == "collapse":
            items = ", ".join(f"{fmt(a)}: {fmt(b)}" for a, b in self.mapping)
            return f"{prefix}SingletonSpec.collapse({{{items}}})"
        strata = "[" + ", ".join(fmt(v) for v in self.strata) + "]"
        name = "" if self.method != "pool" or self.name == _POOLED else f", name={self.name!r}"
        return f"{prefix}SingletonSpec.{self.method}({strata}{name})"


def _value_code(value: Any) -> str:
    if isinstance(value, tuple):
        inner = ", ".join(_value_code(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, float) and value != value:
        return "float('nan')"
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return f"float({str(value)!r})"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    # Dates and other non-literal values: their ISO/str form, which the stratum
    # matching reads back against the column's type.
    iso = getattr(value, "isoformat", None)
    return repr(iso() if callable(iso) else str(value))


def _struct_code(obj: msgspec.Struct, qualname: str) -> str:
    """``qualname(field=value, ...)`` with the fields that differ from their defaults."""
    args = []
    for f in msgspec.structs.fields(obj):
        value = getattr(obj, f.name)
        if f.default is not msgspec.NODEFAULT and value == f.default:
            continue
        args.append(f"{f.name}={_value_code(value)}")
    return f"{qualname}({', '.join(args)})"


# =============================================================================
# Design Definition
# =============================================================================

_FIELDS: tuple[str, ...] = (
    "case_id",
    "wave",
    "stratum",
    "wgt",
    "prob",
    "hit",
    "mos",
    "psu",
    "ssu",
    "pop_size",
    "wr",
)


class Design:
    case_id: str | None
    wave: str | None
    stratum: str | tuple[str, ...] | None
    wgt: str | None
    prob: str | None
    hit: str | None
    mos: str | None
    psu: str | tuple[str, ...] | None
    ssu: str | tuple[str, ...] | None
    pop_size: str | PopSize | None
    wr: bool
    _parts: dict[str, Any]
    _frozen: bool

    PRINT_WIDTH: int | None = None

    # _FIELDS holds the column-name design parameters. Everything else a design
    # carries (replicate weights, the weight-adjustment record, singleton
    # handling) is a design part, one value per registered part in `_parts`;
    # see svy.core.design_parts. `_frozen` stays last: copy/pickle restore the
    # slots in order and the guard must be the last one set.
    __slots__ = (*_FIELDS, "_parts", "_frozen")

    def __init__(
        self,
        case_id: str | None = None,
        wave: str | None = None,
        stratum: str | Sequence[str] | None = None,
        wgt: str | None = None,
        prob: str | None = None,
        hit: str | None = None,
        mos: str | None = None,
        psu: str | Sequence[str] | None = None,
        ssu: str | Sequence[str] | None = None,
        pop_size: str | PopSize | None = None,
        wr: bool = False,
        rep_wgts: RepWgts | None = None,
        wgt_adjustment: WgtAdjustment | None = None,
        singleton: SingletonSpec | None = None,
        **parts: Any,
    ) -> None:
        object.__setattr__(self, "_frozen", False)

        norm_stratum = _norm_spec("stratum", stratum)
        norm_psu = _norm_spec("psu", psu)
        norm_ssu = _norm_spec("ssu", ssu)
        norm_pop_size = _norm_pop_size(pop_size)

        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "wave", wave)
        object.__setattr__(self, "stratum", norm_stratum)
        object.__setattr__(self, "wgt", wgt)
        object.__setattr__(self, "prob", prob)
        object.__setattr__(self, "hit", hit)
        object.__setattr__(self, "mos", mos)
        object.__setattr__(self, "psu", norm_psu)
        object.__setattr__(self, "ssu", norm_ssu)
        object.__setattr__(self, "pop_size", norm_pop_size)
        object.__setattr__(self, "wr", wr)

        # Validate simple string-or-None fields (pop_size excluded — handled by _norm_pop_size)
        for name in ("case_id", "wave", "wgt", "prob", "hit", "mos"):
            val = getattr(self, name)
            if val is not None and not isinstance(val, str):
                raise TypeError(f"{name!r} must be str | None, got {type(val).__name__}")
            if isinstance(val, str) and not val:
                raise ValueError(f"{name!r} must not be an empty string when provided")

        if not isinstance(self.wr, bool):
            raise TypeError(f"'wr' must be bool, got {type(self.wr).__name__}")

        given = {
            "rep_wgts": rep_wgts,
            "wgt_adjustment": wgt_adjustment,
            "singleton": singleton,
            **parts,
        }
        _dp.check_part_names(given, where="Design()")
        fields = self._fields()
        object.__setattr__(
            self,
            "_parts",
            {p.name: p.check(given.get(p.name), fields) for p in _dp.registered()},
        )

        object.__setattr__(self, "_frozen", True)

    def _fields(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in _FIELDS}

    # -----------------------------
    # Design parts
    # -----------------------------
    @property
    def rep_wgts(self) -> RepWgts | None:
        """Replicate weights, or None for a Taylor design."""
        return self._parts.get("rep_wgts")

    @property
    def wgt_adjustment(self) -> WgtAdjustment | None:
        """How the design's weight was last produced, or None."""
        return self._parts.get("wgt_adjustment")

    @property
    def singleton(self) -> SingletonSpec | None:
        """How the singleton strata are handled for variance, or None."""
        return self._parts.get("singleton")

    def __getattr__(self, name: str) -> Any:
        # Only reached for names that are not slots or class attributes: a part
        # registered after the class was defined has no property of its own.
        try:
            parts = object.__getattribute__(self, "_parts")
        except AttributeError:
            raise AttributeError(name) from None
        if name in parts:
            return parts[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _with_part(self, name: str, value: Any) -> Self:
        """This design with one part replaced, bypassing the update rules."""
        return type(self)(**self._fields(), **{**self._parts, name: value})

    # -----------------------------
    # Immutability Guards
    # -----------------------------
    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Design is frozen; use .update(...) to create a modified copy.")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Design is frozen; attributes cannot be deleted.")
        object.__delattr__(self, name)

    # -----------------------------
    # Properties
    # -----------------------------
    @property
    def n_reps(self) -> int | None:
        """How many replicate weights this design carries, or None for Taylor.

        Saves the ``if design.rep_wgts is not None`` that every caller of
        ``design.rep_wgts.n_reps`` otherwise needs.
        """
        return None if self.rep_wgts is None else self.rep_wgts.n_reps

    @property
    def method(self) -> str:
        """Convenience accessor for the estimation method, as a display label.

        ``"Taylor"`` when the design carries no replicate weights, otherwise
        the variant's own label. For display and reporting only.
        """
        if self.rep_wgts is None:
            return "Taylor"
        return self.rep_wgts.method

    # -----------------------------
    # Update Methods
    # -----------------------------
    def update(
        self,
        *,
        case_id: str | None | _MissingType = _MISSING,
        wave: str | None | _MissingType = _MISSING,
        stratum: str | Sequence[str] | None | _MissingType = _MISSING,
        wgt: str | None | _MissingType = _MISSING,
        prob: str | None | _MissingType = _MISSING,
        hit: str | None | _MissingType = _MISSING,
        mos: str | None | _MissingType = _MISSING,
        psu: str | Sequence[str] | None | _MissingType = _MISSING,
        ssu: str | Sequence[str] | None | _MissingType = _MISSING,
        pop_size: str | PopSize | None | _MissingType = _MISSING,
        wr: bool | _MissingType = _MISSING,
        rep_wgts: RepWgts | _MissingType | None = _MISSING,
        wgt_adjustment: WgtAdjustment | _MissingType | None = _MISSING,
        singleton: SingletonSpec | _MissingType | None = _MISSING,
        **parts: Any,
    ) -> Self:
        """A copy with the named fields replaced.

        What is not passed follows its part's rule: replicate weights and the
        weight-adjustment record stay while they describe the new weight, and
        singleton handling stays unless the stratum, PSU or SSU columns change
        (then it is cleared with a warning).
        """
        return self._merge(
            only_if_none=False,
            case_id=case_id,
            wave=wave,
            stratum=stratum,
            wgt=wgt,
            prob=prob,
            hit=hit,
            mos=mos,
            psu=psu,
            ssu=ssu,
            pop_size=pop_size,
            wr=wr,
            rep_wgts=rep_wgts,
            wgt_adjustment=wgt_adjustment,
            singleton=singleton,
            **parts,
        )

    def fill_missing(
        self,
        *,
        case_id: str | None | _MissingType = _MISSING,
        wave: str | None | _MissingType = _MISSING,
        stratum: str | Sequence[str] | None | _MissingType = _MISSING,
        wgt: str | None | _MissingType = _MISSING,
        prob: str | None | _MissingType = _MISSING,
        hit: str | None | _MissingType = _MISSING,
        mos: str | None | _MissingType = _MISSING,
        psu: str | Sequence[str] | None | _MissingType = _MISSING,
        ssu: str | Sequence[str] | None | _MissingType = _MISSING,
        pop_size: str | PopSize | None | _MissingType = _MISSING,
        wr: bool | _MissingType = _MISSING,
        rep_wgts: RepWgts | Sequence[str] | _MissingType | None = _MISSING,
        **parts: Any,
    ) -> Self:
        # Sequence[str] is handled by the rep_wgts part (it counts as not passed).
        return self._merge(
            only_if_none=True,
            case_id=case_id,
            wave=wave,
            stratum=stratum,
            wgt=wgt,
            prob=prob,
            hit=hit,
            mos=mos,
            psu=psu,
            ssu=ssu,
            pop_size=pop_size,
            wr=wr,
            rep_wgts=rep_wgts,
            **parts,
        )

    def update_rep_weights(
        self,
        *,
        method: Literal["brr", "bootstrap", "jackknife", "sdr"] | None | _MissingType = _MISSING,
        prefix: str | _MissingType = _MISSING,
        n_reps: int | _MissingType = _MISSING,
        fay_coef: float | _MissingType = _MISSING,
        df: float | None | _MissingType = _MISSING,
        padding: int | None | _MissingType = _MISSING,
        scale: float | Sequence[float] | None | _MissingType = _MISSING,
        rep_coefs: tuple[float, ...] | None | _MissingType = _MISSING,
        kind: str | None | _MissingType = _MISSING,
    ) -> Self:
        """Return a new Design with selected RepWeights fields updated.

        ``method=None`` clears the replicate weights. Creating them for the
        first time requires ``method``, ``prefix`` and ``n_reps``; afterwards
        each is taken from the current design unless named.

        Internal code prefers
        ``design.update(rep_wgts=msgspec.structs.replace(rw, ...))``, which is
        the same operation without the string round-trip. This exists for
        callers holding a method *name* rather than a variant.
        """
        supplied: dict[str, Any] = {
            name: value
            for name, value in (
                ("method", method),
                ("prefix", prefix),
                ("n_reps", n_reps),
                ("fay_coef", fay_coef),
                ("df", df),
                ("padding", padding),
                ("scale", scale),
                ("rep_coefs", rep_coefs),
                ("kind", kind),
            )
            if not isinstance(value, _MissingType)
        }
        if not supplied:
            return self
        if "method" in supplied and supplied["method"] is None:
            return self.update(rep_wgts=None)

        cur = self.rep_wgts
        for name in ("method", "prefix", "n_reps"):
            if name not in supplied:
                if cur is None:
                    raise ValueError(
                        f"When initializing RepWeights for the first time, '{name}' is mandatory."
                    )
                supplied[name] = getattr(cur, name)

        method_name = supplied.pop("method")

        # Same method: `replace` preserves every field the caller did not name,
        # including the variant's own, and re-runs the struct's validation.
        if cur is not None and type(cur) is resolve_rep_variant(method_name):
            return self.update(rep_wgts=msgspec.structs.replace(cur, **supplied))

        # A different method, or none yet. The shared fields carry over; the
        # outgoing variant's own do not, because a bootstrap kind means nothing
        # on a jackknife. Anything named explicitly still applies to the new
        # variant, and a parameter the new variant does not carry is refused
        # rather than dropped.
        if cur is not None:
            for name in ("df", "padding", "scale", "rep_coefs"):
                supplied.setdefault(name, getattr(cur, name))
        return self.update(rep_wgts=RepWeights(method=method_name, **supplied))

    # -----------------------------
    # Internal Merge Logic
    # -----------------------------
    def _merge(
        self,
        *,
        only_if_none: bool,
        case_id: str | None | _MissingType = _MISSING,
        wave: str | None | _MissingType = _MISSING,
        stratum: str | Sequence[str] | None | _MissingType = _MISSING,
        wgt: str | None | _MissingType = _MISSING,
        prob: str | None | _MissingType = _MISSING,
        hit: str | None | _MissingType = _MISSING,
        mos: str | None | _MissingType = _MISSING,
        psu: str | Sequence[str] | None | _MissingType = _MISSING,
        ssu: str | Sequence[str] | None | _MissingType = _MISSING,
        pop_size: str | PopSize | None | _MissingType = _MISSING,
        wr: bool | _MissingType = _MISSING,
        **parts: Any,
    ) -> Self:
        """
        Internal: merge fields either by overwriting or only filling when current is None.
        """
        pick = _pick_if_none if only_if_none else _pick

        def is_missing(x: object, /) -> TypeGuard[_MissingType]:
            return x is _MISSING

        def _norm_multi_arg(
            field_name: str, val: str | Sequence[str] | None | _MissingType
        ) -> str | tuple[str, ...] | None | _MissingType:
            if is_missing(val):
                return _MISSING
            if val is None:
                return None
            if isinstance(val, str):
                if val == "":
                    raise ValueError(f"'{field_name}' must not be an empty string when provided")
                return val
            if not isinstance(val, Sequence) or isinstance(val, (bytes, bytearray)):
                raise TypeError(f"'{field_name}' must be a sequence of str")
            if len(val) == 0:
                raise ValueError(f"'{field_name}' sequence must not be empty")
            for x in val:
                if not isinstance(x, str):
                    raise TypeError(f"'{field_name}' items must be str")
                if x == "":
                    raise ValueError(f"'{field_name}' items must not be empty")
            return cast(tuple[str, ...], tuple(val))

        def _norm_pop_size_arg(
            val: str | PopSize | None | _MissingType,
        ) -> str | PopSize | None | _MissingType:
            if is_missing(val):
                return _MISSING
            return _norm_pop_size(val)

        stratum_arg = _norm_multi_arg("stratum", stratum)
        psu_arg = _norm_multi_arg("psu", psu)
        ssu_arg = _norm_multi_arg("ssu", ssu)
        pop_size_arg = _norm_pop_size_arg(pop_size)

        fields: dict[str, Any] = {
            "case_id": pick(self.case_id, case_id),
            "wave": pick(self.wave, wave),
            "stratum": pick(self.stratum, stratum_arg),
            "wgt": pick(self.wgt, wgt),
            "prob": pick(self.prob, prob),
            "hit": pick(self.hit, hit),
            "mos": pick(self.mos, mos),
            "psu": pick(self.psu, psu_arg),
            "ssu": pick(self.ssu, ssu_arg),
            "pop_size": pick(self.pop_size, pop_size_arg),
            "wr": _pick(self.wr, wr),
        }

        # Each part decides what it becomes: a passed value is taken (the part
        # may pair it with the new fields), one not passed follows its rule.
        _dp.check_part_names(parts, where="Design.update()")
        new_parts: dict[str, Any] = {}
        for part in _dp.registered():
            arg = parts.get(part.name, _MISSING)
            passed = not is_missing(arg) and not (
                only_if_none and self._parts.get(part.name) is not None
            )
            new_parts[part.name] = part.after_update(self, arg if passed else None, passed, fields)

        return type(self)(**fields, **new_parts)

    # -----------------------------
    # Introspection
    # -----------------------------
    @property
    def variance_psu(self) -> str | tuple[str, ...] | None:
        """PSU for variance: the declared one, else the case on a panel.

        On a long panel with element sampling a case's rows are repeated
        measures and must form one cluster, otherwise ``by=wave`` contrasts
        treat them as independent and the change variance is wrong. The
        fallback is restricted to panels (``wave`` set): on a cross-section
        a case is one row, so the element path gives the same numbers.
        """
        if self.psu is not None:
            return self.psu
        return self.case_id if self.wave is not None else None

    @property
    def is_panel(self) -> bool:
        """True when both ``case_id`` and ``wave`` are declared."""
        return self.case_id is not None and self.wave is not None

    def specified_fields(
        self,
        *,
        ignore_cols: Sequence[str] | None = None,
        data_columns: Sequence[str] | None = None,
    ) -> list[str]:
        """
        Return a de-duplicated (order-preserving) list of column names referenced
        by the design (stratum/psu/ssu/etc.), including replicate weight columns.

        Parameters
        ----------
        ignore_cols : Sequence[str], optional
            Column names to ignore
        data_columns : Sequence[str], optional
            Actual data column names (used for auto-detecting padding in rep weights)

        Returns
        -------
        list[str]
            List of all column names referenced by this design
        """
        default_ignores = {"wr"}
        ignore = default_ignores | (set(ignore_cols) if ignore_cols else set())

        out: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name and name not in ignore and name not in seen:
                out.append(name)
                seen.add(name)

        # 1. Add standard fields
        for name in _FIELDS:
            if name in ignore:
                continue

            val = getattr(self, name, None)
            if not val:
                continue

            # Handle multi-column fields
            if name in {"stratum", "psu", "ssu"}:
                if isinstance(val, str):
                    add(val)
                elif isinstance(val, (tuple, list)):
                    for s in val:
                        add(s)
                continue

            # Handle PopSize
            if name == "pop_size":
                if isinstance(val, PopSize):
                    add(val.psu)
                    add(val.ssu)
                elif isinstance(val, str):
                    add(val)
                continue

            # Handle standard string fields
            if isinstance(val, str):
                add(val)

        # 2. Add Replicate Weight columns with auto-detection
        if self.rep_wgts:
            if data_columns is not None:
                # Use auto-detection from actual data
                rep_cols = self.rep_wgts.columns_from_data(data_columns)
            else:
                # Fall back to default columns (explicit padding or no padding)
                rep_cols = self.rep_wgts.columns

            for col in rep_cols:
                add(col)

        return out

    def columns(self, data_columns: Sequence[str] | None = None) -> list[str]:
        """Every column this design needs from the data, de-duplicated, in order.

        Design fields, ``pop_size``, then each design part's columns:
        replicate weights and the units they were built from, then the
        weight-adjustment record's ``new_wgt``, ``prev_wgt``, ``cells`` and
        ``aux``. A frame missing any of them does not belong to this design.

        Parameters
        ----------
        data_columns : Sequence[str], optional
            The data's column names, used to resolve auto-detected replicate
            padding. Without them replicate names use the explicit padding.
        """
        out = self.specified_fields(data_columns=data_columns)
        seen = set(out)

        def add(name: str | tuple[str, ...] | None) -> None:
            for c in (name,) if isinstance(name, str) else (name or ()):
                if c not in seen:
                    out.append(c)
                    seen.add(c)

        for part, value in self._part_items():
            for c in part.columns(value, self, data_columns):
                add(c)
        return out

    def _part_items(self) -> list[tuple[_dp.DesignPart, Any]]:
        """(part, value) for every registered part this design carries."""
        return [(p, v) for p in _dp.registered() if (v := self._parts.get(p.name)) is not None]

    # -----------------------------
    # Printing & Rendering
    # -----------------------------
    @staticmethod
    def _pad_and_surround(text: str, *, indent: int = 2, surround: bool = False) -> str:
        if text is None:
            return ""
        text = str(text).rstrip("\n")
        if indent > 0:
            pad = " " * indent
            text = "\n".join(pad + line if line else pad for line in text.splitlines())
        return f"\n{text}\n" if surround else text

    @staticmethod
    def _fmt_tuple_names(x) -> str:
        if x is None:
            return "None"
        if isinstance(x, (tuple, list)):
            inner = ", ".join(str(v) for v in x)
            if len(x) == 1:
                inner += ","
            return f"({inner})"
        return str(x)

    def _fmt_psu(self) -> str:
        if self.psu is None and self.variance_psu is not None:
            return f"None (variance: {self.variance_psu})"
        return self._fmt_tuple_names(self.psu)

    @staticmethod
    def _fmt_pop_size(x) -> str:
        if x is None:
            return "None"
        if isinstance(x, PopSize):
            return f"PopSize(psu='{x.psu}', ssu='{x.ssu}')"
        return str(x)

    def _repweights_summary(self) -> str:
        if self.rep_wgts is None:
            return "None"
        fn = getattr(self.rep_wgts, "__plain_str__", None)
        return fn() if callable(fn) else repr(self.rep_wgts)

    def __rich_console__(self, console, options):
        from rich.table import Table as RTable
        from rich.text import Text

        t = RTable(
            show_header=False,
            box=None,
            show_edge=False,
            show_lines=False,
            pad_edge=False,
            expand=False,
        )
        t.add_column("Field", justify="left", no_wrap=True)
        t.add_column("Value", justify="left", no_wrap=False, overflow="fold")

        rows: list[tuple[str, str]] = [
            ("Case id", str(self.case_id)),
            ("Wave", str(self.wave)),
            ("Stratum", self._fmt_tuple_names(self.stratum)),
            ("PSU", self._fmt_psu()),
            ("SSU", self._fmt_tuple_names(self.ssu)),
            ("Weight", str(self.wgt)),
            ("With replacement", str(bool(self.wr))),
            ("Prob", str(self.prob)),
            ("Hit", str(self.hit)),
            ("MOS", str(self.mos)),
            ("Population size", self._fmt_pop_size(self.pop_size)),
        ]
        if self.singleton is not None:
            rows.append(("Singleton", repr(self.singleton)))
        for k, v in rows:
            t.add_row(k, v)

        # Rep weights — sub-fields as separate rows, not bold
        if self.rep_wgts is None:
            t.add_row("Replicate weights", "None")
        else:
            sub_lines = self._repweights_summary().splitlines()
            t.add_row("Replicate weights", "")
            for sub_line in sub_lines[1:]:
                t.add_row(Text(f"    {sub_line}", style="not bold"), "")

        yield make_panel([t], title="Design", obj=self, kind="estimate")

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed."""
        lines: list[str] = [
            "Design",
            f"  Case id          : {self.case_id}",
            f"  Wave             : {self.wave}",
            f"  Stratum          : {self._fmt_tuple_names(self.stratum)}",
            f"  PSU              : {self._fmt_psu()}",
            f"  SSU              : {self._fmt_tuple_names(self.ssu)}",
            f"  Weight           : {self.wgt}",
            f"  With replacement : {bool(self.wr)}",
            f"  Prob             : {self.prob}",
            f"  Hit              : {self.hit}",
            f"  MOS              : {self.mos}",
            f"  Population size  : {self._fmt_pop_size(self.pop_size)}",
        ]
        if self.singleton is not None:
            lines.append(f"  Singleton        : {self.singleton!r}")
        if self.rep_wgts is not None:
            sub_lines = self._repweights_summary().splitlines()
            lines.append("  Replicate weights")
            for sub_line in sub_lines[1:]:
                lines.append(f"      {sub_line}")
        else:
            lines.append("  Replicate weights : None")
        return "\n".join(lines)

    def __str__(self) -> str:
        result = render_rich_to_str(self, width=resolve_width(self))
        return self._pad_and_surround(result, indent=2, surround=False)

    @classmethod
    def set_default_print_width(cls, width: int | None) -> None:
        if width is None:
            cls.PRINT_WIDTH = None
            return
        try:
            w = int(width)
        except Exception as ex:
            raise TypeError(f"print width must be int or None; got {width!r}") from ex
        if w <= 20:
            raise ValueError("print width must be > 20 characters.")
        cls.PRINT_WIDTH = w

    def show(self, *, use_rich: bool = True) -> None:
        from svy.ui.printing import rich_available

        if use_rich and rich_available():
            import sys

            from rich.console import Console

            Console(
                file=sys.stdout,
                force_terminal=True,
                emoji=False,
                width=resolve_width(self),
                soft_wrap=True,
            ).print(self)
            return
        print(self.__plain_str__())

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.case_id is not None:
            parts.append(f"case_id={self.case_id!r}")
        if self.wave is not None:
            parts.append(f"wave={self.wave!r}")

        def add_nonempty(name: str, value) -> None:
            if value is None:
                return
            if isinstance(value, (tuple, list)) and not value:
                return
            parts.append(f"{name}={value!r}")

        add_nonempty("stratum", self.stratum)
        add_nonempty("psu", self.psu)
        add_nonempty("ssu", self.ssu)
        add_nonempty("wgt", self.wgt)
        add_nonempty("prob", self.prob)
        add_nonempty("hit", self.hit)
        add_nonempty("mos", self.mos)
        add_nonempty("pop_size", self.pop_size)
        if self.wr:
            parts.append("wr=True")

        for part in _dp.registered():
            shown = part.repr(self._parts.get(part.name))
            if shown:
                parts.append(shown)

        return f"Design({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Design):
            return False
        names = [p.name for p in _dp.registered()]
        return all(getattr(self, f) == getattr(other, f) for f in _FIELDS) and all(
            self._parts.get(n) == other._parts.get(n) for n in names
        )

    def __hash__(self) -> int:
        return hash(
            (
                tuple(getattr(self, f) for f in _FIELDS),
                tuple(self._parts.get(p.name) for p in _dp.registered()),
            )
        )

    def _to_code(self) -> str:
        """Source that rebuilds this design: ``svy.Design(...)``, runnable with
        only ``import svy``."""
        args = []
        for f in _FIELDS:
            value = getattr(self, f)
            if value is None or (f == "wr" and value is False):
                continue
            if isinstance(value, PopSize):
                code = f"svy.PopSize(psu={value.psu!r}, ssu={value.ssu!r})"
            else:
                code = _value_code(value)
            args.append(f"{f}={code}")
        for part, value in self._part_items():
            args.append(f"{part.name}={part.to_code(value)}")
        return f"svy.Design({', '.join(args)})"
