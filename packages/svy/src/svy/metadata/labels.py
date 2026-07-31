# src/svy/metadata/labels.py
from __future__ import annotations

import builtins
import logging
import math
import threading

from typing import Iterable, Mapping, Self

import msgspec
import polars as pl

from msgspec.structs import force_setattr, replace

from svy.core.enumerations import MissingKind
from svy.core.types import Category
from svy.errors.label_errors import LabelError
from svy.metadata.variable_meta import ValueLabel, _coerce_labels


log = logging.getLogger(__name__)
# ============================= # Labels & Schemes # =============================


class Label(msgspec.Struct, frozen=True):
    """Variable label + optional value labels (code -> text).

    Notes
    -----
    Label/value-labels are intended for variables measured as
    NOMINAL, ORDINAL, or BOOLEAN (see MeasurementType).

    ``categories`` holds ``ValueLabel`` pairs and accepts a dict when
    constructing. It was ``dict[Category, str] | None | _MissingType``, which
    could not be decoded at all — two defects stacked. A ``Category``-keyed
    dict loses its code types through JSON (see ``ValueLabel``), and msgspec
    refuses any union containing a custom type, so the struct raised on every
    round trip regardless.

    The ``_MissingType`` member is gone with it. Nothing ever set this field to
    the sentinel — ``None`` already means "no value labels" — so it bought no
    distinction and cost the ability to decode.
    """

    label: str
    categories: tuple[ValueLabel, ...] | None = None

    def __post_init__(self) -> None:
        force_setattr(self, "categories", _coerce_labels(self.categories))

    @property
    def label_map(self) -> dict[Category, str]:
        """The categories as a mapping, for lookup. Empty when unset."""
        return {vl.code: vl.label for vl in self.categories or ()}

    def clone(self, **overrides) -> Self:
        return replace(self, **overrides)


class SchemeEntry(msgspec.Struct, frozen=True):
    """One category: its code, its label, and everything else about that code.

    Everything a scheme knows about a code lives here rather than in a separate
    collection keyed by it. That is what makes the scheme's invariants
    unrepresentable instead of checked: a missing code that is not in the
    scheme, or a kind for a code that is not missing, cannot be written down.
    """

    code: Category
    label: str
    #: The code's parent in another scheme — a district's region. This is what
    #: lets a cascading list (region → district → ward → enumeration area) be
    #: referenced by concept rather than inlined into a survey specification,
    #: which matters because a geography is revised on its own cycle: a census
    #: redraw should not read as a change to the questionnaire.
    #:
    #: Not validated here. The parent belongs to a *different* scheme, which
    #: this one has no way to see.
    parent: Category | None = None
    #: Set when this code is a non-substantive answer, and says why. ``None``
    #: means an ordinary category.
    missing: MissingKind | None = None

    @property
    def is_missing(self) -> bool:
        return self.missing is not None


class CategoryScheme(msgspec.Struct, kw_only=True, frozen=True):
    """One value-label scheme for a given (concept, locale).

    A **list of entries**, not four parallel collections keyed by code. The
    previous shape held ``mapping``, ``missing``, ``missing_kinds`` and a
    hierarchy separately, which had two costs. Three of the four were dicts or
    sets keyed by ``Category``, and JSON object keys are always strings — so
    ``{101: "Banjul"}`` returned as ``{"101": "Banjul"}`` unless it went through
    a bespoke encoder, silently and with no error. And because the four could
    disagree, seventy-odd lines existed only to check that they did not.

    One entry per code removes both. Every field here is JSON-native, so the
    scheme round-trips through plain msgspec with no custom encoder at all, and
    the invariants hold by construction rather than by validation.

    A dict is still accepted when constructing, since ``{101: "Banjul"}`` reads
    better than a list of structs for the common case::

        CategoryScheme(concept="sex", entries={1: "Male", 2: "Female"})
        CategoryScheme(concept="gm_district", entries=[
            SchemeEntry(code=101, label="Banjul", parent=1),
            SchemeEntry(code=99, label="Refused", missing=MissingKind.REFUSED),
        ])
    """

    concept: str
    entries: tuple[SchemeEntry, ...] = ()
    id: str | None = None
    locale: str | None = None
    title: str | None = None
    ordered: bool = False

    def __post_init__(self):
        # force_setattr, not object.__setattr__: the latter raises "can't apply
        # this __setattr__" on a frozen msgspec Struct under 3.11 and 3.12, and
        # works on 3.13+, so a matrix is the only thing that catches it.
        force_setattr(self, "entries", _coerce_entries(self.entries))
        if self.id is None:
            force_setattr(self, "id", f"{self.concept}:{self.locale or 'default'}")

        seen: set[Category] = set()
        for entry in self.entries:
            if _is_nan(entry.code):
                raise LabelError.nan_key_forbidden(where="CategoryScheme")
            if entry.code in seen:
                raise LabelError.duplicate_code(where="CategoryScheme", code=entry.code)
            seen.add(entry.code)

    # -- lookups -----------------------------------------------------------

    def entry(self, code: Category) -> SchemeEntry | None:
        for e in self.entries:
            if e.code == code:
                return e
        return None

    @property
    def codes(self) -> tuple[Category, ...]:
        return tuple(e.code for e in self.entries)

    @property
    def labels(self) -> dict[Category, str]:
        """Every code and its label, including non-substantive ones."""
        return {e.code: e.label for e in self.entries}

    @property
    def substantive(self) -> tuple[SchemeEntry, ...]:
        """Entries that are real answers — what a frequency table's base is."""
        return tuple(e for e in self.entries if e.missing is None)

    @property
    def missing_codes(self) -> frozenset[Category]:
        return frozenset(e.code for e in self.entries if e.missing is not None)

    def kind_of(self, code: Category) -> MissingKind | None:
        entry = self.entry(code)
        return entry.missing if entry else None

    def codes_of_kind(self, *kinds: MissingKind) -> frozenset[Category]:
        wanted = set(kinds)
        return frozenset(e.code for e in self.entries if e.missing in wanted)

    def parent_of(self, code: Category) -> Category | None:
        entry = self.entry(code)
        return entry.parent if entry else None

    def children_of(self, parent: Category) -> tuple[Category, ...]:
        """Every code under a given parent, in declaration order."""
        return tuple(e.code for e in self.entries if e.parent == parent)

    def clone(self, **overrides) -> Self:
        return replace(self, **overrides)


def _coerce_entries(value) -> tuple[SchemeEntry, ...]:
    """Accept a dict, pairs, or entries; store entries.

    The dict is authoring sugar and never reaches serialization, so the
    string-key problem it would otherwise carry cannot arise.
    """
    if isinstance(value, Mapping):
        return tuple(SchemeEntry(code=c, label=lbl) for c, lbl in value.items())
    return tuple(v if isinstance(v, SchemeEntry) else SchemeEntry(*v) for v in value)


# =============================
# Helpers (normalization, scoring)
# =============================


def _primary(lang: str | None) -> str | None:
    """'fr-CA' -> 'fr'."""
    if not lang:
        return None
    return lang.split("-")[0].lower()


def _match_score(s_locale: str | None, want: str | None) -> int:
    """Higher is better: 3 exact, 2 primary match, 1 neutral."""
    if want is None:
        return 1
    if s_locale == want:
        return 3
    if _primary(s_locale) == _primary(want):
        return 2
    return 1


def _norm_concept(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _norm_locale(loc: str | None) -> str | None:
    return None if not loc else loc.replace("_", "-").lower()


def _is_nan(x: object) -> bool:
    try:
        return math.isnan(x)  # type: ignore[arg-type]
    except Exception:
        return False


# =============================
# Validation & normalization
# =============================


def validate_scheme_missing(s: CategoryScheme, *, strict: bool = True) -> None:
    """Retained as a no-op seam.

    Everything it used to check is now unrepresentable. A missing code outside
    the scheme, a kind for a code that is not missing, a NaN code, a duplicate —
    each was a way for four parallel collections to disagree, and there is one
    collection now. NaN and duplicate codes are rejected in ``__post_init__``,
    where they cannot be skipped by a caller who forgets to validate.
    """
    return None


def normalize_scheme_missing(s: CategoryScheme) -> CategoryScheme:
    """Retained as a no-op seam.

    There is nothing left to derive: ``missing`` used to be recoverable from
    ``missing_kinds``, and both are now one field on the entry.
    """
    return s


def missing_codes_by_kind(s: CategoryScheme, kinds: set[MissingKind]) -> set[Category]:
    """Collect codes matching any of the requested kinds."""
    return set(s.codes_of_kind(*kinds))


# =============================
# Scheme factory
# =============================


def make_scheme(
    *,
    concept: str,
    mapping: Mapping[Category, str] | None = None,
    entries: Iterable[SchemeEntry] | Mapping[Category, str] | None = None,
    locale: str | None = None,
    title: str | None = None,
    ordered: bool = False,
    missing: Iterable[Category] | None = None,
    missing_kinds: Mapping[Category, MissingKind] | None = None,
    parents: Mapping[Category, Category] | None = None,
    id: str | None = None,
) -> CategoryScheme:
    """Build a scheme with a predictable id (``concept:locale``).

    Takes the four facets separately — labels, which codes are missing, why, and
    the hierarchy — and folds them into one entry per code, which is how the
    scheme stores them. That is a convenience for callers who already hold the
    pieces apart; constructing ``CategoryScheme`` directly is the direct route.

    ``mapping`` and ``entries`` are the same argument under two names, since the
    former reads better for a plain code→label list.
    """
    source = entries if entries is not None else (mapping or {})
    concept_key = _norm_concept(concept)
    loc = _norm_locale(locale)
    sid = id if id is not None else (f"{concept_key}:{loc}" if loc else concept_key)

    kinds = dict(missing_kinds or {})
    flagged = set(missing or ()) | set(kinds)
    parent_of = dict(parents or {})

    built = []
    for entry in _coerce_entries(source):
        built.append(
            SchemeEntry(
                code=entry.code,
                label=entry.label,
                parent=entry.parent if entry.parent is not None else parent_of.get(entry.code),
                missing=(
                    entry.missing
                    if entry.missing is not None
                    else (
                        kinds.get(entry.code, MissingKind.NO_ANSWER)
                        if entry.code in flagged
                        else None
                    )
                ),
            )
        )

    stray = flagged - {e.code for e in built}
    if stray:
        raise LabelError.invalid_missing_codes(
            where="labels.make_scheme",
            param="missing",
            not_in_mapping=sorted(stray, key=repr),
        )

    return CategoryScheme(
        id=sid,
        concept=concept_key,
        entries=tuple(built),
        locale=loc,
        title=title,
        ordered=ordered,
    )


# =============================
# Catalog (locale-aware, chainable)
# =============================


class LabellingCatalog:
    """Catalogue of reusable value-label schemes (thread-safe, locale-aware).

    Notes
    -----
    Intended for variables measured as NOMINAL, ORDINAL, or BOOLEAN.
    """

    def __init__(
        self,
        schemes: Iterable[CategoryScheme] = (),
        name: str = "default",
        *,
        locale: str | None = None,
    ):
        self._name = name
        self._lock = threading.RLock()
        self._schemes: dict[str, CategoryScheme] = {s.id: s for s in schemes if s.id is not None}
        self._locale: str | None = _norm_locale(locale)

    # Locale
    @property
    def locale(self) -> str | None:
        return self._locale

    def set_locale(self, locale: str | None) -> None:
        self._locale = _norm_locale(locale)

    # CRUD (chainable)
    def register(self, scheme: CategoryScheme, *, overwrite: bool = False) -> "LabellingCatalog":
        with self._lock:
            if not overwrite and scheme.id in self._schemes:
                raise LabelError.scheme_exists(
                    where="labels.LabellingCatalog.register",
                    scheme_id=scheme.id,
                )
            # Validate again here (defensive) when bringing external schemes
            validate_scheme_missing(scheme, strict=True)
            if scheme.id is not None:
                self._schemes[scheme.id] = scheme
        return self

    def register_many(
        self, *schemes: CategoryScheme, overwrite: bool = False
    ) -> "LabellingCatalog":
        for s in schemes:
            self.register(s, overwrite=overwrite)
        return self

    def add_scheme(
        self,
        *,
        concept: str,
        mapping: Mapping[Category, str],
        locale: str | None = None,
        title: str | None = None,
        ordered: bool = False,
        missing: set[Category] | None = None,
        missing_kinds: dict[Category, MissingKind] | None = None,
        id: str | None = None,
        overwrite: bool = False,
    ) -> "LabellingCatalog":
        """High-level convenience: build a scheme from kwargs and register it."""
        scheme = make_scheme(
            concept=concept,
            mapping=mapping,
            locale=locale or self._locale,
            title=title,
            ordered=ordered,
            missing=missing,
            missing_kinds=missing_kinds,
            id=id,
        )
        return self.register(scheme, overwrite=overwrite)

    def add_schemes(self, *defs: dict, overwrite: bool = False) -> "LabellingCatalog":
        """Batch add from dictionaries of kwargs accepted by add_scheme."""
        for d in defs:
            self.add_scheme(**d, overwrite=overwrite)
        return self

    def get(self, scheme_id: str) -> CategoryScheme:
        try:
            return self._schemes[scheme_id]
        except KeyError as e:
            raise LabelError.unknown_scheme(
                where="labels.LabellingCatalog.get",
                param="scheme_id",
                got=scheme_id,
            ) from e

    def remove(self, scheme_id: str) -> "LabellingCatalog":
        with self._lock:
            self._schemes.pop(scheme_id, None)
        return self

    # Browse/search
    def list(
        self,
        *,
        locale: str | None = None,
        concept: str | None = None,
        ordered: bool | None = None,
    ) -> builtins.list[CategoryScheme]:
        xs = builtins.list(self._schemes.values())
        if concept is not None:
            cpt = _norm_concept(concept)
            xs = [s for s in xs if s.concept == cpt]
        if ordered is not None:
            xs = [s for s in xs if s.ordered == ordered]
        if locale is not None:
            want = _norm_locale(locale)
            xs.sort(key=lambda s: _match_score(s.locale, want), reverse=True)
        else:
            xs.sort(key=lambda s: (s.concept, s.locale or ""))
        return xs

    def search(self, q: str) -> builtins.list[CategoryScheme]:
        ql = q.lower()
        xs = [
            s
            for s in self._schemes.values()
            if (s.id is not None and ql in s.id.lower())
            or ql in s.concept.lower()
            or (s.title and ql in s.title.lower())
            or any(ql in e.label.lower() for e in s.entries)
        ]
        xs.sort(key=lambda s: (s.concept, s.locale or ""))
        return xs

    # Pick best scheme for a concept (locale fallback)
    def pick(self, concept: str, *, locale: str | None = None) -> CategoryScheme:
        want = _norm_locale(locale if locale is not None else self._locale)
        cpt = _norm_concept(concept)
        candidates = [s for s in self._schemes.values() if s.concept == cpt]
        if not candidates:
            raise LabelError.unknown_scheme(
                where="labels.LabellingCatalog.pick",
                param="concept",
                got=concept,
            )
        return max(candidates, key=lambda s: _match_score(s.locale, want))

    # Build Labels
    def to_label(
        self, var_label: str, scheme_id: str, *, overrides: Mapping[Category, str] | None = None
    ) -> Label:
        base = self.get(scheme_id).labels
        if overrides:
            base.update(overrides)
        return Label(label=var_label, categories=base)

    def to_label_by_concept(
        self,
        var_label: str,
        concept: str,
        *,
        locale: str | None = None,
        overrides: Mapping[Category, str] | None = None,
    ) -> Label:
        s = self.pick(concept, locale=locale)
        base = s.labels
        if overrides:
            base.update(overrides)
        return Label(label=var_label, categories=base)

    # -----------------------------
    # Persistence (JSON-friendly)
    # -----------------------------

    # The bespoke encoder is gone. It existed because `mapping`, `missing` and
    # `missing_kinds` were dicts and sets keyed by Category, and JSON object
    # keys are always strings — so a scheme could only survive a round trip
    # through pairs-and-lists written by hand. One entry per code makes every
    # field JSON-native, and msgspec handles it.

    def to_bytes(self) -> bytes:
        try:
            return msgspec.json.encode(list(self._schemes.values()))
        except Exception as e:
            raise LabelError.serialization_error(
                where="labels.LabellingCatalog.to_bytes",
                reason=str(e),
            ) from e

    @classmethod
    def from_bytes(
        cls, data: bytes, *, name: str = "loaded", locale: str | None = None
    ) -> "LabellingCatalog":
        try:
            schemes = msgspec.json.decode(data, type=list[CategoryScheme])
            return cls(schemes, name=name, locale=locale)
        except Exception as e:
            raise LabelError.serialization_error(
                where="labels.LabellingCatalog.from_bytes",
                reason=str(e),
            ) from e

    def save(self, path: str) -> None:
        try:
            with open(path, "wb") as f:
                f.write(self.to_bytes())
        except Exception as e:
            raise LabelError.serialization_error(
                where="labels.LabellingCatalog.save",
                reason=str(e),
                extra={"path": path},
            ) from e

    @classmethod
    def load(
        cls, path: str, *, name: str = "loaded", locale: str | None = None
    ) -> "LabellingCatalog":
        try:
            with open(path, "rb") as f:
                data = f.read()
            return cls.from_bytes(data, name=name, locale=locale)
        except Exception as e:
            raise LabelError.serialization_error(
                where="labels.LabellingCatalog.load",
                reason=str(e),
                extra={"path": path},
            ) from e


# =============================
# Read-only view
# =============================


class SchemeCatalogView:
    def __init__(self, catalog: LabellingCatalog):
        self._c = catalog

    @property
    def locale(self):
        return self._c.locale

    def set_locale(self, locale: str | None):
        self._c.set_locale(locale)

    def list(self, **kw):
        return self._c.list(**kw)

    def search(self, q: str):
        return self._c.search(q)

    def get(self, scheme_id: str):
        return self._c.get(scheme_id)

    def pick(self, concept: str, *, locale: str | None = None):
        return self._c.pick(concept, locale=locale)

    def to_label(self, var_label: str, scheme_id: str, **kw):
        return self._c.to_label(var_label, scheme_id, **kw)

    def to_label_by_concept(self, var_label: str, concept: str, **kw):
        return self._c.to_label_by_concept(var_label, concept, **kw)


# =============================
# Missing policies & simple transforms
# =============================


def is_missing_value(
    value: Category | None,
    *,
    scheme: CategoryScheme | None,
    kinds: set[MissingKind] | None = None,
    treat_null: bool = True,
    treat_nan: bool = True,
) -> bool:
    """Test missingness with an optional policy by kind."""
    if value is None and treat_null:
        return True
    if _is_nan(value) and treat_nan:
        return True
    if not scheme:
        return False

    if kinds is None:
        return value in scheme.missing_codes

    if not scheme.missing_codes:
        return False
    if value is None:
        return False
    k = scheme.kind_of(value)
    return (k in kinds) if k is not None else False


def recode_for_analysis(
    seq: Iterable[Category | None],
    *,
    scheme: CategoryScheme | None,
    kinds: set[MissingKind] | None = None,
    treat_null: bool = True,
    treat_nan: bool = True,
) -> list[Category | None]:
    """Return a new list where selected missing codes are turned into None."""
    out: list[Category | None] = []
    for v in seq:
        out.append(
            None
            if is_missing_value(
                v, scheme=scheme, kinds=kinds, treat_null=treat_null, treat_nan=treat_nan
            )
            else v
        )
    return out


def display_text(
    value: Category | None,
    *,
    scheme: CategoryScheme | None,
    null_text: str = "",
) -> str:
    """Return display label (or empty/null_text for NA)."""
    if value is None or _is_nan(value):
        return null_text
    if scheme is not None:
        entry = scheme.entry(value)
        if entry is not None:
            return entry.label
    return str(value)


# =============================
# Optional adapters (Polars/Pandas)
# =============================


def polars_mask(col, scheme: CategoryScheme | None, kinds: set[MissingKind] | None = None):
    """
    Returns a Polars expression masking values that are missing by policy.
    Safe on non-float columns (guards is_nan via cast).
    Usage:
      df.with_columns(pl.when(polars_mask("q1", s)).then(None).otherwise(pl.col("q1")))
    """
    import polars as pl

    expr = pl.col(col) if isinstance(col, str) else col
    mask = expr.is_null()
    # Guarded NaN check via permissive cast
    mask = mask | expr.cast(pl.Float64, strict=False).is_nan()
    if scheme:
        if kinds is None and scheme.missing_codes:
            mask = mask | expr.is_in(list(scheme.missing_codes))
        elif kinds and scheme.missing_codes:
            codes = list(scheme.codes_of_kind(*kinds))
            if codes:
                mask = mask | expr.is_in(codes)
    # Null-safe: under Kleene logic `False | null = null`, and a null mask
    # entry behaves differently depending on the consumer (when() takes the
    # otherwise-branch; filter() drops the row). A null in any sub-check can
    # only arise from a null input, which IS missing-by-policy, so pin the
    # mask to True there explicitly.
    return mask.fill_null(True)


def polars_to_analysis(
    col,
    scheme: CategoryScheme | None,
    kinds: set[MissingKind] | None = None,
    alias: str | None = None,
):
    expr = pl.col(col) if isinstance(col, str) else col
    alias = alias or (col if isinstance(col, str) else None)
    return pl.when(polars_mask(expr, scheme, kinds)).then(None).otherwise(expr).alias(alias)


def polars_to_display(col, scheme: CategoryScheme | None, alias: str | None = None):
    import polars as pl

    expr = pl.col(col) if isinstance(col, str) else col
    alias = alias or (f"{col}_label" if isinstance(col, str) else "label")
    mapping = scheme.labels if scheme else {}
    return expr.replace(mapping).cast(pl.Utf8).alias(alias)
