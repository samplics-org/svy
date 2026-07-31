# src/svy/metadata/labels.py
from __future__ import annotations

import builtins
import logging
import math
import threading

from typing import Iterable, Mapping, Self

import msgspec

from msgspec.structs import force_setattr, replace

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
    """One category: a code and what to call it.

    Two fields, because svy labels values so results print nicely and that is
    the whole job (design 001 §2.2). A code's parent belongs to a cascading
    choice list, and whether a code is a non-answer is a questionnaire fact;
    both live in svy-spec, which is where anything can act on them.
    """

    code: Category
    label: str


class CategoryScheme(msgspec.Struct, kw_only=True, frozen=True):
    """A reusable code→label map, registered under a concept.

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

    A dict is accepted when constructing, since ``{101: "Banjul"}`` reads better
    than a list of structs::

        CategoryScheme(concept="sex", entries={1: "Male", 2: "Female"})

    There is no ``locale``. A label is a string: write "Femme" and svy prints
    "Femme". What svy does not do is *switch* between languages — that is
    translation, and it lives in svy-spec. Registering ``sex_en`` and ``sex_fr``
    as two concepts does everything a locale field did, without a matching
    algorithm (§2.2).

    There is no ``ordered`` either: order lives in the codes, and "is this
    ordinal" is ``VariableMeta.mtype``.
    """

    concept: str
    entries: tuple[SchemeEntry, ...] = ()
    title: str | None = None

    def __post_init__(self):
        # force_setattr, not object.__setattr__: the latter raises "can't apply
        # this __setattr__" on a frozen msgspec Struct under 3.11 and 3.12, and
        # works on 3.13+, so a matrix is the only thing that catches it.
        force_setattr(self, "entries", _coerce_entries(self.entries))

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
        """Every code and its label."""
        return {e.code: e.label for e in self.entries}

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


def _norm_concept(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _is_nan(x: object) -> bool:
    try:
        return math.isnan(x)  # type: ignore[arg-type]
    except Exception:
        return False


# =============================
# Validation & normalization
# =============================


# =============================
# Scheme factory
# =============================


def make_scheme(
    *,
    concept: str,
    mapping: Mapping[Category, str] | None = None,
    entries: Iterable[SchemeEntry] | Mapping[Category, str] | None = None,
    title: str | None = None,
) -> CategoryScheme:
    """Build a scheme, normalising the concept the way the catalogue keys it.

    ``mapping`` and ``entries`` are the same argument under two names, since the
    former reads better for a plain code→label list.
    """
    source = entries if entries is not None else (mapping or {})
    return CategoryScheme(
        concept=_norm_concept(concept),
        entries=_coerce_entries(source),
        title=title,
    )


# =============================
# Catalog (concept-keyed, chainable)
# =============================


class LabellingCatalog:
    """Catalogue of reusable value-label schemes, keyed by concept (thread-safe).

    Register one "yes/no" scheme and point thirty variables at it. Concepts are
    the key: there is no separate scheme id, because the id only ever existed to
    disambiguate one concept's locales, and svy does not switch languages
    (§2.2).

    Notes
    -----
    Intended for variables measured as NOMINAL, ORDINAL, or BOOLEAN.
    """

    def __init__(self, schemes: Iterable[CategoryScheme] = (), name: str = "default"):
        self._name = name
        self._lock = threading.RLock()
        self._schemes: dict[str, CategoryScheme] = {s.concept: s for s in schemes}

    # CRUD (chainable)
    def register(self, scheme: CategoryScheme, *, overwrite: bool = False) -> "LabellingCatalog":
        with self._lock:
            if not overwrite and scheme.concept in self._schemes:
                raise LabelError.scheme_exists(
                    where="labels.LabellingCatalog.register",
                    scheme_id=scheme.concept,
                )
            self._schemes[scheme.concept] = scheme
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
        title: str | None = None,
        overwrite: bool = False,
    ) -> "LabellingCatalog":
        """High-level convenience: build a scheme from kwargs and register it."""
        scheme = make_scheme(concept=concept, mapping=mapping, title=title)
        return self.register(scheme, overwrite=overwrite)

    def add_schemes(self, *defs: dict, overwrite: bool = False) -> "LabellingCatalog":
        """Batch add from dictionaries of kwargs accepted by add_scheme."""
        for d in defs:
            self.add_scheme(**d, overwrite=overwrite)
        return self

    def get(self, concept: str) -> CategoryScheme:
        try:
            return self._schemes[_norm_concept(concept)]
        except KeyError as e:
            raise LabelError.unknown_scheme(
                where="labels.LabellingCatalog.get",
                param="concept",
                got=concept,
            ) from e

    def remove(self, concept: str) -> "LabellingCatalog":
        with self._lock:
            self._schemes.pop(_norm_concept(concept), None)
        return self

    # Browse/search
    def list(self, *, concept: str | None = None) -> builtins.list[CategoryScheme]:
        xs = builtins.list(self._schemes.values())
        if concept is not None:
            cpt = _norm_concept(concept)
            xs = [s for s in xs if s.concept == cpt]
        xs.sort(key=lambda s: s.concept)
        return xs

    def search(self, q: str) -> builtins.list[CategoryScheme]:
        ql = q.lower()
        xs = [
            s
            for s in self._schemes.values()
            if ql in s.concept.lower()
            or (s.title and ql in s.title.lower())
            or any(ql in e.label.lower() for e in s.entries)
        ]
        xs.sort(key=lambda s: s.concept)
        return xs

    #: One concept, one scheme, so picking is a lookup. Kept as a distinct name
    #: because it is the method svy-spec's catalog protocol calls.
    def pick(self, concept: str) -> CategoryScheme:
        return self.get(concept)

    # Build Labels
    def to_label(
        self, var_label: str, concept: str, *, overrides: Mapping[Category, str] | None = None
    ) -> Label:
        base = self.get(concept).labels
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
    def from_bytes(cls, data: bytes, *, name: str = "loaded") -> "LabellingCatalog":
        try:
            schemes = msgspec.json.decode(data, type=list[CategoryScheme])
            return cls(schemes, name=name)
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
    def load(cls, path: str, *, name: str = "loaded") -> "LabellingCatalog":
        try:
            with open(path, "rb") as f:
                data = f.read()
            return cls.from_bytes(data, name=name)
        except Exception as e:
            raise LabelError.serialization_error(
                where="labels.LabellingCatalog.load",
                reason=str(e),
                extra={"path": path},
            ) from e


# =============================
# Read-only view
# =============================


# =============================
# Missing policies & simple transforms
# =============================


# =============================
# Optional adapters (Polars/Pandas)
# =============================
