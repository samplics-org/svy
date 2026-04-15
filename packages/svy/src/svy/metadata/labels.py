# src/svy/metadata/labels.py
from __future__ import annotations

import builtins
import logging
import math
import threading

from typing import Any, Iterable, Mapping, Self

import msgspec
import polars as pl

from msgspec.structs import replace

from svy.core.enumerations import MissingKind
from svy.core.types import Category, _MissingType
from svy.errors.label_errors import LabelError


log = logging.getLogger(__name__)
# ============================= # Labels & Schemes # =============================


class Label(msgspec.Struct, frozen=True):
    """Variable label + optional value labels (code -> text).

    Notes
    -----
    Label/value-labels are intended for variables measured as
    NOMINAL, ORDINAL, or BOOLEAN (see MeasurementType).
    """

    label: str
    categories: dict[Category, str] | None | _MissingType = None

    def clone(self, **overrides) -> Self:
        return replace(self, **overrides)


class CategoryScheme(msgspec.Struct, kw_only=True, frozen=True):
    """
    One value-label scheme for a given (concept, locale), with optional missing semantics.

    JSON Persistence Note
    ---------------------
    JSON object keys must be strings, and sets are not JSON-native. The catalog
    serializes schemes using a custom encoder (pairs for mappings; lists for sets),
    see LabellingCatalog.to_bytes/from_bytes.
    """

    concept: str
    mapping: dict[Category, str]
    id: str | None = None
    locale: str | None = None
    title: str | None = None
    ordered: bool = False

    # Missing semantics
    missing: set[Category] | None = None
    missing_kinds: dict[Category, MissingKind] | None = None

    def __post_init__(self):
        # Auto-generate id if not provided
        if self.id is None:
            generated_id = f"{self.concept}:{self.locale or 'default'}"
            # For frozen structs, use object.__setattr__
            object.__setattr__(self, "id", generated_id)

    def clone(self, **overrides) -> Self:
        return replace(self, **overrides)


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
    """
    Ensures:
      - No NaN keys in mapping (brittle as dict keys).
      - missing ⊆ mapping.keys()
      - missing_kinds keys ⊆ mapping.keys()
      - If both provided, missing_kinds.keys() ⊆ missing
    """
    keys = set(s.mapping.keys())

    for k in keys:
        if _is_nan(k):
            raise LabelError.nan_key_forbidden(where="labels.validate_scheme_missing")

    if s.missing is not None:
        diff = set(s.missing) - keys
        if diff and strict:
            raise LabelError.invalid_missing_codes(
                where="labels.validate_scheme_missing",
                param="missing",
                not_in_mapping=sorted(diff),
            )

    if s.missing_kinds is not None:
        kkeys = set(s.missing_kinds.keys())
        diff = kkeys - keys
        if diff and strict:
            raise LabelError.invalid_missing_codes(
                where="labels.validate_scheme_missing",
                param="missing_kinds",
                not_in_mapping=sorted(diff),
            )
        if s.missing is not None:
            diff2 = kkeys - set(s.missing)
            if diff2 and strict:
                raise LabelError.inconsistent_missing_kinds(
                    where="labels.validate_scheme_missing",
                    offending_keys=sorted(diff2),
                )


def normalize_scheme_missing(s: CategoryScheme) -> CategoryScheme:
    """If only missing_kinds given, derive missing as its keys."""
    if s.missing is None and s.missing_kinds:
        return CategoryScheme(
            id=s.id,
            concept=s.concept,
            mapping=s.mapping,
            locale=s.locale,
            title=s.title,
            ordered=s.ordered,
            missing=set(s.missing_kinds.keys()),
            missing_kinds=s.missing_kinds,
        )
    return s


def missing_codes_by_kind(s: CategoryScheme, kinds: set[MissingKind]) -> set[Category]:
    """Collect codes matching any of the requested kinds."""
    if not s.missing_kinds:
        return set()
    return {code for code, mk in s.missing_kinds.items() if mk in kinds}


# =============================
# Scheme factory
# =============================


def make_scheme(
    *,
    concept: str,
    mapping: Mapping[Category, str],
    locale: str | None = None,
    title: str | None = None,
    ordered: bool = False,
    missing: set[Category] | None = None,
    missing_kinds: dict[Category, MissingKind] | None = None,
    id: str | None = None,
) -> CategoryScheme:
    """
    Factory to create a CategoryScheme with a predictable id (concept:locale).
    Pass an explicit id to override.
    """
    concept_key = _norm_concept(concept)
    loc = _norm_locale(locale)
    sid = id if id is not None else (f"{concept_key}:{loc}" if loc else concept_key)
    scheme = CategoryScheme(
        id=sid,
        concept=concept_key,
        mapping=dict(mapping),
        locale=loc,
        title=title,
        ordered=ordered,
        missing=missing,
        missing_kinds=missing_kinds,
    )
    # Strict validation to catch issues early
    validate_scheme_missing(scheme, strict=True)
    return normalize_scheme_missing(scheme)


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
            or any(ql in v.lower() for v in s.mapping.values())
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
        base = dict(self.get(scheme_id).mapping)
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
        base = dict(s.mapping)
        if overrides:
            base.update(overrides)
        return Label(label=var_label, categories=base)

    # -----------------------------
    # Persistence (JSON-friendly)
    # -----------------------------

    @staticmethod
    def _scheme_to_jsonable(s: CategoryScheme) -> dict[str, Any]:
        return {
            "id": s.id,
            "concept": s.concept,
            "mapping_pairs": [[k, v] for k, v in s.mapping.items()],
            "locale": s.locale,
            "title": s.title,
            "ordered": s.ordered,
            "missing_list": list(s.missing) if s.missing is not None else None,
            "missing_kind_pairs": (
                [[k, mk.value] for k, mk in s.missing_kinds.items()] if s.missing_kinds else None
            ),
        }

    @staticmethod
    def _scheme_from_jsonable(d: Mapping[str, Any]) -> CategoryScheme:
        mapping = {k: v for k, v in d.get("mapping_pairs") or []}
        missing_set = set(d.get("missing_list") or []) or None
        mk_pairs = d.get("missing_kind_pairs") or []
        missing_kinds = {k: MissingKind(mv) for k, mv in mk_pairs} if mk_pairs else None
        sch = CategoryScheme(
            id=d["id"],
            concept=d["concept"],
            mapping=mapping,
            locale=d.get("locale"),
            title=d.get("title"),
            ordered=bool(d.get("ordered", False)),
            missing=missing_set,
            missing_kinds=missing_kinds,
        )
        # Validate on load as well
        validate_scheme_missing(sch, strict=True)
        return normalize_scheme_missing(sch)

    def to_bytes(self) -> bytes:
        try:
            payload = [self._scheme_to_jsonable(s) for s in self._schemes.values()]
            return msgspec.json.encode(payload)
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
            raw = msgspec.json.decode(data, type=list[dict[str, Any]])
            schemes = [cls._scheme_from_jsonable(d) for d in raw]
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
        return bool(scheme.missing and (value in scheme.missing))

    if not scheme.missing_kinds:
        return False
    if value is None:
        return False
    k = scheme.missing_kinds.get(value)
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
    if scheme and value in scheme.mapping:
        return scheme.mapping[value]
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
        if kinds is None and scheme.missing:
            mask = mask | expr.is_in(list(scheme.missing))
        elif kinds and scheme.missing_kinds:
            codes = [code for code, mk in scheme.missing_kinds.items() if mk in kinds]
            if codes:
                mask = mask | expr.is_in(codes)
    return mask


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
    mapping = scheme.mapping if scheme else {}
    return expr.replace(mapping).cast(pl.Utf8).alias(alias)
