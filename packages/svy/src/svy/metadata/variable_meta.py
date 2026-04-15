# src/svy/core/variable_meta.py
"""
Unified variable metadata system for svy.

This module provides a comprehensive metadata model that unifies:
- Variable labels (question text)
- Value labels (code → display text mappings)
- Measurement types (nominal, ordinal, continuous, etc.)
- Missing value definitions (with semantic kinds)

The core types are:
- VariableMeta: Complete metadata for a single variable
- MissingDef: Unified missing value definition
- SchemeRef: Lazy reference to a catalog scheme
- ResolvedLabels: Cached, ready-to-use labels for display
- MetadataStore: Per-Sample registry with resolution and caching
"""

from __future__ import annotations

import logging
import math

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self

import msgspec
import polars as pl

from msgspec.structs import replace


if TYPE_CHECKING:
    from svy.metadata.labels import CategoryScheme, LabellingCatalog
    from svy.questionnaire import Questionnaire

from svy.core.enumerations import MeasurementType, MetadataSource, MissingKind
from svy.core.types import Category


log = logging.getLogger(__name__)


# =============================================================================
# Helper functions
# =============================================================================


def _is_nan(x: object) -> bool:
    """Check if value is NaN (float or numpy)."""
    try:
        return math.isnan(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _normalize_codes(codes: Iterable[Category] | None) -> frozenset[Category]:
    """Convert codes iterable to frozenset, handling None."""
    if codes is None:
        return frozenset()
    return frozenset(codes)


# =============================================================================
# MissingDef: Unified missing value definition
# =============================================================================


class MissingDef(msgspec.Struct, frozen=True, eq=False):
    """
    Unified missing value definition.

    Consolidates the various missing value representations in svy into
    a single, immutable structure that supports:
    - Simple missing codes (any value treated as missing)
    - Semantic kinds (don't know, refusal, not applicable, etc.)
    - Control over null/NaN handling

    Parameters
    ----------
    codes : frozenset[Category]
        All values that should be treated as missing.
    kinds : dict[Category, MissingKind] | None
        Optional mapping of codes to their semantic meaning.
        Keys must be a subset of `codes`.
    na_is_missing : bool
        Whether to treat None/null as missing (default True).
    nan_is_missing : bool
        Whether to treat NaN as missing (default True).

    Examples
    --------
    >>> # Simple missing codes
    >>> m = MissingDef(codes=frozenset([-99, -98]))

    >>> # With semantic kinds
    >>> m = MissingDef(
    ...     codes=frozenset([-99, -98, -97]),
    ...     kinds={
    ...         -99: MissingKind.DONT_KNOW,
    ...         -98: MissingKind.REFUSAL,
    ...         -97: MissingKind.NOT_APPLICABLE,
    ...     }
    ... )

    >>> # Check if a value is missing
    >>> m.is_missing(-99)  # True
    >>> m.is_missing(1)    # False
    """

    codes: frozenset[Category] = frozenset()
    kinds: dict[Category, MissingKind] | None = None
    na_is_missing: bool = True
    nan_is_missing: bool = True

    def __post_init__(self) -> None:
        """Validate that kinds keys are subset of codes."""
        if self.kinds is not None:
            invalid = set(self.kinds.keys()) - self.codes
            if invalid:
                raise ValueError(f"missing_kinds contains codes not in codes: {invalid}")

    def __eq__(self, other: object) -> bool:
        """Custom equality that handles NaN in frozensets."""
        if not isinstance(other, MissingDef):
            return NotImplemented

        if self.na_is_missing != other.na_is_missing:
            return False
        if self.nan_is_missing != other.nan_is_missing:
            return False

        # Compare codes (need special NaN handling)
        if not self._codes_equal(self.codes, other.codes):
            return False

        # Compare kinds
        if self.kinds is None and other.kinds is None:
            return True
        if self.kinds is None or other.kinds is None:
            return False
        if set(self.kinds.keys()) != set(other.kinds.keys()):
            return False
        for k, v in self.kinds.items():
            if other.kinds.get(k) != v:
                return False

        return True

    def __hash__(self) -> int:
        """Hash that works with NaN values."""
        # Convert codes to a hashable representation
        code_tuple = tuple(sorted(str(c) for c in self.codes))
        kinds_tuple = (
            tuple(sorted((str(k), v.value) for k, v in self.kinds.items())) if self.kinds else None
        )
        return hash((code_tuple, kinds_tuple, self.na_is_missing, self.nan_is_missing))

    @staticmethod
    def _codes_equal(a: frozenset[Category], b: frozenset[Category]) -> bool:
        """Compare code sets with NaN handling."""
        if len(a) != len(b):
            return False

        a_no_nan = {x for x in a if not _is_nan(x)}
        b_no_nan = {x for x in b if not _is_nan(x)}

        if a_no_nan != b_no_nan:
            return False

        # Check NaN presence matches
        a_has_nan = any(_is_nan(x) for x in a)
        b_has_nan = any(_is_nan(x) for x in b)

        return a_has_nan == b_has_nan

    def is_missing(self, value: Category | None) -> bool:
        """
        Test if a value should be treated as missing.

        Parameters
        ----------
        value : Category | None
            The value to test.

        Returns
        -------
        bool
            True if the value is missing.
        """
        # Handle None/null
        if value is None:
            return self.na_is_missing

        # Handle NaN
        if _is_nan(value):
            return self.nan_is_missing

        # Handle explicit codes
        return value in self.codes

    def is_missing_by_kind(self, value: Category | None, *kinds: MissingKind) -> bool:
        """
        Test if a value is missing with one of the specified kinds.

        Parameters
        ----------
        value : Category | None
            The value to test.
        *kinds : MissingKind
            The kinds to check for.

        Returns
        -------
        bool
            True if value is missing and has one of the specified kinds.
        """
        if value is None or _is_nan(value):
            return False  # null/NaN don't have kinds

        if not self.kinds:
            return False

        kind = self.kinds.get(value)
        return kind in kinds if kind is not None else False

    def by_kind(self, *kinds: MissingKind) -> frozenset[Category]:
        """
        Get codes matching specific missing kinds.

        Parameters
        ----------
        *kinds : MissingKind
            The kinds to filter by.

        Returns
        -------
        frozenset[Category]
            Codes that have any of the specified kinds.

        Examples
        --------
        >>> m.by_kind(MissingKind.DONT_KNOW, MissingKind.REFUSED)
        frozenset({-99, -98})
        """
        if not self.kinds or not kinds:
            return frozenset()

        kind_set = set(kinds)
        return frozenset(code for code, kind in self.kinds.items() if kind in kind_set)

    def user_missing(self) -> frozenset[Category]:
        """
        Get codes representing user/respondent-generated missing values.

        This includes: DONT_KNOW, REFUSED, NO_ANSWER.
        """
        return self.by_kind(
            MissingKind.DONT_KNOW,
            MissingKind.REFUSED,
            MissingKind.NO_ANSWER,
        )

    def system_missing(self) -> frozenset[Category]:
        """
        Get codes representing system/design-generated missing values.

        This includes: SYSTEM, STRUCTURAL.
        """
        return self.by_kind(
            MissingKind.SYSTEM,
            MissingKind.STRUCTURAL,
        )

    def clone(self, **overrides: Any) -> MissingDef:
        """Create a copy with optional field overrides."""
        return replace(self, **overrides)

    @classmethod
    def from_codes(
        cls,
        codes: Iterable[Category],
        *,
        na_is_missing: bool = True,
        nan_is_missing: bool = True,
    ) -> MissingDef:
        """
        Create a MissingDef from simple codes (no kinds).

        Parameters
        ----------
        codes : Iterable[Category]
            Values to treat as missing.
        na_is_missing : bool
            Whether None is missing.
        nan_is_missing : bool
            Whether NaN is missing.
        """
        return cls(
            codes=frozenset(codes),
            kinds=None,
            na_is_missing=na_is_missing,
            nan_is_missing=nan_is_missing,
        )

    @classmethod
    def from_kinds(
        cls,
        kinds: Mapping[Category, MissingKind],
        *,
        na_is_missing: bool = True,
        nan_is_missing: bool = True,
    ) -> MissingDef:
        """
        Create a MissingDef from a kinds mapping.

        The codes set is automatically derived from the mapping keys.

        Parameters
        ----------
        kinds : Mapping[Category, MissingKind]
            Mapping of code → kind.
        na_is_missing : bool
            Whether None is missing.
        nan_is_missing : bool
            Whether NaN is missing.
        """
        return cls(
            codes=frozenset(kinds.keys()),
            kinds=dict(kinds),
            na_is_missing=na_is_missing,
            nan_is_missing=nan_is_missing,
        )


# =============================================================================
# SchemeRef: Lazy reference to catalog scheme
# =============================================================================


class SchemeRef(msgspec.Struct, frozen=True):
    """
    Pointer to a LabellingCatalog scheme (resolved lazily).

    This allows VariableMeta to reference reusable label schemes
    without copying the actual labels. Resolution happens at display time.

    Parameters
    ----------
    concept : str
        The concept identifier in the catalog (e.g., "agreement", "yes_no").
    locale : str | None
        Optional locale override. If None, uses the catalog's default.

    Examples
    --------
    >>> ref = SchemeRef(concept="agreement", locale="en")
    >>> # Later, resolve against a catalog:
    >>> scheme = catalog.pick(ref.concept, locale=ref.locale)
    """

    concept: str
    locale: str | None = None

    def resolve(self, catalog: LabellingCatalog) -> CategoryScheme:
        """
        Resolve this reference against a catalog.

        Parameters
        ----------
        catalog : LabellingCatalog
            The catalog to resolve against.

        Returns
        -------
        CategoryScheme
            The resolved scheme.

        Raises
        ------
        LabelError
            If the concept is not found in the catalog.
        """
        return catalog.pick(self.concept, locale=self.locale)


# =============================================================================
# VariableMeta: The unified variable metadata
# =============================================================================


class VariableMeta(msgspec.Struct, frozen=True):
    """
    Complete metadata for a single variable.

    This is the unified metadata model that consolidates:
    - Labels (variable label + value labels)
    - Measurement type
    - Categories/valid values
    - Missing value definitions
    - Additional metadata (unit, notes, etc.)

    Labels can be provided either:
    - Directly via `value_labels` dict
    - By reference via `scheme_ref` (resolved from a catalog)

    Parameters
    ----------
    name : str
        Column/variable name (stable identifier).
    label : str | None
        Human-readable variable label (e.g., question text).
    value_labels : dict[Category, str] | None
        Direct mapping of codes to display labels.
    scheme_ref : SchemeRef | None
        Reference to a catalog scheme (alternative to value_labels).
    mtype : MeasurementType
        The measurement level (nominal, ordinal, continuous, etc.).
    categories : tuple[Category, ...] | None
        Valid values for categorical variables (order matters for ordinal).
    missing : MissingDef | None
        Missing value definition.
    unit : str | None
        Unit of measurement (e.g., "kg", "years", "$").
    notes : str | None
        Free-text notes about the variable.
    source : MetadataSource
        Where this metadata came from.

    Examples
    --------
    >>> # Variable with direct labels
    >>> meta = VariableMeta(
    ...     name="gender",
    ...     label="What is your gender?",
    ...     value_labels={1: "Male", 2: "Female", 3: "Other", -99: "Prefer not to say"},
    ...     mtype=MeasurementType.NOMINAL,
    ...     categories=(1, 2, 3, -99),
    ...     missing=MissingDef.from_codes([-99]),
    ... )

    >>> # Variable referencing a catalog scheme
    >>> meta = VariableMeta(
    ...     name="q1",
    ...     label="How satisfied are you with the service?",
    ...     scheme_ref=SchemeRef(concept="satisfaction", locale="en"),
    ...     mtype=MeasurementType.ORDINAL,
    ... )
    """

    name: str
    label: str | None = None
    value_labels: dict[Category, str] | None = None
    scheme_ref: SchemeRef | None = None
    mtype: MeasurementType = MeasurementType.STRING
    categories: tuple[Category, ...] | None = None
    missing: MissingDef | None = None
    na_as_level: bool = False  # Whether to treat NA as a category level in analysis
    unit: str | None = None
    notes: str | None = None
    source: MetadataSource = MetadataSource.INFERRED

    def __post_init__(self) -> None:
        """Validate that value_labels and scheme_ref aren't both set."""
        if self.value_labels is not None and self.scheme_ref is not None:
            raise ValueError(
                f"Variable {self.name!r}: cannot specify both value_labels and scheme_ref"
            )

    @property
    def has_labels(self) -> bool:
        """Check if this variable has labels (direct or by reference)."""
        return self.value_labels is not None or self.scheme_ref is not None

    @property
    def has_missing(self) -> bool:
        """Check if this variable has missing value definitions."""
        return self.missing is not None and len(self.missing.codes) > 0

    @property
    def is_categorical(self) -> bool:
        """Check if this is a categorical variable."""
        return self.mtype in (
            MeasurementType.NOMINAL,
            MeasurementType.ORDINAL,
            MeasurementType.BOOLEAN,
        )

    @property
    def is_numeric(self) -> bool:
        """Check if this is a numeric variable."""
        return self.mtype in (
            MeasurementType.CONTINUOUS,
            MeasurementType.DISCRETE,
        )

    @property
    def is_ordered(self) -> bool:
        """Check if this is an ordered categorical variable."""
        return self.mtype == MeasurementType.ORDINAL

    def clone(self, **overrides: Any) -> VariableMeta:
        """Create a copy with optional field overrides."""
        return replace(self, **overrides)

    def with_label(self, label: str) -> VariableMeta:
        """Return a copy with updated variable label."""
        return self.clone(label=label, source=MetadataSource.USER)

    def with_value_labels(
        self, labels: dict[Category, str], *, clear_scheme: bool = True
    ) -> VariableMeta:
        """Return a copy with updated value labels."""
        return self.clone(
            value_labels=dict(labels),
            scheme_ref=None if clear_scheme else self.scheme_ref,
            source=MetadataSource.USER,
        )

    def with_scheme_ref(
        self, concept: str, locale: str | None = None, *, clear_labels: bool = True
    ) -> VariableMeta:
        """Return a copy referencing a catalog scheme."""
        return self.clone(
            scheme_ref=SchemeRef(concept=concept, locale=locale),
            value_labels=None if clear_labels else self.value_labels,
            source=MetadataSource.USER,
        )

    def with_missing(self, missing: MissingDef) -> VariableMeta:
        """Return a copy with updated missing definition."""
        return self.clone(missing=missing, source=MetadataSource.USER)

    def with_categories(
        self, categories: Iterable[Category], *, ordered: bool | None = None
    ) -> VariableMeta:
        """Return a copy with updated categories."""
        new_mtype = self.mtype
        if ordered is True:
            new_mtype = MeasurementType.ORDINAL
        elif ordered is False and self.mtype == MeasurementType.ORDINAL:
            new_mtype = MeasurementType.NOMINAL

        return self.clone(
            categories=tuple(categories),
            mtype=new_mtype,
            source=MetadataSource.USER,
        )


# =============================================================================
# ResolvedLabels: Cached, ready-to-use labels
# =============================================================================


class ResolvedLabels(msgspec.Struct, frozen=True):
    """
    Fully resolved labels ready for display.

    This is the output of resolving a VariableMeta against a catalog.
    It contains all labels in expanded form for efficient display operations.

    Parameters
    ----------
    var_label : str
        The variable label (empty string if none).
    value_labels : dict[Category, str]
        Mapping of codes to display text (empty dict if none).
    missing_codes : frozenset[Category]
        Set of codes that are missing values.

    Examples
    --------
    >>> resolved = store.resolve_labels("q1")
    >>> resolved.display(1)  # "Strongly disagree"
    >>> resolved.display(99)  # "99" (no label, falls back to str)
    >>> resolved.display(None)  # ""
    """

    var_label: str = ""
    value_labels: dict[Category, str] = msgspec.field(default_factory=dict)
    missing_codes: frozenset[Category] = frozenset()

    @property
    def has_var_label(self) -> bool:
        """Check if a variable label is present."""
        return bool(self.var_label)

    @property
    def has_value_labels(self) -> bool:
        """Check if value labels are present."""
        return bool(self.value_labels)

    def display(self, value: Category | None, null_text: str = "") -> str:
        """
        Get display text for a single value.

        Parameters
        ----------
        value : Category | None
            The value to display.
        null_text : str
            Text to show for None/null values.

        Returns
        -------
        str
            The display label, or str(value) if no label exists.
        """
        if value is None:
            return null_text

        if _is_nan(value):
            return null_text

        return self.value_labels.get(value, str(value))

    def display_series(
        self,
        s: pl.Series,
        *,
        null_text: str = "",
        unmapped_to_str: bool = True,
    ) -> pl.Series:
        """
        Apply labels to an entire Polars Series.

        Parameters
        ----------
        s : pl.Series
            The series to label.
        null_text : str
            Text to show for null values.
        unmapped_to_str : bool
            If True, convert unmapped values to strings.
            If False, leave them as-is (may cause type issues).

        Returns
        -------
        pl.Series
            A string series with labels applied.
        """
        if not self.value_labels:
            # No labels, just convert to string
            return s.cast(pl.Utf8).fill_null(null_text)

        # Apply mapping using replace
        # We need to work with a DataFrame context for complex operations
        name = s.name
        df = pl.DataFrame({name: s})

        if unmapped_to_str:
            # Replace known values, keep others as string representation
            # Use map_dict for partial replacement (returns null for unmapped)
            result = df.select(
                pl.col(name)
                .replace_strict(self.value_labels, default=None)
                .fill_null(pl.col(name).cast(pl.Utf8))
                .fill_null(null_text)
                .alias(name)
            ).to_series()
        else:
            result = df.select(
                pl.col(name).replace_strict(self.value_labels, default=null_text).alias(name)
            ).to_series()

        return result

    def is_missing(self, value: Category | None) -> bool:
        """Check if a value is a missing code."""
        if value is None:
            return True
        return value in self.missing_codes

    def non_missing_labels(self) -> dict[Category, str]:
        """Get value labels excluding missing codes."""
        return {k: v for k, v in self.value_labels.items() if k not in self.missing_codes}


# =============================================================================
# MetadataStore: Per-Sample registry
# =============================================================================


class MetadataStore:
    """
    Registry of variable metadata for a Sample.

    Handles:
    - Storage of VariableMeta per variable
    - Resolution of labels (direct or from catalog)
    - Caching of resolved labels
    - Bulk operations (inference, import)

    Parameters
    ----------
    catalog : LabellingCatalog | None
        Optional catalog for resolving scheme references.
    default_locale : str | None
        Default locale for catalog lookups.

    Examples
    --------
    >>> store = MetadataStore(catalog=my_catalog, default_locale="en")
    >>> store.infer_from_dataframe(df)
    >>> store.set_label("q1", "How satisfied are you?")
    >>> store.set_scheme("q1", "satisfaction")
    >>> resolved = store.resolve_labels("q1")
    """

    __slots__ = ("_vars", "_catalog", "_locale", "_resolved_cache")

    def __init__(
        self,
        catalog: LabellingCatalog | None = None,
        default_locale: str | None = None,
    ):
        self._vars: dict[str, VariableMeta] = {}
        self._catalog = catalog
        self._locale = default_locale
        self._resolved_cache: dict[str, ResolvedLabels] = {}

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def catalog(self) -> LabellingCatalog | None:
        """The attached label catalog."""
        return self._catalog

    @catalog.setter
    def catalog(self, value: LabellingCatalog | None) -> None:
        """Set the catalog (clears resolved cache)."""
        self._catalog = value
        self._resolved_cache.clear()

    @property
    def locale(self) -> str | None:
        """Default locale for catalog lookups."""
        return self._locale

    @locale.setter
    def locale(self, value: str | None) -> None:
        """Set default locale (clears resolved cache)."""
        self._locale = value
        self._resolved_cache.clear()

    @property
    def variables(self) -> list[str]:
        """List of all variable names with metadata."""
        return list(self._vars.keys())

    def __len__(self) -> int:
        """Number of variables with metadata."""
        return len(self._vars)

    def __contains__(self, var: str) -> bool:
        """Check if variable has metadata."""
        return var in self._vars

    def __iter__(self):
        """Iterate over variable names."""
        return iter(self._vars)

    # =========================================================================
    # Core access
    # =========================================================================

    def get(self, var: str) -> VariableMeta | None:
        """
        Get metadata for a variable.

        Parameters
        ----------
        var : str
            Variable name.

        Returns
        -------
        VariableMeta | None
            The metadata, or None if not found.
        """
        return self._vars.get(var)

    def require(self, var: str) -> VariableMeta:
        """
        Get metadata for a variable, raising if not found.

        Parameters
        ----------
        var : str
            Variable name.

        Returns
        -------
        VariableMeta
            The metadata.

        Raises
        ------
        KeyError
            If variable not found.
        """
        meta = self._vars.get(var)
        if meta is None:
            raise KeyError(f"No metadata for variable: {var!r}")
        return meta

    def set(self, var: str, meta: VariableMeta) -> None:
        """
        Set metadata for a variable.

        Parameters
        ----------
        var : str
            Variable name.
        meta : VariableMeta
            The metadata to set.

        Note
        ----
        If meta.name doesn't match var, a new VariableMeta is created
        with the correct name.
        """
        if meta.name != var:
            meta = meta.clone(name=var)
        self._vars[var] = meta
        self._invalidate_cache(var)

    def remove(self, var: str) -> VariableMeta | None:
        """
        Remove metadata for a variable.

        Parameters
        ----------
        var : str
            Variable name.

        Returns
        -------
        VariableMeta | None
            The removed metadata, or None if not found.
        """
        self._invalidate_cache(var)
        return self._vars.pop(var, None)

    def _invalidate_cache(self, var: str) -> None:
        """Invalidate the resolved cache for a variable."""
        self._resolved_cache.pop(var, None)

    # =========================================================================
    # Label resolution
    # =========================================================================

    def resolve_labels(self, var: str) -> ResolvedLabels:
        """
        Get fully resolved labels for a variable.

        Resolution order:
        1. If value_labels present, use them
        2. Elif scheme_ref present, resolve from catalog
        3. Else return empty labels

        Results are cached until the variable's metadata changes.

        Parameters
        ----------
        var : str
            Variable name.

        Returns
        -------
        ResolvedLabels
            The resolved labels (never None, may be empty).
        """
        # Check cache
        if var in self._resolved_cache:
            return self._resolved_cache[var]

        meta = self._vars.get(var)
        if meta is None:
            # No metadata - return empty
            resolved = ResolvedLabels()
            self._resolved_cache[var] = resolved
            return resolved

        var_label = meta.label or ""
        value_labels: dict[Category, str] = {}
        missing_codes: frozenset[Category] = frozenset()

        # Resolve value labels
        if meta.value_labels is not None:
            value_labels = dict(meta.value_labels)
        elif meta.scheme_ref is not None and self._catalog is not None:
            try:
                scheme = meta.scheme_ref.resolve(self._catalog)
                value_labels = dict(scheme.mapping)
                # Also get missing from scheme if not defined on meta
                if meta.missing is None and scheme.missing:
                    missing_codes = frozenset(scheme.missing)
            except Exception as e:
                log.warning(
                    f"Failed to resolve scheme {meta.scheme_ref.concept!r} "
                    f"for variable {var!r}: {e}"
                )

        # Get missing codes
        if meta.missing is not None:
            missing_codes = meta.missing.codes

        resolved = ResolvedLabels(
            var_label=var_label,
            value_labels=value_labels,
            missing_codes=missing_codes,
        )
        self._resolved_cache[var] = resolved
        return resolved

    def resolve_all(self) -> dict[str, ResolvedLabels]:
        """
        Resolve labels for all variables.

        Returns
        -------
        dict[str, ResolvedLabels]
            Mapping of variable name to resolved labels.
        """
        return {var: self.resolve_labels(var) for var in self._vars}

    # =========================================================================
    # Convenience setters
    # =========================================================================

    def set_label(self, var: str, label: str) -> Self:
        """
        Set variable label (question text).

        Creates metadata if it doesn't exist.

        Parameters
        ----------
        var : str
            Variable name.
        label : str
            The variable label.

        Returns
        -------
        Self
            For method chaining.
        """
        meta = self._vars.get(var)
        if meta is None:
            meta = VariableMeta(name=var, label=label, source=MetadataSource.USER)
        else:
            meta = meta.with_label(label)
        self.set(var, meta)
        return self

    def set_labels(self, **labels: str) -> Self:
        """
        Set multiple variable labels.

        Parameters
        ----------
        **labels : str
            Mapping of variable name to label.

        Returns
        -------
        Self
            For method chaining.

        Examples
        --------
        >>> store.set_labels(
        ...     q1="How satisfied are you?",
        ...     q2="Would you recommend us?",
        ...     age="What is your age?",
        ... )
        """
        for var, label in labels.items():
            self.set_label(var, label)
        return self

    def set_value_labels(self, var: str, labels: dict[Category, str]) -> Self:
        """
        Set value labels for a variable.

        Creates metadata if it doesn't exist.

        Parameters
        ----------
        var : str
            Variable name.
        labels : dict[Category, str]
            Mapping of codes to display text.

        Returns
        -------
        Self
            For method chaining.
        """
        meta = self._vars.get(var)
        if meta is None:
            meta = VariableMeta(
                name=var,
                value_labels=dict(labels),
                source=MetadataSource.USER,
            )
        else:
            meta = meta.with_value_labels(labels)
        self.set(var, meta)
        return self

    def set_scheme(self, var: str, concept: str, locale: str | None = None) -> Self:
        """
        Link a variable to a catalog scheme.

        Creates metadata if it doesn't exist.

        Parameters
        ----------
        var : str
            Variable name.
        concept : str
            The concept identifier in the catalog.
        locale : str | None
            Optional locale override.

        Returns
        -------
        Self
            For method chaining.
        """
        meta = self._vars.get(var)
        if meta is None:
            meta = VariableMeta(
                name=var,
                scheme_ref=SchemeRef(concept=concept, locale=locale),
                source=MetadataSource.USER,
            )
        else:
            meta = meta.with_scheme_ref(concept, locale)
        self.set(var, meta)
        return self

    def set_missing(
        self,
        var: str,
        codes: Iterable[Category] | None = None,
        *,
        # User-friendly parameter names
        dont_know: Iterable[Category] | None = None,
        refused: Iterable[Category] | None = None,
        no_answer: Iterable[Category] | None = None,
        skipped: Iterable[Category] | None = None,
        not_applicable: Iterable[Category] | None = None,
        system: Iterable[Category] | None = None,
        structural: Iterable[Category] | None = None,
        na_is_missing: bool = True,
        nan_is_missing: bool = True,
    ) -> Self:
        """
        Define missing values for a variable.

        Can specify either simple codes or codes with semantic kinds.
        User-friendly parameter names are mapped to underlying missing mechanisms:

        - dont_know → DONT_KNOW (typically MNAR)
        - refused → REFUSED (typically MNAR)
        - no_answer → NO_ANSWER (ambiguous)
        - skipped → STRUCTURAL (design-driven, typically MAR)
        - not_applicable → STRUCTURAL (design-driven, typically MAR)
        - system → SYSTEM (typically MCAR)
        - structural → STRUCTURAL (design-driven, typically MAR)

        Parameters
        ----------
        var : str
            Variable name.
        codes : Iterable[Category] | None
            Simple missing codes (no kind attached).
        dont_know : Iterable[Category] | None
            Codes meaning "don't know" → DONT_KNOW mechanism.
        refused : Iterable[Category] | None
            Codes meaning "refused to answer" → REFUSED mechanism.
        no_answer : Iterable[Category] | None
            Codes for no answer provided → NO_ANSWER mechanism.
        skipped : Iterable[Category] | None
            Codes for skipped questions (routing/skip logic) → STRUCTURAL mechanism.
        not_applicable : Iterable[Category] | None
            Codes meaning "not applicable" → STRUCTURAL mechanism.
        system : Iterable[Category] | None
            System-generated missing codes → SYSTEM mechanism.
        structural : Iterable[Category] | None
            Codes for values missing by study design → STRUCTURAL mechanism.
        na_is_missing : bool
            Whether None is treated as missing.
        nan_is_missing : bool
            Whether NaN is treated as missing.

        Returns
        -------
        Self
            For method chaining.

        Examples
        --------
        >>> # Simple codes
        >>> store.set_missing("q1", codes=[-99, -98])

        >>> # With user-friendly names (mapped to mechanisms)
        >>> store.set_missing(
        ...     "q1",
        ...     dont_know=[-99],
        ...     refused=[-98],
        ...     not_applicable=[-97],  # maps to STRUCTURAL
        ...     skipped=[-96],          # maps to STRUCTURAL
        ... )
        """
        all_codes: set[Category] = set()
        kinds: dict[Category, MissingKind] = {}

        # Add simple codes
        if codes is not None:
            all_codes.update(codes)

        # Map user-friendly names to MissingKind mechanisms
        # Note: skipped and not_applicable both map to STRUCTURAL
        kind_mapping = [
            (dont_know, MissingKind.DONT_KNOW),
            (refused, MissingKind.REFUSED),
            (no_answer, MissingKind.NO_ANSWER),
            (skipped, MissingKind.STRUCTURAL),  # user-friendly → mechanism
            (not_applicable, MissingKind.STRUCTURAL),  # user-friendly → mechanism
            (system, MissingKind.SYSTEM),
            (structural, MissingKind.STRUCTURAL),
        ]

        for code_iter, kind in kind_mapping:
            if code_iter is not None:
                for code in code_iter:
                    all_codes.add(code)
                    kinds[code] = kind

        missing_def = MissingDef(
            codes=frozenset(all_codes),
            kinds=kinds if kinds else None,
            na_is_missing=na_is_missing,
            nan_is_missing=nan_is_missing,
        )

        meta = self._vars.get(var)
        if meta is None:
            meta = VariableMeta(
                name=var,
                missing=missing_def,
                source=MetadataSource.USER,
            )
        else:
            meta = meta.with_missing(missing_def)
        self.set(var, meta)
        return self

    def set_type(self, var: str, mtype: MeasurementType) -> Self:
        """
        Set the measurement type for a variable.

        Parameters
        ----------
        var : str
            Variable name.
        mtype : MeasurementType
            The measurement type.

        Returns
        -------
        Self
            For method chaining.
        """
        meta = self._vars.get(var)
        if meta is None:
            meta = VariableMeta(name=var, mtype=mtype, source=MetadataSource.USER)
        else:
            meta = meta.clone(mtype=mtype, source=MetadataSource.USER)
        self.set(var, meta)
        return self

    def set_categories(
        self, var: str, categories: Iterable[Category], *, ordered: bool | None = None
    ) -> Self:
        """
        Set valid categories for a variable.

        Parameters
        ----------
        var : str
            Variable name.
        categories : Iterable[Category]
            The valid category values.
        ordered : bool | None
            If True, set type to ORDINAL. If False, NOMINAL.

        Returns
        -------
        Self
            For method chaining.
        """
        meta = self._vars.get(var)
        if meta is None:
            mtype = (
                MeasurementType.ORDINAL
                if ordered
                else MeasurementType.NOMINAL
                if ordered is False
                else MeasurementType.NOMINAL
            )
            meta = VariableMeta(
                name=var,
                categories=tuple(categories),
                mtype=mtype,
                source=MetadataSource.USER,
            )
        else:
            meta = meta.with_categories(categories, ordered=ordered)
        self.set(var, meta)
        return self

    def set_na_as_level(self, var: str, flag: bool = True) -> Self:
        """
        Set whether NA should be treated as a category level in analysis.

        Parameters
        ----------
        var : str
            Variable name.
        flag : bool
            If True, NA values are treated as a valid category level.

        Returns
        -------
        Self
            For method chaining.
        """
        meta = self._vars.get(var)
        if meta is None:
            meta = VariableMeta(name=var, na_as_level=flag, source=MetadataSource.USER)
        else:
            meta = meta.clone(na_as_level=flag, source=MetadataSource.USER)
        self.set(var, meta)
        return self

    def rename_variables(self, renames: Mapping[str, str]) -> Self:
        """
        Rename variables in the store.

        Updates variable keys and the `name` field in metadata.
        Clears cached resolutions for renamed variables.

        Parameters
        ----------
        renames : Mapping[str, str]
            Mapping from old names to new names.

        Returns
        -------
        Self
            For method chaining.

        Examples
        --------
        >>> store.rename_variables({"old_name": "new_name", "q1": "question_1"})
        """
        for old_name, new_name in renames.items():
            if old_name in self._vars and old_name != new_name:
                meta = self._vars.pop(old_name)
                # Create updated metadata with new name
                updated = VariableMeta(
                    name=new_name,
                    label=meta.label,
                    value_labels=meta.value_labels,
                    scheme_ref=meta.scheme_ref,
                    mtype=meta.mtype,
                    categories=meta.categories,
                    missing=meta.missing,
                    na_as_level=meta.na_as_level,
                    unit=meta.unit,
                    notes=meta.notes,
                    source=meta.source,
                )
                self._vars[new_name] = updated
                # Clear cache for old name
                self._resolved_cache.pop(old_name, None)
        return self

    # =========================================================================
    # Bulk operations
    # =========================================================================

    def infer_from_dataframe(
        self,
        df: pl.DataFrame,
        *,
        overwrite: bool = False,
        max_categories: int = 1000,
    ) -> Self:
        """
        Auto-populate metadata from a Polars DataFrame.

        Infers measurement types and categories from data types and values.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame to infer from.
        overwrite : bool
            If True, overwrite existing metadata.
            If False, only add metadata for new variables.
        max_categories : int
            Maximum unique values to store as categories.

        Returns
        -------
        Self
            For method chaining.
        """
        for name in df.columns:
            if not overwrite and name in self._vars:
                continue

            dtype = df.schema[name]
            mtype = self._infer_mtype(dtype)
            categories: tuple[Category, ...] | None = None

            # Extract categories for categorical types
            if mtype in (MeasurementType.NOMINAL, MeasurementType.ORDINAL):
                if dtype.base_type() == pl.Enum:
                    try:
                        categories = tuple(dtype.categories)  # type: ignore[union-attr]
                    except Exception:
                        pass
                elif dtype.base_type() == pl.Categorical:
                    try:
                        n_unique = df.select(pl.col(name).n_unique()).item()
                        if n_unique <= max_categories:
                            cats = df.get_column(name).unique(maintain_order=True).to_list()
                            categories = tuple(c for c in cats if c is not None)
                    except Exception:
                        pass
            elif mtype == MeasurementType.BOOLEAN:
                categories = (False, True)
            elif mtype == MeasurementType.STRING:
                # Check if low-cardinality string should be treated as nominal
                try:
                    n_unique = df.select(pl.col(name).n_unique()).item()
                    if n_unique <= max_categories:
                        cats = df.get_column(name).unique(maintain_order=True).to_list()
                        categories = tuple(c for c in cats if c is not None)
                        # Treat low-cardinality strings as nominal
                        mtype = MeasurementType.NOMINAL
                except Exception:
                    pass

            meta = VariableMeta(
                name=name,
                mtype=mtype,
                categories=categories,
                source=MetadataSource.INFERRED,
            )
            self._vars[name] = meta
            self._invalidate_cache(name)

        return self

    def _infer_mtype(self, dtype: pl.DataType) -> MeasurementType:
        """Infer MeasurementType from Polars dtype."""
        base = dtype.base_type()

        if base.is_float():
            return MeasurementType.CONTINUOUS
        if base.is_integer():
            return MeasurementType.DISCRETE
        if base == pl.Boolean:
            return MeasurementType.BOOLEAN
        if base == pl.String:
            return MeasurementType.STRING
        if base in (pl.Categorical, pl.Enum):
            return MeasurementType.NOMINAL
        if base.is_temporal():
            return MeasurementType.DATETIME

        return MeasurementType.STRING

    def import_from_schema(self, schema: Any) -> Self:
        """
        Import measurement info from a Schema object.

        Merges measurement types, categories, and units from Schema
        into existing metadata.

        Parameters
        ----------
        schema : Schema
            The schema to import from.

        Returns
        -------
        Self
            For method chaining.
        """
        # Handle the Schema from schema.py
        measurements = getattr(schema, "measurements", {})

        for name, meas in measurements.items():
            existing = self._vars.get(name)

            mtype = getattr(meas, "mtype", MeasurementType.STRING)
            categories = getattr(meas, "categories", None)
            unit = getattr(meas, "unit", None)
            notes = getattr(meas, "notes", None)

            if categories is not None:
                categories = tuple(categories)

            if existing is None:
                meta = VariableMeta(
                    name=name,
                    mtype=mtype,
                    categories=categories,
                    unit=unit,
                    notes=notes,
                    source=MetadataSource.SCHEMA,
                )
            else:
                # Merge: prefer existing labels, update type info
                meta = existing.clone(
                    mtype=mtype,
                    categories=categories if categories else existing.categories,
                    unit=unit if unit else existing.unit,
                    notes=notes if notes else existing.notes,
                    source=MetadataSource.SCHEMA,
                )

            self._vars[name] = meta
            self._invalidate_cache(name)

        return self

    def import_from_questionnaire(
        self,
        questionnaire: Questionnaire,
        catalog: LabellingCatalog | None = None,
    ) -> Self:
        """
        Import variable and value labels from a Questionnaire.

        Parameters
        ----------
        questionnaire : Questionnaire
            The questionnaire to import from.
        catalog : LabellingCatalog | None
            Catalog for resolving choice concepts.
            Falls back to self._catalog if not provided.

        Returns
        -------
        Self
            For method chaining.
        """
        cat = catalog or self._catalog

        for q in questionnaire.questions:
            existing = self._vars.get(q.name)

            # Determine measurement type from question type
            from svy.questionnaire import QuestionType

            mtype = MeasurementType.STRING
            if q.qtype == QuestionType.SINGLE:
                mtype = MeasurementType.NOMINAL
            elif q.qtype == QuestionType.MULTI:
                mtype = MeasurementType.NOMINAL
            elif q.qtype == QuestionType.BOOLEAN:
                mtype = MeasurementType.BOOLEAN
            elif q.qtype == QuestionType.NUMERIC:
                mtype = MeasurementType.CONTINUOUS
            elif q.qtype == QuestionType.DATE:
                mtype = MeasurementType.DATETIME

            # Get value labels
            value_labels: dict[Category, str] | None = None
            scheme_ref: SchemeRef | None = None
            categories: tuple[Category, ...] | None = None

            if q.choices is not None:
                if q.choices.mapping is not None:
                    value_labels = dict(q.choices.mapping)
                    categories = tuple(q.choices.mapping.keys())
                elif q.choices.concept is not None:
                    scheme_ref = SchemeRef(
                        concept=q.choices.concept,
                        locale=q.choices.locale,
                    )
                    # Try to resolve categories from catalog
                    if cat is not None:
                        try:
                            scheme = cat.pick(q.choices.concept, locale=q.choices.locale)
                            categories = tuple(scheme.mapping.keys())
                        except Exception:
                            pass

            # Build/update metadata
            if existing is None:
                meta = VariableMeta(
                    name=q.name,
                    label=q.text,
                    value_labels=value_labels,
                    scheme_ref=scheme_ref,
                    mtype=mtype,
                    categories=categories,
                    source=MetadataSource.QUESTIONNAIRE,
                )
            else:
                # Merge: questionnaire provides labels if not already set
                meta = existing.clone(
                    label=q.text if not existing.label else existing.label,
                    value_labels=value_labels
                    if not existing.value_labels
                    else existing.value_labels,
                    scheme_ref=scheme_ref if not existing.scheme_ref else existing.scheme_ref,
                    mtype=mtype,
                    categories=categories if categories else existing.categories,
                    source=MetadataSource.QUESTIONNAIRE,
                )

            self._vars[q.name] = meta
            self._invalidate_cache(q.name)

        return self

    def align_to_dataframe(self, df: pl.DataFrame) -> Self:
        """
        Sync metadata with DataFrame columns.

        - Removes metadata for columns no longer in df
        - Adds inferred metadata for new columns

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame to align with.

        Returns
        -------
        Self
            For method chaining.
        """
        df_cols = set(df.columns)
        existing_cols = set(self._vars.keys())

        # Remove metadata for dropped columns
        for col in existing_cols - df_cols:
            self.remove(col)

        # Add metadata for new columns
        new_cols = df_cols - existing_cols
        if new_cols:
            subset = df.select(list(new_cols))
            self.infer_from_dataframe(subset, overwrite=False)

        return self

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:
        n = len(self._vars)
        preview = ", ".join(list(self._vars.keys())[:5])
        if n > 5:
            preview += f", ... (+{n - 5} more)"
        return f"MetadataStore(n={n}, vars=[{preview}])"

    def summary(self, vars: str | list[str] | None = None) -> pl.DataFrame:
        """
        Return a summary DataFrame of metadata.

        Parameters
        ----------
        vars : str | list[str] | None
            Variable(s) to summarize. If None, summarize all.

        Returns
        -------
        pl.DataFrame
            Summary with columns: name, label, mtype, has_value_labels,
            has_missing, n_categories, source.
        """
        if vars is None:
            var_list = list(self._vars.keys())
        elif isinstance(vars, str):
            var_list = [vars]
        else:
            var_list = list(vars)

        rows = []
        for name in var_list:
            meta = self._vars.get(name)
            if meta is None:
                rows.append(
                    {
                        "name": name,
                        "label": None,
                        "mtype": None,
                        "has_value_labels": False,
                        "has_scheme_ref": False,
                        "has_missing": False,
                        "n_categories": None,
                        "source": None,
                    }
                )
            else:
                rows.append(
                    {
                        "name": name,
                        "label": meta.label or "",
                        "mtype": meta.mtype.value,
                        "has_value_labels": meta.value_labels is not None,
                        "has_scheme_ref": meta.scheme_ref is not None,
                        "has_missing": meta.has_missing,
                        "n_categories": len(meta.categories) if meta.categories else None,
                        "source": meta.source.value,
                    }
                )

        return pl.DataFrame(rows)

    def inspect(self, vars: str | list[str]) -> pl.DataFrame:
        """
        Return detailed metadata for one or more variables.

        Provides a comprehensive view including value labels and missing codes,
        suitable for reviewing metadata before analysis or export.

        Parameters
        ----------
        vars : str | list[str]
            Variable name(s) to inspect.

        Returns
        -------
        pl.DataFrame
            Detailed metadata with columns: name, label, mtype, categories,
            value_labels, missing_codes, missing_kinds, scheme_ref, unit,
            notes, source.

        Examples
        --------
        >>> store.inspect("q1")
        >>> store.inspect(["q1", "q2", "age"])
        """
        if isinstance(vars, str):
            var_list = [vars]
        else:
            var_list = list(vars)

        rows = []
        for name in var_list:
            meta = self._vars.get(name)
            if meta is None:
                rows.append(
                    {
                        "name": name,
                        "label": None,
                        "mtype": None,
                        "categories": None,
                        "value_labels": None,
                        "missing_codes": None,
                        "missing_kinds": None,
                        "scheme_ref": None,
                        "unit": None,
                        "notes": None,
                        "source": None,
                    }
                )
            else:
                # Format value labels as string for display
                vl_str = None
                if meta.value_labels:
                    vl_str = "; ".join(f"{k}={v}" for k, v in meta.value_labels.items())

                # Format categories
                cat_str = None
                if meta.categories:
                    cat_str = ", ".join(str(c) for c in meta.categories)

                # Format missing
                missing_codes_str = None
                missing_kinds_str = None
                if meta.missing:
                    if meta.missing.codes:
                        missing_codes_str = ", ".join(
                            str(c)
                            for c in sorted(meta.missing.codes, key=str)  # type: ignore[type-var]
                        )
                    if meta.missing.kinds:
                        missing_kinds_str = "; ".join(
                            f"{k}={v.value}" for k, v in meta.missing.kinds.items()
                        )

                # Format scheme ref
                scheme_str = None
                if meta.scheme_ref:
                    scheme_str = meta.scheme_ref.concept
                    if meta.scheme_ref.locale:
                        scheme_str += f" ({meta.scheme_ref.locale})"

                rows.append(
                    {
                        "name": name,
                        "label": meta.label,
                        "mtype": meta.mtype.value if meta.mtype else None,
                        "categories": cat_str,
                        "value_labels": vl_str,
                        "missing_codes": missing_codes_str,
                        "missing_kinds": missing_kinds_str,
                        "scheme_ref": scheme_str,
                        "unit": meta.unit,
                        "notes": meta.notes,
                        "source": meta.source.value if meta.source else None,
                    }
                )

        return pl.DataFrame(rows)

    def coverage(self, data: pl.DataFrame | None = None) -> pl.DataFrame:
        """
        Show metadata coverage relative to data columns.

        Parameters
        ----------
        data : pl.DataFrame | None
            DataFrame to check against. If None, only shows metadata info.

        Returns
        -------
        pl.DataFrame
            Coverage report with columns: name, in_data, has_label,
            has_value_labels, has_missing, source.

        Examples
        --------
        >>> store.coverage(sample.data)
        """
        data_cols = set(data.columns) if data is not None else set()
        all_vars = set(self._vars.keys()) | data_cols

        rows = []
        for name in sorted(all_vars):
            meta = self._vars.get(name)
            rows.append(
                {
                    "name": name,
                    "in_data": name in data_cols if data is not None else None,
                    "in_metadata": name in self._vars,
                    "has_label": meta.label is not None if meta else False,
                    "has_value_labels": meta.value_labels is not None if meta else False,
                    "has_missing": meta.has_missing if meta else False,
                    "source": meta.source.value if meta else None,
                }
            )

        return pl.DataFrame(rows)

    def unlabeled(self, data: pl.DataFrame) -> list[str]:
        """
        Return variable names in data that have no variable label.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame to check.

        Returns
        -------
        list[str]
            Column names without labels.

        Examples
        --------
        >>> store.unlabeled(sample.data)
        ['id', 'weight', 'temp_col']
        """
        result = []
        for col in data.columns:
            meta = self._vars.get(col)
            if meta is None or meta.label is None:
                result.append(col)
        return result

    def orphaned(self, data: pl.DataFrame) -> list[str]:
        """
        Return variable names with metadata but not in data.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame to check.

        Returns
        -------
        list[str]
            Variable names in metadata but not in data.

        Examples
        --------
        >>> store.orphaned(sample.data)
        ['q99', 'old_variable']
        """
        data_cols = set(data.columns)
        return [name for name in self._vars if name not in data_cols]
