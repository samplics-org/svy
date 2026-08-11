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
- SchemeRef: Lazy reference to a catalog scheme
- ResolvedLabels: Cached, ready-to-use labels for display
- MetadataStore: Per-Sample registry with resolution and caching
"""

from __future__ import annotations

import logging
import math

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, TypeVar

import msgspec
import polars as pl

from msgspec.structs import asdict, force_setattr


if TYPE_CHECKING:
    from svy.metadata.labels import CategoryScheme, LabellingCatalog

from svy.core.enumerations import MeasurementType, MetadataSource
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
# ValueLabel: one code and its display text
# =============================================================================


class ValueLabel(msgspec.Struct, frozen=True):
    """One code and the text shown for it.

    A *pair*, not an entry in a ``dict[Category, str]``. JSON object keys are
    always strings, so a Category-keyed dict returns ``{"1": "Male"}`` for
    ``{1: "Male"}`` — silently, with no error, and every join against an
    integer-coded column then misses. Holding the code as a value keeps its
    type through serialization by any route.

    This is the same shape, and the same reason, as ``CategoryScheme``'s
    ``SchemeEntry`` and svy-spec's ``ChoiceOption``.
    """

    code: Category
    label: str


def _coerce_labels(value) -> tuple[ValueLabel, ...] | None:
    """Accept a dict, pairs, or ValueLabels; store pairs.

    The dict is authoring sugar — ``{1: "Male"}`` reads better than a list of
    structs — and never reaches serialization, so it carries none of the risk.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return tuple(ValueLabel(code=c, label=lbl) for c, lbl in value.items())
    return tuple(v if isinstance(v, ValueLabel) else ValueLabel(*v) for v in value)


_StructT = TypeVar("_StructT", bound=msgspec.Struct)


def _clone_struct(struct: _StructT, overrides: dict[str, Any]) -> _StructT:
    """Copy a frozen struct through its constructor, so ``__post_init__`` runs.

    Not ``msgspec.structs.replace``: it skips ``__post_init__`` before msgspec
    0.21, and svy supports 0.19+. The structs here normalize in
    ``__post_init__`` — a dict of value labels becomes ``ValueLabel`` pairs —
    so ``replace`` on an older msgspec stored the authoring dict verbatim and
    every later read of ``.labels`` raised ``AttributeError: 'int' object has
    no attribute 'code'``. The constructor normalizes on every version we
    support, and re-checks the invariants while it is there.
    """
    return type(struct)(**{**asdict(struct), **overrides})


# =============================================================================
# =============================================================================


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

    Examples
    --------
    >>> ref = SchemeRef(concept="agreement")
    >>> # Later, resolve against a catalog:
    >>> scheme = catalog.pick(ref.concept)
    """

    concept: str

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
        return catalog.pick(self.concept)


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
    ... )

    >>> # Variable referencing a catalog scheme
    >>> meta = VariableMeta(
    ...     name="q1",
    ...     label="How satisfied are you with the service?",
    ...     scheme_ref=SchemeRef(concept="satisfaction"),
    ...     mtype=MeasurementType.ORDINAL,
    ... )
    """

    name: str
    label: str | None = None
    #: ``(code, label)`` pairs. A dict is accepted when constructing and
    #: coerced; ``labels`` gives the mapping back for lookup. Pairs because a
    #: Category-keyed dict does not survive JSON — see ``ValueLabel``.
    value_labels: tuple[ValueLabel, ...] | None = None
    scheme_ref: SchemeRef | None = None
    mtype: MeasurementType = MeasurementType.STRING
    categories: tuple[Category, ...] | None = None
    unit: str | None = None
    notes: str | None = None
    source: MetadataSource = MetadataSource.INFERRED

    def __post_init__(self) -> None:
        """Coerce labels to pairs; a variable cannot have both sources."""
        force_setattr(self, "value_labels", _coerce_labels(self.value_labels))
        if self.value_labels is not None and self.scheme_ref is not None:
            raise ValueError(
                f"Variable {self.name!r}: cannot specify both value_labels and scheme_ref"
            )

    @property
    def labels(self) -> dict[Category, str]:
        """The value labels as a mapping, for lookup. Empty when unset."""
        return {vl.code: vl.label for vl in self.value_labels or ()}

    @property
    def has_labels(self) -> bool:
        """Check if this variable has labels (direct or by reference)."""
        return self.value_labels is not None or self.scheme_ref is not None

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
        return _clone_struct(self, overrides)

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

    def with_scheme_ref(self, concept: str, *, clear_labels: bool = True) -> VariableMeta:
        """Return a copy referencing a catalog scheme."""
        return self.clone(
            scheme_ref=SchemeRef(concept=concept),
            value_labels=None if clear_labels else self.value_labels,
            source=MetadataSource.USER,
        )

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


def _numeric_variants(value: object) -> tuple[Category, ...]:
    """The same number written the other ways a code is commonly stored."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ()
    if f != f:  # NaN
        return ()
    out: list[Category] = [f]
    if f.is_integer():
        out += [int(f), str(int(f))]
    return tuple(v for v in out if v != value)


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

    Examples
    --------
    >>> resolved = store.resolve_labels("q1")
    >>> resolved.display(1)  # "Strongly disagree"
    >>> resolved.display(99)  # "99" (no label, falls back to str)
    >>> resolved.display(None)  # ""
    """

    var_label: str = ""
    #: ``(code, label)`` pairs; ``labels`` is the mapping view. See
    #: ``ValueLabel`` for why this is not a dict.
    value_labels: tuple[ValueLabel, ...] = ()

    def __post_init__(self) -> None:
        force_setattr(self, "value_labels", _coerce_labels(self.value_labels) or ())

    @property
    def labels(self) -> dict[Category, str]:
        return {vl.code: vl.label for vl in self.value_labels}

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

        labels = self.labels
        if value in labels:
            return labels[value]

        # Codes and values can disagree in *type* without disagreeing in
        # meaning, and it happens in both directions:
        #
        #   - SPSS stores value-label keys as strings, so a `.sav` read back
        #     gives {"1": "Yes"} against a Float64 column;
        #   - a Table's `rowvals` holds stringified codes ("1") while the
        #     labels are keyed by the ints they came from ({1: "Yes"}).
        #
        # A literal lookup misses both. `display_series` never showed it
        # because it stringifies both sides before replacing.
        text = str(value)
        if text in labels:
            return labels[text]
        for key in _numeric_variants(value):
            if key in labels:
                return labels[key]
        return text

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
                .replace_strict(self.labels, default=None)
                .fill_null(pl.col(name).cast(pl.Utf8))
                .fill_null(null_text)
                .alias(name)
            ).to_series()
        else:
            result = df.select(
                pl.col(name).replace_strict(self.labels, default=null_text).alias(name)
            ).to_series()

        return result


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

    Examples
    --------
    >>> store = MetadataStore(catalog=my_catalog)
    >>> store.infer_from_dataframe(df)
    >>> store.set_label("q1", "How satisfied are you?")
    >>> store.set_scheme("q1", "satisfaction")
    >>> resolved = store.resolve_labels("q1")
    """

    __slots__ = ("_vars", "_catalog", "_resolved_cache")

    def __init__(self, catalog: LabellingCatalog | None = None):
        self._vars: dict[str, VariableMeta] = {}
        self._catalog = catalog
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

        if meta.value_labels is not None:
            value_labels = meta.labels
        elif meta.scheme_ref is not None and self._catalog is not None:
            try:
                value_labels = meta.scheme_ref.resolve(self._catalog).labels
            except Exception as e:
                log.warning(
                    f"Failed to resolve scheme {meta.scheme_ref.concept!r} "
                    f"for variable {var!r}: {e}"
                )

        resolved = ResolvedLabels(var_label=var_label, value_labels=value_labels)
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

    def set_scheme(self, var: str, concept: str) -> Self:
        """
        Link a variable to a catalog scheme.

        Creates metadata if it doesn't exist.

        Parameters
        ----------
        var : str
            Variable name.
        concept : str
            The concept identifier in the catalog.

        Returns
        -------
        Self
            For method chaining.
        """
        meta = self._vars.get(var)
        if meta is None:
            meta = VariableMeta(
                name=var,
                scheme_ref=SchemeRef(concept=concept),
                source=MetadataSource.USER,
            )
        else:
            meta = meta.with_scheme_ref(concept)
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

    def update(self, other: MetadataStore, *, overwrite: bool = False) -> Self:
        """
        Merge another store into this one, field by field.

        Metadata for one variable usually arrives from several places that each
        know a different part of it: types inferred from the data, missing-value
        codes declared by the analyst, question wording carried by an instrument
        spec. ``set`` replaces a whole ``VariableMeta``, so combining sources
        with it drops whatever the incoming record does not model. This merges
        per field instead, which means a source can only ever *add* what it
        knows — never clear what it has no opinion about.

        Parameters
        ----------
        other : MetadataStore
            The store to merge in. Left unmodified.
        overwrite : bool
            How to resolve a field both stores have set. ``False`` (the default)
            keeps this store's value, so the merge only fills gaps. ``True``
            lets `other` win — useful when it is the authority, e.g. applying an
            instrument spec whose question wording should be definitive.

            Either way a field `other` has not set is left alone, so a store
            that does not model missing codes cannot wipe them.

        Returns
        -------
        Self
            For method chaining.

        Notes
        -----
        Merging a store built from an instrument spec wants ``overwrite=True``.
        A `Sample` runs `infer_from_dataframe` at construction, which sets
        `mtype` by guessing from each column's storage type — so under the
        default that guess is already present and the spec's *declared* level
        never lands. An ordered single-select stays `NUMERICAL_DISCRETE` rather
        than becoming `CATEGORICAL_ORDINAL`. Apply the authoritative source
        first and make local adjustments after it.

        Examples
        --------
        >>> store.update(other)                    # fill gaps only
        >>> store.update(other, overwrite=True)    # `other` wins on conflicts
        """
        for var in other.variables:
            incoming = other.get(var)
            if incoming is None:  # pragma: no cover - variables() lists real keys
                continue

            existing = self._vars.get(var)
            if existing is None:
                self.set(var, incoming)
                continue

            patch = {}
            for field in msgspec.structs.fields(VariableMeta):
                if field.name == "name":
                    continue
                default = None if field.default is msgspec.NODEFAULT else field.default
                new = getattr(incoming, field.name)
                if new == default:
                    continue  # `other` has nothing to say about this field
                if overwrite or getattr(existing, field.name) == default:
                    patch[field.name] = new

            if patch:
                self._vars[var] = existing.clone(**patch)
                self._invalidate_cache(var)

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
            n_categories, source.
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
            value_labels, scheme_ref, unit,
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
                    vl_str = "; ".join(f"{k}={v}" for k, v in meta.labels.items())

                # Format categories
                cat_str = None
                if meta.categories:
                    cat_str = ", ".join(str(c) for c in meta.categories)

                # Format scheme ref
                scheme_str = None
                if meta.scheme_ref:
                    scheme_str = meta.scheme_ref.concept

                rows.append(
                    {
                        "name": name,
                        "label": meta.label,
                        "mtype": meta.mtype.value if meta.mtype else None,
                        "categories": cat_str,
                        "value_labels": vl_str,
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
            has_value_labels, source.

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
