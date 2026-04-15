# src/svy/wrangling/base.py
"""
Wrangling facade.

The Wrangling class is a thin dispatcher.  Each method is a one-liner
that delegates to the corresponding module-level function.  No logic
lives here.

Adding a new wrangling method:
  1. Implement it in svy/wrangling/<topic>.py as a module function.
  2. Import it here and add a one-liner delegation method.
  3. If it also has a pure algorithm, put that in
     svy/engine/wrangling/<algo>.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import polars as pl

from svy.core.types import MutateValue, WhereArg
from svy.wrangling.columns import clean_names as _clean_names
from svy.wrangling.columns import keep_columns as _keep_columns
from svy.wrangling.columns import remove_columns as _remove_columns
from svy.wrangling.columns import rename_columns as _rename_columns
from svy.wrangling.labels import apply_labels as _apply_labels
from svy.wrangling.mutate import mutate as _mutate
from svy.wrangling.rows import distinct as _distinct
from svy.wrangling.rows import filter_records as _filter_records
from svy.wrangling.rows import order_by as _order_by
from svy.wrangling.rows import with_row_index as _with_row_index
from svy.wrangling.values import bottom_and_top_code as _bottom_and_top_code
from svy.wrangling.values import bottom_code as _bottom_code
from svy.wrangling.values import cast_columns as _cast_columns
from svy.wrangling.values import categorize as _categorize
from svy.wrangling.values import fill_null as _fill_null
from svy.wrangling.values import recode as _recode
from svy.wrangling.values import top_code as _top_code


if TYPE_CHECKING:
    from svy.core.sample import Sample


class Wrangling:
    """
    Data wrangling operations for survey samples.

    Accessed via ``sample.wrangling`` attribute.  Provides methods for
    cleaning, transforming, and preparing survey data while maintaining
    design metadata consistency.

    All methods return a **new** Sample by default (copy-on-write),
    leaving the original unchanged.  Pass ``inplace=True`` to mutate
    the original Sample instead.

    Examples
    --------
    >>> sample.wrangling.clean_names().wrangling.mutate({"x": 1})
    """

    def __init__(self, sample: "Sample") -> None:
        self._sample = sample

    # ------------------------------------------------------------------ #
    # Column Operations
    # ------------------------------------------------------------------ #

    def clean_names(
        self,
        minimal: bool = False,
        remove: str | None = None,
        case_style: Literal["snake", "camel", "pascal", "kebab"] = "snake",
        letter_case: Literal["lower", "upper", "title", "original"] = "lower",
        *,
        inplace: bool = False,
    ) -> "Sample":
        """Standardize column names for easier downstream work."""
        return _clean_names(
            self._sample,
            minimal=minimal,
            remove=remove,
            case_style=case_style,
            letter_case=letter_case,
            inplace=inplace,
        )

    def rename_columns(
        self,
        renames: dict[str, str],
        *,
        inplace: bool = False,
    ) -> "Sample":
        """Rename columns directly."""
        return _rename_columns(self._sample, renames, inplace=inplace)

    def remove_columns(
        self,
        columns: str | Sequence[str],
        *,
        force: bool = False,
        inplace: bool = False,
    ) -> "Sample":
        """Remove columns from the sample."""
        return _remove_columns(self._sample, columns, force=force, inplace=inplace)

    def drop(
        self,
        columns: str | Sequence[str],
        *,
        force: bool = False,
        inplace: bool = False,
    ) -> "Sample":
        """Alias for :meth:`remove_columns`."""
        return _remove_columns(self._sample, columns, force=force, inplace=inplace)

    def keep_columns(
        self,
        columns: str | Sequence[str],
        *,
        force: bool = False,
        inplace: bool = False,
    ) -> "Sample":
        """Keep only specified columns, removing all others."""
        return _keep_columns(self._sample, columns, force=force, inplace=inplace)

    def select(
        self,
        columns: str | Sequence[str],
        *,
        force: bool = False,
        inplace: bool = False,
    ) -> "Sample":
        """Alias for :meth:`keep_columns`."""
        return _keep_columns(self._sample, columns, force=force, inplace=inplace)

    # ------------------------------------------------------------------ #
    # Value Transformation
    # ------------------------------------------------------------------ #

    def top_code(
        self,
        top_codes: Mapping[str, float],
        *,
        replace: bool = False,
        into: str | Mapping[str, str] | None = None,
        inplace: bool = False,
    ) -> "Sample":
        """Cap values at upper bounds (top coding)."""
        return _top_code(
            self._sample,
            top_codes,
            replace=replace,
            into=into,
            inplace=inplace,
        )

    def bottom_code(
        self,
        bottom_codes: Mapping[str, float],
        *,
        replace: bool = False,
        into: str | Mapping[str, str] | None = None,
        inplace: bool = False,
    ) -> "Sample":
        """Cap values at lower bounds (bottom coding)."""
        return _bottom_code(
            self._sample,
            bottom_codes,
            replace=replace,
            into=into,
            inplace=inplace,
        )

    def bottom_and_top_code(
        self,
        bottom_and_top_codes: Mapping[str, tuple[float, float] | list[float]],
        *,
        replace: bool = False,
        into: str | Mapping[str, str] | None = None,
        inplace: bool = False,
    ) -> "Sample":
        """Cap values at both lower and upper bounds."""
        return _bottom_and_top_code(
            self._sample,
            bottom_and_top_codes,
            replace=replace,
            into=into,
            inplace=inplace,
        )

    def recode(
        self,
        cols: str | list[str],
        recodes: Mapping[Any, Sequence[Any]],
        *,
        replace: bool = False,
        into: Mapping[str, str] | str | None = None,
        inplace: bool = False,
    ) -> "Sample":
        """Map old values to new labels."""
        return _recode(
            self._sample,
            cols,
            recodes,
            replace=replace,
            into=into,
            inplace=inplace,
        )

    def categorize(
        self,
        col: str,
        bins: list[float] | None = None,
        labels: list[str] | None = None,
        *,
        percentiles: int | tuple[float, ...] | None = None,
        right: bool = True,
        replace: bool = False,
        into: str | None = None,
        inplace: bool = False,
    ) -> "Sample":
        """Bin continuous values into labeled categories."""
        return _categorize(
            self._sample,
            col,
            bins=bins,
            labels=labels,
            percentiles=percentiles,
            right=right,
            replace=replace,
            into=into,
            inplace=inplace,
        )

    # ------------------------------------------------------------------ #
    # Data Type Operations
    # ------------------------------------------------------------------ #

    def cast(
        self,
        cols: str | Sequence[str] | Mapping[str, pl.DataType],
        dtype: pl.DataType | None = None,
        *,
        strict: bool = True,
        inplace: bool = False,
    ) -> "Sample":
        """Cast columns to specified data type(s)."""
        return _cast_columns(
            self._sample,
            cols,
            dtype=dtype,
            strict=strict,
            inplace=inplace,
        )

    def fill_null(
        self,
        cols: str | Sequence[str],
        value: Any = None,
        *,
        strategy: Literal["forward", "backward", "mean", "min", "max", "zero", "one"]
        | None = None,
        inplace: bool = False,
    ) -> "Sample":
        """Fill null values in specified columns."""
        return _fill_null(
            self._sample,
            cols,
            value=value,
            strategy=strategy,
            inplace=inplace,
        )

    # ------------------------------------------------------------------ #
    # Row Operations
    # ------------------------------------------------------------------ #

    def filter_records(
        self,
        where: WhereArg | None = None,
        *,
        negate: bool = False,
        check_singletons: bool = False,
        on_singletons: Literal["ignore", "warn", "error"] = "ignore",
        inplace: bool = False,
    ) -> "Sample":
        """Filter rows based on conditions."""
        return _filter_records(
            self._sample,
            where,
            negate=negate,
            check_singletons=check_singletons,
            on_singletons=on_singletons,
            inplace=inplace,
        )

    def order_by(
        self,
        cols: str | Sequence[str],
        *,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = True,
        inplace: bool = False,
    ) -> "Sample":
        """Sort rows by one or more columns."""
        return _order_by(
            self._sample,
            cols,
            descending=descending,
            nulls_last=nulls_last,
            inplace=inplace,
        )

    def distinct(
        self,
        cols: str | Sequence[str] | None = None,
        *,
        keep: Literal["first", "last", "any", "none"] = "first",
        maintain_order: bool = True,
        inplace: bool = False,
    ) -> "Sample":
        """Remove duplicate rows."""
        return _distinct(
            self._sample,
            cols,
            keep=keep,
            maintain_order=maintain_order,
            inplace=inplace,
        )

    def with_row_index(
        self,
        name: str = "row_index",
        offset: int = 0,
        *,
        inplace: bool = False,
    ) -> "Sample":
        """Add a row index column."""
        return _with_row_index(self._sample, name=name, offset=offset, inplace=inplace)

    # ------------------------------------------------------------------ #
    # Mutate
    # ------------------------------------------------------------------ #

    def mutate(
        self,
        specs: Mapping[str, MutateValue],
        *,
        inplace: bool = False,
    ) -> "Sample":
        """Create or transform columns using expressions, scalars, or arrays."""
        return _mutate(self._sample, specs, inplace=inplace)

    # ------------------------------------------------------------------ #
    # Labels
    # ------------------------------------------------------------------ #

    def apply_labels(
        self,
        labels: dict[str, str] | None = None,
        categories: dict[str, dict[Any, str]] | None = None,
        *,
        strict: bool = True,
        overwrite: bool = True,
        inplace: bool = False,
    ) -> "Sample":
        """Apply variable labels and/or value labels."""
        return _apply_labels(
            self._sample,
            labels=labels,
            categories=categories,
            strict=strict,
            overwrite=overwrite,
            inplace=inplace,
        )
