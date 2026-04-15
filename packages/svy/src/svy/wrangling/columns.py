# src/svy/wrangling/columns.py
"""
Column operations: clean, rename, remove, keep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence, cast

import polars as pl

from svy.core.constants import SVY_HIT, SVY_PROB, SVY_ROW_INDEX, SVY_WEIGHT
from svy.engine.wrangling.cleaning import _clean_names, _rename
from svy.errors import MethodError
from svy.utils.helpers import _normalize_columns_arg
from svy.wrangling._helpers import (
    _auto_clean_design,
    _design_source_columns,
    _internal_columns,
    _rebuild_concat_columns,
    _required_columns,
    _resolve_target,
)
from svy.wrangling._naming import (
    _design_with_renamed_columns,
    _normalize_case_style,
    _normalize_letter_case,
    _update_metadata_keys,
)


if TYPE_CHECKING:
    from svy.core.sample import Sample


def clean_names(
    sample: "Sample",
    minimal: bool = False,
    remove: str | None = None,
    case_style: Literal["snake", "camel", "pascal", "kebab"] = "snake",
    letter_case: Literal["lower", "upper", "title", "original"] = "lower",
    *,
    inplace: bool = False,
) -> "Sample":
    """Standardize column names for easier downstream work."""
    _raw_data = sample._data
    _df_data: pl.DataFrame = (
        cast(pl.DataFrame, _raw_data)
        if not isinstance(_raw_data, pl.LazyFrame)
        else cast(pl.DataFrame, _raw_data.collect())
    )
    cleaned_data, renames = _clean_names(
        data=_df_data,
        minimal=minimal,
        remove=remove,
        case_style=_normalize_case_style(case_style),
        letter_case=_normalize_letter_case(letter_case),
    )

    target = _resolve_target(sample, cleaned_data, inplace=inplace)
    if renames:
        _update_metadata_keys(target, renames)
        if getattr(target, "_design", None) is not None:
            target._design = _design_with_renamed_columns(target._design, renames)
        _rebuild_concat_columns(target)
    return target


def rename_columns(
    sample: "Sample",
    renames: dict[str, str],
    *,
    inplace: bool = False,
) -> "Sample":
    """Rename columns directly."""
    if not renames:
        return sample

    forbidden = {SVY_ROW_INDEX, SVY_WEIGHT, SVY_PROB, SVY_HIT}
    if any(k in forbidden or v in forbidden for k, v in renames.items()):
        raise MethodError(
            title="Reserved column names cannot be renamed",
            detail=f"Forbidden: {sorted(forbidden)}",
            code="RENAME_FORBIDDEN",
            where="wrangling.rename_columns",
        )

    try:
        _raw2 = sample._data
        _df2: pl.DataFrame = (
            cast(pl.DataFrame, _raw2)
            if not isinstance(_raw2, pl.LazyFrame)
            else cast(pl.DataFrame, _raw2.collect())
        )
        renamed_data = _rename(_df2, renames=renames)
    except (ValueError, KeyError):
        raise
    except Exception as ex:
        raise MethodError(
            title="Rename failed",
            detail=str(ex),
            code="RENAME_FAILED",
            where="wrangling.rename_columns",
        ) from ex

    target = _resolve_target(sample, renamed_data, inplace=inplace)
    _update_metadata_keys(target, renames)

    if getattr(target, "_design", None) is not None:
        target._design = _design_with_renamed_columns(target._design, renames)

    if set(renames.keys()) & _design_source_columns(sample):
        _rebuild_concat_columns(target)

    return target


def remove_columns(
    sample: "Sample",
    columns: str | Sequence[str],
    *,
    force: bool = False,
    inplace: bool = False,
) -> "Sample":
    """Remove columns from the sample."""
    cols = _normalize_columns_arg(data=sample._data, columns=columns)

    internal = _internal_columns(sample)
    cols = [c for c in cols if c not in internal]

    if not cols:
        return sample

    protected = _required_columns(sample)
    blocked = [c for c in cols if c in protected]
    if blocked and not force:
        raise MethodError(
            title="Cannot remove design-referenced columns",
            detail=", ".join(blocked),
            code="DROP_PROTECTED_COLUMNS",
            where="wrangling.remove_columns",
            hint="Pass force=True to drop them and automatically clean the design.",
        )

    new_data = sample._data.drop(cols)
    target = _resolve_target(sample, new_data, inplace=inplace)
    if blocked:
        _auto_clean_design(target)
    return target


def keep_columns(
    sample: "Sample",
    columns: str | Sequence[str],
    *,
    force: bool = False,
    inplace: bool = False,
) -> "Sample":
    """Keep only specified columns, removing all others."""
    cols = _normalize_columns_arg(data=sample._data, columns=columns)

    internal = _internal_columns(sample)
    cols_with_internal = list(cols) + [c for c in internal if c not in cols]

    protected = _required_columns(sample)
    to_drop = [c for c in protected if c not in cols]
    if to_drop and not force:
        raise MethodError(
            title="Keeping these columns would remove design-referenced columns",
            detail=", ".join(to_drop),
            code="KEEP_DROPS_PROTECTED",
            where="wrangling.keep_columns",
            hint="Pass force=True to proceed; the design will be cleaned automatically.",
        )

    new_data = sample._data.select(cols_with_internal)
    target = _resolve_target(sample, new_data, inplace=inplace)
    if to_drop:
        _auto_clean_design(target)
    return target
