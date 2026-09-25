# src/svy/wrangling/_naming.py
"""
Column-naming utilities for the wrangling namespace.

Pure functions for case/style normalisation, design-column renaming,
and metadata-key updates.  Nothing here imports from Sample directly.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING, Literal, Sequence

from svy.core import design_parts as _dp
from svy.core.design import Design, PopSize

# Kept importable for svy.engine.wrangling.cleaning.
from svy.core.design_parts import _rep_wgts_with_renames as _rep_wgts_with_renames  # noqa: F401
from svy.core.enumerations import (
    CaseStyle as _CaseStyle,
)
from svy.core.enumerations import (
    LetterCase as _LetterCase,
)
from svy.errors import LabelError


if TYPE_CHECKING:
    from svy.core.sample import Sample


# -------------------------------------------------------------------
# NaN-key guard
# -------------------------------------------------------------------


def _check_nan_keys(mapping: dict, *, where: str, var: str | None = None) -> None:
    """Raise LabelError if any dict key is NaN."""
    for k in mapping.keys():
        try:
            if isinstance(k, float) and math.isnan(k):
                where_tag = f"{where} ({var})" if var else where
                raise LabelError.nan_key_forbidden(where=where_tag)
        except TypeError:
            pass


# -------------------------------------------------------------------
# Design-field renaming helpers
# -------------------------------------------------------------------


def _map_name_in_design(design_field: str | None, renames: dict[str, str]) -> str | None:
    """Map a single string design field through *renames*."""
    return (
        renames.get(design_field, design_field) if isinstance(design_field, str) else design_field
    )


def _map_tuple_in_design(
    design_field: str | tuple[str, ...] | None, renames: dict[str, str]
) -> str | tuple[str, ...] | None:
    """Map a string-or-tuple design field through *renames*."""
    if design_field is None or isinstance(design_field, str):
        return _map_name_in_design(design_field, renames)
    return tuple(renames.get(s, s) for s in design_field)


def _pop_size_with_renames(pop_size, renames: dict[str, str]):
    if isinstance(pop_size, PopSize):
        return PopSize(
            psu=renames.get(pop_size.psu, pop_size.psu),
            ssu=None if pop_size.ssu is None else renames.get(pop_size.ssu, pop_size.ssu),
        )
    return _map_name_in_design(pop_size, renames)


def _design_with_renamed_columns(
    design: Design,
    renames: dict[str, str],
    data_columns: Sequence[str] | None = None,
) -> Design:
    """Return a new Design with all column references updated by *renames*.

    Covers the design fields and every design part (``DesignPart.renamed``):
    the replicate columns and units, the weight the replicates go with, the
    weight-adjustment record. Parts are passed explicitly, so they are carried
    rather than re-decided by the update rules.
    """
    if not renames:
        return design

    parts = {
        part.name: None if value is None else part.renamed(value, renames, data_columns)
        for part in _dp.registered()
        for value in (design._parts.get(part.name),)
    }
    return design.update(
        case_id=_map_name_in_design(design.case_id, renames),
        wave=_map_name_in_design(design.wave, renames),
        stratum=_map_tuple_in_design(design.stratum, renames),
        wgt=_map_name_in_design(design.wgt, renames),
        prob=_map_name_in_design(design.prob, renames),
        hit=_map_name_in_design(design.hit, renames),
        mos=_map_name_in_design(design.mos, renames),
        psu=_map_tuple_in_design(design.psu, renames),
        ssu=_map_tuple_in_design(design.ssu, renames),
        pop_size=_pop_size_with_renames(design.pop_size, renames),
        **parts,
    )


def _history_with_renamed_columns(
    sample: "Sample",
    renames: dict[str, str],
    data_columns: Sequence[str] | None = None,
) -> None:
    """Carry a rename into the sample's earlier designs.

    A rename relabels a column without changing it, so an earlier design that
    read the column still describes it under the new name, and restoring it
    from history (``update_design(wgt=...)``) keeps working. An entry the
    rename cannot be applied to (a partial rename of its replicate columns) is
    kept as it was; its old names then no longer resolve and it is not restored.
    """
    history = getattr(sample, "_design_history", ())
    if not history or not renames:
        return
    renamed: list[Design] = []
    for d in history:
        try:
            renamed.append(_design_with_renamed_columns(d, renames, data_columns))
        except ValueError:
            renamed.append(d)
    sample._design_history = tuple(renamed)


# -------------------------------------------------------------------
# Metadata-key updates after rename
# -------------------------------------------------------------------


def _update_metadata_keys(sample: "Sample", renames: dict[str, str]) -> None:
    """Update metadata store keys after column rename."""
    if not renames:
        return

    meta = sample._metadata
    for old_name, new_name in renames.items():
        var_meta = meta.get(old_name)
        if var_meta is not None:
            meta.remove(old_name)
            meta.set(new_name, var_meta.clone(name=new_name))


# -------------------------------------------------------------------
# Case / letter normalisation
# -------------------------------------------------------------------


def _normalize_case_style(
    case_style: Literal["snake", "camel", "pascal", "kebab"] | None,
) -> _CaseStyle:
    """Normalize user-facing *case_style* string to internal enum."""
    _MAP = {
        "snake": _CaseStyle.SNAKE,
        "camel": _CaseStyle.CAMEL,
        "pascal": _CaseStyle.PASCAL,
        "kebab": _CaseStyle.KEBAB,
    }
    if case_style is None:
        return _CaseStyle.SNAKE
    if not isinstance(case_style, str):
        raise TypeError(
            f"'case_style' must be a string or None, got "
            f"{type(case_style).__name__}. "
            f"Use 'snake', 'camel', 'pascal', or 'kebab'."
        )
    result = _MAP.get(case_style.strip().lower())
    if result is None:
        raise ValueError(
            f"Unknown case_style {case_style!r}. Use 'snake', 'camel', 'pascal', or 'kebab'."
        )
    return result


def _normalize_letter_case(
    letter_case: Literal["lower", "upper", "title", "original"] | None,
) -> _LetterCase:
    """Normalize user-facing *letter_case* string to internal enum."""
    _MAP = {
        "lower": _LetterCase.LOWER,
        "upper": _LetterCase.UPPER,
        "title": _LetterCase.TITLE,
        "original": _LetterCase.ORIGINAL,
    }
    if letter_case is None:
        return _LetterCase.LOWER
    if not isinstance(letter_case, str):
        raise TypeError(
            f"'letter_case' must be a string or None, got "
            f"{type(letter_case).__name__}. "
            f"Use 'lower', 'upper', 'title', or 'original'."
        )
    result = _MAP.get(letter_case.strip().lower())
    if result is None:
        raise ValueError(
            f"Unknown letter_case {letter_case!r}. Use 'lower', 'upper', 'title', or 'original'."
        )
    return result
