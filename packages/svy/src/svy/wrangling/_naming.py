# src/svy/wrangling/_naming.py
"""
Column-naming utilities for the wrangling namespace.

Pure functions for case/style normalisation, design-column renaming,
and metadata-key updates.  Nothing here imports from Sample directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from svy.core.design import Design
from svy.core.enumerations import (
    CaseStyle as _CaseStyle,
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


def _map_name_in_design(
    design_field: str | None, renames: dict[str, str]
) -> str | None:
    """Map a single string design field through *renames*."""
    return (
        renames.get(design_field, design_field)
        if isinstance(design_field, str)
        else design_field
    )


def _map_tuple_in_design(
    design_field: str | tuple[str, ...] | None, renames: dict[str, str]
) -> str | tuple[str, ...] | None:
    """Map a string-or-tuple design field through *renames*."""
    if design_field is None or isinstance(design_field, str):
        return _map_name_in_design(design_field, renames)
    return tuple(renames.get(s, s) for s in design_field)


def _design_with_renamed_columns(
    design: Design, renames: dict[str, str]
) -> Design:
    """Return a new Design with all column references updated by *renames*."""
    if not renames:
        return design

    new_rep = design.rep_wgts
    if design.rep_wgts is not None:
        mapped_wgts = tuple(
            renames.get(s, s) for s in design.rep_wgts.columns
        )
        if mapped_wgts != tuple(design.rep_wgts.columns):
            new_rep = design.rep_wgts.clone(
                wgts=mapped_wgts, n_reps=len(mapped_wgts)
            )

    return design.update(
        row_index=_map_name_in_design(design.row_index, renames),
        stratum=_map_tuple_in_design(design.stratum, renames),
        wgt=_map_name_in_design(design.wgt, renames),
        prob=_map_name_in_design(design.prob, renames),
        hit=_map_name_in_design(design.hit, renames),
        mos=_map_name_in_design(design.mos, renames),
        psu=_map_tuple_in_design(design.psu, renames),
        ssu=_map_tuple_in_design(design.ssu, renames),
        pop_size=_map_name_in_design(design.pop_size, renames),
        rep_wgts=new_rep,
    )


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
            f"Unknown case_style {case_style!r}. "
            f"Use 'snake', 'camel', 'pascal', or 'kebab'."
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
            f"Unknown letter_case {letter_case!r}. "
            f"Use 'lower', 'upper', 'title', or 'original'."
        )
    return result
