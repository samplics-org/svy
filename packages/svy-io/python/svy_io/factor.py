# python/svy_io/factor.py
from __future__ import annotations

from typing import Any, Dict, Optional

import polars as pl


def ordered_categories(mapping: Dict[Any, str], *, levels: str) -> list[str]:
    """
    Category list for an ordered factor, in the order the codes imply.

    Sorted by the numeric value of the code, not its text. Readers hand back
    value labels in a BTreeMap keyed by the code's *string* form, so iteration
    order puts "10" between "1" and "2"; taking that order as given would put
    an 11-category scale in the wrong sequence. Codes that are not numeric
    keep their string order, which is the only order they have.
    """

    def sort_key(code: Any):
        try:
            return (0, float(code), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(code))

    codes = sorted(mapping, key=sort_key)
    if levels == "values":
        return [str(c) for c in codes]
    return [mapping[c] for c in codes]


def check_ordered_levels(levels: str) -> None:
    """
    An ordered factor needs a closed set of categories, and only the 'labels'
    and 'values' modes have one -- both are fully described by the label set.
    'default' and 'both' fall back to the raw data value for anything
    unlabelled, so their categories, and therefore their order, depend on data
    the mapping cannot describe.
    """
    if levels not in {"labels", "values"}:
        raise ValueError(
            f"ordered=True needs levels='labels' or 'values', got {levels!r}. "
            f"The {levels!r} mode falls back to the raw value for anything "
            "unlabelled, so its categories are not known from the label set "
            "and have no defined order."
        )


def as_factor(
    s: pl.Series,
    labels: Optional[Dict[Any, str]] = None,
    *,
    value_labels: Optional[Dict[Any, str]] = None,  # alias (preferred name in readers)
    levels: str = "default",  # "default" | "labels" | "values" | "both"
    ordered: bool = False,
) -> pl.Series:
    """
    Convert a labelled series to a categorical using a haven-like policy.

    You can pass the mapping as either `labels` or `value_labels`.
    Mapping keys may be raw-typed (e.g., 1, 5) or strings ("1", "5").
    """
    # accept both param names
    mapping = value_labels if value_labels is not None else labels

    levels = levels.lower()
    if levels not in {"default", "labels", "values", "both"}:
        raise ValueError("levels must be one of: default, labels, values, both")

    if ordered:
        check_ordered_levels(levels)
        if not mapping:
            raise ValueError(
                "ordered=True needs value labels: the code order is what "
                "defines the order of an ordered factor."
            )

    # No mapping: just categorize the raw values
    # (stringify first: numeric -> Categorical casts are invalid in Polars)
    if not mapping:
        return s.cast(pl.Utf8).cast(pl.Categorical)

    # tolerant lookup: exact match first, else try str(value)
    def _lookup(val: Any) -> Optional[str]:
        if val is None:
            return None
        if val in mapping:
            return mapping[val]
        return mapping.get(str(val))

    if levels == "values":
        # Keep raw values as categories (stringified for non-string dtypes)
        out = s.cast(pl.Utf8)
        if ordered:
            # Enum, not Categorical: Categorical sorts its categories
            # alphabetically, which puts "Higher" before "None" on an
            # education scale. Enum keeps the order it is given.
            return out.cast(pl.Enum(ordered_categories(mapping, levels="values")))
        return out.cast(pl.Categorical)

    if levels == "labels":
        # Only labels; unlabelled values become null
        out = s.map_elements(_lookup)  # -> Optional[str]
        out = out.cast(pl.Utf8)
        if ordered:
            return out.cast(pl.Enum(ordered_categories(mapping, levels="labels")))
        return out.cast(pl.Categorical)

    if levels == "both":
        # Prefer label; display as "[raw] label" when labelled, else raw as string
        def _both(val: Any) -> Optional[str]:
            if val is None:
                return None
            lab = _lookup(val)
            raw = str(val)
            return f"[{raw}] {lab}" if lab is not None else raw

        out = s.map_elements(_both)
        return out.cast(pl.Utf8).cast(pl.Categorical)

    # "default": prefer label where available; otherwise raw value (stringified)
    def _default(val: Any) -> Optional[str]:
        if val is None:
            return None
        lab = _lookup(val)
        return lab if lab is not None else str(val)

    out = s.map_elements(_default)
    return out.cast(pl.Utf8).cast(pl.Categorical)
