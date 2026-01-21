# python/svy_io/factor.py
from __future__ import annotations

from typing import Any, Dict, Optional

import polars as pl


def as_factor(
    s: pl.Series,
    labels: Optional[Dict[Any, str]] = None,
    *,
    value_labels: Optional[Dict[Any, str]] = None,  # alias (preferred name in readers)
    levels: str = "default",  # "default" | "labels" | "values" | "both"
    ordered: bool = False,  # reserved; Polars categoricals are unordered by default
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

    # No mapping: just categorize the raw values
    if not mapping:
        return s.cast(pl.Categorical)

    # tolerant lookup: exact match first, else try str(value)
    def _lookup(val: Any) -> Optional[str]:
        if val is None:
            return None
        if val in mapping:
            return mapping[val]
        return mapping.get(str(val))

    if levels == "values":
        # Keep raw values as categories
        return s.cast(pl.Categorical)

    if levels == "labels":
        # Only labels; unlabelled values become null
        out = s.map_elements(_lookup)  # -> Optional[str]
        return out.cast(pl.Utf8).cast(pl.Categorical)

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
