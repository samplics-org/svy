from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VarMeta:
    name: str
    label: Optional[str]
    label_set: Optional[str]
    fmt: Optional[str]
    kind: str


@dataclass
class ValueLabels:
    set_name: str
    mapping: Dict[str, str]


@dataclass
class MissingRule:
    var: str
    discrete: List[str]
    ranges: List[Tuple[str, str]]


@dataclass
class SvyMetadata:
    file_label: Optional[str]
    vars: List[VarMeta]
    value_labels: Dict[str, ValueLabels]
    user_missing: Dict[str, MissingRule]
    n_rows: int


def _maybe_float(v: Any) -> Any:
    """Parse numeric strings from the native layer; keep everything else."""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def normalize_user_missing(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the canonical ``meta["user_missing"]`` schema.

    Canonical (haven-compatible) shape, one entry per column::

        {"col": <name>, "na_values": [...], "na_range": (lo, hi) | None}

    Historically three shapes coexisted — the native layer's
    ``{var, discrete, ranges}``, the per-variable ``{values, range}``, and
    the ``{col, values, range}`` rewrite in ``read_sav(user_na=True)`` —
    and ``zap_missing`` (which reads ``na_values``/``na_range``) matched
    none of them, silently doing nothing on real metadata.  Every reader
    now funnels through this one converter.

    Sources, later ones winning per column:
      1. any existing meta-level entries (canonical or legacy native shape),
      2. per-variable ``vars[i]["user_missing"]`` definitions.
    """
    by_col: Dict[str, Dict[str, Any]] = {}

    for rule in meta.get("user_missing") or []:
        col = rule.get("col") or rule.get("var") or rule.get("name")
        if not col:
            continue
        if "na_values" in rule or "na_range" in rule:
            # Already canonical — keep extra keys (e.g. "type") as-is.
            by_col[col] = dict(rule)
            continue
        # Native MissingRule shape: {var, discrete: [str], ranges: [(lo, hi)]}
        na_values = [_maybe_float(v) for v in rule.get("discrete") or []]
        ranges = rule.get("ranges") or []
        na_range = tuple(_maybe_float(v) for v in ranges[0]) if ranges else None
        # Legacy hydrate shape: {col, values, range}
        if not na_values:
            na_values = [_maybe_float(v) for v in rule.get("values") or []]
        if na_range is None and rule.get("range"):
            na_range = tuple(_maybe_float(v) for v in rule["range"])
        if na_values or na_range:
            by_col[col] = {"col": col, "na_values": na_values, "na_range": na_range}

    for var in meta.get("vars") or []:
        name = var.get("name")
        um = var.get("user_missing")
        if not name or not um:
            continue
        na_values = [_maybe_float(v) for v in um.get("values") or []]
        na_range = tuple(_maybe_float(v) for v in um["range"]) if um.get("range") else None
        if na_values or na_range:
            by_col[name] = {"col": name, "na_values": na_values, "na_range": na_range}

    return list(by_col.values())
