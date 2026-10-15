from __future__ import annotations

from typing import Any, Dict, List


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


# ---------------- Writer-side specs ---------------- #
# Every writer funnels value labels and user missings through these, so a
# spec that one format accepts is never silently dropped by another.

_STRING_DTYPES = ("String", "Utf8", "Categorical", "Enum")


def canonical_code(code: Any) -> Any:
    """``"1"``, ``1.0`` and ``True`` all become ``1``; other numbers stay floats, other text stays text."""
    if isinstance(code, bool):
        return int(code)
    if isinstance(code, str):
        try:
            code = float(code)
        except ValueError:
            return code
    if isinstance(code, float) and code.is_integer():
        return int(code)
    return code


def normalize_value_labels(
    value_labels: Any, schema: Dict[str, Any] | None = None
) -> Dict[str, Dict[Any, str]]:
    """
    Accept ``{col: {code: text}}`` or ``[{"col": ..., "labels": {...}}]``; return the dict form.

    Codes of numeric columns are made numeric via ``canonical_code``; codes of
    string columns (per ``schema``) are kept as text.
    """
    if not value_labels:
        return {}
    if isinstance(value_labels, dict):
        items = list(value_labels.items())
    elif isinstance(value_labels, (list, tuple)):
        items = []
        for entry in value_labels:
            if not isinstance(entry, dict) or "col" not in entry:
                raise ValueError(f"value_labels entry needs a 'col' key: {entry!r}")
            mapping = entry.get("labels", entry.get("mapping"))
            if mapping is None:
                raise ValueError(f"value_labels entry for {entry['col']!r} has no 'labels'")
            items.append((entry["col"], mapping))
    else:
        raise TypeError("value_labels must be {col: {code: text}} or a list of {'col', 'labels'}")

    out: Dict[str, Dict[Any, str]] = {}
    for col, mapping in items:
        if not isinstance(mapping, dict):
            raise TypeError(f"value labels for {col!r} must be a dict of code -> text")
        is_text = schema is not None and str(schema.get(col, "")).startswith(_STRING_DTYPES)
        out[col] = {
            (str(k) if is_text else canonical_code(k)): ("" if v is None else str(v))
            for k, v in mapping.items()
        }
    return out


def normalize_user_missing_spec(user_missing: Any) -> List[Dict[str, Any]]:
    """
    Accept the reader's canonical ``{col, na_values, na_range}`` or the older
    ``{col, values, range}``, as a list or as ``{col: {...}}``; return the canonical list.
    """
    if not user_missing:
        return []
    if isinstance(user_missing, dict):
        entries = [{"col": col, **spec} for col, spec in user_missing.items()]
    elif isinstance(user_missing, (list, tuple)):
        entries = list(user_missing)
    else:
        raise TypeError(
            "user_missing must be a list of {'col', 'na_values', 'na_range'} or a dict"
        )

    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or "col" not in entry:
            raise ValueError(f"user_missing entry needs a 'col' key: {entry!r}")
        values = entry.get("na_values", entry.get("values")) or []
        rng = entry.get("na_range", entry.get("range"))
        if not values and not rng:
            raise ValueError(f"user_missing entry for {entry['col']!r} has no values or range")
        if rng is not None and len(rng) != 2:
            raise ValueError(f"user_missing range for {entry['col']!r} must be (low, high)")
        out.append(
            {
                "col": entry["col"],
                "na_values": [canonical_code(v) for v in values],
                "na_range": tuple(canonical_code(v) for v in rng) if rng is not None else None,
            }
        )
    return out


def reject_temporal_columns(df: Any, fmt: str) -> None:
    """Raise for date/time columns a writer cannot encode, instead of writing them as missing."""
    temporal = [name for name, dtype in df.schema.items() if dtype.is_temporal()]
    if temporal:
        raise ValueError(
            f"The {fmt} writer cannot store date/time columns yet: {temporal}. "
            "Convert them to numbers or strings first, or write SPSS, which keeps them."
        )
