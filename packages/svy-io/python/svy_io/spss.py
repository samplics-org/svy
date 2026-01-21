# python/svy_io/spss.py
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import svy_io.svyreadstat_rs as native
from polars.exceptions import ComputeError

from .helpers import _normalize_n_max, _as_path
from .labelled import LabelledSPSS, labelled_spss


# -------------------- User-defined missing integration --------------------


def _apply_user_missing_to_column(
    values: List[Any],
    var_meta: Dict[str, Any],
    value_labels: Optional[Dict[str, str]],
) -> LabelledSPSS:
    """Convert a column to LabelledSPSS if it has user-defined missing values."""
    user_miss = var_meta.get("user_missing")
    if not user_miss:
        return labelled_spss(values, labels=value_labels, label=var_meta.get("label"))

    na_values = user_miss.get("values")
    na_range_list = user_miss.get("range")
    na_range = tuple(na_range_list) if na_range_list and len(na_range_list) == 2 else None

    # Convert value_labels keys to match data type
    if (
        value_labels
        and values
        and isinstance(values[0], (int, float))
        and not isinstance(values[0], bool)
    ):
        converted_labels: Dict[Any, str] = {}
        for k, v in value_labels.items():
            try:
                num_key = float(k)
                if num_key == int(num_key):
                    num_key = int(num_key)
                converted_labels[num_key] = v
            except (ValueError, TypeError):
                converted_labels[k] = v
        value_labels = converted_labels

    return labelled_spss(
        values,
        labels=value_labels,
        na_values=na_values,
        na_range=na_range,
        label=var_meta.get("label"),
    )


def _hydrate_user_missing(
    df: pl.DataFrame,
    meta: Dict[str, Any],
    user_na: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Batch process user-defined missing values.

    If user_na=False: Convert to None/null (vectorized).
    If user_na=True: Create LabelledSPSS objects recorded in meta["labelled_columns"].
    """
    # Build value label lookup once
    value_label_sets = {vl["set_name"]: vl["mapping"] for vl in meta.get("value_labels", [])}

    if user_na:
        labelled_columns: Dict[str, LabelledSPSS] = {}
        user_missing_list = []

        for var in meta.get("vars", []):
            col_name = var["name"]
            if col_name not in df.columns or not var.get("user_missing"):
                continue

            user_miss = var["user_missing"]
            label_set = var.get("label_set")
            value_labels = value_label_sets.get(label_set) if label_set else None

            values = df[col_name].to_list()
            labelled_col = _apply_user_missing_to_column(values, var, value_labels)
            labelled_columns[col_name] = labelled_col

            # Build missing spec
            missing_spec = {"col": col_name}
            if user_miss.get("values"):
                missing_spec["values"] = user_miss["values"]
            if user_miss.get("range"):
                missing_spec["range"] = user_miss["range"]
            user_missing_list.append(missing_spec)

        meta["labelled_columns"] = labelled_columns
        meta["user_missing"] = user_missing_list
    else:
        # Vectorized conversion to null
        replacements = []

        for var in meta.get("vars", []):
            col_name = var["name"]
            if col_name not in df.columns:
                continue

            user_miss = var.get("user_missing")
            if not user_miss:
                continue

            na_values = user_miss.get("values", [])
            na_range = user_miss.get("range")

            if not na_values and not na_range:
                continue

            col = pl.col(col_name)
            conditions = []

            if na_values:
                conditions.append(col.is_in(na_values))

            if na_range and len(na_range) == 2:
                low, high = na_range
                if low is not None and high is not None:
                    conditions.append((col >= low) & (col <= high))
                elif low is not None:
                    conditions.append(col >= low)
                elif high is not None:
                    conditions.append(col <= high)

            if conditions:
                mask = conditions[0]
                for cond in conditions[1:]:
                    mask = mask | cond
                replacements.append(pl.when(mask).then(None).otherwise(col).alias(col_name))

        if replacements:
            df = df.with_columns(replacements)

    return df, meta


# -------------------- Metadata convenience --------------------


def _build_value_label_lookup(meta: dict) -> dict[str, dict[str, str]]:
    return {vl["set_name"]: vl["mapping"] for vl in meta.get("value_labels", [])}


def _column_label_map(meta: dict) -> dict[str, dict[str, Any]]:
    return {
        v["name"]: {
            "label": v.get("label"),
            "label_set": v.get("label_set"),
            "fmt": v.get("fmt"),
            "user_missing": v.get("user_missing"),
        }
        for v in meta.get("vars", [])
    }


def get_column_labels(meta: dict) -> dict[str, str | None]:
    return {v["name"]: v.get("label") for v in meta.get("vars", [])}


def get_value_labels_for_column(meta: dict, col_name: str) -> dict[str, str] | None:
    col_info = next((v for v in meta.get("vars", []) if v["name"] == col_name), None)
    if not col_info:
        return None
    set_name = col_info.get("label_set")
    if not set_name:
        return None
    return next(
        (vl["mapping"] for vl in meta.get("value_labels", []) if vl["set_name"] == set_name), None
    )


def get_user_missing_for_column(meta: dict, col_name: str) -> dict[str, Any] | None:
    var_info = next((v for v in meta.get("vars", []) if v["name"] == col_name), None)
    return var_info.get("user_missing") if var_info else None


# ---------------- Name normalization ----------------


def _normalize_names(
    df: pl.DataFrame, meta: Dict[str, Any]
) -> tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Normalize column names:
    - strip whitespace
    - lowercase
    - replace dots/spaces/dashes with underscores
    - collapse multiple underscores
    """
    rename: Dict[str, str] = {}

    for c in df.columns:
        nc = c.strip().lower().replace(".", "_").replace(" ", "_").replace("-", "_")
        while "__" in nc:
            nc = nc.replace("__", "_")
        nc = nc.strip("_")

        if nc != c:
            rename[c] = nc

    if rename:
        df = df.rename(rename)

    for v in meta.get("vars", []):
        if isinstance(v.get("name"), str):
            orig = v["name"]
            normalized = orig.strip().lower().replace(".", "_").replace(" ", "_").replace("-", "_")
            while "__" in normalized:
                normalized = normalized.replace("__", "_")
            v["name"] = normalized.strip("_")

    return df, meta


# ---------------- SPSS READERS ----------------


def _normalize_cols_skip(cols_skip: list[str] | None) -> list[str] | None:
    """Allow callers to pass names with . or _ interchangeably."""
    if not cols_skip:
        return None
    skip_set: set[str] = set()
    for col in cols_skip:
        skip_set.add(col)
        if "_" in col:
            skip_set.add(col.replace("_", "."))
        if "." in col:
            skip_set.add(col.replace(".", "_"))
    return list(skip_set)


def read_sav(
    data_path: str | os.PathLike | io.BufferedIOBase,
    *,
    encoding: str | None = None,
    user_na: bool = False,
    cols_skip: list[str] | None = None,
    n_max: int | None = None,
    rows_skip: int = 0,
    coerce_temporals: bool = True,
    zap_empty_str: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Fast, file-like–safe SPSS .sav reader (ReadStat backend)."""
    n_max = _normalize_n_max(n_max)

    if n_max == 0:
        return pl.DataFrame({}), {
            "file_label": None,
            "vars": [],
            "value_labels": [],
            "user_missing": [],
            "n_rows": 0,
        }

    normalized_cols_skip = _normalize_cols_skip(cols_skip)

    # Native parse (GIL released)
    with _as_path(data_path) as _path:
        ipc_bytes, meta_json = native.df_parse_sav_file(
            _path,
            encoding or None,
            user_na,
            normalized_cols_skip,
            n_max,
            rows_skip,
        )

    meta: Dict[str, Any] = json.loads(meta_json)

    bio = io.BytesIO(ipc_bytes)
    try:
        df = pl.read_ipc(bio, memory_map=False)
    except ComputeError as e:
        if "InvalidFooter" in str(e):
            bio.seek(0)
            df = pl.read_ipc_stream(bio)
        else:
            raise

    # Normalize names BEFORE downstream processing
    df, meta = _normalize_names(df, meta)

    if coerce_temporals:
        from svy_io.temporals import coerce_spss_temporals

        df = coerce_spss_temporals(df, meta)

    if zap_empty_str:
        from svy_io.zap import zap_empty

        df = zap_empty(df)

    df, meta = _hydrate_user_missing(df, meta, user_na)

    return df, meta


def read_por(
    data_path: str | os.PathLike | io.BufferedIOBase,
    *,
    user_na: bool = False,
    cols_skip: list[str] | None = None,
    n_max: int | None = None,
    rows_skip: int = 0,
    coerce_temporals: bool = False,
    zap_empty_str: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Fast, file-like–safe SPSS .por reader (ReadStat backend)."""
    n_max = _normalize_n_max(n_max)

    if n_max == 0:
        return pl.DataFrame({}), {
            "file_label": None,
            "vars": [],
            "value_labels": [],
            "user_missing": [],
            "n_rows": 0,
        }

    normalized_cols_skip = _normalize_cols_skip(cols_skip)

    with _as_path(data_path) as _path:
        ipc_bytes, meta_json = native.df_parse_por_file(
            _path,
            None,
            user_na,
            normalized_cols_skip,
            n_max,
            rows_skip,
        )

    meta: Dict[str, Any] = json.loads(meta_json)

    bio = io.BytesIO(ipc_bytes)
    try:
        df = pl.read_ipc(bio, memory_map=False)
    except ComputeError as e:
        if "InvalidFooter" in str(e):
            bio.seek(0)
            df = pl.read_ipc_stream(bio)
        else:
            raise

    df, meta = _normalize_names(df, meta)

    if coerce_temporals:
        from svy_io.temporals import coerce_spss_temporals

        df = coerce_spss_temporals(df, meta)

    if zap_empty_str:
        from svy_io.zap import zap_empty

        df = zap_empty(df)

    df, meta = _hydrate_user_missing(df, meta, user_na)

    return df, meta


def read_spss(
    data_path: str | os.PathLike,
    *,
    encoding: str | None = None,
    user_na: bool = False,
    cols_skip: list[str] | None = None,
    n_max: int | None = None,
    rows_skip: int = 0,
    coerce_temporals: bool = False,
    zap_empty_str: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Auto-dispatch based on extension. Requires a real filesystem path since we
    must inspect the suffix. For file-like inputs, call read_sav/read_por directly.
    """
    ext = Path(os.fspath(data_path)).suffix.lower()

    if ext in (".sav", ".zsav"):
        return read_sav(
            data_path,
            encoding=encoding,
            user_na=user_na,
            cols_skip=cols_skip,
            n_max=n_max,
            rows_skip=rows_skip,
            coerce_temporals=coerce_temporals,
            zap_empty_str=zap_empty_str,
        )
    elif ext == ".por":
        return read_por(
            data_path,
            user_na=user_na,
            cols_skip=cols_skip,
            n_max=n_max,
            rows_skip=rows_skip,
            coerce_temporals=coerce_temporals,
            zap_empty_str=zap_empty_str,
        )
    else:
        raise ValueError(f"Unknown SPSS file extension: {ext}")


# ---------------- SPSS WRITERS ----------------

_SPSS_RESERVED = {
    "ALL",
    "AND",
    "BY",
    "EQ",
    "GE",
    "GT",
    "LE",
    "LT",
    "NE",
    "NOT",
    "OR",
    "TO",
    "WITH",
}


def _is_valid_varname(name: str) -> bool:
    """Early exits, simpler checks"""
    if not name or len(name.encode("utf-8")) > 64:
        return False
    if not name[0].isalpha():
        return False
    return all(ch.isalnum() or ch == "_" for ch in name[1:])


def _validate_sav(df: pl.DataFrame) -> None:
    """Single pass validation"""
    seen_ci = set()

    for name in df.columns:
        key = name.casefold()
        if key in seen_ci:
            raise ValueError(
                f"SPSS does not allow duplicate variable names (case-insensitive): {name!r}"
            )
        seen_ci.add(key)

        if not _is_valid_varname(name):
            raise ValueError(f"Invalid variable name: {name!r}")

        if name.upper() in _SPSS_RESERVED:
            raise ValueError(f"Invalid/reserved variable name: {name!r}")


def write_sav(
    df: pl.DataFrame,
    path: str | Path,
    *,
    compress: str = "byte",
    adjust_tz: bool = True,
    var_labels: Optional[Dict[str, str]] = None,
    user_missing: Optional[List[Dict[str, Any]]] = None,
    value_labels: Optional[List[Dict[str, Any]]] = None,
) -> pl.DataFrame:
    """Batch categorical conversion, fewer copies"""
    path = os.fspath(path)

    if compress not in ("byte", "none", "zsav"):
        raise ValueError(f"compress must be 'byte', 'none', or 'zsav', got {compress!r}")

    _validate_sav(df)

    to_write = df
    if adjust_tz:
        from .stata import _adjust_temporals

        to_write = _adjust_temporals(df, adjust_tz=True)

    # Batch process categorical columns
    categorical_labels: Dict[str, Dict[str, str]] = {}
    categorical_exprs: List[pl.Expr] = []

    for col_name in to_write.columns:
        if to_write[col_name].dtype == pl.Categorical:
            cats = to_write[col_name].cat.get_categories()

            # Convert to 1-based codes
            codes = (to_write[col_name].to_physical() + 1).cast(pl.Float64)
            categorical_exprs.append(codes.alias(col_name))

            # Create value labels
            labels = {str(i + 1): cats[i] for i in range(len(cats))}
            categorical_labels[col_name] = labels

    if categorical_exprs:
        to_write = to_write.with_columns(categorical_exprs)

    # Merge categorical labels with user-provided
    if categorical_labels:
        if value_labels is None:
            value_labels = []

        for col_name, labels in categorical_labels.items():
            existing = next((vl for vl in value_labels if vl["col"] == col_name), None)
            if existing:
                existing["labels"].update(labels)
            else:
                value_labels.append({"col": col_name, "labels": labels})

    # Convert to Arrow IPC
    bio = io.BytesIO()
    to_write.write_ipc(bio)
    ipc_bytes = bio.getvalue()

    # Call native writer
    native.df_write_sav_file(
        ipc_bytes,
        path,
        compress=compress,
        var_labels=var_labels,
        user_missing=user_missing,
        value_labels=value_labels,
    )

    return df
