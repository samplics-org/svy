# python/svy_io/stata.py
from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import svy_io.svyreadstat_rs as native

from polars.exceptions import ComputeError

from .helpers import _as_path, _normalize_n_max
from .tagged_na import TaggedNA


_STATA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _hydrate_tagged_na(df: pl.DataFrame, meta: Dict[str, Any]) -> pl.DataFrame:
    """
    Replace markers described in meta["tagged_missings"] with TaggedNA objects.
    Only builds replacement series if something actually changes.
    """
    specs: List[Dict[str, Any]] = meta.get("tagged_missings") or []
    if not specs:
        return df

    # Group specs by column for batch processing
    col_specs: dict[str, list] = {}
    for spec in specs:
        col = spec.get("col") or spec.get("name")
        if col and col in df.columns:
            col_specs.setdefault(col, []).append(spec)

    if not col_specs:
        return df

    replacements: list[pl.Series] = []
    for col, specs_for_col in col_specs.items():
        vals = df[col].to_list()
        changed = False

        for spec in specs_for_col:
            # A) explicit rows+tags
            rows, tags = spec.get("rows"), spec.get("tags")
            if rows and tags and len(rows) == len(tags):
                for r, t in zip(rows, tags):
                    if 0 <= r < len(vals):
                        vals[r] = TaggedNA(t)
                        changed = True
                continue

            # B) value-based mapping
            by_value = spec.get("by_value")
            if isinstance(by_value, dict) and by_value:
                for i, v in enumerate(vals):
                    t = by_value.get(v)
                    if t is not None:
                        vals[i] = TaggedNA(t)
                        changed = True

        if changed:
            replacements.append(pl.Series(name=col, values=vals, dtype=pl.Object))

    return df.with_columns(replacements) if replacements else df


def _build_value_label_lookup(meta: dict) -> dict[str, dict[str, str]]:
    """Direct dict comprehension"""
    return {vl["set_name"]: vl["mapping"] for vl in meta.get("value_labels", [])}


def _column_label_map(meta: dict) -> dict[str, dict[str, Any]]:
    """Direct dict comprehension"""
    return {
        v["name"]: {
            "label": v.get("label"),
            "label_set": v.get("label_set"),
            "fmt": v.get("fmt"),
        }
        for v in meta.get("vars", [])
    }


def get_column_labels(meta: dict) -> dict[str, str | None]:
    """Single pass instead of building intermediate dict"""
    return {v["name"]: v.get("label") for v in meta.get("vars", [])}


def get_value_labels_for_column(meta: dict, col_name: str) -> dict[str, str] | None:
    # Find column metadata in single pass
    col_info = next((v for v in meta.get("vars", []) if v["name"] == col_name), None)
    if not col_info:
        return None

    set_name = col_info.get("label_set")
    if not set_name:
        return None

    # Find label set in single pass
    labels = next(
        (vl["mapping"] for vl in meta.get("value_labels", []) if vl["set_name"] == set_name), None
    )
    return labels


def read_dta(
    data_path: str | os.PathLike | io.BufferedIOBase,
    *,
    cols_skip: list[str] | None = None,
    n_max: int | None = None,
    rows_skip: int = 0,
    coerce_temporals: bool = False,
    zap_empty_str: bool = False,
    factorize: bool = False,
    levels: str = "default",
    ordered: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    # Lazy imports only when needed
    if coerce_temporals:
        from svy_io.temporals import coerce_stata_temporals  # type: ignore
    if zap_empty_str:
        from svy_io.zap import zap_empty  # type: ignore
    if factorize:
        from svy_io.sas import apply_value_labels  # reuse shared impl

    n_max = _normalize_n_max(n_max)
    if n_max == 0:
        empty = pl.DataFrame({})
        meta: Dict[str, Any] = {
            "file_label": None,
            "vars": [],
            "value_labels": [],
            "user_missing": [],
            "n_rows": 0,
        }
        return empty, meta

    # Rust does the heavy lifting here (with GIL released).
    with _as_path(data_path) as _path:
        ipc_bytes, meta_json = native.df_parse_dta_file(  # type: ignore[attr-defined]
            _path,
            cols_skip,
            n_max,
            rows_skip,
        )

    # Parse JSON once
    meta: Dict[str, Any] = json.loads(meta_json)

    # Read IPC with proper error handling
    bio = io.BytesIO(ipc_bytes)
    try:
        df = pl.read_ipc(bio, memory_map=False)
    except ComputeError as e:
        if "InvalidFooter" in str(e):
            bio.seek(0)
            df = pl.read_ipc_stream(bio)
        else:
            raise

    # Apply transformations
    if coerce_temporals:
        df = coerce_stata_temporals(df, meta)  # type: ignore
    if zap_empty_str:
        df = zap_empty(df)  # type: ignore
    if factorize:
        df = apply_value_labels(df, meta, levels=levels, ordered=ordered)  # type: ignore

    # Only hydrate if there are tagged missings
    if meta.get("tagged_missings"):
        df = _hydrate_tagged_na(df, meta)

    return df, meta


read_stata = read_dta


def read_stata_arrow(
    data_path: str | os.PathLike | io.BufferedIOBase,
    *,
    cols_skip: list[str] | None = None,
    n_max: int | None = None,
    rows_skip: int = 0,
):
    """Arrow-table variant."""
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc

    from pyarrow import ArrowInvalid

    n_max = _normalize_n_max(n_max)
    if n_max == 0:
        empty = pa.table({})
        meta = {
            "file_label": None,
            "vars": [],
            "value_labels": [],
            "user_missing": [],
            "n_rows": 0,
        }
        return empty, meta

    with _as_path(data_path) as _path:
        ipc_bytes, meta_json = native.df_parse_dta_file(  # type: ignore[attr-defined]
            _path,
            cols_skip,
            n_max,
            rows_skip,
        )

    bio = io.BytesIO(ipc_bytes)
    try:
        table = pa_ipc.open_file(bio).read_all()
    except ArrowInvalid:
        bio.seek(0)
        table = pa_ipc.open_stream(bio).read_all()

    meta = json.loads(meta_json)
    return table, meta


# ─────────────────────────────────────────────────────────────────────────────
# write_dta: Rust-backed writer (svyreadstat_rs.df_write_dta_file)
# ─────────────────────────────────────────────────────────────────────────────


def _stata_file_format(version: int) -> int:
    """Pre-computed dict lookup"""
    v = int(version)
    if v >= 113:
        return v

    version_map = {
        15: 119,
        14: 118,
        13: 117,
        12: 115,
        11: 114,
        10: 114,
        9: 113,
        8: 113,
    }
    result = version_map.get(v)
    if result is None:
        raise ValueError(f"Unsupported Stata version {version!r} (use 8–15)")
    return result


def _validate_dta_names_and_labels(
    df: pl.DataFrame,
    *,
    version_human: int,
    file_label: Optional[str],
    value_labels: dict[str, dict] | None,
):
    """Single pass validation where possible"""
    # Check column name lengths
    too_long = [c for c in df.columns if len(c) > 32]
    if too_long:
        raise ValueError(f"Variables must have Stata-compatible names (≤32). Too long: {too_long}")

    # Check name format for old Stata versions
    if version_human < 14:
        bad = [c for c in df.columns if not _STATA_NAME_RE.match(c)]
        if bad:
            raise ValueError(
                "Variables for Stata <14 must match ^[A-Za-z_][A-Za-z0-9_]*$: " + ", ".join(bad)
            )

    # Check file label length
    if file_label is not None and len(file_label) > 80:
        raise ValueError("file_label must be 80 characters or fewer.")

    # Validate value labels
    if value_labels:
        offenders: dict[str, list] = {}
        for var, mp in value_labels.items():
            bad_keys = [
                k
                for k in mp.keys()
                if not (
                    isinstance(k, (int, bool))
                    or (isinstance(k, float) and math.isfinite(k) and int(k) == k)
                )
            ]
            if bad_keys:
                offenders[var] = bad_keys

        if offenders:
            raise ValueError(
                "Value labels must use integer codes; offending keys: " + str(offenders)
            )


def _ensure_strl_policy(df: pl.DataFrame, *, version_human: int, strl_threshold: int):
    """Early exit if possible"""
    if not (0 <= int(strl_threshold) <= 2045):
        raise ValueError("strl_threshold must be within [0, 2045]")

    if version_human >= 13:
        return  # strL supported, no check needed

    # Only check string columns for old versions
    for c, t in df.schema.items():
        if t == pl.Utf8:
            max_len = df.select(pl.col(c).str.len_bytes().fill_null(0).max()).item() or 0
            if int(max_len) > 244:
                raise ValueError(
                    f"Column '{c}' contains strings of length {max_len}, but Stata < 13 has no strL support."
                )


def _apply_inf_policy(df: pl.DataFrame, *, na_policy: str) -> pl.DataFrame:
    """
    Vectorized ±Inf handling on float columns.
    - 'keep': no-op
    - 'error': raise if any ±Inf present
    - 'nan': replace ±Inf with nulls (preserve NaN as NaN)
    """
    if na_policy not in {"nan", "error", "keep"}:
        raise ValueError("na_policy must be one of {'nan','error','keep'}")
    if na_policy == "keep":
        return df

    float_cols = [name for name, dt in df.schema.items() if dt in (pl.Float32, pl.Float64)]
    if not float_cols:
        return df

    if na_policy == "error":
        # detect columns that have infinite values (but ignore NaN)
        inf_any_exprs = [
            ((~pl.col(c).is_finite()) & (~pl.col(c).is_nan())).any().alias(c) for c in float_cols
        ]
        flags = df.select(inf_any_exprs).row(0)
        bad = [c for c, has_inf in zip(float_cols, flags) if has_inf]
        if bad:
            raise ValueError(f"Found ±Inf values but na_policy='error' in columns: {bad}")
        return df

    # na_policy == "nan": replace only infinities with nulls, keep NaN as NaN
    exprs = []
    for c in float_cols:
        dt = df.schema[c]
        col = pl.col(c)
        exprs.append(
            pl.when(col.is_finite() | col.is_null() | col.is_nan())
            .then(col)
            .otherwise(None)
            .cast(dt)  # preserve original Float32/Float64 dtype
            .alias(c)
        )
    return df.with_columns(exprs)


def _adjust_temporals(df: pl.DataFrame, *, adjust_tz: bool) -> pl.DataFrame:
    """Batch process datetime columns"""
    datetime_cols = [
        (name, dt)
        for name, dt in df.schema.items()
        if isinstance(dt, pl.datatypes.Datetime) and getattr(dt, "time_zone", None)
    ]
    if not datetime_cols:
        return df

    new_series = []
    for name, _dt in datetime_cols:
        s = df[name]
        try:
            s2 = (
                s.dt.replace_time_zone(None)
                if adjust_tz
                else s.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
            )
            new_series.append(s2.alias(name))
        except Exception:
            # If any conversion fails, skip that column silently
            pass

    return df.with_columns(new_series) if new_series else df


def _extract_tagged_missings(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    """Single pass, batch update; returns DF (with TaggedNA -> None) + specs to round-trip tags."""
    specs: list[dict[str, object]] = []
    new_series = []

    for name in df.columns:
        vals = df[name].to_list()
        rows: list[int] = []
        tags: list[str] = []
        changed = False

        for i, v in enumerate(vals):
            if isinstance(v, TaggedNA):
                rows.append(i)
                tags.append(v.tag)
                vals[i] = None
                changed = True

        if changed:
            # Stata tagged-missings only apply to numeric; we store as f64
            new_series.append(pl.Series(name=name, values=vals, dtype=pl.Float64))
            specs.append({"col": name, "rows": rows, "tags": tags})

    out = df.with_columns(new_series) if new_series else df
    return out, specs


def _coerce_ints_to_f64_for_stata(df: pl.DataFrame) -> pl.DataFrame:
    """Cast all integer columns to Float64 (Stata writer expects numeric doubles)."""
    int_dtypes = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }
    casts = [
        pl.col(name).cast(pl.Float64) for name, dtype in df.schema.items() if dtype in int_dtypes
    ]
    return df.with_columns(casts) if casts else df


def _df_to_ipc_bytes(df: pl.DataFrame) -> bytes:
    """Fast IPC serialization"""
    bio = io.BytesIO()
    df.write_ipc(bio)
    return bio.getvalue()


def _columns_with_interior_nul(df: pl.DataFrame) -> list[str]:
    """Early exit on first match per column"""
    hits: list[str] = []
    for name, dt in df.schema.items():
        if dt == pl.Utf8:
            for v in df[name].to_list():
                if isinstance(v, str) and "\x00" in v:
                    hits.append(name)
                    break
    return hits


def write_dta(
    df: pl.DataFrame,
    path: str | os.PathLike | io.BufferedIOBase,
    *,
    version: int = 15,
    file_label: str | None = None,
    var_labels: dict[str, str] | None = None,
    value_labels: dict[str, dict] | None = None,
    strl_threshold: int = 2045,
    adjust_tz: bool = True,
    na_policy: str = "nan",
) -> pl.DataFrame:
    """
    Write a Stata .dta using the Rust native writer (ReadStat backend).
    """
    if isinstance(path, (str, os.PathLike)):
        out_path = str(Path(path).resolve())
        file_like = None
    elif hasattr(path, "write"):
        tmp = tempfile.NamedTemporaryFile(suffix=".dta", delete=False)
        out_path, file_like = tmp.name, path
        tmp.close()
    else:
        raise TypeError("path must be a filesystem path or a writable file-like object")

    version_internal = _stata_file_format(int(version))
    version_human = (
        int(version)
        if int(version) < 113
        else {119: 15, 118: 14, 117: 13, 115: 12, 114: 11, 113: 9}.get(int(version), 15)
    )

    _validate_dta_names_and_labels(
        df,
        version_human=version_human,
        file_label=file_label,
        value_labels=value_labels,
    )
    _ensure_strl_policy(df, version_human=version_human, strl_threshold=int(strl_threshold))

    nul_cols = _columns_with_interior_nul(df)
    if nul_cols:
        raise ValueError(
            "Strings containing embedded NUL (\\x00) are not supported by the underlying ReadStat "
            f"parser and cannot be round-tripped: columns {nul_cols}. Consider removing or replacing "
            "the NULs before writing."
        )

    # Pipeline transformations
    df_w = _apply_inf_policy(df, na_policy=na_policy)
    df_w = _adjust_temporals(df_w, adjust_tz=adjust_tz)
    df_w, user_missing_specs = _extract_tagged_missings(df_w)
    df_w = _coerce_ints_to_f64_for_stata(df_w)

    ipc_bytes = _df_to_ipc_bytes(df_w)
    var_labels_json = json.dumps(var_labels) if var_labels else None
    value_labels_json = json.dumps(value_labels) if value_labels else None
    user_missing_json = json.dumps(user_missing_specs) if user_missing_specs else None

    if not hasattr(native, "df_write_dta_file"):
        raise NotImplementedError(
            "svyreadstat_rs.df_write_dta_file is missing. "
            "Implement it in Rust (see native/svyreadstat_rs/src/stata.rs)."
        )

    native.df_write_dta_file(  # type: ignore[attr-defined]
        ipc_bytes,
        out_path,
        int(version_internal),
        file_label,
        var_labels_json,
        value_labels_json,
        int(strl_threshold),
        user_missing_json,
    )

    if file_like is not None:
        with open(out_path, "rb") as fsrc:
            file_like.write(fsrc.read())
        try:
            os.remove(out_path)
        except Exception:
            pass

    return df


write_stata = write_dta
