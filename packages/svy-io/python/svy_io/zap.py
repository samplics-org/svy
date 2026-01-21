# python/svy_io/zap.py
from __future__ import annotations

import copy

from typing import Any, Dict, List, Tuple, cast, overload

import polars as pl

from svy_io.tagged_na import TaggedNA


INT_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)

FLOAT_DTYPES = (
    pl.Float32,
    pl.Float64,
)

# ───────────────────────── internal helpers ─────────────────────────


def _require_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        raise TypeError("expected a metadata dict")
    return meta


def _zap_label_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(_require_meta(meta))
    for v in out.get("vars", []):
        v["label"] = None
    # dataset/file label lives at top-level
    out["file_label"] = None
    return out


def _zap_labels_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(_require_meta(meta))
    # remove per-column link to label sets
    for v in out.get("vars", []):
        v["label_set"] = None
    # remove global value label sets
    out["value_labels"] = []
    return out


def _zap_formats_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(_require_meta(meta))
    for v in out.get("vars", []):
        # R haven uses format.sas; we store fmt (from your reader)
        v.pop("fmt", None)
        v.pop("format.sas", None)
        v.pop("format.stata", None)
        v.pop("format.spss", None)
    return out


def _zap_widths_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(_require_meta(meta))
    for v in out.get("vars", []):
        v.pop("display_width", None)
        v.pop("width", None)
    return out


# ───────────────────────── zap_label ─────────────────────────


@overload
def zap_label(meta: Dict[str, Any]) -> Dict[str, Any]: ...
@overload
def zap_label(
    df: pl.DataFrame, meta: Dict[str, Any]
) -> Tuple[pl.DataFrame, Dict[str, Any]]: ...


def zap_label(arg1, meta: Dict[str, Any] | None = None):
    """Remove variable/dataset labels (metadata only)."""
    if meta is None:
        # zap_label(meta)
        return _zap_label_meta(arg1)
    # zap_label(df, meta)
    if not isinstance(arg1, pl.DataFrame):
        raise TypeError("zap_label(df, meta): first arg must be a Polars DataFrame")
    return arg1, _zap_label_meta(meta)


# ───────────────────────── zap_labels (value labels) ─────────────────────────


@overload
def zap_labels(meta: Dict[str, Any], *, user_na: bool = False) -> Dict[str, Any]: ...
@overload
def zap_labels(
    df: pl.DataFrame, meta: Dict[str, Any], *, user_na: bool = False
) -> Tuple[pl.DataFrame, Dict[str, Any]]: ...


def _zap_labels_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(meta)
    for v in out.get("vars", []):
        v["label_set"] = None
    out["value_labels"] = []
    return out


def zap_labels(obj, meta: Dict[str, Any] | None = None, *, user_na: bool = False):
    """
    Remove value labels.

    Two call forms:
      • meta_out = zap_labels(meta_dict)
      • (df_out, meta_out) = zap_labels(df, meta_dict)

    For SAS, `user_na` is ignored (kept for parity with haven).
    """
    # Meta-only form
    if meta is None:
        if not isinstance(obj, dict):
            raise TypeError("zap_labels(meta): meta must be a dict")
        return _zap_labels_meta(obj)

    # (df, meta) form — align meta.vars to df.columns and clear label sets
    if not isinstance(obj, pl.DataFrame):
        raise TypeError("zap_labels(df, meta): df must be a polars.DataFrame")

    df: pl.DataFrame = obj
    meta_in: Dict[str, Any] = meta

    # Build quick lookup from incoming meta vars (may not match df exactly)
    in_map = {v.get("name"): v for v in meta_in.get("vars", [])}

    aligned_vars = []
    for name in df.columns:
        v = copy.deepcopy(in_map.get(name, {}))
        # ensure required keys
        v["name"] = name
        v["label"] = v.get("label")
        v["fmt"] = v.get("fmt")
        # zap value-label attachment
        v["label_set"] = None
        aligned_vars.append(v)

    meta_out: Dict[str, Any] = {
        **{k: v for k, v in meta_in.items() if k not in ("vars", "value_labels")},
        "vars": aligned_vars,
        # drop all value label sets entirely
        "value_labels": [],
    }

    return df, meta_out


# ───────────────────────── zap_formats ─────────────────────────


@overload
def zap_formats(meta: Dict[str, Any]) -> Dict[str, Any]: ...
@overload
def zap_formats(
    df: pl.DataFrame, meta: Dict[str, Any]
) -> Tuple[pl.DataFrame, Dict[str, Any]]: ...


def zap_formats(arg1, meta: Dict[str, Any] | None = None):
    """Remove format attributes from per-column metadata."""
    if meta is None:
        return _zap_formats_meta(arg1)
    if not isinstance(arg1, pl.DataFrame):
        raise TypeError("zap_formats(df, meta): first arg must be a Polars DataFrame")
    return arg1, _zap_formats_meta(meta)


# ───────────────────────── zap_widths ─────────────────────────


@overload
def zap_widths(meta: Dict[str, Any]) -> Dict[str, Any]: ...
@overload
def zap_widths(
    df: pl.DataFrame, meta: Dict[str, Any]
) -> Tuple[pl.DataFrame, Dict[str, Any]]: ...


def zap_widths(obj, meta: Dict[str, Any] | None = None):
    """
    Remove 'display_width' from per-column metadata.
    - zap_widths(meta_dict) -> meta_out
    - zap_widths(df, meta_dict) -> (df_unchanged, meta_out)
    """
    if meta is None:
        if not isinstance(obj, dict):
            raise TypeError("zap_widths(meta): meta must be a dict")
        out = copy.deepcopy(obj)
        for v in out.get("vars", []):
            v.pop("display_width", None)
        return out

    if not isinstance(obj, pl.DataFrame):
        raise TypeError("zap_widths(df, meta): df must be a polars.DataFrame")

    df = obj
    meta_out = copy.deepcopy(meta)
    for v in meta_out.get("vars", []):
        v.pop("display_width", None)
    return df, meta_out


# ───────────────────────── zap_empty (string empty→NA) ─────────────────────────


def zap_empty(x: Any):
    """
    Replace empty strings "" with null/None.

    - pl.DataFrame: apply to all Utf8 columns (in place via with_columns)
    - pl.Series:    cast to Utf8 (lossless) and replace "" -> null
    - list/tuple/np.ndarray: return a Python list with "" -> None
    """
    # ── Polars DataFrame ────────────────────────────────────────────────────────
    if isinstance(x, pl.DataFrame):
        # build expressions only for Utf8 columns
        mods = []
        for name, dtype in x.schema.items():
            if dtype == pl.Utf8:
                mods.append(
                    pl.when(pl.col(name) == "")
                    .then(None)
                    .otherwise(pl.col(name))
                    .alias(name)
                )
        return x.with_columns(mods) if mods else x

    # ── Polars Series ──────────────────────────────────────────────────────────
    if isinstance(x, pl.Series):
        s = x
        if s.dtype != pl.Utf8:
            s = s.cast(pl.Utf8, strict=False)
        return s.map_elements(lambda v: None if v == "" else v)

    # ── Python sequence / numpy array ──────────────────────────────────────────
    if isinstance(x, (list, tuple)):
        return [None if v == "" else v for v in x]

    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(x, np.ndarray):
            return [None if v == "" else v for v in x.tolist()]
    except Exception:
        pass

    raise TypeError(
        "zap_empty(x): x must be a polars.DataFrame, polars.Series, or a sequence of strings"
    )


# ───────────────────────── zap_missing (special/user missings→NA) ─────────────────────────


def _user_missing_map(meta: dict) -> dict:
    umap = {}
    for spec in meta.get("user_missing", []) or []:
        # add "col" to the acceptable aliases
        col = (
            spec.get("col") or spec.get("name") or spec.get("column") or spec.get("var")
        )
        if col:
            umap[col] = {
                k: spec.get(k)
                for k in ("na_values", "na_range")
                if spec.get(k) is not None
            }
    return umap


def zap_missing(df: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """
    Convert special/user missings to null (NA).

    Behavior:
      • Replace TaggedNA(...) objects anywhere with null.
      • Floats: replace NaN/±inf with null.
      • If meta['user_missing'] has per-column specs:
          - 'na_values': exact values become null
          - 'na_range' : inclusive range [lo, hi] becomes null
        Works for numeric and string columns; string ranges are lexicographic.

    Returns a new DataFrame (metadata unchanged).
    """
    umap = _user_missing_map(meta)
    exprs: List[pl.Expr] = []

    for name, dtype in df.schema.items():
        col = pl.col(name)

        # 1) always clean TaggedNA -> null (works regardless of dtype)
        col_clean = col.map_elements(
            lambda v: None if isinstance(v, TaggedNA) else v,
            return_dtype=dtype,
        )

        # 2) start building condition
        cond = None

        # floats: null out non-finite
        if dtype in (pl.Float32, pl.Float64):
            nf = ~col_clean.is_finite()
            cond = nf if cond is None else (cond | nf)

        # 3) user-defined missings
        spec = umap.get(name)
        if spec:
            # --- exact values ---
            nv = spec.get("na_values") or []
            if nv:
                if dtype in INT_DTYPES:
                    try:
                        nv_cast = [int(v) for v in nv if v is not None]
                    except Exception:
                        nv_cast = nv
                    c = col_clean.is_in(nv_cast)
                elif dtype in FLOAT_DTYPES:
                    try:
                        nv_cast = [float(v) for v in nv if v is not None]
                    except Exception:
                        nv_cast = nv
                    # keep non-finite handling via cond above
                    c = col_clean.is_in(nv_cast)
                elif dtype == pl.Utf8:
                    nv_cast = [str(v) for v in nv if v is not None]
                    c = col_clean.is_in(nv_cast)
                else:
                    # fallback: try direct inclusion
                    c = col_clean.is_in(nv)
                cond = c if cond is None else (cond | c)

            # --- inclusive range ---
            r = spec.get("na_range")
            if r is not None:
                lo, hi = r
                # use pl.lit so Polars will coerce scalars appropriately
                c = (col_clean >= pl.lit(lo)) & (col_clean <= pl.lit(hi))
                cond = c if cond is None else (cond | c)

        if cond is not None:
            exprs.append(
                pl.when(cond).then(pl.lit(None)).otherwise(col_clean).alias(name)
            )
        else:
            # still apply the TaggedNA sweep even if no cond
            exprs.append(col_clean.alias(name))

    return df.with_columns(exprs) if exprs else df


def zap_missing_with_meta(df: pl.DataFrame, meta: dict) -> tuple[pl.DataFrame, dict]:
    """
    Like zap_missing, but also updates meta by removing value-label entries
    that were turned into NA by user-missing rules.
    """
    out = zap_missing(df, meta)

    # deep copy to avoid mutating caller's meta
    meta_out = copy.deepcopy(meta)

    # index vars by name
    vmap = {v["name"]: v for v in meta_out.get("vars", [])}
    # map set_name -> mapping dict
    vlists = meta_out.get("value_labels", [])
    vl_by_name = {vl["set_name"]: vl for vl in vlists}

    # helper to drop specific codes from a label-set mapping
    def _drop_codes(
        set_name: str | None,
        dtype: pl.DataType,
        na_values: list | None,
        na_range: tuple | None,
    ):
        if not set_name or set_name not in vl_by_name:
            return
        mapping = vl_by_name[set_name]["mapping"]
        # keys are strings in meta; compare as strings for Utf8, numeric stringification otherwise
        if na_values:
            if dtype == pl.Utf8:
                for v in list(na_values):
                    mapping.pop(str(v), None)
            else:
                for v in list(na_values):
                    mapping.pop(str(v), None)
        if na_range:
            lo, hi = na_range
            # remove keys in inclusive range
            to_remove = []
            if dtype == pl.Utf8:
                for k in mapping.keys():
                    if lo <= k <= hi:
                        to_remove.append(k)
            else:
                # parse numeric keys safely
                def _num(x):
                    try:
                        return float(x)
                    except Exception:
                        return None

                lo_n = _num(lo)
                hi_n = _num(hi)
                if lo_n is not None and hi_n is not None:
                    for k in mapping.keys():
                        kv = _num(k)
                        if kv is not None and lo_n <= kv <= hi_n:
                            to_remove.append(k)
            for k in to_remove:
                mapping.pop(k, None)

    for um in meta_out.get("user_missing", []):
        col = um.get("col")
        if not col or col not in vmap:
            continue
        set_name = vmap[col].get("label_set")

        # Normalize schema dtype -> concrete pl.DataType
        dtype_any = df.schema.get(col)

        # Default
        dtype_norm = pl.Utf8

        # Handle common variants across Polars versions
        if dtype_any is not None:
            # Polars >= 1.9 may expose pl.String as a separate class
            if hasattr(pl, "String") and dtype_any == getattr(pl, "String"):
                dtype_norm = pl.Utf8
            # Some versions expose an Unknown dtype
            elif hasattr(pl, "Unknown") and dtype_any == getattr(pl, "Unknown"):
                dtype_norm = pl.Utf8
            # If it's already a real dtype, keep it
            elif isinstance(dtype_any, pl.DataType):
                dtype_norm = dtype_any
            else:
                # Anything else (eg older aliases) -> Utf8
                dtype_norm = pl.Utf8

        dtype = cast(pl.DataType, dtype_norm)

        _drop_codes(set_name, dtype, um.get("na_values"), um.get("na_range"))

    return out, meta_out
