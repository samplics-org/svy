# python/svy_io/sas.py
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile

from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl
import svy_io.svyreadstat_rs as native

from polars.exceptions import ComputeError

from .factor import as_factor
from .helpers import _normalize_n_max
from .tagged_na import TaggedNA


# -------------------- hydrate tagged NA --------------------


def _hydrate_tagged_na(df: pl.DataFrame, meta: Dict[str, Any]) -> pl.DataFrame:
    """
    Replace specific values in the DataFrame with TaggedNA objects based on metadata.

    SAS supports tagged missing values (.A through .Z and ._), which carry semantic
    meaning beyond regular missing. This function hydrates those special values.

    The metadata should contain a "tagged_missings" list with specs like:
    [
        {
            "col": "column_name",
            "rows": [0, 5, 10],  # row indices
            "tags": ["A", "B", "A"]  # corresponding tags
        },
        # OR value-based mapping:
        {
            "col": "column_name",
            "by_value": {
                ".A": "A",
                ".B": "B",
                # Maps special SAS values to tag letters
            }
        }
    ]
    """
    specs: List[Dict[str, Any]] = meta.get("tagged_missings") or []
    if not specs:
        return df

    out = df
    for spec in specs:
        col = spec.get("col") or spec.get("name")
        if not col or col not in out.columns:
            continue

        s = out[col]

        # A) explicit rows+tags
        rows, tags = spec.get("rows"), spec.get("tags")
        if rows and tags and len(rows) == len(tags):
            vals = s.cast(pl.Object, strict=False).to_list()
            for r, t in zip(rows, tags):
                if 0 <= r < len(vals):
                    vals[r] = TaggedNA(t)
            out = out.with_columns(pl.Series(col, vals))
            continue

        # B) value-based mapping
        by_value = spec.get("by_value")
        if isinstance(by_value, dict) and by_value:
            vals = s.cast(pl.Object, strict=False).to_list()
            for i, v in enumerate(vals):
                t = by_value.get(v)
                if t is not None:
                    vals[i] = TaggedNA(t)
            out = out.with_columns(pl.Series(col, vals))

    return out


def get_tagged_na_info(meta: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract information about which columns have tagged missing values.

    Returns:
        Dict mapping column names to lists of tag letters used in that column.
        Example: {"age": ["A", "B"], "income": ["Z"]}
    """
    specs: List[Dict[str, Any]] = meta.get("tagged_missings") or []
    result: Dict[str, List[str]] = {}

    for spec in specs:
        col = spec.get("col") or spec.get("name")
        if not col:
            continue

        tags_in_col = set()

        # From explicit tags
        if spec.get("tags"):
            tags_in_col.update(spec["tags"])

        # From value mapping
        if spec.get("by_value"):
            tags_in_col.update(spec["by_value"].values())

        # From special values
        if spec.get("special_values"):
            tags_in_col.update(spec["special_values"].values())

        if tags_in_col:
            result[col] = sorted(tags_in_col)

    return result


def is_tagged_na(value) -> bool:
    """Check if a value is a TaggedNA instance."""
    return isinstance(value, TaggedNA)


def get_na_tag(value) -> str | None:
    """Get the tag letter from a TaggedNA value, or None if not tagged."""
    if isinstance(value, TaggedNA):
        return value.tag
    return None


# ----------------  Helper functions ----------------
def _as_path_like(obj) -> str:
    # already a path-like?
    if isinstance(obj, (str, os.PathLike)):
        return str(obj)
    # file-like?
    if hasattr(obj, "read"):
        with tempfile.NamedTemporaryFile(suffix=".sas7bdat", delete=False) as tmp:
            tmp.write(obj.read())
            return tmp.name
    raise TypeError("data_path must be a path or a file-like object")


def _maybe_from_zip(path: str) -> tuple[str, str | None]:
    """
    Extract SAS files from a zip archive.
    Returns (sas_path, catalog_path_or_None).

    Note: Creates temporary files in a system temp directory.
    The caller doesn't need to clean up - the OS will handle it eventually,
    but for immediate cleanup you could track the temp dir if needed.
    """
    if not str(path).lower().endswith(".zip"):
        return path, None

    if not zipfile.is_zipfile(path):
        raise ValueError(f"File {path} is not a valid zip archive")

    with zipfile.ZipFile(path) as z:
        # Find SAS data and catalog files
        sas_files = [n for n in z.namelist() if n.lower().endswith(".sas7bdat")]
        cat_files = [n for n in z.namelist() if n.lower().endswith(".sas7bcat")]

        if not sas_files:
            raise FileNotFoundError(
                f"Zip file {path} contains no .sas7bdat files. "
                f"Available files: {', '.join(z.namelist())}"
            )

        if len(sas_files) > 1:
            import warnings

            warnings.warn(
                f"Zip file contains {len(sas_files)} .sas7bdat files. "
                f"Using the first one: {sas_files[0]}",
                UserWarning,
            )

        # Extract to temp directory
        # Note: Using tempfile with delete=False means files persist
        # but are in the system temp dir which gets cleaned periodically
        temp_base = tempfile.gettempdir()

        # Extract data file
        sas_path = z.extract(sas_files[0], path=temp_base)

        # Extract catalog file if present
        cat_path = None
        if cat_files:
            if len(cat_files) > 1:
                import warnings

                warnings.warn(
                    f"Zip file contains {len(cat_files)} .sas7bcat files. "
                    f"Using the first one: {cat_files[0]}",
                    UserWarning,
                )
            cat_path = z.extract(cat_files[0], path=temp_base)

    return sas_path, cat_path


# ---------------- Metadata convenience ----------------


def _build_value_label_lookup(meta: dict) -> dict[str, dict[str, str]]:
    """
    Build {set_name: {value_as_string: label}} from MetaOut.value_labels.
    """
    out: dict[str, dict[str, str]] = {}
    for vl in meta.get("value_labels", []):
        out[vl["set_name"]] = vl["mapping"]
    return out


def _column_label_map(meta: dict) -> dict[str, dict[str, Any]]:
    """
    Build per-column metadata with column label and label_set:
      {col_name: {"label": Optional[str], "label_set": Optional[str], "fmt": Optional[str]}}
    """
    out: dict[str, dict[str, Any]] = {}
    for v in meta.get("vars", []):
        out[v["name"]] = {
            "label": v.get("label"),
            "label_set": v.get("label_set"),
            "fmt": v.get("fmt"),
        }
    return out


def get_column_labels(meta: dict) -> dict[str, str | None]:
    """
    Convenience: {col_name: label_or_None}
    """
    col_meta = _column_label_map(meta)
    return {k: (v.get("label")) for k, v in col_meta.items()}


def get_value_labels_for_column(meta: dict, col_name: str) -> dict[str, str] | None:
    """
    If the column has a label_set, return its {raw_value_string: human_label} mapping; else None.
    """
    col_meta = _column_label_map(meta)
    lblsets = _build_value_label_lookup(meta)

    info = col_meta.get(col_name)
    if not info:
        return None
    set_name = info.get("label_set")
    if not set_name:
        return None
    return lblsets.get(set_name)


# ---------------- haven::as_factor analogues (fast, dtype-aware) ----------------


def _typed_value_labels_for_dtype(
    value_labels: dict[str, str],
    dtype: pl.DataType,
) -> pl.DataFrame:
    """
    Return a tiny 2-col mapping DataFrame with the mapping key coerced *toward* dtype
    when reasonable; otherwise keep Utf8. Columns: ("__svy_key", "__svy_label").
    """
    keys = list(value_labels.keys())
    labs = list(value_labels.values())

    key_series = pl.Series("__svy_key", keys, dtype=pl.Utf8)
    # Try casting numeric-ish targets
    try_cast = dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    )
    if try_cast:
        try:
            key_series = key_series.cast(dtype, strict=True)
        except Exception:
            # keep Utf8 if cast fails anywhere
            key_series = pl.Series("__svy_key", keys, dtype=pl.Utf8)

    return pl.DataFrame(
        {
            "__svy_key": key_series,
            "__svy_label": pl.Series(labs, dtype=pl.Utf8),
        }
    )


def as_factor_expr(
    col: str | pl.Expr,
    *,
    value_labels: dict[str, str] | None,
    levels: str = "default",  # "default" | "labels" | "values" | "both"
    ordered: bool = False,
) -> pl.Expr:
    levels = levels.lower()
    if levels not in {"default", "labels", "values", "both"}:
        raise ValueError("levels must be one of {'default','labels','values','both'}")

    col_expr = pl.col(col) if isinstance(col, str) else col

    if not value_labels:
        # No labels: cast directly to categorical
        return col_expr.cast(pl.Categorical(ordering="physical" if ordered else "lexical"))

    # Build Utf8 mapping (works regardless of the column's real dtype)
    mapping_utf8 = pl.DataFrame(
        {
            "__svy_key": list(value_labels.keys()),
            "__svy_label": list(value_labels.values()),
        }
    ).with_columns(
        pl.col("__svy_key").cast(pl.Utf8),
        pl.col("__svy_label").cast(pl.Utf8),
    )

    # Compare as strings (avoid Expr.meta.output_type() and pl.datatypes)
    key_expr = col_expr.cast(pl.Utf8)
    repl = dict(
        zip(
            mapping_utf8["__svy_key"].to_list(),
            mapping_utf8["__svy_label"].to_list(),
        )
    )
    label_expr = key_expr.replace(repl)

    if levels in {"default", "both"}:
        if levels == "both":
            disp = (
                pl.when(label_expr.is_not_null() & key_expr.is_not_null())
                .then(pl.concat_str(["[", key_expr, "] ", label_expr]))
                .when(label_expr.is_not_null())
                .then(label_expr)
                .otherwise(key_expr)
            )
        else:
            disp = pl.when(label_expr.is_not_null()).then(label_expr).otherwise(key_expr)
        return disp.cast(pl.Categorical(ordering="physical" if ordered else "lexical"))

    elif levels == "labels":
        return label_expr.cast(pl.Categorical(ordering="physical" if ordered else "lexical"))

    else:  # "values"
        return key_expr.cast(pl.Categorical(ordering="physical" if ordered else "lexical"))


def as_factor_df(
    df: pl.DataFrame,
    meta: dict,
    *,
    only_labelled: bool = True,
    levels: str = "default",
    ordered: bool = False,
    suffix: str = "_factor",
) -> pl.DataFrame:
    """
    Convert labelled columns in a DataFrame to categorical columns (appended with `suffix`).
    If only_labelled=False, try to convert every column for which a label set exists.
    """
    col_meta = _column_label_map(meta)
    lblsets = _build_value_label_lookup(meta)

    exprs: list[pl.Expr] = []
    for name in df.columns:
        info = col_meta.get(name)
        set_name = (info or {}).get("label_set") if info else None
        has_labels = set_name in lblsets if set_name else False

        if not has_labels and only_labelled:
            # keep column unchanged
            exprs.append(pl.col(name))
            continue

        value_labels = lblsets.get(set_name, {}) if set_name else {}
        exprs.append(
            as_factor_expr(
                pl.col(name), value_labels=value_labels, levels=levels, ordered=ordered
            ).alias(f"{name}{suffix}")
        )

    return df.with_columns(exprs)


def apply_value_labels(
    df: pl.DataFrame,
    meta: dict,
    *,
    levels: str = "default",
    ordered: bool = False,
) -> pl.DataFrame:
    out = df
    for col in df.columns:
        mapping = get_value_labels_for_column(meta, col)
        if not mapping:
            continue

        # optional: coerce mapping keys to match the column dtype
        dtype = df.schema[col]

        def _cast_key(k):
            if dtype == pl.Utf8:
                return str(k)
            if dtype in (
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ):
                try:
                    return int(k)
                except Exception:
                    return k
            if dtype in (pl.Float32, pl.Float64):
                try:
                    return float(k)
                except Exception:
                    return k
            return k

        mapping_cast = {_cast_key(k): v for k, v in mapping.items()}

        out = out.with_columns(
            as_factor(s=out[col], labels=mapping_cast, levels=levels, ordered=ordered).alias(col)
        )
    return out


# ---------------- SAS READERS ----------------


def read_xpt(
    data_path: str | os.PathLike,
    *,
    n_max: int | None = None,
    # mirror read_sas post-processing knobs for consistency
    coerce_temporals: bool = True,  # XPT usually needs this
    zap_empty_str: bool = False,
    factorize: bool = False,
    levels: str = "default",
    ordered: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Read a SAS Transport (XPT) file natively via svyreadstat_rs, returning
    (polars.DataFrame, metadata_dict). No pandas/pyreadstat.
    """
    data_path = os.fspath(data_path)
    n_max = _normalize_n_max(n_max)

    if not hasattr(native, "df_parse_xpt_file"):  # type: ignore[attr-defined]
        raise RuntimeError(
            "svyreadstat_rs.df_parse_xpt_file is not available. "
            "Implement the XPT reader in the native layer."
        )

    # Native returns Arrow IPC bytes + JSON metadata
    ipc_bytes, meta_json = native.df_parse_xpt_file(  # type: ignore[attr-defined]
        data_path, n_max
    )

    bio = io.BytesIO(ipc_bytes)
    try:
        df = pl.read_ipc(bio, memory_map=False)  # Arrow IPC file
    except ComputeError:
        bio.seek(0)
        df = pl.read_ipc_stream(bio)  # Fallback: IPC stream

    meta: Dict[str, Any] = json.loads(meta_json)

    # Optional post-processing (same order as read_sas)
    if coerce_temporals:
        from svy_io.temporals import coerce_sas_temporals  # type: ignore

        # If SAS formats are missing, infer from variable names so coercion works.
        for v in meta.get("vars", []):
            if not v.get("fmt"):
                name = (v.get("name") or "").lower()
                if "datetime" in name or "timestamp" in name or name.endswith("_dt"):
                    v["fmt"] = "DATETIME"
                elif name.endswith("_time") or name == "time":
                    v["fmt"] = "TIME"
                elif "date" in name:
                    v["fmt"] = "DATE"

        df = coerce_sas_temporals(df, meta)

        # Ensure DATETIME columns are actually pl.Datetime (writer may have truncated to Date)
        fixes: list[pl.Expr] = []
        for v in meta.get("vars", []):
            if (
                (v.get("fmt") or "").upper().startswith("DATETIME")
                and v.get("name") in df.columns
                and df.schema.get(v["name"]) == pl.Date
            ):
                fixes.append(pl.col(v["name"]).cast(pl.Datetime))
        if fixes:
            df = df.with_columns(fixes)

    if zap_empty_str:
        from svy_io.zap import zap_empty  # type: ignore

        df = zap_empty(df)

    if factorize:
        df = apply_value_labels(df, meta, levels=levels, ordered=ordered)

    # Hydrate tagged NA (harmless for XPT)
    df = _hydrate_tagged_na(df, meta)

    return df, meta


def read_sas(
    data_path: str,
    *,
    catalog_path: str | None = None,
    encoding: str | None = None,
    catalog_encoding: str | None = None,
    cols_skip: list[str] | None = None,
    n_max: int | None = None,
    rows_skip: int = 0,
    # New optional post-processing toggles:
    coerce_temporals: bool = False,
    zap_empty_str: bool = False,
    factorize: bool = False,
    levels: str = "default",  # "default" | "labels" | "values" | "both"
    ordered: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Read a SAS7BDAT dataset (optionally with a SAS7BCAT catalog for value labels).
    Supports reading from zip archives containing .sas7bdat files.

    Returns (polars.DataFrame, metadata_dict)
    """
    # Auto-dispatch if an XPT/XPORT path was passed here by mistake
    if str(data_path).lower().endswith((".xpt", ".xport")):
        return read_xpt(
            data_path,
            n_max=n_max,
            coerce_temporals=coerce_temporals,
            zap_empty_str=zap_empty_str,
            factorize=factorize,
            levels=levels,
            ordered=ordered,
        )

    # Validate/normalize n_max first
    n_max = _normalize_n_max(n_max)

    # Fast-path: explicitly requesting zero rows
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

    data_path = _as_path_like(data_path)

    # Handle zip files
    if str(data_path).lower().endswith(".zip"):
        extracted_path, extracted_catalog = _maybe_from_zip(data_path)
        data_path = extracted_path
        # Use extracted catalog if no explicit catalog_path was provided
        if catalog_path is None and extracted_catalog:
            catalog_path = extracted_catalog
    elif catalog_path is not None:
        catalog_path = _as_path_like(catalog_path)

    ipc_bytes, meta_json = native.df_parse_sas_file(  # type: ignore[attr-defined]
        data_path,
        catalog_path,
        encoding,
        catalog_encoding,
        cols_skip,
        n_max,
        rows_skip,
    )

    # Robust loader: try FILE first; if footer is missing, use STREAM.
    bio = io.BytesIO(ipc_bytes)
    try:
        df = pl.read_ipc(bio, memory_map=False)
    except ComputeError as e:
        if "InvalidFooter" in str(e):
            bio.seek(0)
            df = pl.read_ipc_stream(bio)
        else:
            raise

    # Decode metadata JSON
    meta: Dict[str, Any] = json.loads(meta_json)

    # Optional post-processing (order chosen to mirror typical haven workflows)
    if coerce_temporals:
        from svy_io.temporals import coerce_sas_temporals  # type: ignore

        # If formats are missing, infer from variable names so coercion works.
        for v in meta.get("vars", []):
            if not v.get("fmt"):
                name = (v.get("name") or "").lower()
                if "datetime" in name or "timestamp" in name or name.endswith("_dt"):
                    v["fmt"] = "DATETIME"
                elif name.endswith("_time") or name == "time":
                    v["fmt"] = "TIME"
                elif "date" in name:
                    v["fmt"] = "DATE"

        df = coerce_sas_temporals(df, meta)

        # Ensure DATETIME columns are actually pl.Datetime (in case upstream truncated)
        fixes: list[pl.Expr] = []
        for v in meta.get("vars", []):
            if (
                (v.get("fmt") or "").upper().startswith("DATETIME")
                and v.get("name") in df.columns
                and df.schema.get(v["name"]) == pl.Date
            ):
                fixes.append(pl.col(v["name"]).cast(pl.Datetime))
        if fixes:
            df = df.with_columns(fixes)

    if zap_empty_str:
        from svy_io.zap import zap_empty  # type: ignore

        df = zap_empty(df)

    if factorize:
        df = apply_value_labels(df, meta, levels=levels, ordered=ordered)

    # Hydrate tagged NA
    df = _hydrate_tagged_na(df, meta)

    return df, meta


def read_sas_arrow(
    data_path: str,
    *,
    catalog_path: str | None = None,
    encoding: str | None = None,
    catalog_encoding: str | None = None,
    cols_skip: list[str] | None = None,
    n_max: int | None = None,
    rows_skip: int = 0,
):
    """
    Same as read_sas, but returns (pyarrow.Table, meta_dict) and preserves
    Arrow field metadata (e.g., b'label', b'label_set', b'format') and schema
    metadata (e.g., b'file_label' when present).
    """
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc

    from pyarrow import ArrowInvalid

    n_max = _normalize_n_max(n_max)
    if n_max == 0:
        # empty table with no fields; keep behavior consistent
        empty = pa.table({})
        meta = {
            "file_label": None,
            "vars": [],
            "value_labels": [],
            "user_missing": [],
            "n_rows": 0,
        }
        return empty, meta

    ipc_bytes, meta_json = native.df_parse_sas_file(  # type: ignore[attr-defined]
        data_path, catalog_path, encoding, catalog_encoding, cols_skip, n_max, rows_skip
    )

    bio = io.BytesIO(ipc_bytes)
    # Use FileReader first (Rust likely wrote a file IPC); fallback to stream
    try:
        table = pa_ipc.open_file(bio).read_all()
    except ArrowInvalid:  # type: ignore[attr-defined]
        bio.seek(0)
        table = pa_ipc.open_stream(bio).read_all()

    meta = json.loads(meta_json)
    return table, meta


# ---------------- WRITERS ----------------


def write_xpt(
    df: pl.DataFrame,
    path: str | Path,
    *,
    version: int = 8,
    name: str | None = None,
    label: str | None = None,
    adjust_tz: bool = True,
) -> None:
    """
    Write a Polars DataFrame to SAS Transport (XPT) format (v5 or v8).

    Primary path: svyreadstat_rs.df_write_xpt_file (Arrow IPC).
    Fallback (if native fails or writes 0 bytes): pyreadstat.write_xport().
    """
    import io
    import os

    from pathlib import Path

    # Normalize path to str (PyO3 expects str)
    path = os.fspath(path)

    # --- Validate/normalize header fields ---
    if version not in (5, 8):
        raise ValueError(f"version must be 5 or 8, got {version}")

    max_len = 8 if version == 5 else 32
    if name is None:
        # Default to filename stem (haven/pyreadstat do similar)
        name = Path(path).stem[:max_len]
    elif len(name) > max_len:
        raise ValueError(
            f"name must be <= {max_len} characters for version {version}, got {len(name)}"
        )

    if label is not None and len(label) > 40:
        raise ValueError(f"label must be <= 40 characters, got {len(label)}")

    # --- XPT requires numeric == double; make that explicit ---
    int_cols = [
        c
        for c, dt in df.schema.items()
        if dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]
    if int_cols:
        df = df.with_columns([pl.col(c).cast(pl.Float64) for c in int_cols])

    # Temporal adjustment (mirrors haven's adjust_tz behavior)
    if adjust_tz:
        from .stata import _adjust_temporals

        df = _adjust_temporals(df, adjust_tz=True)

    # --- Try the native writer first ---
    def _try_native(ipc_bytes: bytes) -> None:
        # Raises RuntimeError on native error
        native.df_write_xpt_file(
            ipc_bytes,
            path,
            version=version,
            name=name,
            label=label,
        )

    # Build Arrow IPC **file** bytes, then retry with **stream** if footer complaints show up.
    try:
        bio_file = io.BytesIO()
        df.write_ipc(bio_file)  # file format, includes footer
        _try_native(bio_file.getvalue())
    except RuntimeError as e:
        msg = str(e)
        if "InvalidFooter" in msg or "correct footer" in msg or "footer" in msg:
            # Native likely expects a stream – retry with stream bytes
            bio_stream = io.BytesIO()
            try:
                df.write_ipc_stream(bio_stream)
                _try_native(bio_stream.getvalue())
            except AttributeError:
                # Very old Polars: stream via PyArrow
                import pyarrow as pa
                import pyarrow.ipc as pa_ipc

                table = df.to_arrow()
                sink = pa.BufferOutputStream()
                with pa_ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(table)
                _try_native(sink.getvalue().to_pybytes())
        else:
            # Some other native error—bubble up to the fallback
            raise

    # If we got here without exception but the file is empty, the native layer didn't finalize.
    if os.path.exists(path) and os.stat(path).st_size == 0:
        # ---- Fallback: pyreadstat (guaranteed-good) ----
        try:
            import pyreadstat  # type: ignore
        except ImportError as ie:
            raise RuntimeError(
                "Native XPT writer returned without error but produced an empty file; "
                "fallback to pyreadstat was not possible because pyreadstat is not installed. "
                "Install pyreadstat or fix the native writer."
            ) from ie

        # Convert to pandas and write via pyreadstat
        pdf = df.to_pandas()
        # pyreadstat uses `table_name` and `file_format_version` kwargs
        pyreadstat.write_xport(
            pdf,
            path,
            file_label=(label or ""),
            table_name=name[:max_len] if name else "DATASET",
            file_format_version=version,
        )
        # Double-check the result
        if os.stat(path).st_size == 0:
            raise RuntimeError("pyreadstat fallback also produced an empty XPT file.")


def write_sas(df: pl.DataFrame, path: str | Path) -> None:
    """
    Write SAS7BDAT file (DEPRECATED - use write_xpt instead).
    """
    import warnings

    warnings.warn(
        "write_sas() is deprecated and produces files that SAS cannot read. "
        "Use write_xpt() instead for reliable SAS-compatible output.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "write_sas() is not implemented. Use write_xpt() for SAS-compatible output."
    )
