# svy/io/base.py
from __future__ import annotations

import logging
import time

from pathlib import Path
from typing import Any, Callable, cast

import polars as pl

from svy.core.sample import Sample
from svy.core.types import ColumnsArg
from svy.engine.io import (
    _read_sas,
    _read_spss,
    _read_stata,
    _write_sas,
    _write_spss,
    _write_stata,
    import_labels_from_svyio_meta,
)
from svy.errors.io_errors import IoError, map_os_error


log = logging.getLogger(__name__)


# ---------------- internal helpers: error wrapping ---------------- #

_EXPECTED_EXTS = {
    "spss": (".sav", ".zsav"),
    "stata": (".dta",),
    "sas": (".sas7bdat", ".xpt"),
    "csv": (".csv", ".tsv", ".txt"),
    "parquet": (".parquet", ".pqt"),
}


def _summarize_columns(cols: ColumnsArg) -> str:
    if cols is None:
        return "all"
    try:
        preview = list(cols)[:5]
        return f"{len(cols)} (preview={preview})"
    except Exception:
        return "<unknown>"


def _summarize_df(df: Any) -> tuple[Any, Any]:
    try:
        if hasattr(df, "height") and hasattr(df, "columns"):
            return df.height, len(df.columns)
        if hasattr(df, "collect_schema"):
            return "lazy", len(df.collect_schema().names())
    except Exception:
        pass
    return "?", "?"


def _preflight_read(path: str | Path, *, where: str, fmt: str) -> None:
    p = Path(path)
    if not p.exists():
        raise IoError.not_found(where=where, path=p)
    if not p.is_file():
        raise IoError.not_a_file(where=where, path=p)

    valid_exts = _EXPECTED_EXTS.get(fmt)
    if valid_exts and p.suffix.lower() not in valid_exts:
        log.warning(
            "File extension '%s' may not match format '%s' (expected %s)",
            p.suffix,
            fmt,
            valid_exts,
        )


def _wrap_io_op(op_name: str, fmt: str, path: str | Path, engine_fn: Callable, **kwargs) -> Any:
    """
    Generic wrapper to handle logging, timing, and error mapping uniformly.
    """
    where = f"io.{op_name}_{fmt}"
    path_obj = Path(path)

    safe_kw = {k: v for k, v in kwargs.items() if k not in {"df", "store"}}
    log.debug("%s: path=%s, opts=%s", where, path_obj, safe_kw)

    if op_name.startswith("read"):
        _preflight_read(path_obj, where=where, fmt=fmt)

    t0 = time.perf_counter()
    try:
        # --- EXECUTE ENGINE (Fix for SyntaxError) ---

        # 1. If path is already in kwargs (e.g. read_csv(source=...)), just call engine
        if any(k in kwargs for k in ("source", "file", "path")):
            result = engine_fn(**kwargs)

        # 2. READ: Default to positional path (standard for Polars & Internal readers)
        elif op_name.startswith("read"):
            result = engine_fn(path, **kwargs)

        # 3. WRITE: Default to keyword path (standard for Internal writers)
        else:
            result = engine_fn(path=path, **kwargs)

    except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
        log.debug("%s os error: %s", where, e)
        raise map_os_error(e, where=where, path=path_obj) from e
    except RuntimeError as e:
        log.debug("%s parse failed: %s", where, e)
        raise IoError.parse_failed(where=where, fmt=fmt, path=path_obj, engine_msg=str(e)) from e
    except Exception as e:
        log.debug("%s failed: %s", where, e, exc_info=True)
        # Check if it's a known Polars error we want to wrap, otherwise re-raise or wrap generic
        raise IoError.read_failed(where=where, path=path_obj, reason=str(e)) from e

    ms = (time.perf_counter() - t0) * 1000.0

    if op_name.startswith("read"):
        df_part = result[0] if isinstance(result, tuple) else result
        r, c = _summarize_df(df_part)
        log.debug("%s ok: rows=%s, cols=%s, elapsed=%.1f ms", where, r, c, ms)
    else:
        log.debug("%s done in %.1f ms", where, ms)

    return result


# ---------------- Public API: Standard Formats ---------------- #


def read_csv(path: str | Path, **kwargs) -> pl.DataFrame:
    return _wrap_io_op("read", "csv", path, pl.read_csv, source=path, **kwargs)


def scan_csv(path: str | Path, **kwargs) -> pl.LazyFrame:
    return _wrap_io_op("scan", "csv", path, pl.scan_csv, source=path, **kwargs)


def write_csv(sample: Sample | pl.DataFrame, path: str | Path, **kwargs) -> None:
    _raw = sample._data if isinstance(sample, Sample) else sample
    df: pl.DataFrame = (
        cast(pl.DataFrame, _raw)
        if not isinstance(_raw, pl.LazyFrame)
        else cast(pl.DataFrame, _raw.collect())
    )
    _wrap_io_op("write", "csv", path, df.write_csv, file=path, **kwargs)


def read_parquet(path: str | Path, **kwargs) -> pl.DataFrame:
    return _wrap_io_op("read", "parquet", path, pl.read_parquet, source=path, **kwargs)


def scan_parquet(path: str | Path, **kwargs) -> pl.LazyFrame:
    return _wrap_io_op("scan", "parquet", path, pl.scan_parquet, source=path, **kwargs)


def write_parquet(sample: Sample | pl.DataFrame, path: str | Path, **kwargs) -> None:
    _raw = sample._data if isinstance(sample, Sample) else sample
    df: pl.DataFrame = (
        cast(pl.DataFrame, _raw)
        if not isinstance(_raw, pl.LazyFrame)
        else cast(pl.DataFrame, _raw.collect())
    )
    _wrap_io_op("write", "parquet", path, df.write_parquet, file=path, **kwargs)


# ---------------- Public API: Statistical Formats ---------------- #


def read_spss(path: str | Path, *, columns: ColumnsArg = None, **kwargs) -> pl.DataFrame:
    df, _, _ = _wrap_io_op("read", "spss", path, _read_spss, columns=columns, **kwargs)
    return df


def read_spss_with_labels(
    path: str | Path, *, columns: ColumnsArg = None, **kwargs
) -> tuple[pl.DataFrame, dict, dict]:
    return _wrap_io_op("read", "spss", path, _read_spss, columns=columns, **kwargs)


def write_spss(sample: Sample, path: str | Path, **kwargs) -> None:
    _wrap_io_op(
        "write",
        "spss",
        path,
        _write_spss,
        df=sample._data,
        store=sample.meta,
        **kwargs,
    )


def read_stata(path: str | Path, *, columns: ColumnsArg = None, **kwargs) -> pl.DataFrame:
    df, _, _ = _wrap_io_op("read", "stata", path, _read_stata, columns=columns, **kwargs)
    return df


def read_stata_with_labels(
    path: str | Path, *, columns: ColumnsArg = None, **kwargs
) -> tuple[pl.DataFrame, dict, dict]:
    return _wrap_io_op("read", "stata", path, _read_stata, columns=columns, **kwargs)


def write_stata(sample: Sample, path: str | Path, **kwargs) -> None:
    _wrap_io_op(
        "write",
        "stata",
        path,
        _write_stata,
        df=sample._data,
        store=sample.meta,
        **kwargs,
    )


def read_sas(path: str | Path, *, columns: ColumnsArg = None, **kwargs) -> pl.DataFrame:
    df, _, _ = _wrap_io_op("read", "sas", path, _read_sas, columns=columns, **kwargs)
    return df


def read_sas_with_labels(
    path: str | Path, *, columns: ColumnsArg = None, **kwargs
) -> tuple[pl.DataFrame, dict, dict]:
    return _wrap_io_op("read", "sas", path, _read_sas, columns=columns, **kwargs)


def write_sas(sample: Sample, path: str | Path, **kwargs) -> None:
    _wrap_io_op(
        "write",
        "sas",
        path,
        _write_sas,
        df=sample._data,
        store=sample.meta,
        **kwargs,
    )


# ---------------- Ergonomic creators ---------------- #


def _create_sample_with_labels(
    path: str | Path,
    reader_fn: Callable,
    name: str | None = None,
    **kwargs,
) -> Sample:
    """
    Create a Sample from a statistical format file, importing labels into MetadataStore.
    """
    # Read data and raw metadata
    df, raw_meta, _ = reader_fn(path, **kwargs)

    # Create sample (this initializes MetadataStore and infers types)
    sample = Sample(data=df)

    # Import labels from file into MetadataStore
    import_labels_from_svyio_meta(sample.meta, raw_meta)

    if name:
        setattr(sample, "name", name)

    return sample


def _create_sample_simple(
    path: str | Path,
    reader_fn: Callable,
    name: str | None = None,
    **kwargs,
) -> Sample:
    """
    Create a Sample from a simple format (CSV, Parquet) without labels.
    """
    df = reader_fn(path, **kwargs)
    sample = Sample(data=df)

    if name:
        setattr(sample, "name", name)

    return sample


def create_from_csv(path: str | Path, name: str | None = None, **kwargs) -> Sample:
    return _create_sample_simple(path, read_csv, name=name, **kwargs)


def create_from_parquet(path: str | Path, name: str | None = None, **kwargs) -> Sample:
    return _create_sample_simple(path, read_parquet, name=name, **kwargs)


def create_from_spss(path: str | Path, name: str | None = None, **kwargs) -> Sample:
    return _create_sample_with_labels(path, read_spss_with_labels, name=name, **kwargs)


def create_from_stata(path: str | Path, name: str | None = None, **kwargs) -> Sample:
    return _create_sample_with_labels(path, read_stata_with_labels, name=name, **kwargs)


def create_from_sas(path: str | Path, name: str | None = None, **kwargs) -> Sample:
    return _create_sample_with_labels(path, read_sas_with_labels, name=name, **kwargs)


# friendly aliases
create_from_sav = create_from_spss
read_sav = read_spss
read_sav_with_labels = read_spss_with_labels
write_sav = write_spss

create_from_dta = create_from_stata
read_dta = read_stata
read_dta_with_labels = read_stata_with_labels
write_dta = write_stata
