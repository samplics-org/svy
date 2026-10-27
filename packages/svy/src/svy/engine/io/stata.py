# svy/engine/io/stata.py
from __future__ import annotations

import logging
import re

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import polars as pl

from svy.errors import Code, IoWarning
from svy.errors._findings import emit_finding
from svy.metadata import MetadataStore

from .core import (
    _canonical_code,
    build_metadata_for_export,
    to_polars,
    to_writer_table,
)


log = logging.getLogger(__name__)

_DTA_CODE_MIN, _DTA_CODE_MAX = -2147483647, 2147483620  # int32, minus . and .a-.z

try:
    import svy_io as sio
except Exception as e:
    raise ImportError("svy.engine.io requires 'svy-io' (pip install svy-io).") from e


def _read_stata(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    encoding: str | None = None,
    **kwargs,
) -> Tuple[pl.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Read a Stata file.

    Returns
    -------
    Tuple[pl.DataFrame, dict, dict]
        - DataFrame with the data
        - Raw metadata dict (to be imported via import_labels_from_svyio_meta)
        - File info dict
    """
    # 1. Forward the file encoding (iconv name, or "utf8-lossy")
    if encoding is not None:
        kwargs["encoding"] = encoding

    # 2. Call Engine
    # 'columns' is applied here, after the read, and NOT passed to sio.
    kwargs.setdefault("coerce_temporals", True)
    res = sio.read_stata(str(path), **kwargs)

    # 3. Normalize Result
    if isinstance(res, tuple):
        if len(res) == 2:
            raw_data, raw_meta = res
            file_info = {}
        elif len(res) >= 3:
            raw_data = res[0]
            raw_meta = res[1]
            file_info = res[2]
        else:
            raise ValueError(f"Unexpected tuple length from engine: {len(res)}")
    elif isinstance(res, dict):
        raw_data = res["data"]
        raw_meta = res.get("metadata", {})
        file_info = res.get("file_info", {})
    else:
        raise TypeError(f"Unexpected return type from engine: {type(res)}")

    # 4. Convert to Polars
    df = to_polars(raw_data)

    # 5. Apply Column Selection
    if columns is not None:
        df = df.select(list(columns))

    # 6. Return raw metadata (caller will import into MetadataStore)
    return df, raw_meta, file_info


def _stata_temporals(df: pl.DataFrame) -> pl.DataFrame:
    """Times and durations as milliseconds; Stata has no type for them. Dates go through as %td/%tc."""
    exprs = {}
    for name, dtype in df.schema.items():
        col = pl.col(name)
        if dtype == pl.Time:
            exprs[name] = (col.cast(pl.Int64) // 1_000_000).cast(pl.Float64)
        elif isinstance(dtype, pl.Duration):
            exprs[name] = col.dt.total_milliseconds().cast(pl.Float64)
    if not exprs:
        return df
    emit_finding(
        IoWarning(
            code=Code.IO_TEMPORAL_AS_NUMBER,
            title="Times written as numbers",
            detail=f"{', '.join(exprs)} are written as milliseconds (since midnight, for times of day); "
            "Stata has no time-of-day or duration type.",
            where="io.write_stata",
        )
    )
    return df.with_columns(**exprs)


def _write_stata(
    df: pl.DataFrame,
    store: MetadataStore,
    path: str | Path,
    *,
    version: int | None = None,
    encoding: str | None = None,
    **kwargs,
) -> None:
    """
    Write a Stata file.

    Parameters
    ----------
    df : pl.DataFrame
        The data to write.
    store : MetadataStore
        The metadata store with labels.
    path : str | Path
        Output file path.
    version : int | None
        Stata version (default 15).
    encoding : str | None
        File encoding (ignored for Stata).
    """
    meta = build_metadata_for_export(df, store)
    df = _stata_temporals(df)
    table = to_writer_table(df)

    if encoding is not None:
        log.warning("Stata engine does not support 'encoding'. It will be ignored.")

    # Stata labels integer codes of numeric variables only, below its missing-value
    # codes; SPSS also labels text and fractional codes, so a converted file can carry
    # sets Stata cannot hold.
    unfit = sorted(
        var
        for var, labels in meta["value_labels"].items()
        if not df.schema[var].is_numeric()
        or not all(
            re.fullmatch(r"-?\d+", code := _canonical_code(k))
            and _DTA_CODE_MIN <= int(code) <= _DTA_CODE_MAX
            for k in labels
        )
    )
    if unfit:
        emit_finding(
            IoWarning(
                code=Code.IO_LABELS_NOT_WRITTEN,
                title="Value labels not written",
                detail=f"Stata labels integer codes of numeric variables only, so value labels for "
                f"{', '.join(unfit)} are not written. Use write_spss to keep them.",
                where="io.write_stata",
            )
        )
        meta["value_labels"] = {v: m for v, m in meta["value_labels"].items() if v not in unfit}

    sio.write_stata(
        table,
        str(path),
        var_labels=meta["var_labels"] or None,
        value_labels=meta["value_labels"] or None,
        version=version if version is not None else 15,
        **kwargs,
    )
