# svy/engine/io/stata.py
from __future__ import annotations

import logging

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import polars as pl

from svy.metadata import MetadataStore

from .core import (
    to_polars,
    to_writer_table,
)


log = logging.getLogger(__name__)

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
    # 1. Warning for unsupported arguments
    if encoding is not None:
        log.warning("Stata engine does not support 'encoding'. It will be ignored.")

    # 2. Call Engine
    # CRITICAL: We pass ONLY kwargs.
    # 'columns' and 'encoding' are captured by the signature above and NOT passed to sio.
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
    # 1. Prepare Metadata
    var_labels: Dict[str, str] = {}
    value_labels: Dict[str, Dict[Any, Any]] = {}

    for var in df.columns:
        meta = store.get(var)
        if meta is not None:
            if meta.label:
                var_labels[var] = str(meta.label)

            # Get value labels (direct or resolved from scheme)
            resolved = store.resolve_labels(var)
            if resolved.has_value_labels:
                value_labels[var] = dict(resolved.value_labels)

    # 2. Prepare Table
    table = to_writer_table(df)

    # 3. Handle Encoding
    if encoding is not None:
        log.warning("Stata engine does not support 'encoding'. It will be ignored.")

    # 4. Call Engine
    # We pass var_labels/value_labels instead of 'metadata'
    sio.write_stata(
        table,
        str(path),
        var_labels=var_labels if var_labels else None,
        value_labels=value_labels if value_labels else None,
        version=version if version is not None else 15,
        **kwargs,
    )
