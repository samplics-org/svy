# svy/engine/io/sas.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import polars as pl

from svy.metadata import MetadataStore

from .core import (
    to_polars,
    to_writer_table,
)


# ---- hard-require your ReadStat port (no fallbacks) ----
try:
    import svy_io as sio  # your C/Rust-backed ReadStat port
except Exception as e:  # pragma: no cover
    raise ImportError("svy.engine.io requires 'svy-io' (pip install svy-io).") from e


def _read_sas(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    encoding: str | None = None,
    **kwargs,
) -> Tuple[pl.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Read a SAS file.

    Returns
    -------
    Tuple[pl.DataFrame, dict, dict]
        - DataFrame with the data
        - Raw metadata dict (to be imported via import_labels_from_svyio_meta)
        - File info dict
    """
    # 1. Call Engine
    # SAS supports encoding, so we pass it. It does NOT support 'columns'.
    res = sio.read_sas(str(path), encoding=encoding, **kwargs)

    # 2. Normalize Result (Robustness for Tests vs Production)
    if isinstance(res, tuple):
        if len(res) == 2:
            raw_data, raw_meta = res
            file_info = {}
        elif len(res) >= 3:
            raw_data = res[0]
            raw_meta = res[1]
            file_info = res[2]
        else:
            raise ValueError(f"Unexpected tuple length: {len(res)}")
    elif isinstance(res, dict):
        raw_data = res["data"]
        raw_meta = res.get("metadata", {})
        file_info = res.get("file_info", {})
    else:
        raise TypeError(f"Unexpected return type: {type(res)}")

    # 3. Convert to Polars
    df = to_polars(raw_data)

    # 4. Select Columns
    if columns is not None:
        df = df.select(list(columns))

    # 5. Return raw metadata (caller will import into MetadataStore)
    return df, raw_meta, file_info


def _write_sas(
    df: pl.DataFrame,
    store: MetadataStore,
    path: str | Path,
    *,
    format: str | None = None,
    encoding: str | None = None,
    **kwargs,
) -> None:
    """
    Write a SAS file.

    Parameters
    ----------
    df : pl.DataFrame
        The data to write.
    store : MetadataStore
        The metadata store with labels.
    path : str | Path
        Output file path.
    format : str | None
        SAS format (e.g., 'sas7bdat').
    encoding : str | None
        File encoding.
    """
    variables_meta: Dict[str, Any] = {"variables": {}}

    for var in df.columns:
        meta = store.get(var)
        entry: Dict[str, Any] = {"label": None, "values": {}, "missing": []}

        if meta is not None:
            entry["label"] = meta.label

            # Get value labels (direct or resolved from scheme)
            resolved = store.resolve_labels(var)
            if resolved.has_value_labels:
                entry["values"] = dict(resolved.value_labels)

            # Get missing codes
            if meta.missing is not None:
                entry["missing"] = list(meta.missing.codes)

        variables_meta["variables"][var] = entry

    table = to_writer_table(df)
    sio.write_sas(  # type: ignore[attr-defined]
        table, str(path), metadata=variables_meta, format=format, encoding=encoding, **kwargs
    )
