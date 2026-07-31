# svy/engine/io/spss.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import polars as pl

from svy.metadata import MetadataStore

from .core import (
    to_polars,
    to_writer_table,
)


try:
    import svy_io as sio
except Exception as e:
    raise ImportError("svy.engine.io requires 'svy-io' (pip install svy-io).") from e


def _read_spss(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    encoding: str | None = None,
    **kwargs,
) -> Tuple[pl.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Read an SPSS file.

    Returns
    -------
    Tuple[pl.DataFrame, dict, dict]
        - DataFrame with the data
        - Raw metadata dict (to be imported via import_labels_from_svyio_meta)
        - File info dict
    """
    # 1. Call Engine (Capture generic result, DO NOT unpack here)
    res = sio.read_spss(str(path), encoding=encoding, **kwargs)

    # 2. Normalize Result (Handle 2-tuple vs 3-tuple vs Dict)
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


def _write_spss(
    df: pl.DataFrame,
    store: MetadataStore,
    path: str | Path,
    **kwargs,
) -> None:
    """
    Write an SPSS file.

    Parameters
    ----------
    df : pl.DataFrame
        The data to write.
    store : MetadataStore
        The metadata store with labels.
    path : str | Path
        Output file path.

    Notes
    -----
    There is no ``encoding`` parameter: ``svy_io.write_sav`` writes UTF-8 and
    takes none. The old signature accepted one and passed it to a function that
    did not exist.
    """
    var_labels: Dict[str, str] = {}
    value_labels: list[Dict[str, Any]] = []

    for var in df.columns:
        meta = store.get(var)
        if meta is None:
            continue
        if meta.label:
            var_labels[var] = meta.label
        resolved = store.resolve_labels(var)
        if resolved.has_value_labels:
            # SPSS stores value-label keys as strings.
            value_labels.append(
                {"col": var, "labels": {str(k): v for k, v in resolved.labels.items()}}
            )

    table = to_writer_table(df)
    # `write_sav`, not `write_spss`: the latter has never existed in svy-io, and
    # the `# type: ignore[attr-defined]` that used to sit here was the type
    # checker saying so. `user_missing` is deliberately not passed — svy holds
    # no model of it (design 001 §2.2), and an exporter that wants one supplies
    # it here, at the boundary that has the concept.
    sio.write_sav(
        table,
        str(path),
        var_labels=var_labels or None,
        value_labels=value_labels or None,
        **kwargs,
    )
