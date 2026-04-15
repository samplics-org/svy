# svy/engine/io/core.py
from __future__ import annotations

from typing import Any, Dict

import polars as pl

from svy.core.enumerations import MetadataSource
from svy.core.types import Category
from svy.metadata import MetadataStore, VariableMeta


# ---- hard-require your ReadStat port (no fallbacks) ----
try:
    import svy_io as sio  # your C/Rust-backed ReadStat port
except Exception as e:  # pragma: no cover
    raise ImportError("svy.engine.io requires 'svy-io' (pip install svy-io).") from e

# --------------------- shared helpers --------------------- #


def _py_scalar(x: Any) -> Any:
    """Normalize numpy/arrow scalars to plain Python primitives."""
    try:
        import numpy as np  # lazy

        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    if hasattr(x, "as_py"):
        try:
            return x.as_py()
        except Exception:
            pass
    return x


def to_polars(df_like: Any) -> pl.DataFrame:
    """Best-effort conversion to Polars without adding hard deps."""
    if isinstance(df_like, pl.DataFrame):
        return df_like
    # Try Arrow first (common interop path), then pandas/iterables
    try:
        import pyarrow as pa

        if isinstance(df_like, pa.Table):
            return pl.from_arrow(df_like)  # type: ignore[return-value]
    except Exception:
        pass
    try:
        return pl.from_dataframe(df_like)  # handles pandas/arrow table
    except Exception:
        try:
            return pl.DataFrame(df_like)
        except Exception as e:
            raise TypeError("svy-io returned an unsupported table type.") from e


def to_writer_table(df: pl.DataFrame) -> Any:
    """
    Produce a table type that svy-io's writers accept.
    Prefer Polars; fall back to Arrow; then Pandas.
    """
    # Many svy-io backends accept Polars directly
    try:
        _ = df.schema  # cheap sanity
        return df
    except Exception:
        pass
    try:
        return df.to_arrow()
    except Exception:
        pass
    return df.to_pandas()


def import_labels_from_svyio_meta(
    store: MetadataStore,
    meta: Dict[str, Any],
) -> None:
    """
    Import variable and value labels from svy-io metadata into a MetadataStore.

    Accepts either:
      A) {"variables": {
             "<var>": {"label": "...", "values": {code: "label", ...}}
         }}
      B) {"vars": [{"name":..., "label":..., "label_set":...}],
          "value_labels": [{"set_name":..., "mapping": {...}}, ...]}

    Parameters
    ----------
    store : MetadataStore
        The metadata store to populate.
    meta : dict
        The metadata dict from svy-io.
    """
    # New/normalized form (A)
    variables = meta.get("variables")
    if isinstance(variables, dict) and variables:
        for var, vmeta in variables.items():
            var_label = vmeta.get("label") or None
            values = vmeta.get("values") or {}

            existing = store.get(var)
            if existing is not None:
                # Update existing metadata
                new_meta = existing.clone(
                    label=var_label if var_label else existing.label,
                    value_labels=dict(values) if values else existing.value_labels,
                    source=MetadataSource.IMPORTED,
                )
                store.set(var, new_meta)
            else:
                # Create new metadata
                store.set(
                    var,
                    VariableMeta(
                        name=var,
                        label=var_label,
                        value_labels=dict(values) if values else None,
                        source=MetadataSource.IMPORTED,
                    ),
                )
        return

    # Classic split form (B)
    vars_list = meta.get("vars") or []
    vlabels = meta.get("value_labels") or []
    if vars_list or vlabels:
        lblsets: dict[str, dict[Category, str]] = {
            vl.get("set_name"): (vl.get("mapping") or {}) for vl in vlabels
        }
        for v in vars_list:
            var = v.get("name")
            if not var:
                continue
            var_label = v.get("label") or None
            set_name = v.get("label_set")
            values = lblsets.get(set_name) or {}

            existing = store.get(var)
            if existing is not None:
                # Update existing metadata
                new_meta = existing.clone(
                    label=var_label if var_label else existing.label,
                    value_labels=dict(values) if values else existing.value_labels,
                    source=MetadataSource.IMPORTED,
                )
                store.set(var, new_meta)
            else:
                # Create new metadata
                store.set(
                    var,
                    VariableMeta(
                        name=var,
                        label=var_label,
                        value_labels=dict(values) if values else None,
                        source=MetadataSource.IMPORTED,
                    ),
                )


def build_metadata_for_export(
    df: pl.DataFrame,
    store: MetadataStore,
) -> Dict[str, Any]:
    """
    Produce {"var_labels": {...}, "value_labels": {...}} for svy-io writers.

    - var_labels: {var: "Variable label"}
    - value_labels: {var: {code: "text", ...}}

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame being written.
    store : MetadataStore
        The metadata store.

    Returns
    -------
    dict
        Metadata dict suitable for svy-io writers.
    """
    var_labels: Dict[str, str] = {}
    value_labels: Dict[str, dict] = {}

    for var in df.columns:
        meta = store.get(var)
        if meta is None:
            continue

        if meta.label:
            var_labels[var] = str(meta.label)

        # Get value labels (direct or resolved from scheme)
        resolved = store.resolve_labels(var)
        if resolved.has_value_labels:
            # ensure codes are JSON-serializable primitives; texts are str
            value_labels[var] = {
                _py_scalar(k): ("" if v is None else str(v))
                for k, v in resolved.value_labels.items()
            }

    return {"var_labels": var_labels, "value_labels": value_labels}


__all__ = [
    "sio",
    "to_polars",
    "to_writer_table",
    "import_labels_from_svyio_meta",
    "build_metadata_for_export",
]
