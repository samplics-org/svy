# svy/engine/io/__init__.py
from .core import (
    build_metadata_for_export,
    import_labels_from_svyio_meta,
    to_polars,
    to_writer_table,
)
from .sas import _read_sas, _write_sas
from .spss import _read_spss, _write_spss
from .stata import _read_stata, _write_stata


# ---- hard-require your ReadStat port (no fallbacks) ----
try:
    import svy_io as sio  # your C/Rust-backed ReadStat port
except Exception as e:  # pragma: no cover
    raise ImportError("svy.engine.io requires 'svy-io' (pip install svy-io).") from e

__all__ = [
    # core helpers
    "sio",
    "to_polars",
    "to_writer_table",
    "import_labels_from_svyio_meta",
    "build_metadata_for_export",
    # format dispatch
    "_read_spss",
    "_write_spss",
    "_read_stata",
    "_write_stata",
    "_read_sas",
    "_write_sas",
]
