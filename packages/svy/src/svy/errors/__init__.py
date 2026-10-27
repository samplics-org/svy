"""svy's errors and warnings: ``SvyError`` stops a call, ``SvyWarning`` reports without stopping."""

from ._api import UnknownNameError, UsageError
from ._base import Severity, SvyError, SvyWarning
from ._dataset import DatasetError
from ._dimension import DimensionError
from ._findings import WarnCode, WarningStore
from ._io import IoError
from ._label import LabelError
from ._method import MethodError
from ._model import ModelError
from ._serialization import SerializationError
from ._singleton import SingletonError
from ._weighting import WeightingError


__all__ = [
    "DatasetError",
    "DimensionError",
    "IoError",
    "LabelError",
    "MethodError",
    "ModelError",
    "SerializationError",
    "Severity",
    "SingletonError",
    "SvyError",
    "SvyWarning",
    "UnknownNameError",
    "UsageError",
    "WarnCode",
    "WarningStore",
    "WeightingError",
]


def __getattr__(name: str):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    raise UnknownNameError.unknown(
        owner="svy.errors", label="svy.errors", name=name, available=__all__
    )
