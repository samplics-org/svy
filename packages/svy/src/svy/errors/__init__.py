"""svy's errors and warnings: ``SvyError`` stops a call, ``SvyWarning`` reports without stopping.

- Each class names the part of svy that raised it; ``ModelError.ValueError`` is also a ``ValueError``.
- ``except svy.errors.WeightingError`` catches one area; ``category=WeightingWarning`` filters one.
"""

from ._api import UnknownNameError, UsageError
from ._base import _KINDS, Severity, SvyError, SvyWarning
from ._data import DataError, DataWarning
from ._dataset import DatasetError, DatasetWarning
from ._design import DesignError, DesignWarning
from ._estimation import EstimationError, EstimationWarning
from ._findings import WarnCode, WarningStore
from ._io import IoError, IoWarning
from ._label import LabelError, LabelWarning
from ._model import ModelError, ModelWarning
from ._selection import SelectionError, SelectionWarning
from ._serialization import SerializationError
from ._singleton import SingletonError, SingletonWarning
from ._weighting import WeightingError, WeightingWarning


__all__ = [
    "DataError",
    "DataWarning",
    "DatasetError",
    "DatasetWarning",
    "DesignError",
    "DesignWarning",
    "EstimationError",
    "EstimationWarning",
    "IoError",
    "IoWarning",
    "LabelError",
    "LabelWarning",
    "ModelError",
    "ModelWarning",
    "SelectionError",
    "SelectionWarning",
    "SerializationError",
    "Severity",
    "SingletonError",
    "SingletonWarning",
    "SvyError",
    "SvyWarning",
    "UnknownNameError",
    "UsageError",
    "WarnCode",
    "WarningStore",
    "WeightingError",
    "WeightingWarning",
]


# Tracebacks and reprs name the public home, svy.errors.ModelError.ValueError, not
# the private module; pickle finds each class through it.
for _name in __all__:
    _cls = globals()[_name]
    if isinstance(_cls, type) and issubclass(_cls, (SvyError, SvyWarning)):
        _cls.__module__ = __name__
        for _kind in _KINDS:
            _pair = _cls.__dict__.get(_kind.__name__)
            if _pair is not None:
                _pair.__module__ = __name__
del _name, _cls


def __getattr__(name: str):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    raise UnknownNameError.unknown(
        owner="svy.errors", label="svy.errors", name=name, available=__all__
    )
