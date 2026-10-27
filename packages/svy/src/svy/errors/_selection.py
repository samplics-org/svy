# svy/errors/_selection.py
"""Selection: ``sample.sampling``, sample size and power."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ._base import SvyError, SvyWarning


@dataclass(eq=False)
class SelectionError(SvyError):
    _DEFAULT_CODE: ClassVar[str] = "SELECTION_ERROR"


@dataclass(eq=False)
class SelectionWarning(SvyWarning):
    pass
