# svy/errors/_selection.py
"""Selection: ``sample.sampling``, sample size and power."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ._base import SvyError, SvyWarning
from ._codes import Code


@dataclass(eq=False)
class SelectionError(SvyError):
    _DEFAULT_CODE: ClassVar[str] = Code.SELECTION_ERROR


@dataclass(eq=False)
class SelectionWarning(SvyWarning):
    pass
