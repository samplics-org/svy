# svy/errors/_data.py
"""Data: ``sample.wrangling`` and checks on the data itself."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ._base import SvyError, SvyWarning
from ._codes import Code


@dataclass(eq=False)
class DataError(SvyError):
    _DEFAULT_CODE: ClassVar[str] = Code.DATA_ERROR


@dataclass(eq=False)
class DataWarning(SvyWarning):
    pass
