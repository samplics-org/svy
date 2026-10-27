# svy/errors/_data.py
"""Data: ``sample.wrangling`` and checks on the data itself."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ._base import SvyError, SvyWarning


@dataclass(eq=False)
class DataError(SvyError):
    _DEFAULT_CODE: ClassVar[str] = "DATA_ERROR"


@dataclass(eq=False)
class DataWarning(SvyWarning):
    pass
