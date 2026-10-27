# svy/errors/_estimation.py
"""Estimation: ``sample.estimation``, ``sample.categorical``, contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ._base import SvyError, SvyWarning


@dataclass(eq=False)
class EstimationError(SvyError):
    _DEFAULT_CODE: ClassVar[str] = "ESTIMATION_ERROR"


@dataclass(eq=False)
class EstimationWarning(SvyWarning):
    pass
