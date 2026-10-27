# svy/errors/_estimation.py
"""Estimation: ``sample.estimation``, ``sample.categorical``, contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ._base import SvyError, SvyWarning
from ._codes import Code


@dataclass(eq=False)
class EstimationError(SvyError):
    _DEFAULT_CODE: ClassVar[str] = Code.ESTIMATION_ERROR


@dataclass(eq=False)
class EstimationWarning(SvyWarning):
    pass
