# svy/errors/_design.py
"""Design: ``Sample``, ``Design``, replicate weights, ``combine_samples``, variance setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ._base import SvyError, SvyWarning
from ._codes import Code


@dataclass(eq=False)
class DesignError(SvyError):
    _DEFAULT_CODE: ClassVar[str] = Code.DESIGN_ERROR


@dataclass(eq=False)
class DesignWarning(SvyWarning):
    pass
