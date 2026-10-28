# src/svy/size/_normalize.py
"""
Normalization helpers for size subpackage.

Converts user-facing Literal strings to internal enum values used
by the statistical engine.
"""

from __future__ import annotations

from typing import Literal

from svy.core.enumerations import (
    OnePropSizeMethod as _OnePropSizeMethod,
)
from svy.core.enumerations import (
    OnePropTestMethod as _OnePropTestMethod,
)
from svy.core.enumerations import (
    PropVarMode as _PropVarMode,
)
from svy.errors import SelectionError


_ONE_PROP_METHOD_MAP = {
    "wald": _OnePropSizeMethod.WALD,
    "fleiss": _OnePropSizeMethod.FLEISS,
}


def _normalize_one_prop_method(
    method: Literal["wald", "fleiss"] | None,
) -> _OnePropSizeMethod:
    """
    Normalize user-facing method string to internal OnePropSizeMethod enum.

    Accepts (case-insensitive):
      - "wald"   → OnePropSizeMethod.WALD  (default)
      - "fleiss" → OnePropSizeMethod.FLEISS
    """
    if method is None:
        return _OnePropSizeMethod.WALD
    if not isinstance(method, str):
        raise SelectionError.invalid_type(
            where="SampleSize.estimate_prop",
            param="method",
            got=method,
            expected="'wald', 'fleiss' or None",
        )
    result = _ONE_PROP_METHOD_MAP.get(method.strip().lower())
    if result is None:
        raise SelectionError.invalid_choice(
            where="SampleSize.estimate_prop",
            param="method",
            got=method,
            allowed=list(_ONE_PROP_METHOD_MAP),
        )
    return result


_PROP_VAR_MODE_MAP = {
    "alt-props": _PropVarMode.ALT_PROPS,
    "pooled-prop": _PropVarMode.POOLED_PROP,
    "max-var": _PropVarMode.MAX_VAR,
}


def _normalize_prop_var_mode(
    var_mode: Literal["alt-props", "pooled-prop", "max-var"] | None,
    *,
    where: str = "SampleSize.compare_props",
) -> _PropVarMode:
    """
    Normalize user-facing var_mode string to internal PropVarMode enum.

    Accepts (case-insensitive):
      - "alt-props"   → PropVarMode.ALT_PROPS   (default)
      - "pooled-prop" → PropVarMode.POOLED_PROP
      - "max-var"     → PropVarMode.MAX_VAR
    """
    if var_mode is None:
        return _PropVarMode.ALT_PROPS
    if not isinstance(var_mode, str):
        raise SelectionError.invalid_type(
            where=where,
            param="var_mode",
            got=var_mode,
            expected="'alt-props', 'pooled-prop', 'max-var' or None",
        )
    result = _PROP_VAR_MODE_MAP.get(var_mode.strip().lower())
    if result is None:
        raise SelectionError.invalid_choice(
            where=where,
            param="var_mode",
            got=var_mode,
            allowed=list(_PROP_VAR_MODE_MAP),
        )
    return result


_ONE_PROP_TEST_METHOD_MAP = {
    "wald": _OnePropTestMethod.WALD,
    "score": _OnePropTestMethod.SCORE,
    "arcsine": _OnePropTestMethod.ARCSINE,
}


def _normalize_one_prop_test_method(
    method: Literal["wald", "score", "arcsine"], *, where: str
) -> _OnePropTestMethod:
    """Normalize the test of one proportion ("wald", "score", "arcsine"), case-insensitive."""
    if not isinstance(method, str):
        raise SelectionError.invalid_type(
            where=where,
            param="method",
            got=method,
            expected="'wald', 'score' or 'arcsine'",
        )
    result = _ONE_PROP_TEST_METHOD_MAP.get(method.strip().lower())
    if result is None:
        raise SelectionError.invalid_choice(
            where=where,
            param="method",
            got=method,
            allowed=list(_ONE_PROP_TEST_METHOD_MAP),
        )
    return result
