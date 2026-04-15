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
    PropVarMode as _PropVarMode,
)
from svy.core.enumerations import (
    TwoPropsSizeMethod as _TwoPropsSizeMethod,
)


def _normalize_one_prop_method(
    method: Literal["wald", "fleiss"] | None,
) -> _OnePropSizeMethod:
    """
    Normalize user-facing method string to internal OnePropSizeMethod enum.

    Accepts (case-insensitive):
      - "wald"   → OnePropSizeMethod.WALD  (default)
      - "fleiss" → OnePropSizeMethod.FLEISS
    """
    _MAP = {
        "wald": _OnePropSizeMethod.WALD,
        "fleiss": _OnePropSizeMethod.FLEISS,
    }
    if method is None:
        return _OnePropSizeMethod.WALD
    if not isinstance(method, str):
        raise TypeError(
            f"'method' must be a string or None, got {type(method).__name__}. "
            f"Use 'wald' or 'fleiss'."
        )
    result = _MAP.get(method.strip().lower())
    if result is None:
        raise ValueError(f"Unknown method {method!r}. Use 'wald' or 'fleiss'.")
    return result


def _normalize_two_props_method(
    method: Literal["wald", "miettinen-nurminen", "newcombe", "farrington-manning"] | None,
) -> _TwoPropsSizeMethod:
    """
    Normalize user-facing method string to internal TwoPropsSizeMethod enum.

    Accepts (case-insensitive):
      - "wald"                  → TwoPropsSizeMethod.WALD  (default)
      - "miettinen-nurminen"    → TwoPropsSizeMethod.MIETTINEN_NURMINEN
      - "newcombe"              → TwoPropsSizeMethod.NEWCOMBE
      - "farrington-manning"    → TwoPropsSizeMethod.FARRINGTON_MANNING
    """
    _MAP = {
        "wald": _TwoPropsSizeMethod.WALD,
        "miettinen-nurminen": _TwoPropsSizeMethod.MIETTINEN_NURMINEN,
        "newcombe": _TwoPropsSizeMethod.NEWCOMBE,
        "farrington-manning": _TwoPropsSizeMethod.FARRINGTON_MANNING,
    }
    if method is None:
        return _TwoPropsSizeMethod.WALD
    if not isinstance(method, str):
        raise TypeError(
            f"'method' must be a string or None, got {type(method).__name__}. "
            f"Use 'wald', 'miettinen-nurminen', 'newcombe', or 'farrington-manning'."
        )
    result = _MAP.get(method.strip().lower())
    if result is None:
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Use 'wald', 'miettinen-nurminen', 'newcombe', or 'farrington-manning'."
        )
    return result


def _normalize_prop_var_mode(
    var_mode: Literal["alt-props", "pooled-prop", "max-var"] | None,
) -> _PropVarMode:
    """
    Normalize user-facing var_mode string to internal PropVarMode enum.

    Accepts (case-insensitive):
      - "alt-props"   → PropVarMode.ALT_PROPS   (default)
      - "pooled-prop" → PropVarMode.POOLED_PROP
      - "max-var"     → PropVarMode.MAX_VAR
    """
    _MAP = {
        "alt-props": _PropVarMode.ALT_PROPS,
        "pooled-prop": _PropVarMode.POOLED_PROP,
        "max-var": _PropVarMode.MAX_VAR,
    }
    if var_mode is None:
        return _PropVarMode.ALT_PROPS
    if not isinstance(var_mode, str):
        raise TypeError(
            f"'var_mode' must be a string or None, got {type(var_mode).__name__}. "
            f"Use 'alt-props', 'pooled-prop', or 'max-var'."
        )
    result = _MAP.get(var_mode.strip().lower())
    if result is None:
        raise ValueError(
            f"Unknown var_mode {var_mode!r}. Use 'alt-props', 'pooled-prop', or 'max-var'."
        )
    return result
