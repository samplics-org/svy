# src/svy/utils/formats.py
from __future__ import annotations

import logging
import math

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any


log = logging.getLogger(__name__)

# Cache common quantizers to avoid repetitive Decimal instantiation
# Covers 0 to 12 decimal places, which handles 99% of survey reporting needs
_QUANTIZERS = {i: Decimal(1).scaleb(-i) for i in range(13)}


def _fmt_fixed(x: Any, *, dec: int = 5, thousands: bool = False) -> str:
    """
    Fixed-decimal formatter with optional thousands separators.
    Enforces 'Round Half Up' behavior (e.g., 2.5 -> 3, 3.5 -> 4).
    """
    if x is None:
        return ""

    # Fast path: ensure inputs are numeric before expensive logic
    if not isinstance(x, (int, float, Decimal)):
        return str(x)

    # Handle special float cases
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return str(x)
        # Decimal(str(x)) is safer than Decimal(x) to avoid float precision artifacts
        # e.g. Decimal(1.1) -> 1.1000000000000000888...
        d_val = Decimal(str(x))
    else:
        d_val = Decimal(x)

    try:
        # Retrieve or create the quantizer
        q = _QUANTIZERS.get(dec)
        if q is None:
            q = Decimal(1).scaleb(-dec)

        # 1. Round first using the strict arithmetic rule
        rounded = d_val.quantize(q, rounding=ROUND_HALF_UP)

        # 2. Format string (Python's format handles commas natively and efficiently)
        # Since 'rounded' is already quantized, this formatting just handles layout
        if thousands:
            return f"{rounded:,.{dec}f}"
        return f"{rounded:.{dec}f}"

    except (InvalidOperation, ValueError):
        # Fallback for edge cases where Decimal conversion might fail
        return f"{x:.{dec}f}"


def _fmt_num(x: Any, *, sig: int | None = 7, dec: int | None = None) -> str:
    """Format numbers with either significant digits (sig) or fixed decimals (dec)."""
    if x is None:
        return ""

    if isinstance(x, (int, float)):
        # Handle special floats
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return str(x)

        # Fixed decimal mode
        if dec is not None:
            return f"{x:.{dec}f}"

        # Significant digits mode (default)
        if sig is not None:
            return f"{x:.{sig}g}"

    return str(x)


# def _fmt_fixed(x: float | None, precision: int = 4) -> str:
#     """Format float with fixed precision, handling None."""
#     if x is None:
#         return "-"
#     return f"{x:.{precision}f}"


def _fmt_p(p: float | None, *, small: float = 0.001) -> str:
    """Format p-values (e.g. <0.001)."""
    if p is None:
        return "-"
    if p < small:
        return f"<{small:g}"
    return f"{p:.4f}"


def _fmt_smart(x: float | None) -> str:
    """
    Format numbers smartly:
    - Use scientific notation for |x| >= 1e6 or |x| < 1e-4 (non-zero).
    - Use fixed point otherwise.
    """
    if x is None:
        return "-"
    if x == 0:
        return "0.0000"

    abs_x = abs(x)
    if abs_x >= 1_000_000 or abs_x < 1e-4:
        return f"{x:.4e}"

    return f"{x:.4f}"
