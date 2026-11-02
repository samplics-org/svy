# src/svy/engine/size_and_power/mde.py
"""Minimum detectable effect: the smallest effect whose power reaches a target."""

from __future__ import annotations

from collections.abc import Callable

from scipy.optimize import brentq


def smallest_effect(
    power_at: Callable[[float], float], *, target: float, lo: float, hi: float, steps: int = 400
) -> float | None:
    """Smallest ``e`` in ``(lo, hi]`` with ``power_at(e) == target``; None if none reaches it.

    ``power_at(lo)`` is below ``target``. The grid finds the first crossing, so a power that is
    not monotone in the effect still returns the smallest root; brentq then refines it.
    """
    prev = lo
    for i in range(1, steps + 1):
        x = lo + (hi - lo) * i / steps
        if power_at(x) >= target:
            return float(brentq(lambda e: power_at(e) - target, prev, x, xtol=1e-14))
        prev = x
    return None


__all__ = ["smallest_effect"]
