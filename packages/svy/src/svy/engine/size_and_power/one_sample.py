# src/svy/engine/size_and_power/one_sample.py
"""z tests of one mean or one proportion: sample size, and power given n."""

from __future__ import annotations

import math

from scipy.stats import norm

from svy.core.enumerations import OnePropTestMethod
from svy.engine.size_and_power.size import _test_denominator, _zcrit_pair


def terms_prop(
    *, p0: float, p1: float, method: OnePropTestMethod
) -> tuple[float, float, float, float]:
    """(p1 - p0, effect on the statistic's scale, sd under H0, sd under H1), per sqrt(n)."""
    s1 = math.sqrt(p1 * (1.0 - p1))
    if method is OnePropTestMethod.WALD:
        return p1 - p0, p1 - p0, s1, s1
    if method is OnePropTestMethod.SCORE:
        return p1 - p0, p1 - p0, math.sqrt(p0 * (1.0 - p0)), s1
    h = 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p0))
    return p1 - p0, h, 1.0, 1.0


def size_z(
    *,
    effect: float,
    delta: float,
    s0: float,
    s1: float,
    two_sides: bool,
    alpha: float,
    power: float,
    where: str,
) -> float:
    """SRS sample size, rounded up: ((z_a s0 + z_b s1) / (effect - delta))^2."""
    z_a, z_b = _zcrit_pair(two_sides=two_sides, alpha=alpha, power=power, delta=delta)
    denom = _test_denominator(epsilon=effect, delta=delta, two_sides=two_sides, where=where)
    return float(math.ceil(((z_a * s0 + z_b * s1) / denom) ** 2))


def _shift(x: float, sqrt_n: float) -> float:
    """``x * sqrt_n``, with 0 * inf = 0 so a census keeps a zero effect at zero."""
    return 0.0 if x == 0.0 else x * sqrt_n


def power_z(
    *,
    effect: float,
    delta: float,
    s0: float,
    s1: float,
    two_sides: bool,
    alpha: float,
    n_eff: float,
) -> float:
    """Power of the test at ``n_eff`` SRS-equivalent units; both tails when two-sided."""
    sqrt_n = math.sqrt(n_eff) if math.isfinite(n_eff) else math.inf
    if two_sides and delta != 0.0:
        z = float(norm.isf(alpha))
        low = norm.cdf(_shift(delta - effect, sqrt_n) / s1 - z)
        high = norm.cdf(_shift(delta + effect, sqrt_n) / s1 - z)
        return max(0.0, float(low + high - 1.0))
    if two_sides:
        z = float(norm.isf(alpha / 2.0))
        shift = _shift(effect, sqrt_n)
        return float(norm.cdf((shift - z * s0) / s1) + norm.cdf((-shift - z * s0) / s1))
    z = float(norm.isf(alpha))
    shift = _shift(abs(effect) if delta == 0.0 else effect - delta, sqrt_n)
    return float(norm.cdf((shift - z * s0) / s1))


def effective_n(*, n: float, resp_rate: float, deff: float, pop_size: float | None) -> float:
    """SRS-equivalent n of ``n`` selected units: the exact inverse of nonresponse, FPC and deff."""
    n_resp = n * resp_rate
    if pop_size is None:
        return n_resp / deff
    if n_resp >= pop_size:
        return math.inf
    return n_resp * (pop_size - 1.0) / (pop_size - n_resp) / deff


__all__ = ["effective_n", "power_z", "size_z", "terms_prop"]
