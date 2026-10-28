# src/svy/engine/size_and_power/two_sample.py
"""z tests of a difference between two means or two proportions: power given n."""

from __future__ import annotations

import math

from svy.core.enumerations import PropVarMode
from svy.engine.size_and_power.one_sample import power_z
from svy.engine.size_and_power.size import _no_variance_error


def prop_variances(
    *, p1: float, p2: float, n1_eff: float, n2_eff: float, var_mode: PropVarMode
) -> tuple[float, float]:
    """Per-unit variances (v1, v2) of the two groups under ``var_mode``."""
    if var_mode is PropVarMode.ALT_PROPS:
        return p1 * (1.0 - p1), p2 * (1.0 - p2)
    if var_mode is PropVarMode.MAX_VAR:
        return 0.25, 0.25
    if math.isinf(n1_eff) and math.isinf(n2_eff):
        w1 = 0.5
    elif math.isinf(n1_eff):
        w1 = 1.0
    elif math.isinf(n2_eff):
        w1 = 0.0
    else:
        w1 = n1_eff / (n1_eff + n2_eff)
    pooled = w1 * p1 + (1.0 - w1) * p2
    return pooled * (1.0 - pooled), pooled * (1.0 - pooled)


def power_two_groups(
    *,
    effect: float,
    delta: float,
    v1: float,
    v2: float,
    n1_eff: float,
    n2_eff: float,
    rho: float,
    overlap: float,
    two_sides: bool,
    alpha: float,
    where: str,
) -> float:
    """Power for ``effect = group 2 - group 1`` at the groups' SRS-equivalent sizes.

    Var(d) = v1/n1 + v2/n2 - 2 rho m sqrt(v1 v2) / (n1 n2), m = overlap * min(n1, n2).
    """
    if math.isinf(n1_eff) and math.isinf(n2_eff):
        var = 0.0
    else:
        cross = 2.0 * rho * overlap * min(n1_eff, n2_eff) * math.sqrt(v1 * v2) / (n1_eff * n2_eff)
        var = v1 / n1_eff + v2 / n2_eff - cross
        if var <= 0.0:
            raise _no_variance_error(rho=rho, overlap=overlap, var=var, where=where)
    n_eff = math.inf if var == 0.0 else 1.0 / var
    return power_z(
        effect=effect, delta=delta, s0=1.0, s1=1.0, two_sides=two_sides, alpha=alpha, n_eff=n_eff
    )


__all__ = ["power_two_groups", "prop_variances"]
