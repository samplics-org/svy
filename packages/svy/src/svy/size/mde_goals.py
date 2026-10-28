# src/svy/size/mde_goals.py
"""Minimum detectable effect given n and power: the inverse of the Power goals."""

from __future__ import annotations

import math

from typing import TYPE_CHECKING, Any, Literal

from svy.core.enumerations import OnePropTestMethod, PropVarMode
from svy.core.types import DomainScalarMap, Number
from svy.engine.size_and_power.mde import smallest_effect
from svy.engine.size_and_power.one_sample import power_z, terms_prop
from svy.engine.size_and_power.size import _validate_overlap
from svy.engine.size_and_power.two_sample import power_two_groups, prop_variances
from svy.errors import Code, SelectionError
from svy.size._normalize import _normalize_one_prop_test_method, _normalize_prop_var_mode
from svy.size.estimation_goals import _as_input, _checked_alpha, _require_positive
from svy.size.one_sample_goals import (
    _check_margin,
    _check_mean_method,
    _check_two_sides,
    _number,
    _open_unit,
    _positive,
    _rows,
)
from svy.size.power_compare_goals import _group_sizes
from svy.size.power_goals import _finish, _selected_n
from svy.size.types import PowerRow


if TYPE_CHECKING:
    from svy.size.power import Power


_UPPER_BOUND = 1.0 - 1e-9  # proportions are searched up to here
_SE_SPAN = 50.0  # means are searched up to this many standard errors past the margin


def _check_direction(direction: Any, *, where: str) -> bool:
    """True for ``"lower"``."""
    if direction not in ("upper", "lower"):
        raise SelectionError.invalid_choice(
            where=where, param="direction", got=direction, allowed=["upper", "lower"]
        )
    return direction == "lower"


def _check_target(
    r: dict[str, Any], *, two_sides: bool, lower: bool, where: str
) -> tuple[float, float]:
    """(target power, margin): power must exceed alpha, and equivalence has no minimum effect."""
    delta = _number(r["delta"], param="delta", where=where)
    if two_sides and delta != 0.0:
        raise SelectionError.invalid_value(
            where=where,
            param="delta",
            got=delta,
            detail="A two-sided test with a margin is an equivalence test: the smaller the "
            "effect, the higher the power, so it has no minimum detectable effect.",
            hint="Use two_sides=False for a one-sided margin, or delta=0.",
        )
    if lower and delta != 0.0:
        raise SelectionError.invalid_value(
            where=where,
            param="direction",
            got="lower",
            detail="A one-sided test against a margin has H0: epsilon <= delta, "
            "so it looks upward only.",
            hint="Use direction='upper', or delta=0.",
        )
    target = _number(r["power"], param="power", where=where)
    if not r["alpha"] < target < 1.0:
        raise SelectionError.invalid_range(
            where=where,
            param="power",
            got=target,
            min_=r["alpha"],
            max_=1.0,
            hint="power must exceed alpha: with no effect, power equals alpha.",
        )
    return target, delta


def _unreachable(*, where: str, target: float, detail: str) -> SelectionError:
    return SelectionError.ValueError(
        title="Power cannot be reached",
        detail=f"No effect gives power {target:g}: {detail}",
        code=Code.POWER_UNREACHABLE,
        where=where,
        param="power",
        got=target,
        hint="Raise n, or lower power.",
    )


def _solve(power_at, *, target: float, lo: float, hi: float, where: str, detail: str) -> float:
    found = smallest_effect(power_at, target=target, lo=lo, hi=hi)
    if found is None:
        raise _unreachable(where=where, target=target, detail=detail)
    return found


def _row(k, *, n, n_eff, effect: float, target: float) -> PowerRow:
    return PowerRow(stratum=k, n=n, n_eff=n_eff, effect=effect, power=target)


def _common_args(**kw: Any) -> dict[str, Any]:
    return {k: _as_input(v) for k, v in kw.items()}


def mde_mean(
    pw: Power,
    mu0: Number | DomainScalarMap,
    sigma: Number | DomainScalarMap,
    n: Number | DomainScalarMap,
    *,
    power: Number | DomainScalarMap = 0.80,
    direction: Literal["upper", "lower"] = "upper",
    pop_size: Number | DomainScalarMap | None = None,
    two_sides: bool = True,
    delta: Number | DomainScalarMap = 0.0,
    method: Literal["wald"] = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
) -> Power:
    where = "Power.mde_mean"
    _check_two_sides(two_sides, where=where)
    _check_mean_method(method, where=where)
    lower = _check_direction(direction, where=where)
    alpha = _checked_alpha(alpha, where=where)
    _require_positive(where=where, pop_size=pop_size)
    args = _common_args(
        mu0=mu0,
        sigma=sigma,
        n=n,
        power=power,
        direction=direction,
        method=method,
        two_sides=two_sides,
        alpha=alpha,
        delta=delta,
        pop_size=pop_size,
        deff=deff,
        resp_rate=resp_rate,
    )
    _, keys, rows = _rows(
        dict(
            mu0=mu0,
            sigma=sigma,
            n=n,
            power=power,
            delta=delta,
            alpha=alpha,
            pop_size=pop_size,
            deff=deff,
            resp_rate=resp_rate,
        )
    )
    out = []
    for k, r in zip(keys, rows):
        sigma_k = _positive(
            r["sigma"], param="sigma", where=where, hint="sigma must be a positive finite number."
        )
        target, delta_k = _check_target(r, two_sides=two_sides, lower=lower, where=where)
        n_k, n_eff = _selected_n(r, where=where)
        lo = delta_k if not two_sides else 0.0
        if math.isinf(n_eff):
            effect = lo
        else:
            hi = lo + _SE_SPAN * sigma_k / math.sqrt(n_eff)
            effect = _solve(
                lambda e: power_z(
                    effect=e,
                    delta=delta_k,
                    s0=sigma_k,
                    s1=sigma_k,
                    two_sides=two_sides,
                    alpha=r["alpha"],
                    n_eff=n_eff,
                ),
                target=target,
                lo=lo,
                hi=hi,
                where=where,
                detail="the search ran out of range.",
            )
        out.append(_row(k, n=n_k, n_eff=n_eff, effect=-effect if lower else effect, target=target))
    return _finish(pw, goal="mde_mean", args=args, rows=out, solve="effect")


def mde_prop(
    pw: Power,
    p0: Number | DomainScalarMap,
    n: Number | DomainScalarMap,
    *,
    power: Number | DomainScalarMap = 0.80,
    direction: Literal["upper", "lower"] = "upper",
    pop_size: Number | DomainScalarMap | None = None,
    two_sides: bool = True,
    delta: Number | DomainScalarMap = 0.0,
    method: Literal["wald", "score", "arcsine"] = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
) -> Power:
    where = "Power.mde_prop"
    _check_two_sides(two_sides, where=where)
    _method: OnePropTestMethod = _normalize_one_prop_test_method(method, where=where)
    lower = _check_direction(direction, where=where)
    alpha = _checked_alpha(alpha, where=where)
    _require_positive(where=where, pop_size=pop_size)
    args = _common_args(
        p0=p0,
        n=n,
        power=power,
        direction=direction,
        method=method,
        two_sides=two_sides,
        alpha=alpha,
        delta=delta,
        pop_size=pop_size,
        deff=deff,
        resp_rate=resp_rate,
    )
    _, keys, rows = _rows(
        dict(
            p0=p0,
            n=n,
            power=power,
            delta=delta,
            alpha=alpha,
            pop_size=pop_size,
            deff=deff,
            resp_rate=resp_rate,
        )
    )
    out = []
    for k, r in zip(keys, rows):
        p0_k = _open_unit(r["p0"], param="p0", where=where, hint="p0 is a proportion in (0, 1).")
        target, delta_k = _check_target(r, two_sides=two_sides, lower=lower, where=where)
        _check_margin(_method, delta_k, where=where)
        n_k, n_eff = _selected_n(r, where=where)
        base = 1.0 - p0_k if lower else p0_k  # "lower" is "upper" on the complement
        lo = max(base + delta_k, 1e-9) if not two_sides else base

        def power_at(p1: float) -> float:
            _, stat, s0, s1 = terms_prop(p0=base, p1=p1, method=_method)
            return power_z(
                effect=stat,
                delta=delta_k,
                s0=s0,
                s1=s1,
                two_sides=two_sides,
                alpha=r["alpha"],
                n_eff=n_eff,
            )

        if math.isinf(n_eff):
            p1 = lo
        else:
            if lo >= _UPPER_BOUND:
                raise _unreachable(where=where, target=target, detail="the margin leaves no room.")
            p1 = _solve(
                power_at,
                target=target,
                lo=lo,
                hi=_UPPER_BOUND,
                where=where,
                detail="no proportion on that side of p0 is detectable with this n.",
            )
        effect = p1 - base
        out.append(_row(k, n=n_k, n_eff=n_eff, effect=-effect if lower else effect, target=target))
    return _finish(pw, goal="mde_prop", args=args, rows=out, solve="effect")


def _two_group_rows(inputs: dict[str, Any], *, two_sides, lower, where):
    """Per stratum: its inputs, target power, margin, sizes, and rho and overlap."""
    _, keys, rows = _rows(inputs)
    for k, r in zip(keys, rows):
        target, delta = _check_target(r, two_sides=two_sides, lower=lower, where=where)
        (n1, n2), (e1, e2) = _group_sizes(r, where=where)
        rho, overlap = _validate_overlap(
            _number(r["rho"], param="rho", where=where),
            _number(r["overlap"], param="overlap", where=where),
        )
        yield k, r, target, delta, (n1, n2), (e1, e2), rho, overlap


def mde_means(
    pw: Power,
    mu1: Number | DomainScalarMap,
    sigma1: Number | DomainScalarMap,
    n: Number | tuple[Number, Number] | DomainScalarMap,
    *,
    sigma2: Number | DomainScalarMap | None = None,
    power: Number | DomainScalarMap = 0.80,
    direction: Literal["upper", "lower"] = "upper",
    pop_size: Number | DomainScalarMap | None = None,
    two_sides: bool = True,
    delta: Number | DomainScalarMap = 0.0,
    alloc_ratio: Number | DomainScalarMap = 1.0,
    method: Literal["wald"] = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
    rho: Number | DomainScalarMap = 0.0,
    overlap: Number | DomainScalarMap = 0.0,
) -> Power:
    where = "Power.mde_means"
    _check_two_sides(two_sides, where=where)
    _check_mean_method(method, where=where)
    lower = _check_direction(direction, where=where)
    alpha = _checked_alpha(alpha, where=where)
    _require_positive(where=where, pop_size=pop_size)
    args = _common_args(
        mu1=mu1,
        sigma1=sigma1,
        sigma2=sigma2,
        n=n,
        power=power,
        direction=direction,
        alloc_ratio=alloc_ratio,
        method=method,
        two_sides=two_sides,
        alpha=alpha,
        delta=delta,
        pop_size=pop_size,
        deff=deff,
        resp_rate=resp_rate,
        rho=rho,
        overlap=overlap,
    )
    inputs = dict(
        mu1=mu1,
        sigma1=sigma1,
        sigma2=sigma2,
        n=n,
        power=power,
        alloc_ratio=alloc_ratio,
        delta=delta,
        alpha=alpha,
        pop_size=pop_size,
        deff=deff,
        resp_rate=resp_rate,
        rho=rho,
        overlap=overlap,
    )
    out = []
    for k, r, target, delta_k, (n1, n2), (e1, e2), rho_k, overlap_k in _two_group_rows(
        inputs, two_sides=two_sides, lower=lower, where=where
    ):
        hint = "sigma must be a positive finite number."
        v1 = _positive(r["sigma1"], param="sigma1", where=where, hint=hint) ** 2
        v2 = (
            v1
            if r["sigma2"] is None
            else _positive(r["sigma2"], param="sigma2", where=where, hint=hint) ** 2
        )

        def power_at(e: float) -> float:
            return power_two_groups(
                effect=e,
                delta=delta_k,
                v1=v1,
                v2=v2,
                n1_eff=e1,
                n2_eff=e2,
                rho=rho_k,
                overlap=overlap_k,
                two_sides=two_sides,
                alpha=r["alpha"],
                where=where,
            )

        lo = delta_k if not two_sides else 0.0
        if math.isinf(e1) and math.isinf(e2):
            effect = lo
        else:
            se = math.sqrt(v1 / e1 + v2 / e2)
            effect = _solve(
                power_at,
                target=target,
                lo=lo,
                hi=lo + _SE_SPAN * se,
                where=where,
                detail="the search ran out of range.",
            )
        out.append(
            _row(k, n=(n1, n2), n_eff=(e1, e2), effect=-effect if lower else effect, target=target)
        )
    return _finish(pw, goal="mde_means", args=args, rows=out, solve="effect")


def mde_props(
    pw: Power,
    p1: Number | DomainScalarMap,
    n: Number | tuple[Number, Number] | DomainScalarMap,
    *,
    power: Number | DomainScalarMap = 0.80,
    direction: Literal["upper", "lower"] = "upper",
    pop_size: Number | DomainScalarMap | None = None,
    two_sides: bool = True,
    delta: Number | DomainScalarMap = 0.0,
    alloc_ratio: Number | DomainScalarMap = 1.0,
    method: Literal["wald"] = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    var_mode: Literal["alt-props", "pooled-prop", "max-var"] = "alt-props",
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
    rho: Number | DomainScalarMap = 0.0,
    overlap: Number | DomainScalarMap = 0.0,
) -> Power:
    where = "Power.mde_props"
    _check_two_sides(two_sides, where=where)
    if method != "wald":
        raise SelectionError.invalid_choice(
            where=where,
            param="method",
            got=method,
            allowed=["wald"],
            hint="Use method='wald'; var_mode='alt-props', 'pooled-prop' or 'max-var' "
            "sets its variance.",
        )
    _var_mode: PropVarMode = _normalize_prop_var_mode(var_mode, where=where)
    lower = _check_direction(direction, where=where)
    alpha = _checked_alpha(alpha, where=where)
    _require_positive(where=where, pop_size=pop_size)
    args = _common_args(
        p1=p1,
        n=n,
        power=power,
        direction=direction,
        alloc_ratio=alloc_ratio,
        method=method,
        var_mode=var_mode,
        two_sides=two_sides,
        alpha=alpha,
        delta=delta,
        pop_size=pop_size,
        deff=deff,
        resp_rate=resp_rate,
        rho=rho,
        overlap=overlap,
    )
    inputs = dict(
        p1=p1,
        n=n,
        power=power,
        alloc_ratio=alloc_ratio,
        delta=delta,
        alpha=alpha,
        pop_size=pop_size,
        deff=deff,
        resp_rate=resp_rate,
        rho=rho,
        overlap=overlap,
    )
    out = []
    for k, r, target, delta_k, (n1, n2), (e1, e2), rho_k, overlap_k in _two_group_rows(
        inputs, two_sides=two_sides, lower=lower, where=where
    ):
        base = _open_unit(r["p1"], param="p1", where=where, hint="p1 is a proportion in (0, 1).")
        if lower:
            base = 1.0 - base  # "lower" is "upper" on the complement

        def power_at(p2: float) -> float:
            v1, v2 = prop_variances(p1=base, p2=p2, n1_eff=e1, n2_eff=e2, var_mode=_var_mode)
            return power_two_groups(
                effect=p2 - base,
                delta=delta_k,
                v1=v1,
                v2=v2,
                n1_eff=e1,
                n2_eff=e2,
                rho=rho_k,
                overlap=overlap_k,
                two_sides=two_sides,
                alpha=r["alpha"],
                where=where,
            )

        lo = max(base + delta_k, 1e-9) if not two_sides else base
        if math.isinf(e1) and math.isinf(e2):
            p2 = lo
        else:
            if lo >= _UPPER_BOUND:
                raise _unreachable(where=where, target=target, detail="the margin leaves no room.")
            p2 = _solve(
                power_at,
                target=target,
                lo=lo,
                hi=_UPPER_BOUND,
                where=where,
                detail="no proportion on that side of p1 is detectable with these sizes.",
            )
        effect = p2 - base
        out.append(
            _row(k, n=(n1, n2), n_eff=(e1, e2), effect=-effect if lower else effect, target=target)
        )
    return _finish(pw, goal="mde_props", args=args, rows=out, solve="effect")
