# src/svy/size/power_goals.py
"""
Power-given-n goals: the inverse of the one-sample test sizing goals.

``n`` is the number of units selected, the quantity ``SampleSize`` returns as ``Size.n``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from svy.core.types import DomainScalarMap, Number
from svy.engine.size_and_power.one_sample import effective_n, power_z, terms_prop
from svy.errors import Code, SelectionError
from svy.size._normalize import _normalize_one_prop_test_method
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
from svy.size.types import PowerRow, TargetPower


if TYPE_CHECKING:
    from svy.size.power import Power


def _group_n(
    *, n: Any, deff: Any, resp_rate: Any, pop_size: Any, where: str
) -> tuple[float, float]:
    """(n, n_eff) of one group; a sample larger than its population is refused."""
    n = _positive(n, param="n", where=where, hint="n counts the units selected.")
    deff = _positive(
        deff, param="deff", where=where, hint="deff must be a positive finite number."
    )
    rr = _number(resp_rate, param="resp_rate", where=where)
    if not 0.0 < rr <= 1.0:
        raise SelectionError.invalid_range(
            where=where,
            param="resp_rate",
            got=rr,
            min_=0.0,
            max_=1.0,
            hint="resp_rate is a fraction in (0, 1], e.g. 0.85 for 85% response.",
        )
    pop = None if pop_size is None else _number(pop_size, param="pop_size", where=where)
    if pop is not None and n * rr > pop:
        raise SelectionError.ValueError(
            title="Sample larger than the population",
            detail=f"{n * rr:g} respondents (n={n:g} at response rate {rr:g}) exceed pop_size={pop:g}.",
            code=Code.POWER_N_EXCEEDS_POPULATION,
            where=where,
            param="n",
            got=n,
            hint="n counts the units selected; it cannot give more respondents than the population has.",
        )
    return n, effective_n(n=n, resp_rate=rr, deff=deff, pop_size=pop)


def _selected_n(row: dict[str, Any], *, where: str) -> tuple[float, float]:
    return _group_n(
        n=row["n"],
        deff=row["deff"],
        resp_rate=row["resp_rate"],
        pop_size=row["pop_size"],
        where=where,
    )


def _finish(
    pw: Power, *, goal: str, args: dict[str, Any], rows: list[PowerRow], solve: str = "power"
) -> Power:
    pw._target = TargetPower(goal=goal, solve=solve, args=args)
    pw._rows = rows
    return pw


def power_test_mean(
    pw: Power,
    mu0: Number | DomainScalarMap,
    mu1: Number | DomainScalarMap,
    sigma: Number | DomainScalarMap,
    n: Number | DomainScalarMap,
    *,
    pop_size: Number | DomainScalarMap | None = None,
    two_sides: bool = True,
    delta: Number | DomainScalarMap = 0.0,
    method: Literal["wald"] = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
) -> Power:
    """Power of the z test of H0: mu = mu0 when the true mean is mu1, given n selected units."""
    where = "Power.test_mean"
    _check_two_sides(two_sides, where=where)
    _check_mean_method(method, where=where)
    alpha = _checked_alpha(alpha, where=where)
    _require_positive(where=where, pop_size=pop_size)
    args = dict(
        mu0=_as_input(mu0),
        mu1=_as_input(mu1),
        sigma=_as_input(sigma),
        n=_as_input(n),
        alpha=_as_input(alpha),
        method=method,
        two_sides=two_sides,
        delta=_as_input(delta),
        pop_size=_as_input(pop_size),
        deff=_as_input(deff),
        resp_rate=_as_input(resp_rate),
    )
    _, keys, rows = _rows(
        dict(
            mu0=mu0,
            mu1=mu1,
            sigma=sigma,
            n=n,
            delta=delta,
            alpha=alpha,
            pop_size=pop_size,
            deff=deff,
            resp_rate=resp_rate,
        )
    )
    out = []
    for k, r in zip(keys, rows):
        effect = _number(r["mu1"], param="mu1", where=where) - _number(
            r["mu0"], param="mu0", where=where
        )
        sigma_k = _positive(
            r["sigma"], param="sigma", where=where, hint="sigma must be a positive finite number."
        )
        delta_k = _number(r["delta"], param="delta", where=where)
        n_k, n_eff = _selected_n(r, where=where)
        out.append(
            PowerRow(
                stratum=k,
                n=n_k,
                n_eff=n_eff,
                effect=effect,
                power=power_z(
                    effect=effect,
                    delta=delta_k,
                    s0=sigma_k,
                    s1=sigma_k,
                    two_sides=two_sides,
                    alpha=r["alpha"],
                    n_eff=n_eff,
                ),
            )
        )
    return _finish(pw, goal="test_mean", args=args, rows=out)


def power_test_prop(
    pw: Power,
    p0: Number | DomainScalarMap,
    p1: Number | DomainScalarMap,
    n: Number | DomainScalarMap,
    *,
    pop_size: Number | DomainScalarMap | None = None,
    two_sides: bool = True,
    delta: Number | DomainScalarMap = 0.0,
    method: Literal["wald", "score", "arcsine"] = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
) -> Power:
    """Power of the test of H0: p = p0 when the true proportion is p1, given n selected units."""
    where = "Power.test_prop"
    _check_two_sides(two_sides, where=where)
    _method = _normalize_one_prop_test_method(method, where=where)
    alpha = _checked_alpha(alpha, where=where)
    _require_positive(where=where, pop_size=pop_size)
    args = dict(
        p0=_as_input(p0),
        p1=_as_input(p1),
        n=_as_input(n),
        alpha=_as_input(alpha),
        method=method,
        two_sides=two_sides,
        delta=_as_input(delta),
        pop_size=_as_input(pop_size),
        deff=_as_input(deff),
        resp_rate=_as_input(resp_rate),
    )
    _, keys, rows = _rows(
        dict(
            p0=p0,
            p1=p1,
            n=n,
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
        p1_k = _open_unit(r["p1"], param="p1", where=where, hint="p1 is a proportion in (0, 1).")
        delta_k = _number(r["delta"], param="delta", where=where)
        _check_margin(_method, delta_k, where=where)
        n_k, n_eff = _selected_n(r, where=where)
        effect, stat_effect, s0, s1 = terms_prop(p0=p0_k, p1=p1_k, method=_method)
        out.append(
            PowerRow(
                stratum=k,
                n=n_k,
                n_eff=n_eff,
                effect=effect,
                power=power_z(
                    effect=stat_effect,
                    delta=delta_k,
                    s0=s0,
                    s1=s1,
                    two_sides=two_sides,
                    alpha=r["alpha"],
                    n_eff=n_eff,
                ),
            )
        )
    return _finish(pw, goal="test_prop", args=args, rows=out)
