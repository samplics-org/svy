# src/svy/size/power_compare_goals.py
"""Power given n for the two-group comparisons: the inverse of ``compare_means`` / ``compare_props``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from svy.core.enumerations import PropVarMode
from svy.core.types import DomainScalarMap, Number
from svy.engine.size_and_power.size import _validate_overlap
from svy.engine.size_and_power.two_sample import power_two_groups, prop_variances
from svy.errors import Code, SelectionError, UsageError
from svy.size._normalize import _normalize_prop_var_mode
from svy.size.estimation_goals import _as_input, _checked_alpha, _require_positive
from svy.size.one_sample_goals import _check_two_sides, _number, _open_unit, _positive, _rows
from svy.size.power_goals import _finish, _group_n
from svy.size.types import PowerRow


if TYPE_CHECKING:
    from svy.size.power import Power


def _pair(x: Any, *, param: str, where: str) -> tuple[Any, Any]:
    """One value for both groups, or a ``(group 1, group 2)`` tuple."""
    if isinstance(x, tuple):
        if len(x) != 2:
            raise SelectionError.invalid_type(
                where=where, param=param, got=x, expected="a number or a (group 1, group 2) pair"
            )
        return x
    return x, x


def _group_sizes(
    r: dict[str, Any], *, where: str
) -> tuple[tuple[float, float], tuple[float, float]]:
    """((n1, n2), (n1_eff, n2_eff)) of one stratum."""
    n, ratio = (
        r["n"],
        _positive(
            r["alloc_ratio"],
            param="alloc_ratio",
            where=where,
            hint="alloc_ratio is n2 / n1, the second group's size relative to the first.",
        ),
    )
    if isinstance(n, tuple):
        if len(n) != 2:
            raise SelectionError.invalid_type(
                where=where, param="n", got=n, expected="a number (n1) or an (n1, n2) pair"
            )
        if ratio != 1.0:
            raise UsageError.TypeError(
                title="Argument given twice",
                detail="n is an (n1, n2) pair and alloc_ratio is not 1: the two disagree on n2.",
                code=Code.PARAM_GIVEN_TWICE,
                where=where,
                param="alloc_ratio",
                got=ratio,
                hint="Give n as a pair, or n as n1 with alloc_ratio, not both.",
            )
        n1, n2 = n
    else:
        n1 = n
        n2 = _positive(n, param="n", where=where, hint="n counts the units selected.") * ratio
    deff = _pair(r["deff"], param="deff", where=where)
    rr = _pair(r["resp_rate"], param="resp_rate", where=where)
    pop = _pair(r["pop_size"], param="pop_size", where=where)
    g1 = _group_n(n=n1, deff=deff[0], resp_rate=rr[0], pop_size=pop[0], where=where)
    g2 = _group_n(n=n2, deff=deff[1], resp_rate=rr[1], pop_size=pop[1], where=where)
    return (g1[0], g2[0]), (g1[1], g2[1])


def _compare(
    pw: Power,
    *,
    goal: str,
    where: str,
    args: dict[str, Any],
    inputs: dict[str, Any],
    two_sides: bool,
    variances,
    effect_of,
) -> Power:
    _, keys, rows = _rows(inputs)
    out = []
    for k, r in zip(keys, rows):
        (n1, n2), (e1, e2) = _group_sizes(r, where=where)
        rho, overlap = _validate_overlap(
            _number(r["rho"], param="rho", where=where),
            _number(r["overlap"], param="overlap", where=where),
        )
        effect = effect_of(r)
        v1, v2 = variances(r, e1, e2)
        out.append(
            PowerRow(
                stratum=k,
                n=(n1, n2),
                n_eff=(e1, e2),
                effect=effect,
                power=power_two_groups(
                    effect=effect,
                    delta=_number(r["delta"], param="delta", where=where),
                    v1=v1,
                    v2=v2,
                    n1_eff=e1,
                    n2_eff=e2,
                    rho=rho,
                    overlap=overlap,
                    two_sides=two_sides,
                    alpha=r["alpha"],
                    where=where,
                ),
            )
        )
    return _finish(pw, goal=goal, args=args, rows=out)


def power_compare_means(
    pw: Power,
    mu1: Number | DomainScalarMap,
    mu2: Number | DomainScalarMap,
    sigma1: Number | DomainScalarMap,
    n: Number | tuple[Number, Number] | DomainScalarMap,
    *,
    sigma2: Number | DomainScalarMap | None = None,
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
    where = "Power.compare_means"
    _check_two_sides(two_sides, where=where)
    if method != "wald":
        raise SelectionError.invalid_choice(
            where=where, param="method", got=method, allowed=["wald"]
        )
    alpha = _checked_alpha(alpha, where=where)
    _require_positive(where=where, pop_size=pop_size)
    args = dict(
        mu1=_as_input(mu1),
        mu2=_as_input(mu2),
        sigma1=_as_input(sigma1),
        sigma2=_as_input(sigma2),
        n=_as_input(n),
        alloc_ratio=_as_input(alloc_ratio),
        alpha=_as_input(alpha),
        method=method,
        two_sides=two_sides,
        delta=_as_input(delta),
        pop_size=_as_input(pop_size),
        deff=_as_input(deff),
        resp_rate=_as_input(resp_rate),
        rho=_as_input(rho),
        overlap=_as_input(overlap),
    )
    inputs = dict(
        mu1=mu1,
        mu2=mu2,
        sigma1=sigma1,
        sigma2=sigma2,
        n=n,
        alloc_ratio=alloc_ratio,
        delta=delta,
        alpha=alpha,
        pop_size=pop_size,
        deff=deff,
        resp_rate=resp_rate,
        rho=rho,
        overlap=overlap,
    )

    def variances(r: dict[str, Any], e1: float, e2: float) -> tuple[float, float]:
        hint = "sigma must be a positive finite number."
        s1 = _positive(r["sigma1"], param="sigma1", where=where, hint=hint)
        if r["sigma2"] is None:
            return s1 * s1, s1 * s1
        s2 = _positive(r["sigma2"], param="sigma2", where=where, hint=hint)
        return s1 * s1, s2 * s2

    def effect_of(r: dict[str, Any]) -> float:
        return _number(r["mu2"], param="mu2", where=where) - _number(
            r["mu1"], param="mu1", where=where
        )

    return _compare(
        pw,
        goal="compare_means",
        where=where,
        args=args,
        inputs=inputs,
        two_sides=two_sides,
        variances=variances,
        effect_of=effect_of,
    )


def power_compare_props(
    pw: Power,
    p1: Number | DomainScalarMap,
    p2: Number | DomainScalarMap,
    n: Number | tuple[Number, Number] | DomainScalarMap,
    *,
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
    where = "Power.compare_props"
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
    alpha = _checked_alpha(alpha, where=where)
    _require_positive(where=where, pop_size=pop_size)
    args = dict(
        p1=_as_input(p1),
        p2=_as_input(p2),
        n=_as_input(n),
        alloc_ratio=_as_input(alloc_ratio),
        alpha=_as_input(alpha),
        method=method,
        var_mode=var_mode,
        two_sides=two_sides,
        delta=_as_input(delta),
        pop_size=_as_input(pop_size),
        deff=_as_input(deff),
        resp_rate=_as_input(resp_rate),
        rho=_as_input(rho),
        overlap=_as_input(overlap),
    )
    inputs = dict(
        p1=p1,
        p2=p2,
        n=n,
        alloc_ratio=alloc_ratio,
        delta=delta,
        alpha=alpha,
        pop_size=pop_size,
        deff=deff,
        resp_rate=resp_rate,
        rho=rho,
        overlap=overlap,
    )
    hint = "p1 and p2 are proportions in (0, 1)."

    def variances(r: dict[str, Any], e1: float, e2: float) -> tuple[float, float]:
        return prop_variances(
            p1=_open_unit(r["p1"], param="p1", where=where, hint=hint),
            p2=_open_unit(r["p2"], param="p2", where=where, hint=hint),
            n1_eff=e1,
            n2_eff=e2,
            var_mode=_var_mode,
        )

    def effect_of(r: dict[str, Any]) -> float:
        return _open_unit(r["p2"], param="p2", where=where, hint=hint) - _open_unit(
            r["p1"], param="p1", where=where, hint=hint
        )

    return _compare(
        pw,
        goal="compare_props",
        where=where,
        args=args,
        inputs=inputs,
        two_sides=two_sides,
        variances=variances,
        effect_of=effect_of,
    )
