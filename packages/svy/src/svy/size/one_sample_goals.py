# src/svy/size/one_sample_goals.py
"""
One-sample test sizing: H0: mu = mu0 and H0: p = p0.

The checks and the per-stratum reading of the inputs are shared with
svy/size/power_goals.py, which answers the same questions given n.
"""

from __future__ import annotations

import math
import numbers

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from svy.core.enumerations import OnePropTestMethod, PopParam
from svy.core.types import DomainScalarMap, Number
from svy.engine.size_and_power.one_sample import size_z, terms_prop
from svy.engine.size_and_power.size import _apply_deff, _apply_fpc_srswor, _apply_nonresponse
from svy.errors import Code, SelectionError
from svy.size._normalize import _normalize_one_prop_test_method
from svy.size.estimation_goals import _as_input, _build_sizes, _checked_alpha, _require_positive
from svy.size.types import TargetTestMean, TargetTestProp
from svy.utils.helpers import _get_keys_from_maps


if TYPE_CHECKING:
    from svy.size.base import SampleSize


# ---------------------------------------------------------------------------
# Shared with power_goals
# ---------------------------------------------------------------------------


def _number(x: Any, *, param: str, where: str) -> float:
    """``x`` as a finite float; bools and strings are refused."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise SelectionError.invalid_type(
            where=where,
            param=param,
            got=x,
            expected="a number, or a mapping of numbers by stratum",
        )
    if not math.isfinite(x):
        raise SelectionError.invalid_range(where=where, param=param, got=x)
    return float(x)


def _open_unit(x: Any, *, param: str, where: str, hint: str) -> float:
    """``x`` strictly inside (0, 1)."""
    v = _number(x, param=param, where=where)
    if not 0.0 < v < 1.0:
        raise SelectionError.invalid_range(
            where=where, param=param, got=x, min_=0.0, max_=1.0, hint=hint
        )
    return v


def _positive(x: Any, *, param: str, where: str, hint: str) -> float:
    v = _number(x, param=param, where=where)
    if not v > 0.0:
        raise SelectionError.invalid_range(where=where, param=param, got=x, min_=0, hint=hint)
    return v


def _check_two_sides(two_sides: Any, *, where: str) -> None:
    if not isinstance(two_sides, bool):
        raise SelectionError.invalid_type(
            where=where, param="two_sides", got=two_sides, expected="True or False"
        )


def _check_mean_method(method: Any, *, where: str) -> None:
    if method != "wald":
        raise SelectionError.invalid_choice(
            where=where,
            param="method",
            got=method,
            allowed=["wald"],
            hint="The test of a mean is the z test with known sigma.",
        )


def _check_margin(method: OnePropTestMethod, delta: float, *, where: str) -> None:
    if method is not OnePropTestMethod.WALD and delta != 0.0:
        raise SelectionError.ValueError(
            title="Method does not take a margin",
            detail=f"method='{method.name.lower()}' tests against the null value only, not a margin.",
            code=Code.SIZE_METHOD_MARGIN,
            where=where,
            param="delta",
            got=delta,
            hint="Use method='wald' for a non-inferiority or equivalence margin, or delta=0.",
        )


def _rows(params: dict[str, Any]) -> tuple[bool, list[Any], list[dict[str, Any]]]:
    """(stratified, stratum keys, one dict of scalar inputs per stratum)."""
    maps = {k: v for k, v in params.items() if isinstance(v, Mapping)}
    if not maps:
        return False, [None], [dict(params)]
    keys = _get_keys_from_maps(**maps)
    rows = [{k: (v[s] if isinstance(v, Mapping) else v) for k, v in params.items()} for s in keys]
    return True, keys, rows


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


def _size_goal(ss: SampleSize, *, stratified, keys, n0_by_key, row_by_key) -> SampleSize:
    """The deff -> FPC -> nonresponse pipeline of every goal, per stratum."""
    n1: dict[Any, Number] = {}
    n2: dict[Any, Number] = {}
    nf: dict[Any, Number] = {}
    for k in keys:
        r = row_by_key[k]
        n1[k] = _apply_deff(n=n0_by_key[k], deff=r["deff"])
        pop = r["pop_size"]
        n2[k] = _apply_fpc_srswor(n0=n1[k], pop_size=pop) if pop is not None else n1[k]
        nf[k] = _apply_nonresponse(n=n2[k], resp_rate=r["resp_rate"])
    _build_sizes(
        ss,
        stratified=stratified,
        strata=keys if stratified else None,
        n0=n0_by_key if stratified else n0_by_key[None],
        n1_deff=n1 if stratified else n1[None],
        n2_fpc=n2 if stratified else n2[None],
        n_final=nf if stratified else nf[None],
    )
    return ss


def test_mean(
    ss: SampleSize,
    mu0: Number | DomainScalarMap,
    mu1: Number | DomainScalarMap,
    sigma: Number | DomainScalarMap,
    *,
    pop_size: Number | DomainScalarMap | None = None,
    two_sides: bool = True,
    delta: Number | DomainScalarMap = 0.0,
    method: Literal["wald"] = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    power: Number | DomainScalarMap = 0.80,
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
) -> SampleSize:
    """
    Compute required sample size for testing H0: mu = mu0 when the true mean is mu1.

    The test is a z test with known ``sigma``. ``epsilon = mu1 - mu0`` is signed.

    Parameters
    ----------
    ss : SampleSize
        The SampleSize instance to update.
    mu0 : Number | DomainScalarMap
        Mean under the null. Scalar or per-stratum mapping.
    mu1 : Number | DomainScalarMap
        True mean under the alternative.
    sigma : Number | DomainScalarMap
        Population standard deviation.
    pop_size : Number | DomainScalarMap | None, default None
        Target population size. If None, no finite population correction is applied.
    two_sides : bool, default True
        Two-sided test. One-sided with ``delta == 0`` tests on the side where
        ``mu1 - mu0`` lies.
    delta : Number | DomainScalarMap, default 0.0
        Margin. One-sided: H0: epsilon <= delta (non-inferiority: delta < 0).
        Two-sided and ``delta != 0``: equivalence, H0: |epsilon| >= delta.
    method : str, default 'wald'
        Only ``'wald'`` is available.
    alpha, power : Number | DomainScalarMap
        Significance level (0.05) and desired power (0.80).
    deff, resp_rate : Number | DomainScalarMap
        Design effect (1.0) and anticipated response rate (1.0).

    Returns
    -------
    SampleSize
        The updated SampleSize instance (chainable).

    Raises
    ------
    SelectionError
        ``INFEASIBLE_DESIGN`` when no sample size can power the test:
        ``epsilon == delta``, a one-sided ``epsilon < delta``, or an
        equivalence with ``|epsilon| >= delta``.
    """
    where = "SampleSize.test_mean"
    _check_two_sides(two_sides, where=where)
    _check_mean_method(method, where=where)
    alpha = _checked_alpha(alpha, where=where)
    _require_positive(where=where, pop_size=pop_size)

    ss._param = PopParam.MEAN
    ss._target = TargetTestMean(
        mu0=_as_input(mu0),
        mu1=_as_input(mu1),
        sigma=_as_input(sigma),
        alpha=_as_input(alpha),
        power=_as_input(power),
        method=method,
        two_sides=two_sides,
        delta=_as_input(delta),
        pop_size=_as_input(pop_size),
        deff=_as_input(deff),
        resp_rate=_as_input(resp_rate),
    )

    stratified, keys, rows = _rows(
        dict(
            mu0=mu0,
            mu1=mu1,
            sigma=sigma,
            delta=delta,
            alpha=alpha,
            power=power,
            pop_size=pop_size,
            deff=deff,
            resp_rate=resp_rate,
        )
    )
    n0: dict[Any, Number] = {}
    for k, r in zip(keys, rows):
        mu0_k = _number(r["mu0"], param="mu0", where=where)
        mu1_k = _number(r["mu1"], param="mu1", where=where)
        sigma_k = _positive(
            r["sigma"], param="sigma", where=where, hint="sigma must be a positive finite number."
        )
        power_k = _open_unit(
            r["power"], param="power", where=where, hint="power is a probability, e.g. 0.80."
        )
        n0[k] = size_z(
            effect=mu1_k - mu0_k,
            delta=_number(r["delta"], param="delta", where=where),
            s0=sigma_k,
            s1=sigma_k,
            two_sides=two_sides,
            alpha=r["alpha"],
            power=power_k,
            where=where,
        )
    return _size_goal(
        ss, stratified=stratified, keys=keys, n0_by_key=n0, row_by_key=dict(zip(keys, rows))
    )


def test_prop(
    ss: SampleSize,
    p0: Number | DomainScalarMap,
    p1: Number | DomainScalarMap,
    *,
    pop_size: Number | DomainScalarMap | None = None,
    two_sides: bool = True,
    delta: Number | DomainScalarMap = 0.0,
    method: Literal["wald", "score", "arcsine"] = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    power: Number | DomainScalarMap = 0.80,
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
) -> SampleSize:
    """
    Compute required sample size for testing H0: p = p0 when the true proportion is p1.

    ``epsilon = p1 - p0`` is signed.

    Parameters
    ----------
    ss : SampleSize
        The SampleSize instance to update.
    p0 : Number | DomainScalarMap
        Proportion under the null. Scalar or per-stratum mapping.
    p1 : Number | DomainScalarMap
        True proportion under the alternative.
    pop_size : Number | DomainScalarMap | None, default None
        Target population size. If None, no finite population correction is applied.
    two_sides : bool, default True
        Two-sided test. One-sided with ``delta == 0`` tests on the side where
        ``p1 - p0`` lies.
    delta : Number | DomainScalarMap, default 0.0
        Margin, as in ``test_mean``. Only ``method='wald'`` takes a nonzero one.
    method : {'wald', 'score', 'arcsine'}, default 'wald'
        ``'wald'``: the statistic's variance is estimated under the alternative, as a
        design-based analysis tests a proportion. ``'score'``: variance under the null.
        ``'arcsine'``: the variance-stabilizing statistic (Cohen's h).
    alpha, power : Number | DomainScalarMap
        Significance level (0.05) and desired power (0.80).
    deff, resp_rate : Number | DomainScalarMap
        Design effect (1.0) and anticipated response rate (1.0).

    Returns
    -------
    SampleSize
        The updated SampleSize instance (chainable).

    Raises
    ------
    SelectionError
        ``INFEASIBLE_DESIGN`` as in ``test_mean``; ``SIZE_METHOD_MARGIN`` for a
        margin with ``'score'`` or ``'arcsine'``.
    """
    where = "SampleSize.test_prop"
    _check_two_sides(two_sides, where=where)
    _method = _normalize_one_prop_test_method(method, where=where)
    alpha = _checked_alpha(alpha, where=where)
    _require_positive(where=where, pop_size=pop_size)

    ss._param = PopParam.PROP
    ss._target = TargetTestProp(
        p0=_as_input(p0),
        p1=_as_input(p1),
        alpha=_as_input(alpha),
        power=_as_input(power),
        method=method,
        two_sides=two_sides,
        delta=_as_input(delta),
        pop_size=_as_input(pop_size),
        deff=_as_input(deff),
        resp_rate=_as_input(resp_rate),
    )

    stratified, keys, rows = _rows(
        dict(
            p0=p0,
            p1=p1,
            delta=delta,
            alpha=alpha,
            power=power,
            pop_size=pop_size,
            deff=deff,
            resp_rate=resp_rate,
        )
    )
    n0: dict[Any, Number] = {}
    for k, r in zip(keys, rows):
        p0_k = _open_unit(r["p0"], param="p0", where=where, hint="p0 is a proportion in (0, 1).")
        p1_k = _open_unit(r["p1"], param="p1", where=where, hint="p1 is a proportion in (0, 1).")
        delta_k = _number(r["delta"], param="delta", where=where)
        _check_margin(_method, delta_k, where=where)
        power_k = _open_unit(
            r["power"], param="power", where=where, hint="power is a probability, e.g. 0.80."
        )
        _, stat_effect, s0, s1 = terms_prop(p0=p0_k, p1=p1_k, method=_method)
        n0[k] = size_z(
            effect=stat_effect,
            delta=delta_k,
            s0=s0,
            s1=s1,
            two_sides=two_sides,
            alpha=r["alpha"],
            power=power_k,
            where=where,
        )
    return _size_goal(
        ss, stratified=stratified, keys=keys, n0_by_key=n0, row_by_key=dict(zip(keys, rows))
    )
