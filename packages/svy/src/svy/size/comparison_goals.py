# src/svy/size/comparison_goals.py
"""
Comparison-goal sample size functions.

Each function takes a SampleSize instance and returns it (chainable).
The SampleSize class in base.py delegates to these functions directly.

Adding a new comparison goal:
  1. Implement it here as a module-level function.
  2. Import it in base.py and add a one-liner delegation method.
  3. If it has a pure algorithm, put that in svy/engine/size_and_power/size.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

from svy.core.enumerations import MeanVarMode as _MeanVarMode
from svy.core.enumerations import PopParam
from svy.core.enumerations import (
    TwoPropsSizeMethod as _TwoPropsSizeMethod,
)
from svy.core.types import DomainScalarMap, Number
from svy.engine.size_and_power.size import (
    _apply_deff_pair,
    _apply_fpc_srswor_pair,
    _apply_nonresponse_pair,
    _wald_sample_size_two_means,
    _wald_sample_size_two_props,
)
from svy.size._normalize import (
    _normalize_prop_var_mode,
    _normalize_two_props_method,
)
from svy.size.estimation_goals import _broadcast_scalars, _build_sizes, _has_pop
from svy.utils.helpers import _get_keys_from_maps


if TYPE_CHECKING:
    from svy.size.base import SampleSize


def compare_props(
    ss: SampleSize,
    p1: Number | DomainScalarMap,
    p2: Number | DomainScalarMap,
    *,
    pop_size: Number | DomainScalarMap | None = None,
    two_sides: bool = True,
    delta: Number | DomainScalarMap = 0.0,
    alloc_ratio: Number | DomainScalarMap = 1.0,
    method: Literal["wald", "miettinen-nurminen", "newcombe", "farrington-manning"] = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    power: Number | DomainScalarMap = 0.80,
    var_mode: Literal["alt-props", "pooled-prop", "max-var"] = "alt-props",
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
) -> SampleSize:
    """
    Compute required sample size for comparing two proportions.

    Parameters
    ----------
    ss : SampleSize
        The SampleSize instance to update.
    p1 : Number | DomainScalarMap
        Proportion in group 1. Scalar or per-stratum mapping.
    p2 : Number | DomainScalarMap
        Proportion in group 2. Scalar or per-stratum mapping.
    pop_size : Number | DomainScalarMap | None, default None
        Target population size. If None, no finite population correction is applied.
        Scalar or per-stratum mapping.
    two_sides : bool, default True
        Whether to use a two-sided test.
    delta : Number | DomainScalarMap, default 0.0
        Non-inferiority / equivalence margin.
    alloc_ratio : Number | DomainScalarMap, default 1.0
        Allocation ratio n2/n1.
    method : str, default 'wald'
        Calculation method: ``'wald'``, ``'miettinen-nurminen'``, ``'newcombe'``, or
        ``'farrington-manning'``. Only ``'wald'`` is currently implemented.
    alpha : Number | DomainScalarMap, default 0.05
        Significance level.
    power : Number | DomainScalarMap, default 0.80
        Desired statistical power.
    var_mode : str, default 'alt-props'
        Variance mode for the Wald statistic: ``'alt-props'``, ``'pooled-prop'``, or ``'max-var'``.
    deff : Number | DomainScalarMap, default 1.0
        Design effect. Scalar or per-stratum mapping.
    resp_rate : Number | DomainScalarMap, default 1.0
        Anticipated response rate. Scalar or per-stratum mapping.

    Returns
    -------
    SampleSize
        The updated SampleSize instance (chainable).
    """
    ss._param = PopParam.PROP

    stratified = any(
        isinstance(v, Mapping)
        for v in [p1, p2, pop_size, delta, alloc_ratio, alpha, power, deff, resp_rate]
    )

    if stratified:
        strata = _get_keys_from_maps(
            **{
                k: v
                for k, v in dict(
                    p1=p1,
                    p2=p2,
                    pop_size=pop_size,
                    delta=delta,
                    alloc_ratio=alloc_ratio,
                    alpha=alpha,
                    power=power,
                    deff=deff,
                    resp_rate=resp_rate,
                ).items()
                if isinstance(v, Mapping)
            }
        )
        m = len(strata)
        params = {
            "p1": p1,
            "p2": p2,
            "pop_size": pop_size,
            "delta": delta,
            "alloc_ratio": alloc_ratio,
            "alpha": alpha,
            "power": power,
            "deff": deff,
            "resp_rate": resp_rate,
        }
        _broadcast_scalars(strata, params)
        p1, p2, pop_size, delta, alloc_ratio, alpha, power, deff, resp_rate = (
            params["p1"],
            params["p2"],
            params["pop_size"],
            params["delta"],
            params["alloc_ratio"],
            params["alpha"],
            params["power"],
            params["deff"],
            params["resp_rate"],
        )
    else:
        all_scalars = all(
            isinstance(x, (int, float)) for x in [p1, p2, delta, alloc_ratio, alpha, power]
        )
        assert all_scalars, "All inputs must be scalars when not stratified."

    if stratified:
        _p1_map = cast(DomainScalarMap, p1)
        _p2_map = cast(DomainScalarMap, p2)
        epsilon: DomainScalarMap = {s: _p2_map[s] - _p1_map[s] for s in _p1_map}
    else:
        epsilon = cast(Number, p2) - cast(Number, p1)

    _method = _normalize_two_props_method(method)
    _var_mode = _normalize_prop_var_mode(var_mode)
    if _method is _TwoPropsSizeMethod.WALD:
        n0 = _wald_sample_size_two_props(
            two_sides=two_sides,
            epsilon=epsilon,
            prop_1=p1,
            prop_2=p2,
            delta=delta,
            kappa=alloc_ratio,
            alpha=alpha,
            power=power,
            var_mode=_var_mode,
        )
    elif _method is _TwoPropsSizeMethod.MIETTINEN_NURMINEN:
        raise NotImplementedError("miettinen-nurminen method is not implemented yet.")
    elif _method is _TwoPropsSizeMethod.NEWCOMBE:
        raise NotImplementedError("newcombe method is not implemented yet.")
    else:
        raise NotImplementedError("farrington-manning method is not implemented yet.")

    # Pipeline: n0 (SRS) -> DEFF -> FPC -> nonresponse (see estimate_prop).
    n1_deff = _apply_deff_pair(n=n0, deff=deff)
    n2_fpc = _apply_fpc_srswor_pair(n0=n1_deff, pop_size=pop_size) if _has_pop(pop_size) else n1_deff
    n_final = _apply_nonresponse_pair(n=n2_fpc, resp_rate=resp_rate)
    _build_sizes(
        ss,
        stratified=stratified,
        strata=strata if stratified else None,
        n0=n0,
        n1_deff=n1_deff,
        n2_fpc=n2_fpc,
        n_final=n_final,
    )
    return ss


def compare_means(
    ss: SampleSize,
    mu1: Number | DomainScalarMap,
    mu2: Number | DomainScalarMap,
    sigma1: Number | DomainScalarMap,
    sigma2: Number | DomainScalarMap | None = None,
    *,
    pop_size: Number | DomainScalarMap | None = None,
    two_sides: bool = True,
    delta: Number | DomainScalarMap = 0.0,
    alloc_ratio: Number | DomainScalarMap = 1.0,
    method: Literal["wald"] = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    power: Number | DomainScalarMap = 0.80,
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
) -> SampleSize:
    """
    Compute required sample size for comparing two means.

    Parameters
    ----------
    ss : SampleSize
        The SampleSize instance to update.
    mu1 : Number | DomainScalarMap
        Mean in group 1. Scalar or per-stratum mapping.
    mu2 : Number | DomainScalarMap
        Mean in group 2. Scalar or per-stratum mapping.
    sigma1 : Number | DomainScalarMap
        Standard deviation in group 1. Scalar or per-stratum mapping.
    sigma2 : Number | DomainScalarMap | None, default None
        Standard deviation in group 2. If None, assumes equal variances
        (sigma2 = sigma1).
    pop_size : Number | DomainScalarMap | None, default None
        Target population size. If None, no finite population correction is applied.
    two_sides : bool, default True
        Whether to use a two-sided test.
    delta : Number | DomainScalarMap, default 0.0
        Non-inferiority / equivalence margin.
    alloc_ratio : Number | DomainScalarMap, default 1.0
        Allocation ratio n2/n1.
    method : str, default 'wald'
        Calculation method. Only ``'wald'`` is currently implemented.
    alpha : Number | DomainScalarMap, default 0.05
        Significance level.
    power : Number | DomainScalarMap, default 0.80
        Desired statistical power.
    deff : Number | DomainScalarMap, default 1.0
        Design effect. Scalar or per-stratum mapping.
    resp_rate : Number | DomainScalarMap, default 1.0
        Anticipated response rate. Scalar or per-stratum mapping.

    Returns
    -------
    SampleSize
        The updated SampleSize instance (chainable).
    """
    from svy.errors.method_errors import MethodError

    ss._param = PopParam.MEAN

    if method != "wald":
        raise MethodError.invalid_choice(
            where="SampleSize.compare_means",
            param="method",
            got=method,
            allowed=["wald"],
        )

    stratified = any(
        isinstance(v, Mapping)
        for v in [
            mu1,
            mu2,
            sigma1,
            sigma2,
            pop_size,
            delta,
            alloc_ratio,
            alpha,
            power,
            deff,
            resp_rate,
        ]
    )

    if stratified:
        strata = _get_keys_from_maps(
            **{
                k: v
                for k, v in dict(
                    mu1=mu1,
                    mu2=mu2,
                    sigma1=sigma1,
                    sigma2=sigma2,
                    pop_size=pop_size,
                    delta=delta,
                    alloc_ratio=alloc_ratio,
                    alpha=alpha,
                    power=power,
                    deff=deff,
                    resp_rate=resp_rate,
                ).items()
                if isinstance(v, Mapping)
            }
        )
        params = {
            "mu1": mu1,
            "mu2": mu2,
            "sigma1": sigma1,
            "pop_size": pop_size,
            "delta": delta,
            "alloc_ratio": alloc_ratio,
            "alpha": alpha,
            "power": power,
            "deff": deff,
            "resp_rate": resp_rate,
        }
        if sigma2 is not None:
            params["sigma2"] = sigma2
        _broadcast_scalars(strata, params)
        mu1, mu2, sigma1, pop_size, delta, alloc_ratio, alpha, power, deff, resp_rate = (
            params["mu1"],
            params["mu2"],
            params["sigma1"],
            params["pop_size"],
            params["delta"],
            params["alloc_ratio"],
            params["alpha"],
            params["power"],
            params["deff"],
            params["resp_rate"],
        )
        sigma2 = params.get("sigma2")
    else:
        all_scalars = all(
            isinstance(x, (int, float))
            for x in [mu1, mu2, sigma1, delta, alloc_ratio, alpha, power]
        ) and (sigma2 is None or isinstance(sigma2, (int, float)))
        assert all_scalars, "All inputs must be scalars when not stratified."

    if stratified:
        _mu1_map = cast(DomainScalarMap, mu1)
        _mu2_map = cast(DomainScalarMap, mu2)
        epsilon: Number | DomainScalarMap = {s: _mu2_map[s] - _mu1_map[s] for s in _mu1_map}
    else:
        epsilon = cast(Number, mu2) - cast(Number, mu1)

    var_mode = _MeanVarMode.EQUAL_VAR if sigma2 is None else _MeanVarMode.UNEQUAL_VAR
    n0 = _wald_sample_size_two_means(
        two_sides=two_sides,
        epsilon=epsilon,
        delta=delta,
        sigma_1=sigma1,
        sigma_2=sigma2,
        alloc_ratio=alloc_ratio,
        alpha=alpha,
        power=power,
        var_mode=var_mode,
    )

    # Pipeline: n0 (SRS) -> DEFF -> FPC -> nonresponse (see estimate_prop).
    n1_deff = _apply_deff_pair(n=n0, deff=deff)
    n2_fpc = _apply_fpc_srswor_pair(n0=n1_deff, pop_size=pop_size) if _has_pop(pop_size) else n1_deff
    n_final = _apply_nonresponse_pair(n=n2_fpc, resp_rate=resp_rate)
    _build_sizes(
        ss,
        stratified=stratified,
        strata=strata if stratified else None,
        n0=n0,
        n1_deff=n1_deff,
        n2_fpc=n2_fpc,
        n_final=n_final,
    )
    return ss
