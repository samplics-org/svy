# src/svy/size/estimation_goals.py
"""
Estimation-goal sample size functions.

Each function takes a SampleSize instance and returns it (chainable).
The SampleSize class in base.py delegates to these functions directly.

Adding a new estimation goal:
  1. Implement it here as a module-level function.
  2. Import it in base.py and add a one-liner delegation method.
  3. If it has a pure algorithm, put that in svy/engine/size_and_power/size.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

from svy.core.enumerations import OnePropSizeMethod as _OnePropSizeMethod
from svy.core.enumerations import PopParam
from svy.core.types import DomainScalarMap, Number
from svy.engine.size_and_power.size import (
    _apply_deff,
    _apply_fpc_srswor,
    _apply_nonresponse,
    _fleiss_sample_size_prop,
    _wald_sample_size_mean,
    _wald_sample_size_prop,
)
from svy.size._normalize import _normalize_one_prop_method
from svy.size.types import Size
from svy.utils.helpers import _get_keys_from_maps


if TYPE_CHECKING:
    from svy.size.base import SampleSize


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _broadcast_scalars(strata: list, params: dict) -> None:
    """Broadcast any non-Mapping param to a per-stratum dict in-place."""
    m = len(strata)
    for k, v in list(params.items()):
        if not isinstance(v, Mapping):
            params[k] = dict(zip(strata, [v] * m))


def _has_pop(pop_size) -> bool:
    """True if pop_size carries at least one non-None value."""
    if isinstance(pop_size, Mapping):
        return any(v is not None for v in pop_size.values())
    return pop_size is not None


def _build_sizes(ss, *, stratified: bool, strata, n0, n1_fpc, n2_deff, n_final) -> None:
    """Assign Size object(s) to ss._size."""
    if not stratified:
        ss._size = Size(
            stratum=None,
            n0=cast(Number, n0),
            n1_fpc=cast(Number, n1_fpc),
            n2_deff=cast(Number, n2_deff),
            n=cast(Number, n_final),
        )
    else:
        _n0 = cast(DomainScalarMap, n0)
        _n1 = cast(DomainScalarMap, n1_fpc)
        _n2 = cast(DomainScalarMap, n2_deff)
        _nf = cast(DomainScalarMap, n_final)
        ss._size = [
            Size(stratum=str(s), n0=_n0[s], n1_fpc=_n1[s], n2_deff=_n2[s], n=_nf[s]) for s in _n0
        ]


def estimate_prop(
    ss: SampleSize,
    p: Number | DomainScalarMap,
    moe: Number | DomainScalarMap,
    *,
    pop_size: Number | DomainScalarMap | None = None,
    method: Literal["wald", "fleiss"] | DomainScalarMap = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
) -> SampleSize:
    """
    Compute required sample size for estimating a proportion.

    Parameters
    ----------
    ss : SampleSize
        The SampleSize instance to update.
    p : Number | DomainScalarMap
        Expected proportion. Scalar for unstratified; dict for stratified.
    moe : Number | DomainScalarMap
        Desired margin of error (half-width of CI).
    pop_size : Number | DomainScalarMap | None, default None
        Target population size. If None, no finite population correction is applied.
        Scalar or per-stratum mapping.
    method : str, default 'wald'
        Calculation method: ``'wald'`` or ``'fleiss'``.
    alpha : Number | DomainScalarMap, default 0.05
        Significance level.
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

    stratified = any(isinstance(v, Mapping) for v in [p, moe, pop_size, alpha, deff, resp_rate])

    if stratified:
        strata = _get_keys_from_maps(
            **{
                k: v
                for k, v in dict(
                    p=p,
                    moe=moe,
                    pop_size=pop_size,
                    alpha=alpha,
                    deff=deff,
                    resp_rate=resp_rate,
                ).items()
                if isinstance(v, Mapping)
            }
        )
        m = len(strata)
        params = {
            "p": p,
            "moe": moe,
            "pop_size": pop_size,
            "alpha": alpha,
            "deff": deff,
            "resp_rate": resp_rate,
        }
        _broadcast_scalars(strata, params)
        p, moe, pop_size, alpha, deff, resp_rate = (
            params["p"],
            params["moe"],
            params["pop_size"],
            params["alpha"],
            params["deff"],
            params["resp_rate"],
        )
    else:
        all_scalars = all(isinstance(x, (int, float)) for x in [p, moe, alpha])
        assert all_scalars, "All inputs must be scalars when not stratified."

    _method = _normalize_one_prop_method(method)
    if _method is _OnePropSizeMethod.WALD:
        n0 = _wald_sample_size_prop(target=p, half_ci=moe, alpha=alpha)
    else:
        n0 = _fleiss_sample_size_prop(target=p, half_ci=moe, alpha=alpha)

    n1_fpc = _apply_fpc_srswor(n0=n0, pop_size=pop_size) if _has_pop(pop_size) else n0
    n2_deff = _apply_deff(n=n1_fpc, deff=deff)
    n_final = _apply_nonresponse(n=n2_deff, resp_rate=resp_rate)
    _build_sizes(
        ss,
        stratified=stratified,
        strata=strata if stratified else None,
        n0=n0,
        n1_fpc=n1_fpc,
        n2_deff=n2_deff,
        n_final=n_final,
    )
    return ss


def estimate_mean(
    ss: SampleSize,
    sigma: Number | DomainScalarMap,
    moe: Number | DomainScalarMap,
    *,
    pop_size: Number | DomainScalarMap | None = None,
    method: Literal["wald", "fleiss"] | DomainScalarMap = "wald",
    alpha: Number | DomainScalarMap = 0.05,
    deff: Number | DomainScalarMap = 1.0,
    resp_rate: Number | DomainScalarMap = 1.0,
) -> SampleSize:
    """
    Compute required sample size for estimating a mean.

    Parameters
    ----------
    ss : SampleSize
        The SampleSize instance to update.
    sigma : Number | DomainScalarMap
        Expected population standard deviation. Scalar or per-stratum mapping.
    moe : Number | DomainScalarMap
        Desired margin of error (half-width of CI).
    pop_size : Number | DomainScalarMap | None, default None
        Target population size. If None, no finite population correction is applied.
        Scalar or per-stratum mapping.
    method : str, default 'wald'
        Calculation method: ``'wald'`` or ``'fleiss'``. Only ``'wald'`` is currently implemented.
    alpha : Number | DomainScalarMap, default 0.05
        Significance level.
    deff : Number | DomainScalarMap, default 1.0
        Design effect. Scalar or per-stratum mapping.
    resp_rate : Number | DomainScalarMap, default 1.0
        Anticipated response rate. Scalar or per-stratum mapping.

    Returns
    -------
    SampleSize
        The updated SampleSize instance (chainable).
    """
    ss._param = PopParam.MEAN

    stratified = any(
        isinstance(v, Mapping) for v in [sigma, moe, pop_size, alpha, deff, resp_rate]
    )

    if stratified:
        strata = _get_keys_from_maps(
            **{
                k: v
                for k, v in dict(
                    sigma=sigma,
                    moe=moe,
                    pop_size=pop_size,
                    alpha=alpha,
                    deff=deff,
                    resp_rate=resp_rate,
                ).items()
                if isinstance(v, Mapping)
            }
        )
        m = len(strata)
        params = {
            "sigma": sigma,
            "moe": moe,
            "pop_size": pop_size,
            "alpha": alpha,
            "deff": deff,
            "resp_rate": resp_rate,
        }
        _broadcast_scalars(strata, params)
        sigma, moe, pop_size, alpha, deff, resp_rate = (
            params["sigma"],
            params["moe"],
            params["pop_size"],
            params["alpha"],
            params["deff"],
            params["resp_rate"],
        )
    else:
        all_scalars = all(isinstance(x, (int, float)) for x in [sigma, moe, alpha])
        assert all_scalars, "All inputs must be scalars when not stratified."

    _method = _normalize_one_prop_method(method)
    if _method is _OnePropSizeMethod.WALD:
        n0 = _wald_sample_size_mean(half_ci=moe, sigma=sigma, alpha=alpha)
    else:
        raise NotImplementedError("Fleiss method is not implemented for the mean.")

    n1_fpc = _apply_fpc_srswor(n0=n0, pop_size=pop_size) if _has_pop(pop_size) else n0
    n2_deff = _apply_deff(n=n1_fpc, deff=deff)
    n_final = _apply_nonresponse(n=n2_deff, resp_rate=resp_rate)
    _build_sizes(
        ss,
        stratified=stratified,
        strata=strata if stratified else None,
        n0=n0,
        n1_fpc=n1_fpc,
        n2_deff=n2_deff,
        n_final=n_final,
    )
    return ss
