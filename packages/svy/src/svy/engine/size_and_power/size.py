# src/svy/engine/size_and_power/size.py
from __future__ import annotations

import math

from collections.abc import Mapping
from typing import Literal, cast

import numpy as np

from numpy._core.numeric import inf
from scipy.stats import norm

from svy.core.enumerations import MeanVarMode, PropVarMode
from svy.core.types import (
    Array,
    Category,
    DomainScalarMap,
    FloatArray,
    Number,
)


# ============================================================
# Type-safe lookup helper
# ============================================================


def _lookup(m: DomainScalarMap | Number, k: Category) -> Number:
    """Type-safe key lookup that handles scalar broadcast."""
    if isinstance(m, dict):
        return m[k]
    return m  # already Number


PairMap = dict[Category, tuple[Number, Number]]
PairInput = Number | tuple[Number, Number] | Mapping[Category, Number | tuple[Number, Number]]


def _lookup_pair(
    m: Number | tuple[Number, Number] | Mapping[Category, Number | tuple[Number, Number]],
    k: Category,
) -> Number | tuple[Number, Number]:
    """Type-safe key lookup for pair maps."""
    if isinstance(m, Mapping):
        typed_m = cast(dict[Category, Number | tuple[Number, Number]], m)
        return typed_m[k]
    return m  # Number or tuple — already correct type


def _resolve_maps(
    a: Number | DomainScalarMap,
    b: Number | DomainScalarMap,
    *,
    name_a: str = "a",
    name_b: str = "b",
) -> tuple[dict[Category, Number], dict[Category, Number], list[Category]]:
    """
    Resolve two scalar-or-map arguments into aligned dicts + a shared key list.

    Both-mapping: keys must match.
    One-mapping:  the scalar is broadcast to the other's keys.
    Returns (dict_a, dict_b, keys).
    """
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        da = cast(dict[Category, Number], dict(a))
        db = cast(dict[Category, Number], dict(b))
        ka, kb = set(da), set(db)
        if ka != kb:
            raise ValueError(
                f"Domain key mismatch: missing in {name_b}={ka - kb or '{}'}, "
                f"missing in {name_a}={kb - ka or '{}'}"
            )
        keys = list(ka)
    elif isinstance(a, Mapping):
        da = cast(dict[Category, Number], dict(a))
        keys = list(da)
        scalar_b = cast(Number, b)
        db = {k: scalar_b for k in keys}
    else:
        db = cast(dict[Category, Number], dict(b))
        keys = list(db)
        scalar_a = cast(Number, a)
        da = {k: scalar_a for k in keys}
    return da, db, keys


# ============================================================
# Adjsut for nonrespone
# ============================================================


def _apply_nonresponse(
    *,
    n: Number | DomainScalarMap,
    resp_rate: Number | DomainScalarMap,
    strict: bool = False,
) -> Number | DomainScalarMap:
    """
    Adjust for nonresponse:
        n_nr = n / resp_rate

    Behavior:
    - Scalar/scalar -> scalar.
    - If either arg is a mapping -> returns a dict keyed by domains.
      * Scalars are broadcast to the mapping's keys.
      * If both mappings, keys must match.

    Validation:
    - resp_rate must be in (0, 1].
      * strict=False: if invalid (<=0, >1, non-finite) -> leave n unchanged for that key.
      * strict=True: raise ValueError on invalid entries.
    """

    def _adj(nv: Number, rv: Number) -> float:
        n_f = float(nv)
        r_f = float(rv)
        valid = (r_f > 0.0) and (r_f <= 1.0)
        if not valid or not (r_f == r_f):  # NaN check
            if strict:
                raise ValueError(f"Invalid response rate: {rv} (must be in (0,1]).")
            return float(math.ceil(n_f))
        return float(math.ceil(n_f / r_f))

    # scalar-scalar case
    if not isinstance(n, Mapping) and not isinstance(resp_rate, Mapping):
        return _adj(n, resp_rate)

    n_d, r_d, typed_keys = _resolve_maps(n, resp_rate, name_a="n", name_b="resp_rate")
    return {k: _adj(n_d[k], r_d[k]) for k in typed_keys}


# ============================================================
# Adjsut for the design effect
# ============================================================


def _apply_deff(
    *,
    n: Number | DomainScalarMap,
    deff: Number | DomainScalarMap,
) -> Number | DomainScalarMap:
    """
    Apply design effect to sample size:
        n_deff = deff * n

    Behavior:
      - If both inputs are scalars -> returns a scalar.
      - If either input is a mapping -> returns a dict keyed by domains.
        * Scalars are broadcast to the mapping's keys.
        * Mapping keys must match if both are mappings.

    Safeguards:
      - If deff is non-finite or <= 0 for a key, returns the unadjusted n for that key.
    """

    def _adjust(n_val: Number, d_val: Number) -> float:
        n_f = float(n_val)
        d_f = float(d_val)
        if not (d_f > 0.0):
            return float(math.ceil(n_f))
        return float(math.ceil(d_f * n_f))

    # Scalar-scalar
    if not isinstance(n, Mapping) and not isinstance(deff, Mapping):
        return _adjust(n, deff)

    n_d, d_d, typed_keys = _resolve_maps(n, deff, name_a="n", name_b="deff")
    return {k: _adjust(n_d[k], d_d[k]) for k in typed_keys}


# ============================================================
# Apply the fpc
# ============================================================


def _apply_fpc_srswor(
    *,
    n0: Number | DomainScalarMap,
    pop_size: Number | DomainScalarMap,
) -> Number | DomainScalarMap:
    """
    Adjusts base sample size n0 by the finite population correction (FPC) for SRSWOR:
        n_fpc = (N * n0) / (N + n0 - 1)

    Behavior:
      - If both inputs are scalars -> returns a scalar.
      - If either input is a mapping -> returns a dict keyed by domains.
        * Scalars are broadcast to the mapping's keys.
        * Mapping keys must match if both are mappings.

    Safeguards:
      - If N <= 0 or denominator <= 0, returns n0 (no adjustment).
    """

    def _adjust(n0_val: Number, N_val: Number) -> float:
        n0f = float(n0_val)
        Nf = float(N_val) if N_val is not None else 0.0
        denom = Nf + n0f - 1.0
        if Nf <= 0.0 or denom <= 0.0:
            return float(math.ceil(n0f))
        return float(math.ceil((Nf * n0f) / denom))

    # Scalar-scalar
    if not isinstance(n0, Mapping) and not isinstance(pop_size, Mapping):
        return _adjust(n0, pop_size)

    n0_d, N_d, typed_keys = _resolve_maps(n0, pop_size, name_a="n0", name_b="pop_size")
    return {k: _adjust(n0_d[k], N_d[k]) for k in typed_keys}


# ============================================================
# Utilities
# ============================================================

_EPS = 1e-12


def _as_float(x: Number) -> float:
    return float(x)


def _as_f64(a: Array | Number) -> FloatArray:
    """Typed ndarray[float64] view/copy."""
    return cast(FloatArray, np.asarray(a, dtype=np.float64))


def _as_like(x: float | Array, ref: FloatArray) -> FloatArray:
    """Broadcast a scalar or array x to the shape of ref as float64."""
    if isinstance(x, (int, float)):
        return cast(FloatArray, np.full(ref.shape, float(x), dtype=np.float64))
    arr = cast(FloatArray, np.asarray(x, dtype=np.float64))
    if arr.shape == ref.shape:
        return arr
    return cast(FloatArray, np.broadcast_to(arr, ref.shape).astype(np.float64, copy=False))


def _zcrit(alpha: float, two_sides: bool) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    return float(norm.ppf(1 - alpha / 2.0 if two_sides else 1 - alpha))


def _zcrit_from_type(
    alpha: float | Array, ttype: Literal["two-sided", "less", "greater"]
) -> float | FloatArray:
    two_sides = ttype == "two-sided"
    if isinstance(alpha, (int, float)):
        return _zcrit(float(alpha), two_sides)
    a = _as_f64(alpha)
    z = norm.ppf(1 - a / 2.0 if two_sides else 1 - a)
    return cast(FloatArray, np.asarray(z, dtype=np.float64))


def _ensure_same_keys(*maps: DomainScalarMap | None) -> set[Category]:
    keys: list[set[Category]] = [set(m.keys()) for m in maps if m is not None]
    if not keys:
        return set()
    base = keys[0]
    for k in keys[1:]:
        if k != base:
            raise KeyError("All provided maps must share identical keys")
    return base


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float))


def _is_array(x: object) -> bool:
    return isinstance(x, np.ndarray)


def _is_map(x: object) -> bool:
    return isinstance(x, dict)


# ============================================================
# Wald SS for a single proportion
# ============================================================


def _wald_sample_size_prop(
    *,
    target: Number | DomainScalarMap,
    half_ci: Number | DomainScalarMap,
    alpha: Number | DomainScalarMap = 0.05,
) -> Number | DomainScalarMap:
    if _is_number(target) and _is_number(alpha):
        return _wald_sample_size_prop_scalar(
            target=cast(Number, target),
            half_ci=cast(Number, half_ci),
            alpha=cast(Number, alpha),
        )
    elif _is_map(target) and _is_map(alpha):
        return _wald_sample_size_prop_map(
            target=cast(DomainScalarMap, target),
            half_ci=cast(DomainScalarMap, half_ci),
            alpha=alpha,
        )
    else:
        raise TypeError(f"Unsupported type for target: {type(target)}")


def _wald_sample_size_prop_scalar(
    *,
    target: Number,
    half_ci: Number,
    alpha: Number,
) -> float:
    p = _as_float(target)
    d = max(_EPS, _as_float(half_ci))
    z = _zcrit(_as_float(alpha), True)
    n = (z * z) * p * (1.0 - p) / (d * d)
    return float(math.ceil(n))


def _wald_sample_size_prop_map(
    *,
    target: DomainScalarMap,
    half_ci: DomainScalarMap,
    alpha: DomainScalarMap | Number,
) -> DomainScalarMap:
    keys = _ensure_same_keys(target, half_ci)
    out: dict[Category, float] = {}
    for k in keys:
        out[k] = _wald_sample_size_prop_scalar(
            target=target[k],
            half_ci=half_ci[k],
            alpha=_lookup(alpha, k),
        )
    return out


# ============================================================
# Fleiss SS for one proportion (better near 0/1)
# ============================================================


def _fleiss_factor_scalar(*, p: float, d: float) -> float:
    if 0.0 <= p < d or 1.0 - d < p <= 1.0:
        return 8.0 * d * (1.0 - 2.0 * d)
    if d <= p < 0.3:
        return 4.0 * (p + d) * (1.0 - p - d)
    if 0.7 < p <= 1.0 - d:
        return 4.0 * (p - d) * (1.0 - p + d)
    if 0.3 <= p <= 0.7:
        return 1.0
    raise ValueError("Invalid (p, d) for Fleiss factor.")


def _fleiss_sample_size_prop(
    *,
    target: Number | DomainScalarMap,
    half_ci: Number | DomainScalarMap,
    alpha: Number | DomainScalarMap = 0.05,
) -> Number | DomainScalarMap:
    if _is_number(target) and _is_number(half_ci) and _is_number(alpha):
        return _fleiss_sample_size_prop_scalar(
            target=cast(Number, target),
            half_ci=cast(Number, half_ci),
            alpha=cast(Number, alpha),
        )
    if _is_map(target) and _is_map(half_ci) and _is_map(alpha):
        return _fleiss_sample_size_prop_map(
            target=cast(DomainScalarMap, target),
            half_ci=cast(DomainScalarMap, half_ci),
            alpha=alpha,
        )
    raise ValueError("Invalid input type for Fleiss prop.")


def _fleiss_sample_size_prop_scalar(
    *,
    target: Number,
    half_ci: Number,
    alpha: Number,
) -> float:
    p = _as_float(target)
    d = max(_EPS, _as_float(half_ci))
    z = _zcrit(_as_float(alpha), True)
    f = _fleiss_factor_scalar(p=p, d=d)
    n = (f * (z * z) / (4.0 * d * d)) + (1.0 / d) - (2.0 * z * z) + ((z + 2.0) / f)
    return float(math.ceil(n))


def _fleiss_sample_size_prop_map(
    *,
    target: DomainScalarMap,
    half_ci: DomainScalarMap,
    alpha: DomainScalarMap | Number,
) -> dict[Category, float]:
    keys = _ensure_same_keys(target, half_ci)
    out: dict[Category, float] = {}
    for k in keys:
        out[k] = _fleiss_sample_size_prop_scalar(
            target=target[k],
            half_ci=half_ci[k],
            alpha=_lookup(alpha, k),
        )
    return out


# ============================================================
# Wilson for proportions
# ============================================================


def _wilson_sample_size_prop(
    *,
    target: Number | DomainScalarMap,
    half_ci: Number | DomainScalarMap,
    alpha: Number | DomainScalarMap = 0.05,
) -> Number | DomainScalarMap:
    if _is_number(target) and _is_number(alpha):
        return _wilson_sample_size_prop_scalar(
            target=cast(Number, target),
            half_ci=cast(Number, half_ci),
            alpha=cast(Number, alpha),
        )
    elif _is_map(target) and _is_map(alpha):
        return _wilson_sample_size_prop_map(
            target=cast(DomainScalarMap, target),
            half_ci=cast(DomainScalarMap, half_ci),
            alpha=alpha,
        )
    else:
        raise TypeError(f"Unsupported type for target: {type(target)}")


def _wilson_sample_size_prop_scalar(
    target: Number,
    half_ci: Number,
    alpha: Number,
) -> float:
    p = _as_float(target)
    d = max(_EPS, _as_float(half_ci))
    if not (0.0 < d < 0.5):
        raise ValueError("Wilson requires 0 < half_ci < 0.5 for a proportion.")

    z = _zcrit(_as_float(alpha), True)

    pq = p * (1.0 - p)
    num = -(2.0 * d * d - pq) - math.sqrt(d * d * (1.0 - 2.0 * p) ** 2 + (pq * pq))
    denom = 2.0 * (z * z) * (d * d - 0.25)

    u = num / denom
    if u <= 0.0 or not math.isfinite(u):
        u = _EPS
    n0 = 1.0 / u

    return float(math.ceil(n0))


def _wilson_sample_size_prop_map(
    *,
    target: DomainScalarMap,
    half_ci: DomainScalarMap,
    alpha: DomainScalarMap | Number,
) -> dict[Category, float]:
    keys = _ensure_same_keys(target, half_ci)
    out: dict[Category, float] = {}
    for k in keys:
        out[k] = _wilson_sample_size_prop_scalar(
            target=target[k],
            half_ci=half_ci[k],
            alpha=_lookup(alpha, k),
        )
    return out


# ============================================================
# Wald SS for a single mean (known sigma)
# ============================================================


def _wald_sample_size_mean(
    *,
    half_ci: Number | DomainScalarMap,
    sigma: Number | DomainScalarMap,
    alpha: Number | DomainScalarMap = 0.05,
) -> Number | DomainScalarMap:
    if _is_number(half_ci) and _is_number(sigma) and _is_number(alpha):
        return _wald_sample_size_mean_scalar(
            half_ci=cast(Number, half_ci),
            sigma=cast(Number, sigma),
            alpha=cast(Number, alpha),
        )
    elif _is_map(half_ci) and _is_map(sigma) and _is_map(alpha):
        return _wald_sample_size_mean_map(
            half_ci=cast(DomainScalarMap, half_ci),
            sigma=cast(DomainScalarMap, sigma),
            alpha=alpha,
        )
    else:
        raise ValueError("Invalid input types")


def _wald_sample_size_mean_scalar(
    *,
    half_ci: Number,
    sigma: Number,
    alpha: Number,
) -> float:
    d = max(_EPS, _as_float(half_ci))
    s2 = max(_EPS, _as_float(sigma)) ** 2
    z2 = _zcrit(_as_float(alpha), True) ** 2
    n = z2 * s2 / (d * d)
    return float(math.ceil(n))


def _wald_sample_size_mean_map(
    *,
    half_ci: DomainScalarMap,
    sigma: DomainScalarMap,
    alpha: DomainScalarMap | Number,
) -> dict[Category, float]:
    keys = _ensure_same_keys(half_ci, sigma)
    out: dict[Category, float] = {}
    for k in keys:
        out[k] = _wald_sample_size_mean_scalar(
            half_ci=half_ci[k],
            sigma=sigma[k],
            alpha=_lookup(alpha, k),
        )
    return out


# ============================================================
# Power-based SS for one mean
# ============================================================


def _wald_sample_size_one_mean(
    *,
    two_sides: bool,
    epsilon: Number | DomainScalarMap,
    delta: Number | DomainScalarMap,
    sigma: Number | DomainScalarMap,
    alpha: Number | DomainScalarMap,
    power: Number | DomainScalarMap,
) -> Number | DomainScalarMap:
    if (
        _is_number(epsilon)
        and _is_number(delta)
        and _is_number(sigma)
        and _is_number(alpha)
        and _is_number(power)
    ):
        return _wald_sample_size_one_mean_scalar(
            two_sides=two_sides,
            epsilon=cast(Number, epsilon),
            delta=cast(Number, delta),
            sigma=cast(Number, sigma),
            alpha=cast(Number, alpha),
            power=cast(Number, power),
        )
    elif _is_map(epsilon) and _is_map(alpha) and _is_map(power):
        return _wald_sample_size_one_mean_map(
            two_sides=two_sides,
            epsilon=cast(DomainScalarMap, epsilon),
            delta=cast(DomainScalarMap, delta),
            sigma=cast(DomainScalarMap, sigma),
            alpha=alpha,
            power=power,
        )
    else:
        raise ValueError("Invalid input types")


def _wald_sample_size_one_mean_scalar(
    *,
    two_sides: bool,
    epsilon: Number,
    delta: Number,
    sigma: Number,
    alpha: Number,
    power: Number,
) -> float:
    d0 = _as_float(delta)
    eps = abs(_as_float(epsilon))
    s = max(_EPS, _as_float(sigma))
    a = _as_float(alpha)
    b = _as_float(power)

    if two_sides and d0 == 0.0:
        z_a = _zcrit(a, True)
        z_b = float(norm.ppf(b))
    elif two_sides:
        z_a = _zcrit(a, False)
        z_b = float(norm.ppf((1.0 + b) / 2.0))
    else:
        z_a = _zcrit(a, False)
        z_b = float(norm.ppf(b))

    denom = max(_EPS, d0 - eps)
    n = ((z_a + z_b) * s / denom) ** 2
    return float(math.ceil(n))


def _wald_sample_size_one_mean_map(
    *,
    two_sides: bool,
    epsilon: DomainScalarMap,
    delta: DomainScalarMap,
    sigma: DomainScalarMap,
    alpha: DomainScalarMap | Number,
    power: DomainScalarMap | Number,
) -> dict[Category, Number]:
    keys = _ensure_same_keys(
        epsilon,
        delta,
        sigma,
        alpha if isinstance(alpha, dict) else None,
        power if isinstance(power, dict) else None,
    )
    out: dict[Category, Number] = {}
    for k in keys:
        out[k] = _wald_sample_size_one_mean_scalar(
            two_sides=two_sides,
            epsilon=epsilon[k],
            delta=delta[k],
            sigma=sigma[k],
            alpha=_lookup(alpha, k),
            power=_lookup(power, k),
        )
    return out


# ============================================================
# Sample size for two means or proportions
# ============================================================


def _to_pair(x: Number | tuple[Number, Number]) -> tuple[float, float]:
    """Coerce a scalar or 2-tuple to (float, float), broadcasting scalars."""
    if isinstance(x, tuple):
        if len(x) != 2:
            raise ValueError(f"Expected a 2-tuple, got length {len(x)}.")
        return float(x[0]), float(x[1])
    xf = float(x)
    return xf, xf


def _apply_nonresponse_pair(
    *,
    n: tuple[Number, Number] | Mapping[Category, tuple[Number, Number]],
    resp_rate: Number | tuple[Number, Number] | Mapping[Category, Number | tuple[Number, Number]],
    strict: bool = False,
) -> tuple[Number, Number] | dict[Category, tuple[Number, Number]]:
    def _valid_rr(r: float) -> bool:
        return math.isfinite(r) and (r > 0.0) and (r <= 1.0)

    def _adj_scalar(nv: Number, rv: Number) -> float:
        n_f = float(nv)
        r_f = float(rv)
        if not _valid_rr(r_f):
            if strict:
                raise ValueError(f"Invalid response rate: {rv} (must be finite and in (0, 1]).")
            return float(math.ceil(n_f))
        return float(math.ceil(n_f / r_f))

    def _adj_pair(
        n_pair: tuple[Number, Number], r_pair: tuple[Number, Number]
    ) -> tuple[float, float]:
        return (
            _adj_scalar(n_pair[0], r_pair[0]),
            _adj_scalar(n_pair[1], r_pair[1]),
        )

    if not isinstance(n, Mapping) and not isinstance(resp_rate, Mapping):
        n_pair = _to_pair(cast(Number | tuple[Number, Number], n))
        r_pair = _to_pair(cast(Number | tuple[Number, Number], resp_rate))
        return _adj_pair(n_pair, r_pair)

    n_typed = cast(Mapping[Category, tuple[Number, Number]], n) if isinstance(n, Mapping) else n
    rr_typed = (
        cast(Mapping[Category, Number | tuple[Number, Number]], resp_rate)
        if isinstance(resp_rate, Mapping)
        else resp_rate
    )

    if isinstance(n_typed, Mapping) and isinstance(rr_typed, Mapping):
        keys_n: set[Category] = set(n_typed.keys())
        keys_r: set[Category] = set(rr_typed.keys())
        if keys_n != keys_r:
            missing_in_r = keys_n - keys_r
            missing_in_n = keys_r - keys_n
            raise ValueError(
                f"Domain key mismatch: missing in resp_rate={missing_in_r or '{}'}, "
                f"missing in n={missing_in_n or '{}'}"
            )
        typed_keys: list[Category] = list(keys_n)
        n_map: Mapping[Category, tuple[Number, Number]] = n_typed
        r_map: dict[Category, tuple[float, float]] = {
            k: _to_pair(_lookup_pair(rr_typed, k)) for k in typed_keys
        }
    elif isinstance(n_typed, Mapping):
        typed_keys = list(set(n_typed.keys()))
        n_map = n_typed
        r_pair = _to_pair(cast(Number | tuple[Number, Number], rr_typed))
        r_map = {k: r_pair for k in typed_keys}
    else:
        assert isinstance(rr_typed, Mapping)
        typed_keys = list(set(rr_typed.keys()))
        r_map = {k: _to_pair(_lookup_pair(rr_typed, k)) for k in typed_keys}
        n_pair = _to_pair(cast(Number | tuple[Number, Number], n_typed))
        n_map = {k: n_pair for k in typed_keys}

    for k, v in n_map.items():
        if not isinstance(v, tuple) or len(v) != 2:
            raise TypeError(f"n[{k!r}] must be a 2-tuple, got {type(v)} with value {v!r}")

    out: dict[Category, tuple[float, float]] = {}
    for k in typed_keys:
        out[k] = _adj_pair(n_map[k], r_map[k])

    return out


def _apply_deff_pair(
    *,
    n: tuple[Number, Number] | Mapping[Category, tuple[Number, Number]],
    deff: Number | tuple[Number, Number] | Mapping[Category, Number | tuple[Number, Number]],
) -> tuple[Number, Number] | dict[Category, tuple[Number, Number]]:
    def _adjust_scalar(n_val: Number, d_val: Number) -> float:
        n_f = float(n_val)
        d_f = float(d_val)
        if not (d_f > 0.0):
            return float(math.ceil(n_f))
        return float(math.ceil(d_f * n_f))

    def _adjust_pair(
        n_pair: tuple[Number, Number], d_pair: tuple[Number, Number]
    ) -> tuple[float, float]:
        return (
            _adjust_scalar(n_pair[0], d_pair[0]),
            _adjust_scalar(n_pair[1], d_pair[1]),
        )

    if not isinstance(n, Mapping) and not isinstance(deff, Mapping):
        n_pair = _to_pair(cast(Number | tuple[Number, Number], n))
        d_pair = _to_pair(cast(Number | tuple[Number, Number], deff))
        return _adjust_pair(n_pair, d_pair)

    n_typed2 = cast(Mapping[Category, tuple[Number, Number]], n) if isinstance(n, Mapping) else n
    deff_typed = (
        cast(Mapping[Category, Number | tuple[Number, Number]], deff)
        if isinstance(deff, Mapping)
        else deff
    )

    if isinstance(n_typed2, Mapping) and isinstance(deff_typed, Mapping):
        keys_n2: set[Category] = set(n_typed2.keys())
        keys_d: set[Category] = set(deff_typed.keys())
        if keys_n2 != keys_d:
            missing_in_deff = keys_n2 - keys_d
            missing_in_n2 = keys_d - keys_n2
            raise ValueError(
                f"Domain key mismatch: "
                f"missing in deff={missing_in_deff or '{}'}, "
                f"missing in n={missing_in_n2 or '{}'}"
            )
        typed_keys2: list[Category] = list(keys_n2)
        n_map2: Mapping[Category, tuple[Number, Number]] = n_typed2
        d_map: dict[Category, tuple[float, float]] = {
            k: _to_pair(_lookup_pair(deff_typed, k)) for k in typed_keys2
        }
    elif isinstance(n_typed2, Mapping):
        typed_keys2 = list(set(n_typed2.keys()))
        n_map2 = n_typed2
        d_pair = _to_pair(cast(Number | tuple[Number, Number], deff_typed))
        d_map = {k: d_pair for k in typed_keys2}
    else:
        assert isinstance(deff_typed, Mapping)
        typed_keys2 = list(set(deff_typed.keys()))
        d_map = {k: _to_pair(_lookup_pair(deff_typed, k)) for k in typed_keys2}
        n_pair2 = _to_pair(cast(Number | tuple[Number, Number], n_typed2))
        n_map2 = {k: n_pair2 for k in typed_keys2}

    for k, v in n_map2.items():
        if not isinstance(v, tuple) or len(v) != 2:
            raise TypeError(f"n[{k!r}] must be a 2-tuple, got {type(v)} with value {v!r}")

    out2: dict[Category, tuple[float, float]] = {}
    for k in typed_keys2:
        out2[k] = _adjust_pair(n_map2[k], d_map[k])

    return out2


def _apply_fpc_srswor_pair(
    *,
    n0: tuple[Number, Number] | Mapping[Category, tuple[Number, Number]],
    pop_size: Number | tuple[Number, Number] | Mapping[Category, Number | tuple[Number, Number]],
) -> tuple[Number, Number] | dict[Category, tuple[Number, Number]]:
    def _adjust_scalar(n0_val: Number, N_val: Number) -> float:
        n0f = float(n0_val)
        Nf = float(N_val)
        denom = Nf + n0f - 1.0
        if Nf <= 0.0 or denom <= 0.0:
            return float(math.ceil(n0f))
        return float(math.ceil((Nf * n0f) / denom))

    def _adjust_pair(
        n0_pair: tuple[Number, Number], N_pair: tuple[Number, Number]
    ) -> tuple[float, float]:
        return (
            _adjust_scalar(n0_pair[0], N_pair[0]),
            _adjust_scalar(n0_pair[1], N_pair[1]),
        )

    if not isinstance(n0, Mapping) and not isinstance(pop_size, Mapping):
        n0_pair = _to_pair(cast(Number | tuple[Number, Number], n0))
        N_pair = _to_pair(cast(Number | tuple[Number, Number], pop_size))
        return _adjust_pair(n0_pair, N_pair)

    n0_typed = (
        cast(Mapping[Category, tuple[Number, Number]], n0) if isinstance(n0, Mapping) else n0
    )
    ps_typed = (
        cast(Mapping[Category, Number | tuple[Number, Number]], pop_size)
        if isinstance(pop_size, Mapping)
        else pop_size
    )

    if isinstance(n0_typed, Mapping) and isinstance(ps_typed, Mapping):
        keys_n03: set[Category] = set(n0_typed.keys())
        keys_N: set[Category] = set(ps_typed.keys())
        if keys_n03 != keys_N:
            missing_in_N = keys_n03 - keys_N
            missing_in_n03 = keys_N - keys_n03
            raise ValueError(
                f"Domain key mismatch: "
                f"missing in pop_size={missing_in_N or '{}'}, "
                f"missing in n0={missing_in_n03 or '{}'}"
            )
        typed_keys3: list[Category] = list(keys_n03)
        n0_map: Mapping[Category, tuple[Number, Number]] = n0_typed
        N_map: dict[Category, tuple[float, float]] = {
            k: _to_pair(_lookup_pair(ps_typed, k)) for k in typed_keys3
        }
    elif isinstance(n0_typed, Mapping):
        typed_keys3 = list(set(n0_typed.keys()))
        n0_map = n0_typed
        N_pair = _to_pair(cast(Number | tuple[Number, Number], ps_typed))
        N_map = {k: N_pair for k in typed_keys3}
    else:
        assert isinstance(ps_typed, Mapping)
        typed_keys3 = list(set(ps_typed.keys()))
        N_map = {k: _to_pair(_lookup_pair(ps_typed, k)) for k in typed_keys3}
        n0_pair2 = _to_pair(cast(Number | tuple[Number, Number], n0_typed))
        n0_map = {k: n0_pair2 for k in typed_keys3}

    for k, v in n0_map.items():
        if not isinstance(v, tuple) or len(v) != 2:
            raise TypeError(f"n0[{k!r}] must be a 2-tuple, got {type(v)} with value {v!r}")

    out3: dict[Category, tuple[float, float]] = {}
    for k in typed_keys3:
        out3[k] = _adjust_pair(n0_map[k], N_map[k])

    return out3


# --- Core utilities (internal) ---


def _zcrit_pair(
    *, two_sides: bool, alpha: float, power: float, delta: float
) -> tuple[float, float]:
    if two_sides and delta == 0.0:
        z_a = _zcrit(alpha, True)
        z_b = float(norm.ppf(power))
    elif two_sides:
        z_a = _zcrit(alpha, False)
        z_b = float(norm.ppf((1.0 + power) / 2.0))
    else:
        z_a = _zcrit(alpha, False)
        z_b = float(norm.ppf(power))
    return z_a, z_b


def _wald_n2_from_varfactor(*, var_factor: float, denom: float, z_a: float, z_b: float) -> int:
    n2 = var_factor * (((z_a + z_b) / denom) ** 2) if abs(denom) > _EPS else inf
    return int(math.ceil(n2))


def _prob_clip(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


def _choose_kappa(user_kappa: float | None, default_kappa: float) -> float:
    return max(_EPS, _as_float(user_kappa)) if user_kappa is not None else max(_EPS, default_kappa)


## Two MEANS


def _wald_two_group_sizes(
    *,
    v1: Number,
    v2: float,
    kappa: Number,
    denom: Number,
    two_sides: bool,
    alpha: Number,
    power: Number,
    delta: Number,
) -> tuple[Number, Number]:
    z_a, z_b = _zcrit_pair(
        two_sides=two_sides,
        alpha=float(alpha),
        power=float(power),
        delta=float(delta),
    )
    var_factor = v1 + (float(v2) / float(kappa))  # (float(v1) / float(kappa)) + v2
    n1 = _wald_n2_from_varfactor(var_factor=var_factor, denom=float(denom), z_a=z_a, z_b=z_b)
    n2 = int(math.ceil(float(kappa) * n1))
    return n1, n2


def _wald_sample_size_two_means(
    *,
    two_sides: bool,
    epsilon: Number | DomainScalarMap,
    delta: Number | DomainScalarMap,
    sigma_1: Number | DomainScalarMap,
    sigma_2: Number | DomainScalarMap | None,
    equal_var: bool | DomainScalarMap,
    alloc_ratio: Number | DomainScalarMap,
    alpha: Number | DomainScalarMap,
    power: Number | DomainScalarMap,
) -> tuple[Number, Number] | dict[Category, tuple[Number, Number]]:
    if (
        _is_number(epsilon)
        and _is_number(delta)
        and _is_number(sigma_1)
        and (_is_number(sigma_2) or sigma_2 is None)
        and _is_number(alpha)
        and _is_number(power)
    ):
        return _wald_sample_size_two_means_scalar(
            two_sides=two_sides,
            epsilon=cast(Number, epsilon),
            delta=cast(Number, delta),
            sigma_1=cast(Number, sigma_1),
            sigma_2=cast(Number, sigma_2) if sigma_2 is not None else None,
            kappa=cast(Number, alloc_ratio),
            alpha=cast(Number, alpha),
            power=cast(Number, power),
        )
    elif _is_map(epsilon) and _is_map(alpha) and _is_map(power):
        return _wald_sample_size_two_means_map(
            two_sides=two_sides,
            epsilon=cast(DomainScalarMap, epsilon),
            delta=cast(DomainScalarMap, delta),
            sigma_1=cast(DomainScalarMap, sigma_1),
            sigma_2=sigma_2 if isinstance(sigma_2, dict) else None,
            kappa=cast(DomainScalarMap, alloc_ratio),
            alpha=alpha,
            power=power,
        )
    else:
        raise ValueError("Invalid input types")


def _wald_sample_size_two_means_scalar(
    *,
    two_sides: bool,
    epsilon: Number,
    delta: Number,
    sigma_1: Number,
    sigma_2: Number | None = None,
    kappa: Number | None = None,
    alpha: Number,
    power: Number,
    var_mode: MeanVarMode = MeanVarMode.EQUAL_VAR,
) -> tuple[Number, Number]:
    d0 = _as_float(delta)
    eps = abs(_as_float(epsilon))
    s1 = max(_EPS, _as_float(sigma_1))
    s2 = max(_EPS, _as_float(sigma_2) if sigma_2 is not None else s1)
    a = _as_float(alpha)
    b = _as_float(power)
    denom = d0 - eps

    match var_mode:
        case MeanVarMode.EQUAL_VAR:
            v1 = v2 = s1 * s1
            k = _choose_kappa(float(kappa) if kappa is not None else None, default_kappa=1.0)
        case MeanVarMode.UNEQUAL_VAR:
            v1, v2 = s1 * s1, s2 * s2
            k_opt = s1 / s2
            k = _choose_kappa(float(kappa) if kappa is not None else None, default_kappa=k_opt)

    return _wald_two_group_sizes(
        v1=v1, v2=v2, kappa=k, denom=denom, two_sides=two_sides, alpha=a, power=b, delta=d0
    )


def _wald_sample_size_two_means_map(
    *,
    two_sides: bool,
    epsilon: DomainScalarMap,
    delta: DomainScalarMap,
    sigma_1: DomainScalarMap,
    sigma_2: DomainScalarMap | None,
    kappa: DomainScalarMap,
    alpha: DomainScalarMap | Number,
    power: DomainScalarMap | Number,
    var_mode: MeanVarMode = MeanVarMode.EQUAL_VAR,
) -> dict[Category, tuple[Number, Number]]:
    keys = _ensure_same_keys(
        epsilon,
        delta,
        sigma_1,
        kappa,
        sigma_2 if sigma_2 is not None else None,
    )
    out: dict[Category, tuple[Number, Number]] = {}
    for k in keys:
        s2 = sigma_2[k] if isinstance(sigma_2, dict) else None
        n1k, n2k = _wald_sample_size_two_means_scalar(
            two_sides=two_sides,
            epsilon=epsilon[k],
            delta=delta[k],
            sigma_1=sigma_1[k],
            sigma_2=s2,
            kappa=kappa[k],
            alpha=_lookup(alpha, k),
            power=_lookup(power, k),
            var_mode=var_mode,
        )
        out[k] = (n1k, n2k)
    return out


## Two PROPORTIONS


def _wald_sample_size_two_props(
    *,
    two_sides: bool,
    epsilon: Number | DomainScalarMap,
    prop_1: Number | DomainScalarMap,
    prop_2: Number | DomainScalarMap,
    delta: Number | DomainScalarMap,
    kappa: Number | DomainScalarMap,
    alpha: Number | DomainScalarMap,
    power: Number | DomainScalarMap,
    var_mode: PropVarMode = PropVarMode.ALT_PROPS,
) -> tuple[Number, Number] | dict[Category, tuple[Number, Number]]:
    if (
        _is_number(epsilon)
        and _is_number(delta)
        and _is_number(prop_1)
        and _is_number(prop_2)
        and (_is_number(kappa) or kappa is None)
        and _is_number(alpha)
        and _is_number(power)
    ):
        return _wald_sample_size_two_props_scalar(
            two_sides=two_sides,
            epsilon=cast(Number, epsilon),
            p1=cast(Number, prop_1),
            p2=cast(Number, prop_2),
            delta=cast(Number, delta),
            kappa=cast(Number, kappa) if kappa is not None else None,
            alpha=cast(Number, alpha),
            power=cast(Number, power),
            var_mode=var_mode,
        )
    elif (
        _is_map(epsilon)
        and _is_map(delta)
        and _is_map(prop_1)
        and _is_map(prop_2)
        and _is_map(kappa)
        and _is_map(alpha)
        and _is_map(power)
    ):
        return _wald_sample_size_two_props_map(
            two_sides=two_sides,
            epsilon=cast(DomainScalarMap, epsilon),
            p1=cast(DomainScalarMap, prop_1),
            p2=cast(DomainScalarMap, prop_2),
            delta=cast(DomainScalarMap, delta),
            kappa=cast(DomainScalarMap, kappa),
            alpha=alpha,
            power=power,
            var_mode=var_mode,
        )
    else:
        raise ValueError("Invalid input types")


def _wald_sample_size_two_props_scalar(
    *,
    two_sides: bool,
    epsilon: Number,
    p1: Number,
    p2: Number,
    delta: Number,
    alpha: Number,
    power: Number,
    kappa: Number | None = None,
    var_mode: PropVarMode = PropVarMode.ALT_PROPS,
) -> tuple[Number, Number]:
    d0 = _as_float(delta)
    eps = abs(_as_float(epsilon))
    pa = _prob_clip(_as_float(p1))
    pb = _prob_clip(_as_float(p2))
    a = _as_float(alpha)
    b = _as_float(power)
    denom = d0 - eps

    match var_mode:
        case PropVarMode.POOLED_PROP:
            k_tmp = _choose_kappa(float(kappa) if kappa is not None else None, default_kappa=1.0)
            ppool = (k_tmp * pa + pb) / (k_tmp + 1.0)
            v1 = v2 = ppool * (1.0 - ppool)
            k = k_tmp
        case PropVarMode.ALT_PROPS:
            v1 = pa * (1.0 - pa)
            v2 = pb * (1.0 - pb)
            k_opt = math.sqrt(v1 / v2) if v2 > 0 else 1.0
            k = _choose_kappa(float(kappa) if kappa is not None else None, default_kappa=k_opt)
        case PropVarMode.MAX_VAR:
            v1 = v2 = 0.25
            k = _choose_kappa(float(kappa) if kappa is not None else None, default_kappa=1.0)

    return _wald_two_group_sizes(
        v1=v1, v2=v2, kappa=k, denom=denom, two_sides=two_sides, alpha=a, power=b, delta=d0
    )


def _wald_sample_size_two_props_map(
    *,
    two_sides: bool,
    epsilon: DomainScalarMap,
    p1: DomainScalarMap,
    p2: DomainScalarMap,
    delta: DomainScalarMap,
    kappa: DomainScalarMap,
    alpha: DomainScalarMap | Number,
    power: DomainScalarMap | Number,
    var_mode: PropVarMode = PropVarMode.ALT_PROPS,
) -> dict[Category, tuple[Number, Number]]:
    keys = _ensure_same_keys(
        epsilon,
        p1,
        p2,
        delta,
        kappa,
        alpha if isinstance(alpha, dict) else None,
        power if isinstance(power, dict) else None,
    )
    out: dict[Category, tuple[Number, Number]] = {}
    for k in keys:
        n1k, n2k = _wald_sample_size_two_props_scalar(
            two_sides=two_sides,
            epsilon=epsilon[k],
            p1=p1[k],
            p2=p2[k],
            delta=delta[k],
            kappa=kappa[k],
            alpha=_lookup(alpha, k),
            power=_lookup(power, k),
            var_mode=var_mode,
        )
        out[k] = (n1k, n2k)
    return out
