from __future__ import annotations

import math

from typing import Literal, cast, overload

import numpy as np

from scipy.stats import norm as normal

from svy.core.types import Array, Category, DomainScalarMap, FloatArray, Number


_EPS = 1e-12  # numeric guard for sqrt/arcsin


# =========================
# HELPERS
# =========================


def _zcrit(alpha: float, two_sides: bool) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    return float(normal.isf(alpha / 2.0 if two_sides else alpha))


def _zcrit_from_type(
    alpha: float | Array,
    testing_type: Literal["two-sided", "less", "greater"],
) -> float | FloatArray:
    """Return z critical value(s) as float (scalar alpha) or FloatArray (array alpha)."""
    two_sided = testing_type == "two-sided"

    if isinstance(alpha, (int, float)):
        a = float(alpha)
        return float(normal.isf(a / 2.0 if two_sided else a))

    arr = np.asarray(alpha, dtype=np.float64)
    z = normal.isf(arr / 2.0 if two_sided else arr)
    return cast(float | FloatArray, z)


def _split_total_to_groups_array(
    total: Array | Number, ratio: Array | Number | None
) -> tuple[FloatArray, FloatArray]:
    """Return (n_a, n_b) from total and ratio=r=n_b/n_a; equal allocation if ratio is None."""
    t = np.asarray(total, dtype=np.float64)
    if ratio is None:
        n_a = t / 2.0
        n_b = t - n_a
    else:
        r = np.asarray(ratio, dtype=np.float64)
        # n_a = N/(1+r), n_b = N - n_a
        n_a = t / (1.0 + r)
        n_b = t - n_a
    return n_a, n_b


def _se_two_props_array(pa: Array, pb: Array, na: Array, nb: Array) -> FloatArray:
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    na = np.asarray(na, dtype=np.float64)
    nb = np.asarray(nb, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(pa * (1.0 - pa) / na + pb * (1.0 - pb) / nb)
    return cast(FloatArray, se)


def _power_from_lam_array(
    lam: Array, zc: float | Array, testing_type: Literal["two-sided", "less", "greater"]
) -> FloatArray:
    lam = np.asarray(lam, dtype=np.float64)
    zc_arr = np.asarray(zc, dtype=np.float64)
    if testing_type == "two-sided":
        power = normal.cdf(lam - zc_arr) + normal.cdf(-lam - zc_arr)
    elif testing_type == "greater":
        power = 1.0 - normal.cdf(zc_arr - lam)
    else:  # "less"
        power = normal.cdf(-zc_arr - lam)
    power = np.clip(power, 0.0, 1.0).astype(np.float64, copy=False)
    return cast(FloatArray, power)


def _clamp01(v: float) -> float:
    """Clamp a float to [0, 1]."""
    return float(min(1.0, max(0.0, v)))


def _scalar_power(
    lam: float,
    zc: float,
    ttype: Literal["two-sided", "less", "greater"],
) -> float:
    """Compute clipped scalar power from non-centrality parameter lam and critical value zc."""
    if ttype == "two-sided":
        val = float(normal.cdf(lam - zc) + normal.cdf(-lam - zc))
    elif ttype == "greater":
        val = float(1.0 - normal.cdf(zc - lam))
    else:  # "less"
        val = float(normal.cdf(-zc - lam))
    return _clamp01(val)


# =========================
# Mean-power (array flavor)
# =========================


def calculate_power_array(
    two_sides: bool,
    delta: Array,
    sigma: Array,
    samp_size: Array,
    alpha: float,
) -> FloatArray:
    """
    Typed array variant with explicit float64 arrays and a cast on return to
    satisfy mypy/pyright (some numpy overloads are typed as Any).
    """
    zc = _zcrit(alpha, two_sides)

    d = np.asarray(delta, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    n = np.asarray(samp_size, dtype=np.float64)

    lam = d * np.sqrt(n) / s
    if two_sides:
        power = normal.cdf(lam - zc) + normal.cdf(-lam - zc)
    else:
        power = np.where(d >= 0.0, 1.0 - normal.cdf(zc - lam), normal.cdf(-zc - lam))

    power = np.clip(power, 0.0, 1.0).astype(np.float64, copy=False)
    return cast(FloatArray, power)


# =========================
# Mean-power (scalar flavor)
# =========================


def calculate_power_number(
    two_sides: bool,
    delta: Number,
    sigma: Number,
    samp_size: Number,
    alpha: float,
) -> float:
    zc = _zcrit(alpha, two_sides)
    if samp_size <= 0 or sigma <= 0:
        return float("nan")
    lam = float(delta) * math.sqrt(float(samp_size)) / float(sigma)
    ttype: Literal["two-sided", "less", "greater"] = "two-sided" if two_sides else "greater"
    return _scalar_power(lam, zc, ttype)


# =========================
# Mean-power (map flavor)
# =========================


def calculate_power_map(
    two_sides: bool,
    delta: DomainScalarMap,
    sigma: DomainScalarMap,
    samp_size: DomainScalarMap,
    alpha: float,
) -> dict[Category, float]:
    zc = _zcrit(alpha, two_sides)
    kd, ks, kn = set(delta.keys()), set(sigma.keys()), set(samp_size.keys())
    if not (kd == ks == kn):
        raise KeyError("delta, sigma, and samp_size must have identical keys")
    out: dict[Category, float] = {}
    ttype: Literal["two-sided", "less", "greater"] = "two-sided" if two_sides else "greater"
    for k in kd:
        n = float(samp_size[k])
        s = float(sigma[k])
        if n <= 0 or s <= 0:
            out[k] = float("nan")
            continue
        lam = float(delta[k]) * math.sqrt(n) / s
        out[k] = _scalar_power(lam, zc, ttype)
    return out


# =========================
# Mean-power: dispatcher
# =========================


@overload
def calculate_power(
    two_sides: bool, delta: Number, sigma: Number, samp_size: Number, alpha: float
) -> float: ...
@overload
def calculate_power(
    two_sides: bool, delta: Array, sigma: Array, samp_size: Array, alpha: float
) -> FloatArray: ...
@overload
def calculate_power(
    two_sides: bool,
    delta: DomainScalarMap,
    sigma: DomainScalarMap,
    samp_size: DomainScalarMap,
    alpha: float,
) -> dict[Category, float]: ...


def calculate_power(
    two_sides: bool,
    delta: Number | Array | DomainScalarMap,
    sigma: Number | Array | DomainScalarMap,
    samp_size: Number | Array | DomainScalarMap,
    alpha: float,
) -> float | FloatArray | dict[Category, float]:
    if isinstance(delta, dict) and isinstance(sigma, dict) and isinstance(samp_size, dict):
        return calculate_power_map(two_sides, delta, sigma, samp_size, alpha)  # type: ignore[arg-type]
    if (
        isinstance(delta, np.ndarray)
        and isinstance(sigma, np.ndarray)
        and isinstance(samp_size, np.ndarray)
    ):
        return calculate_power_array(two_sides, delta, sigma, samp_size, alpha)
    if (
        isinstance(delta, (int, float))
        and isinstance(sigma, (int, float))
        and isinstance(samp_size, (int, float))
    ):
        return calculate_power_number(
            two_sides, float(delta), float(sigma), float(samp_size), alpha
        )
    raise TypeError(
        "delta, sigma, samp_size must all be scalars, all ndarrays, or all dicts with identical keys."
    )


# =========================
# POWER FOR ONE PROPORTION
# =========================


def power_for_one_proportion_number(
    prop_0: Number,
    prop_1: Number,
    samp_size: Number,
    *,
    arcsin: bool = True,
    testing_type: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
) -> float:
    p0 = float(prop_0)
    p1 = float(prop_1)
    n = float(samp_size)
    if not (0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0) or n <= 0:
        return float("nan")

    zc = float(_zcrit_from_type(alpha, testing_type))
    if arcsin:
        # Cohen's h
        h = 2.0 * math.asin(math.sqrt(max(_EPS, min(1.0 - _EPS, p1)))) - 2.0 * math.asin(
            math.sqrt(max(_EPS, min(1.0 - _EPS, p0)))
        )
        z = h * math.sqrt(n)
    else:
        denom = math.sqrt(max(_EPS, p1 * (1.0 - p1)) / n)
        z = (p1 - p0) / denom

    if testing_type == "two-sided":
        return _scalar_power(z, zc, "two-sided")
    elif testing_type == "greater":
        return _scalar_power(z, zc, "greater")
    else:  # "less"
        return _scalar_power(z, zc, "less")


def power_for_one_proportion_array(
    prop_0: Array,
    prop_1: Array,
    samp_size: Array,
    *,
    arcsin: bool = True,
    testing_type: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float | Array = 0.05,
) -> FloatArray:
    p0 = np.asarray(prop_0, dtype=np.float64)
    p1 = np.asarray(prop_1, dtype=np.float64)
    n = np.asarray(samp_size, dtype=np.float64)

    p0c = np.clip(p0, _EPS, 1.0 - _EPS)
    p1c = np.clip(p1, _EPS, 1.0 - _EPS)
    n = np.maximum(n, _EPS)

    zc = _zcrit_from_type(alpha, testing_type)

    if arcsin:
        z = (2.0 * np.arcsin(np.sqrt(p1c)) - 2.0 * np.arcsin(np.sqrt(p0c))) * np.sqrt(n)
    else:
        z = (p1c - p0c) / np.sqrt((p1c * (1.0 - p1c)) / n)

    if testing_type == "two-sided":
        power = normal.cdf(z - zc) + normal.cdf(-z - zc)
    elif testing_type == "greater":
        power = 1.0 - normal.cdf(zc - z)
    else:  # "less"
        power = normal.cdf(-zc - z)

    power = np.clip(power, 0.0, 1.0).astype(np.float64, copy=False)
    return cast(FloatArray, power)


def power_for_one_proportion_map(
    prop_0: DomainScalarMap,
    prop_1: DomainScalarMap,
    samp_size: DomainScalarMap,
    *,
    arcsin: bool = True,
    testing_type: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
) -> dict[Category, float]:
    k0, k1, kn = set(prop_0.keys()), set(prop_1.keys()), set(samp_size.keys())
    if not (k0 == k1 == kn):
        raise KeyError("prop_0, prop_1, and samp_size must share identical keys")

    zc = float(_zcrit_from_type(alpha, testing_type))
    out: dict[Category, float] = {}
    for k in k0:
        p0 = float(prop_0[k])
        p1 = float(prop_1[k])
        n = float(samp_size[k])
        if not (0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0) or n <= 0:
            out[k] = float("nan")
            continue

        if arcsin:
            h = 2.0 * math.asin(math.sqrt(max(_EPS, min(1.0 - _EPS, p1)))) - 2.0 * math.asin(
                math.sqrt(max(_EPS, min(1.0 - _EPS, p0)))
            )
            z = h * math.sqrt(n)
        else:
            denom = math.sqrt(max(_EPS, p1 * (1.0 - p1)) / n)
            z = (p1 - p0) / denom

        if testing_type == "two-sided":
            out[k] = _scalar_power(z, zc, "two-sided")
        elif testing_type == "greater":
            out[k] = _scalar_power(z, zc, "greater")
        else:
            out[k] = _scalar_power(z, zc, "less")
    return out


@overload
def power_for_one_proportion(
    prop_0: Number,
    prop_1: Number,
    samp_size: Number,
    *,
    arcsin: bool = ...,
    testing_type: Literal["two-sided", "less", "greater"] = ...,
    alpha: float = ...,
) -> float: ...
@overload
def power_for_one_proportion(
    prop_0: Array,
    prop_1: Array,
    samp_size: Array,
    *,
    arcsin: bool = ...,
    testing_type: Literal["two-sided", "less", "greater"] = ...,
    alpha: float | Array = ...,
) -> FloatArray: ...
@overload
def power_for_one_proportion(
    prop_0: DomainScalarMap,
    prop_1: DomainScalarMap,
    samp_size: DomainScalarMap,
    *,
    arcsin: bool = ...,
    testing_type: Literal["two-sided", "less", "greater"] = ...,
    alpha: float = ...,
) -> dict[Category, float]: ...


def power_for_one_proportion(
    prop_0: Number | Array | DomainScalarMap,
    prop_1: Number | Array | DomainScalarMap,
    samp_size: Number | Array | DomainScalarMap,
    *,
    arcsin: bool = True,
    testing_type: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float | Array = 0.05,
) -> float | FloatArray | dict[Category, float]:
    if isinstance(prop_0, dict) and isinstance(prop_1, dict) and isinstance(samp_size, dict):
        from typing import cast as _cast

        from svy.core.types import DomainScalarMap as _DSM

        return power_for_one_proportion_map(
            _cast(_DSM, prop_0),
            _cast(_DSM, prop_1),
            _cast(_DSM, samp_size),
            arcsin=arcsin,
            testing_type=testing_type,
            alpha=float(alpha),
        )
    if (
        isinstance(prop_0, np.ndarray)
        and isinstance(prop_1, np.ndarray)
        and isinstance(samp_size, np.ndarray)
    ):
        return power_for_one_proportion_array(
            prop_0, prop_1, samp_size, arcsin=arcsin, testing_type=testing_type, alpha=alpha
        )
    if (
        isinstance(prop_0, (int, float))
        and isinstance(prop_1, (int, float))
        and isinstance(samp_size, (int, float))
    ):
        return power_for_one_proportion_number(
            float(prop_0),
            float(prop_1),
            float(samp_size),
            arcsin=arcsin,
            testing_type=testing_type,
            alpha=float(alpha),
        )
    raise TypeError(
        "prop_0, prop_1, samp_size must all be scalars, all ndarrays, or all dicts with identical keys."
    )


# =========================
# POWER FOR TWO PROPORTIONS
# =========================


def power_for_two_proportions(
    prop_a: DomainScalarMap | Number | Array,
    prop_b: DomainScalarMap | Number | Array,
    samp_size: DomainScalarMap | Number | Array | None = None,  # total N (optional)
    ratio: DomainScalarMap | Number | Array | None = None,  # r = n_b / n_a
    samp_size_a: DomainScalarMap | Number | Array | None = None,  # per-group overrides
    samp_size_b: DomainScalarMap | Number | Array | None = None,
    testing_type: str = "two-sided",
    alpha: Number | Array = 0.05,
) -> DomainScalarMap | Number | Array:
    """
    Power for testing H0: p_a - p_b = 0 (z-approx). Independent samples.
    If samp_size_a/b are provided, they are used. Otherwise samp_size (+ ratio) is split.
    """
    ttype_str = testing_type.lower()
    if ttype_str not in ("two-sided", "less", "greater"):
        raise AssertionError("testing_type must be 'two-sided', 'less', or 'greater'.")
    ttype: Literal["two-sided", "less", "greater"] = cast(
        Literal["two-sided", "less", "greater"], ttype_str
    )

    # ----------------
    # Dict / map flavor
    # ----------------
    if isinstance(prop_a, dict) and isinstance(prop_b, dict):
        if not isinstance(alpha, (int, float)):
            raise TypeError("For dict inputs, alpha must be a scalar Number.")
        keys_a, keys_b = set(prop_a.keys()), set(prop_b.keys())
        if keys_a != keys_b:
            raise KeyError("prop_a and prop_b must have identical keys")

        if isinstance(samp_size_a, dict) and isinstance(samp_size_b, dict):
            if set(samp_size_a.keys()) != keys_a or set(samp_size_b.keys()) != keys_a:
                raise KeyError("samp_size_a/samp_size_b keys must match prop_a/prop_b keys")
            na_map = {k: float(samp_size_a[k]) for k in keys_a}  # type: ignore[index]
            nb_map = {k: float(samp_size_b[k]) for k in keys_a}  # type: ignore[index]
        else:
            if not isinstance(samp_size, dict):
                raise TypeError(
                    "For dict inputs, provide either samp_size_a & samp_size_b as dicts, "
                    "or samp_size as dict (optionally with ratio as dict/number)."
                )
            if ratio is None:
                na_map = {k: float(samp_size[k]) / 2.0 for k in keys_a}  # type: ignore[index]
                nb_map = {k: float(samp_size[k]) - na_map[k] for k in keys_a}  # type: ignore[index]
            else:
                if isinstance(ratio, dict):
                    if set(ratio.keys()) != keys_a:
                        raise KeyError("ratio dict keys must match prop_a/prop_b keys")
                    na_map = {k: float(samp_size[k]) / (1.0 + float(ratio[k])) for k in keys_a}  # type: ignore[index]
                    nb_map = {k: float(samp_size[k]) - na_map[k] for k in keys_a}  # type: ignore[index]
                elif isinstance(ratio, (int, float)):
                    r = float(ratio)
                    na_map = {k: float(samp_size[k]) / (1.0 + r) for k in keys_a}  # type: ignore[index]
                    nb_map = {k: float(samp_size[k]) - na_map[k] for k in keys_a}  # type: ignore[index]
                else:
                    raise TypeError("ratio must be dict or scalar when using dict inputs")

        zc_dict = float(_zcrit_from_type(float(alpha), ttype))
        out: dict[Category, float] = {}
        for k in prop_a:
            pa = float(prop_a[k])  # type: ignore[arg-type]
            pb = float(prop_b[k])  # type: ignore[arg-type]
            na_f = float(na_map[k])
            nb_f = float(nb_map[k])
            if na_f <= 0 or nb_f <= 0:
                out[k] = float("nan")
                continue
            se_f = math.sqrt(pa * (1 - pa) / na_f + pb * (1 - pb) / nb_f)
            if se_f <= 0.0 or not math.isfinite(se_f):
                out[k] = float("nan")
                continue
            lam_f = (pa - pb) / se_f
            out[k] = _scalar_power(lam_f, zc_dict, ttype)
        return out

    # --------------
    # Array / vector
    # --------------
    if isinstance(prop_a, np.ndarray) and isinstance(prop_b, np.ndarray):
        pa_arr = np.asarray(prop_a, dtype=np.float64)
        pb_arr = np.asarray(prop_b, dtype=np.float64)

        if isinstance(samp_size_a, np.ndarray) and isinstance(samp_size_b, np.ndarray):
            na_arr = np.asarray(samp_size_a, dtype=np.float64)
            nb_arr = np.asarray(samp_size_b, dtype=np.float64)
        else:
            if samp_size is None:
                raise TypeError(
                    "Provide samp_size (total) or both samp_size_a and samp_size_b for array inputs."
                )
            if not isinstance(samp_size, (np.ndarray, int, float)):
                raise TypeError("samp_size must be an ndarray or a scalar for array inputs.")
            if ratio is not None and not isinstance(ratio, (np.ndarray, int, float)):
                raise TypeError("ratio must be ndarray, scalar, or None for array inputs.")
            na_arr, nb_arr = _split_total_to_groups_array(samp_size, ratio)

        se_arr = _se_two_props_array(pa_arr, pb_arr, na_arr, nb_arr)
        zc_arr = _zcrit_from_type(alpha, ttype)  # float or array
        lam_arr = np.divide(
            pa_arr - pb_arr, se_arr, out=np.full_like(se_arr, np.nan), where=se_arr > 0.0
        )
        power_arr = _power_from_lam_array(lam_arr, zc_arr, ttype)
        return power_arr  # FloatArray

    # -------
    # Scalar
    # -------
    if isinstance(prop_a, (int, float)) and isinstance(prop_b, (int, float)):
        pa_s = float(prop_a)
        pb_s = float(prop_b)
        if isinstance(samp_size_a, (int, float)) and isinstance(samp_size_b, (int, float)):
            na_s = float(samp_size_a)
            nb_s = float(samp_size_b)
        else:
            if samp_size is None:
                raise TypeError(
                    "Provide samp_size (total) or both samp_size_a and samp_size_b for scalar inputs."
                )
            if not isinstance(samp_size, (int, float)):
                raise TypeError(
                    "For scalar props, samp_size must be a scalar when per-group sizes aren't provided."
                )
            if ratio is None:
                na_s = float(samp_size) / 2.0
                nb_s = float(samp_size) - na_s
            elif isinstance(ratio, (int, float)):
                r = float(ratio)
                na_s = float(samp_size) / (1.0 + r)
                nb_s = float(samp_size) - na_s
            else:
                raise TypeError("For scalar props, ratio must be a scalar or None.")

        if na_s <= 0 or nb_s <= 0:
            return float("nan")
        se_s = math.sqrt(pa_s * (1 - pa_s) / na_s + pb_s * (1 - pb_s) / nb_s)
        if se_s <= 0.0 or not math.isfinite(se_s):
            return float("nan")
        zc_s = float(_zcrit_from_type(float(alpha), ttype))
        lam_s = (pa_s - pb_s) / se_s
        return _scalar_power(lam_s, zc_s, ttype)

    raise TypeError(
        "prop_a/prop_b must both be scalars, both ndarrays, or both dicts (with compatible size inputs)."
    )


# =========================
# POWER FOR ONE MEAN
# =========================


@overload
def power_for_one_mean(
    mean_0: Number,
    mean_1: Number,
    sigma: Number,
    samp_size: Number,
    testing_type: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
) -> float: ...
@overload
def power_for_one_mean(
    mean_0: Array,
    mean_1: Array,
    sigma: Array,
    samp_size: Array,
    testing_type: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float | Array = 0.05,
) -> FloatArray: ...
@overload
def power_for_one_mean(
    mean_0: DomainScalarMap,
    mean_1: DomainScalarMap,
    sigma: DomainScalarMap,
    samp_size: DomainScalarMap,
    testing_type: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
) -> dict[Category, float]: ...


def power_for_one_mean(
    mean_0: DomainScalarMap | Number | Array,
    mean_1: DomainScalarMap | Number | Array,
    sigma: DomainScalarMap | Number | Array,
    samp_size: DomainScalarMap | Number | Array,
    testing_type: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float | Array = 0.05,
) -> dict[Category, float] | float | FloatArray:
    """
    Power for testing H0: mu = mean_0 when true mean is mean_1 (known sigma),
    using z-approximation:
        λ = (mean_1 - mean_0) * sqrt(n) / sigma
        two-sided:  Φ(λ − zα/2) + Φ(−λ − zα/2)
        greater:    1 − Φ(zα − λ)
        less:       Φ(−zα − λ)
    """
    ttype = testing_type

    # --- dict/map flavor ---
    if (
        isinstance(mean_0, dict)
        and isinstance(mean_1, dict)
        and isinstance(sigma, dict)
        and isinstance(samp_size, dict)
    ):
        k0, k1, ks, kn = (
            set(mean_0.keys()),
            set(mean_1.keys()),
            set(sigma.keys()),
            set(samp_size.keys()),
        )
        if not (k0 == k1 == ks == kn):
            raise KeyError("mean_0, mean_1, sigma, and samp_size must have identical keys")

        zc_map = float(_zcrit_from_type(float(alpha), ttype))
        out: dict[Category, float] = {}
        for k in mean_0:
            n_s = float(samp_size[k])  # type: ignore[index]
            s_s = float(sigma[k])  # type: ignore[index]
            if n_s <= 0 or s_s <= 0:
                out[k] = float("nan")
                continue
            lam = (float(mean_1[k]) - float(mean_0[k])) * math.sqrt(n_s) / s_s  # type: ignore[index]
            out[k] = _scalar_power(lam, zc_map, ttype)
        return out

    # --- array / vector flavor ---
    if (
        isinstance(mean_0, np.ndarray)
        and isinstance(mean_1, np.ndarray)
        and isinstance(sigma, np.ndarray)
        and isinstance(samp_size, np.ndarray)
    ):
        m0 = np.asarray(mean_0, dtype=np.float64)
        m1 = np.asarray(mean_1, dtype=np.float64)
        s_arr = np.asarray(sigma, dtype=np.float64)
        n_arr = np.asarray(samp_size, dtype=np.float64)

        lam_arr = (m1 - m0) * np.sqrt(n_arr) / s_arr
        zc_arr = _zcrit_from_type(alpha, ttype)  # float or array
        power_arr = _power_from_lam_array(lam_arr, zc_arr, ttype)
        return power_arr  # FloatArray

    # --- scalar flavor ---
    if (
        isinstance(mean_0, (int, float))
        and isinstance(mean_1, (int, float))
        and isinstance(sigma, (int, float))
        and isinstance(samp_size, (int, float))
    ):
        n_s = float(samp_size)
        s_s = float(sigma)
        if n_s <= 0 or s_s <= 0:
            return float("nan")
        lam_s = (float(mean_1) - float(mean_0)) * math.sqrt(n_s) / s_s
        zc_s = float(_zcrit_from_type(float(alpha), ttype))
        return _scalar_power(lam_s, zc_s, ttype)

    raise TypeError(
        "mean_0/mean_1/sigma/samp_size must all be scalars, all ndarrays, or all dicts with identical keys."
    )
