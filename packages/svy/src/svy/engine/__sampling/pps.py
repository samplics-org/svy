# src/svy/engine/sampling/pps.py
from __future__ import annotations

import warnings

from typing import Mapping

import numpy as np
import numpy.typing as npt

from svy.core.enumerations import PPSMethod
from svy.core.types import DT, FloatArray, IntArray
from svy.utils.random_state import RandomState, resolve_random_state, spawn_child_rngs


# ---------------------------------------------------------------------------
# Return type: (selected_frame_indices, hits, probs, is_certainty)
# ---------------------------------------------------------------------------
# All engine functions return a 4-tuple:
#   selected  : frame values for selected units          shape (k,)
#   hits      : integer hit counts                       shape (k,)
#   probs     : first-order inclusion probabilities      shape (k,)
#   certainty : boolean, True if selected as certainty   shape (k,)
# ---------------------------------------------------------------------------

_PPS_RETURN = tuple[
    "npt.NDArray[np.generic]",
    "IntArray",
    "FloatArray",
    "npt.NDArray[np.bool_]",
]


# ---------------------------------------------------------------------------
# Certainty extraction helper (shared by SYS, Brewer, RS)
# ---------------------------------------------------------------------------


def _extract_certainty(
    p0: npt.NDArray[np.float64],
    n: int,
    threshold: float,
) -> tuple[npt.NDArray[np.bool_], int]:
    """
    Iteratively extract certainty units.

    A unit is certain when its effective inclusion probability
    pi_eff_i = n_remaining * p_i / sum_p_remaining >= threshold.

    Iteration is necessary because removing certainty units changes the
    total MOS, which may push previously non-certain units over the threshold.

    Parameters
    ----------
    p0        : original base probabilities (MOS_i / sum(MOS)), shape (N,)
    n         : requested sample size
    threshold : certainty threshold in (0, 1]

    Returns
    -------
    cert_mask : boolean mask of certainty units, shape (N,)
    n_rem     : adjusted sample size after removing all certainty units
    """
    N = p0.size
    cert_mask = np.zeros(N, dtype=bool)
    remaining = np.ones(N, dtype=bool)
    n_rem = int(n)

    while n_rem > 0:
        rem_idx = np.flatnonzero(remaining)
        if rem_idx.size == 0:
            break
        p_rem = p0[rem_idx]
        total_rem = p_rem.sum()
        if total_rem <= 0.0:
            break
        pi_eff = n_rem * p_rem / total_rem
        new_cert_local = pi_eff >= threshold - 1e-12
        if not new_cert_local.any():
            break
        new_cert_global = rem_idx[new_cert_local]
        cert_mask[new_cert_global] = True
        remaining[new_cert_global] = False
        n_rem -= int(new_cert_local.sum())

    return cert_mask, n_rem


# ---------------------------------------------------------------------------
# Shared validation and empty result helpers
# ---------------------------------------------------------------------------


def _validate_pps_inputs(
    arr: npt.NDArray,
    mos_arr: npt.NDArray[np.float64],
    n: int,
) -> None:
    if arr.ndim != 1:
        raise ValueError(f"'frame' must be 1-D; got shape {arr.shape}")
    if mos_arr.ndim != 1:
        raise ValueError(f"'mos' must be 1-D; got shape {mos_arr.shape}")
    if arr.size != mos_arr.size:
        raise ValueError(f"Length mismatch: frame={arr.size} vs mos={mos_arr.size}")
    if np.any(mos_arr < 0):
        raise ValueError("All MOS values must be non-negative")
    if n < 0:
        raise ValueError("'n' must be non-negative")


def _empty_result(dtype: np.dtype) -> _PPS_RETURN:
    return (
        np.asarray([], dtype=dtype),
        np.asarray([], dtype=np.int64),
        np.asarray([], dtype=np.float64),
        np.asarray([], dtype=bool),
    )


def _select_all(arr: npt.NDArray) -> _PPS_RETURN:
    """Select all units with certainty (used when n >= N)."""
    N = arr.size
    return (
        arr.copy(),
        np.ones(N, dtype=np.int64),
        np.ones(N, dtype=np.float64),
        np.ones(N, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _select_pps(
    *,
    frame: npt.NDArray[DT] | npt.ArrayLike,
    n: int | Mapping[DT, int],
    mos: npt.NDArray[np.floating] | npt.ArrayLike,
    stratum: npt.NDArray[DT] | npt.ArrayLike | None,
    method: PPSMethod,
    certainty_threshold: float = 1.0,
    rstate: RandomState = None,
) -> _PPS_RETURN:
    """
    Select a PPS sample from a frame, optionally stratified.

    Parameters
    ----------
    frame               : population frame identifiers
    n                   : sample size (scalar or per-stratum mapping)
    mos                 : measure of size values
    stratum             : stratum labels (None for unstratified)
    method              : PPSMethod enum
    certainty_threshold : units with pi_i >= threshold selected with certainty;
                          must be in (0, 1], default 1.0
    rstate              : random state

    Returns
    -------
    selected  : selected frame values
    hits      : hit counts per selected unit
    probs     : first-order inclusion probabilities
    certainty : True for units selected as certainty units
    """
    if not (0.0 < certainty_threshold <= 1.0):
        raise ValueError(f"certainty_threshold must be in (0, 1], got {certainty_threshold}")

    arr = np.asarray(frame)
    mos_arr = np.asarray(mos, dtype=np.float64)

    if stratum is None:
        assert isinstance(n, int)
        return _select_pps_unstratified(
            method=method,
            frame=arr,
            n=n,
            mos=mos_arr,
            certainty_threshold=certainty_threshold,
            rstate=rstate,
        )
    else:
        stratum_arr = np.asarray(stratum)
        return _select_pps_stratified(
            method=method,
            frame=arr,
            n=n,
            mos=mos_arr,
            stratum=stratum_arr,
            certainty_threshold=certainty_threshold,
            rstate=rstate,
        )


# ---------------------------------------------------------------------------
# Stratified dispatcher
# ---------------------------------------------------------------------------


def _select_pps_stratified(
    *,
    frame: npt.NDArray[DT] | npt.ArrayLike,
    n: int | Mapping[DT, int],
    mos: npt.NDArray[np.floating] | npt.ArrayLike,
    stratum: npt.NDArray[DT] | npt.ArrayLike,
    method: PPSMethod,
    certainty_threshold: float = 1.0,
    rstate: RandomState = None,
) -> _PPS_RETURN:
    arr = np.asarray(frame)
    mos_arr = np.asarray(mos, dtype=np.float64)
    strat = np.asarray(stratum)
    strata_vals = np.unique(strat)

    if isinstance(n, Mapping):
        size_map: Mapping[DT, int] = n  # type: ignore[assignment]
    else:
        if int(n) < 0:
            raise ValueError("'n' must be non-negative")
        size_map = {s: int(n) for s in strata_vals}

    rng_parent = resolve_random_state(rstate)
    child_rng = spawn_child_rngs(rng_parent, strata_vals)

    sel_chunks: list[npt.NDArray] = []
    hits_chunks: list[IntArray] = []
    prob_chunks: list[FloatArray] = []
    cert_chunks: list[npt.NDArray[np.bool_]] = []

    for s in strata_vals:
        m = strat == s
        n_s = int(size_map[s])
        if n_s < 0:
            raise ValueError(f"Sample size for stratum {s!r} must be non-negative")
        sel_s, hits_s, prob_s, cert_s = _select_pps_unstratified(
            method=method,
            frame=arr[m],
            n=n_s,
            mos=mos_arr[m],
            certainty_threshold=certainty_threshold,
            rstate=child_rng[s],
        )
        sel_chunks.append(sel_s.astype(arr.dtype, copy=False))
        hits_chunks.append(hits_s.astype(np.int64, copy=False))
        prob_chunks.append(prob_s.astype(np.float64, copy=False))
        cert_chunks.append(cert_s)

    if not sel_chunks:
        return _empty_result(arr.dtype)

    return (
        np.concatenate(sel_chunks, axis=0),
        np.concatenate(hits_chunks, axis=0),
        np.concatenate(prob_chunks, axis=0),
        np.concatenate(cert_chunks, axis=0),
    )


# ---------------------------------------------------------------------------
# Unstratified dispatcher
# ---------------------------------------------------------------------------


def _select_pps_unstratified(
    *,
    method: PPSMethod,
    frame: npt.NDArray[DT] | npt.ArrayLike,
    n: int,
    mos: npt.NDArray[np.floating] | npt.ArrayLike,
    certainty_threshold: float = 1.0,
    rstate: RandomState = None,
) -> _PPS_RETURN:
    dispatch = {
        PPSMethod.SYS: _select_pps_sys,
        PPSMethod.BREWER: _select_pps_brewer,
        PPSMethod.WR: _select_pps_wr,
        PPSMethod.RS: _select_pps_rs,
        PPSMethod.MURPHY: _select_pps_murphy,
    }
    impl = dispatch.get(method)
    if impl is None:
        raise ValueError(f"Unknown PPS method: {method!r}")

    return impl(
        frame=np.asarray(frame),
        n=n,
        mos=np.asarray(mos, dtype=np.float64),
        certainty_threshold=certainty_threshold,
        rstate=rstate,
    )


# ---------------------------------------------------------------------------
# PPS with replacement (WR)
# ---------------------------------------------------------------------------


def _select_pps_wr(
    *,
    frame: npt.ArrayLike,
    n: int,
    mos: npt.ArrayLike,
    certainty_threshold: float = 1.0,
    rstate: RandomState = None,
) -> _PPS_RETURN:
    """
    PPS sampling with replacement.

    Draw n times with probabilities p_i = MOS_i / sum(MOS).
    First-order inclusion probability: pi_i = 1 - (1 - p_i)^n.

    Certainty does not apply to WR since pi_i < 1 always (when N > 1).
    The certainty_threshold parameter is accepted for API consistency but ignored.
    All svy_certainty flags are False.
    """
    arr = np.asarray(frame)
    mos_arr = np.asarray(mos, dtype=np.float64)
    _validate_pps_inputs(arr, mos_arr, n)

    N = arr.size
    if N == 0 or n == 0:
        return _empty_result(arr.dtype)

    total = float(mos_arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("sum(MOS) must be > 0")

    p0 = mos_arr / total
    rng = resolve_random_state(rstate)
    picks = rng.choice(N, size=int(n), replace=True, p=p0)

    hits_full = np.bincount(picks, minlength=N).astype(np.int64)
    mask = hits_full > 0
    pi_full = (1.0 - np.power(1.0 - p0, int(n))).astype(np.float64)
    cert_full = np.zeros(N, dtype=bool)

    return arr[mask], hits_full[mask], pi_full[mask], cert_full[mask]


# ---------------------------------------------------------------------------
# PPS systematic (SYS)
# ---------------------------------------------------------------------------


def _select_pps_sys(
    *,
    frame: npt.ArrayLike,
    n: int,
    mos: npt.ArrayLike,
    certainty_threshold: float = 1.0,
    rstate: RandomState = None,
) -> _PPS_RETURN:
    """
    PPS systematic sampling without replacement.

    Algorithm:
      1. Iteratively extract certainty units (pi_i >= certainty_threshold).
      2. Run systematic grid on remaining units with adjusted n.

    First-order inclusion probabilities:
      - Certainty units: pi_i = 1.0
      - Non-certainty: pi_i = n * MOS_i / sum(MOS), exact for systematic PPS.

    Reference: Madow (1949), Cochran (1977) s.8.6.
    """
    arr = np.asarray(frame)
    mos_arr = np.asarray(mos, dtype=np.float64)
    _validate_pps_inputs(arr, mos_arr, n)

    N = arr.size
    if N == 0 or n == 0:
        return _empty_result(arr.dtype)

    total = float(mos_arr.sum())
    if total <= 0.0:
        raise ValueError("sum(MOS) must be > 0")

    if n >= N:
        return _select_all(arr)

    p0 = mos_arr / total

    # --- Iterative certainty extraction ---
    cert_mask, n_rem = _extract_certainty(p0, n, certainty_threshold)
    n_cert = int(cert_mask.sum())

    if n_cert > 0:
        warnings.warn(
            f"PPS SYS: {n_cert} certainty unit(s) detected "
            f"(threshold={certainty_threshold}). "
            "They are selected with probability 1.",
            UserWarning,
            stacklevel=4,
        )

    hits = np.zeros(N, dtype=np.int64)
    hits[cert_mask] = 1

    # --- Systematic grid on non-certainty units ---
    if n_rem > 0:
        rem_idx = np.flatnonzero(~cert_mask)
        mos_rem = mos_arr[rem_idx]
        total_rem = float(mos_rem.sum())

        if total_rem > 0 and rem_idx.size > 0:
            interval = total_rem / float(n_rem)
            rng = resolve_random_state(rstate)
            start = rng.random() * interval
            picks_cum = start + interval * np.arange(n_rem, dtype=float)

            edges = np.concatenate(([0.0], np.cumsum(mos_rem)))
            # searchsorted gives bin indices in [1, len(rem_idx)]
            bins = np.searchsorted(edges, picks_cum, side="right") - 1
            bins = np.clip(bins, 0, rem_idx.size - 1)

            selected_global = rem_idx[bins]
            # Use bincount to correctly handle duplicates (rare but possible
            # at floating point boundaries)
            hit_counts = np.bincount(selected_global, minlength=N)
            hits += hit_counts.astype(np.int64)

    mask = hits > 0
    pi_full = np.where(cert_mask, 1.0, np.minimum(1.0, n * p0)).astype(np.float64)
    cert_out = cert_mask[mask]

    return arr[mask], hits[mask], pi_full[mask], cert_out


# ---------------------------------------------------------------------------
# PPS Brewer
# ---------------------------------------------------------------------------


def _select_pps_brewer(
    *,
    frame: npt.ArrayLike,
    n: int,
    mos: npt.ArrayLike,
    certainty_threshold: float = 1.0,
    rstate: RandomState = None,
) -> _PPS_RETURN:
    """
    Brewer PPS without replacement.

    At each sequential draw with t units remaining to select, unit i is
    chosen with probability:
        q_i^(t) proportional to p_i * (1 - p_i) / (1 - t * p_i)
    where p_i = MOS_i / sum(MOS) are the ORIGINAL base probabilities
    (never renormalized between draws).

    First-order inclusion probabilities: pi_i = n * p_i (exact for Brewer).
    Certainty condition: p_i >= 1/t for the current remaining draw count t.

    Reference: Brewer (1963), Sarndal, Swensson & Wretman (1992) s.3.3.
    """
    arr = np.asarray(frame)
    mos_arr = np.asarray(mos, dtype=np.float64)
    _validate_pps_inputs(arr, mos_arr, n)

    N = arr.size
    if N == 0 or n == 0:
        return _empty_result(arr.dtype)

    if n >= N:
        return _select_all(arr)

    total = float(mos_arr.sum())
    if total <= 0.0:
        raise ValueError("sum(MOS) must be > 0")

    p0 = mos_arr / total  # original base probabilities — never renormalized

    # --- Iterative certainty extraction ---
    cert_mask, n_rem = _extract_certainty(p0, n, certainty_threshold)
    n_cert = int(cert_mask.sum())

    if n_cert > 0:
        warnings.warn(
            f"PPS Brewer: {n_cert} certainty unit(s) detected "
            f"(threshold={certainty_threshold}). "
            "They are selected with probability 1.",
            UserWarning,
            stacklevel=4,
        )

    hits = np.zeros(N, dtype=np.int64)
    hits[cert_mask] = 1

    rng = resolve_random_state(rstate)
    remaining = ~cert_mask.copy()
    to_draw = n_rem

    # --- Sequential Brewer draws on non-certainty units ---
    while to_draw > 0:
        rem_idx = np.flatnonzero(remaining)
        if rem_idx.size == 0:
            break

        p_rem = p0[rem_idx]  # original probabilities, NOT renormalized
        t = float(to_draw)

        # Brewer weight: w_i proportional to p_i * (1 - p_i) / (1 - t * p_i)
        denom = 1.0 - t * p_rem
        safe = denom > 1e-12

        if safe.all():
            w = p_rem * (1.0 - p_rem) / denom
        else:
            # Numerical guard: units with denom <= 0 should have been extracted
            # as certainty. Fall back to p_rem for those edge cases.
            w = np.where(safe, p_rem * (1.0 - p_rem) / denom, p_rem)

        s = w.sum()
        if not np.isfinite(s) or s <= 0.0:
            w = p_rem / p_rem.sum()
        else:
            w = w / s

        pick = int(rng.choice(rem_idx, p=w))
        hits[pick] = 1
        remaining[pick] = False
        to_draw -= 1

    mask = hits > 0
    # Exact first-order inclusion probability for Brewer: pi_i = n * p_i
    pi_full = np.where(cert_mask, 1.0, np.minimum(1.0, n * p0)).astype(np.float64)
    cert_out = cert_mask[mask]

    return arr[mask], hits[mask], pi_full[mask], cert_out


# ---------------------------------------------------------------------------
# PPS Murphy (n=2 only)
# ---------------------------------------------------------------------------


def _select_pps_murphy(
    *,
    frame: npt.ArrayLike,
    n: int,
    mos: npt.ArrayLike,
    certainty_threshold: float = 1.0,
    rstate: RandomState = None,
) -> _PPS_RETURN:
    """
    Murphy PPS without replacement (n=2 only).

    Exact first-order inclusion probabilities:
        pi_i = p_i * (1 + S - p_i / (1 - p_i))
    where S = sum_j p_j / (1 - p_j) and p_i = MOS_i / sum(MOS).

    No certainty handling needed: Murphy is defined only for n=2 and
    requires at least two positive-MOS units, so certainty (pi_i >= 1)
    cannot arise unless N=1. All svy_certainty flags are False.

    Reference: Murphy (1967).
    """
    if n != 2:
        raise ValueError("Murphy PPS requires n == 2")

    arr = np.asarray(frame)
    mos_arr = np.asarray(mos, dtype=np.float64)
    _validate_pps_inputs(arr, mos_arr, n)

    N = arr.size
    if N < 2:
        raise ValueError("Murphy PPS requires at least 2 population units")
    if (mos_arr > 0).sum() < 2:
        raise ValueError("Murphy PPS requires at least two units with positive MOS")

    total = float(mos_arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("sum(MOS) must be > 0")

    p0 = mos_arr / total
    rng = resolve_random_state(rstate)

    # Two distinct draws
    idx = np.arange(N)
    first = int(rng.choice(idx, p=p0))

    rem_mask = idx != first
    p_rem = p0[rem_mask]
    p_rem = p_rem / p_rem.sum()
    rem_idx = idx[rem_mask]
    second = int(rng.choice(rem_idx, p=p_rem))

    hits_full = np.zeros(N, dtype=np.int64)
    hits_full[first] = 1
    hits_full[second] = 1

    # Exact Murphy inclusion probabilities
    denom = 1.0 - p0
    if np.any(denom <= 0):
        raise RuntimeError("Invalid state: base probability equals 1 for some unit.")
    S = (p0 / denom).sum()
    pi_full = np.clip(p0 * (1.0 + S - p0 / denom), 0.0, 1.0).astype(np.float64)

    mask = hits_full.astype(bool)
    cert_out = np.zeros(mask.sum(), dtype=bool)

    return arr[mask], hits_full[mask], pi_full[mask], cert_out


# ---------------------------------------------------------------------------
# PPS Rao-Sampford (RS)
# ---------------------------------------------------------------------------


def _select_pps_rs(
    *,
    frame: npt.ArrayLike,
    n: int,
    mos: npt.ArrayLike,
    certainty_threshold: float = 1.0,
    rstate: RandomState = None,
    max_attempts: int = 1000,
) -> _PPS_RETURN:
    """
    Rao-Sampford PPS without replacement.

    Algorithm:
      1. Iteratively extract certainty units (pi_eff_i >= certainty_threshold).
      2. On remaining units with adjusted n_rem:
         a. Compute base probabilities p_i = MOS_i / sum(MOS_remaining).
         b. Compute Sampford weights: w_i = p_i / (1 - n_rem * p_i).
         c. Draw first unit with probability p_i.
         d. Draw remaining n_rem-1 units WITHOUT replacement with prob
            proportional to w_i.
         e. Reject and retry if first unit appears in step (d) draw.

    First-order inclusion probabilities: pi_i = n * p_i (exact for RS
    when certainty units are handled correctly).

    Reference: Rao (1965), Sampford (1967),
               Sarndal, Swensson & Wretman (1992) s.3.4.
    """
    arr = np.asarray(frame)
    mos_arr = np.asarray(mos, dtype=np.float64)
    _validate_pps_inputs(arr, mos_arr, n)

    N = arr.size
    if N == 0 or n == 0:
        return _empty_result(arr.dtype)

    if n >= N:
        return _select_all(arr)

    total = float(mos_arr.sum())
    if total <= 0.0:
        raise ValueError("sum(MOS) must be > 0")

    p0 = mos_arr / total

    # --- Iterative certainty extraction ---
    cert_mask, n_rem = _extract_certainty(p0, n, certainty_threshold)
    n_cert = int(cert_mask.sum())

    if n_cert > 0:
        warnings.warn(
            f"PPS RS: {n_cert} certainty unit(s) detected "
            f"(threshold={certainty_threshold}). "
            "They are selected with probability 1.",
            UserWarning,
            stacklevel=4,
        )

    hits = np.zeros(N, dtype=np.int64)
    hits[cert_mask] = 1

    rng = resolve_random_state(rstate)

    if n_rem > 0:
        rem_idx = np.flatnonzero(~cert_mask)
        mos_rem = mos_arr[rem_idx]
        total_rem = float(mos_rem.sum())

        if total_rem > 0 and rem_idx.size >= n_rem:
            # Base probabilities on remaining units (re-normalized after
            # certainty extraction)
            p_rem = mos_rem / total_rem

            if n_rem == 1:
                # Trivial single draw
                pick = int(rng.choice(rem_idx, p=p_rem))
                hits[pick] = 1
            else:
                # Sampford weights: w_i = p_i / (1 - n_rem * p_i)
                # Well-defined because certainty units have been removed.
                pi_eff = n_rem * p_rem
                sampford_denom = 1.0 - pi_eff
                safe = sampford_denom > 1e-12

                if not safe.all():
                    warnings.warn(
                        "RS: near-zero Sampford denominator after certainty "
                        "extraction. Falling back to proportional weights for "
                        "affected units.",
                        UserWarning,
                        stacklevel=4,
                    )
                sampford_w = np.where(safe, p_rem / sampford_denom, p_rem)
                sampford_w = sampford_w / sampford_w.sum()

                # Rao-Sampford rejection sampling
                selected = None
                for _ in range(max_attempts):
                    first = int(rng.choice(rem_idx, p=p_rem))
                    rest = rng.choice(
                        rem_idx,
                        size=n_rem - 1,
                        replace=False,
                        p=sampford_w,
                    )
                    if first not in rest:
                        selected = np.concatenate(([first], rest))
                        break

                if selected is None:
                    warnings.warn(
                        f"RS: rejection sampling did not converge after "
                        f"{max_attempts} attempts. "
                        "Falling back to plain PPS WOR.",
                        UserWarning,
                        stacklevel=4,
                    )
                    selected = rng.choice(rem_idx, size=n_rem, replace=False, p=p_rem)

                hits[selected] = 1

    mask = hits > 0
    pi_full = np.where(cert_mask, 1.0, np.minimum(1.0, n * p0)).astype(np.float64)
    cert_out = cert_mask[mask]

    return arr[mask], hits[mask], pi_full[mask], cert_out
