# src/svy/engine/sampling/srs.py
from __future__ import annotations

from typing import Mapping, cast

import numpy as np
import numpy.typing as npt

from svy.core.types import DT, FloatArray, IntArray
from svy.utils.random_state import RandomState, resolve_random_state, spawn_child_rngs


def _select_srs(
    frame: npt.NDArray[DT] | npt.ArrayLike,
    n: int | Mapping[DT, int],
    *,
    stratum: npt.NDArray[DT] | npt.ArrayLike | None,
    wr: bool = False,
    rstate: RandomState = None,
) -> tuple[npt.NDArray[DT], IntArray, FloatArray]:
    if stratum is None:
        assert isinstance(n, int), "n must be an integer when stratum is None"
        sel_idx, hits, probs = _select_srs_unstratified(
            frame=frame,
            n=n,
            wr=wr,
            rstate=rstate,
        )
    else:
        sel_idx, hits, probs = _select_srs_stratified(
            frame=frame,
            n=n,
            stratum=stratum,
            wr=wr,
            rstate=rstate,
        )

    return sel_idx, hits, probs


def _select_srs_unstratified(
    frame: npt.NDArray[DT] | npt.ArrayLike,
    n: int,
    *,
    wr: bool = False,
    rstate: RandomState = None,
) -> tuple[npt.NDArray[DT], IntArray, FloatArray]:
    arr = np.asarray(frame)
    arr = cast(npt.NDArray[DT], arr)
    N = arr.size
    if N == 0:
        raise ValueError("Cannot sample from an empty frame.")
    if n < 0:
        raise ValueError("'n' must be non-negative.")
    if not wr and n > N:
        raise ValueError("Cannot sample more than population size without replacement.")

    rng = resolve_random_state(rstate)

    picks = rng.choice(N, size=n, replace=wr) if n > 0 else np.empty(0, dtype=int)
    hits_all = np.bincount(picks, minlength=N) if n > 0 else np.zeros(N, dtype=int)
    mask = hits_all > 0

    sample = arr[mask]  # npt.NDArray[DT]
    hits = hits_all[mask].astype(np.int64)
    pi = 1.0 - (1.0 - 1.0 / N) ** n if wr else n / N
    probs = np.full(hits.shape, pi, dtype=np.float64)

    return sample, hits, probs


def _select_srs_stratified(
    frame: npt.NDArray[DT] | npt.ArrayLike,
    n: int | Mapping[DT, int],
    *,
    stratum: npt.NDArray[DT] | npt.ArrayLike,
    wr: bool = False,
    rstate: RandomState = None,
) -> tuple[npt.NDArray[DT], IntArray, FloatArray]:
    arr = np.asarray(frame)
    strat = np.asarray(stratum)
    if arr.ndim != 1:
        raise ValueError(f"'frame' must be 1-D; got shape {arr.shape}")
    if arr.shape[0] != strat.shape[0]:
        raise ValueError(f"Length mismatch: frame={arr.shape[0]} vs stratum={strat.shape[0]}")

    strata_vals = np.unique(strat)

    # Build per-stratum sample sizes
    if isinstance(n, Mapping):
        size_map: Mapping[DT, int] = n  # type: ignore[assignment]
        # missing = [s for s in strata_vals if s not in size_map]
        # if missing:
        #     raise KeyError(f"Missing sample sizes for strata: {missing!r}")
    else:
        if int(n) < 0:
            raise ValueError("'n' must be non-negative")
        size_map = {s: int(n) for s in strata_vals}

    rng_parent = resolve_random_state(rstate)
    child_rng = spawn_child_rngs(rng_parent, strata_vals)

    sel_chunks: list[npt.NDArray[DT]] = []
    hits_chunks: list[IntArray] = []
    prob_chunks: list[FloatArray] = []

    sel_s: npt.NDArray[np.int_]
    hits_s: npt.NDArray[np.int_]
    prob_s: npt.NDArray[np.float64]
    for s in strata_vals:
        m = strat == s
        sel_s, hits_s, prob_s = _select_srs_unstratified(
            frame=arr[m],
            n=size_map[s],
            wr=wr,
            rstate=child_rng[s],  # avoid re-seeding per stratum
        )
        # collect
        sel_chunks.append(sel_s.astype(arr.dtype, copy=False))
        hits_chunks.append(hits_s.astype(np.int64, copy=False))
        prob_chunks.append(prob_s.astype(np.float64, copy=False))

    if not sel_chunks:
        # empty result with correct dtypes
        return (
            np.asarray([], dtype=arr.dtype),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
        )

    selected = np.concatenate(sel_chunks, axis=0)
    hits = np.concatenate(hits_chunks, axis=0)
    probs = np.concatenate(prob_chunks, axis=0)
    return selected, hits, probs
