# src/svy/engine/weighting/replication.py
"""
Replicate weight creation for variance estimation.

Design principles:
- Returns ReplicateSpec with coefficients + metadata
- PSU index mapping for correct weight application
- No variance scale stored (computed when needed via compute_variance_scale)
- df with sensible defaults, user can override
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Literal

import numpy as np

from numpy.typing import NDArray

from svy.utils import hadamard as hdd
from svy.utils.random_state import RandomState, resolve_random_state, spawn_child_rngs


FloatArr = NDArray[np.float64]
IntArr = NDArray[np.int64]


@dataclass
class ReplicateSpec:
    """
    Replicate weight specification.

    Attributes:
        coefs: Replicate coefficient matrix, shape (n_psus, n_reps)
        psu_index: Array mapping each observation to its PSU row, shape (n_obs,)
        method: Replication method name ("BRR", "JKn", "Bootstrap")
        fay_coef: Fay's coefficient for BRR (0.0 for standard BRR)
        df: Degrees of freedom (user can override, or None for default)
    """

    coefs: FloatArr
    psu_index: IntArr
    method: str
    fay_coef: float = 0.0
    df: int | None = None


# ---------------------------------------------------------------------
# BRR
# ---------------------------------------------------------------------


def brr_replicate_spec(
    *,
    n_reps: int,
    psu: NDArray,
    stratum: NDArray | None = None,
    fay_coef: float = 0.0,
    df: int | None = None,
) -> ReplicateSpec:
    """
    Create BRR replicate specification.

    Args:
        n_reps: Number of replicates (must satisfy BRR constraints)
        psu: PSU identifiers for each observation
        stratum: Stratum identifiers, or None for unstratified
        fay_coef: Fay's adjustment coefficient in [0, 1)
        df: Degrees of freedom (None = n_psus - n_strata)

    Returns:
        ReplicateSpec with coefficients and metadata

    Raises:
        AssertionError: If design not paired (2 PSUs per stratum)
        ValueError: If fay_coef out of range
    """
    if not (0.0 <= fay_coef < 1.0):
        raise ValueError("Fay coefficient must be in [0, 1).")

    # Build ordered PSU list and validate pairing
    if stratum is None:
        unique_psus = np.unique(psu)
        n_psus = len(unique_psus)
        n_strata = (n_psus + 1) // 2
        ordered_psus = unique_psus
    else:
        unique_strata = np.unique(stratum)
        n_strata = len(unique_strata)

        # Validate: each stratum must have exactly 2 PSUs
        for s in unique_strata:
            mask = stratum == s
            n_psus_in_stratum = np.unique(psu[mask]).size
            if n_psus_in_stratum != 2:
                raise AssertionError(
                    f"BRR requires exactly 2 PSUs per stratum. "
                    f"Stratum {s!r} has {n_psus_in_stratum} PSUs."
                )

        # Build ordered PSU list
        ordered_psus = []
        for s in unique_strata:
            mask = stratum == s
            psus_in_stratum = np.unique(psu[mask])
            psus_in_stratum = np.sort(psus_in_stratum)
            ordered_psus.extend(psus_in_stratum)
        ordered_psus = np.array(ordered_psus)

    n_psus_total = len(ordered_psus)

    # Create PSU id -> row index mapping
    psu_to_row = {psu_id: idx for idx, psu_id in enumerate(ordered_psus)}

    # Generate Hadamard-based coefficients
    H_matrix: FloatArr = np.asarray(hdd.hadamard(n_reps), dtype=np.float64)
    hadamard_signs: FloatArr = H_matrix[:, 1 : n_strata + 1]

    coef_plus = fay_coef
    coef_minus = 2.0 - fay_coef

    is_plus_sign = hadamard_signs == 1.0

    rep_coef_psu1 = np.where(is_plus_sign, coef_plus, coef_minus)
    rep_coef_psu2 = np.where(is_plus_sign, coef_minus, coef_plus)

    out_interleaved = np.empty((n_reps, 2 * n_strata), dtype=np.float64)
    out_interleaved[:, ::2] = rep_coef_psu1
    out_interleaved[:, 1::2] = rep_coef_psu2

    # Handle odd number of PSUs
    if n_psus_total < 2 * n_strata:
        coefs = out_interleaved.T[:n_psus_total, :]
    else:
        coefs = out_interleaved.T

    # Create observation -> PSU row mapping
    psu_index = np.array([psu_to_row[p] for p in psu], dtype=np.int64)

    # Default df: n_psus - n_strata
    if df is None:
        df = max(1, n_psus_total - n_strata)

    return ReplicateSpec(
        coefs=coefs,
        psu_index=psu_index,
        method="BRR",
        fay_coef=fay_coef,
        df=df,
    )


def brr_nb_reps(psu: NDArray, stratum: NDArray | None = None, n_reps: int | None = None) -> int:
    """
    Compute valid number of BRR replicates given PSU/stratum structure.

    Args:
        psu: PSU identifiers
        stratum: Stratum identifiers, or None
        n_reps: Requested number (None = compute minimum)

    Returns:
        Valid number of replicates satisfying BRR constraints

    Notes:
        - R >= H (number of strata)
        - R is multiple of 4 (if R ≤ 28)
        - R is power of 2 (if R > 28)
    """
    if stratum is None:
        n_psus = np.unique(psu).size
        H = (n_psus + 1) // 2
    else:
        unique_strata = np.unique(stratum)
        H = unique_strata.size

        # Validate pairing
        for s in unique_strata:
            mask = stratum == s
            n_psus_in_stratum = np.unique(psu[mask]).size
            if n_psus_in_stratum != 2:
                raise AssertionError(
                    f"BRR requires exactly 2 PSUs per stratum. "
                    f"Stratum {s!r} has {n_psus_in_stratum} PSUs."
                )

    if n_reps is None:
        return int(4 * (H // 4 + 1))

    R = int(n_reps)
    if R < H:
        R = H

    if R <= 28:
        if R % 4 != 0:
            R = 4 * (R // 4 + 1)
    else:
        R = 1 << (R - 1).bit_length()

    return R


# ---------------------------------------------------------------------
# Jackknife
# ---------------------------------------------------------------------


def jkn_replicate_spec(
    *,
    psu: NDArray,
    stratum: NDArray | None = None,
    df: int | None = None,
) -> ReplicateSpec:
    """
    Create JKn (delete-1 jackknife) replicate specification.

    Args:
        psu: PSU identifiers for each observation
        stratum: Stratum identifiers, or None
        df: Degrees of freedom (None = n_psus - n_strata)

    Returns:
        ReplicateSpec with JKn coefficients

    Notes:
        Creates block-diagonal coefficient matrix where each PSU is
        dropped once. Coefficients are n/(n-1) for retained PSUs, 0 for dropped.
    """

    def _jkn_block(n_psus: int) -> FloatArr:
        """Create JKn block for n_psus PSUs."""
        if n_psus <= 1:
            raise ValueError("JKn requires at least 2 PSUs.")
        scale = float(n_psus) / float(n_psus - 1)
        return scale * (1.0 - np.eye(n_psus, dtype=np.float64))

    if stratum is None:
        psu_ids = np.unique(psu)
        n_psus = len(psu_ids)
        n_strata = 0

        jk_coefs = _jkn_block(n_psus)
        ordered_psus = psu_ids

    else:
        strata = np.unique(stratum)
        n_strata = len(strata)
        blocks: list[FloatArr] = []
        ordered_psus = []

        for s in strata:
            mask = stratum == s
            psu_ids_s = np.unique(psu[mask])

            if len(psu_ids_s) <= 1:
                raise ValueError(
                    f"Jackknife requires at least 2 PSUs per stratum. "
                    f"Stratum {s!r} has only {len(psu_ids_s)}."
                )

            psu_ids_s = np.sort(psu_ids_s)
            blocks.append(_jkn_block(len(psu_ids_s)))
            ordered_psus.extend(psu_ids_s)

        ordered_psus = np.array(ordered_psus)

        # Assemble block diagonal
        try:
            from scipy.linalg import block_diag

            jk_coefs = block_diag(*blocks)
        except ImportError:
            # Manual assembly
            total_psus = sum(b.shape[0] for b in blocks)
            jk_coefs = np.zeros((total_psus, total_psus), dtype=np.float64)
            row_offset = 0
            for b in blocks:
                n = b.shape[0]
                jk_coefs[row_offset : row_offset + n, row_offset : row_offset + n] = b
                row_offset += n

    n_psus_total = len(ordered_psus)

    # Create PSU mapping
    psu_to_row = {psu_id: idx for idx, psu_id in enumerate(ordered_psus)}
    psu_index = np.array([psu_to_row[p] for p in psu], dtype=np.int64)

    # Default df: n_psus - n_strata
    if df is None:
        df = max(1, n_psus_total - n_strata)

    return ReplicateSpec(
        coefs=jk_coefs,
        psu_index=psu_index,
        method="JKn",
        fay_coef=0.0,
        df=df,
    )


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------


def bootstrap_replicate_spec(
    *,
    n_reps: int,
    psu: NDArray,
    stratum: NDArray | None = None,
    samp_rate: float = 0.0,
    size_gap: int = 1,
    rstate: RandomState = None,
    df: int | None = None,
) -> ReplicateSpec:
    """
    Create bootstrap replicate specification with Rao-Wu-Yue adjustment.

    Args:
        n_reps: Number of bootstrap replicates
        psu: PSU identifiers for each observation
        stratum: Stratum identifiers, or None
        samp_rate: Sampling rate (for finite population correction)
        size_gap: Bootstrap sample size = n_psus - gap
        rstate: Random state for reproducibility
        df: Degrees of freedom (None = n_reps - 1)

    Returns:
        ReplicateSpec with bootstrap coefficients
    """

    def _boot_block(n_psus: int, n_reps: int, samp_rate: float, size_gap: int, rng) -> FloatArr:
        """Create bootstrap block for n_psus PSUs."""
        if n_psus <= size_gap:
            raise ValueError("size_gap must be < n_psus")

        sample_size = n_psus - size_gap
        psu_indices = np.arange(n_psus, dtype=np.int64)

        # Sample with replacement
        psu_boot = rng.choice(psu_indices, size=(n_reps, sample_size), replace=True)

        # Count occurrences
        psu_counts = np.zeros((n_psus, n_reps), dtype=np.float64)
        for rep in range(n_reps):
            ids, counts = np.unique(psu_boot[rep, :], return_counts=True)
            psu_counts[ids, rep] = counts

        # Rao-Wu-Yue adjustment
        c = math.sqrt((1.0 - samp_rate) * sample_size / (n_psus - 1.0))
        coefs = 1.0 - c + c * (n_psus / sample_size) * psu_counts

        return coefs

    if stratum is None:
        psu_ids = np.unique(psu)
        ordered_psus = np.sort(psu_ids)

        rng = resolve_random_state(rstate)
        boot_coefs = _boot_block(len(ordered_psus), n_reps, samp_rate, size_gap, rng)

    else:
        strata = np.unique(stratum)
        rng_parent = resolve_random_state(rstate)
        child_rngs = spawn_child_rngs(rng_parent, strata)

        blocks: list[FloatArr] = []
        ordered_psus = []

        for s in strata:
            mask = stratum == s
            psu_ids_s = np.unique(psu[mask])
            psu_ids_s = np.sort(psu_ids_s)

            block = _boot_block(len(psu_ids_s), n_reps, samp_rate, size_gap, child_rngs[s])
            blocks.append(block)
            ordered_psus.extend(psu_ids_s)

        ordered_psus = np.array(ordered_psus)
        boot_coefs = np.vstack(blocks) if blocks else np.zeros((0, n_reps), dtype=np.float64)

    # Create PSU mapping
    psu_to_row = {psu_id: idx for idx, psu_id in enumerate(ordered_psus)}
    psu_index = np.array([psu_to_row[p] for p in psu], dtype=np.int64)

    # Default df: n_reps - 1
    if df is None:
        df = max(1, n_reps - 1)

    return ReplicateSpec(
        coefs=boot_coefs,
        psu_index=psu_index,
        method="Bootstrap",
        fay_coef=0.0,
        df=df,
    )


# ---------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------


def create_replicate_spec(
    *,
    method: Literal["BRR", "JKn", "Bootstrap"],
    psu: NDArray,
    stratum: NDArray | None = None,
    n_reps: int | None = None,
    fay_coef: float = 0.0,
    samp_rate: float = 0.0,
    size_gap: int = 1,
    rstate: RandomState = None,
    df: int | None = None,
) -> ReplicateSpec:
    """
    Create replicate weight specification (main entry point).

    Args:
        method: Replication method
        psu: PSU identifiers for each observation
        stratum: Stratum identifiers, or None
        n_reps: Number of replicates (None = method default)
        fay_coef: Fay's coefficient for BRR
        samp_rate: Sampling rate for bootstrap
        size_gap: Bootstrap gap
        rstate: Random state for bootstrap
        df: Degrees of freedom (None = method default)

    Returns:
        ReplicateSpec with coefficients and metadata
    """
    _meth = method.upper()

    if _meth == "BRR":
        if n_reps is None:
            n_reps = brr_nb_reps(psu, stratum)
        return brr_replicate_spec(
            n_reps=n_reps,
            psu=psu,
            stratum=stratum,
            fay_coef=fay_coef,
            df=df,
        )

    elif _meth == "JKN":
        return jkn_replicate_spec(psu=psu, stratum=stratum, df=df)

    elif _meth == "BOOTSTRAP":
        if n_reps is None:
            n_reps = 500
        return bootstrap_replicate_spec(
            n_reps=n_reps,
            psu=psu,
            stratum=stratum,
            samp_rate=samp_rate,
            size_gap=size_gap,
            rstate=rstate,
            df=df,
        )

    else:
        raise ValueError(f"Unknown replication method: {_meth}")


def compute_variance_scale(method: str, fay_coef: float = 0.0) -> float:
    """
    Compute variance scale factor for replication method.

    Used in variance formula: var(θ̂) = scale * (1/R) * Σ(θ̂_r - θ̂)²

    Args:
        method: Replication method
        fay_coef: Fay's coefficient (only for BRR)

    Returns:
        Variance scale factor
    """
    _meth = method.upper()

    if _meth == "BRR":
        return 1.0 / ((1.0 - fay_coef) ** 2) if fay_coef > 0 else 1.0
    elif _meth in ("JKN", "BOOTSTRAP"):
        return 1.0
    else:
        raise ValueError(f"Unknown method: {_meth}")
