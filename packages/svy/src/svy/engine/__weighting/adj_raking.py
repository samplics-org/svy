# src/svy/engine/weighting/adj_raking.py
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from svy.core.types import (
    DictStrArray,
    DomainScalarMap,
    Number,
)
from svy.utils.checks import to_stringnumber


class MarginStruct(NamedTuple):
    """
    Pre-computed structure for fast raking iteration.
    Stores the static mapping of rows to groups and the target totals.
    """

    indices: np.ndarray  # (n,) integer indices mapping rows to groups
    n_groups: int  # number of groups for bincount minlength
    targets: np.ndarray  # (n_groups,) target control totals ordered by indices


# --- Private Helper: Core Check Functions ---


def _check_raking_convergence(
    *, current_wgts: np.ndarray, prev_weights: np.ndarray, tol: float
) -> tuple[bool, float]:
    """Checks if the raking process has converged (max relative difference is below tolerance)."""
    # Calculate relative difference: |new - old| / (|old| + tiny)
    diff = np.abs(current_wgts - prev_weights)
    denom = np.abs(prev_weights) + 1e-8
    max_relative_diff = float(np.max(diff / denom))

    converged = max_relative_diff < tol
    return converged, max_relative_diff


def _check_raking_bounds(
    *,
    raked_wgts: np.ndarray,
    initial_wgts: np.ndarray,
    ll_bound: Number | None,
    up_bound: Number | None,
) -> bool:
    """Checks if the raked weights are within the specified bounds (relative to initial weights)."""
    if ll_bound is None and up_bound is None:
        return True

    # Use safe division to handle initial_wgts=0
    # If initial=0, the weight shouldn't change, so ratio is effectively 1 (or 0/0 -> nan).
    # We ignore warnings here; checking min/max handles NaNs appropriately if they occur.
    with np.errstate(divide="ignore", invalid="ignore"):
        # For checking bounds, we usually only care about non-zero weights
        mask = initial_wgts != 0
        if not np.any(mask):
            return True  # All zero weights, trivial pass

        ratios = raked_wgts[mask] / initial_wgts[mask]

    if ll_bound is not None and np.min(ratios) < float(ll_bound):
        return False

    if up_bound is not None and np.max(ratios) > float(up_bound):
        return False

    return True


# --- Private Helper: Pre-computation (The Optimization) ---


def _prepare_margins(
    n: int, margins: DictStrArray, control: dict[str, DomainScalarMap]
) -> list[MarginStruct]:
    """
    Pre-calculates integer indices and aligned target vectors for all margins.
    This moves the expensive O(N*logN) grouping work out of the iterative loop.
    """
    structs = []

    for m_name, arr_raw in margins.items():
        arr = np.asarray(arr_raw)
        if arr.shape[0] != n:
            raise ValueError(
                f"Margin '{m_name}' length mismatch. Expected {n}, got {arr.shape[0]}."
            )

        # 1. Map labels to integer indices [0, G-1]
        if arr.ndim > 1:
            # 2D case: unique rows
            uniq_rows, indices = np.unique(arr, axis=0, return_inverse=True)
            # Convert uniques to lookup keys (tuples) for dictionary matching
            uniq_keys = [tuple(row) for row in uniq_rows.tolist()]
        else:
            # 1D case
            uniq_keys_arr, indices = np.unique(arr, return_inverse=True)
            # Convert numpy scalars to python types for dictionary lookup
            uniq_keys = [k.item() if isinstance(k, np.generic) else k for k in uniq_keys_arr]

        n_groups = len(uniq_keys)

        # 2. Build Dense Target Array from Control Dict
        # The array must align exactly with the integer indices 0..G-1
        ctrl_map = control[m_name]
        targets = np.zeros(n_groups, dtype=float)

        for i, key in enumerate(uniq_keys):
            # Normalize key for lookup (match standard svy logic)
            # Try exact match first, then string/number conversion
            if key in ctrl_map:
                val = ctrl_map[key]
            else:
                k_norm = to_stringnumber(token=key)
                if k_norm in ctrl_map:
                    val = ctrl_map[k_norm]
                else:
                    raise ValueError(
                        f"Control total missing for domain value '{key}' in margin '{m_name}'."
                    )

            targets[i] = float(val)

        structs.append(MarginStruct(indices, n_groups, targets))

    return structs


# --- Private Helper: Controls from Factors ---


def _calculate_controls_from_factors(
    *,
    wgts: np.ndarray,
    margins: DictStrArray,
    factors: dict[str, DomainScalarMap],
) -> dict[str, DomainScalarMap]:
    """
    Calculates control totals based on factors.
    Optimized to use vectorized summation.
    """
    wgts = np.asarray(wgts, dtype=float)
    total_w = float(np.sum(wgts))

    if total_w <= 0:
        raise ValueError("Initial sum of weights must be positive for factor calculation.")

    out: dict[str, DomainScalarMap] = {}

    for m, labels_arr in margins.items():
        if m not in factors:
            raise ValueError(f"Missing factors for margin '{m}'.")

        fac_map = factors[m]
        arr = np.asarray(labels_arr)

        # Vectorized grouping
        if arr.ndim > 1:
            uniq_vals, indices = np.unique(arr, axis=0, return_inverse=True)
            keys = [tuple(r) for r in uniq_vals.tolist()]
        else:
            uniq_vals, indices = np.unique(arr, return_inverse=True)
            keys = [k.item() if isinstance(k, np.generic) else k for k in uniq_vals]

        n_groups = len(keys)

        # Calculate Sample Weights by Category (W_c)
        w_by_cat = np.bincount(indices, weights=wgts, minlength=n_groups)

        # Calculate Sample Shares (P_c)
        p_by_cat = w_by_cat / total_w

        # Look up factors and build target array
        f_arr = np.zeros(n_groups, dtype=float)
        for i, key in enumerate(keys):
            # Normalize key
            k_norm = to_stringnumber(token=key)
            if k_norm in fac_map:
                f = float(fac_map[k_norm])
            elif key in fac_map:
                f = float(fac_map[key])
            else:
                raise ValueError(f"Factor missing for category '{key}' in margin '{m}'.")

            if f < 0:
                raise ValueError(f"Invalid factor {f} for category '{key}'.")
            f_arr[i] = f

        # Calculate Targets
        # Target (U_c) = Total * P_c * F_c
        targets_unnorm = total_w * p_by_cat * f_arr

        # Renormalize to preserve total sum
        s = np.sum(targets_unnorm)
        if s > 0 and abs(s - total_w) > 1e-8:
            targets_unnorm *= total_w / s

        # Build Output Map
        ctrl_m = {}
        for i, key in enumerate(keys):
            k_norm = to_stringnumber(token=key)
            ctrl_m[k_norm] = targets_unnorm[i]

        out[m] = ctrl_m

    return out


# --- RAKE method (Main Public-facing Logic) ---


def _rake(
    *,
    wgt: np.ndarray,
    margins: DictStrArray,
    control: dict[str, DomainScalarMap],
    ll_bound: Number | None = None,
    up_bound: Number | None = None,
    tol: float = 1e-4,
    max_iter: int = 100,
    display_iter: bool = False,
) -> np.ndarray:
    """
    Optimized Raking Implementation (IPF).
    Performs one-time grouping preparation, then runs a tight loop using
    fast numpy operations (bincount, broadcast).
    """
    n = wgt.shape[0]

    # 1. Pre-compute structures (Indices, Targets)
    #    This avoids repeating np.unique inside the loop.
    margin_structs = _prepare_margins(n, margins, control)

    raked_weights = wgt.astype(float, copy=True)
    iteration = 0
    converged = False

    # 2. Iteration Loop
    while not converged and iteration < max_iter:
        iteration += 1
        weights_start = raked_weights.copy()

        # --- Cycle through margins (IPF steps) ---
        for struct in margin_structs:
            # struct.indices: (n,) int array mapping rows to groups
            # struct.targets: (n_groups,) float array of target totals

            # Sum current weights by group: O(N)
            current_sums = np.bincount(
                struct.indices, weights=raked_weights, minlength=struct.n_groups
            )

            # Calculate adjustment factors: O(G)
            # Handle division by zero: if sum is 0, no adjustment possible (factor=1 or 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                factors = struct.targets / current_sums

            # Clean up Infs/NaNs from zero sums
            mask_zero = current_sums == 0
            if np.any(mask_zero):
                # If target is also 0, factor is 1 (steady). If target > 0, we can't solve (use 0/error logic).
                # Convention: leave 0 weights as 0.
                factors[mask_zero] = np.where(struct.targets[mask_zero] == 0, 1.0, 0.0)

            # Update weights: O(N) broadcast
            # w_new = w_old * factor[group_id]
            raked_weights *= factors[struct.indices]

        # --- Check Convergence ---
        converged, max_diff = _check_raking_convergence(
            current_wgts=raked_weights, prev_weights=weights_start, tol=tol
        )

        # --- Check Bounds ---
        if ll_bound is not None or up_bound is not None:
            is_bounded = _check_raking_bounds(
                raked_wgts=raked_weights, initial_wgts=wgt, ll_bound=ll_bound, up_bound=up_bound
            )
            if not is_bounded:
                if display_iter:
                    print(f"Iteration {iteration}: Bounds exceeded.")
                raise ValueError(
                    "Raking failed: Weight ratios exceeded specified bounds. "
                    "Check `ll_bound` and `up_bound`."
                )

        if display_iter:
            print(f"  Iteration {iteration}: Max Rel Diff = {max_diff:.6g}")

    # 3. Finalization
    if not converged:
        print(f"Warning: Raking did not converge after {max_iter} iterations.")

    return raked_weights
