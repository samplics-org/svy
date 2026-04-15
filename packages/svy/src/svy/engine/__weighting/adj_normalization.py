# src/svy/engine/weighting/adj_normalization.py
from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np

from svy.core.types import (
    Category,
    ControlsType,
    DomainScalarMap,
    Number,
)
from svy.utils.checks import to_stringnumber


## Private Helper: Control/Factor Normalization
def _normalize_controls_like(*, x: ControlsType | None) -> ControlsType | None:
    """
    Ensures control/factor keys are canonical string/number types and values are floats.
    """
    if x is None:
        return None

    out: ControlsType = {}
    for margin, inner in x.items():
        norm_inner: DomainScalarMap = {}

        if not isinstance(inner, Mapping):
            raise TypeError(
                f"Control mapping for '{margin}' must be a dictionary of (value: control_target)."
            )

        for raw_key, raw_val in inner.items():
            if isinstance(raw_val, Mapping):
                raise TypeError(
                    f"Invalid nested mapping for margin '{margin}'. "
                    "Control/factor values must be numeric."
                )

            # Normalize keys to canonical token type (Category)
            k_sn: Category = to_stringnumber(token=raw_key)
            # Ensure value is float (numeric)
            v_num: Number = float(raw_val)

            norm_inner[k_sn] = v_num

        if not norm_inner:
            continue

        out[margin] = norm_inner

    return out if out else None


## NORMALIZATION method (Vectorized)
def _normalize(
    *,
    wgt: np.ndarray,
    control: DomainScalarMap | Number | None = None,
    by_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, DomainScalarMap | Number, DomainScalarMap | Number]:
    """
    Normalizes weights, either for the whole sample or within domains.
    Optimized to use O(N) vectorized operations instead of O(N*G) iteration.
    Handles 1D (main only) and 2D (main + replicates) weight arrays.
    """
    # Ensure wgt is float for calculation
    w_float = wgt.astype(float, copy=False)
    is_2d = w_float.ndim == 2

    # --- 1. Global Adjustment (Fast Path) ---
    if by_arr is None:
        # Sum across axis 0 (rows) -> Scalar if 1D, (R,) if 2D
        current_sum = np.sum(w_float, axis=0)

        if control is None:
            target_val = float(wgt.shape[0])
        else:
            target_val = float(cast(Number, control))

        # Calculate factor: Scalar or (R,)
        # Avoid div/0
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = target_val / current_sum

        # Fix Infs/NaNs
        if np.ndim(factor) == 0:
            if not np.isfinite(factor):
                factor = 1.0 if target_val == 0 else 0.0
        else:
            # Vector handling for replicates
            factor[~np.isfinite(factor)] = 0.0  # simplified fallback

        # Broadcast multiplication: (N, R) * (R,) works automatically in numpy
        return w_float * factor, factor, target_val

    # --- 2. Domain-wise Adjustment (Vectorized) ---

    # Map domains to integer indices [0, G-1]
    arr = np.asarray(by_arr)

    if arr.ndim > 1:
        uniq_rows, inverse_indices = np.unique(arr, axis=0, return_inverse=True)
        uniq_keys = [tuple(row) for row in uniq_rows]
    else:
        uniq_keys, inverse_indices = np.unique(arr, return_inverse=True)

    n_groups = len(uniq_keys)

    # Calculate current sum of weights per group: O(N)
    # np.bincount only works for 1D weights. Use np.add.at for 2D.
    if is_2d:
        n_reps = w_float.shape[1]
        current_sums = np.zeros((n_groups, n_reps), dtype=float)
        # np.add.at is unbuffered accumulation, suitable for grouping (N, R) into (G, R)
        np.add.at(current_sums, inverse_indices, w_float)
    else:
        current_sums = np.bincount(inverse_indices, weights=w_float, minlength=n_groups)

    # Resolve target controls: O(G) lookup
    # Targets are typically invariant across replicates, so shape is (G,)
    target_controls = np.zeros(n_groups, dtype=float)

    if control is None:
        # Target is simple count of items in group
        counts = np.bincount(inverse_indices, minlength=n_groups)
        target_controls = counts.astype(float)

    elif isinstance(control, dict):
        for i, k in enumerate(uniq_keys):
            key_lookup = k.item() if isinstance(k, np.generic) else k
            if key_lookup not in control:
                raise KeyError(f"Control mapping is missing key for domain: {key_lookup}")
            target_controls[i] = float(control[key_lookup])

    else:
        # Scalar broadcast
        target_controls[:] = float(control)

    # Prepare for broadcasting against (G, R) if necessary
    if is_2d:
        # Target (G,) -> (G, 1) so it broadcasts against Current (G, R)
        numerator = target_controls[:, None]
        denominator = current_sums
    else:
        numerator = target_controls
        denominator = current_sums

    # Calculate Factors: O(G) or O(G*R)
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = numerator / denominator

    # Fix infinite/NaNs where current_sum is 0
    # Logic: if current_sum is 0, factor is 1.0 if target is 0, else 0.0
    zero_sum_mask = denominator == 0
    if np.any(zero_sum_mask):
        # We need to broadcast the condition check if shapes differ
        t_check = numerator if is_2d else target_controls
        if is_2d:
            # Broadcast t_check (G, 1) to (G, R) for the condition
            t_broadcast = np.broadcast_to(t_check, factors.shape)
            factors[zero_sum_mask] = np.where(t_broadcast[zero_sum_mask] == 0, 1.0, 0.0)
        else:
            factors[zero_sum_mask] = np.where(t_check[zero_sum_mask] == 0, 1.0, 0.0)

    # Apply Factors: O(N) broadcast
    # inverse_indices maps N rows to G groups.
    # factors[inverse_indices] performs the lookup/expansion:
    # If 1D: factors (G,) -> (N,)
    # If 2D: factors (G, R) -> (N, R) via fancy indexing
    expanded_factors = factors[inverse_indices]

    norm_weight_arr = w_float * expanded_factors

    # Construct Output Maps: O(G)
    # NOTE: For 2D inputs, we return the factors for the FIRST column (Main Weight)
    # in the map to satisfy the DomainScalarMap signature, as map is mostly for inspection.
    adj_factor_map: DomainScalarMap = {}
    control2_map: DomainScalarMap = {}

    factors_for_map = factors[:, 0] if is_2d else factors

    for i, k in enumerate(uniq_keys):
        key_val = cast(Category, k.item() if isinstance(k, np.generic) else k)
        adj_factor_map[key_val] = float(factors_for_map[i])
        control2_map[key_val] = float(target_controls[i])

    return norm_weight_arr, adj_factor_map, control2_map
