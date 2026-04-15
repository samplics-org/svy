# src/svy/engine/weighting/adj_nonresponse.py
from __future__ import annotations

import numpy as np

from svy.core.types import (
    DomainScalarMap,
)


# --- Private Helper for Non-Response Codes ---


def _response(resp_status: np.ndarray, resp_mapping: DomainScalarMap | None) -> np.ndarray:
    """
    Normalizes response status codes (e.g., 1, 2, 9) into standard strings ('rr', 'nr', 'in', 'uk').
    """
    if resp_mapping is None:
        # Assume resp_status is already standardized
        # Fast path check only if we suspect issues, otherwise strict conversion
        # Vectorized check is faster than manual all() loop for strings
        return np.asarray(resp_status, dtype=str)

    if "rr" not in resp_mapping:
        raise ValueError("The response dictionary must contain the key 'rr'!")

    # 1. Convert inputs to arrays
    status_arr = np.asarray(resp_status)

    # 2. Vectorized mapping using np.select or a lookup
    # Since standard_codes map multiple custom codes to one standard code,
    # we iterate the mapping (small K) instead of the data (large N).

    # Initialize with a placeholder that indicates "unmapped"
    # Use object type for string codes to avoid fixed-width truncation issues during setup
    resp_code = np.full(status_arr.shape, " ", dtype=object)

    standard_codes = {"rr": "rr", "in": "in", "nr": "nr", "uk": "uk"}

    for standard, custom_code in resp_mapping.items():
        if standard in standard_codes:
            # Vectorized assignment
            resp_code[status_arr == custom_code] = standard_codes[standard]  # type: ignore[index]

    # Check for unmapped statuses
    # We rely on the placeholder
    if np.any(resp_code == " "):
        print(
            "Warning: Some response statuses were not mapped and will be treated as neither "
            "Respondent ('rr'), Ineligible ('in'), Nonrespondent ('nr'), nor Unknown ('uk')."
        )

    return resp_code.astype(str)


# --- Private Helper: Vectorized Group Summation ---


def _sum_by_group(values: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Sums values per group_id. Handles 1D (vector) or 2D (matrix of replicates) values.
    """
    if values.ndim == 1:
        return np.bincount(group_ids, weights=values, minlength=n_groups)

    # 2D Case: Loop over columns (replicates).
    # Since replicates (M) are usually < 500, this loop is acceptable and much faster
    # than looping over N rows.
    n, m = values.shape
    out = np.zeros((n_groups, m), dtype=float)
    for i in range(m):
        out[:, i] = np.bincount(group_ids, weights=values[:, i], minlength=n_groups)
    return out


# --- Private Helper: Safe Division ---


def _safe_div_arr(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """
    Vectorized safe division for arrays.
    """
    tiny = 1e-10
    out = np.zeros_like(num, dtype=float)

    # Masking for safety
    zero_den = den == 0
    valid = ~zero_den

    # Normal division
    out[valid] = num[valid] / den[valid]

    # Denom zero, Num non-zero -> huge number
    # (Matches original logic: avoid NaN propogation for empty cells unless empty num too)
    non_zero_num = num != 0
    unsafe = zero_den & non_zero_num
    out[unsafe] = num[unsafe] / tiny

    # Both zero -> NaN
    both_zero = zero_den & ~non_zero_num
    out[both_zero] = np.nan

    return out


# --- ADJUST method (Main Public-facing Logic) ---


def _adjust_nr(
    *,
    wgts: np.ndarray,
    adj_class: np.ndarray | None,
    resp_status: np.ndarray,
    resp_mapping: DomainScalarMap | None = None,
    unknown_to_inelig: bool = True,
) -> np.ndarray:
    """
    Vectorized Non-Response Adjustment.
    Complexity: O(N) instead of O(G * N).
    """
    # 1. Standardize Inputs
    w = np.asarray(wgts, dtype=float)

    # Handle 1D vs 2D weights uniformly for math
    orig_1d = w.ndim == 1
    if orig_1d:
        w = w[:, None]  # (N, 1)

    n_rows, n_cols = w.shape

    # 2. Get Normalized Response Codes
    rc = _response(resp_status=resp_status, resp_mapping=resp_mapping)

    # Create masks (N,)
    is_rr = rc == "rr"
    is_in = rc == "in"
    is_nr = rc == "nr"
    is_uk = rc == "uk"

    # 3. Vectorize Grouping (Map classes to integers 0..G-1)
    if adj_class is None:
        # Single global group
        group_ids = np.zeros(n_rows, dtype=int)
        n_groups = 1
    else:
        # Handle Multi-column (2D) grouping efficiently
        ac = np.asarray(adj_class)
        if ac.ndim > 1:
            # For 2D keys, np.unique(axis=0) creates unique rows
            _, group_ids = np.unique(ac, axis=0, return_inverse=True)
        else:
            _, group_ids = np.unique(ac, return_inverse=True)
        n_groups = group_ids.max() + 1

    # 4. Calculate Group Sums (The heavy lifting)
    #    We sum weights for each status within each group.
    #    Result shape: (n_groups, n_cols)

    # Sum weights where status == RR
    sum_rr = _sum_by_group(w * is_rr[:, None], group_ids, n_groups)

    # Sum weights where status == NR
    sum_nr = _sum_by_group(w * is_nr[:, None], group_ids, n_groups)

    # Sum weights where status == IN
    # Optimization: Only calculate if we actually have ineligibles
    sum_in = (
        _sum_by_group(w * is_in[:, None], group_ids, n_groups)
        if is_in.any()
        else np.zeros((n_groups, n_cols))
    )

    # Sum weights where status == UK
    sum_uk = (
        _sum_by_group(w * is_uk[:, None], group_ids, n_groups)
        if is_uk.any()
        else np.zeros((n_groups, n_cols))
    )

    # 5. Calculate Factors per Group
    #    (Math matches the original scalar implementation)

    # Initialize group factors to 1.0
    factor_rr = np.ones((n_groups, n_cols))
    factor_in = np.ones((n_groups, n_cols))

    if unknown_to_inelig:
        # Unknowns distributed over eligible (in + rr + nr)
        denom_uk = sum_in + sum_rr + sum_nr
        adj_uk = _safe_div_arr(denom_uk + sum_uk, denom_uk)  # Inflation factor for unknowns

        # NR distributed over RR
        denom_rr = sum_rr
        adj_rr = _safe_div_arr(sum_rr + sum_nr, denom_rr)  # Inflation factor for NR

        # Combined factors
        factor_rr = adj_rr * adj_uk
        factor_in = adj_uk
    else:
        # Unknowns join NR pool -> distributed to RR
        # Ineligibles get no adjustment
        denom_rr = sum_rr
        adj_rr = _safe_div_arr(sum_rr + sum_nr + sum_uk, denom_rr)

        factor_rr = adj_rr
        factor_in[:] = 1.0  # Explicitly 1.0

    # 6. Broadcast Factors back to Rows (O(N) operation)
    #    We create the final adjustment matrix initialized to 0.0
    #    (NR/UK rows effectively get dropped/zeroed out)
    adj_weights = np.zeros((n_rows, n_cols), dtype=float)

    # Broadcast group factors to individual rows using group_ids
    # Row-level factor = Group-level factor[group_ids[row]]

    if is_rr.any():
        # w_new = w_old * factor_rr
        adj_weights[is_rr] = w[is_rr] * factor_rr[group_ids[is_rr]]

    if is_in.any():
        # w_new = w_old * factor_in
        adj_weights[is_in] = w[is_in] * factor_in[group_ids[is_in]]

    # Note: NR and UK rows remain 0.0 in adj_weights, effectively removing them.
    # If the original code returned NaN for them, we can do:
    # adj_weights[is_nr | is_uk] = np.nan
    # But usually 0 is safer for sums. If strictly NaN required, uncomment below:
    # adj_weights[~(is_rr | is_in)] = np.nan

    # 7. Return correct shape
    if orig_1d:
        return adj_weights.ravel()

    return adj_weights
