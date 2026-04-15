# src/svy/engine/weighting/adj_calibration.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Sequence, Union

import numpy as np

from svy.core.types import (
    Category,
    Number,
)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """Return x as shape (n, k) float64."""
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(float, copy=False)


def _as_1d_float(x: Union[Sequence[Number], np.ndarray, Number]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


# ---------------------------------------------------------------------
# Core calibration linear algebra (Deville–Särndal linear calibration)
# ---------------------------------------------------------------------
def _core_matrix(
    *,
    w: np.ndarray,  # (n,)
    X: np.ndarray,  # (n,k)
    s: np.ndarray,  # (n,)
    tx: np.ndarray,  # (k,) control totals
) -> np.ndarray:
    """
    Solve for Lagrange multipliers lambda (lam) for a single group.
    System: (X^T diag(w/s) X) * lam = (tx - X^T w)
    """
    # 1. Compute weighted totals (observed)
    #    xw = sum(w * x)
    xw = (X.T * w).sum(axis=1)

    # 2. Check shapes
    if X.shape[1] != tx.shape[0]:
        # Scalar promotion if k=1
        if tx.size == 1 and X.shape[1] == 1:
            tx = tx.reshape(1)
        else:
            raise ValueError("x_control length must match the number of columns in X.")

    # 3. Construct System Matrix A
    #    A = X^T * diag(w/s) * X
    #    Optimization: broadcast (w/s) first
    ws = w / s
    X_ws = X * ws[:, None]  # (n,k)
    A = X.T @ X_ws  # (k,k)

    # 4. Construct RHS vector b
    b = tx - xw

    # 5. Solve
    try:
        lam = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback for singular/ill-conditioned matrices (rare but possible in survey data)
        lam = np.linalg.pinv(A) @ b

    return lam


def _calibrate(
    *,
    samp_weight: Union[Sequence[Number], np.ndarray],
    aux_vars: Union[Sequence[Number], np.ndarray],
    control: Union[
        Mapping[Category, Union[Mapping[Category, Number], Number]],
        Mapping[Category, Number],
        Sequence[Number],
        np.ndarray,
        Number,
    ],
    domain: Optional[Union[Sequence[Category], np.ndarray]] = None,
    scale: Union[Sequence[Number], np.ndarray, Number] = 1.0,
    bounded: bool = False,
    additive: bool = False,
) -> np.ndarray:
    """
    Calibrate sample weights using auxiliary variables and control totals.
    """
    if bounded:
        raise NotImplementedError("bounded=True is not implemented in the core engine function.")

    # --- 1. Standardize Inputs ---
    w = _as_1d_float(samp_weight)
    X = _ensure_2d(np.asarray(aux_vars, dtype=float))
    n, k = X.shape

    if w.shape[0] != n:
        raise ValueError("aux_vars must have the same number of rows as samp_weight.")

    # Scale s
    if np.ndim(scale) == 0:
        s = np.full(n, float(scale), dtype=float)  # type: ignore[arg-type]
    else:
        s = _as_1d_float(scale)
        if s.shape[0] != n:
            raise ValueError("scale must be scalar or length n.")

    # Avoid division by zero in scale
    s = np.where(s <= 0, 1e-10, s)

    # --- 2. Global Calibration (Fast Path) ---
    if domain is None:
        # Resolve control vector
        if isinstance(control, Mapping):
            # Assumes values are in column order.
            # Ideally should match keys to column names, but engine receives raw X.
            tx = np.array(list(control.values()), dtype=float)
        else:
            tx = _as_1d_float(control)

        lam = _core_matrix(w=w, X=X, s=s, tx=tx)

        # g = 1 + (X * lam) / s
        # w_cal = w * g
        correction = (X @ lam) / s

        if additive:
            # Legacy: additive shift? Usually calibration is multiplicative g-weight.
            # If additive requested: w_new = w + correction?
            # Standard regression estimator: w_cal = w + w/s * x'lam
            # The formula above computes w * (1 + x'lam/s) = w + w/s * x'lam.
            # "Additive" usually means g_j term added to weights?
            # Replicating original code's logic: return ONLY the adjustment factor?
            return 1.0 + correction

        return w * (1.0 + correction)

    # --- 3. Domain Calibration (Sort-and-Slice Optimization) ---

    # 3a. Normalize Domain Labels to Hashable
    d_raw = np.asarray(domain, dtype=object)
    if d_raw.ndim > 2 or d_raw.shape[0] != n:
        raise ValueError("domain must have length n.")

    # Vectorized tuple conversion for 2D domains
    if d_raw.ndim == 2:
        # Optimization: View as void type or use object conversion loop
        # Object conversion loop is O(N) but unavoidable for arbitrary types
        domain_keys = np.array([tuple(row) for row in d_raw.tolist()], dtype=object)
    else:
        domain_keys = d_raw

    # 3b. Sort Data by Domain (O(N log N))
    # This groups all rows for the same domain together contiguously
    sort_idx = np.argsort(domain_keys)

    # Apply sort to inputs
    w_sorted = w[sort_idx]
    X_sorted = X[sort_idx]
    s_sorted = s[sort_idx]
    dom_sorted = domain_keys[sort_idx]

    # Find boundaries of each domain slice
    unique_doms, start_indices = np.unique(dom_sorted, return_index=True)

    # Prepare output array (will be unsorted later)
    g_factors_sorted = np.ones(n, dtype=float)

    # 3c. Iterate Slices (O(G))
    # This loop is fast because it operates on contiguous blocks
    if not isinstance(control, Mapping):
        raise TypeError("When domain is provided, 'control' must be a mapping keyed by domain.")

    for i, d_val in enumerate(unique_doms):
        start = start_indices[i]
        end = start_indices[i + 1] if i + 1 < len(unique_doms) else n

        # Extract slice
        w_slice = w_sorted[start:end]
        X_slice = X_sorted[start:end]
        s_slice = s_sorted[start:end]

        # Get control for this domain
        ctrl_j = control.get(d_val)
        if ctrl_j is None:
            # Skip domains with no controls? Or raise?
            # Standard: if data exists but no control, variance/weights unadjusted (g=1)
            continue

        if isinstance(ctrl_j, Mapping):
            tx_j = np.array(list(ctrl_j.values()), dtype=float)
        elif isinstance(ctrl_j, (int, float, np.integer, np.floating)):
            tx_j = np.array([float(ctrl_j)], dtype=float)
        else:
            raise TypeError(f"Invalid control type for domain {d_val}")

        # Solve local system
        lam_j = _core_matrix(w=w_slice, X=X_slice, s=s_slice, tx=tx_j)

        # Compute local g-weights
        # g = 1 + (x'lam)/s
        correction_j = (X_slice @ lam_j) / s_slice
        g_factors_sorted[start:end] = 1.0 + correction_j

    # 3d. Restore Original Order (O(N))
    # Invert the permutation
    unsort_idx = np.argsort(sort_idx)
    g_factors = g_factors_sorted[unsort_idx]

    if additive:
        return g_factors  # Return the factor g itself

    return w * g_factors
