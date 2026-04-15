# src/svy/engine/weighting/adj_poststratification.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from svy.core.types import (
    Category,
    DomainScalarMap,
    Number,
)
from svy.engine.weighting.adj_normalization import _normalize


def _get_unique_keys_efficient(arr: np.ndarray) -> set[Category]:
    """
    Efficiently extract unique keys from 1D or 2D arrays for validation.
    Returns a set of hashable items (scalars or tuples).
    """
    if arr.ndim > 1:
        # Optimization: Use vectorized unique on axis 0 first.
        # This avoids converting N rows to tuples in Python.
        # Complexity drops from O(N) Python overhead to O(G) Python overhead.
        uniq_rows = np.unique(arr, axis=0)
        return {tuple(row) for row in uniq_rows.tolist()}  # type: ignore[return-value]

    # 1D Case
    return set(np.unique(arr).tolist())


def _poststratify(
    *,
    wgt: np.ndarray,
    control: DomainScalarMap | Number | None,
    factor: DomainScalarMap | Number | None,
    by_arr: np.ndarray | None,
) -> np.ndarray:
    # --- 1. Resolution of Control/Factor ---
    # Post-stratification often uses "factors" (population proportions)
    # which we must convert to "controls" (population totals).

    if control is None and factor is not None:
        sum_weights = float(np.sum(wgt))

        if isinstance(factor, (int, float, np.integer, np.floating)):
            # Scalar factor: Target = Total_Wgt * Factor
            control = float(sum_weights * float(factor))

        elif isinstance(factor, Mapping):
            if by_arr is None:
                raise ValueError("Cannot use a factor dictionary without a domain ('by_arr').")

            # Map {domain: multiplier} -> {domain: target_total}
            control = {k: sum_weights * float(v) for k, v in factor.items()}
        else:
            raise TypeError("factor must be a mapping or a real number.")

    # --- 2. Validation (Strict Key Matching) ---
    # Unlike generic normalization, post-stratification strictly requires that
    # the control set matches the data set exactly (bijective mapping).

    if isinstance(control, Mapping):
        if by_arr is None:
            raise ValueError("Control dictionary provided but no domain array ('by_arr').")

        # Optimized Validation: Extract unique keys efficiently
        unique_data_keys = _get_unique_keys_efficient(by_arr)
        control_keys = set(control.keys())

        if unique_data_keys != control_keys:
            missing_in_data = control_keys - unique_data_keys
            missing_in_control = unique_data_keys - control_keys

            err_parts = []
            if missing_in_data:
                err_parts.append(
                    f"Keys in control but missing in data: {sorted(list(missing_in_data))}"
                )
            if missing_in_control:
                err_parts.append(
                    f"Keys in data but missing in control: {sorted(list(missing_in_control))}"
                )

            raise ValueError("Domain mismatch in post-stratification.\n" + "\n".join(err_parts))

    # --- 3. Delegate to Normalized Adjustment ---
    # Use the optimized vectorized normalization routine to compute weights.
    poststratified_weight, _, _ = _normalize(wgt=wgt, control=control, by_arr=by_arr)

    return poststratified_weight
