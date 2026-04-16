# src/svy/weighting/poststratification.py
"""
Post-stratification weight adjustment.

The poststratify() function takes a Sample and returns a Sample (for chaining).
The Weighting class in base.py delegates to this function directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import polars as pl


try:
    from svy_rs._internal import poststratify as rust_poststratify  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_poststratify = None

from svy.core.types import DomainScalarMap, Number
from svy.errors import MethodError
from svy.weighting._helpers import _build_by_array, _normalize_dict_keys
from svy.weighting.raking import _trim_constraints_satisfied
from svy.weighting.types import TrimConfig, resolve_threshold


try:
    from svy_rs._internal import trim_weights as rust_trim_weights  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_trim_weights = None

if TYPE_CHECKING:
    from svy.core.sample import Sample


def _to_hashable(v) -> object:
    """Convert a value to a hashable Python object.

    Multi-column by= produces numpy arrays as elements; convert to tuples.
    """
    if isinstance(v, (list, np.ndarray)):
        return tuple(v)
    return v


def _encode_strata(by_arr: np.ndarray) -> tuple[np.ndarray, list, list]:
    """Encode stratum labels to contiguous int64 indices for Rust.

    Uses FIRST-SEEN order to match Rust's create_group_mapping which uses
    HashMap insertion order. Since we assign 0,1,2,...,n-1, Rust's BTreeSet
    validation sees [0,1,...,n-1] which matches our control_arr order.

    Returns
    -------
    by_indices  : int64 array, shape (n,)  — contiguous 0..n_strata-1
    unique_vals : list of stratum labels in first-seen order
    normed      : list of hashable values (same length as by_arr)
    """
    normed = [_to_hashable(v) for v in by_arr.tolist()]
    seen: dict = {}
    next_id = 0
    for v in normed:
        if v not in seen:
            seen[v] = next_id
            next_id += 1
    unique_vals = [v for v, _ in sorted(seen.items(), key=lambda x: x[1])]
    by_indices = np.array([seen[v] for v in normed], dtype=np.int64)
    return by_indices, unique_vals, normed


def _stratum_sums(wgt_arr: np.ndarray, by_indices: np.ndarray, n_strata: int) -> np.ndarray:
    """Sum weights per stratum using bincount — single C pass, no Python loop."""
    return np.bincount(by_indices, weights=wgt_arr, minlength=n_strata)


def _resolve_to_controls(
    controls: DomainScalarMap | Number | None,
    factors: DomainScalarMap | Number | None,
    by_arr: np.ndarray | None,
    wgt_arr: np.ndarray,
    unique_vals: list | None,
    normed: list | None,
    by_indices: np.ndarray | None,
) -> np.ndarray:
    """Convert controls or factors to an absolute control array for Rust.

    Rust takes absolute totals only. Factors are arbitrary multipliers
    (not required to sum to 1) converted as: control[s] = factor[s] * sum(w[s]).

    Returns float64 array with one value per stratum (sorted order).
    """
    if by_arr is None:
        # Single stratum
        if controls is not None:
            val = (
                float(next(iter(controls.values())))
                if isinstance(controls, dict)
                else float(controls)  # type: ignore[arg-type]
            )
        else:
            fval = (
                float(next(iter(factors.values())))  # type: ignore[union-attr]
                if isinstance(factors, dict)
                else float(factors)  # type: ignore[arg-type]
            )
            # factor × grand_total → target
            val = fval * float(wgt_arr.sum())
        return np.array([val], dtype=np.float64)

    assert unique_vals is not None and normed is not None and by_indices is not None
    n_strata = len(unique_vals)

    if controls is not None:
        if isinstance(controls, dict):
            controls = _normalize_dict_keys(controls)
            data_keys = set(unique_vals)
            ctrl_keys = set(controls.keys())
            extra = ctrl_keys - data_keys
            missing = data_keys - ctrl_keys
            if extra or missing:
                raise MethodError.invalid_mapping_keys(
                    where="Sample.weighting.poststratify",
                    param="controls",
                    missing=sorted(str(k) for k in missing),
                    extra=sorted(str(k) for k in extra),
                )
            return np.array([float(controls[v]) for v in unique_vals], dtype=np.float64)
        else:
            # Scalar: proportional split by current stratum weight sums
            total = float(wgt_arr.sum())
            sums = _stratum_sums(wgt_arr, by_indices, n_strata)
            return (float(controls) / total * sums).astype(np.float64)  # type: ignore[arg-type]
    else:
        grand_total = float(wgt_arr.sum())
        if isinstance(factors, dict):
            factors = _normalize_dict_keys(factors)
            data_keys = set(unique_vals)
            fact_keys = set(factors.keys())  # type: ignore[union-attr]
            extra = fact_keys - data_keys
            missing = data_keys - fact_keys
            if extra or missing:
                raise MethodError.invalid_mapping_keys(
                    where="Sample.weighting.poststratify",
                    param="factors",
                    missing=sorted(str(k) for k in missing),
                    extra=sorted(str(k) for k in extra),
                )
            # factor × grand_total → target for each stratum
            return np.array(
                [
                    float(factors[v]) * grand_total  # type: ignore[index]
                    for v in unique_vals
                ],
                dtype=np.float64,
            )
        else:
            # Scalar factor: same proportion of grand total for every stratum
            return np.full(n_strata, float(factors) * grand_total, dtype=np.float64)  # type: ignore[arg-type]


def poststratify(
    sample: Sample,
    controls: DomainScalarMap | Number | None = None,
    *,
    factors: DomainScalarMap | Number | None = None,
    by: str | Sequence[str] | None = None,
    wgt_name: str = "ps_wgt",
    ignore_reps: bool = False,
    update_design_wgts: bool = True,
    strict: bool = True,
    trimming: TrimConfig | None = None,
) -> Sample:
    if controls is None and factors is None:
        raise MethodError.not_applicable(
            where="Sample.weighting.poststratify",
            method="poststratify",
            reason="Either controls= or factors= must be specified.",
        )

    where = "Sample.weighting.poststratify"
    df = sample._data
    design = sample._design

    if design.wgt is None:
        raise MethodError.not_applicable(
            where=where,
            method="poststratify",
            reason="Sample weight is None. Set design.wgt before calling poststratify().",
        )
    wgt = design.wgt
    if wgt not in df.columns:
        raise MethodError.invalid_choice(
            where=where,
            param="design.wgt",
            got=wgt,
            allowed=list(df.columns),
            hint="Check that the weight column exists in the data.",
        )

    existing_cols = set(df.columns)
    if wgt_name in existing_cols:
        raise MethodError.not_applicable(
            where=where,
            method="poststratify",
            reason=f"Column '{wgt_name}' already exists. Choose a different wgt_name.",
        )

    by_arr = _build_by_array(df, by, where="Sample.weighting.poststratify")

    wgt_arr = df.get_column(wgt).to_numpy().astype(np.float64)

    # Encode strata and resolve controls/factors to absolute totals once
    if by_arr is not None:
        by_indices, unique_vals, normed = _encode_strata(by_arr)
    else:
        by_indices = np.zeros(len(wgt_arr), dtype=np.int64)
        unique_vals = None
        normed = None

    control_arr = _resolve_to_controls(
        controls, factors, by_arr, wgt_arr, unique_vals, normed, by_indices
    )

    # Main weight: reshape to (n, 1), call Rust, extract column 0
    assert rust_poststratify is not None  # noqa: S101
    ps_wgt_arr = rust_poststratify(wgt_arr.reshape(-1, 1), by_indices, control_arr)[:, 0]

    df = df.with_columns(pl.Series(name=wgt_name, values=ps_wgt_arr))

    if update_design_wgts:
        sample._design = sample._design.update(wgt=wgt_name)

    if not ignore_reps and design.rep_wgts is not None:
        rep_cols = design.rep_wgts.columns
        if rep_cols:
            wgts_arr = df.select(rep_cols).to_numpy()

            adj_wgts_arr = rust_poststratify(wgts_arr, by_indices, control_arr)

            n_reps = len(rep_cols)
            new_rep_names = [f"{wgt_name}{i}" for i in range(1, n_reps + 1)]
            wgts_df = pl.DataFrame(adj_wgts_arr, schema=new_rep_names)

            sample._data = df.hstack(wgts_df)
            df = sample._data

            if update_design_wgts:
                sample._design = sample._design.update_rep_weights(
                    method=design.rep_wgts.method,
                    prefix=wgt_name,
                    n_reps=n_reps,
                    fay_coef=design.rep_wgts.fay_coef,
                    df=design.rep_wgts.df,
                )

    sample._data = df

    # ── Trim-poststratify cycle ───────────────────────────────────────────
    if trimming is not None:
        _w_pos = ps_wgt_arr[ps_wgt_arr > 0].astype(np.float64)
        _upper_val = (
            resolve_threshold(trimming.upper, _w_pos) if trimming.upper is not None else None
        )
        _lower_val = (
            resolve_threshold(trimming.lower, _w_pos) if trimming.lower is not None else None
        )

        assert rust_trim_weights is not None  # noqa: S101
        assert rust_poststratify is not None  # noqa: S101

        _current_w = ps_wgt_arr.copy()
        _trim_ok = False

        for _cycle in range(trimming.max_iter):
            # Poststratify step — restore margins
            _ps_result = rust_poststratify(_current_w.reshape(-1, 1), by_indices, control_arr)
            _current_w = _ps_result[:, 0]

            # Trim step
            (_trimmed_w, *_) = rust_trim_weights(
                _current_w,
                _upper_val,
                _lower_val,
                trimming.redistribute,
                trimming.max_iter,
                trimming.tol,
            )
            _current_w = _trimmed_w
            _trim_ok = _trim_constraints_satisfied(_current_w, _upper_val, _lower_val, 1e-4)

            if _trim_ok:
                # Final poststratify to restore margins after last trim
                _final = rust_poststratify(_current_w.reshape(-1, 1), by_indices, control_arr)
                _current_w = _final[:, 0]
                # Re-check trim after final poststratify
                _trim_ok = _trim_constraints_satisfied(_current_w, _upper_val, _lower_val, 1e-4)
                break

        if strict and not _trim_ok:
            raise MethodError.not_applicable(
                where=where,
                method="poststratify",
                reason=(
                    f"Trim-poststratify cycle did not converge after {trimming.max_iter} cycles. "
                    "The design has NOT been modified. "
                    "Pass strict=False to store partial results."
                ),
                hint="Increase TrimConfig.max_iter or use a less restrictive trim threshold.",
            )

        # Update stored weight with final cycled result
        df = df.with_columns(pl.Series(name=wgt_name, values=_current_w))
        sample._data = df

    return sample
