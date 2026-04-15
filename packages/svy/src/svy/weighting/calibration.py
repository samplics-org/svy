# src/svy/weighting/calibration.py
"""
Calibration weight adjustment (GREG / generalised raking).

All public functions take a Sample and return a Sample (for chaining).
The Weighting class in base.py delegates to these functions directly.

Note on additive GREG
---------------------
The samplics library exposed an ``additive`` parameter on calibrate() that
returned a (n_units, n_domains) weight matrix where each column was a
full-sample GREG calibration toward one domain's totals, with the property
that domain estimates sum to the overall GREG total.

This is a valid statistical feature but it does not fit the svy weighting
pipeline (one active weight column per Sample) and neither R survey nor
Stata svycal expose it as a weight-construction step — both handle the
additive property at estimation time.

``additive`` has been removed from the public API.  It is reserved for a
future dedicated API (e.g. ``Sample.weighting.greg_additive()``) once the
estimation layer has a clear contract for consuming domain weight matrices.
The Rust engine's internal ``additive`` flag (return g-factors instead of
calibrated weights) is an implementation detail that must never be exposed
to Python callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import polars as pl

try:
    from svy_rs._internal import calibrate as rust_calibrate  # type: ignore[import-untyped]
    from svy_rs._internal import calibrate_by_domain as rust_calibrate_by_domain  # type: ignore[import-untyped]
    from svy_rs._internal import calibrate_parallel as rust_calibrate_parallel  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_calibrate = None
    rust_calibrate_by_domain = None
    rust_calibrate_parallel = None

from svy.core.terms import Feature
from svy.core.types import Category, Number
from svy.errors import DimensionError, MethodError
from svy.weighting._calibration_utils import _expand_term, _match_term_targets
from svy.weighting._helpers import _build_by_array, _by_to_cols, _normalize_dict_keys

from svy.weighting.raking import _trim_constraints_satisfied
from svy.weighting.types import TrimConfig, resolve_threshold

try:
    from svy_rs._internal import trim_weights as rust_trim_weights  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_trim_weights = None

if TYPE_CHECKING:
    from svy.core.sample import Sample


def control_aux_template(
    sample: Sample,
    *,
    x: Sequence[Feature],
    by: str | Sequence[str] | None = None,
    by_na: Literal["error", "level", "drop"] = "error",
    na_label: str = "__NA__",
) -> dict[Category, Number] | dict[Category, dict[Category, Number]]:
    _, shape = build_aux_matrix(sample, x=x, by=by, by_na=by_na, na_label=na_label)
    return shape


def build_aux_matrix(
    sample: Sample,
    *,
    x: Sequence[Feature],
    by: str | Sequence[str] | None = None,
    by_na: Literal["error", "level", "drop"] = "error",
    na_label: str = "__NA__",
) -> tuple[np.ndarray, dict[Category, Number] | dict[Category, dict[Category, Number]]]:
    df: pl.DataFrame = sample.data
    where = "Sample.weighting.build_aux_matrix"

    by_n = _by_to_cols(by)

    # Validate by columns exist
    if by_n is not None:
        missing_by = [c for c in by_n if c not in df.columns]
        if missing_by:
            raise MethodError.invalid_choice(
                where=where,
                param="by",
                got=missing_by,
                allowed=list(df.columns),
                hint="All `by` columns must exist in the data.",
            )

    if not x:
        raise MethodError.not_applicable(
            where=where, method="aux builder", reason="No terms specified."
        )

    all_exprs = []
    all_labels = []

    for term in x:
        t_exprs, t_labs = _expand_term(term, df, where)
        all_exprs.extend(t_exprs)
        all_labels.extend(t_labs)

    if not all_exprs:
        raise MethodError.not_applicable(
            where=where, method="aux builder", reason="Terms produced no columns."
        )

    final_exprs = [e.alias(f"__aux_col_{i}") for i, e in enumerate(all_exprs)]

    if by_n is None:
        X_df = df.select(final_exprs)
        X = X_df.to_numpy()
        return X, {lab: np.nan for lab in all_labels}

    if by_na == "error":
        if (
            df.select([pl.col(c).is_null().any() for c in by_n])
            .select(pl.any_horizontal(pl.all()))
            .item()
        ):
            raise DimensionError(
                title="Missing values in `by` columns",
                detail=f"Nulls found in {by_n}.",
                code="BY_NA",
                where=where,
                param="by",
                hint="Use by_na='level' or 'drop', or fix data.",
            )
        by_exprs = [pl.col(c) for c in by_n]
    elif by_na == "level":
        by_exprs = [pl.col(c).fill_null(na_label) for c in by_n]
    elif by_na == "drop":
        filter_expr = pl.all_horizontal([pl.col(c).is_not_null() for c in by_n])
        df = df.filter(filter_expr)
        by_exprs = [pl.col(c) for c in by_n]
    else:
        raise MethodError.invalid_choice(
            where=where,
            param="by_na",
            got=by_na,
            allowed=["error", "level", "drop"],
        )

    combined_df = df.select(final_exprs + by_exprs)
    n_x_cols = len(final_exprs)
    X = combined_df[:, :n_x_cols].to_numpy()
    by_arr = combined_df[:, n_x_cols:].to_numpy()

    _keys_raw = [row[0] if by_arr.shape[1] == 1 else tuple(row.tolist()) for row in by_arr]
    keys: list[Category] = _keys_raw  # type: ignore[assignment]

    try:
        uniq_keys = sorted(list(set(keys)))
    except TypeError:
        uniq_keys = list(set(keys))

    inner_template = {lab: np.nan for lab in all_labels}
    return X, {k: inner_template.copy() for k in uniq_keys}


def calibrate(
    sample: Sample,
    *,
    controls: dict[Feature, Any],
    by: str | Sequence[str] | None = None,
    scale: Number | list[Number] | np.ndarray = 1.0,
    bounded: bool = False,
    wgt_name: str = "calib_wgt",
    update_design_wgts: bool = True,
    ignore_reps: bool = False,
    strict: bool = True,
    trimming: TrimConfig | None = None,
) -> Sample:
    where = "Sample.weighting.calibrate"

    if not isinstance(controls, dict) or not controls:
        raise MethodError.not_applicable(
            where=where,
            method="calibrate",
            reason="`controls` must be a non-empty dictionary.",
            param="controls",
        )

    terms: list[Feature]
    is_global = by is None

    if is_global:
        terms = list(controls.keys())
    else:
        first_domain_val = next(iter(controls.values()))
        if not isinstance(first_domain_val, dict):
            raise MethodError.invalid_type(
                where=where,
                param="controls",
                got=first_domain_val,
                expected="dict mapping domain -> targets when by= is used",
                hint="When by= is set, controls values must be dicts keyed by domain.",
            )
        terms = list(first_domain_val.keys())

    X, shape_template = build_aux_matrix(sample, x=terms, by=by)

    if is_global:
        x_labels = list(shape_template.keys())
    else:
        x_labels = list(next(iter(shape_template.values())).keys())

    final_control_arg: Any

    if is_global:
        flat_targets = []
        cursor = 0
        for term in terms:
            _, term_labs = _expand_term(term, sample.data, where)
            k = len(term_labs)
            segment_labels = x_labels[cursor : cursor + k]
            if segment_labels != term_labs:
                raise RuntimeError("Internal error: Design matrix label alignment mismatch.")
            spec = controls[term]
            vals = _match_term_targets(segment_labels, spec, str(term))
            flat_targets.extend(vals)
            cursor += k
        final_control_arg = np.array(flat_targets, dtype=float)
    else:
        final_control_arg = {}
        expected_domains = set(shape_template.keys())
        provided_domains = set(controls.keys())
        missing = expected_domains - provided_domains
        extra = provided_domains - expected_domains
        if missing or extra:
            raise MethodError.invalid_mapping_keys(
                where=where,
                param="controls",
                missing=list(missing),
                extra=list(extra),
            )

        for domain, domain_specs in controls.items():
            flat_targets = []
            cursor = 0
            for term in terms:
                if term not in domain_specs:
                    raise MethodError.invalid_mapping_keys(
                        where=where,
                        param=f"controls[{domain!r}]",
                        missing=[str(term)],
                    )
                _, term_labs = _expand_term(term, sample.data, where)
                k = len(term_labs)
                segment_labels = x_labels[cursor : cursor + k]
                spec = domain_specs[term]
                vals = _match_term_targets(segment_labels, spec, str(term))
                flat_targets.extend(vals)
                cursor += k
            final_control_arg[domain] = dict(zip(x_labels, flat_targets))

    return calibrate_matrix(
        sample,
        aux_vars=X,
        control=final_control_arg,
        by=by,
        scale=scale,
        bounded=bounded,
        wgt_name=wgt_name,
        update_design_wgts=update_design_wgts,
        labels=x_labels,
        weights_only=False,
        ignore_reps=ignore_reps,
        strict=strict,
        trimming=trimming,
    )


def _check_calibration_fit(
    new_w: np.ndarray,
    X: np.ndarray,
    totals: np.ndarray,
    tol: float = 1e-4,
) -> bool:
    """Return True if calibrated weights satisfy X'w ≈ totals within tol."""
    achieved = X.T @ new_w
    denom = np.abs(totals)
    denom = np.where(denom > 1e-10, denom, 1.0)
    max_rel_err = float(np.max(np.abs(achieved - totals) / denom))
    return max_rel_err <= tol


def calibrate_matrix(
    sample: Sample,
    *,
    aux_vars: np.ndarray,
    control: Any,
    by: str | Sequence[str] | None = None,
    scale: Number | Sequence[Number] | np.ndarray = 1.0,
    wgt_name: str = "calib_wgt",
    update_design_wgts: bool = True,
    labels: Sequence[Category] | None = None,
    weights_only: bool = False,
    bounded: bool = False,
    ignore_reps: bool = False,
    strict: bool = True,
    trimming: TrimConfig | None = None,
) -> Any:
    where = "Sample.weighting.calibrate_matrix"
    df: pl.DataFrame = sample.data
    design = sample._design

    wgt_col = getattr(sample.design, "wgt")
    if not isinstance(wgt_col, str) or wgt_col not in df.columns:
        raise MethodError.invalid_choice(
            where=where,
            param="design.wgt",
            got=wgt_col,
            allowed=df.columns,
            hint="Design must reference an existing weight column.",
        )
    w = df.get_column(wgt_col).to_numpy()

    X = np.asarray(aux_vars, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != df.height:
        raise DimensionError(
            title="Shape mismatch",
            detail=f"aux_vars has {X.shape[0]} rows; data has {df.height}.",
            code="SHAPE_MISMATCH",
            where=where,
            param="aux_vars",
            expected=f"(n,{X.shape[1]}) with n={df.height}",
            got=f"(n={X.shape[0]},{X.shape[1]})",
        )

    domain_vec = _build_by_array(df, by, where=where)

    if domain_vec is None and isinstance(control, dict) and labels is not None:
        missing_labels = [lbl for lbl in labels if lbl not in control]
        extra_keys = [k for k in control.keys() if k not in labels]
        if missing_labels or extra_keys:
            raise MethodError.invalid_mapping_keys(
                where=where,
                param="control",
                missing=missing_labels,
                extra=extra_keys,
            )
        control = np.array([float(control[lbl]) for lbl in labels])

    if isinstance(scale, (int, float)):
        scale_arr = np.full(len(w), float(scale), dtype=np.float64)
    else:
        scale_arr = np.asarray(scale, dtype=np.float64)

    # When by= is used, control must be a dict keyed by domain
    if domain_vec is not None and not isinstance(control, dict):
        raise MethodError.invalid_type(
            where=where,
            param="control",
            got=control,
            expected="dict mapping domain values to control totals when by= is used",
            hint="When by= is specified, control must be a dict keyed by domain values.",
        )
    if domain_vec is not None and isinstance(control, dict):
        control = _normalize_dict_keys(control)

    # ── Initialise variables referenced by both main calibration and the
    #    trim-calibrate cycle below.  Must be set before the if/else branch
    #    so the cycle can safely reference them regardless of which path ran.
    totals_arr: np.ndarray = np.empty(0, dtype=np.float64)
    controls_dict_main: dict[int, list[float]] = {}
    domain_indices: np.ndarray = np.empty(0, dtype=np.int64)
    unique_domains: np.ndarray = np.empty(0)
    domain_to_idx: dict[Any, int] = {}

    # Main weight: reshape to (n, 1), call Rust, extract column 0
    if domain_vec is None:
        if isinstance(control, dict):
            totals_arr = (
                np.array([float(control[lbl]) for lbl in labels], dtype=np.float64)
                if labels is not None
                else np.array(list(control.values()), dtype=np.float64)
            )
        else:
            totals_arr = np.asarray(control, dtype=np.float64)
        assert rust_calibrate is not None  # noqa: S101
        # Always pass additive=False — the Rust flag returns g-factors rather
        # than calibrated weights and must not be exposed to callers.
        new_w = rust_calibrate(w.reshape(-1, 1), X, totals_arr, scale_arr, False)[:, 0]
    else:
        unique_domains = np.unique(domain_vec)
        domain_to_idx = {d: idx for idx, d in enumerate(unique_domains)}
        domain_indices = np.array([domain_to_idx[d] for d in domain_vec], dtype=np.int64)
        for domain in unique_domains:
            domain_idx = domain_to_idx[domain]
            if isinstance(control, dict):
                domain_control = control[domain]
                if isinstance(domain_control, dict):
                    totals = (
                        [float(domain_control[lbl]) for lbl in labels]
                        if labels is not None
                        else list(domain_control.values())
                    )
                else:
                    totals = list(np.asarray(domain_control, dtype=float))
            else:
                totals = list(np.asarray(control, dtype=float))
            controls_dict_main[domain_idx] = totals
        assert rust_calibrate_by_domain is not None  # noqa: S101
        # Always pass additive=False — see note above.
        new_w = rust_calibrate_by_domain(
            w.reshape(-1, 1), X, domain_indices, controls_dict_main, scale_arr, False
        )[:, 0]

    if weights_only:
        # No cycling for weights_only mode — just check fit and return
        if strict:
            if domain_vec is None:
                if not _check_calibration_fit(new_w, X, totals_arr):
                    raise MethodError.not_applicable(
                        where=where,
                        method="calibrate",
                        reason=(
                            "Calibration did not satisfy control totals within tolerance. "
                            "The design matrix may be singular or ill-conditioned. "
                            "Pass strict=False to store the approximate solution."
                        ),
                        hint="Check for multicollinearity in aux_vars or reduce the number of calibration variables.",
                    )
            else:
                # FIX: validate fit per domain, not skipped entirely
                for domain in unique_domains:
                    mask = domain_vec == domain
                    d_idx = domain_to_idx[domain]
                    d_totals = np.array(controls_dict_main[d_idx], dtype=np.float64)
                    if not _check_calibration_fit(new_w[mask], X[mask], d_totals):
                        raise MethodError.not_applicable(
                            where=where,
                            method="calibrate",
                            reason=(
                                f"Calibration did not satisfy control totals for domain "
                                f"{domain!r} within tolerance. "
                                "The design matrix may be singular or ill-conditioned. "
                                "Pass strict=False to store the approximate solution."
                            ),
                            hint="Check for multicollinearity in aux_vars or reduce the number of calibration variables.",
                        )
        return new_w

    existing_cols = set(df.columns)

    if wgt_name in existing_cols:
        raise MethodError.not_applicable(
            where=where,
            method="calibrate_matrix",
            reason=f"Column '{wgt_name}' already exists. Choose a different wgt_name.",
        )

    df = df.with_columns(pl.Series(name=wgt_name, values=new_w))

    if update_design_wgts:
        sample._design = sample._design.update(wgt=wgt_name)

    if not ignore_reps and design.rep_wgts is not None:
        rep_cols = design.rep_wgts.columns

        if rep_cols:
            wgts_arr = df.select(rep_cols).to_numpy()

            if isinstance(scale, (int, float)):
                scale_arr = np.full(len(w), float(scale), dtype=np.float64)
            else:
                scale_arr = np.asarray(scale, dtype=np.float64)

            if domain_vec is None:
                if isinstance(control, dict):
                    if labels is not None:
                        totals_arr = np.array(
                            [float(control[lbl]) for lbl in labels],  # type: ignore[index]
                            dtype=np.float64,
                        )
                    else:
                        totals_arr = np.array(list(control.values()), dtype=np.float64)
                else:
                    totals_arr = np.asarray(control, dtype=np.float64)

                n_reps = len(rep_cols)
                if n_reps >= 10:
                    assert rust_calibrate_parallel is not None  # noqa: S101
                    calib_reps = rust_calibrate_parallel(wgts_arr, X, totals_arr, scale_arr)
                else:
                    assert rust_calibrate is not None  # noqa: S101
                    # Always pass additive=False — see module docstring.
                    calib_reps = rust_calibrate(wgts_arr, X, totals_arr, scale_arr, False)
            else:
                # domain_indices and controls_dict_main are already populated above
                assert rust_calibrate_by_domain is not None  # noqa: S101
                # Always pass additive=False — see module docstring.
                calib_reps = rust_calibrate_by_domain(
                    wgts_arr, X, domain_indices, controls_dict_main, scale_arr, False
                )

            n_reps = len(rep_cols)
            new_rep_names = [f"{wgt_name}{i}" for i in range(1, n_reps + 1)]

            wgts_df = pl.DataFrame(calib_reps, schema=new_rep_names)
            df = df.hstack(wgts_df)

            if update_design_wgts:
                sample._design = sample._design.update_rep_weights(
                    method=design.rep_wgts.method,
                    prefix=wgt_name,
                    n_reps=n_reps,
                    fay_coef=design.rep_wgts.fay_coef,
                    df=design.rep_wgts.df,
                )

    sample._data = df

    # ── Trim-calibrate cycle ──────────────────────────────────────────────
    if trimming is not None:
        # Resolve thresholds once from the initial calibrated weights
        _w_pos = new_w[new_w > 0].astype(np.float64)
        _upper_val = (
            resolve_threshold(trimming.upper, _w_pos) if trimming.upper is not None else None
        )
        _lower_val = (
            resolve_threshold(trimming.lower, _w_pos) if trimming.lower is not None else None
        )

        assert rust_trim_weights is not None  # noqa: S101

        _current_w = new_w.copy()
        _calib_converged = True  # already checked above if strict
        _trim_ok = False
        _scale_arr_cycle = (
            np.full(len(w), float(scale), dtype=np.float64)
            if isinstance(scale, (int, float))
            else np.asarray(scale, dtype=np.float64)
        )
        _n_cycles = trimming.max_iter

        for _cycle in range(_n_cycles):
            # Calibrate step — restore control totals
            if domain_vec is None:
                _cal_result = rust_calibrate(
                    _current_w.reshape(-1, 1),
                    X,
                    totals_arr,
                    _scale_arr_cycle,
                    False,  # additive=False always — see module docstring
                )
            else:
                # domain_indices and controls_dict_main are populated above
                _cal_result = rust_calibrate_by_domain(
                    _current_w.reshape(-1, 1),
                    X,
                    domain_indices,
                    controls_dict_main,
                    _scale_arr_cycle,
                    False,  # additive=False always — see module docstring
                )
            _current_w = _cal_result[:, 0]

            # FIX: check calibration fit for domain case too
            if domain_vec is None:
                _calib_converged = _check_calibration_fit(_current_w, X, totals_arr)
            else:
                _calib_converged = all(
                    _check_calibration_fit(
                        _current_w[domain_vec == d],
                        X[domain_vec == d],
                        np.array(controls_dict_main[domain_to_idx[d]], dtype=np.float64),
                    )
                    for d in unique_domains
                )

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
                # Final calibrate to restore totals after last trim
                if domain_vec is None:
                    _final = rust_calibrate(
                        _current_w.reshape(-1, 1),
                        X,
                        totals_arr,
                        _scale_arr_cycle,
                        False,  # additive=False always — see module docstring
                    )
                else:
                    _final = rust_calibrate_by_domain(
                        _current_w.reshape(-1, 1),
                        X,
                        domain_indices,
                        controls_dict_main,
                        _scale_arr_cycle,
                        False,  # additive=False always — see module docstring
                    )
                _current_w = _final[:, 0]
                # Re-check trim and fit after final calibrate
                _trim_ok = _trim_constraints_satisfied(_current_w, _upper_val, _lower_val, 1e-4)
                if domain_vec is None:
                    _calib_converged = _check_calibration_fit(_current_w, X, totals_arr)
                else:
                    _calib_converged = all(
                        _check_calibration_fit(
                            _current_w[domain_vec == d],
                            X[domain_vec == d],
                            np.array(controls_dict_main[domain_to_idx[d]], dtype=np.float64),
                        )
                        for d in unique_domains
                    )
                break

        if strict and not (_calib_converged and _trim_ok):
            raise MethodError.not_applicable(
                where=where,
                method="calibrate",
                reason=(
                    f"Trim-calibrate cycle did not converge after {trimming.max_iter} cycles. "
                    "The design has NOT been modified. "
                    "Pass strict=False to store partial results."
                ),
                hint="Increase max_iter, relax tol, or use a less restrictive TrimConfig.",
            )

        # Update the stored weight with the final cycled result
        new_w = _current_w
        # FIX: was wgt_col_name (NameError) — correct variable is wgt_name
        df = df.with_columns(pl.Series(name=wgt_name, values=new_w))

    sample._data = df
    return sample
