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
    from svy_rs._internal import (
        calibrate_by_domain as rust_calibrate_by_domain,  # type: ignore[import-untyped]
    )
    from svy_rs._internal import (
        calibrate_parallel as rust_calibrate_parallel,  # type: ignore[import-untyped]
    )
except ImportError:  # pragma: no cover
    rust_calibrate = None
    rust_calibrate_by_domain = None
    rust_calibrate_parallel = None

from svy.core.terms import Feature
from svy.core.types import Category, Number
from svy.core.warnings import Severity, WarnCode
from svy.errors import DimensionError, MethodError
from svy.weighting._calibration_utils import _expand_term, _match_term_targets
from svy.weighting._helpers import _build_by_array, _by_to_cols, _normalize_dict_keys
from svy.weighting.raking import _trim_constraints_satisfied
from svy.weighting.trimming import _build_domain_array
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

    # Ordered per-term label lists. Targets are matched per term and kept as
    # ordered flat lists end-to-end — routing them through label-keyed dicts
    # collapsed duplicate level labels across terms (e.g. two Cats sharing
    # numeric codes), breaking the column alignment.
    term_label_lists: list[tuple[Feature, list[Category]]] = []
    x_labels: list[Category] = []
    for term in terms:
        _, term_labs = _expand_term(term, sample.data, where)
        term_label_lists.append((term, term_labs))
        x_labels.extend(term_labs)

    if X.shape[1] != len(x_labels):
        raise RuntimeError("Internal error: Design matrix label alignment mismatch.")

    final_control_arg: Any

    if is_global:
        flat_targets = []
        for term, term_labs in term_label_lists:
            vals = _match_term_targets(term_labs, controls[term], str(term))
            flat_targets.extend(vals)
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
            for term, term_labs in term_label_lists:
                if term not in domain_specs:
                    raise MethodError.invalid_mapping_keys(
                        where=where,
                        param=f"controls[{domain!r}]",
                        missing=[str(term)],
                    )
                vals = _match_term_targets(term_labs, domain_specs[term], str(term))
                flat_targets.extend(vals)
            # Ordered list (not a label-keyed dict): aligned with X columns
            final_control_arg[domain] = flat_targets

    return calibrate_matrix(
        sample,
        aux_vars=X,
        control=final_control_arg,
        by=by,
        scale=scale,
        bounded=bounded,
        wgt_name=wgt_name,
        update_design_wgts=update_design_wgts,
        labels=None,
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

    if bounded:
        # Accepting-and-ignoring this flag previously produced bit-identical
        # results to bounded=False.
        raise NotImplementedError(
            "bounded calibration (distance-bounded g-weights, as in R survey's "
            "calibrate(bounds=)) is not implemented yet. Use trimming= to "
            "constrain calibrated weights, or leave bounded=False."
        )

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
        unique_domains, domain_indices_inv = np.unique(domain_vec, return_inverse=True)
        domain_indices = domain_indices_inv.astype(np.int64)
        domain_to_idx = {d: idx for idx, d in enumerate(unique_domains)}
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

    # ── Trim-calibrate cycle ──────────────────────────────────────────────
    # Runs on arrays BEFORE anything is written to the sample, so a strict
    # failure genuinely leaves the data and design untouched.
    if trimming is not None:
        # Resolve trim domains (honor TrimConfig.by / min_cell_size, matching
        # the contract adjust()/trim() already implement)
        if trimming.by is not None:
            t_by_cols = (
                [trimming.by] if isinstance(trimming.by, str) else list(trimming.by)
            )
            missing_by = [c for c in t_by_cols if c not in df.columns]
            if missing_by:
                raise MethodError.invalid_choice(
                    where=where,
                    param="trimming.by",
                    got=missing_by,
                    allowed=list(df.columns),
                    hint="All trimming by= columns must exist in the data.",
                )
            _trim_dom_arr = _build_domain_array(df, t_by_cols)
            _group_masks = [(str(d), _trim_dom_arr == d) for d in np.unique(_trim_dom_arr)]
        else:
            _group_masks = [("(global)", np.ones(len(new_w), dtype=bool))]

        # Thresholds resolved once per domain from the initial calibrated
        # weights; domains below min_cell_size are skipped with a warning.
        trim_groups: list[tuple[np.ndarray, float | None, float | None]] = []
        for _label, _mask in _group_masks:
            _w_pos = new_w[_mask]
            _w_pos = _w_pos[_w_pos > 0].astype(np.float64)
            if len(_w_pos) < trimming.min_cell_size:
                sample.warn(
                    code=WarnCode.DOMAIN_SKIPPED,
                    title="Domain skipped — cell too small",
                    detail=(
                        f"Trim domain {_label!r} has {len(_w_pos)} positive-weight "
                        f"unit(s), below min_cell_size={trimming.min_cell_size}. "
                        "Trimming skipped."
                    ),
                    where=where,
                    level=Severity.WARNING,
                )
                continue
            _up = (
                resolve_threshold(trimming.upper, _w_pos)
                if trimming.upper is not None
                else None
            )
            _lo = (
                resolve_threshold(trimming.lower, _w_pos)
                if trimming.lower is not None
                else None
            )
            trim_groups.append((_mask, _up, _lo))

        assert rust_trim_weights is not None  # noqa: S101

        _current_w = new_w.copy()
        _calib_converged = True  # already checked above if strict
        _trim_ok = True  # nothing to trim when every domain was skipped
        _scale_arr_cycle = (
            np.full(len(w), float(scale), dtype=np.float64)
            if isinstance(scale, (int, float))
            else np.asarray(scale, dtype=np.float64)
        )

        def _cycle_calibrate(w_in: np.ndarray) -> np.ndarray:
            if domain_vec is None:
                return rust_calibrate(
                    w_in.reshape(-1, 1),
                    X,
                    totals_arr,
                    _scale_arr_cycle,
                    False,  # additive=False always — see module docstring
                )[:, 0]
            return rust_calibrate_by_domain(
                w_in.reshape(-1, 1),
                X,
                domain_indices,
                controls_dict_main,
                _scale_arr_cycle,
                False,  # additive=False always — see module docstring
            )[:, 0]

        def _cycle_fit_ok(w_in: np.ndarray) -> bool:
            if domain_vec is None:
                return _check_calibration_fit(w_in, X, totals_arr)
            return all(
                _check_calibration_fit(
                    w_in[domain_vec == d],
                    X[domain_vec == d],
                    np.array(controls_dict_main[domain_to_idx[d]], dtype=np.float64),
                )
                for d in unique_domains
            )

        def _cycle_trim_ok(w_in: np.ndarray) -> bool:
            return all(
                _trim_constraints_satisfied(w_in[m], u, lo, 1e-4) for m, u, lo in trim_groups
            )

        if trim_groups:
            _trim_ok = False
            for _cycle in range(trimming.max_iter):
                # Calibrate step — restore control totals
                _current_w = _cycle_calibrate(_current_w)
                _calib_converged = _cycle_fit_ok(_current_w)

                # Trim step — per domain with its own thresholds
                for _mask, _up, _lo in trim_groups:
                    (_trimmed_w, *_) = rust_trim_weights(
                        _current_w[_mask],
                        _up,
                        _lo,
                        trimming.redistribute,
                        trimming.max_iter,
                        trimming.tol,
                    )
                    _current_w[_mask] = _trimmed_w
                _trim_ok = _cycle_trim_ok(_current_w)

                if _trim_ok:
                    # Final calibrate to restore totals after last trim
                    _current_w = _cycle_calibrate(_current_w)
                    # Re-check trim and fit after final calibrate
                    _trim_ok = _cycle_trim_ok(_current_w)
                    _calib_converged = _cycle_fit_ok(_current_w)
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

        new_w = _current_w

    # ── Write results (only after the strict guard above) ────────────────
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
    return sample
