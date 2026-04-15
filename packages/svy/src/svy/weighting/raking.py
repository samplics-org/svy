# src/svy/weighting/raking.py
"""
Raking (iterative proportional fitting) weight adjustment.

rake() and controls_margins_template() take a Sample and return a Sample
(for chaining). The Weighting class in base.py delegates to these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import polars as pl


try:
    from svy_rs._internal import rake as rust_rake  # type: ignore[import-untyped]
    from svy_rs._internal import trim_weights as rust_trim_weights  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_rake = None
    rust_trim_weights = None

from svy.core.types import Category, ControlsType
from svy.errors import DimensionError, MethodError
from svy.weighting._helpers import _num_sort_key_label
from svy.weighting.types import TrimConfig, resolve_threshold


if TYPE_CHECKING:
    from svy.core.sample import Sample


# ---------------------------------------------------------------------------
# Helpers (previously in engine)
# ---------------------------------------------------------------------------


def _normalize_controls_like(x: ControlsType | None) -> ControlsType | None:
    """Return x unchanged if it's a non-empty dict, else None."""
    if x is None:
        return None
    if not isinstance(x, dict) or len(x) == 0:
        return None
    return x


def _calculate_controls_from_factors(
    wgts: np.ndarray,
    margins: dict[str, np.ndarray],
    factors: ControlsType,
) -> ControlsType:
    """Convert per-category factors to absolute control totals.

    For each margin column and each category within it:
        control[col][cat] = factor[col][cat] * sum(wgts where col == cat)
    """
    control: ControlsType = {}
    for col, factor_dict in factors.items():
        col_arr = margins[col]
        control[col] = {}
        for cat, factor in factor_dict.items():
            mask = col_arr == cat
            stratum_sum = float(wgts[mask].sum())
            control[col][cat] = float(factor) * stratum_sum  # type: ignore[index]
    return control


def _trim_constraints_satisfied(
    w: np.ndarray,
    upper_val: float | None,
    lower_val: float | None,
    tol: float,
) -> bool:
    """Return True if no weight violates the trim thresholds within tol.

    Checks absolute threshold bounds: convergence means the final weights
    don't exceed upper_val * (1 + tol) or fall below lower_val * (1 - tol).
    This is relative to the threshold itself, not to weight changes — so
    tol=1e-4 means 'within 0.01% of the cap', regardless of weight scale.
    """
    if upper_val is not None and np.any(w > upper_val * (1.0 + tol)):
        return False
    if lower_val is not None and np.any(w[w > 0] < lower_val * (1.0 - tol)):
        return False
    return True


def _build_margin_arrays(
    rake_cols: list[str],
    control_final: ControlsType,
    processed: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build margin_indices and margin_targets arrays for Rust rake."""
    margin_indices = []
    margin_targets = []
    for col in rake_cols:
        cats_sorted = sorted(control_final[col].keys(), key=_num_sort_key_label)  # type: ignore[union-attr]
        cat_to_idx = {cat: idx for idx, cat in enumerate(cats_sorted)}
        indices = np.array([cat_to_idx[val] for val in processed[col]], dtype=np.int64)
        targets = np.array(
            [float(control_final[col][cat]) for cat in cats_sorted],  # type: ignore[index]
            dtype=np.float64,
        )
        margin_indices.append(indices)
        margin_targets.append(targets)
    return margin_indices, margin_targets


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def controls_margins_template(
    sample: Sample,
    *,
    margins: Mapping[str, str],
    cat_na: str = "level",
    na_label: str = "__NA__",
) -> dict[str, dict[Category, float]]:
    df: pl.DataFrame = sample.data
    where = "Sample.weighting.controls_margins_template"

    for mname, col in margins.items():
        if not isinstance(mname, str) or not isinstance(col, str):
            raise MethodError.invalid_type(
                where=where,
                param="margins",
                got=(mname, col),
                expected="dict[str, str]",
                hint="Both margin keys and values must be strings.",
            )
        if col not in df.columns:
            raise MethodError.invalid_choice(
                where=where,
                param=f"margins[{mname!r}]",
                got=col,
                allowed=list(df.columns),
                hint="Check that the margin column exists in the data.",
            )

    result: dict[str, dict[Category, float]] = {}

    for mname, col in margins.items():
        s = df.get_column(col)

        if cat_na not in ("error", "level"):
            raise MethodError.invalid_choice(
                where=where,
                param="cat_na",
                got=cat_na,
                allowed=["error", "level"],
            )

        if cat_na == "error":
            if s.is_null().any():
                raise DimensionError(
                    title="Missing values in margin column",
                    detail=f"Nulls found in {col!r}. Choose cat_na='level' or fix data.",
                    code="MARGIN_NA",
                    where=where,
                    param=col,
                    hint="Use cat_na='level' to include a missing category.",
                )
            s_norm = s.cast(pl.Utf8)
        else:
            s_norm = s.cast(pl.Utf8).fill_null(na_label)

        cats = pl.Series("__cats__", s_norm.unique().to_list(), dtype=pl.Utf8).to_list()
        cats_sorted = sorted(cats, key=_num_sort_key_label)
        result[mname] = {lab: np.nan for lab in cats_sorted}

    return result


def _max_margin_error(
    w: np.ndarray,
    margin_indices: list[np.ndarray],
    margin_targets: list[np.ndarray],
) -> float:
    """Return the maximum relative margin error across all margins and groups."""
    max_err = 0.0
    for indices, targets in zip(margin_indices, margin_targets):
        n_groups = int(indices.max()) + 1
        for g in range(n_groups):
            mask = indices == g
            current = float(w[mask].sum())
            target = float(targets[g])
            if target > 1e-10:
                err = abs(current - target) / target
                if err > max_err:
                    max_err = err
    return max_err


def _check_and_warn_convergence(
    w: np.ndarray,
    margin_indices: list[np.ndarray],
    margin_targets: list[np.ndarray],
    tol: float,
    max_iter: int,
) -> bool:
    """Return True if converged, False if any margin is unsatisfied.
    Prints a warning on non-convergence.
    """
    if _max_margin_error(w, margin_indices, margin_targets) <= tol:
        return True
    print(f"Warning: Raking did not converge after {max_iter} iterations")
    return False


def rake(
    sample: Sample,
    *,
    controls: ControlsType | None = None,
    factors: ControlsType | None = None,
    wgt_name: str = "rk_wgt",
    ignore_reps: bool = False,
    ll_bound: float | None = None,
    up_bound: float | None = None,
    tol: float = 1e-4,
    max_iter: int = 100,
    display_iter: bool = False,
    update_design_wgts: bool = True,
    strict: bool = True,
    trimming: TrimConfig | None = None,
) -> Sample:
    where = "Sample.weighting.rake"
    df = sample._data
    design = sample._design

    if design.wgt is None:
        raise MethodError.not_applicable(
            where=where,
            method="rake",
            reason="Sample weight is None. Set design.wgt before calling rake().",
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
            method="rake",
            reason=f"Column '{wgt_name}' already exists. Choose a different wgt_name.",
        )

    controls_norm: ControlsType | None = _normalize_controls_like(x=controls)
    factors_norm: ControlsType | None = _normalize_controls_like(x=factors)

    if controls_norm is None and factors_norm is None:
        raise MethodError.not_applicable(
            where=where,
            method="rake",
            reason="Either controls= or factors= must be specified.",
        )
    if controls_norm is not None and factors_norm is not None:
        raise MethodError.not_applicable(
            where=where,
            method="rake",
            reason="Provide exactly one of controls= or factors=, not both.",
        )

    if ll_bound is not None and up_bound is not None and ll_bound > up_bound:
        raise MethodError.invalid_range(
            where=where,
            param="ll_bound",
            got=ll_bound,
            hint="ll_bound must be less than or equal to up_bound.",
        )

    rake_cols = (
        list(controls_norm.keys()) if controls_norm is not None else list(factors_norm.keys())  # type: ignore[union-attr]
    )
    if not rake_cols:
        raise MethodError.not_applicable(
            where=where,
            method="rake",
            reason="No raking columns provided in controls/factors keys.",
        )

    processed: dict[str, np.ndarray] = {}
    w0 = df.get_column(wgt).to_numpy().astype(np.float64)

    for col in rake_cols:
        if not isinstance(col, str) or col not in df.columns:
            raise MethodError.invalid_choice(
                where=where,
                param="controls/factors key",
                got=col,
                allowed=list(df.columns),
                hint="All raking column names must exist in the data.",
            )
        s = df.get_column(col)
        if s.len() != w0.size:
            raise DimensionError(
                title="Raking column length mismatch",
                detail=f"Column {col!r} has different length than the weight array.",
                code="LENGTH_MISMATCH",
                where=where,
                param=col,
            )
        if s.null_count() > 0:
            raise DimensionError(
                title="Null values in raking column",
                detail=f"Column {col!r} contains null values. Raking requires complete data.",
                code="NULL_VALUES",
                where=where,
                param=col,
                hint="Drop or impute missing values before raking.",
            )
        processed[col] = s.to_numpy()

    control_final: ControlsType = controls_norm or _calculate_controls_from_factors(
        wgts=w0,
        margins=processed,
        factors=cast(ControlsType, factors_norm),
    )

    missing = [m for m in processed if m not in control_final]
    extra = [m for m in control_final if m not in processed]
    if missing or extra:
        raise MethodError.invalid_mapping_keys(
            where=where,
            param="controls",
            missing=missing,
            extra=extra,
        )

    for col_name, totals in control_final.items():
        if not isinstance(totals, Mapping) or not totals:
            raise MethodError.invalid_type(
                where=where,
                param=f"controls[{col_name!r}]",
                got=totals,
                expected="non-empty dict mapping category -> total",
            )
        vals = np.array(list(totals.values()), dtype=float)
        if not np.all(np.isfinite(vals)) or np.any(vals < 0):
            raise DimensionError(
                title="Invalid control totals",
                detail=f"Control totals for {col_name!r} must be finite and non-negative.",
                code="INVALID_CONTROL_TOTALS",
                where=where,
                param=f"controls[{col_name!r}]",
            )
        if np.all(vals == 0):
            raise DimensionError(
                title="All-zero control totals",
                detail=f"All control totals for {col_name!r} are zero, which is not allowed.",
                code="ZERO_CONTROL_TOTALS",
                where=where,
                param=f"controls[{col_name!r}]",
            )

    # Build margin arrays once — reused across all cycles
    margin_indices, margin_targets = _build_margin_arrays(rake_cols, control_final, processed)

    assert rust_rake is not None  # noqa: S101

    # ── Trim-rake cycle ───────────────────────────────────────────────────
    # When trimming=None: single rake pass (max_iter=1 cycle, no trim step).
    # When trimming is set: iterate up to max_iter cycles:
    #   1. Rake current weights to convergence (up to max_iter IPF steps each)
    #   2. Trim — if no weights changed (within TrimConfig.tol), both
    #      constraints are satisfied and we stop early.
    # Final step is always rake so margins are satisfied.
    # Replicates are raked once with the final main-weight cycle result.

    n_cycles = max_iter if trimming is not None else 1
    current_w = w0.copy()
    rake_converged = False
    trim_unchanged = trimming is None  # trivially true when no trimming

    for cycle in range(n_cycles):
        # ── Rake step ────────────────────────────────────────────────────
        try:
            raked_result = rust_rake(
                current_w.reshape(-1, 1),
                margin_indices,
                margin_targets,
                ll_bound,
                up_bound,
                tol,
                max_iter,
            )
        except ValueError as e:
            msg = str(e)
            if "exceeded weight bounds" in msg or "Raking exceeded" in msg:
                raise ValueError(
                    "Raking failed: Weight ratios exceeded specified bounds."
                ) from None
            raise

        raked_w = raked_result[:, 0]
        rake_converged = _check_and_warn_convergence(
            raked_w, margin_indices, margin_targets, tol, max_iter
        )

        if trimming is None:
            if display_iter:
                margin_err = _max_margin_error(raked_w, margin_indices, margin_targets)
                status = "converged" if rake_converged else "not converged"
                print(f"  Raking: max margin error = {margin_err:.2e}  [{status}]")
            current_w = raked_w
            break

        if display_iter:
            margin_err = _max_margin_error(raked_w, margin_indices, margin_targets)
            rake_status = "✓" if rake_converged else "✗"

        # ── Trim step ────────────────────────────────────────────────────
        w_pos = raked_w[raked_w > 0].astype(np.float64)
        upper_val = (
            resolve_threshold(trimming.upper, w_pos) if trimming.upper is not None else None
        )
        lower_val = (
            resolve_threshold(trimming.lower, w_pos) if trimming.lower is not None else None
        )

        (trimmed_w, *_) = rust_trim_weights(
            raked_w,
            upper_val,
            lower_val,
            trimming.redistribute,
            trimming.max_iter,
            trimming.tol,
        )

        # Check if trim constraints satisfied: no weight violates threshold beyond tol
        # Uses absolute threshold comparison — more meaningful than relative weight change
        trim_unchanged = _trim_constraints_satisfied(trimmed_w, upper_val, lower_val, tol)

        current_w = trimmed_w
        last_trimmed_w = trimmed_w  # saved for post-final-rake trim check

        if display_iter:
            max_w = float(np.max(np.abs(raked_w))) if raked_w.size > 0 else 1.0
            max_w = max_w if max_w > 1e-10 else 1.0
            trim_err = float(np.max(np.abs(trimmed_w - raked_w))) / max_w
            trim_status = "✓" if trim_unchanged else "✗"
            print(
                f"Cycle {cycle + 1:3d} | "
                f"rake margin err = {margin_err:.2e} {rake_status} | "
                f"trim weight change = {trim_err:.2e} {trim_status}"
            )

        if rake_converged and trim_unchanged:
            # Final rake to restore margins after last trim
            try:
                final_result = rust_rake(
                    current_w.reshape(-1, 1),
                    margin_indices,
                    margin_targets,
                    ll_bound,
                    up_bound,
                    tol,
                    max_iter,
                )
                current_w = final_result[:, 0]
                rake_converged = _check_and_warn_convergence(
                    current_w, margin_indices, margin_targets, tol, max_iter
                )
                # Re-check trim after final rake — rake could push weights back above threshold
                trim_unchanged = _trim_constraints_satisfied(current_w, upper_val, lower_val, tol)
            except ValueError as e:
                msg = str(e)
                if "exceeded weight bounds" in msg or "Raking exceeded" in msg:
                    raise ValueError(
                        "Raking failed: Weight ratios exceeded specified bounds."
                    ) from None
                raise
            if display_iter:
                final_margin_err = _max_margin_error(current_w, margin_indices, margin_targets)
                both_ok = rake_converged and trim_unchanged
                print(
                    f"  {'Converged' if both_ok else 'Warning: final rake shifted weights'}: "
                    f"margin err = {final_margin_err:.2e}, "
                    f"trim change = {float(np.max(np.abs(current_w - last_trimmed_w))) / max_w:.2e}"
                )
            break
    else:
        # Loop exhausted without clean convergence — do a final rake
        if trimming is not None:
            try:
                final_result = rust_rake(
                    current_w.reshape(-1, 1),
                    margin_indices,
                    margin_targets,
                    ll_bound,
                    up_bound,
                    tol,
                    max_iter,
                )
                current_w = final_result[:, 0]
                rake_converged = _check_and_warn_convergence(
                    current_w, margin_indices, margin_targets, tol, max_iter
                )
            except ValueError as e:
                msg = str(e)
                if "exceeded weight bounds" in msg or "Raking exceeded" in msg:
                    raise ValueError(
                        "Raking failed: Weight ratios exceeded specified bounds."
                    ) from None
                raise

    raked_w = current_w

    # ── Convergence guard ─────────────────────────────────────────────────
    if (not rake_converged or (trimming is not None and not trim_unchanged)) and strict:
        reason = (
            f"Trim-rake cycle did not converge after {max_iter} cycles. "
            if trimming is not None
            else f"Raking did not converge after {max_iter} iterations. "
        )
        raise MethodError.not_applicable(
            where=where,
            method="rake",
            reason=(
                reason + "The design has NOT been modified. "
                "Increase max_iter, relax tol, or pass strict=False to store partial weights."
            ),
            hint="Try increasing max_iter or relaxing tol.",
        )

    df = df.with_columns(pl.Series(name=wgt_name, values=raked_w))

    if update_design_wgts:
        sample._design = sample._design.update(wgt=wgt_name)

    if not ignore_reps and design.rep_wgts is not None:
        rep_cols = design.rep_wgts.columns

        if rep_cols:
            n_reps = len(rep_cols)
            wgts_arr = df.select(rep_cols).to_numpy()

            # Replicates: one rake pass with the final converged main weights
            # as starting point. Cycling replicates is not standard practice.
            raked_reps = rust_rake(
                wgts_arr,
                margin_indices,
                margin_targets,
                ll_bound,
                up_bound,
                tol,
                max_iter,
            )

            new_rep_names = [f"{wgt_name}{i}" for i in range(1, n_reps + 1)]
            wgts_df = pl.DataFrame(raked_reps, schema=new_rep_names)
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
    return sample
