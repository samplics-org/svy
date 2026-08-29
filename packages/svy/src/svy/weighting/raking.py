# src/svy/weighting/raking.py
"""
Raking (iterative proportional fitting) weight adjustment.

rake() and controls_margins_template() take a Sample and return a Sample
(for chaining). The Weighting class in base.py delegates to these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, cast

import msgspec
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
from svy.weighting._engine import _where_mask
from svy.weighting._helpers import _num_sort_key_label
from svy.weighting.types import TrimConfig, resolve_threshold


if TYPE_CHECKING:
    from svy.core.sample import Sample
    from svy.core.types import WhereArg


# ---------------------------------------------------------------------------
# Helpers (previously in engine)
# ---------------------------------------------------------------------------


def _rake_or_raise(*args, where: str = "Sample.weighting.rake"):
    """Call the Rust raking kernel, translating bounds violations into a
    typed svy error. Bounds violations are errors on every exit path,
    converged or not."""
    assert rust_rake is not None  # noqa: S101
    try:
        return rust_rake(*args)
    except ValueError as e:
        msg = str(e)
        if "exceeded weight bounds" in msg or "Raking exceeded" in msg:
            raise MethodError.not_applicable(
                where=where,
                method="rake",
                reason="Weight ratios exceeded the specified bounds (ll_bound/up_bound).",
                param="ll_bound/up_bound",
                hint="Widen the bounds, relax the margins, or increase max_iter.",
            ) from None
        raise


def _normalize_controls_like(x: ControlsType | None) -> ControlsType | None:
    """Return x unchanged if it's a non-empty dict, else None."""
    if x is None:
        return None
    if not isinstance(x, dict) or len(x) == 0:
        return None
    return x


def _shares_to_controls(
    wgts: np.ndarray,
    shares: ControlsType,
) -> ControlsType:
    """Convert per-margin shares to absolute control totals.

    Shares are marginal proportions, normalized within each margin, so every
    margin resolves against the same grand total:
        control[col][cat] = share[col][cat] / sum(share[col]) * sum(wgts)

    That makes cross-margin consistency structural rather than something the
    caller has to get right -- margins that disagree on a grand total are the
    usual reason IPF fails to converge.
    """
    grand_total = float(wgts.sum())
    control: ControlsType = {}
    for col, share_dict in shares.items():
        total = float(sum(float(v) for v in share_dict.values()))
        if total <= 0:
            raise DimensionError(
                title="Invalid shares",
                detail=f"Shares for {col!r} must include at least one positive value.",
                code="INVALID_SHARES",
                where="Sample.weighting.rake",
                param=f"shares[{col!r}]",
            )
        control[col] = {cat: float(v) / total * grand_total for cat, v in share_dict.items()}
    return control


def _check_margins_agree(control: ControlsType, *, where: str) -> None:
    """Every margin must describe the same population.

    Raking cannot satisfy margins whose totals differ; without this it would
    silently iterate to max_iter and return whatever it reached.
    """
    totals = {
        col: float(sum(float(v) for v in cats.values()))  # type: ignore[union-attr]
        for col, cats in control.items()
    }
    if len(totals) < 2:
        return
    lo, hi = min(totals.values()), max(totals.values())
    if hi > 0 and (hi - lo) / hi > 1e-6:
        raise MethodError.not_applicable(
            where=where,
            method="rake",
            reason=(
                "Margins disagree on the population total: "
                + ", ".join(f"{c}={t:,.4g}" for c, t in sorted(totals.items()))
            ),
            param="controls",
            hint=(
                "Every margin must sum to the same total. Pass shares= to have "
                "them normalized against one grand total automatically."
            ),
        )


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
    shares: ControlsType | None = None,
    where: WhereArg = None,
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
    ctx = "Sample.weighting.rake"
    df = sample._data
    design = sample._design

    if design.wgt is None:
        raise MethodError.not_applicable(
            where=ctx,
            method="rake",
            reason="Sample weight is None. Set design.wgt before calling rake().",
        )
    wgt = design.wgt
    if wgt not in df.columns:
        raise MethodError.invalid_choice(
            where=ctx,
            param="design.wgt",
            got=wgt,
            allowed=list(df.columns),
            hint="Check that the weight column exists in the data.",
        )

    existing_cols = set(df.columns)
    if wgt_name in existing_cols:
        raise MethodError.not_applicable(
            where=ctx,
            method="rake",
            reason=f"Column '{wgt_name}' already exists. Choose a different wgt_name.",
        )

    controls_norm: ControlsType | None = _normalize_controls_like(x=controls)
    shares_norm: ControlsType | None = _normalize_controls_like(x=shares)

    if controls_norm is None and shares_norm is None:
        raise MethodError.not_applicable(
            where=ctx,
            method="rake",
            reason="Either controls= or shares= must be specified.",
        )
    if controls_norm is not None and shares_norm is not None:
        raise MethodError.not_applicable(
            where=ctx,
            method="rake",
            reason="Provide exactly one of controls= or shares=, not both.",
        )

    if ll_bound is not None and up_bound is not None and ll_bound > up_bound:
        raise MethodError.invalid_range(
            where=ctx,
            param="ll_bound",
            got=ll_bound,
            hint="ll_bound must be less than or equal to up_bound.",
        )

    rake_cols = (
        list(controls_norm.keys()) if controls_norm is not None else list(shares_norm.keys())  # type: ignore[union-attr]
    )
    if not rake_cols:
        raise MethodError.not_applicable(
            where=ctx,
            method="rake",
            reason="No raking columns provided in controls/shares keys.",
        )

    processed: dict[str, np.ndarray] = {}
    w0 = df.get_column(wgt).to_numpy().astype(np.float64)

    for col in rake_cols:
        if not isinstance(col, str) or col not in df.columns:
            raise MethodError.invalid_choice(
                where=ctx,
                param="controls/shares key",
                got=col,
                allowed=list(df.columns),
                hint="All raking column names must exist in the data.",
            )

    margin_df = df.select(rake_cols)

    null_counts = margin_df.null_count().row(0)
    for col, n_null in zip(rake_cols, null_counts):
        s = df.get_column(col)
        if s.len() != w0.size:
            raise DimensionError(
                title="Raking column length mismatch",
                detail=f"Column {col!r} has different length than the weight array.",
                code="LENGTH_MISMATCH",
                where=ctx,
                param=col,
            )
        if n_null > 0:
            raise DimensionError(
                title="Null values in raking column",
                detail=f"Column {col!r} contains null values. Raking requires complete data.",
                code="NULL_VALUES",
                where=ctx,
                param=col,
                hint="Drop or impute missing values before raking.",
            )

    margin_np = margin_df.to_numpy()
    for i, col in enumerate(rake_cols):
        processed[col] = margin_np[:, i]

    # `where` scopes the adjustment: only in-scope rows are raked, and the rest
    # keep their weight. The IPF then runs on the subset, so the margins the
    # caller supplies describe the scoped population and nothing else.
    scope = _where_mask(df, where, where=ctx)
    scope_idx = None
    if scope is not None:
        scope_idx = np.flatnonzero(scope)
        if scope_idx.size == 0:
            raise MethodError.not_applicable(
                where=ctx, method="rake", reason="No rows are in scope for this adjustment"
            )
        w_full = w0
        w0 = w0[scope_idx]
        processed = {c: a[scope_idx] for c, a in processed.items()}

    control_final: ControlsType = controls_norm or _shares_to_controls(
        wgts=w0,
        shares=cast(ControlsType, shares_norm),
    )

    missing = [m for m in processed if m not in control_final]
    extra = [m for m in control_final if m not in processed]
    if missing or extra:
        raise MethodError.invalid_mapping_keys(
            where=ctx,
            param="controls",
            missing=missing,
            extra=extra,
        )

    for col_name, totals in control_final.items():
        if not isinstance(totals, Mapping) or not totals:
            raise MethodError.invalid_type(
                where=ctx,
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
                where=ctx,
                param=f"controls[{col_name!r}]",
            )
        if np.all(vals == 0):
            raise DimensionError(
                title="All-zero control totals",
                detail=f"All control totals for {col_name!r} are zero, which is not allowed.",
                code="ZERO_CONTROL_TOTALS",
                where=ctx,
                param=f"controls[{col_name!r}]",
            )

    _check_margins_agree(control_final, where=ctx)

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
        raked_result = _rake_or_raise(
            current_w.reshape(-1, 1),
            margin_indices,
            margin_targets,
            ll_bound,
            up_bound,
            tol,
            max_iter,
        )

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
            final_result = _rake_or_raise(
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
            final_result = _rake_or_raise(
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

    raked_w = current_w

    # ── Convergence guard ─────────────────────────────────────────────────
    if (not rake_converged or (trimming is not None and not trim_unchanged)) and strict:
        reason = (
            f"Trim-rake cycle did not converge after {max_iter} cycles. "
            if trimming is not None
            else f"Raking did not converge after {max_iter} iterations. "
        )
        raise MethodError.not_applicable(
            where=ctx,
            method="rake",
            reason=(
                reason + "The design has NOT been modified. "
                "Increase max_iter, relax tol, or pass strict=False to store partial weights."
            ),
            hint="Try increasing max_iter or relaxing tol.",
        )

    if scope_idx is not None:
        full = w_full.copy()
        full[scope_idx] = raked_w
        raked_w = full

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
            if scope_idx is None:
                raked_reps = _rake_or_raise(
                    wgts_arr, margin_indices, margin_targets, ll_bound, up_bound, tol, max_iter
                )
            else:
                raked_reps = wgts_arr.copy()
                raked_reps[scope_idx] = _rake_or_raise(
                    np.ascontiguousarray(wgts_arr[scope_idx]),
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
                sample._design = sample._design.update(
                    rep_wgts=msgspec.structs.replace(
                        design.rep_wgts, prefix=wgt_name, n_reps=n_reps
                    )
                )

    sample._data = df
    return sample
