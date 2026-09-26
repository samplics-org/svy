# src/svy/weighting/raking.py
"""
Raking (iterative proportional fitting) weight adjustment.

rake() and controls_margins_template() take a Sample and return a Sample
(for chaining). The Weighting class in base.py delegates to these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, cast

import msgspec
import numpy as np
import polars as pl


try:
    from svy_rs._internal import rake as rust_rake  # type: ignore[import-untyped]
    from svy_rs._internal import trim_weights as rust_trim_weights  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_rake = None
    rust_trim_weights = None

from svy.core.design import WgtAdjustment
from svy.core.types import Category, ControlsType
from svy.core.warnings import WarnCode, check_on_finding, finding_level
from svy.errors import DimensionError, MethodError, WeightingError
from svy.weighting._engine import CELLS_PREFIX, _where_mask, resolve_bounds
from svy.weighting._keys import (
    LevelIndex,
    check_na_option,
    match_keys,
    sort_levels,
    target_vector,
    template_levels,
)
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
            raise WeightingError.bounds_exceeded(where=where, bounds=(args[3], args[4])) from None
        raise


def _normalize_controls_like(x: ControlsType | None) -> ControlsType | None:
    """Return x unchanged if it's a non-empty dict, else None."""
    if x is None:
        return None
    if not isinstance(x, dict) or len(x) == 0:
        return None
    return x


def _check_margins_agree(totals: dict[str, float], *, where: str) -> None:
    """Every margin must describe the same population.

    Raking cannot satisfy margins whose totals differ; without this it would
    silently iterate to max_iter and return whatever it reached.
    """
    if len(totals) < 2:
        return
    lo, hi = min(totals.values()), max(totals.values())
    if hi > 0 and (hi - lo) / hi > 1e-6:
        raise WeightingError.margins_disagree(where=where, totals=totals)


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def controls_margins_template(
    sample: Sample,
    *,
    margins: Mapping[str, str],
    na: Literal["error", "level", "drop"] = "error",
    na_label: str = "__NA__",
) -> dict[str, dict[Category, float]]:
    df: pl.DataFrame = sample.data
    where = "Sample.weighting.controls_margins_template"

    check_na_option(na, where=where)
    for mname, col in margins.items():
        if not isinstance(mname, str) or not isinstance(col, str):
            raise MethodError.invalid_type(
                where=where,
                param="margins",
                got=(mname, col),
                expected="dict[str, str]",
                hint="Both margin keys and values must be strings.",
            )
    missing = [col for col in margins.values() if col not in df.columns]
    if missing:
        raise WeightingError.missing_columns(
            where=where,
            param="margins",
            missing=missing,
            available=list(df.columns),
            hint="Check that the margin column exists in the data.",
        )

    result: dict[str, dict[Category, float]] = {}
    for mname, col in margins.items():
        levels = template_levels(
            df, [col], na=na, na_label=na_label, where=where, code="MARGIN_NA"
        )
        result[mname] = {lab: np.nan for lab in levels}
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


def _converged(
    w: np.ndarray,
    margin_indices: list[np.ndarray],
    margin_targets: list[np.ndarray],
    tol: float,
) -> bool:
    """True when every margin is met within ``tol``."""
    return _max_margin_error(w, margin_indices, margin_targets) <= tol


def rake(
    sample: Sample,
    *,
    controls: ControlsType | None = None,
    shares: ControlsType | None = None,
    where: WhereArg = None,
    wgt_name: str = "rk_wgt",
    ignore_reps: bool = False,
    bounds: tuple[float | None, float | None] | None = None,
    tol: float = 1e-4,
    max_iter: int = 100,
    display_iter: bool = False,
    update_design_wgts: bool = True,
    on_nonconvergence: str = "error",
    trimming: TrimConfig | None = None,
) -> Sample:
    ctx = "Sample.weighting.rake"
    df = sample._data
    design = sample._design

    check_on_finding(on_nonconvergence, param="on_nonconvergence", where=ctx)
    ll_bound, up_bound = resolve_bounds(bounds, where=ctx)

    if design.wgt is None:
        raise WeightingError.no_weight(where=ctx, method="rake")
    wgt = design.wgt
    if wgt not in df.columns:
        raise WeightingError.missing_columns(
            where=ctx,
            param="design.wgt",
            missing=[wgt],
            available=list(df.columns),
            hint="Check that the weight column exists in the data.",
        )

    existing_cols = set(df.columns)
    if wgt_name in existing_cols:
        raise WeightingError.wgt_name_exists(
            where=ctx, method="rake", wgt_name=wgt_name, existing=df.columns
        )

    for name, arg in (("controls", controls), ("shares", shares)):
        if arg is not None and not isinstance(arg, Mapping):
            raise WeightingError.targets_type(
                where=ctx,
                param=name,
                got=arg,
                expected="a dict {column: {level: number}}",
                hint=f"e.g. {name}={{'region': {{'North': ..., 'South': ...}}}}.",
            )
    controls_norm: ControlsType | None = _normalize_controls_like(x=controls)
    shares_norm: ControlsType | None = _normalize_controls_like(x=shares)

    if controls_norm is None and shares_norm is None:
        raise WeightingError.targets_missing(where=ctx, method="rake")
    if controls_norm is not None and shares_norm is not None:
        raise WeightingError.targets_conflict(where=ctx, method="rake")

    supplied = cast(ControlsType, controls_norm if controls_norm is not None else shares_norm)
    param = "controls" if controls_norm is not None else "shares"
    rake_cols = list(supplied.keys())

    unknown = [c for c in rake_cols if not isinstance(c, str) or c not in df.columns]
    if unknown:
        raise WeightingError.missing_columns(
            where=ctx,
            param=f"{param} keys",
            missing=unknown,
            available=list(df.columns),
            hint=f"Key {param} by the names of the margin columns.",
        )

    w0 = df.get_column(wgt).to_numpy().astype(np.float64)

    null_counts = dict(zip(rake_cols, df.select(rake_cols).null_count().row(0)))
    with_nulls = {c: int(n) for c, n in null_counts.items() if n > 0}
    if with_nulls:
        raise DimensionError(
            title="Null values in raking column",
            detail=(
                f"{', '.join(f'{c!r} has {n} null value(s)' for c, n in with_nulls.items())}. "
                "Raking requires complete data."
            ),
            code="MARGIN_NA",
            where=ctx,
            param=next(iter(with_nulls)),
            expected=0,
            got=with_nulls,
            hint="Drop or impute missing values before raking, or leave those rows "
            "out with where=.",
        )

    # `where` scopes the adjustment: only in-scope rows are raked, and the rest
    # keep their weight. The IPF then runs on the subset, so the margins the
    # caller supplies describe the scoped population and nothing else.
    scope = _where_mask(df, where, where=ctx)
    scope_idx = None
    margin_df = df.select(rake_cols)
    if scope is not None:
        scope_idx = np.flatnonzero(scope)
        if scope_idx.size == 0:
            raise WeightingError.no_rows_in_scope(where=ctx, method="rake", hint="Check `where`.")
        w_full = w0
        w0 = w0[scope_idx]
        margin_df = margin_df.filter(pl.Series(scope))

    margin_indices: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    for col in rake_cols:
        sub = f"{param}[{col!r}]"
        totals = supplied[col]
        if not isinstance(totals, Mapping) or not totals:
            raise WeightingError.targets_type(
                where=ctx,
                param=sub,
                got=totals,
                expected="a non-empty dict {level: number}",
            )
        values = margin_df.get_column(col).to_list()
        levels = sort_levels(dict.fromkeys(values))
        index = LevelIndex(levels)
        matched = match_keys(totals, index, where=ctx, param=sub, cols=[col])
        vec = np.asarray(target_vector(matched, index.levels, where=ctx, param=sub))
        if not np.any(vec > 0):
            raise WeightingError.all_zero(where=ctx, param=sub)
        code_of = {lv: i for i, lv in enumerate(index.levels)}
        margin_indices.append(np.fromiter((code_of[v] for v in values), np.int64, len(values)))
        raw_targets.append(vec)

    if controls_norm is None:
        # Shares are marginal proportions, normalized within each margin
        # against one grand total, so cross-margin consistency is structural
        # rather than something the caller has to get right.
        grand_total = float(w0.sum())
        margin_targets = [v / v.sum() * grand_total for v in raw_targets]
    else:
        margin_targets = raw_targets
        _check_margins_agree(
            {c: float(v.sum()) for c, v in zip(rake_cols, margin_targets)}, where=ctx
        )

    assert rust_rake is not None  # noqa: S101

    # ── Trim-rake cycle ───────────────────────────────────────────────────
    # When trimming=None: single rake pass, no trim step.
    # When trimming is set: iterate up to trimming.max_iter cycles:
    #   1. Rake current weights to convergence (up to max_iter IPF steps each)
    #   2. Trim — if no weights changed (within TrimConfig.tol), both
    #      constraints are satisfied and we stop early.
    # Final step is always rake so margins are satisfied.
    # Replicates are raked once with the final main-weight cycle result.

    n_cycles = trimming.max_iter if trimming is not None else 1
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
        rake_converged = _converged(raked_w, margin_indices, margin_targets, tol)

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
            rake_converged = _converged(current_w, margin_indices, margin_targets, tol)
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
            rake_converged = _converged(current_w, margin_indices, margin_targets, tol)

    raked_w = current_w

    # ── Convergence guard ─────────────────────────────────────────────────
    converged = rake_converged and (trimming is None or trim_unchanged)
    what = "Trim-rake cycle" if trimming is not None else "Raking"
    cap, cap_param = (
        (trimming.max_iter, "trimming.max_iter")
        if trimming is not None
        else (max_iter, "max_iter")
    )
    miss = {
        "max_iter": cap,
        "max_margin_error": _max_margin_error(raked_w, margin_indices, margin_targets),
    }
    if trimming is not None:
        miss["trim_bounds_met"] = bool(trim_unchanged)
    hint = f"Increase {cap_param} (now {cap}) or relax tol (now {tol:g})."
    if not converged and on_nonconvergence == "error":
        raise WeightingError.not_converged(
            where=ctx,
            method="rake",
            what=what,
            max_iter=cap,
            got=miss,
            expected={"tol": tol},
            hint=hint,
        )

    if scope_idx is not None:
        full = w_full.copy()
        full[scope_idx] = raked_w
        raked_w = full

    df = df.with_columns(pl.Series(name=wgt_name, values=raked_w))

    if update_design_wgts:
        # One column per margin: raking's sweep iterates margins separately,
        # so a single concatenated column would encode poststratification on
        # the full cross -- a different, stronger calibration.
        cells_cols: list[str] = []
        for j, col in enumerate(rake_cols, start=1):
            codes = np.full(len(raked_w), -1, dtype=np.int64)
            if scope_idx is None:
                codes[:] = margin_indices[j - 1]
            else:
                codes[scope_idx] = margin_indices[j - 1]
            name = f"{CELLS_PREFIX}{wgt_name}_{j}"
            df = df.with_columns(
                pl.Series(
                    name=name,
                    values=[None if c < 0 else int(c) for c in codes],
                    dtype=pl.Int32,
                )
            )
            cells_cols.append(name)
        sample._push_design()
        sample._design = sample._design.update(
            wgt=wgt_name,
            # Replaced below by the adjusted replicates. Unadjusted ones do not
            # go with the new weight, so ignore_reps leaves it without any; the
            # previous design in the history keeps them.
            rep_wgts=None if ignore_reps else sample._design.rep_wgts,
            wgt_adjustment=WgtAdjustment(
                kind="raking",
                prev_wgt=wgt,
                new_wgt=wgt_name,
                cells=tuple(cells_cols),
                pins_total=controls_norm is not None,
            ),
        )

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
    if not converged:
        sample.warn(
            code=WarnCode.MAX_ITER_REACHED,
            title=f"{what} did not converge",
            detail=(
                f"{what} did not converge after {cap} "
                f"{'cycles' if trimming is not None else 'iterations'} (max margin error "
                f"{miss['max_margin_error']:.3g}, tol {tol:g}); {wgt_name!r} holds the "
                "last iterate."
            ),
            where=ctx,
            level=finding_level(on_nonconvergence),
            param=cap_param,
            expected={"tol": tol},
            got=miss,
            hint=hint,
        )
    return sample
