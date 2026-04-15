# src/svy/weighting/trimming.py
"""
Sample-aware weight trimming.

trim() is the public entry point called by Weighting.trim() in base.py.
It handles all Sample interaction: extracting weights, resolving thresholds,
domain splitting, replicate weight detection, writing results back,
warnings, and audit.

The pure algorithm lives in the Rust extension (svy_rs._internal.trim_weights)
and works entirely with flat numpy arrays and Python primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from svy.core.warnings import Severity, WarnCode
from svy.errors import DimensionError, MethodError

try:
    from svy_rs._internal import trim_weights as rust_trim_weights  # type: ignore[import-untyped]
    from svy_rs._internal import trim_weights_matrix as rust_trim_weights_matrix  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_trim_weights = None
    rust_trim_weights_matrix = None

from svy.weighting.types import (
    TrimConfig,
    TrimResult,
    resolve_threshold,
)

if TYPE_CHECKING:
    from svy.core.sample import Sample


def trim(
    sample: Sample,
    upper=None,
    lower=None,
    by=None,
    redistribute: bool = True,
    min_cell_size: int = 10,
    max_iter: int = 10,
    tol: float = 1e-6,
    wgt_name: str | None = "trim_wgt",
    update_design_wgts: bool = True,
) -> Sample:
    """
    Trim survey weights and return a new Sample (chainable).

    Parameters
    ----------
    upper : float | Threshold | callable | None
        Upper bound spec.
        float > 1       -> absolute cap
        float in (0, 1] -> quantile of the weight distribution
        Threshold       -> k * stat(w), e.g. Threshold("median", 6.0)
        callable        -> f(w: np.ndarray) -> float
    lower : same type options as upper
    by : str | list[str] | None
        Trim within domains. Thresholds computed per domain;
        redistribution also within each domain.
    redistribute : bool
        Redistribute trimmed mass proportionally to non-trimmed units.
        Default True.
    min_cell_size : int
        Skip and warn for domains with fewer positive-weight units.
        Default 10.
    max_iter : int
        Maximum iterations. Default 10.
    tol : float
        Convergence tolerance (fraction of weights changed). Default 1e-6.
    wgt_name : str | None
        Name for the output weight column. Default "trim_wgt".
        Pass None to trim in-place: the current design weight column is
        modified directly, no new column is created, and replicate weight
        columns are also modified in-place.
    update_design_wgts : bool
        Update design.wgt to point to the trimmed weight column.
        Has no effect when wgt_name=None (in-place mode).

    Returns
    -------
    Sample
    """
    where = "Sample.weighting.trim"

    config = TrimConfig(
        upper=upper,
        lower=lower,
        by=by,
        redistribute=redistribute,
        min_cell_size=min_cell_size,
        max_iter=max_iter,
        tol=tol,
    )

    # wgt_name=None -> in-place (replace=True); otherwise create new column
    return _run_trim(
        sample,
        config,
        wgt_name=wgt_name,
        replace=wgt_name is None,
        update_design_wgts=update_design_wgts,
        where=where,
    )


def _run_trim(
    sample: Sample,
    config: TrimConfig,
    *,
    wgt_name: str | None = None,
    replace: bool = False,
    update_design_wgts: bool = True,
    where: str = "Sample.weighting.trim",
) -> Sample:
    """
    Core Sample-aware trim logic. Accepts a pre-built TrimConfig so that
    other weighting methods (rake, calibrate) can call this directly via
    their trimming= parameter.
    """
    df: pl.DataFrame = sample._data
    design = sample._design

    # ── Validate ─────────────────────────────────────────────────────────
    if design.wgt is None:
        raise MethodError.not_applicable(
            where=where,
            method="trim",
            reason="Sample weight is None. Set design.wgt before calling trim().",
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
    if replace and wgt_name is not None:
        raise MethodError.not_applicable(
            where=where,
            method="trim",
            reason="Pass either replace=True or wgt_name=..., not both.",
        )

    # ── Extract weights ───────────────────────────────────────────────────
    w_orig = df.get_column(wgt).to_numpy().astype(np.float64, copy=False)

    # Negative weights → hard error
    neg_mask = w_orig < 0
    if neg_mask.any():
        n_neg = int(neg_mask.sum())
        sample.warn(
            code=WarnCode.NEGATIVE_WEIGHT,
            title="Negative weights found",
            detail=f"{n_neg} unit(s) have negative weights. Trim cannot proceed.",
            where=where,
            level=Severity.ERROR,
            got=n_neg,
            hint="Remove or correct negative weights before trimming.",
        )
        raise DimensionError(
            title="Negative weights found",
            detail=f"trim() found {n_neg} negative weight(s). Trimming cannot proceed.",
            code="NEGATIVE_WEIGHTS",
            where=where,
            param=wgt,
            got=n_neg,
            expected="all weights >= 0",
            hint="Remove or correct negative weights before trimming.",
        )

    # Zero weights → info
    n_zero = int((w_orig == 0.0).sum())
    if n_zero > 0:
        sample.warn(
            code=WarnCode.ZERO_WEIGHT,
            title="Zero-weight units excluded from trimming",
            detail=(
                f"{n_zero} unit(s) have zero weights and will be preserved "
                "but excluded from trimming and redistribution."
            ),
            where=where,
            level=Severity.INFO,
            got=n_zero,
        )

    # ── Domain splitting or global ────────────────────────────────────────
    w_out = w_orig.copy()
    by_cols: list[str] | None = None

    if config.by is not None:
        by_cols = [config.by] if isinstance(config.by, str) else list(config.by)
        missing = [c for c in by_cols if c not in df.columns]
        if missing:
            raise MethodError.invalid_choice(
                where=where,
                param="by",
                got=missing,
                allowed=list(df.columns),
                hint="Check that all by= columns exist in the data.",
            )

    if by_cols is None:
        result = _trim_domain(w_out, config, domain_label="(global)", sample=sample, where=where)
        if result is not None:
            w_out = result.weights
            _emit_audit(sample, result, domain="(global)", where=where)
    else:
        domain_arr = _build_domain_array(df, by_cols)
        for domain in np.unique(domain_arr):
            mask = domain_arr == domain
            w_domain = w_out[mask]
            result = _trim_domain(
                w_domain, config, domain_label=str(domain), sample=sample, where=where
            )
            if result is not None:
                w_out[mask] = result.weights
                _emit_audit(sample, result, domain=str(domain), where=where)

    # ── Warn if weight sum changed without redistribution ─────────────────
    if not config.redistribute:
        sum_before = float(np.sum(w_orig[w_orig > 0]))
        sum_after = float(np.sum(w_out[w_orig > 0]))
        if abs(sum_after - sum_before) > config.tol * max(sum_before, 1.0):
            sample.warn(
                code=WarnCode.WEIGHT_SUM_CHANGED,
                title="Total weight sum changed",
                detail=(
                    f"Weight sum changed from {sum_before:.4f} to {sum_after:.4f} "
                    f"({100 * (sum_after - sum_before) / sum_before:+.2f}%). "
                    "Set redistribute=True to preserve the total weight sum."
                ),
                where=where,
                level=Severity.WARNING,
                hint="Use redistribute=True to avoid changing the weighted sample size.",
            )

    # ── Write back main weight ────────────────────────────────────────────
    if replace:
        target_wgt = wgt
        df = df.with_columns(pl.Series(name=target_wgt, values=w_out))
    else:
        target_wgt = wgt_name if wgt_name is not None else "trim_wgt"
        if target_wgt in df.columns:
            raise MethodError.not_applicable(
                where=where,
                method="trim",
                reason=f"Column '{target_wgt}' already exists. Choose a different wgt_name.",
            )
        df = df.with_columns(pl.Series(name=target_wgt, values=w_out))

    if update_design_wgts:
        sample._design = sample._design.update(wgt=target_wgt)

    # ── Adjust replicate weights ──────────────────────────────────────────
    # Replicates get the same proportional adjustment as the main weight.
    # factor[i] = w_out[i] / w_orig[i] — computed once, applied to all reps.
    #
    # Scenario A — direct trim (replace=False):
    #   New rep columns are created as {target_wgt}1, {target_wgt}2, ...
    #   rep_wgts.prefix is updated to target_wgt — consistent with rake/poststratify.
    #
    # Scenario B — trim called by parent method (replace=True):
    #   Rep columns are modified in-place under their existing names.
    #   No metadata update — the parent method owns the design update.
    if design.rep_wgts is not None:
        rep_cols = design.rep_wgts.columns
        if rep_cols:
            with np.errstate(divide="ignore", invalid="ignore"):
                factors = np.where(w_orig > 0, w_out / w_orig, 1.0)

            rep_mat = df.select(rep_cols).to_numpy().astype(np.float64, order="C")
            adjusted_rep_mat = rep_mat * factors[:, np.newaxis]

            if replace:
                # In-place: overwrite existing rep columns
                df = df.with_columns(
                    [
                        pl.Series(name=col, values=adjusted_rep_mat[:, i])
                        for i, col in enumerate(rep_cols)
                    ]
                )
            else:
                # New columns: {target_wgt}1, {target_wgt}2, ...
                n_reps = len(rep_cols)
                new_rep_names = [f"{target_wgt}{i}" for i in range(1, n_reps + 1)]
                df = df.with_columns(
                    [
                        pl.Series(name=name, values=adjusted_rep_mat[:, i])
                        for i, name in enumerate(new_rep_names)
                    ]
                )
                if update_design_wgts:
                    sample._design = sample._design.update_rep_weights(
                        method=design.rep_wgts.method,
                        prefix=target_wgt,
                        n_reps=n_reps,
                        fay_coef=design.rep_wgts.fay_coef,
                        df=design.rep_wgts.df,
                    )

    # ── Flush to sample ───────────────────────────────────────────────────
    sample._data = df
    return sample


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------


def _build_domain_array(df: pl.DataFrame, by_cols: list[str]) -> np.ndarray:
    if len(by_cols) == 1:
        return df.get_column(by_cols[0]).to_numpy()
    return np.array(
        ["__".join(str(v) for v in row) for row in df.select(by_cols).iter_rows()],
        dtype=object,
    )


def _trim_domain(
    w: np.ndarray,
    config: TrimConfig,
    domain_label: str,
    sample: Sample,
    where: str,
) -> TrimResult | None:
    """
    Resolve thresholds, call the engine with flat args, pack TrimResult.
    Returns None if domain is skipped (too small).
    """
    n_positive = int(np.sum(w > 0))

    if n_positive < config.min_cell_size:
        sample.warn(
            code=WarnCode.DOMAIN_SKIPPED,
            title="Domain skipped — cell too small",
            detail=(
                f"Domain {domain_label!r} has {n_positive} positive-weight unit(s), "
                f"below min_cell_size={config.min_cell_size}. Trimming skipped."
            ),
            where=where,
            level=Severity.WARNING,
            hint="Increase min_cell_size or combine small domains.",
        )
        return None

    # Resolve ThresholdSpec → float before passing to engine
    w_pos = w[w > 0].astype(np.float64)
    upper_val: float | None = (
        resolve_threshold(config.upper, w_pos) if config.upper is not None else None
    )
    lower_val: float | None = (
        resolve_threshold(config.lower, w_pos) if config.lower is not None else None
    )

    # Call Rust engine with flat primitives only
    assert rust_trim_weights is not None  # noqa: S101
    (
        trimmed_weights,
        n_trimmed_upper,
        n_trimmed_lower,
        weight_sum_before,
        weight_sum_after,
        ess_before,
        ess_after,
        iterations,
        converged,
    ) = rust_trim_weights(
        np.asarray(w, dtype=np.float64),
        upper_val,
        lower_val,
        config.redistribute,
        config.max_iter,
        config.tol,
    )

    if not converged:
        sample.warn(
            code=WarnCode.MAX_ITER_REACHED,
            title="Trimming did not converge",
            detail=(
                f"Domain {domain_label!r}: max_iter={config.max_iter} reached "
                f"without convergence (tol={config.tol}). "
                f"Iterations run: {iterations}."
            ),
            where=where,
            level=Severity.WARNING,
            hint="Increase max_iter or relax tol.",
        )

    # Pack into TrimResult for audit and return
    return TrimResult(
        weights=trimmed_weights,
        upper_threshold=upper_val,
        lower_threshold=lower_val,
        n_trimmed_upper=n_trimmed_upper,
        n_trimmed_lower=n_trimmed_lower,
        weight_sum_before=weight_sum_before,
        weight_sum_after=weight_sum_after,
        ess_before=ess_before,
        ess_after=ess_after,
        iterations=iterations,
        converged=converged,
    )


def _emit_audit(
    sample: Sample,
    result: TrimResult,
    domain: str,
    where: str,
) -> None:
    sample.warn(
        code=WarnCode.WEIGHT_ADJ_AUDIT,
        title=f"Trim audit — domain {domain!r}",
        detail=(
            f"domain={domain!r}, "
            f"upper={result.upper_threshold}, lower={result.lower_threshold}, "
            f"n_trimmed_upper={result.n_trimmed_upper}, "
            f"n_trimmed_lower={result.n_trimmed_lower}, "
            f"ess_before={result.ess_before:.2f}, ess_after={result.ess_after:.2f}, "
            f"iterations={result.iterations}, converged={result.converged}"
        ),
        where=where,
        level=Severity.INFO,
        extra={
            "domain": domain,
            "upper_threshold": result.upper_threshold,
            "lower_threshold": result.lower_threshold,
            "n_trimmed_upper": result.n_trimmed_upper,
            "n_trimmed_lower": result.n_trimmed_lower,
            "weight_sum_before": result.weight_sum_before,
            "weight_sum_after": result.weight_sum_after,
            "ess_before": result.ess_before,
            "ess_after": result.ess_after,
            "iterations": result.iterations,
            "converged": result.converged,
        },
    )
