# src/svy/weighting/poststratification.py
"""
Post-stratification weight adjustment.

Targets here are population constraints, not conveniences -- the distinction from
``normalize``, whose arithmetic is identical for a given set of targets. Because
the constraint is a claim about the population, this adjustment is recorded as
variance-consumed: a poststratified total of a calibration margin has no sampling
variability left in it.

``controls`` sets the total (each cell hits an absolute figure). ``shares``
preserves it (composition is pinned, the grand total carries through) -- the right
form when the cell proportions are known but the population count is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import numpy as np
import polars as pl

from svy.core.design import WgtAdjustment
from svy.core.types import DomainScalarMap, Number
from svy.errors import MethodError
from svy.weighting._engine import (
    CellSpec,
    build_cells,
    materialize_cells,
    resolve_targets,
    scale_to_targets,
)
from svy.weighting.raking import _trim_constraints_satisfied
from svy.weighting.types import TrimConfig, resolve_threshold


try:
    from svy_rs._internal import trim_weights as rust_trim_weights  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_trim_weights = None

if TYPE_CHECKING:
    from collections.abc import Sequence

    from svy.core.sample import Sample
    from svy.core.types import WhereArg


def _trim_cycle(
    wgt_arr: np.ndarray,
    spec: CellSpec,
    targets: np.ndarray,
    trimming: TrimConfig,
) -> tuple[np.ndarray, bool]:
    """Alternate trimming and re-poststratification until both hold.

    Only in-scope rows take part: rows the adjustment never touched keep their
    weight, so trimming them here would be a second, unrequested adjustment.
    """
    idx = np.flatnonzero(spec.in_scope) if spec.in_scope is not None else None
    work = wgt_arr if idx is None else wgt_arr[idx]
    codes = spec.codes if idx is None else spec.codes[idx]

    pos = work[work > 0].astype(np.float64)
    upper = resolve_threshold(trimming.upper, pos) if trimming.upper is not None else None
    lower = resolve_threshold(trimming.lower, pos) if trimming.lower is not None else None

    assert rust_trim_weights is not None  # noqa: S101
    sub = CellSpec(codes, spec.labels, None, spec.cols)

    current = work.copy()
    ok = False
    for _ in range(trimming.max_iter):
        current = scale_to_targets(current.reshape(-1, 1), sub, targets)[:, 0]
        (current, *_) = rust_trim_weights(
            current, upper, lower, trimming.redistribute, trimming.max_iter, trimming.tol
        )
        ok = _trim_constraints_satisfied(current, upper, lower, 1e-4)
        if ok:
            current = scale_to_targets(current.reshape(-1, 1), sub, targets)[:, 0]
            ok = _trim_constraints_satisfied(current, upper, lower, 1e-4)
            break

    if idx is None:
        return current, ok
    out = wgt_arr.copy()
    out[idx] = current
    return out, ok


def poststratify(
    sample: Sample,
    controls: DomainScalarMap | Number | None = None,
    *,
    shares: DomainScalarMap | None = None,
    cells: str | Sequence[str] | None = None,
    where: WhereArg = None,
    wgt_name: str = "ps_wgt",
    ignore_reps: bool = False,
    update_design_wgts: bool = True,
    strict: bool = True,
    trimming: TrimConfig | None = None,
) -> Sample:
    ctx = "Sample.weighting.poststratify"
    df = sample._data
    design = sample._design

    if design.wgt is None:
        raise MethodError.not_applicable(
            where=ctx,
            method="poststratify",
            reason="Sample weight is None. Set design.wgt before calling poststratify().",
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
    if wgt_name in set(df.columns):
        raise MethodError.not_applicable(
            where=ctx,
            method="poststratify",
            reason=f"Column '{wgt_name}' already exists. Choose a different wgt_name.",
        )

    spec = build_cells(df, cells, where, where=ctx)
    wgt_arr = df.get_column(wgt).to_numpy().astype(np.float64)
    targets = resolve_targets(
        controls=controls,
        shares=shares,
        spec=spec,
        wgt_arr=wgt_arr,
        method="poststratify",
        where=ctx,
    )

    ps_arr = scale_to_targets(wgt_arr.reshape(-1, 1), spec, targets)[:, 0]
    df = df.with_columns(pl.Series(name=wgt_name, values=ps_arr))

    if update_design_wgts:
        df, cells_col = materialize_cells(df, spec, wgt_name=wgt_name)
        sample._push_design()
        sample._design = sample._design.update(
            wgt=wgt_name,
            wgt_adjustment=WgtAdjustment(
                kind="poststratification",
                prev_wgt=wgt,
                new_wgt=wgt_name,
                cells=(cells_col,),
                pins_total=controls is not None,
            ),
        )

    if not ignore_reps and design.rep_wgts is not None:
        rep_cols = design.rep_wgts.columns
        if rep_cols:
            adj = scale_to_targets(df.select(rep_cols).to_numpy(), spec, targets)
            n_reps = len(rep_cols)
            new_names = [f"{wgt_name}{i}" for i in range(1, n_reps + 1)]
            sample._data = df.hstack(pl.DataFrame(adj, schema=new_names))
            df = sample._data
            if update_design_wgts:
                sample._design = sample._design.update(
                    rep_wgts=msgspec.structs.replace(
                        design.rep_wgts, prefix=wgt_name, n_reps=n_reps
                    )
                )

    sample._data = df

    if trimming is not None:
        cycled, ok = _trim_cycle(ps_arr, spec, targets, trimming)
        if strict and not ok:
            raise MethodError.not_applicable(
                where=ctx,
                method="poststratify",
                reason=(
                    f"Trim-poststratify cycle did not converge after {trimming.max_iter} "
                    "cycles. The design has NOT been modified. "
                    "Pass strict=False to store partial results."
                ),
                hint="Increase TrimConfig.max_iter or use a less restrictive trim threshold.",
            )
        sample._data = df.with_columns(pl.Series(name=wgt_name, values=cycled))

    return sample
