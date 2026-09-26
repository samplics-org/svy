# src/svy/weighting/normalization.py
"""
Weight normalization.

Targets are conveniences (sum to n, sum to 1, a chosen level per cell), not
population constraints -- which is the only thing that separates this from
``poststratify``. The arithmetic for a given set of targets is identical; the
difference is the claim being made, and so the variance treatment: a
normalization is recorded for provenance only, never consumed by the variance
estimator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import numpy as np
import polars as pl

from svy.core.design import WgtAdjustment
from svy.core.types import DomainScalarMap, Number
from svy.errors import MethodError, WeightingError
from svy.weighting._engine import (
    build_cells,
    record_null_cells,
    resolve_targets,
    scale_to_targets,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from svy.core.sample import Sample
    from svy.core.types import WhereArg


def normalize(
    sample: Sample,
    controls: DomainScalarMap | Number | None = None,
    *,
    factor: Number | None = None,
    shares: DomainScalarMap | None = None,
    cells: str | Sequence[str] | None = None,
    where: WhereArg = None,
    wgt_name: str = "norm_wgt",
    ignore_reps: bool = False,
    update_design_wgts: bool = True,
) -> Sample:
    ctx = "Sample.weighting.normalize"
    if factor is not None:
        given = [
            name
            for name, v in (("controls", controls), ("shares", shares), ("cells", cells))
            if v is not None
        ] + (["where"] if where is not None else [])
        if given:
            raise WeightingError.factor_conflict(where=ctx, given=given)
        if not float(factor) > 0.0:
            raise MethodError.invalid_range(
                where=ctx, param="factor", got=factor, min_=0.0, max_=None
            )
    df = sample._data
    design = sample._design

    if design.wgt is None:
        raise WeightingError.no_weight(where=ctx, method="normalize")
    wgt = design.wgt
    if wgt not in df.columns:
        raise WeightingError.missing_columns(
            where=ctx,
            param="design.wgt",
            missing=[wgt],
            available=list(df.columns),
            hint="Check that the weight column exists in the data.",
        )
    if wgt_name in set(df.columns):
        raise WeightingError.wgt_name_exists(
            where=ctx, method="normalize", wgt_name=wgt_name, existing=df.columns
        )

    wgt_arr = df.get_column(wgt).to_numpy().astype(np.float64)
    spec = None
    if factor is not None:
        f = float(factor)
        norm_arr = wgt_arr * f

        def adjust_reps(arr: np.ndarray) -> np.ndarray:
            return arr * f
    else:
        spec = build_cells(df, cells, where, where=ctx)
        targets = resolve_targets(
            controls=controls,
            shares=shares,
            spec=spec,
            wgt_arr=wgt_arr,
            method="normalize",
            where=ctx,
            counts_when_none=True,
        )
        norm_arr = scale_to_targets(wgt_arr.reshape(-1, 1), spec, targets)[:, 0]
        cell_spec, cell_targets = spec, targets

        def adjust_reps(arr: np.ndarray) -> np.ndarray:
            return scale_to_targets(arr, cell_spec, cell_targets)

    df = df.with_columns(pl.Series(name=wgt_name, values=norm_arr))

    if update_design_wgts:
        sample._push_design()
        # Provenance only: normalization targets are conveniences, not
        # population constraints, so no cells are snapshotted and the variance
        # estimator treats these weights as fixed.
        sample._design = sample._design.update(
            wgt=wgt_name,
            # Replaced below by the adjusted replicates. Unadjusted ones do not
            # go with the new weight, so ignore_reps leaves it without any; the
            # previous design in the history keeps them.
            rep_wgts=None if ignore_reps else sample._design.rep_wgts,
            wgt_adjustment=WgtAdjustment(kind="normalization", prev_wgt=wgt, new_wgt=wgt_name),
        )

    if not ignore_reps and design.rep_wgts is not None:
        rep_cols = design.rep_wgts.columns
        if rep_cols:
            adj = adjust_reps(df.select(rep_cols).to_numpy())
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
    if spec is not None:
        record_null_cells(sample, spec, where=ctx, prev_wgt=wgt, wgt_name=wgt_name)
    return sample
