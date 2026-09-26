# src/svy/weighting/standardization.py
"""
Direct standardization.

Standardization removes confounding by composition: it reweights each domain so
that every domain has the same distribution over a chosen axis, making rates
comparable across domains that differ in that axis. Age is the usual axis --
comparing crude mortality across regions with different age structures compares
age structures as much as mortality -- but the method is general, and any
composition variable works.

Mechanically this is poststratification with derived targets, and it runs on the
same engine: for domain ``g`` and composition cell ``c``,

    target(g, c) = share(c) x W_g

where ``W_g`` is the domain's own estimated total under the same scope. Domain
totals are therefore preserved and only the within-domain composition is
reshaped -- the semantics of R's ``svystandardize``, Stata's ``stdize`` and
SUDAAN's ``STDVAR``.

Standardized weights are analysis-specific: ``where`` bakes one variable's
missingness into the weights and ``by`` bakes in the domain structure, so
estimating a different variable or a different breakdown on the same standardized
sample is silently wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import polars as pl

from svy.core.design import WgtAdjustment
from svy.core.types import DomainScalarMap
from svy.core.warnings import Severity, WarnCode
from svy.errors import WeightingError
from svy.errors.weighting_errors import show
from svy.weighting._engine import (
    CellSpec,
    _cells_to_cols,
    _is_mapping,
    build_cells,
    materialize_cells,
    record_null_cells,
    record_trim_cycle,
    scale_to_targets,
)
from svy.weighting._keys import LevelIndex, match_keys, sort_levels, target_vector
from svy.weighting.types import TrimConfig


if TYPE_CHECKING:
    from collections.abc import Sequence

    from svy.core.sample import Sample
    from svy.core.types import WhereArg

_CTX = "Sample.weighting.standardize"


def _split_label(label: Any, n_by: int) -> tuple[tuple, Any]:
    """Split a cross label into (domain key, composition key).

    ``build_cells`` labels a single column with its raw value and several columns
    with a tuple in column order, and standardize always builds the cross as
    ``by`` columns followed by ``cells`` columns.
    """
    if not isinstance(label, tuple):
        return (), label
    dom = label[:n_by]
    comp = label[n_by:]
    return dom, (comp[0] if len(comp) == 1 else comp)


def _standardize_targets(
    spec: CellSpec,
    wgt_arr: np.ndarray,
    shares: DomainScalarMap,
    n_by: int,
) -> np.ndarray:
    """Derive target(g, c) = share(c) x W_g for every observed (domain, cell)."""
    split = [_split_label(lab, n_by) for lab in spec.labels]
    comp_levels = list(dict.fromkeys(c for _, c in split))

    if not _is_mapping(shares):
        raise WeightingError.targets_type(
            where=_CTX, param="shares", got=shares, expected="a dict[cell, number]"
        )
    comp_cols = (spec.cols or [])[n_by:]
    index = LevelIndex(comp_levels, width=len(comp_cols))
    matched = match_keys(shares, index, where=_CTX, param="shares", cols=comp_cols)
    ordered = sort_levels(index.levels)
    share_vec = np.asarray(target_vector(matched, ordered, where=_CTX, param="shares"))
    share_of = dict(zip(ordered, share_vec))
    if float(share_vec.sum()) <= 0:
        raise WeightingError.all_zero(where=_CTX, param="shares")

    # Domain totals under the same scope, so the `where` used to build the cells
    # is necessarily the one used for W_g -- the mismatch that makes hand-rolled
    # standardization wrong cannot arise here.
    cell_sums = np.bincount(
        spec.codes[spec.codes >= 0],
        weights=wgt_arr[spec.codes >= 0],
        minlength=spec.n_cells,
    )
    dom_total: dict[tuple, float] = {}
    dom_share: dict[tuple, float] = {}
    for code, (dom, comp) in enumerate(split):
        dom_total[dom] = dom_total.get(dom, 0.0) + float(cell_sums[code])
        dom_share[dom] = dom_share.get(dom, 0.0) + float(share_of[comp])

    # R's postStratify(partial=TRUE): a domain missing a level is renormalized
    # over the levels it has rather than refusing the whole adjustment.
    partial = sort_levels(
        dom[0] if len(dom) == 1 else dom
        for dom in dom_total
        if dom_share[dom] < float(share_vec.sum()) - 1e-12
    )

    targets = np.empty(spec.n_cells, dtype=np.float64)
    for code, (dom, comp) in enumerate(split):
        denom = dom_share[dom]
        targets[code] = 0.0 if denom <= 0 else (share_of[comp] / denom) * dom_total[dom]
    return targets, partial


def _record_partial(
    sample: Sample, partial: list[Any], by_cols: list[str], cells_cols: list[str]
) -> None:
    if not partial:
        return
    shown = ", ".join(show(d) for d in partial[:5]) + (", ..." if len(partial) > 5 else "")
    sample.warn(
        code=WarnCode.DOMAIN_LEVELS_PARTIAL,
        title="Domains missing some levels",
        detail=(
            f"{len(partial)} domain(s) of {'/'.join(by_cols)} do not observe every level "
            f"of {'/'.join(cells_cols)}: {shown}. Shares were renormalized over "
            "the levels present, so these domains are standardized to a different "
            "population than the others."
        ),
        where=_CTX,
        level=Severity.WARNING,
        param="by",
        got=list(partial),
        hint="Collapse sparse levels of the composition axis, or accept that these "
        "domains follow a different standard.",
    )


def standardize(
    sample: Sample,
    cells: str | Sequence[str],
    *,
    shares: DomainScalarMap,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    wgt_name: str = "std_wgt",
    ignore_reps: bool = False,
    update_design_wgts: bool = True,
    trimming: TrimConfig | None = None,
) -> Sample:
    df = sample._data
    design = sample._design

    if design.wgt is None:
        raise WeightingError.no_weight(where=_CTX, method="standardize")
    wgt = design.wgt
    if wgt not in df.columns:
        raise WeightingError.missing_columns(
            where=_CTX,
            param="design.wgt",
            missing=[wgt],
            available=list(df.columns),
            hint="Check that the weight column exists in the data.",
        )
    if wgt_name in set(df.columns):
        raise WeightingError.wgt_name_exists(
            where=_CTX, method="standardize", wgt_name=wgt_name, existing=df.columns
        )
    if cells is None:
        raise WeightingError.cells_required(
            where=_CTX,
            param="cells",
            reason="`cells` names the composition axis and is required.",
            hint="e.g. cells='agecat' with shares keyed by age group.",
        )

    by_cols = _cells_to_cols(by, where=_CTX, param="by") or []
    cells_cols = _cells_to_cols(cells, where=_CTX) or []
    overlap = set(by_cols) & set(cells_cols)
    if overlap:
        raise WeightingError.cells_by_overlap(where=_CTX, overlap=sorted(overlap))

    spec = build_cells(df, [*by_cols, *cells_cols], where, where=_CTX)
    wgt_arr = df.get_column(wgt).to_numpy().astype(np.float64)
    targets, partial = _standardize_targets(spec, wgt_arr, shares, len(by_cols))

    std_arr = scale_to_targets(wgt_arr.reshape(-1, 1), spec, targets)[:, 0]
    cycle_ok = True
    if trimming is not None:
        from svy.weighting.poststratification import _trim_cycle

        std_arr, cycle_ok = _trim_cycle(std_arr, spec, targets, trimming)
    df = df.with_columns(pl.Series(name=wgt_name, values=std_arr))

    if update_design_wgts:
        df, cells_col = materialize_cells(df, spec, wgt_name=wgt_name)
        sample._push_design()
        sample._design = sample._design.update(
            wgt=wgt_name,
            # Replaced below by the adjusted replicates. Unadjusted ones do not
            # go with the new weight, so ignore_reps leaves it without any; the
            # previous design in the history keeps them.
            rep_wgts=None if ignore_reps else sample._design.rep_wgts,
            wgt_adjustment=WgtAdjustment(
                kind="standardization",
                prev_wgt=wgt,
                new_wgt=wgt_name,
                cells=(cells_col,),
                # Pins both, matching R. The targets are derived from estimated
                # domain totals, so an argument exists for treating only the
                # composition as pinned -- but R's svystandardize delegates to
                # postStratify with absolute controls and so removes both, and
                # the R-parity SEs (31.9154418732174 / 24.9249948644004 on
                # apiclus1) hold only under that reading.
                pins_total=True,
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

    record_null_cells(sample, spec, where=_CTX, prev_wgt=wgt, wgt_name=wgt_name)
    _record_partial(sample, partial, by_cols, cells_cols)
    if not cycle_ok:
        record_trim_cycle(sample, where=_CTX, what="Trim-standardize cycle", trimming=trimming)
    return sample
