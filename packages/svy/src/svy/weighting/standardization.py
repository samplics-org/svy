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

import warnings

from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import polars as pl

from svy.core.design import WgtAdjustment
from svy.core.types import DomainScalarMap
from svy.errors import MethodError
from svy.weighting._engine import (
    CellSpec,
    _as_float_vector,
    _cells_to_cols,
    _is_mapping,
    build_cells,
    materialize_cells,
    scale_to_targets,
)
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
    comp_keys = {c for _, c in split}

    if not _is_mapping(shares):
        raise MethodError.invalid_type(
            where=_CTX, param="shares", got=shares, expected="dict[cell, number]"
        )
    extra = set(shares.keys()) - comp_keys
    missing = comp_keys - set(shares.keys())
    if extra or missing:
        raise MethodError.invalid_mapping_keys(
            where=_CTX,
            param="shares",
            missing=sorted(missing, key=str),
            extra=sorted(extra, key=str),
            hint=(
                "`shares` needs one entry per level of the composition axis named "
                "by cells=, using the level values themselves as keys."
            ),
        )

    ordered = sorted(comp_keys, key=str)
    share_vec = _as_float_vector(shares, ordered, param="shares", method="standardize", where=_CTX)
    share_of = dict(zip(ordered, share_vec))
    if float(share_vec.sum()) <= 0:
        raise MethodError.not_applicable(
            where=_CTX,
            method="standardize",
            reason="`shares` must include at least one positive value",
            param="shares",
        )

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

    partial = sorted(
        {dom for dom in dom_total if dom_share[dom] < float(share_vec.sum()) - 1e-12}, key=str
    )
    if partial:
        # R's postStratify(partial=TRUE): renormalize over the levels a domain
        # actually has rather than refusing the whole adjustment.
        warnings.warn(
            f"{len(partial)} domain(s) do not observe every level of "
            f"{'/'.join(spec.cols or [])}: {partial[:5]}"
            f"{'...' if len(partial) > 5 else ''}. Shares were renormalized over "
            "the levels present, so these domains are standardized to a different "
            "population than the others.",
            UserWarning,
            stacklevel=3,
        )

    targets = np.empty(spec.n_cells, dtype=np.float64)
    for code, (dom, comp) in enumerate(split):
        denom = dom_share[dom]
        targets[code] = 0.0 if denom <= 0 else (share_of[comp] / denom) * dom_total[dom]
    return targets


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
        raise MethodError.not_applicable(
            where=_CTX,
            method="standardize",
            reason="Sample weight is None. Set design.wgt before calling standardize().",
        )
    wgt = design.wgt
    if wgt not in df.columns:
        raise MethodError.invalid_choice(
            where=_CTX,
            param="design.wgt",
            got=wgt,
            allowed=list(df.columns),
            hint="Check that the weight column exists in the data.",
        )
    if wgt_name in set(df.columns):
        raise MethodError.not_applicable(
            where=_CTX,
            method="standardize",
            reason=f"Column '{wgt_name}' already exists. Choose a different wgt_name.",
        )
    if cells is None:
        raise MethodError.not_applicable(
            where=_CTX,
            method="standardize",
            reason="`cells` names the composition axis and is required",
            hint="e.g. cells='agecat' with shares keyed by age group.",
        )

    by_cols = _cells_to_cols(by, where=_CTX) or []
    cells_cols = _cells_to_cols(cells, where=_CTX) or []
    overlap = set(by_cols) & set(cells_cols)
    if overlap:
        raise MethodError.not_applicable(
            where=_CTX,
            method="standardize",
            reason=(
                f"{sorted(overlap)} appears in both by= and cells=. Domains and the "
                "composition axis must be different variables"
            ),
        )

    spec = build_cells(df, [*by_cols, *cells_cols], where, where=_CTX)
    wgt_arr = df.get_column(wgt).to_numpy().astype(np.float64)
    targets = _standardize_targets(spec, wgt_arr, shares, len(by_cols))

    std_arr = scale_to_targets(wgt_arr.reshape(-1, 1), spec, targets)[:, 0]
    df = df.with_columns(pl.Series(name=wgt_name, values=std_arr))

    if update_design_wgts:
        df, cells_col = materialize_cells(df, spec, wgt_name=wgt_name)
        sample._push_design()
        sample._design = sample._design.update(
            wgt=wgt_name,
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

    if trimming is not None:
        from svy.weighting.poststratification import _trim_cycle

        cycled, ok = _trim_cycle(std_arr, spec, targets, trimming)
        if not ok:
            warnings.warn(
                f"Trim-standardize cycle did not converge after {trimming.max_iter} cycles; "
                "storing the partial result.",
                UserWarning,
                stacklevel=2,
            )
        sample._data = df.with_columns(pl.Series(name=wgt_name, values=cycled))

    return sample
