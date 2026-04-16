# src/svy/selection/multistage.py
"""
Multi-stage / two-phase sampling — add_stage().

Chains a second-stage frame or pre-selected Sample onto the current
stage-1 sample.  Called from the Selection facade in base.py.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

import polars as pl

from svy.core.constants import SVY_PROB, SVY_PROB_STAGE1, SVY_ROW_INDEX, SVY_WEIGHT
from svy.selection.combine_stages import CombineResult, _combine_stages
from svy.errors import MethodError
from svy.selection._helpers import _psu_list

if TYPE_CHECKING:
    from svy.core.sample import Sample


def add_stage(
    sample: "Sample",
    next_stage: "pl.DataFrame | pl.LazyFrame | Sample",
    *,
    prob_name: str | None = None,
    wgt_name: str | None = None,
) -> "Sample":
    """
    Chain a second-stage frame or selected sample onto the current stage.

    The combined ``Sample`` represents the second-stage population.
    Unconditional probabilities (pi_hij = pi_hi * pi_j|hi) are computed
    immediately when ``next_stage`` is a pre-selected ``Sample``, or
    deferred to the next selection call otherwise.

    Parameters
    ----------
    sample     : current stage-1 Sample (injected by the Selection facade)
    next_stage : pl.DataFrame | pl.LazyFrame | Sample
        Three cases:

        * **DataFrame / LazyFrame** — bare frame.  Stage-1 columns are
          joined on the stage-1 PSU column.  The returned Sample is ready
          for a subsequent selection call.
        * **Sample without selection probs** — treated as a bare frame.
        * **Sample with selection probs** — combined probabilities computed
          immediately.

    prob_name : str | None
        Name for the combined probability column. Defaults to svy_prob_selection.
    wgt_name : str | None
        Name for the combined weight column. Defaults to svy_sample_weight.

    Returns
    -------
    Sample
        Combined Sample with stage-1 columns joined in.

    Notes
    -----
    Returned Design fields:
        stratum  — from stage-1 design
        psu      — from stage-1 design
        ssu      — from next_stage.design.psu (if Sample), else None
        prob     — svy_prob_selection_stage1  (chaining trigger)
        wgt      — None; recomputed after stage-2 selection
        hit, mos, pop_size, rep_wgts — None
        wr       — False

    Stage-1 column renames (always applied):
        svy_prob_selection  -> svy_prob_selection_stage1
        svy_sample_weight   -> svy_sample_weight_stage1
        svy_certainty       -> svy_certainty_stage1       (if present)
        svy_number_of_hits  -> svy_number_of_hits_stage1  (if present)
    """
    from svy.core.design import Design as _Design
    from svy.core.sample import Sample as _Sample

    s1 = sample
    s1_design = s1._design

    if s1_design.prob is None:
        raise MethodError.not_applicable(
            where="Sample.sampling.add_stage",
            method="add_stage",
            reason="Current stage has no selection probabilities.",
            hint="Run a selection method (pps_sys, srs, etc.) before add_stage().",
        )
    if s1_design.psu is None:
        raise MethodError.not_applicable(
            where="Sample.sampling.add_stage",
            method="add_stage",
            reason="Current stage Design has no PSU.",
            hint="Set psu= on the Design before calling add_stage().",
        )

    if isinstance(next_stage, (pl.DataFrame, pl.LazyFrame)):
        ns_df = next_stage if isinstance(next_stage, pl.DataFrame) else next_stage.collect()
        ns_df = cast(pl.DataFrame, ns_df)
        ns_design = _Design()
        already_selected = False
    elif isinstance(next_stage, _Sample):
        ns_df = next_stage._data
        if isinstance(ns_df, pl.LazyFrame):
            ns_df = ns_df.collect()
        ns_df = cast(pl.DataFrame, ns_df)
        ns_design = next_stage._design
        already_selected = ns_design.prob is not None
    else:
        raise MethodError.invalid_type(
            where="Sample.sampling.add_stage",
            param="next_stage",
            got=next_stage,
            expected="pl.DataFrame | pl.LazyFrame | Sample",
            hint="Pass a DataFrame, LazyFrame, or Sample as next_stage.",
        )

    s1_df = s1._data
    if isinstance(s1_df, pl.LazyFrame):
        s1_df = s1_df.collect()
    s1_df = cast(pl.DataFrame, s1_df)

    s1_psu = _psu_list(s1_design.psu)
    ns_psu = _psu_list(ns_design.psu) if ns_design.psu else s1_psu

    exclude: set[str] = {SVY_ROW_INDEX}
    internal = s1._internal_design or {}
    for key in ("stratum", "psu", "ssu"):
        val = internal.get(key)
        if isinstance(val, str):
            exclude.add(val)

    result: CombineResult = _combine_stages(
        stage1_df=s1_df,
        stage1_prob_col=s1_design.prob,
        stage1_wgt_col=s1_design.wgt,
        stage1_hit_col=s1_design.hit,
        stage1_psu=s1_psu,
        stage1_exclude_cols=exclude,
        next_df=ns_df,
        next_psu=ns_psu,
        next_prob_col=ns_design.prob if already_selected else None,
        next_wgt_col=ns_design.wgt if already_selected else None,
        already_selected=already_selected,
        out_prob_col=prob_name or SVY_PROB,
        out_wgt_col=wgt_name or SVY_WEIGHT,
    )

    combined_design = _Design(
        row_index=ns_design.row_index,
        stratum=s1_design.stratum,
        psu=s1_design.psu,
        ssu=ns_design.psu if ns_design.psu else None,
        wgt=result.out_wgt_col if result.already_selected else None,
        prob=result.chaining_prob_col,
        hit=ns_design.hit if already_selected else None,
        mos=None,
        pop_size=None,
        wr=False,
        rep_wgts=None,
    )

    combined = _Sample.__new__(_Sample)
    combined._data = result.data
    combined._design = combined_design
    combined._fpc = 1
    combined._metadata = copy.deepcopy(
        next_stage._metadata if isinstance(next_stage, _Sample) else s1._metadata
    )
    combined._internal_design = {
        "stratum": None,
        "psu": None,
        "ssu": None,
        "suffix": s1._internal_design.get("suffix", "_svy_internal_cols_concatenated"),
    }
    combined._warnings = next_stage._warnings if isinstance(next_stage, _Sample) else s1._warnings
    combined._print_width = None
    combined._singleton_result = None
    combined._stage_out_prob = result.out_prob_col
    combined._stage_out_wgt = result.out_wgt_col

    return combined
