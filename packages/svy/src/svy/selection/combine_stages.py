# src/svy/selection/combine_stages.py
from __future__ import annotations

import warnings

from dataclasses import dataclass

import numpy as np
import polars as pl

from svy.core.constants import (
    SVY_CERT_STAGE1,
    SVY_CERTAINTY,
    SVY_HITS_STAGE1,
    SVY_PROB_STAGE1,
    SVY_PROB_STAGE2,
    SVY_WGT_STAGE1,
    SVY_WGT_STAGE2,
)
from svy.errors import DimensionError, MethodError


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombineResult:
    """
    Result of combining two sampling stages.

    Attributes
    ----------
    data : pl.DataFrame
        Combined DataFrame with stage-1 columns joined into next_stage.
    chaining_prob_col : str
        Stage-1 probability column name in ``data``. Set as
        ``Design.prob`` so the next selection call chains automatically.
    out_prob_col : str
        Desired name for the final combined probability column.
    out_wgt_col : str
        Desired name for the final combined weight column.
    already_selected : bool
        True when stage-2 selection was already done. Combined probs
        are in ``data`` under ``out_prob_col``; no further selection
        is needed.
    """

    data: pl.DataFrame
    chaining_prob_col: str
    out_prob_col: str
    out_wgt_col: str
    already_selected: bool


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _combine_stages(
    *,
    stage1_df: pl.DataFrame,
    stage1_prob_col: str,
    stage1_wgt_col: str | None,
    stage1_hit_col: str | None,
    stage1_psu: list[str],
    stage1_exclude_cols: set[str],
    next_df: pl.DataFrame,
    next_psu: list[str],
    next_prob_col: str | None,
    next_wgt_col: str | None,
    already_selected: bool,
    out_prob_col: str,
    out_wgt_col: str,
) -> CombineResult:
    """
    Core engine for combining two sampling stages.

    Handles column renaming (always-rename for design cols, collision
    rule for others), PSU validation, join, and (when
    ``already_selected=True``) immediate computation of combined
    probabilities.

    Parameters
    ----------
    stage1_df : pl.DataFrame
        Selected stage-1 data.
    stage1_prob_col : str
        Probability column in ``stage1_df`` (e.g. ``svy_prob_selection``).
    stage1_wgt_col : str | None
        Weight column in ``stage1_df``, or None.
    stage1_hit_col : str | None
        Hit column in ``stage1_df``, or None.
    stage1_psu : list[str]
        PSU column name(s) from stage-1 Design.
    stage1_exclude_cols : set[str]
        Columns to exclude from stage-1 carry-over (row index, internal
        design helper columns).
    next_df : pl.DataFrame
        Next-stage data.
    next_psu : list[str]
        PSU column name(s) to join on in ``next_df``.
    next_prob_col : str | None
        Stage-2 probability column (only when ``already_selected``).
    next_wgt_col : str | None
        Stage-2 weight column (only when ``already_selected``).
    already_selected : bool
        Whether stage-2 selection has already been performed.
    out_prob_col : str
        Desired name for the combined probability column.
    out_wgt_col : str
        Desired name for the combined weight column.

    Returns
    -------
    CombineResult
    """
    # ------------------------------------------------------------------
    # 1. Validate PSU columns exist
    # ------------------------------------------------------------------
    for col in stage1_psu:
        if col not in stage1_df.columns:
            raise DimensionError.missing_columns(
                where="Sample.sampling.add_stage",
                param="design.psu",
                missing=[col],
                available=sorted(stage1_df.columns),
                hint="The PSU column defined in the stage-1 Design was not found in the data.",
            )
    for col in next_psu:
        if col not in next_df.columns:
            raise DimensionError.missing_columns(
                where="Sample.sampling.add_stage",
                param="next_stage",
                missing=[col],
                available=sorted(next_df.columns),
                hint=(
                    f"PSU column {col!r} not found in next_stage. "
                    "Ensure next_stage contains the same PSU column(s) as the stage-1 design."
                ),
            )

    # ------------------------------------------------------------------
    # 2. Validate PSU matching
    # ------------------------------------------------------------------
    if len(stage1_psu) == 1 and len(next_psu) == 1:
        s1_vals = set(stage1_df[stage1_psu[0]].unique().to_list())
        ns_vals = set(next_df[next_psu[0]].unique().to_list())
    else:
        s1_vals = set(map(tuple, stage1_df.select(stage1_psu).unique().rows()))
        ns_vals = set(map(tuple, next_df.select(next_psu).unique().rows()))

    unmatched = ns_vals - s1_vals
    if unmatched:
        sample_unmatched = sorted(unmatched)[:5]
        raise MethodError.invalid_choice(
            where="Sample.sampling.add_stage",
            param="next_stage PSU values",
            got=sample_unmatched,
            allowed=sorted(s1_vals),
            hint=(
                f"{len(unmatched)} PSU(s) in next_stage have no match in stage-1. "
                "All next_stage PSUs must be present in the selected stage-1 sample. "
                f"Sample of unmatched: {sample_unmatched}" + (" ..." if len(unmatched) > 5 else "")
            ),
        )

    not_in_ns = s1_vals - ns_vals
    if not_in_ns:
        warnings.warn(
            f"{len(not_in_ns)} PSU(s) from stage-1 have no matching "
            f"records in next_stage and will not appear in the combined "
            f"sample.",
            UserWarning,
            stacklevel=5,
        )

    # ------------------------------------------------------------------
    # 3. Always-rename map for stage-1 design columns
    # ------------------------------------------------------------------
    always_rename: dict[str, str] = {}
    for src, dst in [
        (stage1_prob_col, SVY_PROB_STAGE1),
        (stage1_wgt_col, SVY_WGT_STAGE1),
        (SVY_CERTAINTY, SVY_CERT_STAGE1),
        (stage1_hit_col, SVY_HITS_STAGE1),
    ]:
        if src and src in stage1_df.columns:
            always_rename[src] = dst

    # ------------------------------------------------------------------
    # 4. Build stage-1 slice and apply renames
    # ------------------------------------------------------------------
    # PSU columns must always be included — they are the join keys.
    # Exclude everything else in stage1_exclude_cols.
    psu_set = set(stage1_psu)
    s1_cols = [c for c in stage1_df.columns if c not in stage1_exclude_cols or c in psu_set]
    s1_slice = stage1_df.select(s1_cols)

    rename_map = {k: v for k, v in always_rename.items() if k in s1_slice.columns}
    if rename_map:
        s1_slice = s1_slice.rename(rename_map)

    # Collision rule: non-PSU column already in next_df → append _stage1
    # PSU columns from both stages are protected — they are join keys.
    psu_protected: set[str] = psu_set | set(next_psu)
    collision_rename: dict[str, str] = {}
    for col in s1_slice.columns:
        if col in next_df.columns and col not in psu_protected:
            collision_rename[col] = f"{col}_stage1"
    if collision_rename:
        s1_slice = s1_slice.rename(collision_rename)

    # ------------------------------------------------------------------
    # 5. Resolve join keys after renaming
    # ------------------------------------------------------------------
    # s1 PSU cols in s1_slice after renaming (should be unchanged since
    # PSU cols are protected from both always_rename and collision_rename)
    s1_join_cols: list[str] = []
    for col in stage1_psu:
        renamed = rename_map.get(col) or collision_rename.get(col) or col
        s1_join_cols.append(renamed)

    # ------------------------------------------------------------------
    # 6. Join stage-1 columns onto next_stage data
    # ------------------------------------------------------------------
    # Always use left_on / right_on to avoid Polars requiring the join
    # key to exist in both frames under the same name.
    # Drop the stage-1 PSU columns from s1_slice AFTER using them as
    # right_on, so the result does not have duplicate PSU columns.
    combined_data = next_df.join(
        s1_slice,
        left_on=next_psu,
        right_on=s1_join_cols,
        how="left",
    )

    # ------------------------------------------------------------------
    # 7. Already selected: rename stage-2 cols + compute combined probs
    # ------------------------------------------------------------------
    if already_selected:
        s2_rename: dict[str, str] = {}
        if next_prob_col and next_prob_col in combined_data.columns:
            s2_rename[next_prob_col] = SVY_PROB_STAGE2
        if next_wgt_col and next_wgt_col in combined_data.columns:
            s2_rename[next_wgt_col] = SVY_WGT_STAGE2
        if s2_rename:
            combined_data = combined_data.rename(s2_rename)

        combined_data = combined_data.with_columns(
            [
                (pl.col(SVY_PROB_STAGE1) * pl.col(SVY_PROB_STAGE2)).alias(out_prob_col),
                (1.0 / (pl.col(SVY_PROB_STAGE1) * pl.col(SVY_PROB_STAGE2))).alias(out_wgt_col),
            ]
        )

        return CombineResult(
            data=combined_data,
            chaining_prob_col=out_prob_col,
            out_prob_col=out_prob_col,
            out_wgt_col=out_wgt_col,
            already_selected=True,
        )

    # ------------------------------------------------------------------
    # 8. Not yet selected: return with chaining trigger
    # ------------------------------------------------------------------
    return CombineResult(
        data=combined_data,
        chaining_prob_col=SVY_PROB_STAGE1,
        out_prob_col=out_prob_col,
        out_wgt_col=out_wgt_col,
        already_selected=False,
    )


# ---------------------------------------------------------------------------
# Write-back helper used by Selection._pps and Selection.srs
# ---------------------------------------------------------------------------


def _apply_chaining_writeback(
    *,
    src_df: pl.DataFrame,
    sel_idx,
    hits,
    probs,
    certainty,
    row_col: str,
    prev_prob_col: str,
    out_prob_col: str,
    out_wgt_col: str,
    hit_col: str,
    is_pps: bool,
) -> pl.DataFrame:
    """
    Build the write-back temp DataFrame when in chaining mode.

    Retrieves stage-1 probs from ``src_df``, multiplies by stage-2
    probs, and returns a DataFrame ready to join onto ``src_df``.

    Parameters
    ----------
    src_df : pl.DataFrame
        Full source DataFrame containing the stage-1 prob column.
    sel_idx : array-like
        Selected row index values.
    hits : array-like
        Hit counts from the selection engine.
    probs : array-like
        Stage-2 inclusion probabilities from the selection engine.
    certainty : array-like | None
        Certainty flags (PPS only); pass None for SRS.
    row_col : str
        Row index column name.
    prev_prob_col : str
        Stage-1 probability column name (``svy_prob_selection_stage1``).
    out_prob_col : str
        Desired combined probability column name.
    out_wgt_col : str
        Desired combined weight column name.
    hit_col : str
        Hit column name.
    is_pps : bool
        True for PPS (adds certainty column), False for SRS.

    Returns
    -------
    pl.DataFrame
        Temp DataFrame to join onto ``src_df``.
    """
    stage2_probs = np.asarray(probs, dtype=np.float64)

    prev = pl.DataFrame({row_col: sel_idx}).join(
        other=src_df.select([row_col, prev_prob_col]),
        on=row_col,
        how="left",
    )
    stage1_probs = prev[prev_prob_col].fill_null(1.0).to_numpy().astype(np.float64)
    combined_probs = stage1_probs * stage2_probs

    safe_s2 = np.where(stage2_probs > 0, stage2_probs, np.nan)
    safe_combined = np.where(combined_probs > 0, combined_probs, np.nan)

    row_data: dict = {
        row_col: sel_idx,
        SVY_PROB_STAGE2: stage2_probs,
        SVY_WGT_STAGE2: (1.0 / safe_s2),
        out_prob_col: combined_probs,
        out_wgt_col: (1.0 / safe_combined),
        hit_col: np.asarray(hits, dtype=np.int64),
    }
    cast_schema: dict = {
        row_col: src_df[row_col].dtype,
        SVY_PROB_STAGE2: pl.Float64,
        SVY_WGT_STAGE2: pl.Float64,
        out_prob_col: pl.Float64,
        out_wgt_col: pl.Float64,
        hit_col: pl.Int64,
    }

    if is_pps and certainty is not None:
        row_data[SVY_CERTAINTY] = certainty
        cast_schema[SVY_CERTAINTY] = pl.Boolean

    return pl.DataFrame(row_data).cast(cast_schema)
