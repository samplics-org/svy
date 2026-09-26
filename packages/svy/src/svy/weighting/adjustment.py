# src/svy/weighting/adjustment.py
"""
Non-response weight adjustment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import msgspec
import numpy as np
import polars as pl


try:
    from svy_rs._internal import adjust_nr as rust_adjust_nr  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_adjust_nr = None

from svy.core.design import WgtAdjustment
from svy.core.types import DomainScalarMap
from svy.core.warnings import Severity, WarnCode, check_on_finding
from svy.errors import WeightingError
from svy.weighting._engine import CellSpec, _where_mask, build_cells, record_null_cells
from svy.weighting._keys import native, text_forms
from svy.weighting.trimming import _run_trim as _apply_trim
from svy.weighting.types import TrimConfig
from svy.wrangling.mutate import mutate as _mutate
from svy.wrangling.rows import filter_records as _filter_records


if TYPE_CHECKING:
    from collections.abc import Sequence

    from svy.core.design import Design
    from svy.core.sample import Sample
    from svy.core.types import WhereArg

_CANONICAL_TO_INT: dict[str, int] = {"rr": 0, "nr": 1, "in": 2, "uk": 3}


def _status_matches(label: Any, levels: list[Any]) -> list[Any]:
    """The status values ``label`` names: equal to it, or equal as text ignoring
    case (``"refused"`` names ``"Refused"``, ``"1"`` names ``1``, ``"true"``
    names ``True``)."""
    label = native(label)
    wanted = {f.casefold() for f in ((label,) if isinstance(label, str) else text_forms(label))}
    if label is None:
        wanted.add("none")
    out = []
    for lv in levels:
        if lv is None:
            forms: tuple[str, ...] = ("None",)
        else:
            forms = text_forms(lv)
        try:
            same = lv == label and (lv is None) == (label is None)
        except Exception:  # noqa: BLE001
            same = False
        if same or any(f.casefold() in wanted for f in forms):
            out.append(lv)
    return out


def _encode_resp_status(
    values: list[Any],
    resp_mapping: DomainScalarMap | None,
    *,
    column: str,
    where: str = "Sample.weighting.adjust",
) -> np.ndarray:
    """
    Encode response statuses to the Rust integer codes (0=rr, 1=nr, 2=in, 3=uk).

    Every row must match a mapping entry (or a canonical label when no mapping
    is given) — code 0 means "respondent", so letting unmatched rows fall
    through would silently inflate their weights. Mapping values may be
    scalars or collections (e.g. {"nr": ["refusal", "noncontact"]}).
    """
    levels = list(dict.fromkeys(values))
    status_of: dict[Any, str] = {}

    if resp_mapping is not None:
        if not isinstance(resp_mapping, Mapping):
            raise WeightingError.targets_type(
                where=where,
                param="resp_mapping",
                got=resp_mapping,
                expected="a dict {status code: value or list of values}",
                hint="e.g. resp_mapping={'rr': 1, 'nr': [2, 3], 'uk': 9}.",
            )
        allowed: list[Any] = []
        conflicts: dict[Any, list[str]] = {}
        for canonical_key, data_label in resp_mapping.items():
            key_lower = str(canonical_key).lower()
            if key_lower not in _CANONICAL_TO_INT:
                raise WeightingError.resp_mapping_key(where=where, key=native(canonical_key))
            labels = (
                list(data_label)
                if isinstance(data_label, (list, tuple, set, frozenset, np.ndarray))
                else [data_label]
            )
            for lab in labels:
                allowed.append(native(lab))
                for lv in _status_matches(lab, levels):
                    prev = status_of.get(lv)
                    if prev is not None and prev != key_lower:
                        conflicts.setdefault(lv, [prev]).append(key_lower)
                    status_of[lv] = key_lower
        if conflicts:
            raise WeightingError.resp_mapping_conflict(where=where, conflicts=conflicts)
    else:
        allowed = list(_CANONICAL_TO_INT.keys())
        for lv in levels:
            if isinstance(lv, str) and lv.lower() in _CANONICAL_TO_INT:
                status_of[lv] = lv.lower()

    unknown = [lv for lv in levels if lv not in status_of]
    if unknown:
        counts = dict.fromkeys(unknown, 0)
        for v in values:
            if v in counts:
                counts[v] += 1
        raise WeightingError.resp_status_unknown(
            where=where,
            column=column,
            counts=counts,
            allowed=allowed,
            mapping=resp_mapping,
        )

    return np.fromiter(
        (_CANONICAL_TO_INT[status_of[v]] for v in values), dtype=np.int64, count=len(values)
    )


def _apply_nr(
    wgts: np.ndarray,
    spec: CellSpec,
    resp_codes: np.ndarray,
    unknown_to_inelig: bool,
) -> np.ndarray:
    """Run the non-response kernel on in-scope rows only.

    Rows outside the adjustment keep their weight whatever their response
    status: they were never part of this adjustment's classes.
    """
    assert rust_adjust_nr is not None  # noqa: S101
    if spec.in_scope is None:
        return rust_adjust_nr(
            np.ascontiguousarray(wgts, dtype=np.float64),
            spec.codes,
            resp_codes,
            unknown_to_inelig,
        )
    out = np.array(wgts, dtype=np.float64, copy=True)
    idx = np.flatnonzero(spec.in_scope)
    out[idx] = rust_adjust_nr(
        np.ascontiguousarray(out[idx], dtype=np.float64),
        spec.codes[idx],
        resp_codes[idx],
        unknown_to_inelig,
    )
    return out


def _adjust_panel(
    sample: Sample,
    df: pl.DataFrame,
    design: Design,
    wgt_cols: list[str],
    resp_status: str,
    cells: str | Sequence[str] | None,
    where: WhereArg,
    resp_mapping: DomainScalarMap | None,
    unknown_to_inelig: bool,
    ctx: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CellSpec, bool]:
    """Nonresponse factors on a panel: computed on the rows in scope, applied
    to the case.

    Two rules change against the cross-section: a case with a row at an
    earlier wave but none in scope is a nonrespondent (attriters have no
    row to filter on), and the factor is written to every row of the case so
    the longitudinal weight is constant within it. Returns the new weights
    (n x k), the encoded status of the real rows, the scope mask, the cells and
    whether the missing-in-scope rule applied.
    """
    case_id, wave = design.case_id, design.wave
    assert case_id is not None and wave is not None  # noqa: S101 — caller checked
    n = df.height
    mask = _where_mask(df, where, where=ctx)
    scope = np.ones(n, dtype=bool) if mask is None else mask
    resp_codes = _encode_resp_status(
        df.get_column(resp_status).to_list(), resp_mapping, column=resp_status, where=ctx
    )

    wave_vals = df.get_column(wave)
    scope_waves = wave_vals.filter(pl.Series(scope)).unique()
    if scope_waves.is_empty():
        raise WeightingError.no_rows_in_scope(where=ctx, method="adjust", hint="Check `where`.")
    wave_subset = bool(np.array_equal(wave_vals.is_in(scope_waves.implode()).to_numpy(), scope))

    virtual = df.head(0)
    if wave_subset:
        in_scope_cases = df.filter(pl.Series(scope)).select(case_id).unique()
        virtual = (
            df.filter(pl.col(wave) < scope_waves.min())
            .join(in_scope_cases, on=case_id, how="anti")
            .sort(wave)
            .unique(subset=[case_id], keep="last", maintain_order=True)
        )

    n_virtual = virtual.height
    aug = pl.concat([df, virtual], how="vertical") if n_virtual else df
    aug = aug.with_columns(
        pl.Series("__svy_scope__", np.concatenate([scope, np.ones(n_virtual, dtype=bool)]))
    )
    spec = build_cells(aug, cells, pl.col("__svy_scope__"), where=ctx)
    codes_aug = np.concatenate([resp_codes, np.ones(n_virtual, dtype=np.int64)])
    old = aug.select(wgt_cols).to_numpy().astype(np.float64)
    new = _apply_nr(old, spec, codes_aug, unknown_to_inelig)

    factor = np.ones_like(old)
    pos = old > 0
    factor[pos] = new[pos] / old[pos]
    assigned = np.ones(aug.height, dtype=bool) if spec.in_scope is None else spec.in_scope

    fac_names = [f"__svy_f{i}" for i in range(len(wgt_cols))]
    per_case = (
        aug.select(case_id, wave)
        .with_columns([pl.Series(nm, factor[:, i]) for i, nm in enumerate(fac_names)])
        .filter(pl.Series(assigned))
        .sort(wave)
        .unique(subset=[case_id], keep="last", maintain_order=True)
        .drop(wave)
    )
    joined = df.select(case_id).join(per_case, on=case_id, how="left", maintain_order="left")
    case_factor = joined.select(fac_names).fill_null(1.0).to_numpy().astype(np.float64)
    return old[:n] * case_factor, resp_codes, scope, spec, wave_subset


def adjust(
    sample: Sample,
    resp_status: str,
    cells: str | Sequence[str] | None = None,
    *,
    where: WhereArg = None,
    resp_mapping: DomainScalarMap | None = None,
    wgt_name: str = "nr_wgt",
    ignore_reps: bool = False,
    unknown_to_inelig: bool = True,
    update_design_wgts: bool = True,
    respondents_only: bool = True,
    trimming: TrimConfig | None = None,
    on_nonconvergence: str = "error",
) -> Sample:
    ctx = "Sample.weighting.adjust"
    df = sample._data
    design = sample._design
    check_on_finding(on_nonconvergence, param="on_nonconvergence", where=ctx)

    if design.wgt is None:
        raise WeightingError.no_weight(where=ctx, method="adjust")
    wgt = design.wgt
    if wgt not in df.columns:
        raise WeightingError.missing_columns(
            where=ctx,
            param="design.wgt",
            missing=[wgt],
            available=list(df.columns),
            hint="Check that the weight column exists in the data.",
        )
    if not isinstance(resp_status, str) or resp_status not in df.columns:
        raise WeightingError.missing_columns(
            where=ctx,
            param="resp_status",
            missing=[resp_status],
            available=list(df.columns),
            hint="`resp_status` must be a string naming an existing column.",
        )

    existing_cols = set(df.columns)
    if wgt_name in existing_cols:
        raise WeightingError.wgt_name_exists(
            where=ctx, method="adjust", wgt_name=wgt_name, existing=df.columns
        )

    rep_cols = list(design.rep_wgts.columns) if not ignore_reps and design.rep_wgts else []
    wgt_cols = [wgt, *rep_cols]

    keep_mask: np.ndarray
    rule_applied = True
    if design.is_panel:
        new_wgts, resp_codes, scope, spec, rule_applied = _adjust_panel(
            sample,
            df,
            design,
            wgt_cols,
            resp_status,
            cells,
            where,
            resp_mapping,
            unknown_to_inelig,
            ctx,
        )
        # Nonrespondents leave at the scope waves only: their earlier rows
        # stay, since they responded then.
        keep_mask = ~scope | (resp_codes == 0)
    else:
        spec = build_cells(df, cells, where, where=ctx)
        resp_codes = _encode_resp_status(
            df.get_column(resp_status).to_list(), resp_mapping, column=resp_status, where=ctx
        )
        old = df.select(wgt_cols).to_numpy().astype(np.float64)
        new_wgts = _apply_nr(old, spec, resp_codes, unknown_to_inelig)
        # Filter from the encoded codes (0 == respondent) — the single source
        # of truth already used for the adjustment itself. Re-deriving the
        # mask from raw strings was case-sensitive while the encoder is not,
        # which could silently empty the sample.
        keep_mask = resp_codes == 0

    # Only rows in an adjustment class leave: a row in none (a null cell, or
    # outside `where`) was not adjusted, so it keeps its weight and stays,
    # whatever its status -- dropping it would drop its weight with it.
    in_class = (
        np.ones(df.height, dtype=bool) if spec.in_scope is None else spec.in_scope[: df.height]
    )
    keep_mask = keep_mask | ~in_class
    null_rows = np.zeros(df.height, dtype=bool)
    for m in spec.nulls.values():
        null_rows |= m[: df.height]
    n_nonresp_kept = int((null_rows & (resp_codes != 0)).sum())

    # New columns, the design, and the row drop all go through the Sample's
    # own tracking: mutate registers the columns, update_design records the
    # previous design and refreshes the internal state (concat columns,
    # singletons, validation), filter_records re-checks singletons.
    new_cols: dict[str, pl.Series] = {wgt_name: pl.Series(wgt_name, new_wgts[:, 0])}
    new_rep_names = [f"{wgt_name}{i}" for i in range(1, len(rep_cols) + 1)]
    for i, name in enumerate(new_rep_names, start=1):
        new_cols[name] = pl.Series(name, new_wgts[:, i])
    _mutate(sample, new_cols, inplace=True)

    if update_design_wgts:
        # Provenance only: R carries no variance record for non-response
        # either. Modelling the adjustment's own variance is a different
        # estimator (SUDAAN's WTADJUST), not the calibration sweep.
        updates: dict = {
            "wgt": wgt_name,
            "wgt_adjustment": WgtAdjustment(kind="nonresponse", prev_wgt=wgt, new_wgt=wgt_name),
        }
        # Unadjusted replicates do not go with the new weight, so ignore_reps
        # leaves it without any; the previous design in the history keeps them.
        updates["rep_wgts"] = (
            msgspec.structs.replace(design.rep_wgts, prefix=wgt_name, n_reps=len(rep_cols))
            if rep_cols
            else None
        )
        sample.update_design(**updates)

    if respondents_only:
        _filter_records(
            sample,
            pl.lit(pl.Series(keep_mask)),
            check_singletons=True,
            on_singletons="warn",
            inplace=True,
        )

    if trimming is not None:
        if update_design_wgts:
            sample = _apply_trim(
                sample,
                trimming,
                replace=True,
                update_design_wgts=True,
                where="Sample.weighting.adjust",
                on_nonconvergence=on_nonconvergence,
                param="trimming.max_iter",
            )
        else:
            # The trim must target the freshly created adjusted weight (and
            # its replicate columns), not the caller's original design weight:
            # point the design at the new columns for the trim, then restore.
            original_design = sample._design
            sample._design = original_design.update(
                wgt=wgt_name,
                rep_wgts=(
                    msgspec.structs.replace(design.rep_wgts, prefix=wgt_name, n_reps=len(rep_cols))
                    if rep_cols
                    else None
                ),
            )
            sample = _apply_trim(
                sample,
                trimming,
                replace=True,
                update_design_wgts=False,
                where="Sample.weighting.adjust",
                on_nonconvergence=on_nonconvergence,
                param="trimming.max_iter",
            )
            sample._design = original_design

    record_null_cells(
        sample,
        spec,
        where=ctx,
        prev_wgt=wgt,
        wgt_name=wgt_name,
        n_rows=df.height,
        outcome=(
            f"{wgt_name!r} keeps their {wgt!r} value and they stay in the sample whatever "
            f"their response status ({n_nonresp_kept} of them not respondents)"
        ),
        extra={"nonrespondents_kept": n_nonresp_kept},
    )
    if not rule_applied:
        wave = design.wave
        sample.warn(
            code=WarnCode.PANEL_SCOPE_NOT_WAVES,
            title="Missing-in-scope rule skipped",
            detail=(
                "adjust on a panel: `where` is not a set of waves, so cases with earlier "
                "rows but none in scope are NOT added as nonrespondents (the "
                "missing-in-scope rule was skipped)."
            ),
            where=ctx,
            level=Severity.WARNING,
            param="where",
            hint=f"Scope one or more whole waves, e.g. where=col({wave!r}) == 2, to apply it.",
        )
    return sample
