# src/svy/weighting/adjustment.py
"""
Non-response weight adjustment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import numpy as np
import polars as pl


try:
    from svy_rs._internal import adjust_nr as rust_adjust_nr  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_adjust_nr = None

from svy.core.design import WgtAdjustment
from svy.core.types import DomainScalarMap
from svy.errors import MethodError
from svy.weighting._engine import CellSpec, build_cells
from svy.weighting.trimming import _run_trim as _apply_trim
from svy.weighting.types import TrimConfig


if TYPE_CHECKING:
    from collections.abc import Sequence

    from svy.core.sample import Sample
    from svy.core.types import WhereArg

_CANONICAL_TO_INT: dict[str, int] = {"rr": 0, "nr": 1, "in": 2, "uk": 3}


def _encode_resp_status(
    resp_status_arr: np.ndarray,
    resp_mapping: DomainScalarMap | None,
) -> np.ndarray:
    """
    Encode response statuses to the Rust integer codes (0=rr, 1=nr, 2=in, 3=uk).

    Every row must match a mapping entry (or a canonical label when no mapping
    is given) — code 0 means "respondent", so letting unmatched rows fall
    through would silently inflate their weights. Mapping values may be
    scalars or collections (e.g. {"nr": ["refusal", "noncontact"]}).
    """
    n = len(resp_status_arr)
    codes = np.zeros(n, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)
    lower = np.char.lower(resp_status_arr.astype(str))

    if resp_mapping is not None:
        allowed_labels: list[str] = []
        for canonical_key, data_label in resp_mapping.items():
            key_lower = str(canonical_key).lower()
            if key_lower not in _CANONICAL_TO_INT:
                raise MethodError.invalid_choice(
                    where="adjust._encode_resp_status",
                    param="resp_mapping key",
                    got=canonical_key,
                    allowed=list(_CANONICAL_TO_INT.keys()),
                    hint="Use canonical response status codes: rr, nr, in, uk.",
                )
            labels = (
                list(data_label)
                if isinstance(data_label, (list, tuple, set, frozenset, np.ndarray))
                else [data_label]
            )
            for lab in labels:
                mask = lower == str(lab).lower()
                codes[mask] = _CANONICAL_TO_INT[key_lower]
                matched |= mask
                allowed_labels.append(str(lab))
    else:
        allowed_labels = list(_CANONICAL_TO_INT.keys())
        for label, code in _CANONICAL_TO_INT.items():
            mask = lower == label
            codes[mask] = code
            matched |= mask

    if not matched.all():
        unmatched = sorted(set(resp_status_arr[~matched].astype(str)))
        raise MethodError.invalid_choice(
            where="Sample.weighting.adjust",
            param="resp_status",
            got=unmatched[:10],
            allowed=allowed_labels,
            hint="Every response status value (including nulls) must match a "
            "resp_mapping entry, or a canonical code (rr/nr/in/uk) when "
            "resp_mapping is None.",
        )

    return codes


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
) -> Sample:
    ctx = "Sample.weighting.adjust"
    df = sample._data
    design = sample._design

    if design.wgt is None:
        raise MethodError.not_applicable(
            where=ctx,
            method="adjust",
            reason="Sample weight is None. Set design.wgt before calling adjust().",
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
    if not isinstance(resp_status, str) or resp_status not in df.columns:
        raise MethodError.invalid_choice(
            where=ctx,
            param="resp_status",
            got=resp_status,
            allowed=list(df.columns),
            hint="`resp_status` must be a string naming an existing column.",
        )

    existing_cols = set(df.columns)
    if wgt_name in existing_cols:
        raise MethodError.not_applicable(
            where=ctx,
            method="adjust",
            reason=f"Column '{wgt_name}' already exists. Choose a different wgt_name.",
        )

    spec = build_cells(df, cells, where, where=ctx)

    wgt_arr = df.get_column(wgt).to_numpy().astype(np.float64)
    resp_status_arr = df.get_column(resp_status).to_numpy()
    resp_codes = _encode_resp_status(resp_status_arr, resp_mapping)

    adj_wgt_arr = _apply_nr(wgt_arr.reshape(-1, 1), spec, resp_codes, unknown_to_inelig)[:, 0]

    df = df.with_columns(pl.Series(wgt_name, adj_wgt_arr))

    if update_design_wgts:
        sample._push_design()
        # Provenance only: R carries no variance record for non-response
        # either. Modelling the adjustment's own variance is a different
        # estimator (SUDAAN's WTADJUST), not the calibration sweep.
        sample._design = sample._design.update(
            wgt=wgt_name,
            wgt_adjustment=WgtAdjustment(kind="nonresponse", prev_wgt=wgt, new_wgt=wgt_name),
        )

    if not ignore_reps and design.rep_wgts is not None:
        rep_cols = design.rep_wgts.columns
        wgts_arr = df.select(rep_cols).to_numpy()

        adj_wgts_arr = _apply_nr(wgts_arr, spec, resp_codes, unknown_to_inelig)

        n_reps = len(rep_cols)
        new_rep_names = [f"{wgt_name}{i}" for i in range(1, n_reps + 1)]
        wgts_df = pl.DataFrame(adj_wgts_arr, schema=new_rep_names)

        sample._data = df.hstack(wgts_df)
        df = sample._data

        if update_design_wgts:
            sample._design = sample._design.update(
                rep_wgts=msgspec.structs.replace(design.rep_wgts, prefix=wgt_name, n_reps=n_reps)
            )

    sample._data = df

    if respondents_only:
        # Filter from the encoded codes (0 == respondent) — the single source
        # of truth already used for the adjustment itself. Re-deriving the
        # mask from raw strings was case-sensitive while the encoder is not,
        # which could silently empty the sample.
        sample._data = sample._data.filter(pl.Series("__resp_mask__", resp_codes == 0))

    if trimming is not None:
        if update_design_wgts:
            sample = _apply_trim(
                sample,
                trimming,
                replace=True,
                update_design_wgts=True,
                where="Sample.weighting.adjust",
            )
        else:
            # The trim must target the freshly created adjusted weight (and
            # its replicate columns), not the caller's original design weight:
            # point the design at the new columns for the trim, then restore.
            original_design = sample._design
            tmp_design = original_design.update(wgt=wgt_name)
            if not ignore_reps and design.rep_wgts is not None:
                tmp_design = tmp_design.update(
                    rep_wgts=msgspec.structs.replace(
                        design.rep_wgts,
                        prefix=wgt_name,
                        n_reps=len(design.rep_wgts.columns),
                    )
                )
            sample._design = tmp_design
            sample = _apply_trim(
                sample,
                trimming,
                replace=True,
                update_design_wgts=False,
                where="Sample.weighting.adjust",
            )
            sample._design = original_design

    return sample
