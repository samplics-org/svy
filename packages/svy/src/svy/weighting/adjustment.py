# src/svy/weighting/adjustment.py
"""
Non-response weight adjustment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl


try:
    from svy_rs._internal import adjust_nr as rust_adjust_nr  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_adjust_nr = None

from svy.core.types import DomainScalarMap
from svy.errors import MethodError
from svy.weighting._helpers import _build_by_array
from svy.weighting.trimming import _run_trim as _apply_trim
from svy.weighting.types import TrimConfig


if TYPE_CHECKING:
    from collections.abc import Sequence

    from svy.core.sample import Sample

_CANONICAL_TO_INT: dict[str, int] = {"rr": 0, "nr": 1, "in": 2, "uk": 3}


def _encode_resp_status(
    resp_status_arr: np.ndarray,
    resp_mapping: DomainScalarMap | None,
) -> np.ndarray:
    codes = np.zeros(len(resp_status_arr), dtype=np.int64)
    lower = np.char.lower(resp_status_arr.astype(str))

    if resp_mapping is not None:
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
            codes[lower == str(data_label).lower()] = _CANONICAL_TO_INT[key_lower]
    else:
        for label, code in _CANONICAL_TO_INT.items():
            codes[lower == label] = code

    return codes


def _encode_adj_class(adj_class_arr: np.ndarray | None, n: int) -> np.ndarray:
    if adj_class_arr is None:
        return np.zeros(n, dtype=np.int64)
    if np.issubdtype(adj_class_arr.dtype, np.integer):
        return adj_class_arr.astype(np.int64)
    _, result = np.unique(adj_class_arr, return_inverse=True)
    return result.astype(np.int64)


def adjust(
    sample: Sample,
    resp_status: str,
    by: str | Sequence[str] | None,
    *,
    resp_mapping: DomainScalarMap | None = None,
    wgt_name: str = "nr_wgt",
    ignore_reps: bool = False,
    unknown_to_inelig: bool = True,
    update_design_wgts: bool = True,
    respondents_only: bool = True,
    trimming: TrimConfig | None = None,
) -> Sample:
    where = "Sample.weighting.adjust"
    df = sample._data
    design = sample._design

    if design.wgt is None:
        raise MethodError.not_applicable(
            where=where,
            method="adjust",
            reason="Sample weight is None. Set design.wgt before calling adjust().",
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
    if not isinstance(resp_status, str) or resp_status not in df.columns:
        raise MethodError.invalid_choice(
            where=where,
            param="resp_status",
            got=resp_status,
            allowed=list(df.columns),
            hint="`resp_status` must be a string naming an existing column.",
        )

    existing_cols = set(df.columns)
    if wgt_name in existing_cols:
        raise MethodError.not_applicable(
            where=where,
            method="adjust",
            reason=f"Column '{wgt_name}' already exists. Choose a different wgt_name.",
        )

    adj_class_arr = _build_by_array(df, by, where=where)

    wgt_arr = df.get_column(wgt).to_numpy().astype(np.float64)
    resp_status_arr = df.get_column(resp_status).to_numpy()

    adj_class_int = _encode_adj_class(adj_class_arr, len(wgt_arr))
    resp_codes = _encode_resp_status(resp_status_arr, resp_mapping)

    assert rust_adjust_nr is not None  # noqa: S101
    adj_wgt_arr = rust_adjust_nr(
        wgt_arr.reshape(-1, 1),
        adj_class_int,
        resp_codes,
        unknown_to_inelig,
    )[:, 0]

    df = df.with_columns(pl.Series(wgt_name, adj_wgt_arr))

    if update_design_wgts:
        sample._design = sample._design.update(wgt=wgt_name)

    if not ignore_reps and design.rep_wgts is not None:
        rep_cols = design.rep_wgts.columns
        wgts_arr = df.select(rep_cols).to_numpy()

        adj_wgts_arr = rust_adjust_nr(
            wgts_arr,
            adj_class_int,
            resp_codes,
            unknown_to_inelig,
        )

        n_reps = len(rep_cols)
        new_rep_names = [f"{wgt_name}{i}" for i in range(1, n_reps + 1)]
        wgts_df = pl.DataFrame(adj_wgts_arr, schema=new_rep_names)

        sample._data = df.hstack(wgts_df)
        df = sample._data

        if update_design_wgts:
            sample._design = sample._design.update_rep_weights(
                method=design.rep_wgts.method,
                prefix=wgt_name,
                n_reps=n_reps,
                fay_coef=design.rep_wgts.fay_coef,
                df=design.rep_wgts.df,
            )

    sample._data = df

    if respondents_only:

        def _resp_mask_expr(col: str, mapping: dict | None) -> pl.Expr:
            if mapping is None:
                return pl.col(col) == "rr"
            rr_codes = mapping.get("rr", "rr")
            if isinstance(rr_codes, (list, tuple, set, np.ndarray)):
                return pl.col(col).is_in(list(rr_codes))
            return pl.col(col) == rr_codes

        sample._data = sample._data.filter(_resp_mask_expr(resp_status, resp_mapping))

    if trimming is not None:
        sample = _apply_trim(
            sample,
            trimming,
            replace=True,
            update_design_wgts=update_design_wgts,
            where="Sample.weighting.adjust",
        )

    return sample
