# src/svy/weighting/normalization.py
"""
Weight normalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl


try:
    from svy_rs._internal import normalize as rust_normalize  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rust_normalize = None

from svy.core.types import DomainScalarMap, Number
from svy.errors import MethodError
from svy.weighting._helpers import _build_by_array, _normalize_dict_keys


if TYPE_CHECKING:
    from collections.abc import Sequence

    from svy.core.sample import Sample


def _encode_groups(by_arr: np.ndarray | None, n: int) -> np.ndarray | None:
    if by_arr is None:
        return None
    if np.issubdtype(by_arr.dtype, np.integer):
        return by_arr.astype(np.int64)
    mapping: dict = {}
    next_id = 0
    result = np.empty(len(by_arr), dtype=np.int64)
    for i, val in enumerate(by_arr):
        if val not in mapping:
            mapping[val] = next_id
            next_id += 1
        result[i] = mapping[val]
    return result


def _build_control_array(
    controls: DomainScalarMap | Number | None,
    by_arr: np.ndarray | None,
    group_ids: np.ndarray | None,
) -> np.ndarray | None:
    if controls is None:
        return None
    if by_arr is None:
        return np.array([float(controls)], dtype=np.float64)  # type: ignore[arg-type]
    _, idx = np.unique(group_ids, return_index=True)
    ordered_labels = by_arr[np.sort(idx)]
    if isinstance(controls, dict):
        controls = _normalize_dict_keys(controls)
        return np.array([float(controls[lbl]) for lbl in ordered_labels], dtype=np.float64)
    return np.full(len(ordered_labels), float(controls), dtype=np.float64)  # type: ignore[arg-type]


def normalize(
    sample: Sample,
    controls: DomainScalarMap | Number | None = None,
    *,
    by: str | Sequence[str] | None = None,
    wgt_name: str = "norm_wgt",
    ignore_reps: bool = False,
    update_design_wgts: bool = True,
) -> Sample:
    where = "Sample.weighting.normalize"
    df = sample._data
    design = sample._design

    if design.wgt is None:
        raise MethodError.not_applicable(
            where=where,
            method="normalize",
            reason="Sample weight is None. Set design.wgt before calling normalize().",
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

    existing_cols = set(df.columns)
    if wgt_name in existing_cols:
        raise MethodError.not_applicable(
            where=where,
            method="normalize",
            reason=f"Column '{wgt_name}' already exists. Choose a different wgt_name.",
        )

    by_arr = _build_by_array(df, by, where=where)

    wgt_arr = df.get_column(wgt).to_numpy().astype(np.float64)

    group_ids = _encode_groups(by_arr, len(wgt_arr))
    control_arr = _build_control_array(controls, by_arr, group_ids)

    assert rust_normalize is not None  # noqa: S101
    samp_norm = rust_normalize(
        wgt_arr.reshape(-1, 1),
        group_ids,
        control_arr,
    )[:, 0]

    df = df.with_columns(pl.Series(name=wgt_name, values=samp_norm))

    if update_design_wgts:
        sample._design = sample._design.update(wgt=wgt_name)

    if not ignore_reps and design.rep_wgts is not None:
        rep_cols = design.rep_wgts.columns
        if rep_cols:
            wgts_arr = df.select(rep_cols).to_numpy()
            adj_wgts_arr = rust_normalize(wgts_arr, group_ids, control_arr)
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
    return sample
