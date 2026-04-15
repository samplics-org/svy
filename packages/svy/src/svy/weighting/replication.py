# src/svy/weighting/replication.py
"""
Replicate weight creation: BRR, Jackknife, Bootstrap, SDR.

Each function takes a Sample and returns a Sample (for chaining).
The Weighting class in base.py delegates to these functions directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import polars as pl


try:
    from svy_rs._internal import (
        create_bootstrap_wgts as rust_create_bootstrap_wgts,  # type: ignore[import-untyped]
    )
    from svy_rs._internal import (
        create_brr_wgts as rust_create_brr_wgts,  # type: ignore[import-untyped]
    )
    from svy_rs._internal import (
        create_jk_wgts as rust_create_jk_wgts,  # type: ignore[import-untyped]
    )
    from svy_rs._internal import (
        create_sdr_wgts as rust_create_sdr_wgts,  # type: ignore[import-untyped]
    )
except ImportError:  # pragma: no cover
    rust_create_bootstrap_wgts = None
    rust_create_brr_wgts = None
    rust_create_jk_wgts = None
    rust_create_sdr_wgts = None

from svy.core.design import RepWeights
from svy.core.enumerations import EstimationMethod
from svy.errors import DimensionError, MethodError
from svy.utils.checks import drop_missing
from svy.utils.random_state import RandomState, resolve_random_state
from svy.weighting._helpers import (
    _name_rep_cols,
    _to_float_array,
    _to_int_array,
)


if TYPE_CHECKING:
    from svy.core.sample import Sample


def create_variance_strata(
    sample: Sample,
    *,
    method: Literal["brr", "jk2"],
    order_by: str | Sequence[str] | None = None,
    shuffle: bool = False,
    into: str = "svy_var_stratum",
    rstate: int | None = None,
) -> Sample:
    where = "Sample.weighting.create_variance_strata"
    df = sample._data
    design = sample._design

    if design.psu is None:
        raise MethodError.not_applicable(
            where=where,
            method="create_variance_strata",
            reason="Design must have PSU defined.",
        )

    _method = method.lower()
    if _method not in ("brr", "jk2"):
        raise MethodError.invalid_choice(
            where=where,
            param="method",
            got=_method,
            allowed=["brr", "jk2"],
        )

    psu_col = design.psu
    orig_stratum_col = design.stratum

    # When stratum is a tuple (multi-column), use the internal concatenated
    # column which already exists in the data as a single string column.
    if isinstance(orig_stratum_col, (tuple, list)):
        _internal = getattr(sample, "_internal_design", None) or {}
        stratum_col_for_grouping = _internal.get("stratum")
        if stratum_col_for_grouping is None or stratum_col_for_grouping not in df.columns:
            raise MethodError.not_applicable(
                where=where,
                method="create_variance_strata",
                reason=(
                    f"Multi-column stratum {orig_stratum_col} requires an internal "
                    f"concatenated column, but it was not found in the data."
                ),
                hint="Ensure the Sample was constructed with the stratum columns present.",
            )
        # Use individual source columns for select, but the concat column for grouping
        stratum_source_cols = list(orig_stratum_col)
    elif orig_stratum_col is not None:
        stratum_col_for_grouping = orig_stratum_col
        stratum_source_cols = [orig_stratum_col]
    else:
        stratum_col_for_grouping = None
        stratum_source_cols = []

    if order_by is None:
        order_cols = []
    elif isinstance(order_by, str):
        order_cols = [order_by]
    else:
        order_cols = list(order_by)

    missing_cols = [col for col in order_cols if col not in df.columns]
    if missing_cols:
        raise MethodError.invalid_choice(
            where=where,
            param="order_by",
            got=missing_cols,
            allowed=df.columns,
            hint="Check column names.",
        )

    select_cols = [psu_col]
    # Include stratum source columns AND the concat column (if different)
    for c in stratum_source_cols:
        if c not in select_cols:
            select_cols.append(c)
    if stratum_col_for_grouping and stratum_col_for_grouping not in select_cols:
        select_cols.append(stratum_col_for_grouping)
    select_cols.extend([c for c in order_cols if c not in select_cols])

    psu_df = df.select(select_cols).unique(maintain_order=True)

    sort_cols = []
    if stratum_col_for_grouping:
        sort_cols.append(stratum_col_for_grouping)
    sort_cols.extend(order_cols)
    if psu_col not in sort_cols:
        sort_cols.append(psu_col)

    psu_df = psu_df.sort(sort_cols)

    psu_list = psu_df[psu_col].to_list()
    n_psus = len(psu_list)

    if stratum_col_for_grouping:
        orig_strata = psu_df[stratum_col_for_grouping].to_numpy()
    else:
        orig_strata = np.zeros(n_psus, dtype=np.int64)

    unique_orig, counts = np.unique(orig_strata, return_counts=True)
    stratum_counts = dict(zip(unique_orig.tolist(), counts.tolist()))

    small_strata = [(s, c) for s, c in stratum_counts.items() if c < 2]
    if small_strata:
        raise DimensionError(
            title="Insufficient PSUs per stratum",
            detail=(
                f"All strata must have at least 2 PSUs. "
                f"Found {len(small_strata)} strata with fewer."
            ),
            code="INSUFFICIENT_PSU",
            where=where,
            param="stratum",
            expected="≥2 PSUs per stratum",
            got=f"{small_strata[:5]}{'...' if len(small_strata) > 5 else ''}",
            hint="Combine small strata or check design specification.",
        )

    if _method == "brr":
        odd_strata = [(s, c) for s, c in stratum_counts.items() if c % 2 == 1]
        if odd_strata:
            raise DimensionError(
                title="Odd PSU counts for BRR",
                detail=(
                    f"BRR requires even number of PSUs per stratum. "
                    f"Found {len(odd_strata)} strata with odd counts."
                ),
                code="ODD_PSU_COUNT",
                where=where,
                param="stratum",
                expected="Even PSU count per stratum",
                got=f"{odd_strata[:5]}{'...' if len(odd_strata) > 5 else ''}",
                hint="Use method='jk2' which allows 2-3 PSUs per stratum.",
            )

    var_strata = np.empty(n_psus, dtype=np.int64)
    rng = np.random.default_rng(rstate) if shuffle and not order_cols else None
    var_stratum_counter = 0

    for orig_str in unique_orig:
        mask = orig_strata == orig_str
        indices = np.where(mask)[0]
        n_in_stratum = len(indices)

        if rng is not None:
            indices = indices.copy()
            rng.shuffle(indices)

        if _method == "brr":
            for i in range(0, n_in_stratum, 2):
                var_strata[indices[i]] = var_stratum_counter
                var_strata[indices[i + 1]] = var_stratum_counter
                var_stratum_counter += 1
        else:  # jk2
            if n_in_stratum % 2 == 1:
                for i in range(0, n_in_stratum - 3, 2):
                    var_strata[indices[i]] = var_stratum_counter
                    var_strata[indices[i + 1]] = var_stratum_counter
                    var_stratum_counter += 1
                var_strata[indices[-3]] = var_stratum_counter
                var_strata[indices[-2]] = var_stratum_counter
                var_strata[indices[-1]] = var_stratum_counter
                var_stratum_counter += 1
            else:
                for i in range(0, n_in_stratum, 2):
                    var_strata[indices[i]] = var_stratum_counter
                    var_strata[indices[i + 1]] = var_stratum_counter
                    var_stratum_counter += 1

    psu_to_var_stratum = dict(zip(psu_list, var_strata))
    psu_vec = df[psu_col].to_list()
    obs_var_strata = np.array([psu_to_var_stratum[p] for p in psu_vec], dtype=np.int64)

    sample._data = df.with_columns(pl.Series(name=into, values=obs_var_strata))
    sample._design = sample._design.update(stratum=into)

    return sample


def create_brr_wgts(
    sample: Sample,
    n_reps: int | None = None,
    *,
    rep_prefix: str | None = None,
    fay_coef: float = 0.0,
    rstate: int | None = None,
    drop_nulls: bool = False,
) -> Sample:
    df = sample._data
    design = sample._design

    if design.stratum is None:
        raise MethodError.not_applicable(
            where="Sample.weighting.create_brr_wgts",
            method="create_brr_wgts",
            reason="BRR requires stratum in Design (got stratum=None).",
            hint="Call sample.weighting.create_variance_strata() first.",
        )
    if design.psu is None:
        raise MethodError.not_applicable(
            where="Sample.weighting.create_brr_wgts",
            method="create_brr_wgts",
            reason="BRR requires psu in Design (got psu=None).",
        )

    if drop_nulls:
        needed = list({c for c in [design.wgt, design.stratum, design.psu] if isinstance(c, str)})
        data = drop_missing(df=df, cols=needed, treat_infinite_as_missing=True)
    else:
        data = df

    main_weights = _to_float_array(data, design.wgt, len(data))
    stratum_int = _to_int_array(data, design.stratum)
    psu_int = _to_int_array(data, design.psu)

    assert rust_create_brr_wgts is not None  # noqa: S101
    rep_mat, df_val = rust_create_brr_wgts(
        main_weights,
        stratum_int,
        psu_int,
        n_reps,
        fay_coef,
        rstate,
    )
    n_reps_actual = rep_mat.shape[1]

    rep_prefix = rep_prefix or design.wgt
    rep_cols = _name_rep_cols(rep_prefix, n_reps_actual)
    rep_dicts = {col: rep_mat[:, i] for i, col in enumerate(rep_cols)}
    sample._data = data.with_columns(
        [pl.Series(name=col, values=vals) for col, vals in rep_dicts.items()]
    )

    sample._design = sample._design.fill_missing(
        rep_wgts=RepWeights(
            method=EstimationMethod.BRR,
            prefix=rep_prefix,
            n_reps=n_reps_actual,
            fay_coef=fay_coef,
            df=df_val,
        )
    )

    return sample


def create_jk_wgts(
    sample: Sample,
    *,
    paired: bool = False,
    rep_prefix: str | None = None,
    rstate: int | None = None,
    drop_nulls: bool = False,
) -> Sample:
    df = sample._data
    design = sample._design

    if design.psu is None:
        raise MethodError.not_applicable(
            where="Sample.weighting.create_jk_wgts",
            method="create_jk_wgts",
            reason="Jackknife requires psu in Design (got psu=None).",
        )

    if drop_nulls:
        needed = list({c for c in [design.wgt, design.stratum, design.psu] if isinstance(c, str)})
        data = drop_missing(df=df, cols=needed, treat_infinite_as_missing=True)
    else:
        data = df

    main_weights = _to_float_array(data, design.wgt, len(data))
    psu_int = _to_int_array(data, design.psu)
    stratum_int = _to_int_array(data, design.stratum)

    assert rust_create_jk_wgts is not None  # noqa: S101
    rep_mat, df_val = rust_create_jk_wgts(
        main_weights,
        psu_int,
        stratum_int,
        paired,
        rstate,
    )
    n_reps = rep_mat.shape[1]

    rep_prefix = rep_prefix or design.wgt
    rep_cols = _name_rep_cols(rep_prefix, n_reps)
    rep_dicts = {col: rep_mat[:, i] for i, col in enumerate(rep_cols)}
    sample._data = data.with_columns(
        [pl.Series(name=col, values=vals) for col, vals in rep_dicts.items()]
    )

    sample._design = sample._design.fill_missing(
        rep_wgts=RepWeights(
            method=EstimationMethod.JACKKNIFE,
            prefix=rep_prefix,
            n_reps=n_reps,
            df=df_val,
        )
    )

    return sample


def create_bs_wgts(
    sample: Sample,
    n_reps: int = 500,
    *,
    rep_prefix: str | None = None,
    drop_nulls: bool = False,
    rstate: RandomState = None,
) -> Sample:
    df = sample._data
    design = sample._design

    if design.psu is None:
        raise MethodError.not_applicable(
            where="Sample.weighting.create_bs_wgts",
            method="create_bs_wgts",
            reason="Bootstrap requires psu in Design (got psu=None).",
        )
    if n_reps is None:
        raise MethodError.not_applicable(
            where="Sample.weighting.create_bs_wgts",
            method="create_bs_wgts",
            reason="n_reps must be specified for Bootstrap.",
        )

    if drop_nulls:
        needed = list({c for c in [design.wgt, design.stratum, design.psu] if isinstance(c, str)})
        data = drop_missing(df=df, cols=needed, treat_infinite_as_missing=True)
    else:
        data = df

    main_weights = _to_float_array(data, design.wgt, len(data))
    psu_int = _to_int_array(data, design.psu)
    stratum_int = _to_int_array(data, design.stratum)

    rng = resolve_random_state(rstate)
    seed = (
        int(rng.integers(0, 2**63 - 1))
        if hasattr(rng, "integers")
        else int(rng.randint(0, 2**31 - 1))
    )

    assert rust_create_bootstrap_wgts is not None  # noqa: S101
    rep_mat, df_val = rust_create_bootstrap_wgts(
        main_weights,
        psu_int,
        n_reps,
        stratum_int,
        seed,
    )

    rep_prefix = rep_prefix or design.wgt
    rep_cols = _name_rep_cols(rep_prefix, n_reps)
    rep_dicts = {col: rep_mat[:, i] for i, col in enumerate(rep_cols)}
    sample._data = data.with_columns(
        [pl.Series(name=col, values=vals) for col, vals in rep_dicts.items()]
    )

    sample._design = sample._design.update_rep_weights(
        method=EstimationMethod.BOOTSTRAP,
        prefix=rep_prefix,
        n_reps=n_reps,
        df=df_val,
    )

    return sample


def create_sdr_wgts(
    sample: Sample,
    n_reps: int = 4,
    *,
    rep_prefix: str | None = None,
    order_col: str | None = None,
    drop_nulls: bool = False,
) -> Sample:
    df = sample._data
    design = sample._design

    if n_reps < 2:
        raise MethodError.invalid_range(
            where="Sample.weighting.create_sdr_wgts",
            param="n_reps",
            got=n_reps,
            min_=2,
            hint="SDR requires at least 2 replicates.",
        )

    if drop_nulls:
        needed = list({c for c in [design.wgt, design.stratum] if isinstance(c, str)})
        if order_col:
            needed.append(order_col)
        data = drop_missing(df=df, cols=needed, treat_infinite_as_missing=True)
    else:
        data = df

    main_weights = _to_float_array(data, design.wgt, len(data))
    stratum_int = _to_int_array(data, design.stratum)

    order_int: np.ndarray | None = None
    if order_col and order_col in data.columns:
        order_int = _to_int_array(data, order_col)

    assert rust_create_sdr_wgts is not None  # noqa: S101
    rep_mat, df_val = rust_create_sdr_wgts(
        main_weights,
        n_reps,
        stratum_int,
        order_int,
    )

    rep_prefix = rep_prefix or design.wgt
    rep_cols = _name_rep_cols(rep_prefix, n_reps)
    rep_dicts = {col: rep_mat[:, i] for i, col in enumerate(rep_cols)}
    sample._data = data.with_columns(
        [pl.Series(name=col, values=vals) for col, vals in rep_dicts.items()]
    )

    sample._design = sample._design.update_rep_weights(
        method=EstimationMethod.SDR,
        prefix=rep_prefix,
        n_reps=n_reps,
        df=df_val,
    )

    return sample
