# src/svy/weighting/base.py
"""
Weighting facade.

The Weighting class is a thin dispatcher. Each method is a one-liner that
delegates to the corresponding module-level function. No logic lives here.

Adding a new weighting method:
  1. Implement it in svy/weighting/<method>.py as a module function.
  2. Import it here and add a one-liner delegation method.
  3. If it also has a pure algorithm, put that in svy/engine/weighting/adj_<method>.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np

from svy.core.terms import Feature
from svy.core.types import Category, ControlsType, DomainScalarMap, Number
from svy.utils.random_state import RandomState
from svy.weighting.adjustment import adjust as _adjust
from svy.weighting.calibration import build_aux_matrix as _build_aux_matrix
from svy.weighting.calibration import calibrate as _calibrate
from svy.weighting.calibration import calibrate_matrix as _calibrate_matrix
from svy.weighting.calibration import control_aux_template as _control_aux_template
from svy.weighting.normalization import normalize as _normalize
from svy.weighting.poststratification import poststratify as _poststratify
from svy.weighting.raking import controls_margins_template as _controls_margins_template
from svy.weighting.raking import rake as _rake
from svy.weighting.replication import create_brr_wgts as _create_brr_wgts
from svy.weighting.replication import create_bs_wgts as _create_bs_wgts
from svy.weighting.replication import create_jk_wgts as _create_jk_wgts
from svy.weighting.replication import create_sdr_wgts as _create_sdr_wgts
from svy.weighting.replication import create_variance_strata as _create_variance_strata
from svy.weighting.trimming import trim as _trim
from svy.weighting.types import TrimConfig

if TYPE_CHECKING:
    from svy.core.sample import Sample


class Weighting:
    def __init__(self, sample: Any) -> None:
        self._sample = sample

    # ------------------------------------------------------------------ #
    # Variance strata / replicate weights
    # ------------------------------------------------------------------ #

    def create_variance_strata(
        self,
        *,
        method: Literal["brr", "jk2"],
        order_by: str | Sequence[str] | None = None,
        shuffle: bool = False,
        into: str = "svy_var_stratum",
        rstate: int | None = None,
    ) -> Any:
        return _create_variance_strata(
            self._sample,
            method=method,
            order_by=order_by,
            shuffle=shuffle,
            into=into,
            rstate=rstate,
        )

    def create_brr_wgts(
        self,
        n_reps: int | None = None,
        *,
        rep_prefix: str | None = None,
        fay_coef: float = 0.0,
        rstate: int | None = None,
        drop_nulls: bool = False,
    ) -> Any:
        return _create_brr_wgts(
            self._sample,
            n_reps,
            rep_prefix=rep_prefix,
            fay_coef=fay_coef,
            rstate=rstate,
            drop_nulls=drop_nulls,
        )

    def create_jk_wgts(
        self,
        *,
        paired: bool = False,
        rep_prefix: str | None = None,
        rstate: int | None = None,
        drop_nulls: bool = False,
    ) -> Any:
        return _create_jk_wgts(
            self._sample,
            paired=paired,
            rep_prefix=rep_prefix,
            rstate=rstate,
            drop_nulls=drop_nulls,
        )

    def create_bs_wgts(
        self,
        n_reps: int = 500,
        *,
        rep_prefix: str | None = None,
        drop_nulls: bool = False,
        rstate: RandomState = None,
    ) -> Any:
        return _create_bs_wgts(
            self._sample,
            n_reps,
            rep_prefix=rep_prefix,
            drop_nulls=drop_nulls,
            rstate=rstate,
        )

    def create_sdr_wgts(
        self,
        n_reps: int = 4,
        *,
        rep_prefix: str | None = None,
        order_col: str | None = None,
        drop_nulls: bool = False,
    ) -> Any:
        return _create_sdr_wgts(
            self._sample,
            n_reps,
            rep_prefix=rep_prefix,
            order_col=order_col,
            drop_nulls=drop_nulls,
        )

    # ------------------------------------------------------------------ #
    # Adjustment (non-response)
    # ------------------------------------------------------------------ #

    def adjust(
        self,
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
    ) -> Any:
        return _adjust(
            self._sample,
            resp_status,
            by,
            resp_mapping=resp_mapping,
            wgt_name=wgt_name,
            ignore_reps=ignore_reps,
            unknown_to_inelig=unknown_to_inelig,
            update_design_wgts=update_design_wgts,
            respondents_only=respondents_only,
            trimming=trimming,
        )

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #

    def normalize(
        self,
        controls: DomainScalarMap | Number | None = None,
        *,
        by: str | Sequence[str] | None = None,
        wgt_name: str = "norm_wgt",
        ignore_reps: bool = False,
        update_design_wgts: bool = True,
    ) -> Any:
        return _normalize(
            self._sample,
            controls,
            by=by,
            wgt_name=wgt_name,
            ignore_reps=ignore_reps,
            update_design_wgts=update_design_wgts,
        )

    # ------------------------------------------------------------------ #
    # Post-stratification
    # ------------------------------------------------------------------ #

    def poststratify(
        self,
        controls: DomainScalarMap | Number | None = None,
        *,
        factors: DomainScalarMap | Number | None = None,
        by: str | Sequence[str] | None = None,
        wgt_name: str = "ps_wgt",
        ignore_reps: bool = False,
        update_design_wgts: bool = True,
        strict: bool = True,
        trimming: TrimConfig | None = None,
    ) -> Any:
        return _poststratify(
            self._sample,
            controls,
            factors=factors,
            by=by,
            wgt_name=wgt_name,
            ignore_reps=ignore_reps,
            update_design_wgts=update_design_wgts,
            strict=strict,
            trimming=trimming,
        )

    # ------------------------------------------------------------------ #
    # Raking
    # ------------------------------------------------------------------ #

    def controls_margins_template(
        self,
        *,
        margins: Mapping[str, str],
        cat_na: str = "level",
        na_label: str = "__NA__",
    ) -> dict[str, dict[Category, float]]:
        return _controls_margins_template(
            self._sample,
            margins=margins,
            cat_na=cat_na,
            na_label=na_label,
        )

    def rake(
        self,
        *,
        controls: ControlsType | None = None,
        factors: ControlsType | None = None,
        wgt_name: str = "rk_wgt",
        ignore_reps: bool = False,
        ll_bound: float | None = None,
        up_bound: float | None = None,
        tol: float = 1e-4,
        max_iter: int = 100,
        display_iter: bool = False,
        update_design_wgts: bool = True,
        strict: bool = True,
        trimming: TrimConfig | None = None,
    ) -> Sample:
        return _rake(
            self._sample,
            controls=controls,
            factors=factors,
            wgt_name=wgt_name,
            ignore_reps=ignore_reps,
            ll_bound=ll_bound,
            up_bound=up_bound,
            tol=tol,
            max_iter=max_iter,
            display_iter=display_iter,
            update_design_wgts=update_design_wgts,
            strict=strict,
            trimming=trimming,
        )

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def control_aux_template(
        self,
        *,
        x: Sequence[Feature],
        by: str | Sequence[str] | None = None,
        by_na: Literal["error", "level", "drop"] = "error",
        na_label: str = "__NA__",
    ) -> dict[Category, Number] | dict[Category, dict[Category, Number]]:
        return _control_aux_template(
            self._sample,
            x=x,
            by=by,
            by_na=by_na,
            na_label=na_label,
        )

    def build_aux_matrix(
        self,
        *,
        x: Sequence[Feature],
        by: str | Sequence[str] | None = None,
        by_na: Literal["error", "level", "drop"] = "error",
        na_label: str = "__NA__",
    ) -> tuple[np.ndarray, dict[Category, Number] | dict[Category, dict[Category, Number]]]:
        return _build_aux_matrix(
            self._sample,
            x=x,
            by=by,
            by_na=by_na,
            na_label=na_label,
        )

    def calibrate(
        self,
        *,
        controls: dict[Feature, Any],
        by: str | Sequence[str] | None = None,
        scale: Number | list[Number] | np.ndarray = 1.0,
        bounded: bool = False,
        wgt_name: str = "calib_wgt",
        update_design_wgts: bool = True,
        ignore_reps: bool = False,
        strict: bool = True,
        trimming: TrimConfig | None = None,
    ) -> Any:
        return _calibrate(
            self._sample,
            controls=controls,
            by=by,
            scale=scale,
            bounded=bounded,
            wgt_name=wgt_name,
            update_design_wgts=update_design_wgts,
            ignore_reps=ignore_reps,
            strict=strict,
            trimming=trimming,
        )

    def calibrate_matrix(
        self,
        *,
        aux_vars: np.ndarray,
        control: Any,
        by: str | Sequence[str] | None = None,
        scale: Number | Sequence[Number] | np.ndarray = 1.0,
        wgt_name: str = "calib_wgt",
        update_design_wgts: bool = True,
        labels: Sequence[Category] | None = None,
        weights_only: bool = False,
        bounded: bool = False,
        ignore_reps: bool = False,
        strict: bool = True,
        trimming: TrimConfig | None = None,
    ) -> Any:
        return _calibrate_matrix(
            self._sample,
            aux_vars=aux_vars,
            control=control,
            by=by,
            scale=scale,
            wgt_name=wgt_name,
            update_design_wgts=update_design_wgts,
            labels=labels,
            weights_only=weights_only,
            bounded=bounded,
            ignore_reps=ignore_reps,
            strict=strict,
            trimming=trimming,
        )

    # ------------------------------------------------------------------ #
    # Trimming
    # ------------------------------------------------------------------ #

    def trim(
        self,
        upper=None,
        lower=None,
        by=None,
        redistribute: bool = True,
        min_cell_size: int = 10,
        max_iter: int = 10,
        tol: float = 1e-6,
        wgt_name: str | None = "trim_wgt",
        update_design_wgts: bool = True,
    ) -> "Sample":
        return _trim(
            self._sample,
            upper=upper,
            lower=lower,
            by=by,
            redistribute=redistribute,
            min_cell_size=min_cell_size,
            max_iter=max_iter,
            tol=tol,
            wgt_name=wgt_name,
            update_design_wgts=update_design_wgts,
        )
