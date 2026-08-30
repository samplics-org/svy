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

from svy.core.repwgts import BootstrapKind
from svy.core.terms import Feature
from svy.core.types import Category, ControlsType, DomainScalarMap, Number, WhereArg
from svy.errors import MethodError
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
from svy.weighting.standardization import standardize as _standardize
from svy.weighting.trimming import trim as _trim
from svy.weighting.types import TrimConfig


if TYPE_CHECKING:
    from svy.core.sample import Sample


# Renamed and removed parameters, kept only to make the break legible. These
# are not aliases: the call still fails, but it names the replacement instead
# of reporting an unknown keyword.
_RENAMED: dict[str, dict[str, str]] = {
    "adjust": {"by": "cells"},
    "normalize": {"by": "cells"},
    "poststratify": {"by": "cells", "factors": "shares"},
    "rake": {"factors": "shares"},
}


def _reject_legacy_kwargs(method: str, kwargs: dict[str, Any]) -> None:
    if not kwargs:
        return
    renames = _RENAMED.get(method, {})
    for old, new in renames.items():
        if old in kwargs:
            raise MethodError.not_applicable(
                where=f"Sample.weighting.{method}",
                method=method,
                reason=f"`{old}=` was renamed to `{new}=`",
                param=old,
                hint=(
                    f"Replace {old}= with {new}=."
                    + (
                        " shares are normalized internally, so a vector that does not "
                        "sum to 1 now pins composition instead of rescaling the total."
                        if new == "shares"
                        else ""
                    )
                ),
            )
    unknown = next(iter(kwargs))
    raise TypeError(f"{method}() got an unexpected keyword argument {unknown!r}")


class Weighting:
    """Weight adjustments.

    Three relation parameters run through these methods, each with one meaning:

    ``cells=``
        Groups that each receive ONE derived adjustment factor. The adjustment
        is computed per cell, so the cells are what the adjustment pins.
    ``by=``
        Repeat the whole adjustment independently per group. Present only where
        it is not a second spelling of ``cells``: for ``adjust``, ``normalize``
        and ``poststratify``, ``by=g`` would be exactly ``cells=[g, *cells]``.
        This is the same ``by=`` as in estimation.
    ``where=``
        Scope. Matching rows receive the adjustment and the rest keep their
        previous weight, so the new column is complete. Contrast with
        estimation's ``where=``, which zero-weights for subpopulation variance.

    ``standardize`` is the one method taking both axes, because reusing one set
    of shares across domains is what standardization means.

    Targets, where a method takes them, follow one rule: a scalar is one cell
    and a dict is many; ``controls`` sets the total and ``shares`` preserves it.
    """

    def __init__(self, sample: Any) -> None:
        self._sample = sample

    def _target(self, inplace: bool) -> Any:
        """The sample this operation works on: the caller's, or a private fork.

        Every function in ``weighting/`` builds its result by rebinding
        ``sample._data`` and ``sample._design`` as it goes, so the fork has to be
        made here, before the work starts, rather than at the end the way
        ``wrangling._helpers._resolve_target`` does it. Same contract either way:
        ``inplace=False`` leaves the caller's sample untouched, ``inplace=True``
        rewrites it, and both return a ``Sample`` so chaining is unaffected.
        """
        return self._sample if inplace else self._sample._fork()

    # ------------------------------------------------------------------ #
    # Variance strata / replicate weights
    # ------------------------------------------------------------------ #

    def create_brr_wgts(
        self,
        n_reps: int | None = None,
        *,
        stratum: str | None = None,
        psu: str | None = None,
        stratum_name: str = "svy_var_stratum",
        order_by: str | Sequence[str] | None = None,
        shuffle: bool = False,
        rep_prefix: str | None = None,
        fay_coef: float = 0.0,
        rstate: int | None = None,
        drop_nulls: bool = False,
        inplace: bool = False,
    ) -> Any:
        return _create_brr_wgts(
            self._target(inplace),
            n_reps,
            stratum=stratum,
            psu=psu,
            stratum_name=stratum_name,
            order_by=order_by,
            shuffle=shuffle,
            rep_prefix=rep_prefix,
            fay_coef=fay_coef,
            rstate=rstate,
            drop_nulls=drop_nulls,
        )

    def create_jk_wgts(
        self,
        *,
        paired: bool = False,
        stratum: str | None = None,
        psu: str | None = None,
        stratum_name: str = "svy_var_stratum",
        order_by: str | Sequence[str] | None = None,
        shuffle: bool = False,
        rep_prefix: str | None = None,
        rstate: int | None = None,
        drop_nulls: bool = False,
        inplace: bool = False,
    ) -> Any:
        return _create_jk_wgts(
            self._target(inplace),
            paired=paired,
            stratum=stratum,
            psu=psu,
            stratum_name=stratum_name,
            order_by=order_by,
            shuffle=shuffle,
            rep_prefix=rep_prefix,
            rstate=rstate,
            drop_nulls=drop_nulls,
        )

    def create_bs_wgts(
        self,
        n_reps: int = 500,
        *,
        kind: BootstrapKind = "rao-wu",
        stratum: str | None = None,
        psu: str | None = None,
        rep_prefix: str | None = None,
        drop_nulls: bool = False,
        rstate: RandomState = None,
        inplace: bool = False,
    ) -> Any:
        return _create_bs_wgts(
            self._target(inplace),
            n_reps,
            kind=kind,
            stratum=stratum,
            psu=psu,
            rep_prefix=rep_prefix,
            drop_nulls=drop_nulls,
            rstate=rstate,
        )

    def create_sdr_wgts(
        self,
        n_reps: int = 4,
        *,
        psu: str | None = None,
        rep_prefix: str | None = None,
        order_col: str | None = None,
        drop_nulls: bool = False,
        inplace: bool = False,
    ) -> Any:
        return _create_sdr_wgts(
            self._target(inplace),
            n_reps,
            psu=psu,
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
        inplace: bool = False,
        **_legacy: Any,
    ) -> Any:
        """Adjust weights for non-response.

        Redistributes the weight of non-respondents to respondents within each
        adjustment class.

        Parameters
        ----------
        resp_status : str
            Column of response statuses (rr / nr / in / uk, or mapped via
            ``resp_mapping``).
        cells : str | Sequence[str] | None
            Adjustment classes: each receives one factor, derived from the
            response statuses within it. None adjusts the sample as one class.
        where : WhereArg
            Scope. Rows outside it keep their weight whatever their status.

        Recorded as ``kind="nonresponse"`` and provenance only: variance treats
        the adjusted weights as fixed, matching R.
        """
        _reject_legacy_kwargs("adjust", _legacy)
        return _adjust(
            self._target(inplace),
            resp_status,
            cells,
            where=where,
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
        shares: DomainScalarMap | None = None,
        cells: str | Sequence[str] | None = None,
        where: WhereArg = None,
        wgt_name: str = "norm_wgt",
        ignore_reps: bool = False,
        update_design_wgts: bool = True,
        inplace: bool = False,
        **_legacy: Any,
    ) -> Any:
        """Rescale weights to a chosen total.

        Targets here are conveniences -- sum to n, sum to 1, a chosen level per
        cell -- not population constraints. The arithmetic is identical to
        ``poststratify`` for the same targets; what differs is the claim, and
        so the variance treatment.

        Parameters
        ----------
        controls : number | dict | None
            A number is the grand total and requires ``cells=None``; a dict is
            one target per cell. None normalizes to n, or to per-cell counts.
        shares : dict | None
            Composition per cell; the weight total carries through unchanged.
        cells : str | Sequence[str] | None
            Groups that each receive one factor.
        where : WhereArg
            Scope.

        Recorded as ``kind="normalization"`` and provenance only.
        """
        _reject_legacy_kwargs("normalize", _legacy)
        return _normalize(
            self._target(inplace),
            controls,
            shares=shares,
            cells=cells,
            where=where,
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
        shares: DomainScalarMap | None = None,
        cells: str | Sequence[str] | None = None,
        where: WhereArg = None,
        wgt_name: str = "ps_wgt",
        ignore_reps: bool = False,
        update_design_wgts: bool = True,
        strict: bool = True,
        trimming: TrimConfig | None = None,
        inplace: bool = False,
        **_legacy: Any,
    ) -> Any:
        """Adjust weights so cells match known population figures.

        Unlike ``normalize``, the targets are population constraints, so the
        adjustment is variance-consumed: a poststratified total of a pinned
        margin has no sampling error left in it.

        Parameters
        ----------
        controls : number | dict | None
            Absolute totals. A number is the grand total -- a known population
            size -- and requires ``cells=None``; a dict is one total per cell,
            keyed by cell value, or by tuple in ``cells`` order for several
            columns.
        shares : dict | None
            Known cell PROPORTIONS, for when the composition is known but the
            population count is not. Normalized internally, so counts or
            proportions both work, and the weight total carries through.
        cells : str | Sequence[str] | None
            The post-strata: each receives one factor.
        where : WhereArg
            Scope.
        trimming : TrimConfig | None
            Alternate trimming and re-poststratification until both hold. This
            is the supported route to calibrated-and-trimmed weights, since
            trimming afterwards would break the controls.
        """
        _reject_legacy_kwargs("poststratify", _legacy)
        return _poststratify(
            self._target(inplace),
            controls,
            shares=shares,
            cells=cells,
            where=where,
            wgt_name=wgt_name,
            ignore_reps=ignore_reps,
            update_design_wgts=update_design_wgts,
            strict=strict,
            trimming=trimming,
        )

    # ------------------------------------------------------------------ #
    # Standardization
    # ------------------------------------------------------------------ #

    def standardize(
        self,
        cells: str | Sequence[str],
        *,
        shares: DomainScalarMap,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        wgt_name: str = "std_wgt",
        ignore_reps: bool = False,
        update_design_wgts: bool = True,
        trimming: TrimConfig | None = None,
        inplace: bool = False,
    ) -> Any:
        """Standardize weights to a common composition.

        Removes confounding by composition: each domain is reweighted to the
        same distribution over ``cells``, making rates comparable across
        domains that differ in it. Age is the usual axis, but any composition
        variable works.

        ``target(g, c) = share(c) x W_g``, so domain totals are preserved and
        only the within-domain composition is reshaped. ``W_g`` is computed
        under the same scope that built the cells.

        Parameters
        ----------
        cells : str | Sequence[str]
            The composition axis (R's ``by``).
        shares : dict
            Standard population over the ``cells`` levels. Counts or
            proportions; normalized internally.
        by : str | Sequence[str] | None
            Domains to standardize within (R's ``over``). None treats the whole
            sample as one domain.
        where : WhereArg
            Scope (R's ``excluding.missing``).

        Standardized weights are analysis-specific: ``where`` bakes in one
        variable's missingness and ``by`` the domain structure, so estimating a
        different variable or breakdown on the same sample is silently wrong.

        A domain missing a level is renormalized over the levels it has, with a
        warning -- R instead lets that domain's total fall.
        """
        return _standardize(
            self._target(inplace),
            cells,
            shares=shares,
            by=by,
            where=where,
            wgt_name=wgt_name,
            ignore_reps=ignore_reps,
            update_design_wgts=update_design_wgts,
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
        shares: ControlsType | None = None,
        where: WhereArg = None,
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
        inplace: bool = False,
        **_legacy: Any,
    ) -> Sample:
        """Adjust weights to match several marginal distributions at once.

        Iterative proportional fitting: each margin is fitted in turn until all
        hold simultaneously. Takes no ``cells`` -- the margins ARE the
        ``controls`` keys, one entry per margin column.

        Parameters
        ----------
        controls : dict | None
            Absolute marginal totals, ``{column: {level: total}}``. Margins
            that disagree on a population total are rejected: that is the usual
            reason IPF fails to converge.
        shares : dict | None
            Marginal PROPORTIONS, same shape. Normalized per margin against one
            grand total, which makes cross-margin consistency structural.
        where : WhereArg
            Scope.
        """
        _reject_legacy_kwargs("rake", _legacy)
        return _rake(
            self._target(inplace),
            controls=controls,
            shares=shares,
            where=where,
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
        where: WhereArg = None,
        scale: Number | list[Number] | np.ndarray = 1.0,
        bounded: bool = False,
        wgt_name: str = "calib_wgt",
        update_design_wgts: bool = True,
        ignore_reps: bool = False,
        strict: bool = True,
        trimming: TrimConfig | None = None,
        inplace: bool = False,
    ) -> Any:
        """Calibrate weights to auxiliary control totals (GREG).

        Takes no ``cells``: the model is the ``controls`` keys, which may be
        continuous auxiliaries as well as categorical terms.

        Parameters
        ----------
        controls : dict[Feature, Any]
            Target totals keyed by term. A bare string names a continuous
            auxiliary; ``Cat``/``Cross`` name categorical ones.
        by : str | Sequence[str] | None
            Calibrate separately within each group, each with its own controls.
        where : WhereArg
            Scope. Rows outside it keep their weight and take no part in the
            fit.
        """
        return _calibrate(
            self._target(inplace),
            controls=controls,
            by=by,
            where=where,
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
        inplace: bool = False,
    ) -> Any:
        return _calibrate_matrix(
            self._target(inplace),
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
        *,
        where: WhereArg = None,
        inplace: bool = False,
    ) -> "Sample":
        """Cap extreme weights.

        Parameters
        ----------
        upper, lower : float | Threshold | callable | None
            Bounds. A float above 1 is an absolute cap, in (0, 1] a quantile;
            ``Threshold("median", 6.0)`` is k x a statistic.
        by : str | Sequence[str] | None
            Trim within domains: thresholds and redistribution are computed per
            domain.
        where : WhereArg
            Scope. Rows outside it neither inform the threshold nor receive
            redistributed weight.

        Recorded as ``kind="trimming"`` and provenance only, deliberately:
        trimming breaks the constraints a calibration asserted, so centring
        afterwards would claim a calibration that no longer holds. For
        calibrated-and-trimmed weights use ``poststratify(trimming=...)``.
        """
        return _trim(
            self._target(inplace),
            upper=upper,
            lower=lower,
            by=by,
            where=where,
            redistribute=redistribute,
            min_cell_size=min_cell_size,
            max_iter=max_iter,
            tol=tol,
            wgt_name=wgt_name,
            update_design_wgts=update_design_wgts,
        )
