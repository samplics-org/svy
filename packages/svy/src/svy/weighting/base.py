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
from svy.errors import WeightingError
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
    "poststratify": {"by": "cells", "factors": "shares", "strict": "on_nonconvergence"},
    "rake": {
        "factors": "shares",
        "ll_bound": "bounds",
        "up_bound": "bounds",
        "strict": "on_nonconvergence",
    },
    "calibrate": {"bounded": "bounds", "strict": "on_nonconvergence"},
    "calibrate_matrix": {
        "control": "controls",
        "bounded": "bounds",
        "strict": "on_nonconvergence",
    },
    "controls_margins_template": {"cat_na": "na"},
    "control_aux_template": {"by_na": "na"},
    "build_aux_matrix": {"by_na": "na"},
}


def _rename_note(old: str, new: str, kwargs: dict[str, Any]) -> str:
    if new == "shares":
        return (
            "shares are normalized internally, so a vector that does not "
            "sum to 1 now pins composition instead of rescaling the total."
        )
    if new == "na":
        return f"na={kwargs[old]!r} keeps the same meaning." + (
            " The default is now 'error'." if old == "cat_na" else ""
        )
    if new == "on_nonconvergence":
        mode = "error" if kwargs[old] else "warn"
        return (
            f'strict={kwargs[old]!r} is on_nonconvergence="{mode}"; True maps to "error" '
            '(the default) and False to "warn".'
        )
    if new == "bounds" and old == "bounded":
        return (
            "bounds=(lo, hi) bounds the factor g = new/old weight. Bounded calibration "
            "is not supported yet, so leave it unset."
        )
    if new == "bounds":
        lo, hi = kwargs.get("ll_bound"), kwargs.get("up_bound")
        return (
            f"ll_bound= and up_bound= are one parameter now: bounds=({lo!r}, {hi!r}). "
            "None on a side leaves it open."
        )
    return ""


def _reject_legacy_kwargs(method: str, kwargs: dict[str, Any]) -> None:
    if not kwargs:
        return
    renames = _RENAMED.get(method, {})
    for old, new in renames.items():
        if old in kwargs:
            raise WeightingError.param_renamed(
                where=f"Sample.weighting.{method}",
                method=method,
                old=old,
                new=new,
                note=_rename_note(old, new, kwargs),
            )
    unknown = next(iter(kwargs))
    raise TypeError(f"{method}() got an unexpected keyword argument {unknown!r}")


# The Sample state a weighting call may change; `_data` and `_design` go last
# so the adopted sample gets a fresh data version.
_STATE = (
    "_metadata",
    "_schema",
    "_singletons",
    "_singleton_result",
    "_internal_design",
    "_warnings",
    "_fpc",
    "_print_width",
    "_design_history",
    "_design",
    "_data",
)


class Weighting:
    """Weight adjustments.

    Three relation parameters run through these methods, each with one meaning:

    ``cells=``
        The classes the adjustment is computed over: each receives ONE derived
        adjustment factor, so the cells are what the adjustment pins.
    ``by=``
        Run the whole method separately within each domain. Present only where
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
    A dict is keyed by the column's values, or by their text form as JSON gives
    it (``"1"`` for ``1``, ``"true"`` for ``True``, an ISO date); several
    columns take a tuple in column order or its parts joined by ``"_&_"``.

    Iterative methods share ``tol=`` (the convergence tolerance; each
    docstring says what it measures), ``max_iter=`` (iteration cap) and
    ``on_nonconvergence=``; a ``trimming=`` cycle is capped by
    ``trimming.max_iter``. Every ``on_*``
    parameter in svy means the same: "error" raises and leaves the sample as
    it was; "warn" records the finding in ``sample.warnings`` and raises it
    once as a ``SvyUserWarning``; "ignore" records it at INFO level without
    raising.

    Failures raise ``WeightingError`` (a ``MethodError``) with a stable
    ``code`` and ``expected``/``got`` in the data's own values, so
    ``err.to_dict()`` is enough to act on.
    """

    def __init__(self, sample: Any) -> None:
        self._sample = sample

    def _run(self, inplace: bool, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` on a private fork; with ``inplace=True`` adopt the result.

        Every function in ``weighting/`` builds its result by rebinding
        ``sample._data`` and ``sample._design`` as it goes, so the work always
        happens on a fork. A call that fails therefore leaves the caller's data,
        design and metadata as they were, inplace or not; with ``inplace=True``
        the diagnostics recorded before the failure are still kept, since that
        is how a caller asks for them on their own Sample.
        """
        work = self._sample._fork()
        try:
            out = fn(work, *args, **kwargs)
        except Exception:
            if inplace:
                self._sample._warnings = work._warnings
            raise
        if not inplace or not hasattr(out, "_data"):
            return out
        target = self._sample
        for name in _STATE:
            if hasattr(out, name):
                setattr(target, name, getattr(out, name))
        return target

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
        """Create balanced repeated replication (BRR) weights.

        BRR needs two PSUs per stratum. Strata carrying more are paired into
        variance strata first; the paired column is written as
        ``stratum_name`` and recorded on the replicate weights, while the
        Design keeps the true strata for Taylor variance.

        Parameters
        ----------
        n_reps : int | None
            Number of replicates. None uses the Hadamard order for the number
            of strata; fewer are rounded up to it, and more than it is an error.
        stratum : str | None
            Column of the strata to build from. None uses the Design's.
        psu : str | None
            Column of the PSUs to build from. None uses the Design's.
        stratum_name : str
            Name of the paired variance-stratum column, written only when some
            stratum has more than two PSUs.
        order_by : str | Sequence[str] | None
            Pair adjacent PSUs in this order (systematic frames). None pairs in
            data order.
        shuffle : bool
            Pair PSUs at random within each stratum instead.
        rep_prefix : str | None
            Prefix of the replicate columns (``<prefix>1..R``). None uses the
            weight column's name.
        fay_coef : float
            Fay coefficient in [0, 1). 0 is classic BRR.
        rstate : int | None
            Seed for ``shuffle``.
        drop_nulls : bool
            Drop rows with a null or non-finite weight, stratum or PSU first.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.
        """
        return self._run(
            inplace,
            _create_brr_wgts,
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
        """Create delete-one-PSU jackknife replicate weights.

        Parameters
        ----------
        paired : bool
            True builds JK2: one replicate per two-PSU variance stratum, pairing
            strata that carry more than two PSUs first. False builds JK1/JKn on
            the strata exactly as given.
        stratum : str | None
            Column of the strata to build from. None uses the Design's.
        psu : str | None
            Column of the PSUs to build from. None uses the Design's.
        stratum_name : str
            Name of the paired variance-stratum column (``paired=True`` only).
        order_by : str | Sequence[str] | None
            Pair adjacent PSUs in this order (``paired=True`` only).
        shuffle : bool
            Pair PSUs at random within each stratum (``paired=True`` only).
        rep_prefix : str | None
            Prefix of the replicate columns (``<prefix>1..R``). None uses the
            weight column's name.
        rstate : int | None
            Seed for ``shuffle``.
        drop_nulls : bool
            Drop rows with a null or non-finite weight, stratum or PSU first.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.
        """
        return self._run(
            inplace,
            _create_jk_wgts,
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
        """Create bootstrap replicate weights.

        Parameters
        ----------
        n_reps : int
            Number of replicates.
        kind : {"rao-wu", "poisson"}
            ``"rao-wu"`` resamples PSUs within strata (Rao-Wu-Yue rescaling)
            and requires a PSU. ``"poisson"`` (Beaumont-Patak) draws
            independent per-unit factors and needs only a weight, for files
            whose design identifiers are suppressed; it cannot recover
            clustering.
        stratum : str | None
            Column of the strata to resample within. None uses the Design's.
        psu : str | None
            Column of the PSUs to resample. None uses the Design's.
        rep_prefix : str | None
            Prefix of the replicate columns (``<prefix>1..R``). None uses the
            weight column's name.
        drop_nulls : bool
            Drop rows with a null or non-finite weight (and stratum and PSU for
            ``"rao-wu"``) first.
        rstate : RandomState
            Seed or generator for the draws.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.

        Calibrating the replicates is a separate step: ``poststratify``,
        ``rake`` and ``calibrate`` adjust the replicate columns alongside the
        main weight unless ``ignore_reps=True``.
        """
        return self._run(
            inplace,
            _create_bs_wgts,
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
        """Create successive difference replication (SDR) weights.

        Parameters
        ----------
        n_reps : int
            Number of replicates, at least 2.
        psu : str | None
            PSU column recorded on the replicate weights as the units they
            describe. It does not change the replicates. None records the
            Design's.
        rep_prefix : str | None
            Prefix of the replicate columns (``<prefix>1..R``). None uses the
            weight column's name.
        order_col : str | None
            Column giving the sort order the successive differences follow
            (the frame order of a systematic sample). None uses data order.
        drop_nulls : bool
            Drop rows with a null or non-finite weight, stratum or
            ``order_col`` first.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.
        """
        return self._run(
            inplace,
            _create_sdr_wgts,
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
        on_nonconvergence: Literal["error", "warn", "ignore"] = "error",
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
            ``cells=`` are the classes the adjustment is computed over;
            ``by=`` runs the whole method separately within each domain.
        where : WhereArg
            Scope. Rows outside it keep their weight whatever their status.
        resp_mapping : dict | None
            Maps the canonical statuses ``"rr"`` (respondent), ``"nr"``
            (non-respondent), ``"in"`` (ineligible) and ``"uk"`` (unknown
            eligibility) to the column's own values, a value or a list of
            them each, e.g. ``{"rr": 1, "nr": [2, 3], "uk": 9}``. None reads
            the canonical codes from the column.
        wgt_name : str
            Name of the new weight column; replicate columns become
            ``<wgt_name>1..R``. It must not exist yet.
        ignore_reps : bool
            Leave the replicate weights unadjusted. The new weight then has no
            replicate weights (variance is Taylor); the replicate columns stay
            in the data and ``update_design(wgt=<previous weight>)`` restores
            them.
        unknown_to_inelig : bool
            True spreads the weight of unknown-eligibility units over every
            unit of known eligibility (rr, nr and in), so part of it goes to
            the ineligibles. False adds it to the non-respondents' weight,
            which goes to respondents only.
        update_design_wgts : bool
            Point ``design.wgt`` and the replicate weights at the new columns
            and record the adjustment. False only adds the columns.
        respondents_only : bool
            Drop the nonrespondents and ineligibles whose weight the adjustment
            handled. Rows in no adjustment class (outside ``where``, or with a
            null cell) keep their weight and stay, whatever their status.
        trimming : TrimConfig | None
            Trim the adjusted weights afterwards (one pass, no re-adjustment).
        on_nonconvergence : {"error", "warn", "ignore"}
            What to do when trimming (``trimming=``) does not converge within
            ``trimming.max_iter`` iterations; with "warn" or "ignore" the last
            iterate is kept. "error" raises and leaves the sample as it was; "warn" records the finding in
            ``sample.warnings`` and raises it once as a ``SvyUserWarning``; "ignore"
            records it at INFO level without raising.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.

        Recorded as ``kind="nonresponse"`` and provenance only: variance treats
        the adjusted weights as fixed, matching R.
        """
        _reject_legacy_kwargs("adjust", _legacy)
        return self._run(
            inplace,
            _adjust,
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
            on_nonconvergence=on_nonconvergence,
        )

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #

    def normalize(
        self,
        controls: DomainScalarMap | Number | None = None,
        *,
        factor: Number | None = None,
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
        factor : number | None
            Multiply every weight by this instead of hitting a target -- e.g.
            ``factor=4/6`` for the NCHS multi-span combined-weight recipe
            before ``combine_samples(..., adjust="none")``. Cannot be combined
            with controls, shares, cells or where.
        shares : dict | None
            Composition per cell; the weight total carries through unchanged.
        cells : str | Sequence[str] | None
            Groups that each receive one factor. ``cells=`` are the classes
            the adjustment is computed over; ``by=`` runs the whole method
            separately within each domain.
        where : WhereArg
            Scope.
        wgt_name : str
            Name of the new weight column; replicate columns become
            ``<wgt_name>1..R``. It must not exist yet.
        ignore_reps : bool
            Leave the replicate weights unadjusted. The new weight then has no
            replicate weights (variance is Taylor); the replicate columns stay
            in the data and ``update_design(wgt=<previous weight>)`` restores
            them.
        update_design_wgts : bool
            Point ``design.wgt`` and the replicate weights at the new columns
            and record the adjustment. False only adds the columns.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.

        Recorded as ``kind="normalization"`` and provenance only.
        """
        _reject_legacy_kwargs("normalize", _legacy)
        return self._run(
            inplace,
            _normalize,
            controls,
            factor=factor,
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
        on_nonconvergence: Literal["error", "warn", "ignore"] = "error",
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
            The post-strata: each receives one factor. ``cells=`` are the
            classes the adjustment is computed over; ``by=`` runs the whole
            method separately within each domain.
        where : WhereArg
            Scope.
        wgt_name : str
            Name of the new weight column; replicate columns become
            ``<wgt_name>1..R``. It must not exist yet.
        ignore_reps : bool
            Leave the replicate weights unadjusted. The new weight then has no
            replicate weights (variance is Taylor); the replicate columns stay
            in the data and ``update_design(wgt=<previous weight>)`` restores
            them.
        update_design_wgts : bool
            Point ``design.wgt`` and the replicate weights at the new columns
            and record the adjustment. False only adds the columns.
        on_nonconvergence : {"error", "warn", "ignore"}
            What to do when the trim-poststratify cycle (``trimming=``) runs
            out of cycles; with "warn" or "ignore" the last cycle's weights are
            kept. "error" raises and leaves the sample as it was; "warn"
            records the finding in ``sample.warnings`` and raises it once as a
            ``SvyUserWarning``; "ignore" records it at INFO level without
            raising.
        trimming : TrimConfig | None
            Alternate trimming and re-poststratification until both hold, for
            at most ``trimming.max_iter`` cycles. This is the supported route
            to calibrated-and-trimmed weights, since trimming afterwards would
            break the controls.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.
        """
        _reject_legacy_kwargs("poststratify", _legacy)
        return self._run(
            inplace,
            _poststratify,
            controls,
            shares=shares,
            cells=cells,
            where=where,
            wgt_name=wgt_name,
            ignore_reps=ignore_reps,
            update_design_wgts=update_design_wgts,
            on_nonconvergence=on_nonconvergence,
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
        on_nonconvergence: Literal["error", "warn", "ignore"] = "error",
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
            The composition axis (R's ``by``). ``cells=`` are the classes the
            adjustment is computed over; ``by=`` runs the whole method
            separately within each domain.
        shares : dict
            Standard population over the ``cells`` levels. Counts or
            proportions; normalized internally.
        by : str | Sequence[str] | None
            Domains to standardize within (R's ``over``). None treats the whole
            sample as one domain.
        where : WhereArg
            Scope (R's ``excluding.missing``).
        wgt_name : str
            Name of the new weight column; replicate columns become
            ``<wgt_name>1..R``. It must not exist yet.
        ignore_reps : bool
            Leave the replicate weights unadjusted. The new weight then has no
            replicate weights (variance is Taylor); the replicate columns stay
            in the data and ``update_design(wgt=<previous weight>)`` restores
            them.
        update_design_wgts : bool
            Point ``design.wgt`` and the replicate weights at the new columns
            and record the adjustment. False only adds the columns.
        trimming : TrimConfig | None
            Alternate trimming and re-standardization until both hold, for at
            most ``trimming.max_iter`` cycles.
        on_nonconvergence : {"error", "warn", "ignore"}
            What to do when the trim-standardize cycle (``trimming=``) runs out
            of cycles (``MAX_ITER_REACHED``); with "warn" or "ignore" the last
            cycle's weights are kept. "error" raises and leaves the sample as it was; "warn" records the finding in
            ``sample.warnings`` and raises it once as a ``SvyUserWarning``; "ignore"
            records it at INFO level without raising.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.

        Standardized weights are analysis-specific: ``where`` bakes in one
        variable's missingness and ``by`` the domain structure, so estimating a
        different variable or breakdown on the same sample is silently wrong.

        A domain missing a level is renormalized over the levels it has, recorded
        as ``DOMAIN_LEVELS_PARTIAL`` in ``sample.warnings`` -- R instead lets that
        domain's total fall.
        """
        return self._run(
            inplace,
            _standardize,
            cells,
            shares=shares,
            by=by,
            where=where,
            wgt_name=wgt_name,
            ignore_reps=ignore_reps,
            update_design_wgts=update_design_wgts,
            trimming=trimming,
            on_nonconvergence=on_nonconvergence,
        )

    # ------------------------------------------------------------------ #
    # Raking
    # ------------------------------------------------------------------ #

    def controls_margins_template(
        self,
        *,
        margins: Mapping[str, str],
        na: Literal["error", "level", "drop"] = "error",
        na_label: str = "__NA__",
        **_legacy: Any,
    ) -> dict[str, dict[Category, float]]:
        """A ``controls`` skeleton for ``rake``: ``{margin: {level: nan}}``.

        Levels are the columns' own values, so the filled template goes
        straight back to ``rake``.

        Parameters
        ----------
        margins : dict[str, str]
            ``{margin name: column}``. Use the column's name as the margin name
            to pass the filled template to ``rake`` as is.
        na : {"error", "level", "drop"}
            What to do with nulls in a margin column: ``"error"`` refuses
            them, ``"level"`` lists them under ``na_label``, ``"drop"`` leaves
            them out.
        na_label : str
            Key for the null level under ``na="level"``.
        """
        _reject_legacy_kwargs("controls_margins_template", _legacy)
        return _controls_margins_template(
            self._sample,
            margins=margins,
            na=na,
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
        bounds: tuple[float | None, float | None] | None = None,
        tol: float = 1e-4,
        max_iter: int = 100,
        display_iter: bool = False,
        update_design_wgts: bool = True,
        on_nonconvergence: Literal["error", "warn", "ignore"] = "error",
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
        wgt_name : str
            Name of the new weight column; replicate columns become
            ``<wgt_name>1..R``. It must not exist yet.
        ignore_reps : bool
            Leave the replicate weights unadjusted. The new weight then has no
            replicate weights (variance is Taylor); the replicate columns stay
            in the data and ``update_design(wgt=<previous weight>)`` restores
            them.
        bounds : tuple[float | None, float | None] | None
            Bounds ``(lo, hi)`` on the adjustment factor g = new weight / old
            weight. None on a side leaves it open; None sets no bounds. The
            bounds are CHECKED after raking, not enforced: raking runs
            unconstrained, and if any row's g falls outside them (main weight
            or replicates) the call raises ``BOUNDS_EXCEEDED`` and the sample
            is left as it was. With ``trimming=`` each raking pass is checked
            against the weights it started from. To limit the weights
            themselves, use ``trimming=``.
        tol : float
            Convergence tolerance: the largest relative misfit
            ``|achieved - target| / target`` accepted over every margin level.
            With ``trimming=`` it is also the relative slack allowed on the
            trimming bounds.
        max_iter : int
            Iteration cap on the IPF sweeps of each raking pass. With
            ``trimming=`` the trim-rake cycles are capped by
            ``trimming.max_iter``, as in ``poststratify`` and ``calibrate``.
        display_iter : bool
            Print the final margin error (and each trim-rake cycle) to stdout.
        update_design_wgts : bool
            Point ``design.wgt`` and the replicate weights at the new columns
            and record the adjustment. False only adds the columns.
        on_nonconvergence : {"error", "warn", "ignore"}
            What to do when raking does not converge within ``max_iter`` (or
            the trim-rake cycle within ``trimming.max_iter``); with "warn" or
            "ignore" the last iterate is kept. "error" raises and leaves the sample as it was; "warn"
            records the finding in ``sample.warnings`` and raises it once as a
            ``SvyUserWarning``; "ignore" records it at INFO level without
            raising.
        trimming : TrimConfig | None
            Alternate trimming and re-raking until both hold, for at most
            ``trimming.max_iter`` cycles. Replicates are raked once from the
            final main-weight cycle.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.
        """
        _reject_legacy_kwargs("rake", _legacy)
        return self._run(
            inplace,
            _rake,
            controls=controls,
            shares=shares,
            where=where,
            wgt_name=wgt_name,
            ignore_reps=ignore_reps,
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
            display_iter=display_iter,
            update_design_wgts=update_design_wgts,
            on_nonconvergence=on_nonconvergence,
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
        na: Literal["error", "level", "drop"] = "error",
        na_label: str = "__NA__",
        **_legacy: Any,
    ) -> dict[Category, Number] | dict[Category, dict[Category, Number]]:
        """A ``controls`` skeleton for ``calibrate``: ``{level: nan}`` per term, or
        ``{domain: {...}}`` with ``by``.

        Keys are the columns' own values.

        Parameters
        ----------
        x : Sequence[Feature]
            Calibration terms: a column name for a continuous auxiliary,
            ``Cat``/``Cross`` for categorical ones.
        by : str | Sequence[str] | None
            Domains, one skeleton each, for ``calibrate(by=)``. ``cells=`` are
            the classes the adjustment is computed over; ``by=`` runs the whole
            method separately within each domain.
        na : {"error", "level", "drop"}
            What to do with nulls in the ``by`` columns: ``"error"`` refuses
            them, ``"level"`` keys them by ``na_label``, ``"drop"`` leaves them
            out.
        na_label : str
            Key for the null domain under ``na="level"``.
        """
        _reject_legacy_kwargs("control_aux_template", _legacy)
        return _control_aux_template(
            self._sample,
            x=x,
            by=by,
            na=na,
            na_label=na_label,
        )

    def build_aux_matrix(
        self,
        *,
        x: Sequence[Feature],
        by: str | Sequence[str] | None = None,
        na: Literal["error", "level", "drop"] = "error",
        na_label: str = "__NA__",
        **_legacy: Any,
    ) -> tuple[np.ndarray, dict[Category, Number] | dict[Category, dict[Category, Number]]]:
        """The auxiliary matrix ``calibrate`` would solve against, with its
        ``controls`` skeleton.

        Returns ``(X, template)``: one row per record and one column per level
        of each term, in term order, and the skeleton of
        ``control_aux_template``. ``X`` and filled totals go to
        ``calibrate_matrix``.

        Parameters
        ----------
        x : Sequence[Feature]
            Calibration terms: a column name for a continuous auxiliary,
            ``Cat``/``Cross`` for categorical ones.
        by : str | Sequence[str] | None
            Domains, one skeleton each, for ``calibrate_matrix(by=)``.
            ``cells=`` are the classes the adjustment is computed over; ``by=``
            runs the whole method separately within each domain.
        na : {"error", "level", "drop"}
            What to do with nulls in the ``by`` columns: ``"error"`` refuses
            them, ``"level"`` keys them by ``na_label``, ``"drop"`` leaves out
            the domain and its rows.
        na_label : str
            Key for the null domain under ``na="level"``.
        """
        _reject_legacy_kwargs("build_aux_matrix", _legacy)
        return _build_aux_matrix(
            self._sample,
            x=x,
            by=by,
            na=na,
            na_label=na_label,
        )

    def calibrate(
        self,
        *,
        controls: dict[Feature, Any],
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        scale: Number | list[Number] | np.ndarray = 1.0,
        bounds: tuple[float | None, float | None] | None = None,
        wgt_name: str = "calib_wgt",
        update_design_wgts: bool = True,
        ignore_reps: bool = False,
        on_nonconvergence: Literal["error", "warn", "ignore"] = "error",
        trimming: TrimConfig | None = None,
        inplace: bool = False,
        **_legacy: Any,
    ) -> Any:
        """Calibrate weights to auxiliary control totals (GREG).

        Takes no ``cells``: the model is the ``controls`` keys, which may be
        continuous auxiliaries as well as categorical terms. The linear
        solution is closed form, so there is no ``tol`` or ``max_iter``; the
        fit is checked afterwards against the controls at a relative
        tolerance of 1e-4.

        Parameters
        ----------
        controls : dict[Feature, Any]
            Target totals keyed by term. A bare string names a continuous
            auxiliary; ``Cat``/``Cross`` name categorical ones. With ``by``,
            one such dict per domain, keyed by domain.
        by : str | Sequence[str] | None
            Calibrate separately within each group, each with its own controls.
            ``cells=`` are the classes the adjustment is computed over; ``by=``
            runs the whole method separately within each domain.
        where : WhereArg
            Scope. Rows outside it keep their weight and take no part in the
            fit.
        scale : number | Sequence[number] | np.ndarray
            Per-row variance scale of the linear distance (R's
            ``variance=``); a number applies to every row.
        bounds : tuple[float | None, float | None] | None
            Bounds ``(lo, hi)`` on the adjustment factor g = new weight / old
            weight. Not supported yet: setting a side raises
            ``NOT_SUPPORTED``. Use ``trimming=`` to constrain calibrated
            weights.
        wgt_name : str
            Name of the new weight column; replicate columns become
            ``<wgt_name>1..R``. It must not exist yet.
        update_design_wgts : bool
            Point ``design.wgt`` and the replicate weights at the new columns
            and record the adjustment. False only adds the columns.
        ignore_reps : bool
            Leave the replicate weights unadjusted. The new weight then has no
            replicate weights (variance is Taylor); the replicate columns stay
            in the data and ``update_design(wgt=<previous weight>)`` restores
            them.
        on_nonconvergence : {"error", "warn", "ignore"}
            What to do when the weights miss the controls
            (``CALIBRATION_NOT_MET``: a singular system or inconsistent
            controls) or the trim-calibrate cycle runs out of cycles; with
            "warn" or "ignore" the approximate solution is kept. "error"
            raises and leaves the sample as it was; "warn" records the finding
            in ``sample.warnings`` and raises it once as a
            ``SvyUserWarning``; "ignore" records it at INFO level without
            raising.
        trimming : TrimConfig | None
            Alternate trimming and re-calibration until both hold, for at most
            ``trimming.max_iter`` cycles.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.
        """
        _reject_legacy_kwargs("calibrate", _legacy)
        return self._run(
            inplace,
            _calibrate,
            controls=controls,
            by=by,
            where=where,
            scale=scale,
            bounds=bounds,
            wgt_name=wgt_name,
            update_design_wgts=update_design_wgts,
            ignore_reps=ignore_reps,
            on_nonconvergence=on_nonconvergence,
            trimming=trimming,
        )

    def calibrate_matrix(
        self,
        *,
        aux_vars: np.ndarray,
        controls: Any = None,
        by: str | Sequence[str] | None = None,
        scale: Number | Sequence[Number] | np.ndarray = 1.0,
        wgt_name: str = "calib_wgt",
        update_design_wgts: bool = True,
        labels: Sequence[Category] | None = None,
        weights_only: bool = False,
        bounds: tuple[float | None, float | None] | None = None,
        ignore_reps: bool = False,
        on_nonconvergence: Literal["error", "warn", "ignore"] = "error",
        trimming: TrimConfig | None = None,
        inplace: bool = False,
        **_legacy: Any,
    ) -> Any:
        """Calibrate weights against an auxiliary matrix you built (GREG).

        ``calibrate`` without the term layer: ``aux_vars`` is used as given,
        e.g. from ``build_aux_matrix``. Closed form like ``calibrate``, with
        the fit checked at a relative tolerance of 1e-4.

        Parameters
        ----------
        aux_vars : np.ndarray
            Auxiliary matrix, one row per record of this sample and one column
            per control.
        controls : sequence | dict
            One total per column of ``aux_vars``, in column order, or a dict
            keyed by ``labels``. With ``by``, a dict of those keyed by domain.
        by : str | Sequence[str] | None
            Calibrate separately within each group, each with its own controls.
            ``cells=`` are the classes the adjustment is computed over; ``by=``
            runs the whole method separately within each domain.
        scale : number | Sequence[number] | np.ndarray
            Per-row variance scale of the linear distance (R's
            ``variance=``); a number applies to every row.
        wgt_name : str
            Name of the new weight column; replicate columns become
            ``<wgt_name>1..R``. It must not exist yet.
        update_design_wgts : bool
            Point ``design.wgt`` and the replicate weights at the new columns
            and record the adjustment. False only adds the columns.
        labels : Sequence[Category] | None
            Names of the ``aux_vars`` columns, so ``controls`` can be a dict
            keyed by them.
        weights_only : bool
            Return the calibrated main weights as an array and leave the sample
            alone (``inplace`` is then ignored).
        bounds : tuple[float | None, float | None] | None
            Bounds ``(lo, hi)`` on the adjustment factor g = new weight / old
            weight. Not supported yet: setting a side raises
            ``NOT_SUPPORTED``. Use ``trimming=`` to constrain calibrated
            weights.
        ignore_reps : bool
            Leave the replicate weights unadjusted. The new weight then has no
            replicate weights (variance is Taylor); the replicate columns stay
            in the data and ``update_design(wgt=<previous weight>)`` restores
            them.
        on_nonconvergence : {"error", "warn", "ignore"}
            What to do when the weights miss the controls
            (``CALIBRATION_NOT_MET``) or the trim-calibrate cycle runs out of
            cycles; with "warn" or "ignore" the approximate solution is kept.
            "error" raises and leaves the sample as it was; "warn" records the
            finding in ``sample.warnings`` and raises it once as a
            ``SvyUserWarning``; "ignore" records it at INFO level without
            raising.
        trimming : TrimConfig | None
            Alternate trimming and re-calibration until both hold, for at most
            ``trimming.max_iter`` cycles. Ignored with ``weights_only=True``.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.
        """
        _reject_legacy_kwargs("calibrate_matrix", _legacy)
        return self._run(
            inplace,
            _calibrate_matrix,
            aux_vars=aux_vars,
            controls=controls,
            by=by,
            scale=scale,
            wgt_name=wgt_name,
            update_design_wgts=update_design_wgts,
            labels=labels,
            weights_only=weights_only,
            bounds=bounds,
            ignore_reps=ignore_reps,
            on_nonconvergence=on_nonconvergence,
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
        on_nonconvergence: Literal["error", "warn", "ignore"] = "error",
        inplace: bool = False,
    ) -> "Sample":
        """Cap extreme weights.

        Parameters
        ----------
        upper, lower : float | Threshold | callable | None
            Bounds on the weights themselves (not on an adjustment factor). A
            number is an absolute bound (``upper=40``, ``upper=0.9``);
            ``Threshold.quantile(0.99)`` is a quantile of the weights and
            ``Threshold("median", 6.0)`` is k x a statistic. At least one is
            required.
        by : str | Sequence[str] | None
            Trim within domains: thresholds and redistribution are computed per
            domain. ``cells=`` are the classes the adjustment is computed over;
            ``by=`` runs the whole method separately within each domain.
        redistribute : bool
            Spread the weight removed (or added) by trimming over the untrimmed
            units, so the total is kept. False lets the total change.
        min_cell_size : int
            Domains with fewer positive weights are skipped, and recorded as
            ``DOMAIN_SKIPPED``.
        max_iter : int
            Iteration cap on trimming and redistributing, which can push other
            weights over the bound.
        tol : float
            Convergence tolerance, which here is the fraction of weights that
            still changed in the last iteration -- not a relative misfit as in
            ``rake``.
        wgt_name : str | None
            Name of the new weight column. None trims the current weight (and
            its replicates) in place, without a new column.
        update_design_wgts : bool
            Point ``design.wgt`` at the trimmed column. No effect with
            ``wgt_name=None``.
        where : WhereArg
            Scope. Rows outside it neither inform the threshold nor receive
            redistributed weight.
        on_nonconvergence : {"error", "warn", "ignore"}
            What to do when a domain does not converge within ``max_iter``
            (``MAX_ITER_REACHED``); with "warn" or "ignore" the last iterate is
            kept. "error" raises and leaves the sample as it was; "warn" records the finding in
            ``sample.warnings`` and raises it once as a ``SvyUserWarning``; "ignore"
            records it at INFO level without raising.
        inplace : bool
            Adopt the result on this Sample instead of returning a new one.

        Recorded as ``kind="trimming"`` and provenance only, deliberately:
        trimming breaks the constraints a calibration asserted, so centring
        afterwards would claim a calibration that no longer holds. For
        calibrated-and-trimmed weights use ``poststratify(trimming=...)``.
        """
        return self._run(
            inplace,
            _trim,
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
            on_nonconvergence=on_nonconvergence,
        )
