# src/svy/selection/base.py
"""
Selection facade.

The Selection class is a thin dispatcher.  Each method is a one-liner
that delegates to the corresponding module-level function.  No logic
lives here.

Module map
----------
_helpers.py     _psu_list, _apply_order, edge-case warning guards
_group_keys.py  _build_group_keys, _normalize_n_for_groups, _compute_pop_sizes
srs.py          srs() + _srs_writeback, _ensure_row_index, _apply_where
pps.py          pps_sys/wr/brewer/murphy/rs() + _pps() + _pps_writeback
multistage.py   add_stage()
allocation.py   allocate() -- proportional / neyman / equal / rate
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from svy.core.types import Category, Number, WhereArg
from svy.errors.api_errors import guided_getattr
from svy.selection.multistage import add_stage as _add_stage
from svy.selection.pps import pps_brewer as _pps_brewer
from svy.selection.pps import pps_murphy as _pps_murphy
from svy.selection.pps import pps_rs as _pps_rs
from svy.selection.pps import pps_sys as _pps_sys
from svy.selection.pps import pps_wr as _pps_wr
from svy.selection.srs import srs as _srs
from svy.utils.random_state import RandomState


if TYPE_CHECKING:
    from svy.core.sample import Sample


class Selection:
    __getattr__ = guided_getattr("Selection", "sample.sampling")

    def __init__(self, sample: "Sample") -> None:
        self._sample = sample

    # ------------------------------------------------------------------ #
    # Simple Random Sampling
    # ------------------------------------------------------------------ #

    def srs(
        self,
        n: int | Mapping[Category, Number],
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        wr: bool = False,
        order_by: str | Sequence[str] | None = None,
        order_type: Literal["ascending", "descending", "random"] = "ascending",
        prob_name: str | None = None,
        wgt_name: str | None = None,
        hit_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
        n_draws: int | None = None,
    ) -> "Sample | list[Sample]":
        """
        Simple random sample, optionally stratified by stratum x ``by``.

        Parameters
        ----------
        n : int or mapping
            Units per stratum, or per stratum x ``by`` cell. A mapping may be keyed by cell, by
            ``by`` level or by stratum; every key must be given (0 skips one).
        by : str or sequence of str, optional
            Columns that subdivide the design's strata; units are drawn in each stratum x ``by`` cell.
        where : expression or mapping, optional
            Draw only from rows where this holds; the other rows stay in the output with null
            selection columns.
        wr : bool, default False
            With replacement; a unit drawn k times is one row with ``k`` hits.
        order_by : str or sequence of str, optional
            Columns to sort the frame by before drawing.
        order_type : {"ascending", "descending", "random"}, default "ascending"
            Sort direction; ``"random"`` shuffles, within ``order_by`` groups when given.
        prob_name, wgt_name, hit_name : str, optional
            Names of the probability, weight and hit columns (default ``svy_prob_selection``,
            ``svy_sample_weight``, ``svy_number_of_hits``); a name already in the data raises.
        rstate : int, Generator or SeedSequence, optional
            Seed; a Generator advances with each call.
        drop_nulls : bool, default False
            Drop rows missing a value the draw uses (design, ``by`` and ``order_by`` columns);
            otherwise such rows raise.
        n_draws : int, optional
            Draw this many samples and return them as a list, also for 1; draw r equals a single
            call with child r of ``SeedSequence(rstate)``, and this sample is left as it was.

        Returns
        -------
        Sample or list[Sample]
            This sample, holding the drawn rows with their selection columns and its design's
            ``prob``, ``wgt`` and ``hit`` set; with ``n_draws``, a list of new samples.
        """
        return _srs(
            self._sample,
            n,
            by=by,
            where=where,
            wr=wr,
            order_by=order_by,
            order_type=order_type,
            prob_name=prob_name,
            wgt_name=wgt_name,
            hit_name=hit_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
            n_draws=n_draws,
        )

    # ------------------------------------------------------------------ #
    # PPS methods
    # ------------------------------------------------------------------ #

    def pps_sys(
        self,
        n: int | Mapping[Category, Number],
        *,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        order_by: str | Sequence[str] | None = None,
        order_type: Literal["ascending", "descending", "random"] = "ascending",
        prob_name: str | None = None,
        wgt_name: str | None = None,
        hit_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
        n_draws: int | None = None,
    ) -> "Sample | list[Sample]":
        """
        PPS systematic sampling without replacement.

        The size measure is the design's ``mos``. Systematic selection depends on the frame's order (``order_by``).

        Parameters
        ----------
        n : int or mapping
            Units per stratum, or per stratum x ``by`` cell. A mapping may be keyed by cell, by
            ``by`` level or by stratum; every key must be given (0 skips one).
        certainty_threshold : float, default 1.0
            Units whose inclusion probability reaches this, in (0, 1], are taken with certainty,
            and the rest are drawn from what remains.
        by : str or sequence of str, optional
            Columns that subdivide the design's strata; units are drawn in each stratum x ``by`` cell.
        where : expression or mapping, optional
            Draw only from rows where this holds; the other rows stay in the output with null
            selection columns.
        order_by : str or sequence of str, optional
            Columns to sort the frame by before drawing.
        order_type : {"ascending", "descending", "random"}, default "ascending"
            Sort direction; ``"random"`` shuffles, within ``order_by`` groups when given.
        prob_name, wgt_name, hit_name : str, optional
            Names of the probability, weight and hit columns (default ``svy_prob_selection``,
            ``svy_sample_weight``, ``svy_number_of_hits``); a name already in the data raises.
        rstate : int, Generator or SeedSequence, optional
            Seed; a Generator advances with each call.
        drop_nulls : bool, default False
            Drop rows missing a value the draw uses (design, ``by`` and ``order_by`` columns);
            otherwise such rows raise.
        n_draws : int, optional
            Draw this many samples and return them as a list, also for 1; draw r equals a single
            call with child r of ``SeedSequence(rstate)``, and this sample is left as it was.

        Returns
        -------
        Sample or list[Sample]
            This sample, holding the drawn rows with their selection columns and its design's
            ``prob``, ``wgt`` and ``hit`` set; with ``n_draws``, a list of new samples.
        """
        return _pps_sys(
            self._sample,
            n,
            certainty_threshold=certainty_threshold,
            by=by,
            where=where,
            order_by=order_by,
            order_type=order_type,
            prob_name=prob_name,
            wgt_name=wgt_name,
            hit_name=hit_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
            n_draws=n_draws,
        )

    def pps_wr(
        self,
        n: int | Mapping[Category, Number],
        *,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        prob_name: str | None = None,
        wgt_name: str | None = None,
        hit_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
        n_draws: int | None = None,
    ) -> "Sample | list[Sample]":
        """
        PPS sampling with replacement.

        The size measure is the design's ``mos``. A unit drawn k times is one row with ``k`` hits.

        Parameters
        ----------
        n : int or mapping
            Units per stratum, or per stratum x ``by`` cell. A mapping may be keyed by cell, by
            ``by`` level or by stratum; every key must be given (0 skips one).
        certainty_threshold : float, default 1.0
            Not used by this method.
        by : str or sequence of str, optional
            Columns that subdivide the design's strata; units are drawn in each stratum x ``by`` cell.
        where : expression or mapping, optional
            Draw only from rows where this holds; the other rows stay in the output with null
            selection columns.
        prob_name, wgt_name, hit_name : str, optional
            Names of the probability, weight and hit columns (default ``svy_prob_selection``,
            ``svy_sample_weight``, ``svy_number_of_hits``); a name already in the data raises.
        rstate : int, Generator or SeedSequence, optional
            Seed; a Generator advances with each call.
        drop_nulls : bool, default False
            Drop rows missing a value the draw uses (design and ``by`` columns); otherwise such
            rows raise.
        n_draws : int, optional
            Draw this many samples and return them as a list, also for 1; draw r equals a single
            call with child r of ``SeedSequence(rstate)``, and this sample is left as it was.

        Returns
        -------
        Sample or list[Sample]
            This sample, holding the drawn rows with their selection columns and its design's
            ``prob``, ``wgt`` and ``hit`` set; with ``n_draws``, a list of new samples.
        """
        return _pps_wr(
            self._sample,
            n,
            certainty_threshold=certainty_threshold,
            by=by,
            where=where,
            prob_name=prob_name,
            wgt_name=wgt_name,
            hit_name=hit_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
            n_draws=n_draws,
        )

    def pps_brewer(
        self,
        n: int | Mapping[Category, Number],
        *,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        prob_name: str | None = None,
        wgt_name: str | None = None,
        hit_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
        n_draws: int | None = None,
    ) -> "Sample | list[Sample]":
        """
        Brewer PPS sampling without replacement.

        The size measure is the design's ``mos``.

        Parameters
        ----------
        n : int or mapping
            Units per stratum, or per stratum x ``by`` cell. A mapping may be keyed by cell, by
            ``by`` level or by stratum; every key must be given (0 skips one).
        certainty_threshold : float, default 1.0
            Units whose inclusion probability reaches this, in (0, 1], are taken with certainty,
            and the rest are drawn from what remains.
        by : str or sequence of str, optional
            Columns that subdivide the design's strata; units are drawn in each stratum x ``by`` cell.
        where : expression or mapping, optional
            Draw only from rows where this holds; the other rows stay in the output with null
            selection columns.
        prob_name, wgt_name, hit_name : str, optional
            Names of the probability, weight and hit columns (default ``svy_prob_selection``,
            ``svy_sample_weight``, ``svy_number_of_hits``); a name already in the data raises.
        rstate : int, Generator or SeedSequence, optional
            Seed; a Generator advances with each call.
        drop_nulls : bool, default False
            Drop rows missing a value the draw uses (design and ``by`` columns); otherwise such
            rows raise.
        n_draws : int, optional
            Draw this many samples and return them as a list, also for 1; draw r equals a single
            call with child r of ``SeedSequence(rstate)``, and this sample is left as it was.

        Returns
        -------
        Sample or list[Sample]
            This sample, holding the drawn rows with their selection columns and its design's
            ``prob``, ``wgt`` and ``hit`` set; with ``n_draws``, a list of new samples.
        """
        return _pps_brewer(
            self._sample,
            n,
            certainty_threshold=certainty_threshold,
            by=by,
            where=where,
            prob_name=prob_name,
            wgt_name=wgt_name,
            hit_name=hit_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
            n_draws=n_draws,
        )

    def pps_murphy(
        self,
        n: int | Mapping[Category, Number],
        *,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        prob_name: str | None = None,
        wgt_name: str | None = None,
        hit_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
        n_draws: int | None = None,
    ) -> "Sample | list[Sample]":
        """
        Murphy PPS sampling without replacement; ``n`` must be 2 per stratum.

        The size measure is the design's ``mos``.

        Parameters
        ----------
        n : int or mapping
            Units per stratum, or per stratum x ``by`` cell. A mapping may be keyed by cell, by
            ``by`` level or by stratum; every key must be given (0 skips one).
        certainty_threshold : float, default 1.0
            Not used by this method.
        by : str or sequence of str, optional
            Columns that subdivide the design's strata; units are drawn in each stratum x ``by`` cell.
        where : expression or mapping, optional
            Draw only from rows where this holds; the other rows stay in the output with null
            selection columns.
        prob_name, wgt_name, hit_name : str, optional
            Names of the probability, weight and hit columns (default ``svy_prob_selection``,
            ``svy_sample_weight``, ``svy_number_of_hits``); a name already in the data raises.
        rstate : int, Generator or SeedSequence, optional
            Seed; a Generator advances with each call.
        drop_nulls : bool, default False
            Drop rows missing a value the draw uses (design and ``by`` columns); otherwise such
            rows raise.
        n_draws : int, optional
            Draw this many samples and return them as a list, also for 1; draw r equals a single
            call with child r of ``SeedSequence(rstate)``, and this sample is left as it was.

        Returns
        -------
        Sample or list[Sample]
            This sample, holding the drawn rows with their selection columns and its design's
            ``prob``, ``wgt`` and ``hit`` set; with ``n_draws``, a list of new samples.
        """
        return _pps_murphy(
            self._sample,
            n,
            certainty_threshold=certainty_threshold,
            by=by,
            where=where,
            prob_name=prob_name,
            wgt_name=wgt_name,
            hit_name=hit_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
            n_draws=n_draws,
        )

    def pps_rs(
        self,
        n: int | Mapping[Category, Number],
        *,
        certainty_threshold: float = 1.0,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        prob_name: str | None = None,
        wgt_name: str | None = None,
        hit_name: str | None = None,
        rstate: RandomState = None,
        drop_nulls: bool = False,
        n_draws: int | None = None,
    ) -> "Sample | list[Sample]":
        """
        Rao-Sampford (rejective) PPS sampling without replacement.

        The size measure is the design's ``mos``.

        Parameters
        ----------
        n : int or mapping
            Units per stratum, or per stratum x ``by`` cell. A mapping may be keyed by cell, by
            ``by`` level or by stratum; every key must be given (0 skips one).
        certainty_threshold : float, default 1.0
            Units whose inclusion probability reaches this, in (0, 1], are taken with certainty,
            and the rest are drawn from what remains.
        by : str or sequence of str, optional
            Columns that subdivide the design's strata; units are drawn in each stratum x ``by`` cell.
        where : expression or mapping, optional
            Draw only from rows where this holds; the other rows stay in the output with null
            selection columns.
        prob_name, wgt_name, hit_name : str, optional
            Names of the probability, weight and hit columns (default ``svy_prob_selection``,
            ``svy_sample_weight``, ``svy_number_of_hits``); a name already in the data raises.
        rstate : int, Generator or SeedSequence, optional
            Seed; a Generator advances with each call.
        drop_nulls : bool, default False
            Drop rows missing a value the draw uses (design and ``by`` columns); otherwise such
            rows raise.
        n_draws : int, optional
            Draw this many samples and return them as a list, also for 1; draw r equals a single
            call with child r of ``SeedSequence(rstate)``, and this sample is left as it was.

        Returns
        -------
        Sample or list[Sample]
            This sample, holding the drawn rows with their selection columns and its design's
            ``prob``, ``wgt`` and ``hit`` set; with ``n_draws``, a list of new samples.
        """
        return _pps_rs(
            self._sample,
            n,
            certainty_threshold=certainty_threshold,
            by=by,
            where=where,
            prob_name=prob_name,
            wgt_name=wgt_name,
            hit_name=hit_name,
            rstate=rstate,
            drop_nulls=drop_nulls,
            n_draws=n_draws,
        )

    # ------------------------------------------------------------------ #
    # Multi-stage
    # ------------------------------------------------------------------ #

    def add_stage(
        self,
        next_stage: "Any",
        *,
        prob_name: str | None = None,
        wgt_name: str | None = None,
    ) -> "Sample":
        """Chain a second-stage frame or selected sample onto the current stage."""
        return _add_stage(self._sample, next_stage, prob_name=prob_name, wgt_name=wgt_name)
