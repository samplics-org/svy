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
    ) -> "Sample":
        """
        Simple Random Sample, optionally stratified by (stratum x by).

        Parameters
        ----------
        where : WhereArg, optional
            Row filter applied before selection.  Eligible rows (True)
            participate in the draw; non-eligible rows (False) are kept
            in the output with prob=null, weight=null, hit=null.
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
    ) -> "Sample":
        """
        PPS systematic sampling without replacement.

        Parameters
        ----------
        where : WhereArg, optional
            Row filter — eligible rows participate in the draw; others
            are kept with null selection columns.
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
    ) -> "Sample":
        """PPS sampling with replacement."""
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
    ) -> "Sample":
        """Brewer PPS sampling without replacement."""
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
    ) -> "Sample":
        """Murphy PPS sampling without replacement (n=2 only)."""
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
    ) -> "Sample":
        """Rao-Sampford PPS sampling without replacement."""
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
