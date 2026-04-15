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
from svy.utils.random_state import RandomState

from svy.selection.allocation import allocate as _allocate
from svy.selection.multistage import add_stage as _add_stage
from svy.selection.pps import pps_brewer as _pps_brewer
from svy.selection.pps import pps_murphy as _pps_murphy
from svy.selection.pps import pps_rs as _pps_rs
from svy.selection.pps import pps_sys as _pps_sys
from svy.selection.pps import pps_wr as _pps_wr
from svy.selection.srs import srs as _srs
from svy.selection._group_keys import _compute_pop_sizes, _build_group_keys, _normalize_n_for_groups

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
            self._sample, n,
            by=by, where=where, wr=wr,
            order_by=order_by, order_type=order_type,
            prob_name=prob_name, wgt_name=wgt_name, hit_name=hit_name,
            rstate=rstate, drop_nulls=drop_nulls,
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
            self._sample, n,
            certainty_threshold=certainty_threshold,
            by=by, where=where, order_by=order_by, order_type=order_type,
            prob_name=prob_name, wgt_name=wgt_name, hit_name=hit_name,
            rstate=rstate, drop_nulls=drop_nulls,
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
            self._sample, n,
            certainty_threshold=certainty_threshold,
            by=by, where=where, prob_name=prob_name, wgt_name=wgt_name,
            hit_name=hit_name, rstate=rstate, drop_nulls=drop_nulls,
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
            self._sample, n,
            certainty_threshold=certainty_threshold,
            by=by, where=where, prob_name=prob_name, wgt_name=wgt_name,
            hit_name=hit_name, rstate=rstate, drop_nulls=drop_nulls,
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
            self._sample, n,
            certainty_threshold=certainty_threshold,
            by=by, where=where, prob_name=prob_name, wgt_name=wgt_name,
            hit_name=hit_name, rstate=rstate, drop_nulls=drop_nulls,
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
            self._sample, n,
            certainty_threshold=certainty_threshold,
            by=by, where=where, prob_name=prob_name, wgt_name=wgt_name,
            hit_name=hit_name, rstate=rstate, drop_nulls=drop_nulls,
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

    # ------------------------------------------------------------------ #
    # Allocation helpers
    # ------------------------------------------------------------------ #

    def group_sizes(
        self,
        *,
        by: str | Sequence[str] | None = None,
    ) -> dict[str, int]:
        """
        Return per-group frame counts for the current sample.

        The returned dict can be passed directly to ``allocate()`` or used
        to inspect stratum balance before selecting.
        """
        import polars as pl
        from typing import cast

        data = self._sample._data
        if isinstance(data, pl.LazyFrame):
            data = cast(pl.DataFrame, data.collect())

        stratum_col = self._sample._internal_design.get("stratum")
        by_cols = self._sample._to_cols(by) if by is not None else []
        suffix = self._sample._internal_design.get("suffix", "_svy_internal_cols_concatenated")

        stratum_by_col, _, G, _, _, data = _build_group_keys(
            data,
            stratum_col=stratum_col,
            by_cols=by_cols,
            suffix=suffix,
            sample_ref=self._sample,
        )
        return _compute_pop_sizes(data, stratum_by_col, G)

    def allocate(
        self,
        group_sizes: dict[str, int],
        *,
        method: str = "proportional",
        n_total: int | None = None,
        n_per_group: int | None = None,
        rate: float | dict[str, float] | None = None,
        group_sds: dict[str, float] | None = None,
        min_n: int = 1,
        cap_at_population: bool = True,
    ) -> dict[str, int]:
        """
        Compute a per-group ``n`` mapping using a named allocation method.

        Pass the returned dict directly as ``n=`` to srs(), pps_sys(), etc.
        """
        return _allocate(
            group_sizes,
            method=method,
            n_total=n_total,
            n_per_group=n_per_group,
            rate=rate,
            group_sds=group_sds,
            min_n=min_n,
            cap_at_population=cap_at_population,
        )
