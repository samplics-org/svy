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
from svy.selection._group_keys import (
    _build_group_keys,
    _compute_pop_sizes,
)
from svy.selection.allocation import AllocationMethod
from svy.selection.allocation import allocate as _allocate
from svy.selection.multistage import add_stage as _add_stage
from svy.selection.pps import pps_brewer as _pps_brewer
from svy.selection.pps import pps_murphy as _pps_murphy
from svy.selection.pps import pps_rs as _pps_rs
from svy.selection.pps import pps_sys as _pps_sys
from svy.selection.pps import pps_wr as _pps_wr
from svy.selection.srs import srs as _srs
from svy.utils.deprecation import deprecated
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

    # ------------------------------------------------------------------ #
    # Allocation
    # ------------------------------------------------------------------ #

    def _frame_groups(self, by: str | Sequence[str] | None):
        """The frame with its stratum (x by) key column, the key column and the keys."""
        from typing import cast

        import polars as pl

        data = self._sample._data
        if isinstance(data, pl.LazyFrame):
            data = cast(pl.DataFrame, data.collect())
        stratum_col = self._sample._internal_design.get("stratum")
        by_cols = self._sample._to_cols(by) if by is not None else []
        stratum_by_col, _, G, _, _, data = _build_group_keys(
            data,
            stratum_col=stratum_col,
            by_cols=by_cols,
            sample_ref=self._sample,
        )
        return data, stratum_by_col, G

    def _group_mos(self, mos: str | None, by: str | Sequence[str] | None) -> dict[str, float]:
        """Per-group totals of the measure of size, keyed like the frame counts.

        Missing and non-positive values add nothing: such units cannot be
        drawn with probability proportional to size.
        """
        import polars as pl

        from svy.errors import MethodError

        mos_col = mos if mos is not None else self._sample._design.mos
        where = "sampling.allocate"
        if mos_col is None:
            raise MethodError(
                title="No measure of size",
                detail="allocate(method='size') needs a measure of size column.",
                code="MOS_MISSING",
                where=where,
                param="mos",
                hint="Pass mos='column' or declare Design(mos='column').",
            )
        data, stratum_by_col, G = self._frame_groups(by)
        if mos_col not in data.columns:
            raise MethodError(
                title="Measure of size column not found",
                detail=f"Column {mos_col!r} is not in the data.",
                code="MOS_MISSING",
                where=where,
                param="mos",
                got=mos_col,
            )
        if not data.schema[mos_col].is_numeric():
            raise MethodError(
                title="Measure of size must be numeric",
                detail=f"Column {mos_col!r} has type {data.schema[mos_col]}.",
                code="MOS_NOT_NUMERIC",
                where=where,
                param="mos",
                got=str(data.schema[mos_col]),
            )
        positive = pl.col(mos_col).cast(pl.Float64).clip(lower_bound=0.0).sum()
        if not G or stratum_by_col is None:
            return {"__all__": float(data.select(positive).item())}
        agg = data.group_by(stratum_by_col).agg(positive.alias("__mos__"))
        totals = dict(zip(agg[stratum_by_col].to_list(), agg["__mos__"].to_list()))
        return {g: float(totals.get(g, 0.0)) for g in G}

    @deprecated(
        since="0.32.0",
        remove_in="2026.0",
        use="sample.sampling.allocate(..., by=...), which counts the frame itself, "
        "or sample.describe(by=...) to look at the groups",
    )
    def group_sizes(
        self,
        *,
        by: str | Sequence[str] | None = None,
    ) -> dict[str, int]:
        """
        Return per-group frame counts for the current sample.

        Deprecated: ``allocate()`` counts the frame itself.
        """
        data, stratum_by_col, G = self._frame_groups(by)
        return _compute_pop_sizes(data, stratum_by_col, G)

    def allocate(
        self,
        group_sizes: dict[str, int] | None = None,
        *,
        method: AllocationMethod = "proportional",
        n_total: int | None = None,
        n_per_group: int | None = None,
        rate: float | dict[str, float] | None = None,
        by: str | Sequence[str] | None = None,
        mos: str | None = None,
        group_sds: dict[str, float] | None = None,
        group_mos: dict[str, float] | None = None,
        power: float = 1.0,
        min_n: int = 1,
        cap_at_population: bool = True,
    ) -> dict[str, int]:
        """
        Compute a per-group ``n`` mapping using a named allocation method.

        The groups are the design's strata, crossed with ``by=``; the frame's
        units are counted per group, and ``method="size"`` sums the measure of
        size (``mos=``, default the design's ``mos``) per group. Pass the
        returned dict as ``n=`` to srs(), pps_sys(), etc. with the same
        ``by=``.

        Parameters
        ----------
        group_sizes : dict | None
            Custom counts ``{group: N_h}`` from outside the frame (e.g. a
            population register), used instead of counting the frame. With
            ``method="size"`` they need ``group_mos=`` too.
        method : {"proportional", "neyman", "size", "equal", "rate"}
            ``"size"`` allocates in proportion to the groups' measure-of-size
            totals.
        by : str | Sequence[str] | None
            Columns crossed with the design's strata to form the groups.
        mos : str | None
            Measure of size column for ``method="size"``; default the design's.
        group_sds, group_mos : dict | None
            Per-group SDs (``"neyman"``) and custom size totals (``"size"``).
        power : float
            ``N_h ** power`` (``"proportional"``) or ``MOS_h ** power``
            (``"size"``); 0.5 is square-root allocation.

        Examples
        --------
        >>> n_map = sample.sampling.allocate(method="proportional", n_total=500, by="region")
        >>> sample = sample.sampling.srs(n_map, by="region")
        >>> n_map = sample.sampling.allocate(method="size", n_total=60, by="region")
        >>> n_map = sample.sampling.allocate(register_counts, n_total=500)  # custom counts
        """
        from svy.errors import MethodError

        where = "sampling.allocate"
        if group_sizes is not None and by is not None:
            raise MethodError(
                title="Custom counts and by= together",
                detail="by= says how to count the frame, but group_sizes= replaces that count.",
                code="ALLOCATE_SIZES_AND_BY",
                where=where,
                param="by",
                hint="Drop group_sizes= to count the frame by group, or drop by=.",
            )
        if mos is not None and method != "size":
            raise MethodError(
                title="mos= without method='size'",
                detail=f"allocate(method={method!r}) does not use a measure of size.",
                code="ALLOCATE_MOS_UNUSED",
                where=where,
                param="mos",
                hint="Use method='size', or drop mos=.",
            )
        if group_sizes is None:
            data, stratum_by_col, G = self._frame_groups(by)
            group_sizes = _compute_pop_sizes(data, stratum_by_col, G)
            if method == "size" and group_mos is None:
                group_mos = self._group_mos(mos, by)
        elif method == "size" and group_mos is None:
            raise MethodError(
                title="Custom counts need custom size totals",
                detail=(
                    "With group_sizes= the groups are yours, so the frame's measure of "
                    "size cannot be summed over them."
                ),
                code="ALLOCATE_MOS_MISSING",
                where=where,
                param="group_mos",
                hint="Pass group_mos={group: size total} with the same keys as group_sizes.",
            )
        return _allocate(
            group_sizes,
            method=method,
            n_total=n_total,
            n_per_group=n_per_group,
            rate=rate,
            group_sds=group_sds,
            group_mos=group_mos,
            power=power,
            min_n=min_n,
            cap_at_population=cap_at_population,
        )
