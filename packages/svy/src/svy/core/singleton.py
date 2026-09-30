from __future__ import annotations

import copy
import logging

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, cast

import msgspec
import polars as pl

from svy.core.constants import (
    SVY_ROW_INDEX,
    SVY_VAR_EXCLUDE,
    SVY_VAR_IS_SINGLETON,
    SVY_VAR_PSU,
    SVY_VAR_STRATUM,
)
from svy.core.enumerations import SingletonDomains, SingletonMethod
from svy.core.types import WhereArg
from svy.errors.singleton_errors import SingletonError
from svy.utils.random_state import RandomState, resolve_random_state


if TYPE_CHECKING:
    from svy.core.design import Design
    from svy.core.design import Singleton as SingletonRule
    from svy.core.sample import Sample
    from svy.core.warnings import SvyWarning

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL COLUMN NAMES FOR VARIANCE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

# These columns are used internally for variance estimation when singleton
# handling is applied. The original design columns are NEVER modified.
_VAR_STRATUM_COL = SVY_VAR_STRATUM
_VAR_PSU_COL = SVY_VAR_PSU
_VAR_EXCLUDE_COL = SVY_VAR_EXCLUDE
_VAR_IS_SINGLETON_COL = SVY_VAR_IS_SINGLETON  # For CENTER method
_VAR_COLS = (_VAR_STRATUM_COL, _VAR_PSU_COL, _VAR_EXCLUDE_COL, _VAR_IS_SINGLETON_COL)

#: For the collapse options read per stratum: error code, what the column is
#: for, and an example of a column that fits.
_STRATUM_COLUMN_ROLES = {
    "within": (
        "SINGLETON_WITHIN_INVALID",
        "say which strata a singleton may join",
        "the region the stratum lies in",
    ),
    "order_by": (
        "SINGLETON_ORDER_BY_INVALID",
        "give the stratum one place in the order",
        "a stratum code or its position along a frame",
    ),
}

# Must match Sample._concatenate_cols, which builds the internal stratum key.
_KEY_SEP = "__by__"
_KEY_NULL = "__Null__"


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════


class SingletonInfo(msgspec.Struct, frozen=True):
    """Information about a single singleton stratum."""

    stratum_key: str
    stratum_values: dict[str, Any]
    psu_key: str
    n_observations: int


class StratumInfo(msgspec.Struct, frozen=True):
    """Information about a stratum (used for collapse candidates)."""

    stratum_key: str
    stratum_values: dict[str, Any]
    n_psus: int
    n_observations: int
    sort_values: tuple[Any, ...] = ()


class SingletonHandlingConfig(msgspec.Struct, frozen=True):
    """
    Configuration for how singletons are handled during variance estimation.

    This is stored on the Sample and read by the variance estimation engine.
    The original design columns are NEVER modified - this config tells the
    engine how to adjust its calculations.
    """

    method: SingletonMethod
    singleton_keys: tuple[str, ...]

    # For CERTAINTY: maps singleton stratum -> (original_psu becomes stratum, records become PSUs)
    # For COLLAPSE/POOL: maps singleton stratum key -> target stratum key
    # For SKIP: None (just marks strata to exclude)
    # For SCALE/CENTER: None (post-hoc adjustment)
    stratum_mapping: dict[str, str] | None = None

    # For SCALE: design-level singleton fraction (reporting only; the estimator
    # recounts it over the strata present in each estimate)
    singleton_fraction: float | None = None

    # For CENTER: grand mean values (computed at estimation time)
    # This is populated lazily by the estimation engine

    # Internal column names for variance calculation (if data was modified)
    var_stratum_col: str | None = None
    var_psu_col: str | None = None
    var_exclude_col: str | None = None


class SingletonResult(msgspec.Struct, frozen=True):
    """Result of applying a singleton handling method."""

    method: SingletonMethod
    detected: tuple[SingletonInfo, ...]
    applied: dict[str, str] | tuple[str, ...] | None = None
    n_singletons_detected: int = 0
    n_strata_before: int = 0
    n_strata_after: int = 0
    n_psus_before: int = 0
    n_psus_after: int = 0

    # The config to be attached to the sample for variance estimation
    config: SingletonHandlingConfig | None = None


# ═══════════════════════════════════════════════════════════════════════════
# TYPE ALIASES
# ═══════════════════════════════════════════════════════════════════════════

CollapseStrategy = Literal["next", "previous", "smallest", "largest"]
CollapseUsing = (
    CollapseStrategy | dict[str, str] | Callable[[SingletonInfo, list[StratumInfo]], str]
)


# ═══════════════════════════════════════════════════════════════════════════
# THE ENGINE THAT APPLIES THE RULE
# ═══════════════════════════════════════════════════════════════════════════


class _Engine:
    """Detects a sample's singleton strata and applies the design's rule to them.

    Internal: the rule is declared on the design (``svy.Singleton``) and read
    through ``sample.singletons``; this is what ``_rederive`` runs whenever the
    data or design changed. The ``_apply_*`` builders write the variance columns
    the estimators read.
    """

    __slots__ = ("_sample",)

    def __init__(self, sample: Sample, *, _sync: bool = True) -> None:
        self._sample = sample
        # The handling in effect is derived state: bring it up to date with the
        # data and design before anything here reads it.
        sync = getattr(sample, "_sync_parts", None) if _sync else None
        if sync is not None:
            sync()

    def detected(self) -> list[SingletonInfo]:
        """The singleton strata of the sample's current data and design."""
        return self._detect_on_df(self._narrow_data(), self._narrow_design())

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _narrow_data(self) -> "pl.DataFrame":
        """Narrow self._sample._data to pl.DataFrame, collecting if LazyFrame."""
        raw = self._sample._data
        if isinstance(raw, pl.LazyFrame):
            return cast(pl.DataFrame, raw.collect())
        return cast(pl.DataFrame, raw)

    def _narrow_design(self) -> "Design":
        """Narrow self._sample._design to Design."""
        return cast("Design", self._sample._design)

    @staticmethod
    def _to_cols(spec: str | Sequence[str] | None) -> list[str]:
        """Convert column spec to list of column names."""
        if spec is None:
            return []
        if isinstance(spec, str):
            return [spec] if spec else []
        return [s for s in spec if isinstance(s, str) and s]

    def _internal_cols(self) -> tuple[str | None, str]:
        """Resolve internal design column names."""
        idict = getattr(self._sample, "_internal_design", {}) or {}
        stratum_col = idict.get("stratum")
        psu_col = idict.get("psu")

        if not psu_col:
            psu_col = SVY_ROW_INDEX

        return stratum_col, psu_col

    def _counts_before(self, df: pl.DataFrame, stratum_col: str, psu_col: str) -> tuple[int, int]:
        """Returns (n_strata, n_psus) efficiently."""
        # Distinct (stratum, PSU) pairs are the distinct PSUs summed over strata,
        # which avoids hashing a struct of the two.
        counts = df.group_by(stratum_col).agg(pl.col(psu_col).n_unique().alias("p"))
        return counts.height, int(counts.get_column("p").sum())

    def _strata_index(self) -> _StrataIndex | None:
        stratum_col, _ = self._internal_cols()
        data = self._narrow_data()
        if not stratum_col or stratum_col not in data.columns:
            return None
        cols = self._to_cols(self._narrow_design().stratum)
        return _StrataIndex(data, cols, stratum_col)

    def _key_values(self, keys: Sequence[str], index: _StrataIndex | None = None) -> list[Any]:
        """The stratum columns' values of internal stratum keys: a scalar per
        key, a tuple when the design has several stratum columns."""
        index = cast(_StrataIndex, index or self._strata_index())
        single = isinstance(self._narrow_design().stratum, str)
        return [index.values[k][0] if single else index.values[k] for k in keys]

    def _value_keys(
        self, values: Sequence[Any], index: _StrataIndex | None = None
    ) -> list[str | None]:
        """Internal stratum keys of stored stratum values, None where the
        stratum is not in the data."""
        index = index or self._strata_index()
        return [None if index is None else index.key(v, strict=False) for v in values]

    def _resolve(self, stratum: Any, index: _StrataIndex | None = None) -> Any:
        """svy's key for a stratum a caller named, or the input unchanged when
        it names none (the caller's own error then applies)."""
        index = index or self._strata_index()
        key = None if index is None else index.key(stratum)
        return stratum if key is None else key

    def _detect_on_df(self, df: pl.DataFrame, design: Design) -> list[SingletonInfo]:
        """
        Detect singleton strata using optimized Polars aggregation.
        """
        stratum_col, psu_col = self._internal_cols()
        if not stratum_col or stratum_col not in df.columns:
            return []

        # Aggregation: Find strata with exactly 1 unique PSU
        agg = (
            df.lazy()
            .group_by(stratum_col)
            .agg(
                pl.col(psu_col).n_unique().alias("n_psu"),
                pl.len().alias("n_obs"),
                pl.col(psu_col).first().alias("any_psu"),
            )
            .filter(pl.col("n_psu") == 1)
            .collect()
        )
        agg = cast(pl.DataFrame, agg)

        if agg.height == 0:
            return []

        # Extract original stratum values if they exist (for reporting)
        stratum_cols = self._to_cols(getattr(design, "stratum", None))
        values_map: dict[Any, dict[str, Any]] = {}

        if stratum_cols:
            keys = agg.get_column(stratum_col).to_list()

            val_df = (
                df.filter(pl.col(stratum_col).is_in(keys))
                .group_by(stratum_col)
                .head(1)
                .select([stratum_col] + stratum_cols)
            )

            for row in val_df.to_dicts():
                k = row.pop(stratum_col)
                values_map[k] = row

        # Construct objects
        result = [
            SingletonInfo(
                stratum_key=str(row[stratum_col]),
                stratum_values=values_map.get(row[stratum_col], {}),
                psu_key=str(row["any_psu"]),
                n_observations=row["n_obs"],
            )
            for row in agg.to_dicts()
        ]
        # Sort for deterministic order
        return sorted(result, key=lambda s: s.stratum_key)

    def _get_all_strata_info(
        self,
        df: pl.DataFrame | None = None,
        order_by: str | Sequence[str] | None = None,
        stratum_col_override: str | None = None,
    ) -> list[StratumInfo]:
        """Get info for all strata."""
        if df is None:
            df = self._narrow_data()

        stratum_col, psu_col = self._internal_cols()
        if not stratum_col:
            return []

        # Allow override for rebalancing during collapse
        effective_stratum_col = stratum_col_override or stratum_col

        design = self._sample._design
        stratum_cols = self._to_cols(getattr(design, "stratum", None))
        order_cols = self._to_cols(order_by)

        # Build aggregation
        agg_exprs = [
            pl.col(psu_col).n_unique().alias("n_psus"),
            pl.len().alias("n_obs"),
        ]

        # Add order_by columns for sort values
        for col in order_cols:
            if col in df.columns:
                agg_exprs.append(pl.col(col).drop_nulls().first().alias(f"_order_{col}"))

        agg = cast(
            pl.DataFrame, df.lazy().group_by(effective_stratum_col).agg(agg_exprs).collect()
        )

        # Extract stratum values (use original stratum columns, not the override)
        values_map: dict[Any, dict[str, Any]] = {}
        if stratum_cols:
            val_df = (
                df.group_by(effective_stratum_col)
                .head(1)
                .select([effective_stratum_col] + stratum_cols)
            )
            for row in val_df.to_dicts():
                k = row.pop(effective_stratum_col)
                values_map[k] = row

        # Build StratumInfo objects
        result = []
        for row in agg.to_dicts():
            key = row[effective_stratum_col]
            sort_vals = tuple(row.get(f"_order_{col}") for col in order_cols)
            result.append(
                StratumInfo(
                    stratum_key=str(key),
                    stratum_values=values_map.get(key, {}),
                    n_psus=row["n_psus"],
                    n_observations=row["n_obs"],
                    sort_values=sort_vals,
                )
            )

        # Sort for deterministic order
        return sorted(result, key=lambda s: (s.sort_values, s.stratum_key))

    def _get_non_singleton_strata(
        self,
        df: pl.DataFrame | None = None,
        within: str | Sequence[str] | None = None,
        singleton: SingletonInfo | None = None,
        order_by: str | Sequence[str] | None = None,
        stratum_col_override: str | None = None,
    ) -> list[StratumInfo]:
        """Get all non-singleton strata, optionally filtered by `within` constraint."""
        if df is None:
            df = self._narrow_data()
        all_strata = self._get_all_strata_info(
            df, order_by=order_by, stratum_col_override=stratum_col_override
        )
        non_singletons = [s for s in all_strata if s.n_psus > 1]

        within_cols = self._to_cols(within)
        if not within_cols or singleton is None:
            return non_singletons
        stratum_col = stratum_col_override or self._internal_cols()[0]
        values = self._within_values(df, cast(str, stratum_col), within_cols)
        own = values.get(singleton.stratum_key)
        return [s for s in non_singletons if values.get(s.stratum_key) == own]

    def _within_values(
        self, df: pl.DataFrame, stratum_col: str, within_cols: list[str]
    ) -> dict[str, tuple[Any, ...]]:
        """Each stratum's values of the ``within`` columns, which must be
        constant within a stratum."""
        return self._stratum_values(df, stratum_col, within_cols, param="within")

    def _stratum_values(
        self, df: pl.DataFrame, stratum_col: str, cols: list[str], *, param: str
    ) -> dict[str, tuple[Any, ...]]:
        """Each stratum's values of ``cols`` (``within`` or ``order_by``), which
        must be in the data and constant within each stratum; missing values
        are ignored."""
        code, purpose, example = _STRATUM_COLUMN_ROLES[param]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SingletonError(
                title=f"{param} names columns not in the data",
                detail=f"collapse {param}={cols}: {missing} not in the data.",
                code=code,
                where="svy.Singleton",
                param=param,
                got=missing,
                hint=f"{param} names columns constant within each stratum, e.g. {example}.",
            )
        # A missing value says nothing about the stratum.
        grouped = df.group_by(stratum_col).agg(
            *[pl.col(c).drop_nulls().n_unique().alias(f"__n_{i}__") for i, c in enumerate(cols)],
            *[pl.col(c).drop_nulls().first().alias(c) for c in cols],
        )
        varying = [
            (c, grouped.filter(pl.col(f"__n_{i}__") > 1).get_column(stratum_col).to_list())
            for i, c in enumerate(cols)
        ]
        varying = [(c, keys) for c, keys in varying if keys]
        if varying:
            col, keys = varying[0]
            shown = self._key_values(sorted(map(str, keys))[:3])
            raise SingletonError(
                title=f"{param} column varies within a stratum",
                detail=(
                    f"collapse {param}={col!r}: {col!r} takes several values in "
                    f"{len(keys)} strata (e.g. {_shown(shown)}), so it does not {purpose}."
                ),
                code=code,
                where="svy.Singleton",
                param=param,
                got=col,
                hint=f"Use a column constant within each stratum, e.g. {example}.",
            )
        return {
            str(row[0]): tuple(row[1:]) for row in grouped.select(stratum_col, *cols).iter_rows()
        }

    def _select_target(
        self,
        singleton: SingletonInfo,
        candidates: list[StratumInfo],
        *,
        using: CollapseUsing,
        order_by: str | Sequence[str] | None,
        descending: bool,
        rstate: RandomState,
    ) -> StratumInfo:
        """Select target stratum for a singleton using the specified strategy."""
        if not candidates:
            raise SingletonError(
                title="No valid merge targets",
                detail=f"No non-singleton strata available for {singleton.stratum_key!r}",
                code="NO_MERGE_TARGETS",
                where="singleton.collapse",
            )

        # Handle dict mapping
        if isinstance(using, dict):
            target_key = cast(dict[str, str], using).get(singleton.stratum_key)
            if target_key is None:
                raise ValueError(
                    f"Mapping does not contain key for singleton {singleton.stratum_key!r}"
                )
            target = next((c for c in candidates if c.stratum_key == target_key), None)
            if target is None:
                raise ValueError(
                    f"Target {target_key!r} not found in candidates or is a singleton"
                )
            return target

        # Handle callable
        if callable(using):
            target_key = using(singleton, candidates)
            if isinstance(target_key, StratumInfo):
                target_key = target_key.stratum_key
            elif not any(c.stratum_key == target_key for c in candidates):
                target_key = self._resolve(target_key)
            target = next((c for c in candidates if c.stratum_key == target_key), None)
            if target is None:
                raise ValueError(
                    f"Callable returned {target_key!r} which is not a valid candidate"
                )
            return target

        # Sort candidates for deterministic behavior
        candidates_sorted = sorted(candidates, key=lambda c: (c.sort_values, c.stratum_key))

        # Handle string strategies
        if using == "smallest":
            min_size = min(c.n_psus for c in candidates_sorted)
            ties = [c for c in candidates_sorted if c.n_psus == min_size]
        elif using == "largest":
            max_size = max(c.n_psus for c in candidates_sorted)
            ties = [c for c in candidates_sorted if c.n_psus == max_size]
        elif using in ("next", "previous"):
            ties = self._get_adjacent_stratum(singleton, candidates, using, order_by, descending)
        else:
            raise ValueError(f"Unknown strategy: {using!r}")

        if len(ties) == 1:
            return ties[0]

        # Multiple ties - break them
        if rstate is not None:
            rng = resolve_random_state(rstate)
            idx = int(rng.integers(0, len(ties)))
            return ties[idx]
        else:
            # Deterministic: ties already sorted by (sort_values, stratum_key)
            return ties[0]

    def _get_adjacent_stratum(
        self,
        singleton: SingletonInfo,
        candidates: list[StratumInfo],
        direction: Literal["next", "previous"],
        order_by: str | Sequence[str] | None,
        descending: bool,
    ) -> list[StratumInfo]:
        """Get adjacent stratum(a) based on ordering."""
        # Sort all strata (including singleton position)
        all_strata = self._get_all_strata_info(order_by=order_by)

        # Sort by sort_values then key
        all_strata.sort(
            key=lambda s: (s.sort_values, s.stratum_key),
            reverse=descending,
        )

        # Find singleton position
        singleton_idx = next(
            (i for i, s in enumerate(all_strata) if s.stratum_key == singleton.stratum_key), None
        )

        if singleton_idx is None:
            # Fallback to smallest
            min_size = min(c.n_psus for c in candidates)
            return [c for c in candidates if c.n_psus == min_size]

        # Find adjacent non-singleton
        candidate_keys = {c.stratum_key for c in candidates}

        if direction == "next":
            # Look forward
            for i in range(singleton_idx + 1, len(all_strata)):
                if all_strata[i].stratum_key in candidate_keys:
                    return [
                        next(c for c in candidates if c.stratum_key == all_strata[i].stratum_key)
                    ]
            # Wrap around
            for i in range(0, singleton_idx):
                if all_strata[i].stratum_key in candidate_keys:
                    return [
                        next(c for c in candidates if c.stratum_key == all_strata[i].stratum_key)
                    ]
        else:  # previous
            # Look backward
            for i in range(singleton_idx - 1, -1, -1):
                if all_strata[i].stratum_key in candidate_keys:
                    return [
                        next(c for c in candidates if c.stratum_key == all_strata[i].stratum_key)
                    ]
            # Wrap around
            for i in range(len(all_strata) - 1, singleton_idx, -1):
                if all_strata[i].stratum_key in candidate_keys:
                    return [
                        next(c for c in candidates if c.stratum_key == all_strata[i].stratum_key)
                    ]

        # No adjacent found, return all candidates as ties
        return candidates

    # ══════════════════════════════════════════════════════════════════════
    # APPLY IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════════════

    def _apply_certainty(
        self, singles: list[SingletonInfo]
    ) -> tuple[pl.DataFrame, Design, SingletonResult]:
        """
        Apply certainty unit handling.

        For singleton strata, treats the PSU as a stratum and the SSU (or
        individual records if no SSU) as the PSUs for variance estimation.

        This creates internal columns for variance calculation without
        modifying the original design columns.
        """
        data = cast(pl.DataFrame, self._sample._data.clone())
        design = cast("Design", copy.deepcopy(self._sample._design))

        stratum_col, psu_col = self._internal_cols()
        assert stratum_col is not None

        # Get SSU column or fall back to row index
        idict = getattr(self._sample, "_internal_design", {}) or {}
        ssu_col = idict.get("ssu")
        # The effective PSU for variance is SSU if available, else row index
        effective_psu_source = ssu_col if ssu_col and ssu_col in data.columns else SVY_ROW_INDEX

        singleton_keys = [s.stratum_key for s in singles]
        n_strata_before, n_psus_before = self._counts_before(data, stratum_col, psu_col)

        is_singleton = pl.col(stratum_col).is_in(singleton_keys)

        # Create internal variance columns:
        # - For singletons: stratum = original PSU, PSU = SSU or row index
        # - For non-singletons: keep original stratum and PSU
        data = data.with_columns(
            # Effective stratum for variance: singleton PSU becomes stratum
            pl.when(is_singleton)
            .then(pl.col(psu_col).cast(pl.Utf8))
            .otherwise(pl.col(stratum_col).cast(pl.Utf8))
            .alias(_VAR_STRATUM_COL),
            # Effective PSU for variance: SSU or row index becomes PSU
            pl.when(is_singleton)
            .then(pl.col(effective_psu_source).cast(pl.Utf8))
            .otherwise(pl.col(psu_col).cast(pl.Utf8))
            .alias(_VAR_PSU_COL),
            # Not excluded
            pl.lit(False).alias(_VAR_EXCLUDE_COL),
        )

        # Count using internal variance columns
        n_strata_after, n_psus_after = self._counts_before(data, _VAR_STRATUM_COL, _VAR_PSU_COL)

        config = SingletonHandlingConfig(
            method=SingletonMethod.SELF_REPRESENTING,
            singleton_keys=tuple(singleton_keys),
            stratum_mapping=None,  # No mapping, just level shift
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=SingletonMethod.SELF_REPRESENTING,
            detected=tuple(singles),
            applied=tuple(singleton_keys),
            n_singletons_detected=len(singles),
            n_strata_before=n_strata_before,
            n_strata_after=n_strata_after,
            n_psus_before=n_psus_before,
            n_psus_after=n_psus_after,
            config=config,
        )
        return data, design, result

    def _apply_skip(
        self, singles: list[SingletonInfo]
    ) -> tuple[pl.DataFrame, Design, SingletonResult]:
        """
        Apply skip handling.

        Marks singleton strata as excluded from variance calculation. The data
        is NOT removed - rows remain in the dataset but are excluded from
        variance contribution (effectively contributing zero).

        This creates internal columns for variance calculation without
        modifying the original data or design columns.
        """
        data, design, n_strata_before, n_psus_before, n_strata_after, n_psus_after = (
            self._apply_exclude_variance_cols(singles)
        )
        singleton_keys = [s.stratum_key for s in singles]

        config = SingletonHandlingConfig(
            method=SingletonMethod.SKIP,
            singleton_keys=tuple(singleton_keys),
            stratum_mapping=None,
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=SingletonMethod.SKIP,
            detected=tuple(singles),
            applied=tuple(singleton_keys),
            n_singletons_detected=len(singles),
            n_strata_before=n_strata_before,
            n_strata_after=n_strata_after,
            n_psus_before=n_psus_before,
            n_psus_after=n_psus_after,
            config=config,
        )
        return data, design, result

    def _apply_scale(
        self, singles: list[SingletonInfo]
    ) -> tuple[pl.DataFrame, Design, SingletonResult]:
        """
        Apply scale handling.

        Uses the same exclusion logic as skip(), and marks the sample so the
        estimation engine scales each variance by nstrat/nokstrat, counted over
        the strata present in that estimate. ``singleton_fraction`` records the
        design-level fraction for reporting.

        This creates internal columns for variance calculation without
        modifying the original data or design columns.
        """
        data, design, n_strata_before, n_psus_before, n_strata_after, n_psus_after = (
            self._apply_exclude_variance_cols(singles)
        )
        singleton_keys = [s.stratum_key for s in singles]
        singleton_frac = len(singles) / n_strata_before if n_strata_before > 0 else 0.0

        config = SingletonHandlingConfig(
            method=SingletonMethod.SCALE,
            singleton_keys=tuple(singleton_keys),
            stratum_mapping=None,
            singleton_fraction=singleton_frac,
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=SingletonMethod.SCALE,
            detected=tuple(singles),
            applied=None,
            n_singletons_detected=len(singles),
            n_strata_before=n_strata_before,
            n_strata_after=n_strata_after,
            n_psus_before=n_psus_before,
            n_psus_after=n_psus_after,
            config=config,
        )
        return data, design, result

    def _apply_exclude_variance_cols(
        self,
        singles: list[SingletonInfo],
    ) -> tuple[pl.DataFrame, "Design", int, int, int, int]:
        """
        Shared setup for skip/scale: clone data+design, add internal variance
        columns that exclude singleton strata, return counts before/after.
        Returns (data, design, n_strata_before, n_psus_before, n_strata_after, n_psus_after).
        """
        data = cast(pl.DataFrame, self._sample._data.clone())
        design = cast("Design", copy.deepcopy(self._sample._design))

        stratum_col, psu_col = self._internal_cols()
        assert stratum_col is not None

        singleton_keys = [s.stratum_key for s in singles]
        n_strata_before, n_psus_before = self._counts_before(data, stratum_col, psu_col)

        is_singleton = pl.col(stratum_col).is_in(singleton_keys)

        data = data.with_columns(
            pl.col(stratum_col).cast(pl.Utf8).alias(_VAR_STRATUM_COL),
            pl.col(psu_col).cast(pl.Utf8).alias(_VAR_PSU_COL),
            is_singleton.alias(_VAR_EXCLUDE_COL),
        )

        non_excluded = data.filter(~pl.col(_VAR_EXCLUDE_COL))
        n_strata_after, n_psus_after = self._counts_before(
            non_excluded, _VAR_STRATUM_COL, _VAR_PSU_COL
        )
        return data, design, n_strata_before, n_psus_before, n_strata_after, n_psus_after

    def _apply_center(
        self, singles: list[SingletonInfo]
    ) -> tuple[pl.DataFrame, Design, SingletonResult]:
        """
        Apply center (adjust) handling.

        Does NOT exclude singletons. Instead, marks them so the estimation
        engine computes their variance contribution as (stratum_total - grand_mean)².

        This creates internal columns for variance calculation without
        modifying the original data or design columns.
        """
        data = cast(pl.DataFrame, self._sample._data.clone())
        design = cast("Design", copy.deepcopy(self._sample._design))

        stratum_col, psu_col = self._internal_cols()
        assert stratum_col is not None

        singleton_keys = [s.stratum_key for s in singles]
        n_strata_before, n_psus_before = self._counts_before(data, stratum_col, psu_col)

        is_singleton = pl.col(stratum_col).is_in(singleton_keys)

        # Create internal variance columns
        # - Keep original stratum and PSU structure
        # - Do NOT exclude singletons (exclude=False for all)
        # - Engine will use CENTER method for singleton variance
        data = data.with_columns(
            # Effective stratum for variance: same as original
            pl.col(stratum_col).cast(pl.Utf8).alias(_VAR_STRATUM_COL),
            # Effective PSU for variance: same as original
            pl.col(psu_col).cast(pl.Utf8).alias(_VAR_PSU_COL),
            # Do NOT exclude - singletons are included but handled specially
            pl.lit(False).alias(_VAR_EXCLUDE_COL),
            # Mark which rows are in singleton strata (for engine to identify)
            is_singleton.alias(_VAR_IS_SINGLETON_COL),
        )

        config = SingletonHandlingConfig(
            method=SingletonMethod.CENTER,
            singleton_keys=tuple(singleton_keys),
            stratum_mapping=None,
            singleton_fraction=None,
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=SingletonMethod.CENTER,
            detected=tuple(singles),
            applied=None,
            n_singletons_detected=len(singles),
            n_strata_before=n_strata_before,
            n_strata_after=n_strata_before,  # No change - singletons not excluded
            n_psus_before=n_psus_before,
            n_psus_after=n_psus_before,  # No change
            config=config,
        )
        return data, design, result

    def _apply_collapse(
        self,
        singles: list[SingletonInfo],
        *,
        using: CollapseUsing,
        within: str | Sequence[str] | None,
        order_by: str | Sequence[str] | None,
        descending: bool,
        rstate: RandomState,
    ) -> tuple[pl.DataFrame, Design, SingletonResult]:
        """
        Apply collapse handling with rebalancing.

        Merges singleton strata into existing non-singleton strata for
        variance estimation purposes. The original stratum column is NOT
        modified - internal columns are created for variance calculation.

        Rebalancing: Singletons are processed sequentially, with candidate
        stratum PSU counts recomputed after each merge to distribute
        singletons evenly across targets.
        """
        data = cast(pl.DataFrame, self._sample._data.clone())
        design = cast("Design", copy.deepcopy(self._sample._design))

        stratum_col, psu_col = self._internal_cols()
        assert stratum_col is not None

        n_strata_before, n_psus_before = self._counts_before(data, stratum_col, psu_col)

        # Sort singletons for deterministic processing order
        singles = sorted(singles, key=lambda s: s.stratum_key)

        # Track mapping for result
        applied_mapping: dict[str, str] = {}

        # Create a working column for rebalancing (will track effective stratum)
        _WORKING_STRATUM = "__svy_collapse_working__"
        data = data.with_columns(pl.col(stratum_col).alias(_WORKING_STRATUM))

        # Process singletons one by one with rebalancing
        for singleton in singles:
            # Get current candidates using working column (recomputed after each merge)
            candidates = self._get_non_singleton_strata(
                df=data,
                within=within,
                singleton=singleton,
                order_by=order_by,
                stratum_col_override=_WORKING_STRATUM,
            )

            if not candidates:
                raise SingletonError(
                    title="No valid merge targets",
                    detail=(
                        f"No non-singleton strata available for {singleton.stratum_key!r} "
                        f"with within={within}"
                    ),
                    code="NO_MERGE_TARGETS",
                    where="singleton.collapse",
                )

            # Select target
            target = self._select_target(
                singleton,
                candidates,
                using=using,
                order_by=order_by,
                descending=descending,
                rstate=rstate,
            )

            applied_mapping[singleton.stratum_key] = target.stratum_key

            # Update working column for rebalancing (so next iteration sees updated PSU counts)
            data = data.with_columns(
                pl.when(pl.col(_WORKING_STRATUM) == singleton.stratum_key)
                .then(pl.lit(target.stratum_key))
                .otherwise(pl.col(_WORKING_STRATUM))
                .alias(_WORKING_STRATUM)
            )

        # Create final internal variance columns from working column
        data = data.with_columns(
            # Effective stratum for variance: remapped stratum keys
            pl.col(_WORKING_STRATUM).cast(pl.Utf8).alias(_VAR_STRATUM_COL),
            # Effective PSU for variance: same as original
            pl.col(psu_col).cast(pl.Utf8).alias(_VAR_PSU_COL),
            # Not excluded
            pl.lit(False).alias(_VAR_EXCLUDE_COL),
        )

        # Remove working column
        data = data.drop(_WORKING_STRATUM)

        n_strata_after, n_psus_after = self._counts_before(data, _VAR_STRATUM_COL, _VAR_PSU_COL)

        config = SingletonHandlingConfig(
            method=SingletonMethod.COLLAPSE,
            singleton_keys=tuple(applied_mapping.keys()),
            stratum_mapping=applied_mapping,
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=SingletonMethod.COLLAPSE,
            detected=tuple(singles),
            applied=applied_mapping,
            n_singletons_detected=len(singles),
            n_strata_before=n_strata_before,
            n_strata_after=n_strata_after,
            n_psus_before=n_psus_before,
            n_psus_after=n_psus_after,
            config=config,
        )
        return data, design, result

    def _apply_pool(
        self,
        singles: list[SingletonInfo],
        *,
        name: str,
    ) -> tuple[pl.DataFrame, Design, SingletonResult]:
        """
        Apply pool handling - combine all singletons into one pseudo-stratum.

        Creates internal columns for variance calculation without modifying
        the original stratum column.
        """
        data = cast(pl.DataFrame, self._sample._data.clone())
        design = cast("Design", copy.deepcopy(self._sample._design))

        stratum_col, psu_col = self._internal_cols()
        assert stratum_col is not None

        singleton_keys = [s.stratum_key for s in singles]
        n_strata_before, n_psus_before = self._counts_before(data, stratum_col, psu_col)

        is_singleton = pl.col(stratum_col).is_in(singleton_keys)

        # Create internal variance columns:
        # - Singletons get pooled stratum name
        # - Non-singletons keep original stratum
        data = data.with_columns(
            # Effective stratum for variance: pool singletons together
            pl.when(is_singleton)
            .then(pl.lit(name))
            .otherwise(pl.col(stratum_col).cast(pl.Utf8))
            .alias(_VAR_STRATUM_COL),
            # Effective PSU for variance: same as original
            pl.col(psu_col).cast(pl.Utf8).alias(_VAR_PSU_COL),
            # Not excluded
            pl.lit(False).alias(_VAR_EXCLUDE_COL),
        )

        n_strata_after, n_psus_after = self._counts_before(data, _VAR_STRATUM_COL, _VAR_PSU_COL)

        stratum_mapping = {s.stratum_key: name for s in singles}

        config = SingletonHandlingConfig(
            method=SingletonMethod.POOL,
            singleton_keys=tuple(singleton_keys),
            stratum_mapping=stratum_mapping,
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=SingletonMethod.POOL,
            detected=tuple(singles),
            applied=stratum_mapping,
            n_singletons_detected=len(singles),
            n_strata_before=n_strata_before,
            n_strata_after=n_strata_after,
            n_psus_before=n_psus_before,
            n_psus_after=n_psus_after,
            config=config,
        )
        return data, design, result


# ═══════════════════════════════════════════════════════════════════════════
# THE SPEC AND ITS DERIVED STATE
# ═══════════════════════════════════════════════════════════════════════════


class _StrataIndex:
    """The strata of the data, found by their columns' values (a tuple for
    tuple strata), or as a fallback by svy's key string or the str form of the
    values (as contrast keys are); a fallback that names two strata is an
    error."""

    def __init__(self, data: pl.DataFrame, cols: list[str], key_col: str) -> None:
        rows = data.select([*cols, pl.col(key_col).cast(pl.Utf8)]).unique().rows()
        self.width = len(cols)
        self.values: dict[str, tuple[Any, ...]] = {row[-1]: row[:-1] for row in rows}
        self.exact: dict[tuple[Any, ...], str] = {row[:-1]: row[-1] for row in rows}
        self.fallback: dict[Any, str] = {}
        self.ambiguous: dict[Any, set[str]] = {}
        for vals, key in self.exact.items():
            forms = {key, tuple(str(v) for v in vals)}
            if self.width == 1:
                forms.add(str(vals[0]))
            for form in forms:
                self._offer(form, key)

    def _offer(self, form: Any, key: str) -> None:
        if form in self.ambiguous:
            self.ambiguous[form].add(key)
        elif form in self.fallback and self.fallback[form] != key:
            self.ambiguous[form] = {self.fallback.pop(form), key}
        else:
            self.fallback[form] = key

    def key(self, stratum: Any, *, strict: bool = True) -> str | None:
        if isinstance(stratum, list):
            stratum = tuple(stratum)
        t = stratum if isinstance(stratum, tuple) else (stratum,)
        try:
            if len(t) == self.width and t in self.exact:
                return self.exact[t]
        except TypeError:
            return None
        forms = [stratum] if isinstance(stratum, str) else []
        forms.append(tuple(str(v) for v in t))
        for form in forms:
            if form in self.fallback:
                return self.fallback[form]
            if form in self.ambiguous:
                if strict:
                    raise ValueError(
                        f"Stratum {stratum!r} matches several strata "
                        f"({', '.join(map(repr, sorted(self.ambiguous[form])))}); name it "
                        "by its columns' values."
                    )
                return None
        # A value saved in another type (a date read back from JSON as its ISO
        # string): compare in the key's own string form.
        if len(t) == self.width:
            key = _KEY_SEP.join(_key_part(v) for v in t)
            if key in self.values:
                return key
        return None


class DomainSingleton(msgspec.Struct, frozen=True):
    """A stratum with several PSUs whose rows in a domain sit in one of them."""

    #: The ``by`` columns' values; empty for a ``where=`` domain alone.
    domain: tuple[tuple[str, Any], ...]
    #: The stratum columns' values.
    stratum: tuple[tuple[str, Any], ...]

    @property
    def domain_label(self) -> str:
        return ", ".join(f"{c}={_fmt_value(v)}" for c, v in self.domain)

    @property
    def stratum_label(self) -> str:
        return ", ".join(f"{c}={_fmt_value(v)}" for c, v in self.stratum)

    @property
    def label(self) -> str:
        if not self.domain:
            return self.stratum_label
        return f"{self.stratum_label} in {self.domain_label}"


def _fmt_value(value: Any) -> str:
    """A value as the column shows it: ``2002``, not ``2002.0``, for an
    integer-valued float."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def domain_singleton_frame(
    df: pl.DataFrame,
    *,
    strata_col: str,
    psu_col: str | None,
    by_col: str | None = None,
    mask: pl.Expr | None = None,
    name_cols: Sequence[str] = (),
) -> pl.DataFrame:
    """One row per (``by_col`` level, stratum) whose domain rows sit in one of
    the stratum's several units; see :func:`find_domain_singletons`."""
    if mask is None and by_col is None:
        # Every row is in the domain: every stratum keeps all its units.
        return df.clear().select(strata_col)
    active = pl.lit(True) if mask is None else mask
    units = pl.col(psu_col).n_unique() if psu_col else pl.len()
    full = df.group_by(strata_col).agg(units.alias("__svy_n_full__"))
    keys = [by_col, strata_col] if by_col else [strata_col]
    firsts = [pl.col(c).first() for c in dict.fromkeys(name_cols) if c not in keys]
    return (
        df.filter(active)
        .group_by(keys)
        .agg(units.alias("__svy_n_dom__"), *firsts)
        .join(full, on=strata_col)
        .filter((pl.col("__svy_n_dom__") == 1) & (pl.col("__svy_n_full__") > 1))
    )


def find_domain_singletons(
    df: pl.DataFrame,
    *,
    strata_col: str,
    psu_col: str | None,
    stratum_cols: Sequence[str] = (),
    by_col: str | None = None,
    by_cols: Sequence[str] = (),
    mask: pl.Expr | None = None,
) -> list[DomainSingleton]:
    """Strata with several PSUs (rows, without PSUs) of which one holds rows of
    a domain: R's ``nsubset == 1 && nPSU > 1`` after ``subset()``.

    A domain row is in ``mask`` (``where=`` with the analysis variables
    present; every row when ``None``) and in one level of ``by_col`` when
    given, whatever its weight: R's ``subset()`` keeps zero-weight rows. Strata
    are named by ``stratum_cols`` and domains by ``by_cols``, from the rows.
    """
    names = list(dict.fromkeys(c for c in (*by_cols, *stratum_cols) if c in df.columns))
    found = domain_singleton_frame(
        df, strata_col=strata_col, psu_col=psu_col, by_col=by_col, mask=mask, name_cols=names
    )
    if found.is_empty():
        return []
    shown_by = [c for c in by_cols if c in found.columns]
    shown_strata = [c for c in stratum_cols if c in found.columns] or [strata_col]
    found = found.sort(list(dict.fromkeys([*shown_by, *shown_strata])), nulls_last=True)
    return [
        DomainSingleton(
            domain=tuple((c, row[c]) for c in shown_by),
            stratum=tuple(("stratum" if c == strata_col else c, row[c]) for c in shown_strata),
        )
        for row in found.iter_rows(named=True)
    ]


def _domain_text(
    pairs: Sequence[str],
    by_domains: bool,
    method: str | None,
    applied: bool,
    listed: int = 3,
) -> str:
    """The sentence a domain-singleton finding reports and prints: one count,
    of domain × stratum pairs (of strata for a ``where=`` domain alone)."""
    n = len(pairs)
    shown = "; ".join(pairs[:listed])
    if n > listed:
        shown += f"; {n - listed} more"
    if by_domains:
        what = f"{n} domain × stratum {'pair' if n == 1 else 'pairs'}"
    else:
        what = f"{n} {'stratum' if n == 1 else 'strata'}"
    if applied:
        how = (
            "centered at the grand mean"
            if method == SingletonMethod.CENTER
            else "left out and counted with the singletons"
        )
        return f'{what} with one PSU in the domain {how} (domains="apply"): {shown}'
    return f"{what} with one PSU in the domain ({shown}); standard domain variance used"


def _domain_hint(method: str | None, applied: bool) -> str:
    if applied:
        return 'Singleton(..., domains="standard") uses the standard domain variance.'
    rule = method if method in (SingletonMethod.CENTER, SingletonMethod.SCALE) else "center"
    return (
        f'svy.Singleton("{rule}", domains="apply") handles them as singletons; '
        "sample.domain_singletons(by=..., where=...) lists them."
    )


def domain_singleton_findings(
    sample: Sample,
    df: pl.DataFrame,
    *,
    strata_col: str | None,
    psu_col: str | None,
    by_col: str | None = None,
    by_cols: Sequence[str] = (),
    mask: pl.Expr | None = None,
    where: str,
) -> list[SvyWarning]:
    """Detect the strata with one PSU in the analysis's domains and report
    them as the rule's ``on_domain_singletons`` says (``"ignore"`` without a
    rule): the finding is recorded on the sample at INFO (never raised as a
    Python warning) and returned for the result, at WARNING under ``"warn"``
    so it prints as a note; ``"error"`` raises ``DOMAIN_SINGLETON``.

    A calibrated design has none: its scores are nonzero outside the domain,
    and R keeps those rows.
    """
    rule = getattr(sample._design, "singleton", None)
    on = rule.on_domain_singletons if rule is not None else "ignore"
    if strata_col is None or strata_col not in df.columns:
        return []
    from svy.core.data_prep import calib_applies

    if calib_applies(sample, df):
        return []
    stratum = sample._design.stratum
    stratum_cols = [stratum] if isinstance(stratum, str) else list(stratum or ())
    found = find_domain_singletons(
        df,
        strata_col=strata_col,
        psu_col=psu_col,
        stratum_cols=stratum_cols,
        by_col=by_col,
        by_cols=by_cols,
        mask=mask,
    )
    if not found:
        return []
    method = rule.method if rule is not None else None
    applied = domains_applied(sample)
    pairs = [f.label for f in found]
    by_domains = any(f.domain for f in found)
    strata = list(dict.fromkeys(f.stratum_label for f in found))
    doms = list(dict.fromkeys(f.domain_label for f in found))
    if on == "error":
        raise SingletonError.from_domain_singletons(
            pairs, by_domains=by_domains, method=method, applied=applied, where=where
        )
    from svy.core.warnings import Severity, SvyWarning, WarnCode

    fields: dict[str, Any] = dict(
        code=WarnCode.DOMAIN_SINGLETON_PSU,
        title="Strata with a single PSU in a domain",
        detail=_domain_text(pairs, by_domains, method, applied),
        where=where,
        param="on_domain_singletons",
        hint=_domain_hint(method, applied),
        extra={
            "pairs": pairs,
            "strata": strata,
            "domains": doms,
            "by_domains": by_domains,
            "method": None if method is None else method.value,
            "applied": applied,
        },
    )
    warn = getattr(sample, "warn", None)
    kept = warn(level=Severity.INFO, **fields) if warn is not None else SvyWarning(**fields)
    level = Severity.WARNING if on == "warn" else Severity.INFO
    return [msgspec.structs.replace(kept, level=level)]


def domain_singleton_note(findings: Sequence[Any]) -> str | None:
    """One note for every domain-singleton finding of a call that asked for
    one (``on_domain_singletons="warn"``): the pairs of several results
    (variables with different missing values) merged."""
    from svy.core.warnings import Severity

    found = [
        f
        for f in findings
        if f.code == "DOMAIN_SINGLETON_PSU" and f.extra and f.level >= Severity.WARNING
    ]
    if not found:
        return None
    first = found[0].extra
    pairs = list(dict.fromkeys(p for f in found for p in f.extra["pairs"]))
    by_domains = any(f.extra.get("by_domains", True) for f in found)
    text = _domain_text(pairs, by_domains, first["method"], first["applied"])
    return f"note: {text}"


def singleton_config(sample: Sample) -> SingletonHandlingConfig | None:
    """The variance settings of the sample's singleton rule, if it has one."""
    result = getattr(sample, "_singleton_result", None)
    return result.config if result else None


def taylor_singleton_method(sample: Sample) -> str | None:
    """What the Taylor kernels do with singleton strata: ``"center"``,
    ``"scale"``, or ``None`` (they contribute nothing; the other rules recode
    the strata and PSUs instead). ``":domains"`` is appended under
    ``domains="apply"``: the kernels then treat a stratum with one PSU in the
    domain as a singleton too."""
    config = singleton_config(sample)
    if config is None:
        return None
    method = str(getattr(config.method, "value", config.method)).lower()
    if method not in ("center", "scale"):
        return None
    return f"{method}:domains" if domains_applied(sample) else method


def domains_applied(sample: Sample) -> bool:
    """The rule handles a domain's one-PSU strata as singletons (``domains="apply"``)."""
    rule = getattr(sample._design, "singleton", None)
    return rule is not None and rule.domains == SingletonDomains.APPLY


def require_singleton_rule(sample: Sample, *, where: str) -> None:
    """Before a Taylor variance: raise when the design has singleton strata and
    no rule for them (R's ``options(survey.lonely.psu = "fail")``), or when its
    rule cannot be applied to the current data."""
    sync = getattr(sample, "_sync_parts", None)
    if sync is not None:
        sync()
    problem = sample.__dict__.get("_singleton_problem") if hasattr(sample, "__dict__") else None
    if problem is not None:
        raise msgspec_replace_error(problem, where=where)
    if singleton_config(sample) is None and getattr(sample, "_singletons", None):
        singles = _Engine(sample, _sync=False).detected()
        if singles:
            raise SingletonError.from_singletons(singles, where=where)


def singletons_frame(sample: Sample) -> pl.DataFrame:
    """``sample.singletons``: one row per singleton stratum of the current data,
    with its stratum and PSU columns' values, its rows, and how the rule
    handled it (null without a rule, or when the rule cannot handle it)."""
    sample._sync_parts()
    engine = _Engine(sample, _sync=False)
    design = sample._design
    data = engine._narrow_data()
    stratum_cols = _Engine._to_cols(design.stratum)
    psu_cols = _Engine._to_cols(design.variance_psu)
    cols = [c for c in dict.fromkeys([*stratum_cols, *psu_cols]) if c in data.columns]
    schema: dict[str, Any] = {c: data.schema[c] for c in cols}
    schema.update({"n": pl.UInt32, "handled": pl.Utf8})
    singles = engine.detected()
    if not singles:
        return pl.DataFrame(schema=schema)
    stratum_col, _ = engine._internal_cols()
    keys = [s.stratum_key for s in singles]
    handled = _handled(sample, engine, keys)
    key = pl.col(cast(str, stratum_col)).cast(pl.Utf8)
    rows = (
        data.filter(key.is_in(keys))
        .group_by(key.alias("__svy_key__"), maintain_order=True)
        .agg(*[pl.col(c).first() for c in cols], pl.len().cast(pl.UInt32).alias("n"))
    )
    order = {k: i for i, k in enumerate(keys)}
    return (
        rows.with_columns(
            pl.col("__svy_key__")
            .replace_strict(handled, default=None, return_dtype=pl.Utf8)
            .alias("handled"),
            pl.col("__svy_key__").replace_strict(order, return_dtype=pl.UInt32).alias("__o__"),
        )
        .sort("__o__")
        .select(*cols, "n", "handled")
    )


def _handled(sample: Sample, engine: _Engine, keys: list[str]) -> dict[str, str | None]:
    """How the rule handled each singleton stratum, by svy's key."""
    result = getattr(sample, "_singleton_result", None)
    rule = sample._design.singleton
    if result is None or rule is None:
        return {k: None for k in keys}
    method = SingletonMethod(result.method)
    if method is SingletonMethod.COLLAPSE:
        applied = cast(dict[str, str], result.applied)
        targets = dict(zip(applied, engine._key_values(list(applied.values()))))
        return {k: f"collapse -> {_label(targets[k])}" if k in targets else None for k in keys}
    done = {s.stratum_key for s in result.detected}
    text = f"pool -> {rule.name}" if method is SingletonMethod.POOL else method.value
    return {k: text if k in done else None for k in keys}


def _label(value: Any) -> str:
    """A stratum as a reader writes it: ``Center``, ``2002``, ``N, 4``."""
    if isinstance(value, tuple):
        return ", ".join(_fmt_value(v) for v in value)
    return _fmt_value(value)


def domain_singletons_frame(
    sample: Sample, by: str | Sequence[str] | None = None, where: WhereArg = None
) -> pl.DataFrame:
    """``sample.domain_singletons(...)``: strata with several PSUs of which one
    holds the rows of a domain (``by`` level within ``where``)."""
    from svy.utils.where import _compile_where

    sample._sync_parts()
    engine = _Engine(sample, _sync=False)
    stratum_col, psu_col = engine._internal_cols()
    design = sample._design
    data = engine._narrow_data()
    stratum_cols = _Engine._to_cols(design.stratum)
    psu_cols = _Engine._to_cols(design.variance_psu)
    by_cols = _Engine._to_cols(by)
    missing = [c for c in by_cols if c not in data.columns]
    if missing:
        from svy.errors.method_errors import MethodError

        raise MethodError.not_applicable(
            where="Sample.domain_singletons",
            method="domain_singletons",
            param="by",
            reason=f"columns not in the data: {missing}",
        )
    cols = [c for c in dict.fromkeys([*by_cols, *stratum_cols, *psu_cols]) if c in data.columns]
    schema: dict[str, Any] = {c: data.schema[c] for c in cols}
    schema.update({"n": pl.UInt32, "n_psus": pl.UInt32})
    empty = pl.DataFrame(schema=schema)
    if not stratum_col or stratum_col not in data.columns:
        return empty
    mask = _compile_where(where)
    if mask is None and not by_cols:
        return empty
    dom = "__svy_domain__"
    frame = data.with_columns(
        pl.struct(by_cols).alias(dom) if by_cols else pl.lit(None).alias(dom)
    )
    active = frame if mask is None else frame.filter(mask)
    full = frame.group_by(stratum_col).agg(pl.col(psu_col).n_unique().alias("n_psus"))
    found = (
        active.group_by(dom, stratum_col)
        .agg(
            pl.col(psu_col).n_unique().alias("__n_dom__"),
            pl.len().cast(pl.UInt32).alias("n"),
            *[pl.col(c).first() for c in cols],
        )
        .join(full, on=stratum_col)
        .filter((pl.col("__n_dom__") == 1) & (pl.col("n_psus") > 1))
    )
    out = found.select(*cols, "n", pl.col("n_psus").cast(pl.UInt32))
    order = [c for c in dict.fromkeys([*by_cols, *stratum_cols]) if c in data.columns]
    return out.sort(order, nulls_last=True) if order else out


def msgspec_replace_error(err: SingletonError, *, where: str) -> SingletonError:
    """The stored resolution error, raised afresh at the analysis that needs it."""
    import dataclasses

    return dataclasses.replace(err, where=where)


def _key_part(value: Any) -> str:
    if value is None:
        return _KEY_NULL
    return cast(str, pl.Series([value]).cast(pl.Utf8)[0])


def _basis(sample: Sample) -> tuple[tuple[Any, ...], list[str]]:
    """What singleton detection and the variance columns are computed from:
    the rows, the stratum/PSU/SSU columns (and svy's keys of them), the
    columns the rule reads, the variance columns themselves, and the rule."""
    design = sample._design
    rule = design.singleton
    key = (design.stratum, design.variance_psu, design.ssu, rule)
    idict = getattr(sample, "_internal_design", None) or {}
    cols = [
        SVY_ROW_INDEX,
        *_Engine._to_cols(design.stratum),
        *_Engine._to_cols(design.variance_psu),
        *_Engine._to_cols(design.ssu),
        *(idict.get(k) for k in ("stratum", "psu", "ssu")),
        *((rule.within or ()) if rule is not None else ()),
        *((rule.order_by or ()) if rule is not None else ()),
        *_VAR_COLS,
    ]
    names = set(sample._data.collect_schema().names())
    return key, [c for c in dict.fromkeys(cols) if c and c in names]


def _same(a: pl.Series, b: pl.Series) -> bool:
    if a.dtype != b.dtype or a.len() != b.len():
        return False
    try:
        # A column a step did not touch keeps its buffer.
        if a._get_buffer_info() == b._get_buffer_info():
            return True
    except Exception:
        pass
    return a.equals(b, check_names=False, null_equal=True)


def _derive_singleton_state(sample: Sample) -> None:
    """Detect the singletons again and apply the rule to them.

    Skipped when the rows, the strata/PSU/SSU columns, the columns the rule
    reads and the rule are those of the last run: a change elsewhere (a new
    column, a weight) cannot move them.
    """
    key, cols = _basis(sample)
    prev = sample.__dict__.get("_singleton_basis")
    data = cast(pl.DataFrame, sample._data)
    if (
        prev is not None
        and prev[0] == key
        and prev[1].columns == cols
        and prev[1].height == data.height
        and all(_same(prev[1].get_column(c), data.get_column(c)) for c in cols)
    ):
        return
    _rederive(sample)
    key, cols = _basis(sample)
    sample.__dict__["_singleton_basis"] = (key, cast(pl.DataFrame, sample._data).select(cols))


def _rederive(sample: Sample) -> None:
    """Detect the singletons as at construction and apply the declared rule to
    them: the variance columns and ``last_result`` are rebuilt from the rule
    and the data. The rule itself never changes.

    A rule that cannot be applied to this data (an explicit collapse mapping
    missing a singleton, a target gone, no stratum to merge into) is kept, and
    the analyses needing a Taylor variance raise its error.
    """
    ensure = getattr(sample, "_ensure_internal_concat", None)
    if ensure is not None:
        ensure()
    sample._check_for_singletons()
    data = sample._data
    stale = [c for c in _VAR_COLS if c in data.collect_schema().names()]
    if stale:
        sample._data = data.drop(stale)
    state = sample.__dict__
    state["_singleton_problem"] = None
    prev = getattr(sample, "_singleton_result", None)
    rule = sample._design.singleton
    if rule is None:
        # A combine() recode has no rule behind it and its report stays.
        if prev is not None and prev.config is not None:
            sample._singleton_result = None
        return
    sample._singleton_result = None

    facet = _Engine(sample, _sync=False)
    stratum_col, _ = facet._internal_cols()
    if not stratum_col or stratum_col not in facet._narrow_data().columns:
        return
    singles = facet.detected()
    method = rule.method
    domains_only = method in (SingletonMethod.CENTER, SingletonMethod.SCALE) and (
        rule.domains == SingletonDomains.APPLY
    )
    if not singles and not domains_only:
        return
    try:
        if method is SingletonMethod.SELF_REPRESENTING:
            new_data, _, result = facet._apply_certainty(singles)
        elif method is SingletonMethod.SKIP:
            new_data, _, result = facet._apply_skip(singles)
        elif method is SingletonMethod.SCALE:
            new_data, _, result = facet._apply_scale(singles)
        elif method is SingletonMethod.CENTER:
            new_data, _, result = facet._apply_center(singles)
        elif method is SingletonMethod.POOL:
            new_data, _, result = facet._apply_pool(singles, name=cast(str, rule.name))
        else:
            new_data, result = _resolve_collapse(sample, facet, rule, singles, prev)
    except SingletonError as err:
        state["_singleton_problem"] = err
        return
    except ValueError as err:
        state["_singleton_problem"] = SingletonError(
            title="The singleton rule cannot be applied",
            detail=str(err),
            code="SINGLETON_RULE_UNRESOLVED",
            where="svy.Singleton",
            param="singleton",
            hint="Change the rule with sample.update_design(singleton=...).",
        )
        return
    sample._data = new_data
    sample._singleton_result = result


def _resolve_collapse(
    sample: Sample,
    facet: _Engine,
    rule: SingletonRule,
    singles: list[SingletonInfo],
    prev: SingletonResult | None,
) -> tuple[pl.DataFrame, SingletonResult]:
    """Apply a collapse rule: an explicit mapping checked against the data, or
    a strategy re-run (an INFO finding when its mapping changed)."""
    from svy.core.warnings import Severity

    using: Any = rule.using
    if rule.order_by:
        stratum_col = cast(str, facet._internal_cols()[0])
        facet._stratum_values(
            facet._narrow_data(), stratum_col, list(rule.order_by), param="order_by"
        )
    if isinstance(using, tuple):
        using = _checked_mapping(sample, facet, using, singles, within=rule.within)
    new_data, _, result = facet._apply_collapse(
        singles,
        using=using,
        within=list(rule.within) if rule.within else None,
        order_by=list(rule.order_by) if rule.order_by else None,
        descending=rule.descending,
        rstate=rule.rstate,
    )
    if (
        not isinstance(rule.using, tuple)
        and prev is not None
        and prev.method == SingletonMethod.COLLAPSE
        and isinstance(prev.applied, dict)
        and prev.applied != result.applied
    ):
        pairs = cast(dict[str, str], result.applied)
        shown = _mapping_text(facet, pairs)
        sample.warn(
            level=Severity.INFO,
            code="SINGLETON_COLLAPSE_CHANGED",
            title="Singleton collapse changed",
            detail=f"the singletons now collapse as {shown} (using={rule.using!r}).",
            where="svy.Singleton",
            param="singleton",
            extra={"mapping": {str(k): str(v) for k, v in pairs.items()}},
        )
    return new_data, result


def _mapping_text(facet: _Engine, pairs: dict[str, str], listed: int = 5) -> str:
    keys = list(pairs)
    sources = facet._key_values(keys)
    targets = facet._key_values([pairs[k] for k in keys])
    items = [f"{_shown([a])} -> {_shown([b])}" for a, b in zip(sources, targets)]
    text = ", ".join(items[:listed])
    return text + (f", and {len(items) - listed} more" if len(items) > listed else "")


def _checked_mapping(
    sample: Sample,
    facet: _Engine,
    mapping: tuple[tuple[Any, Any], ...],
    singles: list[SingletonInfo],
    *,
    within: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """An explicit collapse mapping as svy's keys, for the singletons now.

    Entries for strata that are no longer singletons are ignored (INFO);
    a singleton it does not map, or a target that is gone, is itself a
    singleton or lies outside the singleton's ``within`` values, raises.
    """
    from svy.core.warnings import Severity

    index = facet._strata_index()
    now = {s.stratum_key for s in singles}
    keyed: dict[str, str] = {}
    unused: list[Any] = []
    gone: list[Any] = []
    lonely: list[Any] = []
    for source, target in mapping:
        s_key = facet._value_keys([source], index)[0]
        if s_key is None:
            s_key = _loose_key(facet, source, index)
        if s_key is None or s_key not in now:
            unused.append(source)
            continue
        t_key = facet._value_keys([target], index)[0]
        if t_key is None:
            t_key = _loose_key(facet, target, index)
        if t_key is None:
            gone.append(target)
        elif t_key in now:
            lonely.append(target)
        else:
            keyed[s_key] = t_key
    unmapped = [k for k in sorted(now) if k not in keyed]
    if unused:
        sample.warn(
            level=Severity.INFO,
            code="SINGLETON_MAPPING_UNUSED",
            title="Collapse mapping entries not used",
            detail=f"not singleton strata in this data, so not collapsed: {_shown(unused)}.",
            where="svy.Singleton",
            param="using",
            extra={"strata": [str(v) for v in unused]},
        )
    if gone:
        raise SingletonError(
            title="Collapse target not in the data",
            detail=f"the collapse mapping merges into strata the data no longer has: "
            f"{_shown(gone)}.",
            code="SINGLETON_TARGET_MISSING",
            where="svy.Singleton",
            param="using",
            got=gone,
            hint="Update the mapping: sample.update_design(singleton=svy.Singleton("
            '"collapse", using={...})).',
        )
    if lonely:
        raise SingletonError(
            title="Collapse target is a singleton",
            detail=f"the collapse mapping merges into strata that now have one PSU: "
            f"{_shown(lonely)}.",
            code="SINGLETON_TARGET_SINGLETON",
            where="svy.Singleton",
            param="using",
            got=lonely,
            hint="Map each singleton stratum to a stratum with two or more PSUs.",
        )
    if within and keyed:
        cols = list(within)
        stratum_col = cast(str, facet._internal_cols()[0])
        where_ = facet._within_values(facet._narrow_data(), stratum_col, cols)
        outside = [(a, b) for a, b in keyed.items() if where_.get(a) != where_.get(b)]
        if outside:
            pairs = [
                f"{_shown([x])} -> {_shown([y])}"
                for x, y in zip(
                    facet._key_values([a for a, _ in outside], index),
                    facet._key_values([b for _, b in outside], index),
                )
            ]
            raise SingletonError(
                title="Collapse target outside within",
                detail=f"the collapse mapping merges strata with different {', '.join(cols)}: "
                f"{', '.join(pairs)}.",
                code="SINGLETON_TARGET_OUTSIDE_WITHIN",
                where="svy.Singleton",
                param="using",
                got=pairs,
                hint=f"Map each singleton to a stratum with the same {', '.join(cols)}, or "
                "leave within out of the rule.",
            )
    if unmapped:
        values = facet._key_values(unmapped, index)
        raise SingletonError(
            title="Singletons missing from the collapse mapping",
            detail=f"{len(values)} singleton {'stratum is' if len(values) == 1 else 'strata are'} "
            f"not in the collapse mapping: {_shown(values[:10])}"
            + (" ..." if len(values) > 10 else "")
            + ".",
            code="SINGLETON_UNMAPPED",
            where="svy.Singleton",
            param="using",
            got=values,
            hint="Add them to the mapping (sample.singletons lists the singletons), or "
            'collapse with a strategy: svy.Singleton("collapse", using="smallest").',
        )
    return keyed


def _shown(values: Sequence[Any]) -> str:
    """Stratum values as the data shows them: strings quoted, ``2002`` for an
    integer-valued float, tuples element-wise."""

    def one(v: Any) -> str:
        if isinstance(v, tuple):
            return "(" + ", ".join(one(x) for x in v) + ")"
        return repr(v) if isinstance(v, str) else _fmt_value(v)

    return ", ".join(one(v) for v in values)


def _loose_key(facet: _Engine, stratum: Any, index: _StrataIndex | None) -> str | None:
    """svy's key for a stratum named loosely (its key string, or its values'
    str form), None when it names none."""
    if index is None:
        return None
    try:
        return index.key(stratum)
    except ValueError:
        return None
