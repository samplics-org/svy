from __future__ import annotations

import copy
import logging

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, cast

import msgspec
import polars as pl

from svy.core.constants import SVY_ROW_INDEX
from svy.core.enumerations import SingletonHandling as _SingletonHandling
from svy.errors.singleton_errors import SingletonError
from svy.utils.random_state import RandomState, resolve_random_state


if TYPE_CHECKING:
    from svy.core.design import Design, SingletonSpec
    from svy.core.sample import Sample

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL COLUMN NAMES FOR VARIANCE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

# These columns are used internally for variance estimation when singleton
# handling is applied. The original design columns are NEVER modified.
_VAR_STRATUM_COL = "__svy_var_stratum__"
_VAR_PSU_COL = "__svy_var_psu__"
_VAR_EXCLUDE_COL = "__svy_var_exclude__"
_VAR_IS_SINGLETON_COL = "__svy_var_is_singleton__"  # For CENTER method
_VAR_COLS = (_VAR_STRATUM_COL, _VAR_PSU_COL, _VAR_EXCLUDE_COL, _VAR_IS_SINGLETON_COL)

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

    method: _SingletonHandling
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

    method: _SingletonHandling
    detected: tuple[SingletonInfo, ...]
    applied: dict[str, str] | tuple[str, ...] | None = None
    n_singletons_detected: int = 0
    n_strata_before: int = 0
    n_strata_after: int = 0
    n_psus_before: int = 0
    n_psus_after: int = 0

    # The config to be attached to the sample for variance estimation
    config: SingletonHandlingConfig | None = None


class SingletonSummary(msgspec.Struct, frozen=True):
    """Summary of singleton situation with recommendation."""

    n_singletons: int
    n_strata: int
    n_psus: int
    pct_singletons: float
    affected_rows: int
    pct_rows_affected: float
    singletons: tuple[SingletonInfo, ...]
    recommendation: _SingletonHandling | None
    recommendation_reason: str | None


class StratumVarianceInfo(msgspec.Struct, frozen=True):
    """Variance contribution from a stratum."""

    stratum_key: str
    n_psus: int
    is_singleton: bool
    variance_contribution: float | None  # None for singletons
    n_observations: int
    psu_totals: tuple[float, ...] | None = None  # Weighted totals per PSU


# ═══════════════════════════════════════════════════════════════════════════
# TYPE ALIASES
# ═══════════════════════════════════════════════════════════════════════════

CollapseStrategy = Literal["next", "previous", "smallest", "largest"]
CollapseUsing = (
    CollapseStrategy | dict[str, str] | Callable[[SingletonInfo, list[StratumInfo]], str]
)


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON FACET
# ═══════════════════════════════════════════════════════════════════════════


class Singleton:
    """
    Facet for detecting and handling singleton PSUs.

    A singleton is a stratum that contains only one PSU. This is problematic
    for variance estimation in complex survey designs because we need at least
    two PSUs per stratum to estimate within-stratum variance.

    Access via: sample.singleton

    Examples
    --------
    >>> # Check for singletons
    >>> if sample.singleton.exists:
    ...     print(f"Found {sample.singleton.count} singletons")
    ...     print(sample.singleton.show())

    >>> # Handle singletons
    >>> new_sample = sample.singleton.collapse(using="smallest")
    >>> new_sample = sample.singleton.pool()
    >>> new_sample = sample.singleton.certainty()
    """

    __slots__ = ("_sample",)

    def __init__(self, sample: Sample, *, _sync: bool = True) -> None:
        self._sample = sample
        # The handling in effect is derived state: bring it up to date with the
        # data and design before anything here reads it.
        sync = getattr(sample, "_sync_parts", None) if _sync else None
        if sync is not None:
            sync()

    # ══════════════════════════════════════════════════════════════════════
    # QUICK CHECKS (cached-style properties)
    # ══════════════════════════════════════════════════════════════════════

    @property
    def exists(self) -> bool:
        """True if any singleton strata exist."""
        return len(self.detected()) > 0

    @property
    def count(self) -> int:
        """Number of singleton strata."""
        return len(self.detected())

    # ══════════════════════════════════════════════════════════════════════
    # INSPECTION
    # ══════════════════════════════════════════════════════════════════════

    def detected(self) -> list[SingletonInfo]:
        """
        Detect singleton strata from the sample's current data/design.

        Returns
        -------
        list[SingletonInfo]
            Information about each singleton stratum.
        """
        return self._detect_on_df(self._narrow_data(), self._narrow_design())

    def show(self) -> pl.DataFrame:
        """
        Tabular view of detected singletons.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: singleton_key, n_obs, psu, plus
            original stratum column values.
        """
        singles = self.detected()
        if not singles:
            return pl.DataFrame(
                schema={"singleton_key": pl.Utf8, "n_obs": pl.Int64, "psu": pl.Utf8}
            )

        data = [
            (s.stratum_key, s.n_observations, s.psu_key, *(s.stratum_values.values()))
            for s in singles
        ]

        extra_headers = list(singles[0].stratum_values.keys()) if singles else []
        headers = ["singleton_key", "n_obs", "psu"] + extra_headers

        return pl.DataFrame(data, schema=headers, orient="row")

    def keys(self) -> list[str]:
        """
        List of singleton stratum keys.

        Returns
        -------
        list[str]
            Internal stratum keys for all singletons.
        """
        return [s.stratum_key for s in self.detected()]

    def summary(self) -> SingletonSummary:
        """
        Rich summary of singleton situation with recommendation.

        Returns
        -------
        SingletonSummary
            Summary including counts, percentages, and a recommended
            handling strategy.
        """
        singles = self.detected()
        stratum_col, psu_col = self._internal_cols()

        if not stratum_col:
            return SingletonSummary(
                n_singletons=0,
                n_strata=0,
                n_psus=0,
                pct_singletons=0.0,
                affected_rows=0,
                pct_rows_affected=0.0,
                singletons=(),
                recommendation=None,
                recommendation_reason="No stratification defined",
            )

        _data = self._narrow_data()
        n_strata, n_psus = self._counts_before(_data, stratum_col, psu_col)
        total_rows = _data.height

        affected_rows = sum(s.n_observations for s in singles)
        pct_singletons = (len(singles) / n_strata * 100) if n_strata > 0 else 0.0
        pct_rows_affected = (affected_rows / total_rows * 100) if total_rows > 0 else 0.0

        # Determine recommendation
        recommendation, reason = self._recommend_strategy(
            singles, n_strata, affected_rows, total_rows
        )

        return SingletonSummary(
            n_singletons=len(singles),
            n_strata=n_strata,
            n_psus=n_psus,
            pct_singletons=pct_singletons,
            affected_rows=affected_rows,
            pct_rows_affected=pct_rows_affected,
            singletons=tuple(singles),
            recommendation=recommendation,
            recommendation_reason=reason,
        )

    # ══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def strata_profile(
        self,
        variables: str | Sequence[str] | None = None,
        *,
        weighted: bool = True,
    ) -> pl.DataFrame:
        """
        Profile all strata with summary statistics.

        Parameters
        ----------
        variables : str or sequence of str, optional
            Variables to summarize (mean). If None, only structural info.
        weighted : bool, default True
            Use survey weights for means.

        Returns
        -------
        pl.DataFrame
            Columns: stratum_key, n_psus, n_obs, is_singleton, plus
            mean of each variable.
        """
        stratum_col, psu_col = self._internal_cols()
        if not stratum_col:
            return pl.DataFrame()

        data = self._narrow_data()
        design = self._narrow_design()
        wgt_col = getattr(design, "wgt", None) if weighted else None

        # Base aggregations
        agg_exprs = [
            pl.col(psu_col).n_unique().alias("n_psus"),
            pl.len().alias("n_obs"),
        ]

        # Variable means
        var_list = self._to_cols(variables)
        for var in var_list:
            if var not in data.columns:
                log.warning(f"Variable {var!r} not in data, skipping")
                continue
            if wgt_col and wgt_col in data.columns:
                # Weighted mean (alias the whole ratio, not just the denominator)
                agg_exprs.append(
                    ((pl.col(var) * pl.col(wgt_col)).sum() / pl.col(wgt_col).sum()).alias(
                        f"mean_{var}"
                    )
                )
            else:
                agg_exprs.append(pl.col(var).mean().alias(f"mean_{var}"))

        result = (
            data.lazy()
            .group_by(stratum_col)
            .agg(agg_exprs)
            .with_columns((pl.col("n_psus") == 1).alias("is_singleton"))
            .sort(stratum_col)
            .collect()
        )

        return cast(pl.DataFrame, result).rename({stratum_col: "stratum_key"})

    def candidates_for(
        self,
        singleton_key: str,
        *,
        by: Literal["size", "similarity"] = "size",
        variables: Sequence[str] | None = None,
        within: str | Sequence[str] | None = None,
        top_k: int = 5,
        weighted: bool = True,
    ) -> pl.DataFrame:
        """
        Find candidate merge targets for a specific singleton.

        Parameters
        ----------
        singleton_key : str
            The stratum key of the singleton to find candidates for.
        by : {"size", "similarity"}, default "size"
            Ranking criterion:
            - "size": rank by number of PSUs (ascending)
            - "similarity": rank by similarity in `variables` (requires variables)
        variables : sequence of str, optional
            Variables to compute similarity. Required if by="similarity".
        within : str or sequence of str, optional
            Constrain candidates to same value(s) in these columns.
        top_k : int, default 5
            Number of top candidates to return.
        weighted : bool, default True
            Use survey weights for similarity calculation.

        Returns
        -------
        pl.DataFrame
            Top candidates with stratum_key, n_psus, n_obs, and ranking metric.
        """
        stratum_col, psu_col = self._internal_cols()
        if not stratum_col:
            return pl.DataFrame()

        singleton_key = self._resolve(singleton_key)
        singles = self.detected()
        singleton = next((s for s in singles if s.stratum_key == singleton_key), None)
        if singleton is None:
            raise ValueError(f"Singleton {singleton_key!r} not found")

        # Get all non-singleton strata
        candidates = self._get_non_singleton_strata(within=within, singleton=singleton)

        if not candidates:
            return pl.DataFrame(
                schema={"stratum_key": pl.Utf8, "n_psus": pl.Int64, "n_obs": pl.Int64}
            )

        if by == "similarity":
            if not variables:
                raise ValueError("variables required when by='similarity'")
            candidates = self._rank_by_similarity(
                singleton, candidates, variables, weighted=weighted
            )
        else:
            # Sort by size (smallest first)
            candidates.sort(key=lambda c: (c.n_psus, c.stratum_key))

        # Convert to DataFrame
        top = candidates[:top_k]
        return pl.DataFrame(
            {
                "stratum_key": [c.stratum_key for c in top],
                "n_psus": [c.n_psus for c in top],
                "n_obs": [c.n_observations for c in top],
            }
        )

    def compare(
        self,
        key1: str,
        key2: str,
        variables: Sequence[str] | None = None,
        *,
        weighted: bool = True,
    ) -> pl.DataFrame:
        """
        Compare two strata side-by-side.

        Parameters
        ----------
        key1, key2 : str
            Stratum keys to compare.
        variables : sequence of str, optional
            Variables to compare. If None, compares all numeric columns.
        weighted : bool, default True
            Use survey weights for statistics.

        Returns
        -------
        pl.DataFrame
            Columns: variable, stratum_1, stratum_2, difference.
        """
        stratum_col, _ = self._internal_cols()
        if not stratum_col:
            return pl.DataFrame()
        index = self._strata_index()
        key1, key2 = self._resolve(key1, index), self._resolve(key2, index)

        data = self._narrow_data()
        design = self._narrow_design()
        wgt_col = getattr(design, "wgt", None) if weighted else None

        # Determine variables to compare
        if variables is None:
            numeric_dtypes = {
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            }
            var_list = [
                c
                for c in data.columns
                if data.schema[c] in numeric_dtypes
                and c not in {stratum_col, wgt_col, SVY_ROW_INDEX}
            ]
        else:
            var_list = list(variables)

        if not var_list:
            return pl.DataFrame(
                schema={
                    "variable": pl.Utf8,
                    "stratum_1": pl.Float64,
                    "stratum_2": pl.Float64,
                    "difference": pl.Float64,
                }
            )

        # Compute means for both strata in a single Polars pass
        both_keys = [key1, key2]
        subset2 = data.filter(pl.col(stratum_col).is_in(both_keys))

        if wgt_col and wgt_col in data.columns:
            mean_exprs = [
                ((pl.col(v) * pl.col(wgt_col)).sum() / pl.col(wgt_col).sum()).alias(v)
                for v in var_list
                if v in data.columns
            ]
        else:
            mean_exprs = [pl.col(v).mean().alias(v) for v in var_list if v in data.columns]

        means_df = subset2.lazy().group_by(stratum_col).agg(mean_exprs).collect()

        def _row_as_means(key: str) -> dict[str, float]:
            rows = means_df.filter(pl.col(stratum_col) == key)
            if rows.is_empty():
                return {v: float("nan") for v in var_list}
            d = rows.row(0, named=True)
            d.pop(stratum_col, None)
            return {v: d.get(v, float("nan")) for v in var_list}

        means1 = _row_as_means(key1)
        means2 = _row_as_means(key2)

        return pl.DataFrame(
            {
                "variable": var_list,
                "stratum_1": [means1.get(v, float("nan")) for v in var_list],
                "stratum_2": [means2.get(v, float("nan")) for v in var_list],
                "difference": [
                    means1.get(v, float("nan")) - means2.get(v, float("nan")) for v in var_list
                ],
            }
        )

    def rank_strata(
        self,
        by: str | Sequence[str],
        *,
        weighted: bool = True,
        descending: bool = False,
    ) -> pl.DataFrame:
        """
        Rank all strata by variable(s).

        Parameters
        ----------
        by : str or sequence of str
            Variable(s) to rank by.
        weighted : bool, default True
            Use survey weights for aggregation.
        descending : bool, default False
            Sort descending (largest first).

        Returns
        -------
        pl.DataFrame
            Columns: rank, stratum_key, n_psus, is_singleton, plus mean of `by` vars.
        """
        by_cols = self._to_cols(by) if isinstance(by, str) else list(by)
        profile = self.strata_profile(variables=by_cols, weighted=weighted)

        if profile.is_empty():
            return pl.DataFrame()

        # Sort by the first variable
        sort_col = f"mean_{by_cols[0]}" if by_cols else "n_psus"
        if sort_col not in profile.columns:
            sort_col = "n_psus"

        result = profile.sort(sort_col, descending=descending).with_row_index("rank", offset=1)

        return result

    def suggest_mapping(
        self,
        *,
        variables: Sequence[str],
        within: str | Sequence[str] | None = None,
        weighted: bool = True,
    ) -> dict[str, str]:
        """
        Suggest merge mapping based on similarity.

        Parameters
        ----------
        variables : sequence of str
            Variables to compute similarity.
        within : str or sequence of str, optional
            Constrain candidates to same value(s) in these columns.
        weighted : bool, default True
            Use survey weights.

        Returns
        -------
        dict[str, str]
            Mapping {singleton_key: suggested_target_key}.
        """
        singles = self.detected()
        if not singles:
            return {}

        mapping = {}
        for singleton in singles:
            candidates = self._get_non_singleton_strata(within=within, singleton=singleton)
            if not candidates:
                log.warning(
                    f"No valid candidates for singleton {singleton.stratum_key!r} "
                    f"with within={within}"
                )
                continue

            ranked = self._rank_by_similarity(singleton, candidates, variables, weighted=weighted)
            if ranked:
                mapping[singleton.stratum_key] = ranked[0].stratum_key

        return mapping

    # ══════════════════════════════════════════════════════════════════════
    # VARIANCE COMPUTATION (for estimation with singletons)
    # ══════════════════════════════════════════════════════════════════════

    def variance_contributions(
        self,
        variable: str,
        *,
        weighted: bool = True,
    ) -> list[StratumVarianceInfo]:
        """
        Compute variance contribution from each stratum for a variable.

        This computes the Taylor series variance contribution from each stratum.
        Singleton strata will have variance_contribution=None since we cannot
        compute within-stratum variance from a single PSU.

        Parameters
        ----------
        variable : str
            The variable to compute variance for.
        weighted : bool, default True
            Use survey weights.

        Returns
        -------
        list[StratumVarianceInfo]
            Variance info for each stratum, sorted by stratum key.

        Notes
        -----
        For Taylor series linearization, the variance contribution from
        stratum h is:

            V_h = (n_h / (n_h - 1)) * Σᵢ(y_hi - ȳ_h)²

        where n_h is the number of PSUs, y_hi is the weighted total for
        PSU i, and ȳ_h is the mean of PSU totals.
        """
        stratum_col, psu_col = self._internal_cols()
        if not stratum_col:
            return []

        data = self._narrow_data()
        design = self._narrow_design()
        wgt_col = getattr(design, "wgt", None) if weighted else None

        if variable not in data.columns:
            raise ValueError(f"Variable {variable!r} not found in data")

        # Compute weighted totals per PSU within each stratum
        if wgt_col and wgt_col in data.columns:
            psu_totals = (
                data.lazy()
                .group_by([stratum_col, psu_col])
                .agg(
                    (pl.col(variable) * pl.col(wgt_col)).sum().alias("psu_total"),
                    pl.len().alias("n_obs"),
                )
                .collect()
            )
        else:
            psu_totals = (
                data.lazy()
                .group_by([stratum_col, psu_col])
                .agg(
                    pl.col(variable).sum().alias("psu_total"),
                    pl.len().alias("n_obs"),
                )
                .collect()
            )
        psu_totals = cast(pl.DataFrame, psu_totals)

        # Compute per-stratum variance in Polars: V_h = (n_h/(n_h-1)) * sum((y_hi - ȳ_h)²)
        stratum_stats = (
            psu_totals.lazy()
            .group_by(stratum_col)
            .agg(
                pl.col("psu_total").count().alias("n_psus"),
                pl.col("n_obs").sum().alias("n_obs_total"),
                pl.col("psu_total").mean().alias("mean_total"),
                pl.col("psu_total").var(ddof=0).alias("var_pop"),  # population variance
            )
            .with_columns(
                # V_h = (n/(n-1)) * n * pop_var  = n²/(n-1) * pop_var
                # But pop_var = sum_sq_dev/n, so V_h = n/(n-1) * sum_sq_dev
                # Use: V_h = n/(n-1) * n * var_pop  when n > 1, else null
                pl.when(pl.col("n_psus") > 1)
                .then(
                    (pl.col("n_psus").cast(pl.Float64) / (pl.col("n_psus") - 1))
                    * pl.col("n_psus").cast(pl.Float64)
                    * pl.col("var_pop")
                )
                .otherwise(pl.lit(None))
                .alias("variance")
            )
            .sort(stratum_col)
            .collect()
        )
        stratum_stats = cast(pl.DataFrame, stratum_stats)

        # Keep per-PSU totals for StratumVarianceInfo.psu_totals
        totals_by_stratum: dict[Any, list[float]] = {}
        for row in psu_totals.to_dicts():
            k = row[stratum_col]
            totals_by_stratum.setdefault(k, []).append(float(row["psu_total"]))

        result = []
        for row in stratum_stats.to_dicts():
            key = row[stratum_col]
            totals = totals_by_stratum.get(key, [])
            result.append(
                StratumVarianceInfo(
                    stratum_key=str(key),
                    n_psus=row["n_psus"],
                    is_singleton=row["n_psus"] == 1,
                    variance_contribution=row["variance"],
                    n_observations=int(row["n_obs_total"] or 0),
                    psu_totals=tuple(totals),
                )
            )

        return result

    # ══════════════════════════════════════════════════════════════════════
    # HANDLING METHODS (all return new Sample)
    # ══════════════════════════════════════════════════════════════════════

    def raise_error(self) -> None:
        """
        Raise an error if any singleton PSUs are detected.

        Raises
        ------
        SingletonError
            If any singletons exist.
        """
        singles = self.detected()
        if singles:
            raise SingletonError.from_singletons(singles)

    def certainty(self) -> Sample:
        """
        Treat singleton PSUs as 'certainty units'.

        Each observation within a singleton stratum gets its own unique PSU ID,
        effectively treating each unit as self-representing.

        Returns
        -------
        Sample
            New sample with singleton observations as individual PSUs.
        """
        singles = self.detected()
        if not singles:
            return self._sample

        stratum_col, psu_col = self._internal_cols()
        if not stratum_col:
            return self._sample

        # Already handled: estimation gates on the recorded decision, not the
        # data, so only an existing certainty result short-circuits. (A former
        # ids-already-conform check was trivially true for no-PSU designs —
        # each row is its own PSU — and returned the sample with no config,
        # so the assertion was never recorded and estimation still failed.)
        existing = getattr(self._sample, "_singleton_result", None)
        if existing is not None and existing.method == _SingletonHandling.CERTAINTY:
            return self._sample

        data, design, result = self._apply_certainty(singles)
        return self._install(data, design, result, lambda: _spec("certainty", self, result))

    def skip(self) -> Sample:
        """
        Let singleton strata contribute nothing to the variance.

        All rows stay in the estimator; only the variance contribution of each
        one-PSU stratum is dropped. This is R's ``lonely.psu = "remove"``.

        Returns
        -------
        Sample
            New sample carrying the skip recipe.

        Warnings
        --------
        The variance is understated by the singletons' share. Consider
        `scale()`, `collapse()` or `pool()` instead.
        """
        singles = self.detected()
        if not singles:
            return self._sample

        data, design, result = self._apply_skip(singles)
        return self._install(data, design, result, lambda: _spec("skip", self, result))

    def combine(self, mapping: dict[str, dict[str, str]]) -> Sample:
        """
        Remap values in stratum/PSU columns per explicit mapping.

        This is a low-level method that directly modifies column values.
        For most use cases, prefer `collapse()` which handles the mapping
        automatically.

        Parameters
        ----------
        mapping : dict[str, dict[str, str]]
            Mapping of {column_name: {old_value: new_value}}.
            Columns must be in design.stratum or design.psu.

        Returns
        -------
        Sample
            New sample with remapped values.

        Raises
        ------
        SingletonError
            If the mapping doesn't resolve all singletons.
        ValueError
            If mapping refers to invalid columns.
        """
        singles = self.detected()
        if not singles:
            return self._sample

        data, design, result = self._apply_combine(mapping, singles)
        # A recode of the data, not a spec: a spec from an earlier rule no
        # longer describes the recoded strata and goes with it, silently.
        if getattr(design, "singleton", None) is not None:
            design = design._with_part("singleton", None)
        return self._clone_with_result(data, design, result)

    def collapse(
        self,
        *,
        using: CollapseUsing = "smallest",
        within: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
        descending: bool = False,
        rstate: RandomState = None,
    ) -> Sample:
        """
        Merge each singleton into an existing non-singleton stratum.

        Parameters
        ----------
        using : str, dict, or callable, default "smallest"
            Strategy for selecting target stratum:
            - "smallest": smallest non-singleton (by PSU count)
            - "largest": largest non-singleton
            - "next": next stratum by `order_by` (or stratum key)
            - "previous": previous stratum by `order_by`
            - dict[str, str]: explicit {singleton_key: target_key}
            - callable(SingletonInfo, list[StratumInfo]) -> str

        within : str or sequence of str, optional
            Constrain candidates to matching values in these columns.
            E.g., within="region" only considers strata in the same region.

        order_by : str or sequence of str, optional
            Column(s) for ordering strata when using "next"/"previous",
            and for deterministic tie-breaking.

        descending : bool, default False
            Sort order for `order_by`.

        rstate : RandomState, optional
            If provided, ties are broken randomly (reproducible with int seed).
            If None (default), ties are broken deterministically by `order_by`
            then stratum key.

        Returns
        -------
        Sample
            New sample with singletons merged into existing strata.

        Notes
        -----
        When multiple strata tie as candidates (e.g., same size), tie-breaking:
        - rstate=None: deterministic (by order_by, then stratum key)
        - rstate=42: random but reproducible
        - rstate=np.random.default_rng(): random

        Singletons are processed with rebalancing: after each merge, candidate
        sizes are recalculated, distributing singletons more evenly.

        Examples
        --------
        >>> # Merge into smallest stratum
        >>> new_sample = sample.singleton.collapse(using="smallest")

        >>> # Merge within same region
        >>> new_sample = sample.singleton.collapse(using="smallest", within="region")

        >>> # Use explicit mapping
        >>> mapping = {"singleton_A": "target_B", "singleton_C": "target_D"}
        >>> new_sample = sample.singleton.collapse(using=mapping)

        >>> # Custom strategy with proximity matrix
        >>> def find_nearest(singleton, candidates):
        ...     return max(candidates, key=lambda c: proximity[singleton.stratum_key, c.stratum_key]).stratum_key
        >>> new_sample = sample.singleton.collapse(using=find_nearest)
        """
        singles = self.detected()
        if not singles:
            return self._sample

        if isinstance(using, dict):
            # Strata by their columns' values (tuples for tuple strata), or by
            # svy's key strings as before.
            index = self._strata_index()
            using = {self._resolve(k, index): self._resolve(v, index) for k, v in using.items()}
        data, design, result = self._apply_collapse(
            singles,
            using=using,
            within=within,
            order_by=order_by,
            descending=descending,
            rstate=rstate,
        )
        return self._install(data, design, result, lambda: _spec("collapse", self, result))

    def pool(self, *, name: str = "__pooled__") -> Sample:
        """
        Combine all singletons into a single new pseudo-stratum.

        Unlike `collapse()`, which merges singletons into existing strata,
        `pool()` creates a new stratum containing only the former singleton PSUs.

        Parameters
        ----------
        name : str, default "__pooled__"
            Name for the new pseudo-stratum.

        Returns
        -------
        Sample
            New sample with all singletons in a single pseudo-stratum.

        Notes
        -----
        The pooled stratum will have multiple PSUs (one per former singleton),
        allowing variance estimation. This is appropriate when:
        - Singletons don't naturally belong to any existing stratum
        - You want to preserve the original structure of non-singleton strata
        - Singletons represent "miscellaneous" or "other" units

        Examples
        --------
        Before: Strata A(1 PSU), B(1 PSU), C(2 PSUs), D(2 PSUs)
        After:  Strata __pooled__(2 PSUs), C(2 PSUs), D(2 PSUs)
        """
        singles = self.detected()
        if not singles:
            return self._sample

        data, design, result = self._apply_pool(singles, name=name)
        return self._install(data, design, result, lambda: _spec("pool", self, result, name=name))

    def scale(self) -> Sample:
        """
        Mark for variance scaling by singleton fraction.

        Singleton strata contribute nothing to the variance, and the variance is
        inflated by 1/(1 - singleton_frac). All rows stay in the estimator.

        This is equivalent to R's `lonely.psu = "average"` option.

        Returns
        -------
        Sample
            New sample marked for variance scaling.

        Notes
        -----
        Point estimates use the full sample. Variances are computed with the
        singleton strata's contributions dropped, then multiplied by
        ``n_strata / n_ok_strata``, i.e. ``1 / (1 - f)`` with
        ``f = n_singletons / n_strata`` — e.g. with 20% singleton strata the
        variance is multiplied by ``1/0.8 = 1.25``. Both counts are taken over
        the strata that hold the rows being estimated, as R does after
        ``subset()`` and inside ``svyby()``: a ``by=`` level or ``where=``
        domain that leaves whole strata out gets its own fraction, a domain
        with no singleton stratum is not inflated, and a domain lying entirely
        in singleton strata has no reference variance and reports ``NaN``.
        The factor is the same for every Taylor estimator (totals, means,
        ratios, proportions, quantiles, correlations) and applies to the whole
        covariance matrix and to the design effect, as in R.

        This assumes singleton strata would have contributed "average"
        variance had they held multiple PSUs.

        Examples
        --------
        >>> sample_scaled = sample.singleton.scale()
        >>> result = sample_scaled.estimation.mean("income")  # Variance auto-scaled
        """

        singles = self.detected()
        if not singles:
            return self._sample

        data, design, result = self._apply_scale(singles)
        return self._install(data, design, result, lambda: _spec("scale", self, result))

    def center(self) -> Sample:
        """
        Mark for grand-mean centering of singleton variance.

        This method does NOT exclude singleton strata. Instead, it tells
        the variance estimation engine to compute singleton variance
        contribution using (stratum_total - grand_mean)².

        This is equivalent to R's `lonely.psu = "adjust"` option and to
        Stata's `singleunit(centered)` — the conventional lonely-PSU
        handling for DHS analyses.

        Returns
        -------
        Sample
            New sample marked for grand-mean centering.

        Notes
        -----
        For singleton strata, instead of the undefined within-stratum
        variance, the contribution is based on how different the
        singleton's total is from the grand mean of PSU totals:

            V_singleton = (y_h - ȳ)²

        where y_h is the singleton's PSU total and ȳ is the mean of
        all PSU totals across all strata.

        This preserves the full sample for point estimation while
        providing a variance estimate that reflects the singleton's
        deviation from the overall pattern.

        Examples
        --------
        >>> sample_centered = sample.singleton.center()
        >>> result = sample_centered.estimation.mean("income")  # Uses centered variance
        """
        singles = self.detected()
        if not singles:
            return self._sample

        data, design, result = self._apply_center(singles)
        return self._install(data, design, result, lambda: _spec("center", self, result))

    def handle(
        self,
        method: Literal[
            "error", "certainty", "skip", "combine", "collapse", "pool", "scale", "center"
        ],
        **kwargs: Any,
    ) -> Sample:
        """
        Unified dispatcher for singleton handling methods.

        Parameters
        ----------
        method : str
            The handling method to apply:
            - ``'error'``: raise error if singletons exist
            - ``'certainty'``: treat as certainty units
            - ``'skip'``: remove singleton rows
            - ``'combine'``: manual remapping
            - ``'collapse'``: merge into existing strata
            - ``'pool'``: combine into pseudo-stratum
            - ``'scale'``: post-hoc variance inflation
            - ``'center'``: grand-mean centering

        **kwargs
            Arguments passed to the specific method.

        Returns
        -------
        Sample
            New sample with singletons handled.

        Examples
        --------
        >>> method = config.get("singleton_handling", "certainty")
        >>> new_sample = sample.singleton.handle(method)

        >>> new_sample = sample.singleton.handle("collapse", using="smallest", within="region")
        """
        if isinstance(method, str):
            method = method.lower()

        dispatch = {
            _SingletonHandling.ERROR: lambda: (self.raise_error(), self._sample)[1],
            _SingletonHandling.CERTAINTY: self.certainty,
            _SingletonHandling.SKIP: self.skip,
            _SingletonHandling.COMBINE: self.combine,
            _SingletonHandling.COLLAPSE: self.collapse,
            _SingletonHandling.POOL: self.pool,
            _SingletonHandling.SCALE: self.scale,
            _SingletonHandling.CENTER: self.center,
            "error": lambda: (self.raise_error(), self._sample)[1],
            "certainty": self.certainty,
            "skip": self.skip,
            "combine": self.combine,
            "collapse": self.collapse,
            "pool": self.pool,
            "scale": self.scale,
            "center": self.center,
        }

        # Accept both the enum and its string value ('.strip()' on a plain
        # Enum raised AttributeError, so handle(SingletonHandling.POOL)
        # crashed while handle("pool") worked).
        key = method.strip().lower() if isinstance(method, str) else method
        handler = dispatch.get(key)
        if handler is None:
            raise ValueError(f"Unknown method {method!r}. Use one of: {tuple(dispatch)}.")

        return handler(**kwargs)

    # ══════════════════════════════════════════════════════════════════════
    # RESULT ACCESS
    # ══════════════════════════════════════════════════════════════════════

    @property
    def last_result(self) -> SingletonResult | None:
        """
        Result from the most recent singleton handling operation.

        Returns
        -------
        SingletonResult or None
            The result if this sample was produced by a singleton handling
            operation, otherwise None.
        """
        return getattr(self._sample, "_singleton_result", None)

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

    def _install(
        self,
        data: pl.DataFrame,
        design: Design,
        result: SingletonResult,
        spec: Callable[[], SingletonSpec],
    ) -> Sample:
        """A new sample whose design carries the resolved rule.

        The rule goes in with ``update_design``, so it is in ``design_history``,
        and the variance columns are derived from it (and rebuilt whenever the
        data or design changes). A host without a design history gets the
        derived columns directly.
        """
        if not hasattr(self._sample, "update_design"):
            return self._clone_with_result(data, design, result)
        new = self._sample._fork()
        new.update_design(singleton=spec())
        return new

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

    def _clone_with_result(
        self,
        data: pl.DataFrame,
        design: Design,
        result: SingletonResult,
    ) -> Sample:
        """Clone sample and attach singleton result."""
        from svy.core.sample import Sample as _Sample

        new_sample = cast(_Sample, self._sample.clone(data=data, design=design))
        object.__setattr__(new_sample, "_singleton_result", result)
        return new_sample

    def _recommend_strategy(
        self,
        singles: list[SingletonInfo],
        n_strata: int,
        affected_rows: int,
        total_rows: int,
    ) -> tuple[_SingletonHandling | None, str | None]:
        """Determine recommended handling strategy."""
        if not singles:
            return None, "No singletons detected"

        pct_singletons = (len(singles) / n_strata * 100) if n_strata > 0 else 0
        pct_rows = (affected_rows / total_rows * 100) if total_rows > 0 else 0

        # Heuristic recommendations
        if pct_rows < 1:
            return _SingletonHandling.SKIP, (
                f"Singletons affect <1% of rows ({pct_rows:.1f}%); "
                "removing them has minimal impact"
            )

        if pct_singletons < 10:
            return _SingletonHandling.COLLAPSE, (
                f"Few singletons ({pct_singletons:.1f}% of strata); "
                "merging into similar strata preserves design"
            )

        if len(singles) >= 2:
            return _SingletonHandling.POOL, (
                f"Multiple singletons ({len(singles)}); "
                "pooling creates a valid pseudo-stratum for variance estimation"
            )

        return _SingletonHandling.CERTAINTY, (
            "Single singleton stratum; treating as certainty unit allows variance estimation"
        )

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
                agg_exprs.append(pl.col(col).first().alias(f"_order_{col}"))

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
        all_strata = self._get_all_strata_info(
            df, order_by=order_by, stratum_col_override=stratum_col_override
        )
        non_singletons = [s for s in all_strata if s.n_psus > 1]

        if within is None or singleton is None:
            return non_singletons

        # Filter by within constraint
        within_cols = self._to_cols(within)
        if not within_cols:
            return non_singletons

        # Get singleton's values for within columns
        singleton_within_values = {col: singleton.stratum_values.get(col) for col in within_cols}

        # Filter candidates
        return [
            s
            for s in non_singletons
            if all(
                s.stratum_values.get(col) == singleton_within_values.get(col)
                for col in within_cols
            )
        ]

    def _rank_by_similarity(
        self,
        singleton: SingletonInfo,
        candidates: list[StratumInfo],
        variables: Sequence[str],
        *,
        weighted: bool = True,
    ) -> list[StratumInfo]:
        """Rank candidates by similarity to singleton."""
        stratum_col, _ = self._internal_cols()
        if not stratum_col:
            return candidates

        data = self._narrow_data()
        design = self._narrow_design()
        wgt_col = getattr(design, "wgt", None) if weighted else None
        var_list = list(variables)

        def compute_means(stratum_key: str) -> dict[str, float]:
            subset = data.filter(pl.col(stratum_col) == stratum_key)
            if subset.height == 0:
                return {}

            means = {}
            for var in var_list:
                if var not in subset.columns:
                    continue
                if wgt_col and wgt_col in subset.columns:
                    total = (subset[var] * subset[wgt_col]).sum()
                    wgt_sum = subset[wgt_col].sum()
                    means[var] = total / wgt_sum if wgt_sum > 0 else float("nan")
                else:
                    means[var] = subset[var].mean()
            return means

        # Compute singleton means
        singleton_means = compute_means(singleton.stratum_key)

        # Compute distances
        distances: list[tuple[StratumInfo, float]] = []
        for cand in candidates:
            cand_means = compute_means(cand.stratum_key)
            # Euclidean distance on standardized variables (simple approach)
            dist = 0.0
            n_vars = 0
            for var in var_list:
                if var in singleton_means and var in cand_means:
                    s_val = singleton_means[var]
                    c_val = cand_means[var]
                    if not (
                        s_val is None
                        or c_val is None
                        or (isinstance(s_val, float) and (s_val != s_val))
                        or (isinstance(c_val, float) and (c_val != c_val))
                    ):
                        dist += (s_val - c_val) ** 2
                        n_vars += 1
            if n_vars > 0:
                dist = (dist / n_vars) ** 0.5
            else:
                dist = float("inf")
            distances.append((cand, dist))

        # Sort by distance (ascending)
        distances.sort(key=lambda x: (x[1], x[0].stratum_key))
        return [d[0] for d in distances]

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
            method=_SingletonHandling.CERTAINTY,
            singleton_keys=tuple(singleton_keys),
            stratum_mapping=None,  # No mapping, just level shift
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=_SingletonHandling.CERTAINTY,
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
            method=_SingletonHandling.SKIP,
            singleton_keys=tuple(singleton_keys),
            stratum_mapping=None,
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=_SingletonHandling.SKIP,
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
            method=_SingletonHandling.SCALE,
            singleton_keys=tuple(singleton_keys),
            stratum_mapping=None,
            singleton_fraction=singleton_frac,
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=_SingletonHandling.SCALE,
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
            method=_SingletonHandling.CENTER,
            singleton_keys=tuple(singleton_keys),
            stratum_mapping=None,
            singleton_fraction=None,
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=_SingletonHandling.CENTER,
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

    def _apply_combine(
        self,
        mapping: dict[str, dict[str, str]],
        singles: list[SingletonInfo],
    ) -> tuple[pl.DataFrame, Design, SingletonResult]:
        """Apply combine handling."""
        data = cast(pl.DataFrame, self._sample._data.clone())
        design = cast("Design", copy.deepcopy(self._sample._design))

        stratum_col, psu_col = self._internal_cols()
        assert stratum_col is not None

        per_col_map = self._require_per_column_mapping(mapping)

        stratum_cols = self._to_cols(getattr(design, "stratum", None))
        psu_cols = self._to_cols(getattr(design, "psu", None))
        allowed = set(stratum_cols) | set(psu_cols)

        bad = [c for c in per_col_map.keys() if c not in allowed]
        if bad:
            raise ValueError(
                f"Combine mapping refers to columns not in design.stratum/psu: {bad}. "
                f"Allowed: {sorted(allowed)}"
            )

        n_strata_before, n_psus_before = self._counts_before(data, stratum_col, psu_col)

        exprs = []
        for col, map_dict in per_col_map.items():
            if col not in data.columns:
                raise ValueError(f"Column {col!r} not found in data.")
            exprs.append(pl.col(col).replace(map_dict).alias(col))

        data = data.with_columns(exprs)

        # Recompute internals
        sep = "__by__"
        null_token = "__Null__"

        if any(c in per_col_map for c in stratum_cols):
            parts = [pl.col(c).cast(pl.Utf8).fill_null(null_token) for c in stratum_cols]
            data = data.with_columns(
                pl.concat_str(parts, separator=sep).cast(pl.Categorical).alias(stratum_col)
            )

        if any(c in per_col_map for c in psu_cols):
            parts = [pl.col(c).cast(pl.Utf8).fill_null(null_token) for c in psu_cols]
            data = data.with_columns(
                pl.concat_str(parts, separator=sep).cast(pl.Utf8).alias(psu_col)
            )

        residual = self._detect_on_df(data, design)
        if residual:
            raise SingletonError.from_singletons(residual, where="singleton.combine")

        n_strata_after, n_psus_after = self._counts_before(data, stratum_col, psu_col)

        result = SingletonResult(
            method=_SingletonHandling.COMBINE,
            detected=tuple(singles),
            applied=tuple(sorted(per_col_map.keys())),
            n_singletons_detected=len(singles),
            n_strata_before=n_strata_before,
            n_strata_after=n_strata_after,
            n_psus_before=n_psus_before,
            n_psus_after=n_psus_after,
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
            method=_SingletonHandling.COLLAPSE,
            singleton_keys=tuple(applied_mapping.keys()),
            stratum_mapping=applied_mapping,
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=_SingletonHandling.COLLAPSE,
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
            method=_SingletonHandling.POOL,
            singleton_keys=tuple(singleton_keys),
            stratum_mapping=stratum_mapping,
            var_stratum_col=_VAR_STRATUM_COL,
            var_psu_col=_VAR_PSU_COL,
            var_exclude_col=_VAR_EXCLUDE_COL,
        )

        result = SingletonResult(
            method=_SingletonHandling.POOL,
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

    def _require_per_column_mapping(self, mapping: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Validate and normalize per-column mapping."""
        if not mapping:
            raise ValueError("mapping cannot be empty")

        normalized = {}
        for col, m in mapping.items():
            if not isinstance(m, dict):
                raise TypeError(
                    "combine() expects {column: {old: new}}. "
                    f"Value for '{col}' was {type(m).__name__}, expected dict."
                )
            normalized[col] = m

        return normalized


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


def _key_part(value: Any) -> str:
    if value is None:
        return _KEY_NULL
    return cast(str, pl.Series([value]).cast(pl.Utf8)[0])


def _spec(method: str, facet: Singleton, result: SingletonResult, **kw: Any) -> SingletonSpec:
    """The resolved decision of a handling result, in the strata's own values."""
    from svy.core.design import SingletonSpec

    if method == "collapse":
        mapping = cast(dict[str, str], result.applied)
        sources = facet._key_values(list(mapping))
        targets = facet._key_values(list(mapping.values()))
        return SingletonSpec.collapse(dict(zip(sources, targets)))
    strata = facet._key_values([s.stratum_key for s in result.detected])
    return getattr(SingletonSpec, method)(strata, **kw)


def _current_singleton_values(sample: Sample) -> list[Any]:
    """The singleton strata of the sample now, as stratum values, from the
    detection run at construction (``Sample._check_for_singletons``)."""
    facet = Singleton(sample, _sync=False)
    stratum_col, _ = facet._internal_cols()
    if not stratum_col or stratum_col not in facet._narrow_data().columns:
        return []
    sample._check_for_singletons()
    keys = sorted(str(d[stratum_col]) for d in (sample._singletons or []))
    return facet._key_values(keys)


def _basis(sample: Sample) -> tuple[tuple[Any, ...], list[str]]:
    """What singleton detection and the variance columns are computed from:
    the rows, the stratum/PSU/SSU columns (and svy's keys of them), the
    variance columns themselves, and the design's rule."""
    design = sample._design
    key = (design.stratum, design.variance_psu, design.ssu, design.singleton)
    idict = getattr(sample, "_internal_design", None) or {}
    cols = [
        SVY_ROW_INDEX,
        *Singleton._to_cols(design.stratum),
        *Singleton._to_cols(design.variance_psu),
        *Singleton._to_cols(design.ssu),
        *(idict.get(k) for k in ("stratum", "psu", "ssu")),
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
    """Detect the singletons again and rebuild the rule's variance columns.

    Skipped when the rows, the strata/PSU/SSU columns and the rule are those of
    the last run: a change elsewhere (a new column, a weight) cannot move them.
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
    """Detect the singletons as at construction, and rebuild the variance
    columns and the handling report from the rule.

    The rule is kept only while it describes the data: the singleton strata
    detected now are exactly the ones it handles, and a collapse target still
    exists. Otherwise it is cleared from the design with one warning; the design
    it replaces goes into ``design_history``.
    """
    ensure = getattr(sample, "_ensure_internal_concat", None)
    if ensure is not None:
        ensure()
    sample._check_for_singletons()
    data = sample._data
    stale = [c for c in _VAR_COLS if c in data.collect_schema().names()]
    if stale:
        sample._data = data.drop(stale)
    design = sample._design
    spec = design.singleton
    if spec is None:
        # A combine() report has no spec behind it and stays; a report derived
        # from a spec that is gone does not.
        prev = getattr(sample, "_singleton_result", None)
        if prev is not None and prev.config is not None:
            sample._singleton_result = None
        return

    facet = Singleton(sample, _sync=False)
    stratum_col, _ = facet._internal_cols()
    reason: str | None = None
    now: list[str] = []
    index: _StrataIndex | None = None
    if not stratum_col or stratum_col not in facet._narrow_data().columns:
        reason = "the design has no stratum"
    else:
        now = sorted(str(d[stratum_col]) for d in (sample._singletons or []))
        index = facet._strata_index()
        handled = facet._value_keys(spec.handled, index)
        if None in handled:
            gone = [v for v, k in zip(spec.handled, handled) if k is None]
            reason = f"handled strata no longer in the data: {', '.join(map(repr, gone))}"
        elif set(handled) != set(now):
            reason = "the singleton strata changed"
        elif spec.method == "collapse":
            targets = facet._value_keys(spec.targets, index)
            gone = [v for v, k in zip(spec.targets, targets) if k is None]
            if gone:
                reason = f"collapse targets no longer in the data: {', '.join(map(repr, gone))}"

    if reason is not None:
        sample._push_design()
        sample._design = design._with_part("singleton", None)
        sample._singleton_result = None
        from svy.core.design_parts import warn_singleton_cleared

        warn_singleton_cleared(
            spec, reason, facet._key_values(now, index) if now else [], sample=sample
        )
        return

    keys = set(handled)
    singles = [s for s in facet.detected() if s.stratum_key in keys]
    if spec.method == "certainty":
        new_data, _, result = facet._apply_certainty(singles)
    elif spec.method == "skip":
        new_data, _, result = facet._apply_skip(singles)
    elif spec.method == "scale":
        new_data, _, result = facet._apply_scale(singles)
    elif spec.method == "center":
        new_data, _, result = facet._apply_center(singles)
    elif spec.method == "pool":
        new_data, _, result = facet._apply_pool(singles, name=cast(str, spec.name))
    else:
        mapping = dict(zip(handled, facet._value_keys([b for _, b in spec.mapping], index)))
        new_data, _, result = facet._apply_collapse(
            singles,
            using=cast(dict[str, str], mapping),
            within=None,
            order_by=None,
            descending=False,
            rstate=None,
        )
    sample._data = new_data
    sample._singleton_result = result
