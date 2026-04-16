# src/svy/categorical/base.py
"""
Categorical data analysis facade on Sample.

All categorical analysis now calls the Rust backend directly:
  - tabulate: rs.tabulate_rs() for proportion/total estimation + Rao-Scott chi-square
  - ttest: rs.ttest_rs() for design-based t-tests
  - ranktest: rs.ranktest_rs() for design-based rank tests
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Callable, Literal, cast

import numpy as np
import polars as pl

from svy.categorical.ranktest import RankTestKSample, RankTestTwoSample
from svy.categorical.table import CellEst, Table, TableStats
from svy.categorical.ttest import TTestByResult, TTestOneGroup, TTestTwoGroups
from svy.core.constants import (
    _INTERNAL_CONCAT_SUFFIX,
)
from svy.core.containers import ChiSquare, FDist
from svy.core.data_prep import prepare_data
from svy.core.enumerations import (
    RankScoreMethod as _RankScoreMethod,
)
from svy.core.enumerations import (
    TableType,
)
from svy.core.enumerations import (
    TableUnits as _TableUnits,
)
from svy.core.types import (
    Number,
    WhereArg,
)
from svy.ui.printing import format_where_clause
from svy.utils.checks import assert_no_missing, drop_missing
from svy.utils.helpers import (
    _scale_weights_for_units,
)


# Rust backend
try:
    from svy_rs import _internal as rs
except ImportError:
    import svy_rs as rs


# For type checkers only
if TYPE_CHECKING:
    from svy.core.sample import Sample

log = logging.getLogger(__name__)


# -------------------------------------------
# Normalization helpers
# -------------------------------------------


def _normalize_units(
    units: Literal["proportion", "percent", "count"] | None,
) -> _TableUnits:
    """
    Normalize user-facing units string to internal TableUnits enum.

    Accepts (case-insensitive):
      - "proportion", "prop"      -> TableUnits.PROPORTION  (default)
      - "percent", "pct", "perc"  -> TableUnits.PERCENT
      - "count"                   -> TableUnits.COUNT
    """
    _MAP = {
        "proportion": _TableUnits.PROPORTION,
        "prop": _TableUnits.PROPORTION,
        "percent": _TableUnits.PERCENT,
        "pct": _TableUnits.PERCENT,
        "perc": _TableUnits.PERCENT,
        "count": _TableUnits.COUNT,
    }
    if units is None:
        return _TableUnits.PROPORTION
    if not isinstance(units, str):
        raise TypeError(
            f"'units' must be a string or None, got {type(units).__name__}. "
            f"Use 'proportion', 'percent', or 'count'."
        )
    result = _MAP.get(units.strip().lower())
    if result is None:
        raise ValueError(f"Unknown units {units!r}. Use 'proportion', 'percent', or 'count'.")
    return result


def _normalize_rank_method(
    method: Literal["kruskal-wallis", "vander-waerden", "median"] | None,
) -> _RankScoreMethod | None:
    """
    Normalize user-facing rank method string to internal RankScoreMethod enum.

    Accepts (case-insensitive):
      - "kruskal-wallis", "kruskal", "kw"  -> RankScoreMethod.KRUSKAL_WALLIS
      - "vander-waerden", "vdw"            -> RankScoreMethod.VANDER_WAERDEN
      - "median"                           -> RankScoreMethod.MEDIAN
    """
    _MAP = {
        "kruskal-wallis": _RankScoreMethod.KRUSKAL_WALLIS,
        "kruskal": _RankScoreMethod.KRUSKAL_WALLIS,
        "kw": _RankScoreMethod.KRUSKAL_WALLIS,
        "vander-waerden": _RankScoreMethod.VANDER_WAERDEN,
        "vdw": _RankScoreMethod.VANDER_WAERDEN,
        "median": _RankScoreMethod.MEDIAN,
    }
    if method is None:
        return None
    if not isinstance(method, str):
        raise TypeError(
            f"'method' must be a string or None, got {type(method).__name__}. "
            f"Use 'kruskal-wallis', 'vander-waerden', or 'median'."
        )
    result = _MAP.get(method.strip().lower())
    if result is None:
        raise ValueError(
            f"Unknown rank method {method!r}. Use 'kruskal-wallis', 'vander-waerden', or 'median'."
        )
    return result


# -------------------------------------------
# Categorical data analysis facade on Sample
# -------------------------------------------


class Categorical:
    def __init__(self, sample: Sample) -> None:
        self._sample = sample

    # ════════════════════════════════════════════════════════════════════════
    # TABULATE (Rust backend)
    # ════════════════════════════════════════════════════════════════════════

    def tabulate(
        self,
        rowvar: str,
        colvar: str | None = None,
        *,
        units: Literal["proportion", "percent", "count"] = "proportion",
        count_total: float | int | None = None,
        alpha: float = 0.05,
        drop_nulls: bool = False,
        use_labels: bool | None = None,
    ) -> Table:
        """
        Create a frequency table or cross-tabulation.

        Parameters
        ----------
        rowvar : str
            Variable for table rows.
        colvar : str | None
            Variable for table columns (creates two-way table). Default None.
        units : str
            Output units: ``'proportion'``, ``'percent'``, or ``'count'``. Default ``'proportion'``.
        count_total : float | int | None
            Total for count scaling. Default None.
        alpha : float
            Significance level for confidence intervals. Default 0.05.
        drop_nulls : bool
            If True, drop rows with missing values. Default False.
        use_labels : bool | None
            If True, display labels instead of codes. None uses Table default.

        Returns
        -------
        Table
            Frequency table with estimates and statistics.
        """
        from scipy.stats import norm as norm_dist
        from scipy.stats import t as t_dist

        _raw = self._sample._data
        local_data: pl.DataFrame = (
            cast(pl.DataFrame, _raw)
            if not isinstance(_raw, pl.LazyFrame)
            else cast(pl.DataFrame, _raw.collect())
        )
        design = self._sample._design

        # required columns
        cols = [rowvar] + ([colvar] if colvar else []) + design.specified_fields()
        cols = self._sample._dedup_preserve_order(cols)
        local_data = local_data.select(cols)

        if drop_nulls:
            valid_data = drop_missing(df=local_data, cols=cols, treat_infinite_as_missing=True)
        else:
            assert_no_missing(df=local_data, subset=cols)
            valid_data = local_data

        # Create concatenated design columns
        _cc, _ = self._sample._create_concatenated_cols_from_lists(
            data=valid_data,
            design=design,
            by=None,
            null_token="__Null__",
            suffix=_INTERNAL_CONCAT_SUFFIX,
            categorical=True,
            drop_original=False,
        )
        concat_data: pl.DataFrame = cast(pl.DataFrame, _cc)

        # Build DataFrame for Rust
        weight_col = design.wgt if design.wgt else "__svy_ones__"
        if not design.wgt:
            concat_data = concat_data.with_columns(pl.lit(1.0).alias(weight_col))

        # Scale weights for units
        _units = _normalize_units(units)
        wgt_arr = concat_data[weight_col].to_numpy().copy()
        wgt_arr = _scale_weights_for_units(wgt_arr, units=_units, count_total=count_total)
        concat_data = concat_data.with_columns(
            pl.Series(name="__svy_scaled_wgt__", values=wgt_arr)
        )

        strata_col = f"stratum{_INTERNAL_CONCAT_SUFFIX}" if design.stratum is not None else None
        psu_col = f"psu{_INTERNAL_CONCAT_SUFFIX}" if design.psu is not None else None
        ssu_col = f"ssu{_INTERNAL_CONCAT_SUFFIX}" if design.ssu is not None else None

        # Cast all design columns to String in a single with_columns call
        _cast_cols = [rowvar] + ([colvar] if colvar else [])
        for _c in [strata_col, psu_col, ssu_col]:
            if _c and _c in concat_data.columns:
                _cast_cols.append(_c)
        concat_data = concat_data.with_columns([pl.col(c).cast(pl.String) for c in _cast_cols])

        # Determine whether to compute totals (weights not normalized)
        compute_totals = abs(float(wgt_arr.sum()) - 1.0) > 1e-6

        # Call Rust backend
        cells_df, stats_df = rs.tabulate_rs(
            concat_data,
            rowvar_col=rowvar,
            weight_col="__svy_scaled_wgt__",
            colvar_col=colvar,
            strata_col=strata_col,
            psu_col=psu_col,
            ssu_col=ssu_col,
            compute_totals=compute_totals,
        )

        # Unpack cells DataFrame into CellEst objects — vectorized CI computation
        df_val = int(cells_df["df"][0]) if cells_df.height > 0 else 1
        t_crit = float(t_dist.ppf(1 - alpha / 2, df_val))
        z_crit = float(norm_dist.ppf(1 - alpha / 2))

        import re

        _NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")

        def _norm_label(v: str) -> str:
            """Normalize '1.0' → '1', '2.0' → '2', etc."""
            s = v.strip()
            if _NUMERIC_RE.fullmatch(s):
                try:
                    x = float(s)
                    return str(int(x)) if x == int(x) else s
                except (ValueError, OverflowError):
                    pass
            return s

        # Extract all columns as numpy arrays — one pass, no iter_rows
        est_arr = cells_df["est"].to_numpy()
        se_arr = cells_df["se"].to_numpy()
        cv_arr = cells_df["cv"].to_numpy()
        rowvar_arr = cells_df["rowvar"].to_list()
        colvar_arr = cells_df["colvar"].to_list()

        if compute_totals:
            lci_arr = est_arr - z_crit * se_arr
            uci_arr = est_arr + z_crit * se_arr
        else:
            # Logit CI — fully vectorized with masked ops
            valid = (est_arr > 0) & (est_arr < 1) & (se_arr > 0)
            # Safe denominators for masked positions
            _p = np.where(valid, est_arr, 0.5)
            scale = np.where(valid, se_arr / (_p * (1.0 - _p)), 0.0)
            logit = np.where(valid, np.log(_p / (1.0 - _p)), 0.0)
            lci_arr = np.where(
                valid,
                1.0 / (1.0 + np.exp(-(logit - t_crit * scale))),
                est_arr,
            )
            uci_arr = np.where(
                valid,
                1.0 / (1.0 + np.exp(-(logit + t_crit * scale))),
                est_arr,
            )

        cell_rows = [
            CellEst(
                rowvar=_norm_label(rowvar_arr[i]),
                colvar=_norm_label(colvar_arr[i]),
                est=float(est_arr[i]),
                se=float(se_arr[i]),
                cv=float(cv_arr[i]),
                lci=float(lci_arr[i]),
                uci=float(uci_arr[i]),
            )
            for i in range(len(est_arr))
        ]

        # Unpack stats DataFrame into TableStats (two-way only)
        tbl_stats = None
        if colvar is not None and stats_df.height > 0:
            cs = stats_df.filter(pl.col("stat") == "chisq").row(0, named=True)
            fs = stats_df.filter(pl.col("stat") == "f").row(0, named=True)

            tbl_stats = TableStats(
                chisq=ChiSquare(
                    df=int(cs["df"]),
                    value=cs["value"],
                    p_value=cs["p_value"],
                ),
                f=FDist(
                    df_num=fs["df"],
                    df_den=fs["df2"],
                    value=fs["value"],
                    p_value=fs["p_value"],
                ),
            )

        # levels for display
        rowvals = concat_data[rowvar].unique().sort().to_list()
        colvals = concat_data[colvar].unique().sort().to_list() if colvar else None

        metadata = getattr(self._sample, "_metadata", None)

        table = Table(
            type=TableType.ONE_WAY if colvar is None else TableType.TWO_WAY,
            rowvar=rowvar,
            colvar=colvar,
            estimates=cell_rows,
            stats=tbl_stats,
            rowvals=rowvals,
            colvals=colvals,
            alpha=alpha,
            metadata=metadata,
        )

        if use_labels is not None:
            table.use_labels = use_labels

        return table

    # ════════════════════════════════════════════════════════════════════════
    # T-TEST: Main entry point (Rust backend)
    # ════════════════════════════════════════════════════════════════════════

    def ttest(
        self,
        y: str,
        *,
        mean_h0: Number = 0,
        group: str | None = None,
        y_pair: str | None = None,
        by: str | None = None,
        where: WhereArg = None,
        alpha: float = 0.05,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        drop_nulls: bool = False,
    ) -> TTestOneGroup | TTestTwoGroups | TTestByResult:
        """
        Perform a design-based t-test.

        Args:
            y: Name of the continuous variable to test.
            mean_h0: Hypothesized mean (one-sample) or difference (two-sample). Default 0.
            group: Grouping variable for two-sample test. If None, performs one-sample test.
            y_pair: Second variable for paired t-test (computes y - y_pair).
            by: Stratification variable for domain estimation. When provided, returns
                a list of test results (one per domain level).
            where: Subpopulation filter. Restricts the analysis to observations
                matching the condition. Accepts Polars expressions, dicts, or lists.
                Default None.
            alpha: Significance level for confidence intervals. Default 0.05.
            alternative: Alternative hypothesis. One of "two-sided", "less", "greater".
                Default "two-sided".
            drop_nulls: If True, drop rows with missing values. Default False.

        Returns:
            TTestOneGroup for one-sample tests, TTestTwoGroups for two-sample tests.
            When `by` is specified, returns a list of test results.
        """
        prep = prepare_data(
            self._sample,
            y=y,
            group=group,
            y_pair=y_pair,
            by=by,
            where=where,
            drop_nulls=drop_nulls,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        y_name = f"svy_{y}_minus_{y_pair}" if y_pair else y

        # Single Rust call — handles by-levels internally
        result_df: pl.DataFrame = rs.ttest_rs(
            prep.df,
            y_col=prep.y_col,
            weight_col=prep.weight_col,
            group_col=group,
            strata_col=prep.strata_col,
            psu_col=prep.psu_col,
            ssu_col=prep.ssu_col,
            singleton_method=prep.singleton_method,
            null_value=float(mean_h0),
            domain_col=prep.domain_col,
            domain_val=prep.domain_val,
            by_col=prep.by_col,
        )

        # Unpack result rows into Python containers
        if by is not None:
            results = []
            for i in range(result_df.height):
                by_level = result_df[prep.by_col][i]
                res = self._unpack_ttest_row(
                    result_df=result_df,
                    row_idx=i,
                    y_name=y_name,
                    group=group,
                    mean_h0=mean_h0,
                    alpha=alpha,
                    alternative=alternative,
                    by=by,
                    by_level=by_level,
                )
                results.append(res)
            # Determine shared metadata for TTestByResult header
            first = results[0] if results else None
            _groups = first.groups if first and isinstance(first, TTestTwoGroups) else None
            _mean_h0 = mean_h0 if group is None else None

            return TTestByResult(
                results,
                by=by,
                y=y_name,
                mean_h0=_mean_h0,
                groups=_groups,
                alpha=alpha,
                where_clause=format_where_clause(where),
            )
        else:
            return self._unpack_ttest_row(
                result_df=result_df,
                row_idx=0,
                y_name=y_name,
                group=group,
                mean_h0=mean_h0,
                alpha=alpha,
                alternative=alternative,
                by=None,
                by_level=None,
            )

    def _unpack_ttest_row(
        self,
        *,
        result_df: pl.DataFrame,
        row_idx: int,
        y_name: str,
        group: str | None,
        mean_h0: Number,
        alpha: float,
        alternative: str,
        by: str | None,
        by_level: object,
    ) -> TTestOneGroup | TTestTwoGroups:
        """Unpack a single row from the ttest result DataFrame into a Python container."""
        from scipy.stats import t as t_dist

        from svy.categorical.ttest import DiffEst, GroupLevels, TtestEst, TTestStats

        row = result_df.row(row_idx, named=True)
        test_type = row["type"]
        t_stat = row["t"]
        df_val = row["df"]
        p_rust = row["p_value"]

        # Recompute p-value for alternative hypothesis if not two-sided
        if alternative != "two-sided":
            p_left = float(t_dist.cdf(t_stat, df_val))
            if alternative == "less":
                p_value = p_left
            else:  # greater
                p_value = 1 - p_left
        else:
            p_value = p_rust

        t_crit = float(t_dist.ppf(1 - alpha / 2, df_val))

        if test_type == "one-sample":
            estimate = row["estimate"]
            diff_value = row["diff"]
            se = row["se"]

            diff_est = DiffEst(
                y=y_name,
                diff=diff_value,
                se=se,
                lci=diff_value - t_crit * se,
                uci=diff_value + t_crit * se,
                by=by,
                by_level=by_level if by else None,
            )
            est_obj = TtestEst(
                by=by,
                by_level=by_level if by else None,
                group=None,
                group_level=None,
                y=y_name,
                y_level=None,
                est=estimate,
                se=se,
                cv=abs(se / estimate) if estimate != 0 else float("nan"),
                lci=estimate - t_crit * se,
                uci=estimate + t_crit * se,
            )
            return TTestOneGroup(
                y=y_name,
                mean_h0=mean_h0,
                alternative=alternative,
                diff=[diff_est],
                estimates=[est_obj],
                stats=TTestStats(df=df_val, t=t_stat, p_value=p_value),
                alpha=alpha,
            )
        else:
            # Two-sample
            diff_value = row["diff"]
            se = row["se"]
            level_0 = row["level_0"]
            level_1 = row["level_1"]
            mean_0 = row["mean_0"]
            mean_1 = row["mean_1"]
            se_0 = row["se_0"]
            se_1 = row["se_1"]

            diff_est = DiffEst(
                y=y_name,
                diff=diff_value,
                se=se,
                lci=diff_value - t_crit * se,
                uci=diff_value + t_crit * se,
                by=by,
                by_level=by_level if by else None,
            )

            def _make_est(grp_level, est_val, se_val):
                return TtestEst(
                    by=by,
                    by_level=by_level if by else None,
                    group=group,
                    group_level=grp_level,
                    y=y_name,
                    y_level=None,
                    est=est_val,
                    se=se_val,
                    cv=abs(se_val / est_val) if est_val != 0 else float("nan"),
                    lci=est_val - t_crit * se_val,
                    uci=est_val + t_crit * se_val,
                )

            return TTestTwoGroups(
                y=y_name,
                groups=GroupLevels(var=group, levels=(level_0, level_1)),
                alternative=alternative,
                diff=[diff_est],
                estimates=[_make_est(level_0, mean_0, se_0), _make_est(level_1, mean_1, se_1)],
                stats=TTestStats(df=df_val, t=t_stat, p_value=p_value),
                alpha=alpha,
            )

    # ════════════════════════════════════════════════════════════════════════
    # RANKTEST: Main entry point (Rust backend)
    # ════════════════════════════════════════════════════════════════════════

    def ranktest(
        self,
        y: str,
        *,
        group: str,
        method: Literal["kruskal-wallis", "vander-waerden", "median"] | None = None,
        score_fn: Callable[[np.ndarray, float], np.ndarray] | None = None,
        by: str | None = None,
        where: WhereArg = None,
        alpha: float = 0.05,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        drop_nulls: bool = False,
    ) -> RankTestTwoSample | RankTestKSample | list[RankTestTwoSample] | list[RankTestKSample]:
        """
        Perform a design-based rank test.

        Implements Lumley & Scott (2013) methodology via the Rust backend.
        Automatically selects two-sample (Wilcoxon) or k-sample (Kruskal-Wallis)
        form based on the number of unique levels in ``group``.
        """
        from svy.errors import MethodError

        # --- Validate method / score_fn ---
        if method is None and score_fn is None:
            raise MethodError.invalid_choice(
                where="ranktest",
                param="method / score_fn",
                got=None,
                allowed=["kruskal-wallis", "vander-waerden", "median"],
                hint="Provide method='kruskal-wallis' or similar.",
            )
        if method is not None and score_fn is not None:
            raise MethodError.not_applicable(
                where="ranktest",
                method="ranktest",
                reason="cannot specify both 'method' and 'score_fn'",
                param="method / score_fn",
                hint="Use one or the other.",
            )
        if score_fn is not None:
            return self._ranktest_custom_score(
                y=y,
                group=group,
                score_fn=score_fn,
                by=by,
                where=where,
                alpha=alpha,
                alternative=alternative,
                drop_nulls=drop_nulls,
            )

        # Normalize and map to Rust string
        _method = _normalize_rank_method(method)
        score_method_str = {
            _RankScoreMethod.KRUSKAL_WALLIS: "wilcoxon",
            _RankScoreMethod.VANDER_WAERDEN: "vanderwaerden",
            _RankScoreMethod.MEDIAN: "median",
        }.get(_method, "wilcoxon")

        prep = prepare_data(
            self._sample,
            y=y,
            group=group,
            by=by,
            where=where,
            drop_nulls=drop_nulls,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )

        # Single Rust call — handles by-levels internally
        result_df: pl.DataFrame = rs.ranktest_rs(
            prep.df,
            y_col=prep.y_col,
            group_col=group,
            weight_col=prep.weight_col,
            strata_col=prep.strata_col,
            psu_col=prep.psu_col,
            ssu_col=prep.ssu_col,
            score_method=score_method_str,
            singleton_method=prep.singleton_method,
            domain_col=prep.domain_col,
            domain_val=prep.domain_val,
            by_col=prep.by_col,
        )

        # Get unique group levels from prepared data for k-sample display
        _group_levels = (
            sorted(str(v) for v in prep.df[group].unique().to_list() if v is not None)
            if group in prep.df.columns
            else []
        )

        # Unpack result rows into Python containers
        if by is not None:
            results = []
            for i in range(result_df.height):
                by_level = result_df[prep.by_col][i]
                res = self._unpack_ranktest_row(
                    result_df=result_df,
                    row_idx=i,
                    y_name=y,
                    group=group,
                    group_levels=_group_levels,
                    alpha=alpha,
                    alternative=alternative,
                    by=by,
                    by_level=by_level,
                )
                results.append(res)
            from svy.categorical.ranktest import RankTestByResult

            first = results[0] if results else None
            _groups = first.groups if first and isinstance(first, RankTestTwoSample) else None
            _method = first.method_name if first else ""
            _by_levels = [result_df[prep.by_col][i] for i in range(result_df.height)]
            return RankTestByResult(
                results,
                by=by,
                y=y,
                group_var=group,
                method_name=_method,
                groups=_groups,
                alpha=alpha,
                where_clause=format_where_clause(where),
                by_levels=_by_levels,
            )
        else:
            return self._unpack_ranktest_row(
                result_df=result_df,
                row_idx=0,
                y_name=y,
                group=group,
                group_levels=_group_levels,
                alpha=alpha,
                alternative=alternative,
                by=None,
                by_level=None,
            )

    def _unpack_ranktest_row(
        self,
        *,
        result_df: pl.DataFrame,
        row_idx: int,
        y_name: str,
        group: str,
        group_levels: list[str],
        alpha: float,
        alternative: str,
        by: str | None,
        by_level: object,
    ) -> RankTestTwoSample | RankTestKSample:
        """Unpack a single row from the ranktest result DataFrame into a Python container."""
        from scipy.stats import t as t_dist

        from svy.categorical.ttest import DiffEst, GroupLevels
        from svy.core.containers import FDist, TDist

        row = result_df.row(row_idx, named=True)
        test_type = row["type"]
        method_name = row["method"]

        _DISPLAY_TWO = {
            "KruskalWallis": "Wilcoxon",
            "vanderWaerden": "van der Waerden",
            "median": "Median",
        }
        _DISPLAY_K = {
            "KruskalWallis": "Kruskal-Wallis",
            "vanderWaerden": "van der Waerden",
            "median": "Median",
        }

        if test_type == "two-sample":
            display_method = _DISPLAY_TWO.get(method_name, method_name)
            delta = row["delta"]
            se = row["se"]
            t_stat = row["t"]
            df_val = row["df"]
            p_rust = row["p_value"]

            if alternative != "two-sided":
                p_left = float(t_dist.cdf(t_stat, df_val))
                p_value = p_left if alternative == "less" else 1 - p_left
            else:
                p_value = p_rust

            t_crit = float(t_dist.ppf(1 - alpha / 2, df_val))

            diff_est = DiffEst(
                y=y_name,
                diff=delta,
                se=se,
                lci=delta - t_crit * se,
                uci=delta + t_crit * se,
                by=by,
                by_level=by_level if by else None,
            )

            # Use actual group levels (sorted); fall back to ("0","1") if unavailable
            g_levels = tuple(group_levels[:2]) if len(group_levels) >= 2 else ("0", "1")

            return RankTestTwoSample(
                y=y_name,
                groups=GroupLevels(var=group, levels=g_levels),
                method_name=display_method,
                alternative=alternative,
                diff=[diff_est],
                estimates=[],
                stats=TDist(df=df_val, value=t_stat, p_value=p_value),
                alpha=alpha,
            )
        else:
            display_method = _DISPLAY_K.get(method_name, method_name)
            ndf = row["ndf"]
            ddf = row["ddf"]
            f_stat = row["f_stat"]
            p_value = row["p_value"]

            return RankTestKSample(
                y=y_name,
                group_var=group,
                group_levels=group_levels,
                method_name=display_method,
                estimates=[],
                stats=FDist(
                    df_num=float(ndf),
                    df_den=ddf,
                    value=f_stat,
                    p_value=p_value,
                ),
                alpha=alpha,
            )

    # ════════════════════════════════════════════════════════════════════════
    # RANKTEST: Custom score function support
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_midranks(y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute estimated population mid-ranks.

        Matches R's:
            ii <- order(y)
            rankhat[ii] <- ave(cumsum(w[ii]) - w[ii]/2, factor(y[ii]))

        Returns (rankhat, N_hat) where N_hat = sum(w).
        """
        n = len(y)
        ii = np.argsort(y, kind="mergesort")
        y_sorted = y[ii]
        w_sorted = w[ii]

        cumw = np.cumsum(w_sorted)
        midrank_sorted = cumw - w_sorted / 2.0

        # Average over ties — vectorized with np.unique + np.bincount
        _, inverse, counts = np.unique(y_sorted, return_inverse=True, return_counts=True)
        group_sums = np.bincount(inverse, weights=midrank_sorted, minlength=len(counts))
        rankhat_sorted = (group_sums / counts)[inverse]

        rankhat = np.empty(n, dtype=np.float64)
        rankhat[ii] = rankhat_sorted
        N_hat = float(np.sum(w))
        return rankhat, N_hat

    def _ranktest_custom_score(
        self,
        y: str,
        group: str,
        score_fn: Callable[[np.ndarray, float], np.ndarray],
        by: str | None,
        where: WhereArg,
        alpha: float,
        alternative: str,
        drop_nulls: bool,
    ) -> RankTestTwoSample | RankTestKSample | list[RankTestTwoSample] | list[RankTestKSample]:
        """
        Rank test with a custom score function.

        Computes weighted mid-ranks and applies score_fn in Python,
        then delegates to rs.ttest_rs on the scores as the y variable.
        """
        from svy.categorical.ttest import DiffEst, GroupLevels
        from svy.core.containers import TDist

        prep = prepare_data(
            self._sample,
            y=y,
            group=group,
            by=by,
            where=where,
            drop_nulls=drop_nulls,
            cast_y_float=True,
            select_columns=True,
            apply_singleton_filter=False,
        )
        y_arr = prep.df[prep.y_col].to_numpy().astype(np.float64)
        w_arr = prep.df[prep.weight_col].to_numpy().astype(np.float64)

        # prepare_data already zeroed weights for non-domain obs,
        # so w_arr > 0 correctly identifies domain-active observations.
        mask = w_arr > 0

        # Compute ranks on active observations
        y_active = y_arr[mask]
        w_active = w_arr[mask]
        rankhat_active, N_hat = self._compute_midranks(y_active, w_active)

        # Apply custom score function
        scores_active = score_fn(rankhat_active, N_hat)

        # Build full-length score column (0 for non-active)
        scores_full = np.zeros(len(y_arr), dtype=np.float64)
        scores_full[mask] = scores_active

        # Add scores as a column and use ttest_rs on scores ~ group
        score_col_name = "__svy_custom_rankscore__"
        df = prep.df.with_columns(pl.Series(name=score_col_name, values=scores_full))

        method_name = getattr(score_fn, "__name__", "custom")

        result_df: pl.DataFrame = rs.ttest_rs(
            df,
            y_col=score_col_name,
            weight_col=prep.weight_col,
            group_col=group,
            strata_col=prep.strata_col,
            psu_col=prep.psu_col,
            ssu_col=prep.ssu_col,
            singleton_method=prep.singleton_method,
            null_value=0.0,
            domain_col=prep.domain_col,
            domain_val=prep.domain_val,
        )

        row = result_df.row(0, named=True)
        test_type = row["type"]

        if test_type == "two-sample":
            from scipy.stats import t as t_dist

            diff_value = row["diff"]
            se = row["se"]
            t_stat = row["t"]
            df_val = row["df"]
            p_rust = row["p_value"]

            if alternative != "two-sided":
                p_left = float(t_dist.cdf(t_stat, df_val))
                p_value = p_left if alternative == "less" else 1 - p_left
            else:
                p_value = p_rust

            t_crit = float(t_dist.ppf(1 - alpha / 2, df_val))

            diff_est = DiffEst(
                y=y,
                diff=diff_value,
                se=se,
                lci=diff_value - t_crit * se,
                uci=diff_value + t_crit * se,
                by=by,
                by_level=None,
            )

            return RankTestTwoSample(
                y=y,
                groups=GroupLevels(var=group, levels=(row["level_0"], row["level_1"])),
                method_name=method_name,
                alternative=alternative,
                diff=[diff_est],
                estimates=[],
                stats=TDist(df=df_val, value=t_stat, p_value=p_value),
                alpha=alpha,
            )
        else:
            from svy.errors import MethodError

            raise MethodError.not_applicable(
                where="ranktest",
                method="ranktest",
                reason="Custom score_fn with more than 2 groups is not yet supported. "
                "Use a built-in method for k-sample tests.",
                param="score_fn",
            )
