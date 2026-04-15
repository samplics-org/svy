# src/svy/estimation/base.py
from __future__ import annotations

import logging
import math

from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

import numpy as np
import polars as pl

from scipy import stats

from svy.core.constants import _BY_SEP, _INTERNAL_CONCAT_SUFFIX
from svy.core.data_prep import PreparedData, prepare_data
from svy.core.design import PopSize
from svy.core.enumerations import EstimationMethod, PopParam, QuantileMethod as _QuantileMethod
from svy.core.types import ExprLike, WhereArg
from svy.errors import DimensionError, MethodError
from svy.estimation.estimate import Estimate, ParamEst
from svy.estimation.taylor import (
    taylor_mean as _taylor_mean,
    taylor_total as _taylor_total,
    taylor_ratio as _taylor_ratio,
    taylor_prop as _taylor_prop,
    taylor_median as _taylor_median,
)
from svy.estimation.replication import (
    replicate_estimate as _replicate_estimate,
    replicate_median as _replicate_median,
)
from svy.ui.printing import format_where_clause
from svy.utils.helpers import _colspec_to_list
from svy.wrangling.rows import _compile_where_to_pl_expr


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from svy.core.sample import Sample


class Estimation:
    def __init__(self, sample: Sample) -> None:
        self._sample = sample
        self._design_cache: dict[str, Any] | None = None
        self._polars_cache: dict[str, Any] | None = None

    def _get_factorized_design(self) -> dict[str, Any]:
        if self._design_cache is not None:
            _d = self._sample._data
            _h = (
                cast(pl.DataFrame, _d.collect()).height
                if isinstance(_d, pl.LazyFrame)
                else cast(pl.DataFrame, _d).height
            )
            if self._design_cache["n_rows"] == _h:
                return self._design_cache
            else:
                self._design_cache = None

        _raw_data = self._sample._data
        local_data: pl.DataFrame = (
            cast(pl.DataFrame, _raw_data.collect())
            if isinstance(_raw_data, pl.LazyFrame)
            else cast(pl.DataFrame, _raw_data)
        )
        design = self._sample._design

        cache: dict[str, Any] = {
            "n_rows": local_data.height,
            "stratum": None,
            "psu": None,
            "ssu": None,
            "wgt": None,
        }

        def _process_component(spec: str | list[str] | tuple[str, ...] | None, name: str):
            if not spec:
                return None, None
            if isinstance(spec, str):
                target_col = spec
            elif isinstance(spec, (list, tuple)) and len(spec) == 1:
                target_col = spec[0]
            else:
                cols = list(spec)
                target_col = f"{name}{_INTERNAL_CONCAT_SUFFIX}"
                if target_col not in local_data.columns:
                    expr = pl.concat_str(
                        [pl.col(c).cast(pl.Utf8).fill_null("__Null__") for c in cols],
                        separator=_BY_SEP,
                    )
                    s_temp = local_data.select(expr.alias(target_col))[target_col]
                else:
                    s_temp = local_data[target_col]

                if s_temp.dtype in (
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                    pl.UInt8,
                    pl.UInt32,
                    pl.UInt64,
                ):
                    return s_temp.to_numpy(), None
                return (
                    s_temp.cast(pl.Categorical).to_physical().to_numpy(),
                    s_temp.unique().to_list(),
                )

            s = local_data[target_col]
            if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt32, pl.UInt64):
                return s.to_numpy(), None
            return (s.cast(pl.Categorical).to_physical().to_numpy(), s.unique().to_list())

        cache["stratum"] = _process_component(design.stratum, "stratum")
        cache["psu"] = _process_component(design.psu, "psu")

        if design.ssu:
            ssu_cols = _colspec_to_list(design.ssu)
            if len(ssu_cols) == 1:
                cache["ssu"] = local_data[ssu_cols[0]].to_numpy()
            else:
                cache["ssu"] = None

        cache["wgt"] = (
            local_data[design.wgt].to_numpy()
            if design.wgt
            else np.ones(local_data.height, dtype=float)
        )

        self._design_cache = cache
        return cache

    def _get_polars_design_info(self) -> dict[str, Any]:
        if self._polars_cache is not None:
            _d2 = self._sample._data
            _h2 = (
                cast(pl.DataFrame, _d2.collect()).height
                if isinstance(_d2, pl.LazyFrame)
                else cast(pl.DataFrame, _d2).height
            )
            if self._polars_cache["n_rows"] == _h2:
                return self._polars_cache
            else:
                self._polars_cache = None

        design = self._sample._design
        _data_raw = self._sample._data
        data: pl.DataFrame = (
            cast(pl.DataFrame, _data_raw.collect())
            if isinstance(_data_raw, pl.LazyFrame)
            else cast(pl.DataFrame, _data_raw)
        )

        singleton_result = getattr(self._sample, "_singleton_result", None)
        config = singleton_result.config if singleton_result else None

        strata_col = None
        psu_col = None
        weight_col = design.wgt

        if config and config.var_stratum_col:
            strata_col = config.var_stratum_col
            psu_col = config.var_psu_col
        else:
            if design.stratum:
                if isinstance(design.stratum, str):
                    strata_col = design.stratum
                elif isinstance(design.stratum, (list, tuple)):
                    if len(design.stratum) == 1:
                        strata_col = design.stratum[0]
                    else:
                        strata_col = f"_strata_{_INTERNAL_CONCAT_SUFFIX}"
                        if strata_col not in data.columns:
                            expr = pl.concat_str(
                                [
                                    pl.col(c).cast(pl.String).fill_null("__Null__")
                                    for c in design.stratum
                                ],
                                separator=_BY_SEP,
                            )
                            data = data.with_columns(expr.alias(strata_col))

            if design.psu:
                if isinstance(design.psu, str):
                    psu_col = design.psu
                elif isinstance(design.psu, (list, tuple)):
                    if len(design.psu) == 1:
                        psu_col = design.psu[0]
                    else:
                        psu_col = f"_psu_{_INTERNAL_CONCAT_SUFFIX}"
                        if psu_col not in data.columns:
                            expr = pl.concat_str(
                                [
                                    pl.col(c).cast(pl.String).fill_null("__Null__")
                                    for c in design.psu
                                ],
                                separator=_BY_SEP,
                            )
                            data = data.with_columns(expr.alias(psu_col))

        casts = []
        if strata_col and data[strata_col].dtype != pl.String:
            casts.append(pl.col(strata_col).cast(pl.String))
        if psu_col and data[psu_col].dtype != pl.String:
            casts.append(pl.col(psu_col).cast(pl.String))

        # Resolve SSU column
        ssu_col = None
        ssu_spec = getattr(design, "ssu", None)
        if ssu_spec:
            if isinstance(ssu_spec, str):
                ssu_col = ssu_spec
            elif isinstance(ssu_spec, (list, tuple)) and len(ssu_spec) == 1:
                ssu_col = ssu_spec[0]
        if ssu_col and ssu_col in data.columns and data[ssu_col].dtype != pl.String:
            casts.append(pl.col(ssu_col).cast(pl.String))

        if casts:
            data = data.with_columns(casts)

        if not weight_col:
            weight_col = "__unit_wgt__"
            if weight_col not in data.columns:
                data = data.with_columns(pl.lit(1.0).alias(weight_col))
        else:
            if data[weight_col].dtype != pl.Float64:
                data = data.with_columns(pl.col(weight_col).cast(pl.Float64))

        # --- FPC column computation ---
        fpc_col = None
        fpc_ssu_col = None
        pop_size = getattr(design, "pop_size", None)

        if pop_size is not None:
            data, fpc_col, fpc_ssu_col = self._compute_fpc_columns(
                data=data,
                pop_size=pop_size,
                strata_col=strata_col,
                psu_col=psu_col,
                ssu_col=ssu_col,
            )

        _fin = self._sample._data
        _fin_h = (
            cast(pl.DataFrame, _fin.collect()).height
            if isinstance(_fin, pl.LazyFrame)
            else cast(pl.DataFrame, _fin).height
        )
        self._polars_cache = {
            "n_rows": _fin_h,
            "data": data,
            "strata_col": strata_col,
            "psu_col": psu_col,
            "ssu_col": ssu_col,
            "weight_col": weight_col,
            "fpc_col": fpc_col,
            "fpc_ssu_col": fpc_ssu_col,
            "singleton_config": config,
        }
        return self._polars_cache

    # ----------------------------------------------------------------
    # FPC — delegated to _fpc.py
    # ----------------------------------------------------------------

    def _compute_fpc_columns(self, data, pop_size, strata_col, psu_col, ssu_col=None):
        from svy.estimation._fpc import compute_fpc_columns

        return compute_fpc_columns(self, data, pop_size, strata_col, psu_col, ssu_col)

    # ----------------------------------------------------------------
    # Internal Helpers
    # ----------------------------------------------------------------

    def _ensure_float64(self, data: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
        casts = [
            pl.col(c).cast(pl.Float64)
            for c in cols
            if c in data.columns and data[c].dtype != pl.Float64
        ]
        return data.with_columns(casts) if casts else data

    def _coerce_y_for_prop(self, data: pl.DataFrame, y: str) -> pl.DataFrame:
        if y not in data.columns:
            return data
        dtype = data[y].dtype
        if dtype in (pl.Float32, pl.Float64):
            raise TypeError(
                f"prop() does not support float column '{y}'. "
                f"Use a String, Categorical, Enum, Boolean, or integer column."
            )
        if dtype.is_integer() and dtype != pl.Int64:
            return data.with_columns(pl.col(y).cast(pl.Int64))
        return data

    def _get_enum_value(self, config, attr="method") -> str:
        if not hasattr(config, attr):
            return ""
        val = getattr(config, attr)
        return str(val.value) if hasattr(val, "value") else str(val)

    def _adjust_variance_for_singletons(
        self, result_df: pl.DataFrame, param: PopParam = PopParam.TOTAL
    ) -> pl.DataFrame:
        return self._apply_scale_adjustment(result_df, result_df, param=param)

    def _get_center_method(self) -> str | None:
        cache = self._get_polars_design_info()
        config = cache.get("singleton_config")
        if config:
            method_str = self._get_enum_value(config, "method").lower()
            if method_str in ("center", "adjust"):
                return "center"
        return None

    def _should_run_double_pass(self) -> bool:
        cache = self._get_polars_design_info()
        config = cache.get("singleton_config")
        if config:
            return self._get_enum_value(config, "method").lower() == "scale"
        return False

    def _apply_scale_adjustment(
        self, full_df: pl.DataFrame, filtered_df: pl.DataFrame, param: PopParam = PopParam.TOTAL
    ) -> pl.DataFrame:
        cache = self._get_polars_design_info()
        config = cache.get("singleton_config")

        if not config:
            return filtered_df

        method_str = self._get_enum_value(config, "method").lower()
        if method_str != "scale":
            return filtered_df

        f = config.singleton_fraction
        if f is None or f >= 1.0:
            return filtered_df

        if param == PopParam.TOTAL:
            inflation_factor = 1.0 / (1.0 - f)
        else:
            inflation_factor = 1.0 - f

        sqrt_factor = math.sqrt(inflation_factor)

        if full_df is filtered_df:
            merged = filtered_df
        else:
            std_cols = {"y", "est", "se", "var", "df", "n", "deff", "level"}
            extra_cols = [c for c in filtered_df.columns if c not in std_cols]
            by_col_name = extra_cols[0] if extra_cols else None

            join_on = ["y"]
            if by_col_name:
                join_on.append(by_col_name)
            if "level" in filtered_df.columns:
                join_on.append("level")

            full_subset = full_df.select(join_on + ["est"])

            if "est" in filtered_df.columns:
                merged = filtered_df.drop("est").join(full_subset, on=join_on, how="left")
            else:
                merged = filtered_df.join(full_subset, on=join_on, how="left")

        return merged.with_columns(
            (pl.col("var") * inflation_factor).alias("var"),
            (pl.col("se") * sqrt_factor).alias("se"),
        )

    @staticmethod
    def _normalize_ci_method(method: str) -> str:
        """Normalize CI method name to canonical form.

        Canonical names: ``"logit"``, ``"beta"``, ``"korn-graubard"``, ``"wilson"``.

        Accepted aliases (case-insensitive):
            ``"clopper-pearson"`` / ``"kg"`` → ``"korn-graubard"``
            ``"score"`` → ``"wilson"``
        """
        m = method.lower().replace("_", "-")
        aliases = {
            "clopper-pearson": "korn-graubard",
            "kg": "korn-graubard",
            "score": "wilson",
        }
        return aliases.get(m, m)

    def _compute_prop_ci(
        self,
        p: float,
        se: float,
        alpha: float,
        df: int,
        n: int,
        method: str,
    ) -> tuple[float, float]:
        """Compute confidence interval for a proportion.

        Parameters
        ----------
        p : float
            Estimated proportion.
        se : float
            Standard error of the proportion.
        alpha : float
            Significance level (e.g. 0.05 for 95% CI).
        df : int
            Degrees of freedom (PSUs - strata).
        n : int
            Nominal sample size (denominator).
        method : str
            One of ``"logit"``, ``"beta"``, ``"korn-graubard"``.

            ``"logit"``
                Wald-type interval on the logit scale, back-transformed.
                Default in svy and matches Stata's ``svy: prop``.

            ``"beta"``
                Korn-Graubard CI matching R's
                ``survey::svyciprop(method="beta")``.  Uses df-adjusted
                effective sample size (no truncation) and the incomplete
                Beta function (Clopper-Pearson formulation).

            ``"korn-graubard"``
                Korn-Graubard CI matching the NCHS SAS reference macro.
                Adds truncation of effective sample size at *n* and
                explicit handling of *p* = 0 or *p* = 1, as required by
                the NCHS Data Presentation Standards for Proportions
                (Parker et al. 2017).

        References
        ----------
        Korn E.L., Graubard B.I. (1998).  Confidence Intervals For
        Proportions With Small Expected Number of Positive Counts
        Estimated From Survey Data.  *Survey Methodology* 24(2):193-201.

        Parker J.D. et al. (2017).  National Center for Health Statistics
        Data Presentation Standards for Proportions.
        *Vital Health Stat* 2(175).
        """
        method = self._normalize_ci_method(method)

        if method == "logit":
            if p <= 0 or p >= 1:
                return (p, p)
            t_crit = stats.t.ppf(1 - alpha / 2, df) if df > 0 else 1.96
            scale = se / (p * (1.0 - p)) if se > 0 else 0
            logit_p = math.log(p / (1 - p))
            lci = 1.0 / (1.0 + math.exp(-(logit_p - t_crit * scale)))
            uci = 1.0 / (1.0 + math.exp(-(logit_p + t_crit * scale)))
            return (lci, uci)

        elif method == "beta":
            # ── R-compatible Korn-Graubard CI ──
            # Matches R survey::svyciprop(method="beta") exactly.
            # Reference: Korn & Graubard (1998), eqs 2.1, 2.2, 1.2.
            from scipy.stats import beta as beta_dist

            if p <= 0 or p >= 1 or se <= 0:
                return (p, p)

            # Eq 2.1: effective sample size
            n_eff = (p * (1 - p)) / (se**2)

            # Eq 2.2: df-adjustment (no truncation, matching R)
            if df > 0 and n > 1:
                t_n = stats.t.ppf(alpha / 2, n - 1)
                t_df = stats.t.ppf(alpha / 2, df)
                n_eff = n_eff * (t_n / t_df) ** 2

            # Clopper-Pearson via Beta distribution (asymmetric +1)
            x = n_eff * p
            lci = beta_dist.ppf(alpha / 2, x, n_eff - x + 1)
            uci = beta_dist.ppf(1 - alpha / 2, x + 1, n_eff - x)
            return (lci, uci)

        elif method == "korn-graubard":
            # ── NCHS SAS macro-compatible Korn-Graubard CI ──
            # Matches KG_macro.sas from CDC/NCHS.
            # Adds: truncation of n_eff at n, p=0/p=1 handling.
            # Reference: Korn & Graubard (1998); Parker et al. (2017).
            from scipy.stats import f as f_dist

            if p <= 0 or p >= 1:
                # Special handling: fall back to nominal sample size,
                # then apply df-adjustment (matching NCHS SAS macro).
                n_eff = float(n)
                if df > 0 and n > 1:
                    t_n = stats.t.ppf(1 - alpha / 2, n - 1)
                    t_df = stats.t.ppf(1 - alpha / 2, df)
                    t_adj = (t_n / t_df) ** 2
                    n_eff_df = min(n, n_eff * t_adj)
                else:
                    n_eff_df = n_eff
                x = p * n_eff_df
                if p == 0:
                    lci = 0.0
                    if n_eff_df > 0:
                        v3 = 2 * (x + 1)
                        v4 = 2 * (n_eff_df - x)
                        if v3 > 0 and v4 > 0:
                            f_upper = f_dist.ppf(1 - alpha / 2, v3, v4)
                            uci = (v3 * f_upper) / (v4 + v3 * f_upper)
                        else:
                            uci = 1.0
                    else:
                        uci = 1.0
                    return (lci, uci)
                else:  # p == 1
                    uci = 1.0
                    if n_eff_df > 0:
                        v1 = 2 * x
                        v2 = 2 * (n_eff_df - x + 1)
                        if v1 > 0 and v2 > 0:
                            f_lower = f_dist.ppf(alpha / 2, v1, v2)
                            lci = (v1 * f_lower) / (v2 + v1 * f_lower)
                        else:
                            lci = 0.0
                    else:
                        lci = 0.0
                    return (lci, uci)

            if se <= 0:
                return (p, p)

            # Eq 2.1: effective sample size
            n_eff = (p * (1 - p)) / (se**2)

            # Eq 2.2: df-adjustment with NCHS truncation
            if df > 0 and n > 1:
                t_n = stats.t.ppf(1 - alpha / 2, n - 1)
                t_df = stats.t.ppf(1 - alpha / 2, df)
                t_adj = (t_n / t_df) ** 2
                n_eff_df = min(n, n_eff * t_adj)
            else:
                n_eff_df = min(n, n_eff)

            # Eqs 4-9: F-distribution formulation
            x = n_eff_df * p
            v1 = 2 * x
            v2 = 2 * (n_eff_df - x + 1)
            v3 = 2 * (x + 1)
            v4 = 2 * (n_eff_df - x)

            if v1 > 0 and v2 > 0:
                f_lower = f_dist.ppf(alpha / 2, v1, v2)
                lci = (v1 * f_lower) / (v2 + v1 * f_lower)
            else:
                lci = 0.0

            if v3 > 0 and v4 > 0:
                f_upper = f_dist.ppf(1 - alpha / 2, v3, v4)
                uci = (v3 * f_upper) / (v4 + v3 * f_upper)
            else:
                uci = 1.0

            return (lci, uci)

        elif method == "wilson":
            # ── Wilson score interval ──
            # Uses the score-test inversion with effective sample size.
            # Replaces n with n_eff = p(1-p)/se² and uses t-quantile for df.
            # Reference: Wilson (1927); Franco et al. (2019, JSSAM).
            if p <= 0 or p >= 1 or se <= 0:
                return (p, p)

            # Effective sample size
            n_eff = (p * (1 - p)) / (se**2)

            # df-adjustment (same as beta method)
            if df > 0 and n > 1:
                t_n = stats.t.ppf(1 - alpha / 2, n - 1)
                t_df = stats.t.ppf(1 - alpha / 2, df)
                n_eff = n_eff * (t_n / t_df) ** 2

            # Wilson score interval: roots of the score-test quadratic
            z = stats.t.ppf(1 - alpha / 2, df) if df > 0 else 1.96
            z2 = z * z
            denom = 1 + z2 / n_eff
            center = (p + z2 / (2 * n_eff)) / denom
            half_width = (z / denom) * math.sqrt(p * (1 - p) / n_eff + z2 / (4 * n_eff * n_eff))
            lci = max(0.0, center - half_width)
            uci = min(1.0, center + half_width)
            return (lci, uci)

        else:
            raise ValueError(f"Unknown CI method: {method!r}")

    def _polars_result_to_param_est(
        self,
        result_df: pl.DataFrame,
        y_name: str,
        param: PopParam,
        alpha: float,
        deff: bool,
        by_col: str | None,
        as_factor: bool,
        x_name: str | None = None,
        ci_method: str = "logit",
    ) -> list[ParamEst]:
        est_list = []
        for row in result_df.iter_rows(named=True):
            est = row["est"]
            se = row["se"]
            df_val = row["df"]
            n_val = row.get("n", 0)
            cv = se / est if est != 0 else float("inf")
            t_crit = stats.t.ppf(1 - alpha / 2, df_val) if df_val > 0 else 1.96
            is_prop = (param == PopParam.PROP) or as_factor
            if is_prop:
                lci, uci = self._compute_prop_ci(
                    p=est,
                    se=se,
                    alpha=alpha,
                    df=df_val,
                    n=n_val,
                    method=ci_method,
                )
            else:
                lci = est - t_crit * se
                uci = est + t_crit * se
            deff_val = row.get("deff") if deff else None
            by_level = None
            if by_col and by_col in row:
                by_level = (row[by_col],)
            y_level = None
            if as_factor and "level" in row:
                level_val = row["level"]
                try:
                    y_level = int(level_val)
                except (ValueError, TypeError):
                    y_level = level_val
            p = ParamEst(
                y=y_name,
                est=est,
                se=se,
                cv=cv,
                lci=lci,
                uci=uci,
                deff=deff_val,
                by=(by_col,) if by_col else None,
                by_level=by_level,
                y_level=y_level,
                x=x_name,
            )
            est_list.append(p)
        return est_list

    def _median_result_to_param_est(
        self,
        result_df,
        y_name,
        alpha,
        by_col,
        data,
        weight_col,
    ) -> list[ParamEst]:
        est_list = []
        for row in result_df.iter_rows(named=True):
            est = row["est"]
            se_p = row["se"]
            df_val = row["df"]
            t_crit = stats.t.ppf(1 - alpha / 2, df_val) if df_val > 0 else 1.96
            p_lower = max(0.0, 0.5 - t_crit * se_p)
            p_upper = min(1.0, 0.5 + t_crit * se_p)
            if by_col and by_col in row:
                domain_val = row[by_col]
                domain_data = data.filter(pl.col(by_col) == domain_val)
            else:
                domain_data = data
            lci = self._invert_cdf(domain_data, y_name, weight_col, p_lower)
            uci = self._invert_cdf(domain_data, y_name, weight_col, p_upper)
            se_q = (uci - lci) / (2 * t_crit) if t_crit > 0 else se_p
            cv = se_q / est if est != 0 else float("inf")
            by_level = None
            if by_col and by_col in row:
                by_level = (row[by_col],)
            p = ParamEst(
                y=y_name,
                est=est,
                se=se_q,
                cv=cv,
                lci=lci,
                uci=uci,
                deff=None,
                by=(by_col,) if by_col else None,
                by_level=by_level,
                y_level=None,
                x=None,
            )
            est_list.append(p)
        return est_list

    def _invert_cdf(self, data, y_col, weight_col, p):
        df = data.select([y_col, weight_col]).drop_nulls()
        if df.height == 0:
            return float("nan")
        df = df.sort(y_col)
        y_vals = df[y_col].to_numpy()
        w_vals = df[weight_col].to_numpy()
        cumsum = np.cumsum(w_vals)
        total = cumsum[-1]
        if total <= 0:
            return float("nan")
        cdf = cumsum / total
        if p <= 0:
            return float(y_vals[0])
        if p >= 1:
            return float(y_vals[-1])
        idx = np.searchsorted(cdf, p)
        if idx == 0:
            return float(y_vals[0])
        if idx >= len(y_vals):
            return float(y_vals[-1])
        p_low = cdf[idx - 1]
        p_high = cdf[idx]
        y_low = y_vals[idx - 1]
        y_high = y_vals[idx]
        if p_high == p_low:
            return float(y_high)
        frac = (p - p_low) / (p_high - p_low)
        return float(y_low + frac * (y_high - y_low))

    def _replicate_median_result_to_param_est(self, result_df, y_name, alpha, by_col):
        est_list = []
        for row in result_df.iter_rows(named=True):
            est = row["est"]
            se = row["se"]
            df_val = row["df"]
            cv = se / est if est != 0 else float("inf")
            t_crit = stats.t.ppf(1 - alpha / 2, df_val) if df_val > 0 else 1.96
            lci = est - t_crit * se
            uci = est + t_crit * se
            by_level = None
            if by_col and by_col in row:
                by_level = (row[by_col],)
            p = ParamEst(
                y=y_name,
                est=est,
                se=se,
                cv=cv,
                lci=lci,
                uci=uci,
                deff=None,
                by=(by_col,) if by_col else None,
                by_level=by_level,
                y_level=None,
                x=None,
            )
            est_list.append(p)
        return est_list

    def _build_estimate_result_light(
        self,
        est_list,
        est_cov,
        param,
        alpha,
        by_cols,
        as_factor,
        method: EstimationMethod = EstimationMethod.TAYLOR,
        rust_df=None,
    ) -> Estimate:
        metadata = getattr(self._sample, "_metadata", None)
        estimate = Estimate(param, alpha=alpha, metadata=metadata)
        estimate.method = method
        estimate.covariance = est_cov
        estimate.as_factor = as_factor
        if by_cols and len(by_cols) > 0:
            by_tuple = tuple(by_cols)
            final_ests = []
            for p in est_list:
                by_level = p.by_level
                if by_level and len(by_cols) > 1 and len(by_level) == 1:
                    parts = str(by_level[0]).split("__by__", maxsplit=len(by_cols) - 1)
                    by_level = tuple(parts) if len(parts) == len(by_cols) else by_level
                new_p = ParamEst(
                    y=p.y,
                    est=p.est,
                    se=p.se,
                    cv=p.cv,
                    lci=p.lci,
                    uci=p.uci,
                    deff=p.deff,
                    by=by_tuple,
                    by_level=by_level,
                    y_level=p.y_level,
                    x=p.x,
                    x_level=p.x_level,
                )
                final_ests.append(new_p)
            estimate.estimates = final_ests
        else:
            estimate.estimates = est_list
        d_cache = self._get_factorized_design()
        if d_cache["stratum"] is not None:
            strata_info = d_cache["stratum"]
            if isinstance(strata_info, tuple) and strata_info[1] is not None:
                estimate.strata = strata_info[1]
                estimate.n_strata = len(strata_info[1])
            else:
                arr = strata_info[0] if isinstance(strata_info, tuple) else strata_info
                estimate.strata = np.unique(arr).tolist()
                estimate.n_strata = len(estimate.strata)
        if d_cache["psu"] is not None:
            psu_info = d_cache["psu"]
            if isinstance(psu_info, tuple) and psu_info[1] is not None:
                estimate.n_psus = len(psu_info[1])
            else:
                arr = psu_info[0] if isinstance(psu_info, tuple) else psu_info
                estimate.n_psus = len(np.unique(arr))
        elif d_cache["wgt"] is not None:
            estimate.n_psus = len(d_cache["wgt"])
        estimate.degrees_freedom = (
            rust_df
            if rust_df is not None
            else max(0, (estimate.n_psus or 0) - (estimate.n_strata or 0))
        )
        return estimate

    @staticmethod
    def _normalize_method(method: str | None) -> Literal["taylor", "replication"] | None:
        """
        Normalize user-facing method string to canonical form.

        Accepts case-insensitive variants:
          - Taylor: "taylor", "Taylor", "TAYLOR", "linearization", "lin"
          - Replication: "replication", "replicate", "rep",
            "bootstrap", "brr", "jackknife", "jk", "sdr"
          - None: auto-detect

        Returns "taylor", "replication", or None.
        """
        if method is None:
            return None
        if not isinstance(method, str):
            raise TypeError(
                f"'method' must be a string or None, got {type(method).__name__}. "
                f"Use method='taylor' or method='replication'."
            )
        m = method.strip().lower()
        if m in ("taylor", "linearization", "lin"):
            return "taylor"
        if m in ("replication", "replicate", "rep", "bootstrap", "brr", "jackknife", "jk", "sdr"):
            return "replication"
        raise ValueError(f"Unknown estimation method {method!r}. Use 'taylor' or 'replication'.")

    @staticmethod
    def _normalize_q_method(
        q_method: Literal["higher", "lower", "nearest", "linear", "middle"] | None,
    ) -> _QuantileMethod:
        """
        Normalize user-facing q_method string to internal QuantileMethod enum.

        Accepts (case-insensitive):
          - "higher"  → QuantileMethod.HIGHER  (default)
          - "lower"   → QuantileMethod.LOWER
          - "nearest" → QuantileMethod.NEAREST
          - "linear"  → QuantileMethod.LINEAR
          - "middle"  → QuantileMethod.MIDDLE
        """
        _MAP = {
            "higher": _QuantileMethod.HIGHER,
            "lower": _QuantileMethod.LOWER,
            "nearest": _QuantileMethod.NEAREST,
            "linear": _QuantileMethod.LINEAR,
            "middle": _QuantileMethod.MIDDLE,
        }
        if q_method is None:
            return _QuantileMethod.HIGHER
        if not isinstance(q_method, str):
            raise TypeError(
                f"'q_method' must be a string or None, got {type(q_method).__name__}. "
                f"Use one of: {tuple(_MAP)}."
            )
        result = _MAP.get(q_method.strip().lower())
        if result is None:
            raise ValueError(f"Unknown quantile method {q_method!r}. Use one of: {tuple(_MAP)}.")
        return result

    def _resolve_method(self, method: str | None) -> EstimationMethod:
        normalized = self._normalize_method(method)

        if normalized == "replication":
            if self._sample._design.rep_wgts is None:
                raise ValueError(
                    "Replication requires rep_wgts in the design. "
                    "Create replicate weights first or use method='taylor'."
                )
            return self._sample._design.rep_wgts.method

        return EstimationMethod.TAYLOR

    # ----------------------------------------------------------------
    # Domain Estimation Helpers
    # ----------------------------------------------------------------
    def _compile_where_expr(self, where: WhereArg) -> pl.Expr:
        """Compile where clause to Polars expression using wrangling helper."""

        result = _compile_where_to_pl_expr(where)
        return cast(pl.Expr, result)

    # ----------------------------------------------------------------
    # Unified Public APIs
    # ----------------------------------------------------------------

    def _empty_estimate(
        self, param: PopParam, alpha: float, by_cols: list[str], method: EstimationMethod
    ) -> Estimate:
        """Return an empty Estimate for cases like zero-weight domains."""
        metadata = getattr(self._sample, "_metadata", None)
        est = Estimate(param, alpha=alpha, metadata=metadata)
        est.method = method
        est.estimates = []
        est.covariance = np.array([])
        return est

    def mean(
        self,
        y: str,
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        deff: bool = False,
        fay_coef: float = 0.0,
        as_factor: bool = False,
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate:
        """Estimate population mean with standard errors.

        Parameters
        ----------
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design (Taylor when strata/PSU
            are available, replication otherwise).
        """
        target_method = self._resolve_method(method)
        prep = prepare_data(
            self._sample,
            y=y,
            by=by,
            where=where,
            drop_nulls=drop_nulls,
            cast_y_float=True,
            apply_singleton_filter=True,
            select_columns=True,
        )

        try:
            if target_method == EstimationMethod.TAYLOR:
                result = _taylor_mean(
                    self,
                    prep=prep,
                    y=y,
                    deff=deff,
                    alpha=alpha,
                    as_factor=as_factor,
                    param=PopParam.MEAN,
                )
            else:
                result = _replicate_estimate(
                    self,
                    prep=prep,
                    method=target_method,
                    param=PopParam.MEAN,
                    y=y,
                    fay_coef=fay_coef,
                    as_factor=as_factor,
                    variance_center=variance_center,
                    alpha=alpha,
                )
        except RuntimeError as e:
            if "weights is zero" in str(e).lower() or "sum of weights" in str(e).lower():
                result = self._empty_estimate(
                    PopParam.MEAN, alpha, _colspec_to_list(by), target_method
                )
            else:
                raise

        if where is not None:
            result.where_clause = format_where_clause(where)
        return result

    def total(
        self,
        y: str,
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        deff: bool = False,
        fay_coef: float = 0.0,
        as_factor: bool = False,
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate:
        """Estimate population total with standard errors.

        Parameters
        ----------
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design.
        """
        target_method = self._resolve_method(method)
        prep = prepare_data(
            self._sample,
            y=y,
            by=by,
            where=where,
            drop_nulls=drop_nulls,
            cast_y_float=True,
            apply_singleton_filter=True,
            select_columns=True,
        )

        try:
            if target_method == EstimationMethod.TAYLOR:
                result = _taylor_total(
                    self,
                    prep=prep,
                    y=y,
                    deff=deff,
                    alpha=alpha,
                    as_factor=as_factor,
                )
            else:
                result = _replicate_estimate(
                    self,
                    prep=prep,
                    method=target_method,
                    param=PopParam.TOTAL,
                    y=y,
                    fay_coef=fay_coef,
                    as_factor=as_factor,
                    variance_center=variance_center,
                    alpha=alpha,
                )
        except RuntimeError as e:
            if "weights is zero" in str(e).lower() or "sum of weights" in str(e).lower():
                result = self._empty_estimate(
                    PopParam.TOTAL, alpha, _colspec_to_list(by), target_method
                )
            else:
                raise

        if where is not None:
            result.where_clause = format_where_clause(where)
        return result

    def prop(
        self,
        y: str,
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        ci_method: Literal["logit", "beta", "korn-graubard", "wilson"] = "logit",
        deff: bool = False,
        fay_coef: float = 0.0,
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate:
        """Estimate population proportion with standard errors.

        Parameters
        ----------
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design.
        """
        target_method = self._resolve_method(method)
        prep = prepare_data(
            self._sample,
            y=y,
            by=by,
            where=where,
            drop_nulls=drop_nulls,
            cast_y_float=False,
            apply_singleton_filter=True,
            select_columns=True,
        )

        try:
            if target_method == EstimationMethod.TAYLOR:
                result = _taylor_prop(
                    self,
                    prep=prep,
                    y=y,
                    deff=deff,
                    alpha=alpha,
                    ci_method=ci_method,
                )
            else:
                result = _replicate_estimate(
                    self,
                    prep=prep,
                    method=target_method,
                    param=PopParam.PROP,
                    y=y,
                    fay_coef=fay_coef,
                    as_factor=True,
                    variance_center=variance_center,
                    alpha=alpha,
                    ci_method=ci_method,
                )
        except RuntimeError as e:
            if "weights is zero" in str(e).lower() or "sum of weights" in str(e).lower():
                result = self._empty_estimate(
                    PopParam.PROP, alpha, _colspec_to_list(by), target_method
                )
            else:
                raise

        if where is not None:
            result.where_clause = format_where_clause(where)
        return result

    def ratio(
        self,
        y: str,
        x: str,
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        deff: bool = False,
        fay_coef: float = 0.0,
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate:
        """Estimate population ratio (y/x) with standard errors.

        Parameters
        ----------
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design.
        """
        target_method = self._resolve_method(method)
        prep = prepare_data(
            self._sample,
            y=y,
            x=x,
            by=by,
            where=where,
            drop_nulls=drop_nulls,
            cast_y_float=True,
            apply_singleton_filter=True,
            select_columns=True,
        )

        try:
            if target_method == EstimationMethod.TAYLOR:
                result = _taylor_ratio(
                    self,
                    prep=prep,
                    y=y,
                    x=x,
                    deff=deff,
                    alpha=alpha,
                )
            else:
                result = _replicate_estimate(
                    self,
                    prep=prep,
                    method=target_method,
                    param=PopParam.RATIO,
                    y=y,
                    x=x,
                    fay_coef=fay_coef,
                    variance_center=variance_center,
                    alpha=alpha,
                )
        except RuntimeError as e:
            if "weights is zero" in str(e).lower() or "sum of weights" in str(e).lower():
                result = self._empty_estimate(
                    PopParam.RATIO, alpha, _colspec_to_list(by), target_method
                )
            else:
                raise

        if where is not None:
            result.where_clause = format_where_clause(where)
        return result

    def median(
        self,
        y: str,
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        fay_coef: float = 0.0,
        q_method: Literal["higher", "lower", "nearest", "linear", "middle"] = "higher",
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate:
        """Estimate population median with standard errors.

        Parameters
        ----------
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design.
        """
        target_method = self._resolve_method(method)
        resolved_q_method = self._normalize_q_method(q_method)
        prep = prepare_data(
            self._sample,
            y=y,
            by=by,
            where=where,
            drop_nulls=drop_nulls,
            cast_y_float=True,
            apply_singleton_filter=True,
            select_columns=True,
        )

        try:
            if target_method == EstimationMethod.TAYLOR:
                result = _taylor_median(
                    self,
                    prep=prep,
                    y=y,
                    q_method=resolved_q_method,
                    alpha=alpha,
                )
            else:
                result = _replicate_median(
                    self,
                    prep=prep,
                    y=y,
                    method=target_method,
                    fay_coef=fay_coef,
                    q_method=resolved_q_method,
                    variance_center=variance_center,
                    alpha=alpha,
                )
        except RuntimeError as e:
            if "weights is zero" in str(e).lower() or "sum of weights" in str(e).lower():
                result = self._empty_estimate(
                    PopParam.MEDIAN, alpha, _colspec_to_list(by), target_method
                )
            else:
                raise

        if where is not None:
            result.where_clause = format_where_clause(where)
        return result
