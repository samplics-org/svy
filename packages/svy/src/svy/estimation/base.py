# src/svy/estimation/base.py
from __future__ import annotations

import logging
import math

from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

import numpy as np
import polars as pl
import svy_rs as rs

from svy.core.constants import _BY_SEP, _INTERNAL_CONCAT_SUFFIX
from svy.core.data_prep import prepare_data
from svy.core.enumerations import PopParam
from svy.core.enumerations import QuantileMethod as _QuantileMethod
from svy.core.repwgts import RepWgts
from svy.core.types import WhereArg
from svy.core.warnings import WarnCode
from svy.errors import DimensionError, MethodError
from svy.errors.singleton_errors import SingletonError
from svy.estimation.estimate import Estimate, EstimateList, ParamEst
from svy.estimation.replication import (
    replicate_estimate as _replicate_estimate,
)
from svy.estimation.replication import (
    replicate_median as _replicate_median,
)
from svy.estimation.replication import (
    replicate_quantile as _replicate_quantile,
)
from svy.estimation.taylor import (
    taylor_mean as _taylor_mean,
)
from svy.estimation.taylor import (
    taylor_mean_multi as _taylor_mean_multi,
)
from svy.estimation.taylor import (
    taylor_median as _taylor_median,
)
from svy.estimation.taylor import (
    taylor_median_multi as _taylor_median_multi,
)
from svy.estimation.taylor import (
    taylor_prop as _taylor_prop,
)
from svy.estimation.taylor import (
    taylor_prop_multi as _taylor_prop_multi,
)
from svy.estimation.taylor import (
    taylor_quantile as _taylor_quantile,
)
from svy.estimation.taylor import (
    taylor_ratio as _taylor_ratio,
)
from svy.estimation.taylor import (
    taylor_ratio_multi as _taylor_ratio_multi,
)
from svy.estimation.taylor import (
    taylor_total as _taylor_total,
)
from svy.estimation.taylor import (
    taylor_total_multi as _taylor_total_multi,
)
from svy.ui.printing import format_where_clause
from svy.utils.helpers import _colspec_to_list
from svy.wrangling.rows import _compile_where_to_pl_expr

from .association import guard_pandas_method as _guard_pandas_method
from .association import normalize_kind as _normalize_assoc_kind
from .association import parse_cols as _parse_assoc_cols
from .association import replicate_assoc as _replicate_assoc
from .association import taylor_assoc as _taylor_assoc


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
            if self._design_cache["_data_version"] == self._sample._data_version:
                return self._design_cache
            self._design_cache = None

        _raw_data = self._sample._data
        local_data: pl.DataFrame = (
            cast(pl.DataFrame, _raw_data.collect())
            if isinstance(_raw_data, pl.LazyFrame)
            else cast(pl.DataFrame, _raw_data)
        )
        design = self._sample._design

        cache: dict[str, Any] = {
            "_data_version": self._sample._data_version,
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
            # Categorical only accepts string input: float-typed design
            # columns (e.g. numeric stratum/PSU codes read as f64) and other
            # non-string dtypes go through Utf8 first.
            if s.dtype not in (pl.Utf8, pl.Categorical, pl.Enum):
                s = s.cast(pl.Utf8)
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
            if self._polars_cache["_data_version"] == self._sample._data_version:
                return self._polars_cache
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

        # Fail-fast on unhandled singletons (Taylor variance path).
        # Singletons are chosen/handled at the sample level; a handled sample
        # carries a config. If no strategy was chosen and the design still has
        # singleton strata, refuse to under-report the variance silently
        # (mirrors R's options(survey.lonely.psu = "fail")).
        if config is None and getattr(self._sample, "_singletons", None):
            singles = self._sample.singleton.detected()
            if singles:
                raise SingletonError.from_singletons(singles, where="estimation")

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

        self._polars_cache = {
            "_data_version": self._sample._data_version,
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

        return compute_fpc_columns(data, pop_size, strata_col, psu_col, ssu_col)

    # ----------------------------------------------------------------
    # Internal Helpers
    # ----------------------------------------------------------------

    def _ensure_float64(self, data: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
        # One `data.schema` lookup answers both "does the column exist" and
        # "what is its dtype". The previous `c in data.columns and data[c].dtype`
        # rebuilt the full column-name list *and* materialised a Series per
        # column, making this O(B^2) when `cols` is a set of replicate weights.
        schema = data.schema
        casts = [
            pl.col(c).cast(pl.Float64)
            for c in cols
            if (dtype := schema.get(c)) is not None and dtype != pl.Float64
        ]
        return data.with_columns(casts) if casts else data

    def _coerce_y_for_prop(self, data: pl.DataFrame, y: str) -> pl.DataFrame:
        if y not in data.columns:
            return data
        dtype = data[y].dtype

        # Floats: accept if non-null values are integer-valued (e.g. 0.0/1.0 indicators,
        # or discrete codes 1.0/2.0/3.0 that ended up as float after a CSV read with nulls).
        if dtype in (pl.Float32, pl.Float64):
            s = data[y]
            # Strip nulls and NaNs before checking integrality.
            non_null = s.drop_nulls().drop_nans()
            if non_null.len() == 0:
                # All-null column: nothing to validate, cast to Int64 and let downstream handle it.
                return data.with_columns(pl.col(y).cast(pl.Int64, strict=False).alias(y))
            # Integer-valued check: floor(x) == x for every non-null value.
            is_integral = (non_null == non_null.floor()).all()
            if not is_integral:
                raise TypeError(
                    f"prop() received float column '{y}' with non-integer values. "
                    f"prop() expects a categorical/indicator variable (e.g. 0/1 or discrete "
                    f"codes). Cast to String/Categorical/Enum if this is a label, or recode "
                    f"to integer levels."
                )
            # NaN → null, then cast to Int64 so the Rust side sees a clean nullable integer.
            return data.with_columns(
                pl.when(pl.col(y).is_nan())
                .then(None)
                .otherwise(pl.col(y))
                .cast(pl.Int64, strict=False)
                .alias(y)
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
    def _normalize_deff(deff: object) -> str | None:
        """Canonicalize ``deff`` to ``"wor"``, ``"wr"`` or ``None``.

        ``deff`` names the simple-random-sample reference the design variance is
        compared against; it says nothing about how the sample was drawn. The
        two references differ by exactly the finite-population correction
        ``1 - n/N``, so they agree closely when the sampling fraction is small
        and diverge sharply when it is not -- at a 50% sampling rate they differ
        by a factor of two.

        Booleans are rejected rather than mapped. ``True`` used to select the
        without-replacement reference, but it never said which reference it
        meant, and quietly accepting an undocumented spelling would leave two
        ways to ask for the same thing indefinitely. Rejecting it loudly also
        avoids the trap that ``Literal`` is not enforced at runtime: a
        string-only implementation would read ``deff=True`` as "off" and
        silently stop reporting a design effect the caller asked for.
        """
        if deff is None:
            return None
        if isinstance(deff, bool):
            hint = (
                "Use deff='wor' for the reference True used to select, or "
                "deff='wr' for the with-replacement reference (Kish's deft^2), "
                "which needs no population size and is unaffected by rescaled "
                "weights."
                if deff
                else "Omit the argument, or pass deff=None."
            )
            raise MethodError(
                title=f"deff no longer accepts {deff!r}",
                detail=(
                    "deff now names the SRS reference the design variance is "
                    "compared against: 'wor' (without replacement, the previous "
                    "behaviour) or 'wr' (with replacement). A boolean cannot say "
                    "which reference is wanted."
                ),
                code="DEFF_BOOL_REJECTED",
                where="estimation.deff",
                param="deff",
                expected="'wor', 'wr', or None",
                got=deff,
                hint=hint,
            )
        m = str(deff).lower().replace("_", "-").strip()
        m = {
            "without-replacement": "wor",
            "srswor": "wor",
            "with-replacement": "wr",
            "srswr": "wr",
            "replace": "wr",  # R spells the with-replacement reference this way
        }.get(m, m)
        if m not in ("wor", "wr"):
            raise MethodError(
                title=f"Unknown deff reference {deff!r}",
                detail="deff selects the SRS reference: 'wor' or 'wr'.",
                code="DEFF_REFERENCE_UNKNOWN",
                where="estimation.deff",
                param="deff",
                expected="'wor', 'wr', or None",
                got=deff,
                hint="Use deff='wor' for Kish's design effect, deff='wr' for deft^2.",
            )
        return m

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

    @staticmethod
    def _t_crit(alpha: float, df: float) -> float:
        """Two-sided t critical value, or NaN when df <= 0.

        R's ``qt(1 - alpha/2, 0)`` is NaN: with no residual degrees of freedom
        the interval width is undefined. Returning NaN — rather than the old
        fallback to a normal 1.96 — makes a domain resting on a single PSU
        report NaN bounds, matching survey, instead of a zero-width CI that
        reads as a point estimate known with certainty (issue #96).
        """
        from scipy import stats

        return float(stats.t.ppf(1 - alpha / 2, df)) if df > 0 else float("nan")

    @staticmethod
    def _t_crit_arr(alpha: float, df_arr: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`_t_crit`; NaN wherever df <= 0."""
        from scipy import stats

        pos = df_arr > 0
        return np.where(pos, stats.t.ppf(1 - alpha / 2, np.where(pos, df_arr, 1.0)), np.nan)

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
        from scipy import stats

        method = self._normalize_ci_method(method)

        if method == "logit":
            if p <= 0 or p >= 1:
                return (p, p)
            t_crit = self._t_crit(alpha, df)
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
            if p <= 0 or p >= 1:
                return (p, p)
            # No residual df: width undefined (issue #96). Checked before the
            # se <= 0 short-circuit, because a lone-PSU cell has se == 0 as the
            # same artifact and must go NaN, not to a point — matching the logit
            # branch. The max/min clamps below would otherwise swallow the NaN.
            if df <= 0:
                return (float("nan"), float("nan"))
            if se <= 0:
                return (p, p)

            # Effective sample size
            n_eff = (p * (1 - p)) / (se**2)

            # df-adjustment (same as beta method)
            if df > 0 and n > 1:
                t_n = stats.t.ppf(1 - alpha / 2, n - 1)
                t_df = stats.t.ppf(1 - alpha / 2, df)
                n_eff = n_eff * (t_n / t_df) ** 2

            # Wilson score interval: roots of the score-test quadratic
            z = self._t_crit(alpha, df)
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
        n_rows = result_df.height
        if n_rows == 0:
            return []

        est_arr = result_df["est"].to_numpy()
        se_arr = result_df["se"].to_numpy()
        df_arr = result_df["df"].to_numpy().astype(np.float64)
        n_arr = (
            result_df["n"].to_numpy()
            if "n" in result_df.columns
            else np.zeros(n_rows, dtype=np.int64)
        )
        deff_arr = result_df["deff"].to_numpy() if (deff and "deff" in result_df.columns) else None

        if deff_arr is not None and n_rows and bool(np.all(np.isnan(deff_arr))):
            # The with-replacement reference has no N in it and cannot go
            # degenerate, so an entirely missing design effect means the
            # without-replacement correction 1 - n/N collapsed: the weights sum
            # to no more than the sample size. Raise rather than hand back a
            # column of NaN, which reads as "no design effect" instead of "this
            # could not be computed". A partially missing column is left alone --
            # one degenerate by-group should not fail the whole call.
            raise MethodError(
                title="Design effect is not computable for this design",
                detail=(
                    "The without-replacement reference divides by 1 - n/N, with "
                    "N taken from the sum of the weights. Here that sum is no "
                    "greater than the sample size, so the correction is zero or "
                    "negative and no design effect exists. Either the weights "
                    "have been rescaled -- normalize() makes them sum to the "
                    "sample size -- and no longer count population units, or "
                    "this is a census, in which case there is no sampling "
                    "variance to compare against."
                ),
                code="DEFF_NOT_COMPUTABLE",
                where="estimation.deff",
                param="deff",
                expected="weights summing to more than the sample size",
                got="sum(weights) <= n",
                hint=(
                    "Use deff='wr', which compares against a with-replacement "
                    "reference, needs no population size and is unaffected by "
                    "rescaled weights."
                ),
            )

        t_crits = self._t_crit_arr(alpha, df_arr)

        with np.errstate(divide="ignore", invalid="ignore"):
            cv_arr = np.where(est_arr != 0, se_arr / est_arr, np.inf)

        by_tuple = (by_col,) if by_col else None

        by_levels: list = [None] * n_rows
        if by_col and by_col in result_df.columns:
            by_levels = [(v,) for v in result_df[by_col].to_list()]

        y_levels: list = [None] * n_rows
        if as_factor and "level" in result_df.columns:
            for i, lv in enumerate(result_df["level"].to_list()):
                try:
                    y_levels[i] = int(lv)
                except (ValueError, TypeError):
                    y_levels[i] = lv

        is_prop = (param == PopParam.PROP) or as_factor

        if not is_prop:
            lci_arr = est_arr - t_crits * se_arr
            uci_arr = est_arr + t_crits * se_arr
            return [
                ParamEst(
                    y=y_name,
                    est=float(est_arr[i]),
                    se=float(se_arr[i]),
                    cv=float(cv_arr[i]),
                    lci=float(lci_arr[i]),
                    uci=float(uci_arr[i]),
                    deff=float(deff_arr[i]) if deff_arr is not None else None,
                    df=int(df_arr[i]),
                    by=by_tuple,
                    by_level=by_levels[i],
                    y_level=y_levels[i],
                    x=x_name,
                )
                for i in range(n_rows)
            ]

        ci_method_norm = self._normalize_ci_method(ci_method)

        if ci_method_norm == "logit":
            p_arr = est_arr
            valid = (p_arr > 0) & (p_arr < 1)
            lci_arr = p_arr.copy()
            uci_arr = p_arr.copy()
            if valid.any():
                pv, sev, tv = p_arr[valid], se_arr[valid], t_crits[valid]
                with np.errstate(divide="ignore", invalid="ignore"):
                    scale = np.where(sev > 0, sev / (pv * (1.0 - pv)), 0.0)
                logit_p = np.log(pv / (1.0 - pv))
                lci_arr[valid] = 1.0 / (1.0 + np.exp(-(logit_p - tv * scale)))
                uci_arr[valid] = 1.0 / (1.0 + np.exp(-(logit_p + tv * scale)))
            return [
                ParamEst(
                    y=y_name,
                    est=float(est_arr[i]),
                    se=float(se_arr[i]),
                    cv=float(cv_arr[i]),
                    lci=float(lci_arr[i]),
                    uci=float(uci_arr[i]),
                    deff=float(deff_arr[i]) if deff_arr is not None else None,
                    df=int(df_arr[i]),
                    by=by_tuple,
                    by_level=by_levels[i],
                    y_level=y_levels[i],
                    x=x_name,
                )
                for i in range(n_rows)
            ]

        # beta / korn-graubard / wilson — per-row scalar fallback
        est_list = []
        for i in range(n_rows):
            lci, uci = self._compute_prop_ci(
                p=float(est_arr[i]),
                se=float(se_arr[i]),
                alpha=alpha,
                df=int(df_arr[i]),
                n=int(n_arr[i]),
                method=ci_method_norm,
            )
            est_list.append(
                ParamEst(
                    y=y_name,
                    est=float(est_arr[i]),
                    se=float(se_arr[i]),
                    cv=float(cv_arr[i]),
                    lci=lci,
                    uci=uci,
                    deff=float(deff_arr[i]) if deff_arr is not None else None,
                    df=int(df_arr[i]),
                    by=by_tuple,
                    by_level=by_levels[i],
                    y_level=y_levels[i],
                    x=x_name,
                )
            )
        return est_list

    def _quantile_result_to_param_est(
        self,
        result_df,
        y_name,
        alpha,
        by_col,
        data,
        weight_col,
        q_method: _QuantileMethod = _QuantileMethod.HIGHER,
        set_prob: bool = True,
    ) -> list[ParamEst]:
        """Turn a Woodruff result frame into ``ParamEst`` rows.

        The Rust side returns the estimate and the standard error *on the
        probability scale*; the interval comes from inverting the weighted CDF
        at ``p ± t·se_p``, and the reported SE is the back-solved half-width.
        The per-domain CDF is built once and reused across probabilities.

        ``set_prob=False`` leaves ``ParamEst.prob`` unset, which is what the
        median path wants — it reports as MEDIAN, not as a quantile at 0.5.
        """
        n_rows = result_df.height
        if n_rows == 0:
            return []

        q_method_str = q_method.value if hasattr(q_method, "value") else str(q_method).lower()

        # The median entry points drop the column; default those rows to 0.5.
        probs = result_df["prob"].to_list() if "prob" in result_df.columns else [0.5] * n_rows

        if by_col and by_col in result_df.columns:
            domain_vals = result_df[by_col].to_list()
            unique_domains = list(dict.fromkeys(domain_vals))
        else:
            domain_vals = [None] * n_rows
            unique_domains = [None]

        domain_cache: dict = {}
        for dv in unique_domains:
            sub = (
                data.filter(pl.col(by_col) == dv).select([y_name, weight_col]).drop_nulls()
                if dv is not None
                else data.select([y_name, weight_col]).drop_nulls()
            )
            if sub.height == 0:
                domain_cache[dv] = None
                continue
            sub_sorted = sub.sort(y_name)
            y_vals = sub_sorted[y_name].to_numpy()
            w_vals = sub_sorted[weight_col].to_numpy()
            cumsum = np.cumsum(w_vals)
            total = cumsum[-1]
            domain_cache[dv] = (y_vals, cumsum / total) if total > 0 else None

        est_vals = result_df["est"].to_list()
        se_vals = result_df["se"].to_list()
        df_vals = result_df["df"].to_list()
        by_tuple = (by_col,) if by_col else None
        est_list = []

        for i in range(n_rows):
            est = float(est_vals[i])
            se_p = float(se_vals[i])
            df_val = int(df_vals[i])
            dv = domain_vals[i]
            p = float(probs[i])

            t_crit = self._t_crit(alpha, df_val)
            cached = domain_cache.get(dv)
            if df_val <= 0 or cached is None:
                # No residual df (issue #96) or an empty domain: the interval
                # is undefined. Skip the CDF inversion, whose probability
                # arguments would otherwise be silently clamped into [0, 1].
                lci = uci = float("nan")
            else:
                p_lower = max(0.0, p - t_crit * se_p)
                p_upper = min(1.0, p + t_crit * se_p)
                y_sorted, cdf = cached
                # Invert with the same rule that located the point estimate.
                # R's oldsvyquantile hands one method/f pair to both its point
                # approxfun and its endpoint approx; inverting linearly here
                # while estimating with, say, "higher" shrinks the interval —
                # badly so in sparse tails, where consecutive order statistics
                # are far apart.
                lci, uci = rs.weighted_quantile_at(
                    y_sorted, cdf, [p_lower, p_upper], quantile_method=q_method_str
                )

            se_q = (uci - lci) / (2.0 * t_crit) if t_crit > 0 else se_p
            cv = se_q / est if est != 0 else float("inf")
            est_list.append(
                ParamEst(
                    y=y_name,
                    est=est,
                    se=se_q,
                    cv=cv,
                    lci=lci,
                    uci=uci,
                    deff=None,
                    df=df_val,
                    by=by_tuple,
                    by_level=(dv,) if dv is not None else None,
                    y_level=None,
                    x=None,
                    prob=p if set_prob else None,
                )
            )
        return est_list

    def _median_result_to_param_est(
        self,
        result_df,
        y_name,
        alpha,
        by_col,
        data,
        weight_col,
        q_method: _QuantileMethod = _QuantileMethod.HIGHER,
    ) -> list[ParamEst]:
        """The p = 0.5 case, reported as a median rather than as a quantile."""
        return self._quantile_result_to_param_est(
            result_df, y_name, alpha, by_col, data, weight_col, q_method, set_prob=False
        )

    def _replicate_quantile_result_to_param_est(
        self, result_df, y_name, alpha, by_col, set_prob: bool = True
    ):
        """Replicate-weight quantiles.

        Unlike Taylor, the replicate variance is already on the quantile scale
        (each replicate re-estimates the quantile itself), so the interval is
        the usual ``est ± t·se`` — no CDF inversion.
        """
        n_rows = result_df.height
        if n_rows == 0:
            return []

        probs = result_df["prob"].to_list() if "prob" in result_df.columns else [0.5] * n_rows

        est_arr = result_df["est"].to_numpy()
        se_arr = result_df["se"].to_numpy()
        df_arr = result_df["df"].to_numpy().astype(np.float64)

        t_crits = self._t_crit_arr(alpha, df_arr)
        lci_arr = est_arr - t_crits * se_arr
        uci_arr = est_arr + t_crits * se_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_arr = np.where(est_arr != 0, se_arr / est_arr, np.inf)

        by_tuple = (by_col,) if by_col else None
        by_levels: list = [None] * n_rows
        if by_col and by_col in result_df.columns:
            by_levels = [(v,) for v in result_df[by_col].to_list()]

        return [
            ParamEst(
                y=y_name,
                est=float(est_arr[i]),
                se=float(se_arr[i]),
                cv=float(cv_arr[i]),
                lci=float(lci_arr[i]),
                uci=float(uci_arr[i]),
                deff=None,
                df=int(df_arr[i]),
                by=by_tuple,
                by_level=by_levels[i],
                y_level=None,
                x=None,
                prob=float(probs[i]) if set_prob else None,
            )
            for i in range(n_rows)
        ]

    def _replicate_median_result_to_param_est(self, result_df, y_name, alpha, by_col):
        """The p = 0.5 case, reported as a median rather than as a quantile."""
        return self._replicate_quantile_result_to_param_est(
            result_df, y_name, alpha, by_col, set_prob=False
        )

    def _build_estimate_result_light(
        self,
        est_list,
        est_cov,
        param,
        alpha,
        by_cols,
        as_factor,
        method: RepWgts | None = None,
        deff_ref: str | None = None,
    ) -> Estimate:
        metadata = getattr(self._sample, "_metadata", None)
        estimate = Estimate(param, alpha=alpha, metadata=metadata)
        estimate.method = method.method if method is not None else "Taylor"
        estimate.deff_ref = deff_ref
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
                    df=p.df,
                    by=by_tuple,
                    by_level=by_level,
                    y_level=p.y_level,
                    x=p.x,
                    x_level=p.x_level,
                    prob=p.prob,
                )
                final_ests.append(new_p)
            estimate.estimates = final_ests
        else:
            estimate.estimates = est_list
        d_cache = self._get_factorized_design()
        strata_labels = self._design_strata_labels(d_cache)
        if strata_labels is not None:
            # Copy: the memo is shared by every Estimate built off this design,
            # and the attribute is a plain mutable list on a user-facing object.
            estimate.strata = list(strata_labels)
            estimate.n_strata = len(strata_labels)
        n_psus = self._design_n_psus(d_cache)
        if n_psus is not None:
            estimate.n_psus = n_psus
        return estimate

    # The two helpers below exist only to populate reporting metadata, but they
    # run per *estimate*. Batched calls produce one Estimate per variable, so an
    # 8-variable mean() was doing 16 `np.unique` passes over the full-length
    # design arrays — 63% of that call's wall time at 1M rows, and entirely
    # serial, which is what held the batched path near 1.8 cores. The unique
    # strata and PSU count are properties of the design, not of the variable, so
    # they are memoised on the design cache (already keyed on `_data_version`,
    # so they invalidate exactly when the design arrays do).

    @staticmethod
    def _design_strata_labels(d_cache: dict[str, Any]) -> list[Any] | None:
        """Sorted unique stratum labels, or None when the design is unstratified."""
        if d_cache["stratum"] is None:
            return None
        if "_strata_labels" not in d_cache:
            strata_info = d_cache["stratum"]
            if isinstance(strata_info, tuple) and strata_info[1] is not None:
                d_cache["_strata_labels"] = strata_info[1]
            else:
                arr = strata_info[0] if isinstance(strata_info, tuple) else strata_info
                d_cache["_strata_labels"] = np.unique(arr).tolist()
        return d_cache["_strata_labels"]

    @staticmethod
    def _design_n_psus(d_cache: dict[str, Any]) -> int | None:
        """Number of distinct PSUs, falling back to the row count when no PSU is set."""
        if d_cache["psu"] is None:
            return len(d_cache["wgt"]) if d_cache["wgt"] is not None else None
        if "_n_psus" not in d_cache:
            psu_info = d_cache["psu"]
            if isinstance(psu_info, tuple) and psu_info[1] is not None:
                d_cache["_n_psus"] = len(psu_info[1])
            else:
                arr = psu_info[0] if isinstance(psu_info, tuple) else psu_info
                d_cache["_n_psus"] = len(np.unique(arr))
        return d_cache["_n_psus"]

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

    def _resolve_method(self, method: str | None) -> RepWgts | None:
        """Resolve to the replicate-weight variant, or None for Taylor.

        Replication is never selected implicitly. ``method=None`` means Taylor,
        which is what the signature ``Literal["taylor", "replication"] | None``
        already implies -- ``None`` is the unstated default, not a third mode.

        It used to be a third mode: ``None`` resolved to replication whenever
        the design carried replicate weights and no ``stratum``/``psu``. That
        made the *estimator* depend on inputs the estimator never reads --
        replication consumes the replicate columns and ``coefficients()``,
        nothing else -- so declaring a single design column silently moved the
        standard error. It was worst for JKn, where ``stratum`` and ``psu`` are
        exactly what svy needs to derive ``(n_h-1)/n_h``: the one action that
        makes JKn usable was the action that switched JKn off.
        """
        normalized = self._normalize_method(method)
        design = self._sample._design

        if normalized == "replication":
            if design.rep_wgts is None:
                raise ValueError(
                    "Replication requires rep_wgts in the design. "
                    "Create replicate weights first or use method='taylor'."
                )
            return design.rep_wgts

        # Taylor, for an explicit "taylor" and for the None default alike.
        #
        # Linearization needs design structure. Without stratum or psu every
        # row is its own PSU in a single stratum, so the variance is SRS-like
        # and df is n-1 rather than n_reps-1 -- on a 200-row file with 8
        # replicates, df=199 against a true 7. Replicate weights sitting on the
        # design is strong evidence that is not the intended estimator, but
        # choosing one is the caller's to make, so this warns and proceeds
        # rather than switching. Warned for the explicit spelling too: the
        # hazard is in the number, not in who asked for it.
        if design.rep_wgts is not None and design.stratum is None and design.psu is None:
            self._sample.warn(
                code=WarnCode.TAYLOR_WITHOUT_DESIGN,
                title="Taylor variance on a design with no stratum or psu",
                detail=(
                    "The design carries replicate weights "
                    f"({design.rep_wgts.method}, n_reps={design.rep_wgts.n_reps}) but no "
                    "stratum or psu, so linearization has no clustering or "
                    "stratification to work from and the variance is SRS-like."
                ),
                where="Sample.estimation",
                param="method",
                hint=(
                    "Pass method='replication' to use the replicate weights, or "
                    "declare stratum/psu if the frame carries them. Pass "
                    "method='taylor' explicitly to keep this variance."
                ),
            )
        return None

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
        self, param: PopParam, alpha: float, by_cols: list[str], method: RepWgts | None
    ) -> Estimate:
        """Return an empty Estimate for cases like zero-weight domains."""
        metadata = getattr(self._sample, "_metadata", None)
        est = Estimate(param, alpha=alpha, metadata=metadata)
        est.method = method.method if method is not None else "Taylor"
        est.estimates = []
        est.covariance = np.array([])
        return est

    def _taylor_multi(
        self,
        items: list,
        *,
        single_call,
        batched_call,
        prep_y: str,
        prep_extra_cols: list[str],
        by: str | Sequence[str] | None,
        where: WhereArg,
        method: str | None,
        drop_nulls: bool,
        as_factor: bool = False,
        cast_y_float: bool = True,
    ) -> EstimateList:
        """Estimate a list of items, sharing one Taylor design build.

        Fast path (ungrouped Taylor, no as_factor/drop_nulls/scale double-pass):
        one batched Rust call that indexes the design once and fans the items out
        in parallel, via ``batched_call(prep)``. Everything else falls back to
        independent per-item calls via ``single_call(item)`` — identical results,
        no design amortisation. Returns one ``Estimate`` per item, in order.
        """
        if not items:
            return []

        target_method = self._resolve_method(method)
        batched = (
            target_method is None
            and by is None
            and not as_factor
            and not drop_nulls
            and not self._should_run_double_pass()
        )
        if not batched:
            return EstimateList(single_call(it) for it in items)

        prep = prepare_data(
            self._sample,
            y=prep_y,
            extra_cols=prep_extra_cols,
            by=None,
            where=where,
            drop_nulls=drop_nulls,
            cast_y_float=cast_y_float,
            apply_singleton_filter=True,
            select_columns=True,
        )
        results = batched_call(prep)
        if where is not None:
            wc = format_where_clause(where)
            for r in results:
                r.where_clause = wc
        return EstimateList(results)

    def mean(
        self,
        y: str | Sequence[str],
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        deff: Literal["wor", "wr"] | None = None,
        fay_coef: float = 0.0,
        as_factor: bool = False,
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate | list[Estimate]:
        """Estimate population mean with standard errors.

        Parameters
        ----------
        y : str or sequence of str
            A single response column, or a list of columns. With a list, each
            variable is estimated independently and a ``list[Estimate]`` is
            returned (one per variable, in order); a single string returns a
            single ``Estimate``. For ungrouped Taylor means the list form shares
            one design build across variables (faster than a manual loop).
        deff : {'wor', 'wr'} | None, default None
            Report the design effect against a simple-random-sample reference.
            ``None`` omits it.

            The reference is a modelling choice about the *denominator*; it says
            nothing about how the sample was drawn. ``'wor'`` compares against
            SRS without replacement (Kish's design effect); ``'wr'`` compares
            against SRS with replacement (the square of Kish's "deft"). The two
            differ by exactly the finite-population correction ``1 - n/N``, so
            they agree closely when the sampling fraction is small and diverge
            sharply when it is not -- at a 50% sampling rate, common in
            evaluation studies, they differ by a factor of two, and ``'wor'``
            grows without bound as the sample approaches a census.

            ``'wor'`` infers ``N`` from the sum of the weights, so it is only
            meaningful while the weights remain reciprocals of selection
            probabilities. After ``normalize``, and to a lesser degree after
            raking or calibration, that sum is no longer a population count and
            the design effect is silently wrong -- svy cannot detect this in
            general, since it cannot tell a rescaled weight vector from a small
            population. ``'wr'`` has no ``N`` in it at all and is therefore
            unaffected; prefer it whenever the weights have been rescaled.

            The one detectable case raises: if the weights sum to no more than
            the sample size, the correction is non-positive and no design effect
            exists to report.
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design (Taylor when strata/PSU
            are available, replication otherwise).
        """
        deff_ref = self._normalize_deff(deff)

        if not isinstance(y, str):
            ys = list(y)
            return self._taylor_multi(
                ys,
                # `single_call` re-enters mean(), which normalizes again, so it
                # must carry the caller's spelling rather than the canonical one.
                single_call=lambda yy: self.mean(
                    yy,
                    by=by,
                    where=where,
                    method=method,
                    deff=deff,
                    fay_coef=fay_coef,
                    as_factor=as_factor,
                    variance_center=variance_center,
                    alpha=alpha,
                    drop_nulls=drop_nulls,
                ),
                batched_call=lambda prep: _taylor_mean_multi(
                    self, prep=prep, ys=ys, deff_ref=deff_ref, alpha=alpha
                ),
                prep_y=(ys[0] if ys else ""),
                prep_extra_cols=ys[1:],
                by=by,
                where=where,
                method=method,
                as_factor=as_factor,
                drop_nulls=drop_nulls,
            )

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
            # Ungrouped replication only: the mask path covers the ungrouped
            # kernels. Grouped (by=) replication keeps the zeroing path.
            domain_mask_for_replication=(target_method is not None and by is None),
        )

        try:
            if target_method is None:
                result = _taylor_mean(
                    self,
                    prep=prep,
                    y=y,
                    deff_ref=deff_ref,
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
        y: str | Sequence[str],
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        deff: Literal["wor", "wr"] | None = None,
        fay_coef: float = 0.0,
        as_factor: bool = False,
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate | list[Estimate]:
        """Estimate population total with standard errors.

        Parameters
        ----------
        y : str or sequence of str
            A single response column, or a list of columns. A list returns a
            ``list[Estimate]`` (one per variable, in order); ungrouped Taylor
            totals share one design build across variables.
        deff : {'wor', 'wr'} | None, default None
            Report the design effect against a simple-random-sample reference.
            ``None`` omits it.

            The reference is a modelling choice about the *denominator*; it says
            nothing about how the sample was drawn. ``'wor'`` compares against
            SRS without replacement (Kish's design effect); ``'wr'`` compares
            against SRS with replacement (the square of Kish's "deft"). The two
            differ by exactly the finite-population correction ``1 - n/N``, so
            they agree closely when the sampling fraction is small and diverge
            sharply when it is not -- at a 50% sampling rate, common in
            evaluation studies, they differ by a factor of two, and ``'wor'``
            grows without bound as the sample approaches a census.

            ``'wor'` infers ``N`` from the sum of the weights, so it is only
            meaningful while the weights remain reciprocals of selection
            probabilities. After ``normalize``, and to a lesser degree after
            raking or calibration, that sum is no longer a population count and
            the design effect is silently wrong -- svy cannot detect this in
            general, since it cannot tell a rescaled weight vector from a small
            population. ``'wr'`` has no ``N`` in it at all and is therefore
            unaffected; prefer it whenever the weights have been rescaled.

            The one detectable case raises: if the weights sum to no more than
            the sample size, the correction is non-positive and no design effect
            exists to report.
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design.
        """
        deff_ref = self._normalize_deff(deff)

        if not isinstance(y, str):
            ys = list(y)
            return self._taylor_multi(
                ys,
                single_call=lambda yy: self.total(
                    yy,
                    by=by,
                    where=where,
                    method=method,
                    deff=deff,
                    fay_coef=fay_coef,
                    as_factor=as_factor,
                    variance_center=variance_center,
                    alpha=alpha,
                    drop_nulls=drop_nulls,
                ),
                batched_call=lambda prep: _taylor_total_multi(
                    self, prep=prep, ys=ys, deff_ref=deff_ref, alpha=alpha
                ),
                prep_y=(ys[0] if ys else ""),
                prep_extra_cols=ys[1:],
                by=by,
                where=where,
                method=method,
                as_factor=as_factor,
                drop_nulls=drop_nulls,
            )

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
            # Ungrouped replication only: the mask path covers the ungrouped
            # kernels. Grouped (by=) replication keeps the zeroing path.
            domain_mask_for_replication=(target_method is not None and by is None),
        )

        try:
            if target_method is None:
                result = _taylor_total(
                    self,
                    prep=prep,
                    y=y,
                    deff_ref=deff_ref,
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
        y: str | Sequence[str],
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        ci_method: Literal["logit", "beta", "korn-graubard", "wilson"] = "logit",
        deff: Literal["wor", "wr"] | None = None,
        fay_coef: float = 0.0,
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate | list[Estimate]:
        """Estimate population proportion with standard errors.

        Parameters
        ----------
        y : str or sequence of str
            A single category column, or a list. A list returns a
            ``list[Estimate]`` (one multi-row, per-level estimate per variable);
            ungrouped Taylor proportions share one design build across variables.
        deff : {'wor', 'wr'} | None, default None
            Report the design effect against a simple-random-sample reference.
            ``None`` omits it.

            The reference is a modelling choice about the *denominator*; it says
            nothing about how the sample was drawn. ``'wor'`` compares against
            SRS without replacement (Kish's design effect); ``'wr'`` compares
            against SRS with replacement (the square of Kish's "deft"). The two
            differ by exactly the finite-population correction ``1 - n/N``, so
            they agree closely when the sampling fraction is small and diverge
            sharply when it is not -- at a 50% sampling rate, common in
            evaluation studies, they differ by a factor of two, and ``'wor'``
            grows without bound as the sample approaches a census.

            ``'wor'` infers ``N`` from the sum of the weights, so it is only
            meaningful while the weights remain reciprocals of selection
            probabilities. After ``normalize``, and to a lesser degree after
            raking or calibration, that sum is no longer a population count and
            the design effect is silently wrong -- svy cannot detect this in
            general, since it cannot tell a rescaled weight vector from a small
            population. ``'wr'`` has no ``N`` in it at all and is therefore
            unaffected; prefer it whenever the weights have been rescaled.

            The one detectable case raises: if the weights sum to no more than
            the sample size, the correction is non-positive and no design effect
            exists to report.
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design.
        """
        deff_ref = self._normalize_deff(deff)

        if not isinstance(y, str):
            ys = list(y)
            return self._taylor_multi(
                ys,
                single_call=lambda yy: self.prop(
                    yy,
                    by=by,
                    where=where,
                    method=method,
                    ci_method=ci_method,
                    deff=deff,
                    fay_coef=fay_coef,
                    variance_center=variance_center,
                    alpha=alpha,
                    drop_nulls=drop_nulls,
                ),
                batched_call=lambda prep: _taylor_prop_multi(
                    self, prep=prep, ys=ys, deff_ref=deff_ref, alpha=alpha, ci_method=ci_method
                ),
                prep_y=(ys[0] if ys else ""),
                prep_extra_cols=ys[1:],
                by=by,
                where=where,
                method=method,
                drop_nulls=drop_nulls,
                cast_y_float=False,
            )

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
            # Ungrouped replication only: the mask path covers the ungrouped
            # kernels. Grouped (by=) replication keeps the zeroing path.
            domain_mask_for_replication=(target_method is not None and by is None),
        )

        try:
            if target_method is None:
                result = _taylor_prop(
                    self,
                    prep=prep,
                    y=y,
                    deff_ref=deff_ref,
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
        y: str | Sequence[str],
        x: str | Sequence[str],
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        deff: Literal["wor", "wr"] | None = None,
        fay_coef: float = 0.0,
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate | list[Estimate]:
        """Estimate population ratio (y/x) with standard errors.

        Parameters
        ----------
        y, x : str or sequence of str
            Numerator and denominator columns. If either is a list, the call is
            batched and returns a ``list[Estimate]``: numerator/denominator are
            paired element-wise (a scalar side is broadcast to the other's
            length). Ungrouped Taylor ratios share one design build.
        deff : {'wor', 'wr'} | None, default None
            Report the design effect against a simple-random-sample reference.
            ``None`` omits it.

            The reference is a modelling choice about the *denominator*; it says
            nothing about how the sample was drawn. ``'wor'`` compares against
            SRS without replacement (Kish's design effect); ``'wr'`` compares
            against SRS with replacement (the square of Kish's "deft"). The two
            differ by exactly the finite-population correction ``1 - n/N``, so
            they agree closely when the sampling fraction is small and diverge
            sharply when it is not -- at a 50% sampling rate, common in
            evaluation studies, they differ by a factor of two, and ``'wor'``
            grows without bound as the sample approaches a census.

            ``'wor'` infers ``N`` from the sum of the weights, so it is only
            meaningful while the weights remain reciprocals of selection
            probabilities. After ``normalize``, and to a lesser degree after
            raking or calibration, that sum is no longer a population count and
            the design effect is silently wrong -- svy cannot detect this in
            general, since it cannot tell a rescaled weight vector from a small
            population. ``'wr'`` has no ``N`` in it at all and is therefore
            unaffected; prefer it whenever the weights have been rescaled.

            The one detectable case raises: if the weights sum to no more than
            the sample size, the correction is non-positive and no design effect
            exists to report.
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design.
        """
        deff_ref = self._normalize_deff(deff)

        if not (isinstance(y, str) and isinstance(x, str)):
            ys = [y] if isinstance(y, str) else list(y)
            xs = [x] if isinstance(x, str) else list(x)
            if len(ys) == 1 and len(xs) > 1:
                ys = ys * len(xs)
            if len(xs) == 1 and len(ys) > 1:
                xs = xs * len(ys)
            if len(ys) != len(xs):
                raise DimensionError(
                    title="Mismatched numerator/denominator lengths",
                    detail=(
                        "ratio() requires the numerator and denominator to be the "
                        "same length (or one of them a single column, which is "
                        "broadcast to the other)."
                    ),
                    code="RATIO_LENGTH_MISMATCH",
                    where="estimation.ratio",
                    param="x",
                    expected=f"len == {len(ys)} (or a single column)",
                    got=len(xs),
                    hint="Pass equal-length y and x lists, or a scalar for one side.",
                )
            pairs = list(zip(ys, xs))
            return self._taylor_multi(
                pairs,
                single_call=lambda pr: self.ratio(
                    pr[0],
                    pr[1],
                    by=by,
                    where=where,
                    method=method,
                    deff=deff,
                    fay_coef=fay_coef,
                    variance_center=variance_center,
                    alpha=alpha,
                    drop_nulls=drop_nulls,
                ),
                batched_call=lambda prep: _taylor_ratio_multi(
                    self,
                    prep=prep,
                    ys=[p[0] for p in pairs],
                    xs=[p[1] for p in pairs],
                    deff_ref=deff_ref,
                    alpha=alpha,
                ),
                prep_y=(ys[0] if ys else ""),
                prep_extra_cols=list(dict.fromkeys([*ys[1:], *xs])),
                by=by,
                where=where,
                method=method,
                drop_nulls=drop_nulls,
            )

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
            # Ungrouped replication only: the mask path covers the ungrouped
            # kernels. Grouped (by=) replication keeps the zeroing path.
            domain_mask_for_replication=(target_method is not None and by is None),
        )

        try:
            if target_method is None:
                result = _taylor_ratio(
                    self,
                    prep=prep,
                    y=y,
                    x=x,
                    deff_ref=deff_ref,
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

    def _assoc(
        self,
        param: PopParam,
        cols,
        *,
        by,
        where,
        method,
        ci_method: str,
        deff_ref: str | None,
        fay_coef: float,
        variance_center: str,
        alpha: float,
        drop_nulls: bool,
    ) -> Estimate:
        """Shared machinery behind ``corr`` and ``cov``.

        The two differ only in which coefficient the kernel computes and, for
        the interval, whether the parameter is bounded.
        """
        verb = "corr" if param == PopParam.CORR else "cov"
        pairs = _parse_assoc_cols(cols, where=f"estimation.{verb}")
        target_method = self._resolve_method(method)

        # Every named column must survive preparation, so the first drives
        # `y` and the rest ride along as extras.
        named = list(dict.fromkeys([c for pair in pairs for c in pair]))
        prep = prepare_data(
            self._sample,
            y=named[0],
            extra_cols=named[1:],
            by=by,
            where=where,
            drop_nulls=drop_nulls,
            cast_y_float=True,
            apply_singleton_filter=True,
            select_columns=True,
            domain_mask_for_replication=(target_method is not None and by is None),
        )

        try:
            if target_method is None:
                result = _taylor_assoc(
                    self,
                    prep=prep,
                    pairs=pairs,
                    param=param,
                    deff_ref=deff_ref,
                    alpha=alpha,
                    ci_method=ci_method,
                )
            else:
                result = _replicate_assoc(
                    self,
                    prep=prep,
                    pairs=pairs,
                    param=param,
                    method=target_method,
                    fay_coef=fay_coef,
                    variance_center=variance_center,
                    alpha=alpha,
                    ci_method=ci_method,
                )
        except RuntimeError as e:
            if "weights is zero" in str(e).lower() or "sum of weights" in str(e).lower():
                result = self._empty_estimate(param, alpha, _colspec_to_list(by), target_method)
            else:
                raise

        if where is not None:
            result.where_clause = format_where_clause(where)
        return result

    def corr(
        self,
        cols: tuple[str, str] | Sequence[str] | Sequence[tuple[str, str]],
        *,
        kind: Literal["pearson"] = "pearson",
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        ci_method: Literal["fisher", "wald"] = "fisher",
        deff: Literal["wor", "wr"] | None = None,
        fay_coef: float = 0.0,
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate:
        """Estimate design-based correlation with standard errors.

        Correlation is symmetric, so there is no numerator/denominator here and
        no ``y``/``x`` argument: name the columns and every requested pair is
        returned as its own row.

        Parameters
        ----------
        cols : pair, sequence of str, or sequence of pairs
            ``("a", "b")`` estimates that one pair. A flat list of three or
            more columns estimates every unique pair, in ``i < j`` order. A
            list of 2-tuples estimates exactly those pairs, which is how you
            ask for one column against several without also getting the pairs
            among the others.
        kind : {'pearson'}, default 'pearson'
            Which coefficient. Rank-based kinds are planned; asking for one
            today reports that it is not implemented rather than silently
            falling back.
        method : {'taylor', 'replication'} | None
            Variance estimator, auto-detected from the design when None. Note
            this selects the *variance* method — pandas spells the coefficient
            ``method=``, which here is ``kind=``.
        ci_method : {'fisher', 'wald'}, default 'fisher'
            ``'fisher'`` builds the interval on the arctanh scale and
            transforms back, keeping it inside [-1, 1]. ``'wald'`` gives the
            symmetric ``est ± t·se``, which can exceed the bounds.

        Returns
        -------
        Estimate
            One row per pair, or per (group, pair) when ``by`` is set.

        Examples
        --------
        >>> sample.est.corr(("income", "age"))                    # doctest: +SKIP
        >>> sample.est.corr(["income", "age", "educ"])            # 3 pairs
        >>> sample.est.corr([("income", "age"), ("income", "educ")])
        """
        _guard_pandas_method(method)
        _normalize_assoc_kind(kind)
        deff_ref = self._normalize_deff(deff)
        deff_ref = self._normalize_deff(deff)
        return self._assoc(
            PopParam.CORR,
            cols,
            by=by,
            where=where,
            method=method,
            ci_method=ci_method,
            deff_ref=deff_ref,
            fay_coef=fay_coef,
            variance_center=variance_center,
            alpha=alpha,
            drop_nulls=drop_nulls,
        )

    def cov(
        self,
        cols: tuple[str, str] | Sequence[str] | Sequence[tuple[str, str]],
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        deff: Literal["wor", "wr"] | None = None,
        fay_coef: float = 0.0,
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate:
        """Estimate design-based covariance with standard errors.

        Takes the same symmetric ``cols`` argument as :meth:`corr`. Estimates
        are on R ``svyvar``'s scale, carrying its n/(n-1) factor. There is no
        ``ci_method``: a covariance is unbounded, so the interval is Wald.

        A flat list gives off-diagonal pairs only, exactly as for ``corr``. To
        get a variance, name the self-pair: ``cov(("income", "income"))``.

        Returns
        -------
        Estimate
            One row per pair, or per (group, pair) when ``by`` is set.
        """
        deff_ref = self._normalize_deff(deff)
        return self._assoc(
            PopParam.COV,
            cols,
            by=by,
            where=where,
            method=method,
            ci_method="wald",
            deff_ref=deff_ref,
            fay_coef=fay_coef,
            variance_center=variance_center,
            alpha=alpha,
            drop_nulls=drop_nulls,
        )

    @staticmethod
    def _normalize_probs(p: float | Sequence[float]) -> tuple[list[float], bool]:
        """Validate ``p`` and report whether it was a scalar.

        A scalar returns a single ``Estimate``; a sequence returns one per
        probability, matching how ``y`` already behaves.
        """
        scalar = isinstance(p, (int, float)) and not isinstance(p, bool)
        probs = [float(p)] if scalar else [float(v) for v in p]

        if not probs:
            raise ValueError("p must contain at least one probability.")
        for v in probs:
            if not (0.0 < v < 1.0):
                raise ValueError(f"Each probability must lie strictly in (0, 1); got {v}.")
        if len(set(probs)) != len(probs):
            raise ValueError(f"p contains duplicate probabilities: {probs}.")
        return probs, scalar

    def quantile(
        self,
        y: str | Sequence[str],
        *,
        p: float | Sequence[float] = (0.25, 0.50, 0.75),
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        fay_coef: float = 0.0,
        q_method: Literal["higher", "lower", "nearest", "linear", "middle"] = "higher",
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate | EstimateList:
        """Estimate population quantiles with standard errors.

        Standard errors follow Woodruff (1952): the design-based variance of
        the estimated proportion ``P(Y <= q)`` is computed on the probability
        scale, and the interval comes from inverting the weighted CDF at
        ``p ± t·se_p`` — the construction behind R's ``svyquantile``. The
        reported ``se`` is the back-solved half-width ``(uci - lci) / (2t)``.

        Parameters
        ----------
        y : str or sequence of str
            A single column, or a list. A list returns one result per variable.
        p : float or sequence of float, default (0.25, 0.50, 0.75)
            Target probabilities, each strictly inside (0, 1). A scalar returns
            a single ``Estimate``; a sequence returns an ``EstimateList`` with
            one member per probability, in the order given.
        by : str or sequence of str, optional
            Domain columns. Each result then carries one row per domain.
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design.
        q_method : str, default "higher"
            Tie-handling rule when a probability falls between observations,
            analogous to R's ``qrule``.

        Returns
        -------
        Estimate or EstimateList
            One ``Estimate`` per (variable, probability) pair. Scalar ``y`` and
            scalar ``p`` give a bare ``Estimate``; anything else is an
            ``EstimateList``.

        See Also
        --------
        median : The p = 0.5 case, reported as ``PopParam.MEDIAN``.
        """
        probs, p_is_scalar = self._normalize_probs(p)

        if not isinstance(y, str):
            # Variables expand across results too; flatten so a list of
            # variables and a list of probabilities compose predictably.
            out = EstimateList()
            for yy in y:
                res = self.quantile(
                    yy,
                    p=probs,
                    by=by,
                    where=where,
                    method=method,
                    fay_coef=fay_coef,
                    q_method=q_method,
                    variance_center=variance_center,
                    alpha=alpha,
                    drop_nulls=drop_nulls,
                )
                out.extend(res if isinstance(res, list) else [res])
            return out

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
            if target_method is None:
                results = _taylor_quantile(
                    self,
                    prep=prep,
                    y=y,
                    probs=probs,
                    q_method=resolved_q_method,
                    alpha=alpha,
                )
            else:
                results = _replicate_quantile(
                    self,
                    prep=prep,
                    y=y,
                    probs=probs,
                    method=target_method,
                    fay_coef=fay_coef,
                    q_method=resolved_q_method,
                    variance_center=variance_center,
                    alpha=alpha,
                )
        except RuntimeError as e:
            if "weights is zero" in str(e).lower() or "sum of weights" in str(e).lower():
                results = [
                    self._empty_estimate(
                        PopParam.QUANTILE, alpha, _colspec_to_list(by), target_method
                    )
                    for _ in probs
                ]
            else:
                raise

        if where is not None:
            wc = format_where_clause(where)
            for r in results:
                r.where_clause = wc

        return results[0] if p_is_scalar else EstimateList(results)

    def median(
        self,
        y: str | Sequence[str],
        *,
        by: str | Sequence[str] | None = None,
        where: WhereArg = None,
        method: Literal["taylor", "replication"] | None = None,
        fay_coef: float = 0.0,
        q_method: Literal["higher", "lower", "nearest", "linear", "middle"] = "higher",
        variance_center: Literal["rep_mean", "estimate"] = "rep_mean",
        alpha: float = 0.05,
        drop_nulls: bool = False,
    ) -> Estimate | list[Estimate]:
        """Estimate population median with standard errors.

        Parameters
        ----------
        y : str or sequence of str
            A single column, or a list. A list returns a ``list[Estimate]``.
            Ungrouped Taylor medians run in parallel across variables (median
            amortises no design build; the sort dominates).
        method : str | None
            Variance estimation method: ``'taylor'`` or ``'replication'``.
            If None, auto-detected from the design.

        See Also
        --------
        quantile : Any probability. ``median(y)`` equals ``quantile(y, p=0.5)``
            numerically; only the reported ``param`` differs.
        """
        if not isinstance(y, str):
            ys = list(y)
            resolved_q = self._normalize_q_method(q_method)
            return self._taylor_multi(
                ys,
                single_call=lambda yy: self.median(
                    yy,
                    by=by,
                    where=where,
                    method=method,
                    fay_coef=fay_coef,
                    q_method=q_method,
                    variance_center=variance_center,
                    alpha=alpha,
                    drop_nulls=drop_nulls,
                ),
                batched_call=lambda prep: _taylor_median_multi(
                    self, prep=prep, ys=ys, q_method=resolved_q, alpha=alpha
                ),
                prep_y=(ys[0] if ys else ""),
                prep_extra_cols=ys[1:],
                by=by,
                where=where,
                method=method,
                drop_nulls=drop_nulls,
            )

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
            if target_method is None:
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
