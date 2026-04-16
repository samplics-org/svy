# src/svy/regression/base.py
"""
GLM fitting for complex survey data.
"""

from __future__ import annotations

import logging
import math

from typing import TYPE_CHECKING, ClassVar, Literal, Sequence

import numpy as np
import polars as pl

from scipy import stats


# Rust backend
try:
    from svy_rs import _internal as rs
except ImportError:
    import svy_rs as rs

from svy.core.containers import FDist, TDist
from svy.core.data_prep import prepare_data
from svy.core.terms import Cat, Cross, Feature
from svy.errors.model_errors import ModelError
from svy.regression.glm import GLMCoef, GLMFit, GLMStats
from svy.regression.links import link_inverse, link_mu_eta, resolve_link
from svy.regression.prediction import GLMPred


if TYPE_CHECKING:
    from svy.core.sample import Sample
    from svy.regression.margins import GLMMargins

log = logging.getLogger(__name__)


def _normalize_family(
    family: Literal["gaussian", "binomial", "poisson", "gamma"],
) -> str:
    """
    Normalize user-facing family string to canonical lowercase form.

    Accepts (case-insensitive):
      - "gaussian"  (default)
      - "binomial"
      - "poisson"
      - "gamma"
    """
    _MAP = {
        "gaussian": "gaussian",
        "binomial": "binomial",
        "poisson": "poisson",
        "gamma": "gamma",
    }
    if not isinstance(family, str):
        raise TypeError(
            f"'family' must be a string, got {type(family).__name__}. "
            f"Use 'gaussian', 'binomial', 'poisson', or 'gamma'."
        )
    result = _MAP.get(family.strip().lower())
    if result is None:
        raise ValueError(
            f"Unknown family {family!r}. Use 'gaussian', 'binomial', 'poisson', or 'gamma'."
        )
    return result


def _normalize_link(
    link: Literal["identity", "logit", "log", "inverse", "inverse_squared"] | None,
) -> str | None:
    """
    Normalize user-facing link string to canonical lowercase form.

    Accepts (case-insensitive):
      - None                → auto (canonical link for family)
      - "identity"
      - "logit"
      - "log"
      - "inverse"
      - "inverse_squared"
    """
    _MAP = {
        "identity": "identity",
        "logit": "logit",
        "log": "log",
        "inverse": "inverse",
        "inverse_squared": "inverse_squared",
    }
    if link is None:
        return None
    if not isinstance(link, str):
        raise TypeError(
            f"'link' must be a string or None, got {type(link).__name__}. "
            f"Use 'identity', 'logit', 'log', 'inverse', or 'inverse_squared'."
        )
    result = _MAP.get(link.strip().lower())
    if result is None:
        raise ValueError(
            f"Unknown link {link!r}. "
            f"Use 'identity', 'logit', 'log', 'inverse', or 'inverse_squared'."
        )
    return result


class GLM:
    """
    Generalized Linear Models for Complex Survey Data.

    Access via `sample.glm`.

    Examples
    --------
    >>> sample.glm.fit(y="outcome", x=["age", Cat("region")])
    >>> sample.glm.predict(new_data)
    """

    PRINT_WIDTH: ClassVar[int | None] = None

    __slots__ = ("_sample", "fitted", "_print_width")

    def __init__(self, sample: Sample) -> None:
        self._sample = sample
        self.fitted: GLMFit | None = None
        self._print_width: int | None = None

    def _ensure_fitted(self) -> GLMFit:
        if self.fitted is None:
            raise ModelError.not_fitted(where="GLM")
        return self.fitted

    # --- Print Width Configuration ---

    def set_print_width(self, width: int | None) -> "GLM":
        """Set the print width for this specific instance."""
        if width is None:
            self._print_width = None
            return self
        try:
            w = int(width)
        except Exception as ex:
            raise TypeError(f"print width must be int or None; got {width!r}") from ex
        if w <= 20:
            raise ValueError("print width must be > 20 characters.")
        self._print_width = w
        return self

    @classmethod
    def set_default_print_width(cls, width: int | None) -> None:
        """Set the default print width for all GLM instances."""
        if width is None:
            cls.PRINT_WIDTH = None
            return
        try:
            w = int(width)
        except Exception as ex:
            raise TypeError(f"class print width must be int or None; got {width!r}") from ex
        if w <= 20:
            raise ValueError("class print width must be > 20 characters.")
        cls.PRINT_WIDTH = w

    # --- Display Methods ---

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed. Never calls str(self)."""
        if self.fitted is None:
            return "GLM (not fitted)"
        return self.fitted.__plain_str__()

    def __str__(self) -> str:
        from svy.ui.printing import render_rich_to_str, resolve_width

        return render_rich_to_str(self, width=resolve_width(self))

    def __repr__(self) -> str:
        if self.fitted:
            return (
                f"<GLM: Fitted {self.fitted.family} ({self.fitted.link}) on y '{self.fitted.y}'>"
            )
        return "<GLM: Unfitted>"

    def __rich_console__(self, console, options):
        if self.fitted:
            yield from self.fitted.__rich_console__(console, options)
        else:
            from rich.text import Text

            yield Text("GLM (not fitted)", style="dim")

    # --- Properties ---

    @property
    def coefs(self) -> list[GLMCoef]:
        return self._ensure_fitted().coefs

    @property
    def stats(self) -> GLMStats:
        return self._ensure_fitted().stats

    def fit(
        self,
        y: str,
        *,
        x: Sequence[Feature] | None = None,
        intercept: bool = True,
        family: Literal["gaussian", "binomial", "poisson", "gamma"] = "gaussian",
        link: Literal["identity", "logit", "log", "inverse", "inverse_squared"] | None = None,
        drop_nulls: bool = True,
        tol: float = 1e-8,
        max_iter: int = 100,
        alpha: float = 0.05,
    ) -> GLM:
        """
        Fit a GLM to the survey data.

        Parameters
        ----------
        y : str
            Response variable column name.
        x : sequence of Feature, optional
            Predictor variables. Can be strings or Cat/Cross terms.
        intercept : bool
            Include intercept term.
        family : str
            Distribution family: ``'gaussian'``, ``'binomial'``, ``'poisson'``, or ``'gamma'``.
        link : str, optional
            Link function: ``'identity'``, ``'logit'``, ``'log'``, ``'inverse'``, or
            ``'inverse_squared'``. If None, uses the canonical link for the family.
        drop_nulls : bool
            Drop rows with missing values.
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum IRLS iterations.
        alpha : float
            Significance level for confidence intervals.

        Returns
        -------
        GLM
            Self, with fitted results in `.fitted`.
        """
        # Resolve family/link
        fam_str = _normalize_family(family)
        link_str = resolve_link(fam_str, _normalize_link(link))

        # Collect x column names for prepare_data extra_cols
        feature_specs = list(x) if x else []
        x_cols = self._collect_feature_cols(feature_specs)

        # ── Centralised data preparation ─────────────────────────────────
        # prepare_data handles: materialise, column selection, missing values,
        # weight casting, singleton filter, and correct strata/psu resolution
        # (including singleton variance columns when present).

        prep = prepare_data(
            self._sample,
            y=y,
            extra_cols=x_cols,
            drop_nulls=drop_nulls,
            cast_y_float=True,
            apply_singleton_filter=True,
            select_columns=True,
        )

        df = prep.df
        w_col = prep.weight_col
        s_col = prep.strata_col
        p_col = prep.psu_col

        # Filter invalid weights
        if self._sample._design.wgt is not None:
            df = df.filter(
                (pl.col(w_col) > 0) & pl.col(w_col).is_not_null() & pl.col(w_col).is_finite()
            )

        # ── Build feature matrix ──────────────────────────────────────────
        feature_exprs: list[pl.Expr] = []
        feature_names: list[str] = []
        term_info: dict = {}

        if intercept:
            feature_exprs.append(pl.lit(1.0).alias("_intercept_"))
            feature_names.append("_intercept_")

        def _resolve_feature(feat: Feature) -> tuple[list[pl.Expr], list[str]]:
            if isinstance(feat, str):
                term_info[feat] = {"type": "continuous"}
                return [pl.col(feat).cast(pl.Float64)], [feat]

            elif isinstance(feat, Cat):
                levels = (
                    df.select(pl.col(feat.name).drop_nulls().unique().sort()).to_series().to_list()
                )

                if len(levels) < 2:
                    log.warning(f"Categorical '{feat.name}' has < 2 levels. Dropped.")
                    return [], []

                ref_val = feat.ref if feat.ref in levels else levels[0]
                levels = [ref_val] + [v for v in levels if v != ref_val]

                term_info[feat.name] = {
                    "type": "categorical",
                    "levels": levels,
                    "ref": ref_val,
                }

                exprs, names = [], []
                for level in levels[1:]:
                    name = f"{feat.name}_{level}"
                    expr = (pl.col(feat.name) == level).cast(pl.Float64).alias(name)
                    exprs.append(expr)
                    names.append(name)
                return exprs, names

            elif isinstance(feat, Cross):
                left_exprs, left_names = _resolve_feature(feat.left)
                right_exprs, right_names = _resolve_feature(feat.right)

                if not left_exprs or not right_exprs:
                    return [], []

                exprs, names = [], []
                for le, ln in zip(left_exprs, left_names):
                    for re, rn in zip(right_exprs, right_names):
                        name = f"{ln}:{rn}"
                        exprs.append((le * re).alias(name))
                        names.append(name)
                return exprs, names

            else:
                raise TypeError(f"Unknown feature type: {type(feat)}")

        for feat in feature_specs:
            exprs, names = _resolve_feature(feat)
            feature_exprs.extend(exprs)
            feature_names.extend(names)

        # Build final selection
        final_selects = (
            [pl.col(y).cast(pl.Float64)] + feature_exprs + [pl.col(w_col).cast(pl.Float64)]
        )
        if s_col and s_col in df.columns:
            final_selects.append(pl.col(s_col))
        if p_col and p_col in df.columns:
            final_selects.append(pl.col(p_col))

        try:
            eng_df: pl.DataFrame = df.select(final_selects)
        except Exception as e:
            raise ValueError(f"Failed to prepare data: {e}")

        # Validate response
        y_data = eng_df.get_column(y)
        self._validate_response(y_data, fam_str)

        # Call Rust
        try:
            res = rs.fit_glm_rs(
                y, feature_names, w_col, s_col, p_col, fam_str, link_str, tol, max_iter, eng_df
            )
        except Exception as e:
            raise RuntimeError(f"Rust GLM engine failed: {e}") from e

        beta, cov_flat, scale, df_resid, dev, null_dev, iters = res

        # Post-process
        k = len(feature_names)
        n = eng_df.height
        cov_mat = np.array(cov_flat).reshape(k, k)
        se_arr = np.sqrt(np.diag(cov_mat))
        params_arr = np.array(beta)

        # Design DF
        df_design = self._compute_design_df(eng_df, s_col, p_col, n)

        # Statistics
        stats_struct = self._build_stats(
            n,
            k,
            scale,
            dev,
            null_dev,
            iters,
            df_design,
            intercept,
            params_arr,
            cov_mat,
        )

        # Coefficients
        t_crit = stats.t.ppf(1 - alpha / 2, df_design)
        coef_list = []
        for i, name in enumerate(feature_names):
            est, se = params_arr[i], se_arr[i]
            t_val = est / se if se > 0 else 0.0
            p_val = 2 * stats.t.sf(abs(t_val), df_design)

            coef_list.append(
                GLMCoef(
                    term=name,
                    est=est,
                    se=se,
                    lci=est - t_crit * se,
                    uci=est + t_crit * se,
                    wald=TDist(value=t_val, df=df_design, p_value=p_val),
                )
            )

        self.fitted = GLMFit(
            y=y,
            family=fam_str.capitalize(),
            link=link_str,
            stats=stats_struct,
            coefs=coef_list,
            cov_matrix=cov_mat,
            term_info=term_info,
            feature_names=feature_names,
        )
        return self

    def predict(
        self,
        new_data: pl.DataFrame,
        alpha: float = 0.05,
        y_col: str | None = None,
    ) -> GLMPred:
        fit = self._ensure_fitted()

        # Build design matrix
        X = self._build_prediction_matrix(new_data, fit)

        # Extract model info
        beta = np.array([c.est for c in fit.coefs])
        cov = fit.cov_matrix
        df = fit.coefs[0].wald.df if fit.coefs[0].wald else 1e6
        if cov is None:
            raise ValueError("Covariance matrix not available")

        # Linear predictor
        eta = X @ beta
        var_eta = np.sum((X @ cov) * X, axis=1)
        se_eta = np.sqrt(np.maximum(var_eta, 0))

        t_crit = stats.t.ppf(1 - alpha / 2, df)
        lci_eta = eta - t_crit * se_eta
        uci_eta = eta + t_crit * se_eta

        yhat = link_inverse(fit.link, eta)
        lci = link_inverse(fit.link, lci_eta)
        uci = link_inverse(fit.link, uci_eta)

        dmu_deta = link_mu_eta(fit.link, eta)
        se = se_eta * np.abs(dmu_deta)

        # Residuals (if y available)
        residuals = None
        if y_col is not None and y_col in new_data.columns:
            y = new_data.get_column(y_col).cast(pl.Float64).to_numpy()
            residuals = y - yhat

        return GLMPred(
            yhat=yhat,
            se=se,
            lci=lci,
            uci=uci,
            df=df,
            residuals=residuals,
            alpha=alpha,
        )

    def margins(
        self,
        at: dict[str, list] | None = None,
        variables: list[str] | None = None,
        alpha: float = 0.05,
    ) -> "GLMMargins | list[GLMMargins]":
        from svy.regression.margins import margins as compute_margins

        return compute_margins(self, at=at, variables=variables, alpha=alpha)

    def _collect_feature_cols(self, feature_specs: list[Feature]) -> list[str]:
        """Recursively collect raw column names from feature specs."""
        cols: list[str] = []
        for feat in feature_specs:
            if isinstance(feat, str):
                cols.append(feat)
            elif isinstance(feat, Cat):
                cols.append(feat.name)
            elif isinstance(feat, Cross):
                cols.extend(self._collect_feature_cols([feat.left, feat.right]))
        return cols

    def _build_prediction_matrix(
        self,
        new_data: pl.DataFrame,
        fit: GLMFit,
    ) -> np.ndarray:
        """Build design matrix for prediction."""
        n = new_data.height
        terms = [c.term for c in fit.coefs]
        term_info = fit.term_info or {}
        k = len(terms)

        # Identify simple continuous columns that can be batch-extracted in one select
        simple_cont = {
            t
            for t in terms
            if t in new_data.columns
            and ":" not in t
            and t != "_intercept_"
            and term_info.get(t, {}).get("type") == "continuous"
        }

        # Batch-extract continuous columns in a single Polars select
        batch_mat: np.ndarray | None = None
        batch_cols: list[str] = []
        if simple_cont:
            batch_cols = [t for t in terms if t in simple_cont]  # preserves order
            batch_mat = new_data.select(
                [pl.col(c).cast(pl.Float64) for c in batch_cols]
            ).to_numpy()

        batch_idx = {col: i for i, col in enumerate(batch_cols)}

        X = np.zeros((n, k))
        for j, term in enumerate(terms):
            if term in batch_idx:
                X[:, j] = batch_mat[:, batch_idx[term]]
            else:
                X[:, j] = self._resolve_pred_term(term, new_data, term_info)

        return X

    def _resolve_pred_term(
        self,
        term: str,
        data: pl.DataFrame,
        term_info: dict,
    ) -> np.ndarray:
        """Resolve a single term for prediction."""
        n = data.height

        if term == "_intercept_":
            return np.ones(n)

        if ":" in term:
            parts = term.split(":")
            result = np.ones(n)
            for part in parts:
                result = result * self._resolve_pred_term(part, data, term_info)
            return result

        for var_name, info in term_info.items():
            if info.get("type") == "categorical":
                prefix = f"{var_name}_"
                if term.startswith(prefix):
                    level = term[len(prefix) :]
                    if level in info["levels"]:
                        col = data.get_column(var_name)
                        return self._compare_level(col, level)

        if term in data.columns:
            return data.get_column(term).cast(pl.Float64).to_numpy()

        if "_" in term:
            for i in range(len(term) - 1, 0, -1):
                if term[i] == "_":
                    var = term[:i]
                    level = term[i + 1 :]
                    if var in data.columns:
                        col = data.get_column(var)
                        return self._compare_level(col, level)

        raise KeyError(f"Cannot resolve term '{term}'")

    def _compare_level(self, col: pl.Series, level) -> np.ndarray:
        """Compare column to level, handling type coercion."""
        col_np = col.to_numpy()

        try:
            return (col_np == level).astype(np.float64)
        except (TypeError, ValueError):
            pass

        try:
            level_num = float(level) if "." in str(level) else int(level)
            return (col_np == level_num).astype(np.float64)
        except (TypeError, ValueError):
            pass

        return (col_np.astype(str) == str(level)).astype(np.float64)

    def to_polars(self) -> pl.DataFrame:
        """Export fitted coefficients to a Polars DataFrame."""
        return self._ensure_fitted().to_polars()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _validate_response(self, y_data: pl.Series, family: str) -> None:
        """Validate response variable for family."""
        if family == "binomial":
            unique = set(y_data.unique().to_list())
            if not unique <= {0, 1, 0.0, 1.0}:
                raise ModelError.domain_violation(
                    where="GLM.fit",
                    family=family,
                    violation="Non-binary target",
                    hint="Must be 0/1",
                )
        elif family in ("gamma", "inversegaussian") and y_data.min() <= 0:  # type: ignore[operator]
            raise ModelError.domain_violation(
                where="GLM.fit", family=family, violation="Non-positive values"
            )
        elif family == "poisson" and y_data.min() < 0:  # type: ignore[operator]
            raise ModelError.domain_violation(
                where="GLM.fit", family=family, violation="Negative values"
            )

    def _compute_design_df(
        self, df: pl.DataFrame, s_col: str | None, p_col: str | None, n: int
    ) -> int:
        """Compute design-based degrees of freedom."""
        if p_col and s_col:
            psu_per_stratum = df.group_by(s_col).agg(pl.col(p_col).n_unique())
            total_psu = psu_per_stratum[p_col].sum()
            n_strata = psu_per_stratum.height  # already one row per stratum
            return int(max(1, total_psu - n_strata))
        elif p_col:
            return max(1, df.get_column(p_col).n_unique() - 1)
        elif s_col:
            return max(1, n - df.get_column(s_col).n_unique())
        else:
            return max(1, n - 1)

    def _build_stats(
        self,
        n: int,
        k: int,
        scale: float,
        dev: float,
        null_dev: float,
        iters: int,
        df_design: int,
        intercept: bool,
        params: np.ndarray | None = None,
        cov_mat: np.ndarray | None = None,
    ) -> GLMStats:
        """Build model statistics including the adjusted Wald F-test."""
        aic = dev + 2 * k
        bic = aic - 2 * k + k * math.log(n) if n > 0 else None

        r2 = 1.0 - dev / null_dev if null_dev > 1e-12 else None
        r2_adj = None
        if r2 is not None:
            p = k - (1 if intercept else 0)
            if n > p + 1:
                r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

        # ── Joint Wald F-test for all predictors (excluding intercept) ────
        # F = (1/p) * beta' * V_inv * beta, where beta and V exclude the
        # intercept term. The adjusted F rescales by (df_design - p + 1) /
        # (df_design * p) to account for the design degrees of freedom.
        f_val = 0.0
        f_adj_val = 0.0
        df_num = 0.0
        p_wald = 1.0
        p_adj = 1.0

        if params is not None and cov_mat is not None:
            # Exclude intercept (first element) if present
            if intercept and k > 1:
                beta_test = params[1:]
                cov_test = cov_mat[1:, 1:]
            else:
                beta_test = params
                cov_test = cov_mat

            p_terms = len(beta_test)
            if p_terms > 0:
                df_num = float(p_terms)
                try:
                    cov_inv = np.linalg.solve(cov_test, beta_test)
                    f_val = float(beta_test @ cov_inv) / p_terms
                    # Adjusted F: accounts for design df
                    df_den_adj = df_design - p_terms + 1
                    if df_den_adj > 0:
                        f_adj_val = f_val * df_den_adj / df_design
                        p_adj = float(stats.f.sf(f_adj_val, p_terms, df_den_adj))
                    else:
                        f_adj_val = f_val
                        p_adj = float(stats.f.sf(f_val, p_terms, df_design))
                    p_wald = float(stats.f.sf(f_val, p_terms, df_design))
                except np.linalg.LinAlgError:
                    log.warning("Singular covariance matrix; Wald F-test not computed.")

        return GLMStats(
            n=n,
            wald=FDist(value=f_val, df_num=df_num, df_den=df_design, p_value=p_wald),
            wald_adj=FDist(value=f_adj_val, df_num=df_num, df_den=df_design, p_value=p_adj),
            scale=scale,
            deviance=dev,
            aic=aic,
            bic=bic,
            r_squared=r2,
            r_squared_adj=r2_adj,
            iterations=iters,
        )
