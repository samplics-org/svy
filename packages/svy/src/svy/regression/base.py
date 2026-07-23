# src/svy/regression/base.py
"""
GLM fitting for complex survey data.
"""

from __future__ import annotations

import logging
import math

from typing import TYPE_CHECKING, ClassVar, Literal, Sequence, cast

import msgspec
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
from svy.core.types import WhereArg
from svy.errors.model_errors import ModelError
from svy.regression.glm import GLMCoef, GLMFit, GLMStats
from svy.regression.links import link_inverse, link_mu_eta, resolve_link
from svy.regression.prediction import GLMPred
from svy.ui.printing import format_where_clause
from svy.wrangling.rows import _compile_where_to_pl_expr


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
    >>> sample.glm.fit(y="outcome", x=["age"], where=svy.col("sex") == "F")
    >>> sample.glm.predict(new_data)
    """

    PRINT_WIDTH: ClassVar[int | None] = None

    __slots__ = (
        "_sample",
        "fitted",
        "_print_width",
        "_fit_frame",
        "_fit_weight_col",
        "_fit_domain_col",
    )

    def __init__(self, sample: Sample) -> None:
        self._sample = sample
        self.fitted: GLMFit | None = None
        self._print_width: int | None = None
        self._fit_frame: pl.DataFrame | None = None
        self._fit_weight_col: str | None = None
        self._fit_domain_col: str | None = None

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
        where: WhereArg = None,
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
        where : WhereArg, optional
            Domain restriction (e.g. ``svy.col("sex") == "F"``). Unlike pre-filtering
            with ``wrangling.filter_records``, this preserves the full design
            structure for the variance computation, producing correct domain-
            estimation standard errors that match R's
            ``svyglm(..., design = subset(d, ...))``.
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

        # If a where predicate was passed, harvest the column names it
        # references and add them to extra_cols. Otherwise prepare_data's
        # projection (select_columns=True) drops them before line 296 can
        # evaluate the predicate, raising ColumnNotFoundError.
        where_cols: list[str] = []
        if where is not None:
            where_expr = _compile_where_to_pl_expr(where)
            where_cols = list(where_expr.meta.root_names())

        # Replicate weight columns (replicate variance is computed when the
        # design carries them) and the FPC population column, both of which
        # must survive prepare_data's projection.
        design0 = self._sample._design
        rep_cols: list[str] = (
            list(design0.rep_wgts.columns) if design0.rep_wgts is not None else []
        )
        pop_size = design0.pop_size
        pop_cols: list[str] = []
        if pop_size is not None:
            _pop_col = pop_size if isinstance(pop_size, str) else pop_size.psu
            if isinstance(_pop_col, str):
                pop_cols = [_pop_col]

        # ── Centralised data preparation ─────────────────────────────────
        # prepare_data handles: materialise, column selection, missing values,
        # weight casting, singleton filter, and correct strata/psu resolution
        # (including singleton variance columns when present).
        # Covariate and where-clause columns go through null_zero_cols: rows
        # with missing values are KEPT with zeroed weights (main + replicate)
        # so the design structure (PSUs in stratum centering, df) is
        # preserved — matching R svyglm — instead of physically dropped.
        prep = prepare_data(
            self._sample,
            y=y,
            extra_cols=x_cols + where_cols + rep_cols + pop_cols,
            null_zero_cols=x_cols + where_cols,
            drop_nulls=drop_nulls,
            cast_y_float=True,
            apply_singleton_filter=True,
            select_columns=True,
        )

        df = prep.df
        w_col = prep.weight_col
        s_col = prep.strata_col
        p_col = prep.psu_col

        # ── Materialise the where predicate as a boolean by-column ───────
        # We don't pass `where` to prepare_data because the GLM engine needs
        # the full design rows in the dataframe (the Rust side does the
        # domain restriction via by_col). Instead, compile the predicate to
        # a polars expression and add it as a string column ("true"/"false").
        # The Rust engine receives this as by_col and produces one fit per
        # level; the Python side then picks the "true" level.
        by_col_name: str | None = None
        if where is not None:
            by_col_name = "__where_domain__"
            bool_expr = _compile_where_to_pl_expr(where)
            df = df.with_columns(cast(pl.Expr, bool_expr).cast(pl.Utf8).alias(by_col_name))

        # Drop rows with INVALID weights (null / non-finite / negative) —
        # unconditionally: prepare_data always provides a weight column,
        # synthesizing ones for unweighted designs. Zero-weight rows are
        # KEPT: prepare_data zero-weights missing-value rows so they
        # preserve the design structure while contributing nothing.
        df = df.filter(
            pl.col(w_col).is_not_null() & pl.col(w_col).is_finite() & (pl.col(w_col) >= 0)
        )

        # In-domain rows: where == true (when set) and positive weight.
        # Cat level enumeration, reference validation, and response
        # validation all use exactly the rows the fit will use.
        _in_dom_expr = pl.col(w_col) > 0
        if by_col_name is not None:
            _in_dom_expr = _in_dom_expr & (
                pl.col(by_col_name).str.to_lowercase() == "true"
            )
        level_df = df.filter(_in_dom_expr)

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
                # Levels enumerated over the rows the fit will use (in-domain
                # under where=), so out-of-domain-only levels cannot create
                # phantom all-zero dummies that inflate df and zero the Wald F.
                levels = (
                    level_df.select(pl.col(feat.name).drop_nulls().unique().sort())
                    .to_series()
                    .to_list()
                )

                if len(levels) < 2:
                    log.warning(f"Categorical '{feat.name}' has < 2 levels. Dropped.")
                    return [], []

                if feat.ref is not None and feat.ref not in levels:
                    raise ModelError(
                        title="Reference level not found",
                        detail=(
                            f"Cat({feat.name!r}, ref={feat.ref!r}): the reference "
                            f"level is not among the observed levels {levels!r}. "
                            "A silently substituted reference would change the "
                            "meaning of every coefficient."
                        ),
                        code="CAT_REF_NOT_FOUND",
                        where="GLM.fit",
                        param=feat.name,
                        got=feat.ref,
                        expected=levels,
                        hint="Check the level's value and dtype (e.g. 1 vs '1').",
                    )
                ref_val = feat.ref if feat.ref is not None else levels[0]
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

        # ── FPC (single-stage) ───────────────────────────────────────────
        # Per-stratum (1 - f_h) factors matching R svyglm on a design with
        # fpc=. Only the PSU-level factor applies (the sandwich meat is
        # first-stage); PopSize SSU information is ignored here.
        fpc_name: str | None = None
        if pop_cols and pop_cols[0] in df.columns:
            from svy.estimation._fpc import build_fpc_psu_column

            df, fpc_name = build_fpc_psu_column(
                df, pop_cols[0], s_col, p_col, "__svy_fpc_psu__"
            )

        # Build final selection — include the by_col when domain is set
        final_selects = (
            [pl.col(y).cast(pl.Float64)] + feature_exprs + [pl.col(w_col).cast(pl.Float64)]
        )
        if s_col and s_col in df.columns:
            final_selects.append(pl.col(s_col))
        if p_col and p_col in df.columns:
            final_selects.append(pl.col(p_col))
        if by_col_name and by_col_name in df.columns:
            final_selects.append(pl.col(by_col_name))
        if fpc_name and fpc_name in df.columns:
            final_selects.append(pl.col(fpc_name))
        for rc in rep_cols:
            if rc in df.columns:
                final_selects.append(pl.col(rc).cast(pl.Float64))

        try:
            eng_df: pl.DataFrame = df.select(final_selects)
        except Exception as e:
            raise ValueError(f"Failed to prepare data: {e}")

        # Zero-weight rows (kept for design structure) may carry nulls in
        # engineered features (e.g. Cat dummies of a null value); the Rust
        # engine requires dense y/X and these rows contribute nothing.
        _dense_cols = [y] + [c for c in feature_names if c != "_intercept_"]
        eng_df = eng_df.with_columns([pl.col(c).fill_null(0.0) for c in _dense_cols])

        # Validate response over the rows the fit will actually use (the
        # full frame may hold out-of-domain rows with other y values).
        y_data = level_df.get_column(y).drop_nulls()
        self._validate_response(y_data, fam_str)

        # ── Call Rust ─────────────────────────────────────────────────────
        # Always returns Vec<(level, ...)> with one entry per by-level, or a
        # single entry with level="" when by_col is None.
        def _run_engine(weight_name: str, fpc: str | None):
            res = rs.fit_glm_rs(
                y_name=y,
                x_names=feature_names,
                weight_name=weight_name,
                stratum_name=s_col,
                psu_name=p_col,
                fpc_name=fpc,
                by_col=by_col_name,
                family=fam_str,
                link=link_str,
                tol=tol,
                max_iter=max_iter,
                data=eng_df,
            )
            if not res:
                raise RuntimeError("GLM engine returned no results.")
            if by_col_name is None:
                return res[0]
            true_res = [r for r in res if str(r[0]).lower() == "true"]
            if not true_res:
                raise RuntimeError(
                    "where clause produced no in-domain observations; cannot fit GLM."
                )
            return true_res[0]

        try:
            chosen = _run_engine(w_col, fpc_name)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Rust GLM engine failed: {e}") from e

        (
            _level,
            beta,
            cov_flat,
            naive_flat,
            scale,
            df_resid,
            dev,
            null_dev,
            iters,
            n_obs,
        ) = chosen

        # ── Post-process ──────────────────────────────────────────────────
        k = len(feature_names)
        # n counts the rows that actually contributed: in-domain, positive
        # weight (zero-weight missing-value rows are structural only).
        n = n_obs
        cov_mat = np.array(cov_flat).reshape(k, k)
        naive_cov = np.array(naive_flat).reshape(k, k)
        params_arr = np.array(beta)

        # ── R svyglm weight scale ────────────────────────────────────────
        # The Rust engine normalizes weights over ALL rows passed; R svyglm
        # normalizes over the fitted rows only. Deviance and the model-based
        # information are linear in that scale; rho converts to R's scale
        # (rho == 1 for full-sample fits with no zero-weight rows).
        w_all = eng_df.get_column(w_col).to_numpy().astype(float)
        if by_col_name is not None:
            _dom_arr = (
                eng_df.get_column(by_col_name).cast(pl.Utf8).str.to_lowercase() == "true"
            ).to_numpy()
            in_dom_mask = _dom_arr & (w_all > 0)
        else:
            in_dom_mask = w_all > 0
        _n_in = int(in_dom_mask.sum())
        _sum_in = float(w_all[in_dom_mask].sum())
        _n_all = int(len(w_all))
        _sum_all = float(w_all.sum())
        rho = (
            (_n_in / _sum_in) * (_sum_all / _n_all)
            if _sum_in > 0 and _n_all > 0 and _sum_all > 0
            else 1.0
        )
        dev = dev * rho
        null_dev = null_dev * rho
        naive_cov = naive_cov / rho

        # Design DF — for GLM we follow the regression convention used by
        # R's svyglm: df_resid = degf(design) - (k - 1), where k - 1 is the
        # number of non-intercept fitted parameters. This differs from the
        # estimation namespace (mean, prop, ratio) which uses the raw design
        # df since no parameters are fitted. The max(1, ...) guard protects
        # against degenerate designs with very few PSUs relative to k.
        # Restricted to in-domain, positive-weight rows (Rust returns its
        # own df_resid; we recompute here for the t-quantile convention).
        in_dom_df = eng_df.filter(pl.Series(in_dom_mask))
        df_design = self._compute_design_df(in_dom_df, s_col, p_col, n)
        df_design = max(1, df_design - (k - 1))

        # ── Replicate variance (R svrepglm) ──────────────────────────────
        # When the design carries replicate weights, V(beta) comes from the
        # spread of per-replicate refits: V = sum_r c_r (b_r - mean)(...)^T
        # with the method's coefficients (previously the replicate design
        # silently fell back to Taylor SEs).
        rep_wgts = design0.rep_wgts
        if rep_wgts is not None and rep_cols:
            rep_betas = []
            for rc in rep_cols:
                try:
                    r_chosen = _run_engine(rc, None)
                except Exception as e:
                    raise ModelError(
                        title="Replicate GLM fit failed",
                        detail=f"Refit with replicate weight {rc!r} failed: {e}",
                        code="REPLICATE_FIT_FAILED",
                        where="GLM.fit",
                    ) from e
                rep_betas.append(np.asarray(r_chosen[1], dtype=float))
            B = np.vstack(rep_betas)
            coefs_r = np.asarray(
                self._rep_coefficients(rep_wgts, B.shape[0]), dtype=float
            )
            Bc = B - B.mean(axis=0)
            cov_mat = (Bc * coefs_r[:, None]).T @ Bc
            rep_df = getattr(rep_wgts, "df", None)
            if rep_df:
                df_design = max(1, int(rep_df) - (k - 1))

        se_arr = np.sqrt(np.diag(cov_mat))

        # ── Design-adjusted AIC (R survey's dAIC, Lumley & Scott 2015) ───
        # Generic families: deviance + 2*eff.p with eff.p the trace of the
        # generalized design-effect matrix. Gaussian/identity models follow
        # R's extractAIC_svylm: profile-likelihood -2l plus a sigma^2
        # design-effect term. Both validated against AIC(svyglm).
        eff_p = self._effective_parameters(naive_cov, cov_mat, intercept)
        if fam_str == "gaussian" and link_str == "identity":
            aic_val = self._gaussian_daic(
                eng_df=eng_df,
                y=y,
                feature_names=feature_names,
                params=params_arr,
                w_all=w_all,
                in_dom_mask=in_dom_mask,
                naive_cov=naive_cov,
                cov=cov_mat,
                intercept=intercept,
            )
        else:
            aic_val = dev + 2.0 * eff_p if eff_p is not None else None

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
            aic_val,
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

        fit_obj = GLMFit(
            y=y,
            family=fam_str.capitalize(),
            link=link_str,
            stats=stats_struct,
            coefs=coef_list,
            cov_matrix=cov_mat,
            term_info=term_info,
            feature_names=feature_names,
        )

        # Stamp the where clause for display (mirrors the estimation namespace).
        # Only attempt if GLMFit declares a where_clause field; otherwise skip
        # silently — display will still work without it.
        if where is not None and hasattr(fit_obj, "where_clause"):
            try:
                fit_obj = msgspec.structs.replace(
                    fit_obj,
                    where_clause=format_where_clause(where),
                )
            except Exception:
                pass

        # Persist the exact rows the fit used (post null-drop, post weight
        # filter, with the domain column) so margins() reproduces the fitted
        # frame instead of recomputing from the raw sample data.
        self._fit_frame = df
        self._fit_weight_col = w_col
        self._fit_domain_col = by_col_name

        self.fitted = fit_obj
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

    @staticmethod
    def _rep_coefficients(rep_wgts, n_reps: int) -> list[float]:
        """Per-replicate variance coefficients c_r (R svrVar's scale*rscales)."""
        rscales = getattr(rep_wgts, "rscales", None)
        if rscales is not None:
            return [float(r) for r in rscales]
        m = str(getattr(rep_wgts, "method", "")).lower()
        if "boot" in m:
            return [1.0 / n_reps] * n_reps
        if "brr" in m or "balanced" in m:
            fay = float(getattr(rep_wgts, "fay_coef", 0.0) or 0.0)
            return [1.0 / (n_reps * (1.0 - fay) ** 2)] * n_reps
        if "jack" in m:
            return [(n_reps - 1.0) / n_reps] * n_reps
        if "sdr" in m:
            return [4.0 / n_reps] * n_reps
        raise ModelError(
            title="Unknown replication method",
            detail=f"Cannot derive replicate variance coefficients for method {m!r}.",
            code="UNKNOWN_REP_METHOD",
            where="GLM.fit",
        )

    @staticmethod
    def _effective_parameters(
        naive_cov: np.ndarray, cov: np.ndarray, intercept: bool
    ) -> float | None:
        """
        Rao-Scott effective parameters: tr of the generalized design-effect
        matrix solve(V0, V) over the non-intercept coefficients — R survey's
        eff.p in extractAIC.svyglm (Lumley & Scott 2015).
        """
        k = naive_cov.shape[0]
        idx = slice(1, k) if (intercept and k > 1) else slice(0, k)
        v0 = naive_cov[idx, idx]
        v = cov[idx, idx]
        if v0.size == 0:
            return None
        try:
            lam = np.linalg.eigvals(np.linalg.solve(v0, v))
            return float(np.sum(np.real(lam)))
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _gaussian_daic(
        *,
        eng_df: pl.DataFrame,
        y: str,
        feature_names: list[str],
        params: np.ndarray,
        w_all: np.ndarray,
        in_dom_mask: np.ndarray,
        naive_cov: np.ndarray,
        cov: np.ndarray,
        intercept: bool,
    ) -> float | None:
        """
        Design-adjusted AIC for gaussian/identity models — mirrors R
        survey's extractAIC_svylm (Lumley & Scott 2015): profile Gaussian
        -2loglik at (beta_hat, sigma2_hat) plus 2*eff.p, where eff.p adds a
        sigma^2 design-effect term to the trace of solve(V0*sigma2, V).
        """
        w_norm = np.zeros_like(w_all)
        sum_in = float(w_all[in_dom_mask].sum())
        n_in = int(in_dom_mask.sum())
        if sum_in <= 0 or n_in == 0:
            return None
        w_norm[in_dom_mask] = w_all[in_dom_mask] * (n_in / sum_in)
        n_hat = float(w_norm.sum())  # == n_in by construction

        x_mat = eng_df.select(feature_names).to_numpy()
        y_arr = eng_df.get_column(y).to_numpy().astype(float)
        r2_arr = (y_arr - x_mat @ params) ** 2
        sigma2 = float((w_norm * r2_arr).sum() / n_hat)
        if sigma2 <= 0:
            return None
        minus2ll = n_hat * math.log(sigma2) + n_hat + n_hat * math.log(2.0 * math.pi)

        k = naive_cov.shape[0]
        idx = slice(1, k) if (intercept and k > 1) else slice(0, k)
        v0s = naive_cov[idx, idx] * sigma2
        v = cov[idx, idx]
        if v0s.size == 0:
            return None
        try:
            tr_mu = float(np.trace(np.linalg.solve(v0s, v)))
        except np.linalg.LinAlgError:
            return None
        # sigma^2 design effect: I_sigma2 / H_sigma2
        i_s2 = n_hat / (2.0 * sigma2**2)
        u = -1.0 / (2.0 * sigma2) + r2_arr / (2.0 * sigma2**2)
        h_s2 = float((w_norm * u**2).sum())
        if h_s2 <= 0:
            return None
        eff_p = tr_mu + i_s2 / h_s2
        return minus2ll + 2.0 * eff_p

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
        aic: float | None = None,
    ) -> GLMStats:
        """Build model statistics including the adjusted Wald F-test.

        AIC is the design-adjusted dAIC computed by the caller (R survey's
        AIC.svyglm convention, Lumley & Scott 2015). BIC is None: R survey
        provides no plain BIC for svyglm (its dBIC requires an explicit
        maximal model).
        """
        bic = None

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
