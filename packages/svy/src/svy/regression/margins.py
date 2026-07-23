# src/svy/regression/margins.py
"""
Marginal effects and predictive margins for fitted GLM models.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, ClassVar

import msgspec
import numpy as np
import polars as pl

from scipy import stats

from svy.regression.links import link_inverse, link_mu_eta, link_mu_eta2
from svy.ui.printing import make_panel, render_plain_table, render_rich_to_str, resolve_width


if TYPE_CHECKING:
    from svy.regression.base import GLM
    from svy.regression.glm import GLMFit

log = logging.getLogger(__name__)


# =============================================================================
# Result Container
# =============================================================================


class GLMMargins(msgspec.Struct, frozen=True):
    """
    Marginal effects or predictive margins from a fitted GLM.

    Attributes
    ----------
    term : str
        Variable name for the margin.
    values : np.ndarray | None
        Values at which margins are computed (for `at` margins).
    margin : np.ndarray
        Marginal effect or predicted mean at each value.
    se : np.ndarray
        Standard errors.
    lci : np.ndarray
        Lower confidence interval bounds.
    uci : np.ndarray
        Upper confidence interval bounds.
    df : float
        Degrees of freedom.
    alpha : float
        Significance level.
    margin_type : str
        Type of margin: "ame" (average marginal effect) or "predictive" (predictive margin).
    """

    PRINT_WIDTH: ClassVar[int | None] = None

    term: str
    values: np.ndarray | None
    margin: np.ndarray
    se: np.ndarray
    lci: np.ndarray
    uci: np.ndarray
    df: float
    alpha: float
    margin_type: str

    @classmethod
    def set_default_print_width(cls, width: int | None) -> None:
        if width is None:
            cls.PRINT_WIDTH = None
            return
        try:
            w = int(width)
        except Exception as ex:
            raise TypeError(f"print width must be int or None; got {width!r}") from ex
        if w <= 20:
            raise ValueError("print width must be > 20 characters.")
        cls.PRINT_WIDTH = w

    @property
    def conf_level(self) -> float:
        """Confidence level (e.g., 0.95 for 95% CI)."""
        return 1.0 - self.alpha

    def to_polars(self) -> pl.DataFrame:
        """Convert margins to DataFrame."""
        data = {
            "term": [self.term] * len(self.margin),
            "margin": self.margin,
            "se": self.se,
            "lci": self.lci,
            "uci": self.uci,
        }
        if self.values is not None:
            data["value"] = self.values
        return pl.DataFrame(data)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {
            "term": self.term,
            "margin": self.margin.tolist(),
            "se": self.se.tolist(),
            "lci": self.lci.tolist(),
            "uci": self.uci.tolist(),
            "df": self.df,
            "alpha": self.alpha,
            "margin_type": self.margin_type,
        }
        if self.values is not None:
            d["values"] = self.values.tolist()
        return d

    def __len__(self) -> int:
        return len(self.margin)

    def __str__(self) -> str:
        return render_rich_to_str(self, width=resolve_width(self))

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed. Never calls str(self)."""
        conf_pct = int(self.conf_level * 100)
        title = f"GLM Margins: {self.term} ({self.margin_type}, {conf_pct}% CI)"
        ci_header = f"{conf_pct}% CI"
        if self.values is not None:
            headers = ["Value", "Margin", "SE", ci_header]
            rows = [
                [
                    f"{v:.2f}",
                    f"{m:.6f}",
                    f"{s:.6f}",
                    f"[{l:.6f}, {u:.6f}]",
                ]
                for v, m, s, l, u in zip(self.values, self.margin, self.se, self.lci, self.uci)
            ]
        else:
            headers = ["Margin", "SE", ci_header]
            rows = [
                [
                    f"{m:.6f}",
                    f"{s:.6f}",
                    f"[{l:.6f}, {u:.6f}]",
                ]
                for m, s, l, u in zip(self.margin, self.se, self.lci, self.uci)
            ]
        return f"{title}\n\n{render_plain_table(headers, rows)}"

    def __repr__(self) -> str:
        conf_pct = int(self.conf_level * 100)
        if self.values is not None:
            return f"GLMMargins(term='{self.term}', n={len(self)}, {conf_pct}% CI, type={self.margin_type})"
        return f"GLMMargins(term='{self.term}', {conf_pct}% CI, type={self.margin_type})"

    def __rich_console__(self, console, options):
        from rich import box
        from rich.table import Table as RTable

        conf_pct = int(self.conf_level * 100)

        table = RTable(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            expand=False,
        )
        if self.values is not None:
            table.add_column("Value", justify="right")
        table.add_column("Margin", justify="right")
        table.add_column("SE", justify="right")
        table.add_column(f"{conf_pct}% CI", justify="right")

        for row_data in zip(
            *(
                ([self.values] if self.values is not None else [])
                + [self.margin, self.se, self.lci, self.uci]
            )
        ):
            if self.values is not None:
                v, m, s, l, u = row_data
                row = [f"{v:.2f}", f"{m:.6f}", f"{s:.6f}", f"[{l:.6f}, {u:.6f}]"]
            else:
                m, s, l, u = row_data
                row = [f"{m:.6f}", f"{s:.6f}", f"[{l:.6f}, {u:.6f}]"]
            table.add_row(*row)

        title = f"GLM Margins: [bold]{self.term}[/bold] ({self.margin_type}, {conf_pct}% CI)"
        yield make_panel([table], title=title, obj=self, kind="estimate")

    def show(self, *, use_rich: bool = True) -> None:
        from svy.ui.printing import rich_available

        if use_rich and rich_available():
            import sys

            from rich.console import Console

            Console(
                file=sys.stdout,
                force_terminal=True,
                emoji=False,
                width=resolve_width(self),
                soft_wrap=True,
            ).print(self)
            return
        print(self.__plain_str__())


# =============================================================================
# Validation
# =============================================================================


def _get_model_variables(glm: GLM) -> set[str]:
    """
    Get all variable names used in the model.

    Returns both raw variable names and categorical base names.
    """
    fit = glm.fitted
    if fit is None:
        return set()
    term_info = fit.term_info or {}

    variables = set()

    # From term_info (continuous and categorical base names)
    for name, info in term_info.items():
        variables.add(name)

    # From coefficient names (for safety)
    for coef in fit.coefs:
        term = coef.term
        if term == "_intercept_":
            continue

        # Handle interaction terms
        if ":" in term:
            for part in term.split(":"):
                # Handle dummy variables (var_level)
                if "_" in part:
                    for i in range(len(part) - 1, 0, -1):
                        if part[i] == "_":
                            variables.add(part[:i])
                            break
                else:
                    variables.add(part)
        # Handle dummy variables
        elif "_" in term:
            for i in range(len(term) - 1, 0, -1):
                if term[i] == "_":
                    variables.add(term[:i])
                    break
        else:
            variables.add(term)

    return variables


def _validate_at_variables(glm: GLM, at: dict[str, list]) -> None:
    """
    Validate that all variables in `at` are in the model.

    Raises
    ------
    ValueError
        If a variable in `at` is not in the model.
    """
    model_vars = _get_model_variables(glm)

    for var in at.keys():
        if var not in model_vars:
            raise ValueError(
                f"Variable '{var}' not found in model. Available variables: {sorted(model_vars)}"
            )


def _validate_ame_variables(glm: GLM, variables: list[str]) -> list[str]:
    """
    Validate variables for AME computation.

    Returns validated list of continuous variables.
    """
    fit = glm.fitted
    if fit is None:
        return []
    term_info = fit.term_info or {}

    valid_vars = []
    for var in variables:
        if var not in term_info:
            raise ValueError(
                f"Variable '{var}' not found in model. "
                f"Available variables: {sorted(term_info.keys())}"
            )

        if term_info[var].get("type") != "continuous":
            log.warning(
                f"Variable '{var}' is categorical. "
                "AME for categorical variables is not yet supported. Skipping."
            )
            continue

        valid_vars.append(var)

    return valid_vars


# =============================================================================
# Margins Computation Functions
# =============================================================================


def _fitted_frame(glm: GLM, fit: GLMFit) -> tuple[pl.DataFrame, np.ndarray]:
    """
    Return the (data, weights) the fit actually used.

    The frame is persisted by ``GLM.fit`` (post null-drop, post weight
    filter) and restricted here to in-domain rows when the fit had a
    ``where=`` clause, so margins average over exactly the fitted
    observations — matching Stata ``margins`` after a domain fit and R
    ``marginaleffects`` on ``svyglm(design = subset(...))``.
    """
    from svy.errors.model_errors import ModelError

    frame = glm._fit_frame
    w_col = glm._fit_weight_col
    if frame is None or w_col is None:
        raise ModelError.not_fitted(where="GLM.margins")

    d_col = glm._fit_domain_col
    if d_col is not None and d_col in frame.columns:
        frame = frame.filter(pl.col(d_col).str.to_lowercase() == "true").drop(d_col)

    weights = frame.get_column(w_col).to_numpy().astype(float)
    return frame, weights


def _model_df(fit: GLMFit) -> float:
    """Design degrees of freedom used for t-based intervals."""
    return fit.coefs[0].wald.df if fit.coefs[0].wald else 1e6


def _term_derivative_matrix(
    glm: GLM,
    fit: GLMFit,
    data: pl.DataFrame,
    var: str,
) -> np.ndarray:
    """
    Build the n x k matrix D with D[:, j] = d X_j / d var.

    Each engineered feature column is a product of factors (split on
    ':'). The derivative w.r.t. a continuous variable follows the
    product rule: sum over each occurrence of `var` of the product of
    the remaining factors (Cat dummies and other covariates evaluated
    at their observed values). Terms not involving `var` have zero
    derivative.
    """
    term_info = fit.term_info or {}
    terms = fit.feature_names
    n = data.height
    D = np.zeros((n, len(terms)))

    for j, term in enumerate(terms):
        if term == "_intercept_":
            continue
        parts = term.split(":")
        occurrences = [i for i, p in enumerate(parts) if p == var]
        if not occurrences:
            continue
        dcol = np.zeros(n)
        for occ in occurrences:
            prod = np.ones(n)
            for i, part in enumerate(parts):
                if i == occ:
                    continue
                prod = prod * glm._resolve_pred_term(part, data, term_info)
            dcol += prod
        D[:, j] = dcol

    return D


def compute_predictive_margins(
    glm: GLM,
    variable: str,
    at_values: list | np.ndarray,
    alpha: float = 0.05,
) -> GLMMargins:
    """
    Compute predictive margins at specific values of a variable.

    For each value v, every fitted observation is set counterfactually
    to ``variable = v``, predictions are averaged with the survey
    weights, and the SE comes from the delta method over the
    design-based V(beta): se^2 = g' V g with
    g = sum_i w_i mu'(eta_i) x_i / sum_i w_i  (Stata ``margins``
    convention; covariates treated as fixed).
    """
    fit = glm.fitted
    if fit is None:
        raise ValueError("Model not fitted")
    if fit.cov_matrix is None:
        raise ValueError("Covariance matrix not available")

    data, weights = _fitted_frame(glm, fit)
    w_sum = weights.sum()
    df = _model_df(fit)

    at_values = np.asarray(at_values)
    n_values = len(at_values)

    margins = np.zeros(n_values)
    se = np.zeros(n_values)

    beta_vec = np.array([c.est for c in fit.coefs])
    cov = fit.cov_matrix
    terms = fit.feature_names

    # The single-column fast path is only valid when `variable` is a plain
    # feature column that appears in no interaction term; otherwise the
    # interaction columns must be rebuilt from the counterfactual data.
    in_interaction = any(":" in t and variable in t.split(":") for t in terms)
    col_idx: int | None
    try:
        col_idx = terms.index(variable)
    except ValueError:
        # e.g. categorical base name — needs the full rebuild path
        col_idx = None

    X_base = glm._build_prediction_matrix(data, fit) if col_idx is not None else None

    for i, val in enumerate(at_values):
        if col_idx is not None and not in_interaction:
            X_cf = X_base.copy()
            X_cf[:, col_idx] = float(val)
        else:
            cf_data = data.with_columns(pl.lit(val).alias(variable))
            X_cf = glm._build_prediction_matrix(cf_data, fit)

        eta = X_cf @ beta_vec
        yhat = link_inverse(fit.link, eta)
        margins[i] = np.sum(weights * yhat) / w_sum

        # Delta method: gradient of the weighted mean prediction w.r.t. beta
        dmu_deta = link_mu_eta(fit.link, eta)
        grad = (weights * dmu_deta) @ X_cf / w_sum
        se[i] = np.sqrt(max(float(grad @ cov @ grad), 0.0))

    # Confidence intervals
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    lci = margins - t_crit * se
    uci = margins + t_crit * se

    return GLMMargins(
        term=variable,
        values=at_values,
        margin=margins,
        se=se,
        lci=lci,
        uci=uci,
        df=df,
        alpha=alpha,
        margin_type="predictive",
    )


def compute_average_marginal_effects(
    glm: GLM,
    variables: list[str] | None = None,
    alpha: float = 0.05,
) -> list[GLMMargins]:
    """
    Compute average marginal effects (AME) for continuous variables.

    The marginal effect differentiates the full linear predictor, so a
    variable appearing in interaction terms contributes
    beta_x + beta_{x:z} * z (evaluated at observed z). SEs use the full
    delta method over the design-based V(beta):
    g_j = mean_w(mu''(eta) X_j deta_dx + mu'(eta) D_j), se^2 = g' V g.
    """
    fit = glm.fitted
    if fit is None:
        raise ValueError("Model not fitted")
    if fit.cov_matrix is None:
        raise ValueError("Covariance matrix not available")

    data, weights = _fitted_frame(glm, fit)
    w_sum = weights.sum()
    df = _model_df(fit)

    beta_vec = np.array([c.est for c in fit.coefs])
    cov = fit.cov_matrix

    # Predictions on the link scale over the fitted rows
    X = glm._build_prediction_matrix(data, fit)
    eta = X @ beta_vec
    dmu_deta = link_mu_eta(fit.link, eta)
    d2mu_deta2 = link_mu_eta2(fit.link, eta)

    # Determine which variables to compute AME for
    term_info = fit.term_info or {}

    if variables is None:
        # All continuous variables (exclude intercept and categoricals)
        variables = [name for name, info in term_info.items() if info.get("type") == "continuous"]
    else:
        # Validate user-provided variables
        variables = _validate_ame_variables(glm, variables)

    results = []
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    for var in variables:
        D = _term_derivative_matrix(glm, fit, data, var)
        if not D.any():
            log.warning(f"Variable '{var}' not found in model coefficients")
            continue

        # d(eta)/d(var) per observation, including interaction terms
        deta_dx = D @ beta_vec
        ame = np.sum(weights * dmu_deta * deta_dx) / w_sum

        # Full delta method: gradient of AME w.r.t. beta
        grad_rows = (weights * d2mu_deta2 * deta_dx)[:, None] * X
        grad_rows += (weights * dmu_deta)[:, None] * D
        grad = grad_rows.sum(axis=0) / w_sum
        se_val = np.sqrt(max(float(grad @ cov @ grad), 0.0))

        lci = ame - t_crit * se_val
        uci = ame + t_crit * se_val

        results.append(
            GLMMargins(
                term=var,
                values=None,
                margin=np.array([ame]),
                se=np.array([se_val]),
                lci=np.array([lci]),
                uci=np.array([uci]),
                df=df,
                alpha=alpha,
                margin_type="ame",
            )
        )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def margins(
    glm: GLM,
    at: dict[str, list] | None = None,
    variables: list[str] | None = None,
    alpha: float = 0.05,
) -> GLMMargins | list[GLMMargins]:
    """
    Compute marginal effects or predictive margins.

    Parameters
    ----------
    glm : GLM
        Fitted GLM model.
    at : dict, optional
        Dictionary mapping variable names to values for predictive margins.
        Example: {"meals": [20, 50, 80]}
        Variables must be in the model.
    variables : list of str, optional
        Variables to compute AME for (when at is None).
        Must be continuous variables in the model.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    GLMMargins or list of GLMMargins

    Raises
    ------
    ValueError
        If a variable in `at` or `variables` is not in the model.
    """
    from svy.errors.model_errors import ModelError

    if glm.fitted is None:
        raise ModelError.not_fitted(where="GLM.margins")

    if at is not None:
        _validate_at_variables(glm, at)
        results = []
        for var, values in at.items():
            result = compute_predictive_margins(glm, var, values, alpha)
            results.append(result)
        return results[0] if len(results) == 1 else results
    else:
        return compute_average_marginal_effects(glm, variables, alpha)
