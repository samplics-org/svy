# src/svy/regression/margins.py
"""
Marginal effects and predictive margins for fitted GLM models.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, ClassVar, cast

import msgspec
import numpy as np
import polars as pl

from scipy import stats

from svy.regression.base import link_mu_eta
from svy.ui.printing import make_panel, render_plain_table, render_rich_to_str, resolve_width


if TYPE_CHECKING:
    from svy.regression.glm import GLM

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
                    f"{self.values[i]:.2f}",
                    f"{self.margin[i]:.6f}",
                    f"{self.se[i]:.6f}",
                    f"[{self.lci[i]:.6f}, {self.uci[i]:.6f}]",
                ]
                for i in range(len(self.margin))
            ]
        else:
            headers = ["Margin", "SE", ci_header]
            rows = [
                [
                    f"{self.margin[i]:.6f}",
                    f"{self.se[i]:.6f}",
                    f"[{self.lci[i]:.6f}, {self.uci[i]:.6f}]",
                ]
                for i in range(len(self.margin))
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

        for i in range(len(self.margin)):
            row = []
            if self.values is not None:
                row.append(f"{self.values[i]:.2f}")
            row.extend(
                [
                    f"{self.margin[i]:.6f}",
                    f"{self.se[i]:.6f}",
                    f"[{self.lci[i]:.6f}, {self.uci[i]:.6f}]",
                ]
            )
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


def compute_predictive_margins(
    glm: GLM,
    variable: str,
    at_values: list | np.ndarray,
    alpha: float = 0.05,
) -> GLMMargins:
    """
    Compute predictive margins at specific values of a variable.
    """
    fit = glm.fitted
    sample = glm._sample

    # Get original data
    _raw = sample._data
    data: pl.DataFrame = (
        _raw if isinstance(_raw, pl.DataFrame) else cast(pl.DataFrame, _raw.collect())
    )

    # Get weights
    design = sample._design
    w_col = design.wgt or "_svy_ones_"
    if design.wgt is None:
        data = data.with_columns(pl.lit(1.0).alias(w_col))

    weights = data.get_column(w_col).to_numpy().astype(float)
    w_sum = weights.sum()

    # Get df from model
    if fit is None:
        raise ValueError("Model not fitted")
    df = fit.coefs[0].wald.df if fit.coefs[0].wald else 1e6

    at_values = np.asarray(at_values)
    n_values = len(at_values)

    margins = np.zeros(n_values)
    se = np.zeros(n_values)

    # For each value, compute mean prediction
    for i, val in enumerate(at_values):
        # Create counterfactual data with variable set to val
        cf_data = data.with_columns(pl.lit(val).alias(variable))

        # Get predictions
        pred = glm.predict(cf_data)

        # Weighted mean
        yhat = pred.yhat
        margins[i] = np.sum(weights * yhat) / w_sum

        # Survey variance of mean
        ybar = margins[i]
        var_est = np.sum(weights**2 * (yhat - ybar) ** 2) / (w_sum**2)

        # Finite population correction: n/(n-1)
        n = len(weights)
        var_est *= n / (n - 1)

        se[i] = np.sqrt(var_est)

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
    """
    fit = glm.fitted
    sample = glm._sample

    # Get original data
    _raw2 = sample._data
    data: pl.DataFrame = (
        _raw2 if isinstance(_raw2, pl.DataFrame) else cast(pl.DataFrame, _raw2.collect())
    )

    # Get weights
    design = sample._design
    w_col = design.wgt or "_svy_ones_"
    if design.wgt is None:
        data = data.with_columns(pl.lit(1.0).alias(w_col))

    weights = data.get_column(w_col).to_numpy().astype(float)
    w_sum = weights.sum()

    # Get df and coefficients
    if fit is None:
        raise ValueError("Model not fitted")
    df = fit.coefs[0].wald.df if fit.coefs[0].wald else 1e6
    beta = {c.term: c.est for c in fit.coefs}
    beta_se = {c.term: c.se for c in fit.coefs}

    # Get predictions on link scale
    X = glm._build_prediction_matrix(data, fit)
    beta_vec = np.array([c.est for c in fit.coefs])
    eta = X @ beta_vec
    dmu_deta = link_mu_eta(fit.link, eta)

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
        if var not in beta:
            log.warning(f"Variable '{var}' not found in model coefficients")
            continue

        # AME = weighted mean of (dmu/deta * beta)
        ame = np.sum(weights * dmu_deta * beta[var]) / w_sum

        # SE via delta method (simplified)
        mean_dmu_deta = np.sum(weights * dmu_deta) / w_sum
        se_val = abs(mean_dmu_deta) * beta_se[var]

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
