# src/svy/regression/margins.py
"""
Marginal effects and predictive margins for fitted GLM models.
"""

from __future__ import annotations

import itertools
import logging

from typing import TYPE_CHECKING, ClassVar

import msgspec
import numpy as np
import polars as pl

from svy.errors import Code, ModelError
from svy.errors._api import guided_getattr
from svy.regression.glm import _print_width, delta_var, offset_values
from svy.regression.links import link_inverse, link_mu_eta, link_mu_eta2
from svy.ui.printing import make_panel, render_plain_table, render_rich_to_str, resolve_width
from svy.utils.formats import _fmt_conf_pct


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
        Variable name for the margin; ``"a#b"`` for a joint margin.
    at : dict[str, np.ndarray] | None
        The value each variable was set to, one column per variable,
        aligned with ``margin``. None for an effect over the observed values.
    contrast : np.ndarray | None
        ``"k - ref"`` labels of a categorical marginal effect.
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

    __getattr__ = guided_getattr("GLMMargins", "GLMMargins")

    term: str
    at: dict[str, np.ndarray] | None
    contrast: np.ndarray | None
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
        cls.PRINT_WIDTH = _print_width(width, where="GLMMargins.set_default_print_width")

    @property
    def conf_level(self) -> float:
        """Confidence level (e.g., 0.95 for 95% CI)."""
        return 1.0 - self.alpha

    def to_polars(self) -> pl.DataFrame:
        """Convert margins to DataFrame."""
        data: dict = {"term": [self.term] * len(self.margin)}
        for var, col in (self.at or {}).items():
            data[var] = col
        if self.contrast is not None:
            data["contrast"] = self.contrast
        data.update(margin=self.margin, se=self.se, lci=self.lci, uci=self.uci)
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
        if self.at is not None:
            d["at"] = {var: col.tolist() for var, col in self.at.items()}
        if self.contrast is not None:
            d["contrast"] = self.contrast.tolist()
        return d

    def __len__(self) -> int:
        return len(self.margin)

    def _table(self) -> tuple[list[str], list[list[str]]]:
        """Headers and formatted rows shared by the rich and plain renderers."""
        conf_pct = _fmt_conf_pct(self.alpha)
        label_cols = [(var, list(col)) for var, col in (self.at or {}).items()]
        if self.contrast is not None:
            label_cols.append(("Contrast", list(self.contrast)))
        headers = [name for name, _ in label_cols] + ["Margin", "SE", f"{conf_pct}% CI"]
        rows = [
            [_fmt_value(col[i]) for _, col in label_cols]
            + [
                f"{self.margin[i]:.6f}",
                f"{self.se[i]:.6f}",
                f"[{self.lci[i]:.6f}, {self.uci[i]:.6f}]",
            ]
            for i in range(len(self.margin))
        ]
        return headers, rows

    def __str__(self) -> str:
        return render_rich_to_str(self, width=resolve_width(self))

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed. Never calls str(self)."""
        conf_pct = _fmt_conf_pct(self.alpha)
        title = f"GLM Margins: {self.term} ({self.margin_type}, {conf_pct}% CI)"
        headers, rows = self._table()
        return f"{title}\n\n{render_plain_table(headers, rows)}"

    def __repr__(self) -> str:
        conf_pct = _fmt_conf_pct(self.alpha)
        if self.at is not None or self.contrast is not None:
            return f"GLMMargins(term='{self.term}', n={len(self)}, {conf_pct}% CI, type={self.margin_type})"
        return f"GLMMargins(term='{self.term}', {conf_pct}% CI, type={self.margin_type})"

    def __rich_console__(self, console, options):
        from rich import box
        from rich.table import Table as RTable

        conf_pct = _fmt_conf_pct(self.alpha)

        table = RTable(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            expand=False,
        )
        headers, rows = self._table()
        for h in headers:
            table.add_column(h, justify="right")
        for row in rows:
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


class GLMMarginsList(list):
    """
    A list of :class:`GLMMargins` that prints every table and stacks into
    one frame.

    A plain ``list`` subclass: indexing, iteration, ``len()`` and
    ``isinstance(result, list)`` behave as before.
    """

    __getattr__ = guided_getattr("GLMMarginsList", "GLMMarginsList")

    __slots__ = ()

    def _members(self) -> list[GLMMargins]:
        return [m for m in self if isinstance(m, GLMMargins)]

    def to_polars(self) -> pl.DataFrame:
        """
        Stack the members, one row per margin: ``term``, one column per
        ``at`` variable (null where a member does not set it), ``contrast``,
        then the estimates.
        """
        frames = [m.to_polars() for m in self._members()]
        if not frames:
            return pl.DataFrame()
        df = pl.concat(frames, how="diagonal_relaxed")
        tail = [c for c in ("contrast", "margin", "se", "lci", "uci") if c in df.columns]
        return df.select("term", *[c for c in df.columns if c not in ("term", *tail)], *tail)

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed. Never calls str(self)."""
        return "\n\n".join(m.__plain_str__() for m in self._members())

    def __str__(self) -> str:
        members = self._members()
        if not members:
            return super().__repr__()
        try:
            return render_rich_to_str(self, width=resolve_width(members[0]))
        except Exception:
            return self.__plain_str__()

    # print([...]) and the REPL both reach for a list's repr.
    __repr__ = __str__

    def __rich_console__(self, console, options):
        yield from self._members()

    def show(self, *, use_rich: bool = True) -> None:
        for m in self._members():
            m.show(use_rich=use_rich)


def _fmt_value(v: object) -> str:
    # Numeric at-values print fixed-point; levels and contrast labels as text.
    return (
        f"{v:.2f}"
        if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
        else str(v)
    )


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
            raise _unknown_variable(var, sorted(model_vars), param="at")


def _unknown_variable(var: str, known: list[str], *, param: str) -> ModelError:
    return ModelError.ValueError(
        title="Variable not in the model",
        detail=f"Variable '{var}' not found in model. Available variables: {known}",
        code=Code.MARGINS_UNKNOWN_VARIABLE,
        where="GLM.margins",
        param=param,
        got=var,
        expected=known,
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
            raise _unknown_variable(var, sorted(term_info.keys()), param="variables")

        kind = term_info[var].get("type")
        if kind not in ("continuous", "categorical"):
            raise ModelError.ValueError(
                title="No marginal effect",
                detail=f"Variable {var!r} has unsupported type {kind!r} for marginal effects.",
                code=Code.MARGINS_NO_EFFECT,
                where="GLM.margins",
                param="variables",
                got=var,
            )

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
    from svy.errors._model import ModelError

    frame = glm._fit_frame
    w_col = glm._fit_weight_col
    if frame is None or w_col is None:
        raise ModelError.not_fitted(where="GLM.margins")

    d_col = glm._fit_domain_col
    if d_col is not None and d_col in frame.columns:
        frame = frame.filter(pl.col(d_col)).drop(d_col)

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
    terms = fit.feature_names
    n = data.height
    D = np.zeros((n, len(terms)))

    # Every factor that survives the product rule, evaluated once.
    needed = sorted(
        {
            part
            for term in terms
            if term != "_intercept_"
            for parts in [term.split(":")]
            if var in parts
            for part in parts
        }
    )
    factors = glm._build_term_matrix(data, fit, needed) if needed else np.zeros((n, 0))
    factor_idx = {name: i for i, name in enumerate(needed)}

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
                prod = prod * factors[:, factor_idx[part]]
            dcol += prod
        D[:, j] = dcol

    return D


def _assign(data: pl.DataFrame, fit: GLMFit, values: dict[str, object]) -> pl.DataFrame:
    """Set each variable to one value on every row (the counterfactual data)."""
    term_info = fit.term_info or {}
    exprs = []
    for var, val in values.items():
        # A categorical level must keep the column's fitted dtype so the rebuilt
        # dummies match; a continuous one must not, or an integer column would
        # truncate a fractional value.
        is_cat = (term_info.get(var) or {}).get("type") == "categorical"
        exprs.append(pl.lit(val, dtype=data.schema.get(var) if is_cat else None).alias(var))
    return data.with_columns(exprs)


def _link_pieces(
    glm: GLM, fit: GLMFit, data: pl.DataFrame, beta_vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design matrix and dmu/deta, d2mu/deta2 over the rows of `data`."""
    X = glm._build_prediction_matrix(data, fit)
    eta = X @ beta_vec + offset_values(fit, data)
    return X, link_mu_eta(fit.link, eta), link_mu_eta2(fit.link, eta)


def _continuous_ame(
    glm: GLM,
    fit: GLMFit,
    data: pl.DataFrame,
    var: str,
    pieces: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    beta_vec: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    w_sum: float,
) -> tuple[float, float]:
    """
    Average derivative of the mean with respect to a continuous `var`.

    The derivative covers the full linear predictor, so interaction terms
    contribute beta_x + beta_{x:z} * z at the observed z. By the delta
    method, g_j = mean_w(mu''(eta) X_j deta_dx + mu'(eta) D_j).
    """
    X, dmu_deta, d2mu_deta2 = pieces
    D = _term_derivative_matrix(glm, fit, data, var)
    if not D.any():
        raise ModelError.ValueError(
            title="No marginal effect",
            detail=f"Variable {var!r} has no derivative with respect to any fitted "
            f"coefficient; it cannot have a marginal effect in this model.",
            code=Code.MARGINS_NO_EFFECT,
            where="GLM.margins",
            param="variables",
            got=var,
        )

    deta_dx = D @ beta_vec
    ame = float(np.sum(weights * dmu_deta * deta_dx) / w_sum)

    grad_rows = (weights * d2mu_deta2 * deta_dx)[:, None] * X
    grad_rows += (weights * dmu_deta)[:, None] * D
    grad = grad_rows.sum(axis=0) / w_sum
    return ame, float(np.sqrt(max(float(delta_var(grad, cov)), 0.0)))


def _categorical_contrasts(
    glm: GLM,
    fit: GLMFit,
    data: pl.DataFrame,
    var: str,
    info: dict,
    *,
    beta_vec: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    w_sum: float,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Average marginal effect of a categorical predictor: a discrete contrast,
    not a derivative.

    For each non-reference level k, with every other covariate held at its
    observed value,

        AME_k = mean_w[ mu(x, var=k) - mu(x, var=ref) ]

    and by the delta method, with mu'(.) = dmu/deta,

        d(AME_k)/dbeta = mean_w[ mu'(eta_k) x_k - mu'(eta_ref) x_ref ]

    Matches `avg_slopes()` in R marginaleffects, which reports the same
    quantity for a factor under the label "k - ref".
    """
    levels = list(info.get("levels") or [])
    labels = info.get("labels") or {}

    def text(level: object) -> str:
        return str(labels.get(level, level))

    ref = info.get("ref")
    others = [lv for lv in levels if lv != ref]
    if not others:
        raise ModelError.ValueError(
            title="No marginal effect",
            detail=f"Categorical {var!r} has no non-reference level to contrast.",
            code=Code.MARGINS_NO_EFFECT,
            where="GLM.margins",
            param="variables",
            got=var,
        )

    offset = offset_values(fit, data)

    def _counterfactual(level: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = glm._build_prediction_matrix(_assign(data, fit, {var: level}), fit)
        eta = X @ beta_vec + offset
        return link_inverse(fit.link, eta), link_mu_eta(fit.link, eta), X

    mu_ref, dmu_ref, X_ref = _counterfactual(ref)
    base_grad = (weights * dmu_ref)[:, None] * X_ref

    margins = np.zeros(len(others))
    se = np.zeros(len(others))

    for i, level in enumerate(others):
        mu_k, dmu_k, X_k = _counterfactual(level)
        margins[i] = float(np.sum(weights * (mu_k - mu_ref)) / w_sum)
        grad = ((weights * dmu_k)[:, None] * X_k - base_grad).sum(axis=0) / w_sum
        se[i] = float(np.sqrt(max(float(delta_var(grad, cov)), 0.0)))

    return [f"{text(lv)} - {text(ref)}" for lv in others], margins, se


def _with_ci(
    term: str,
    margins: np.ndarray,
    se: np.ndarray,
    *,
    df: float,
    alpha: float,
    margin_type: str,
    at: dict[str, np.ndarray] | None = None,
    contrast: np.ndarray | None = None,
) -> GLMMargins:
    from scipy import stats

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return GLMMargins(
        term=term,
        at=at,
        contrast=contrast,
        margin=margins,
        se=se,
        lci=margins - t_crit * se,
        uci=margins + t_crit * se,
        df=df,
        alpha=alpha,
        margin_type=margin_type,
    )


def _fitted_parts(glm: GLM) -> tuple[GLMFit, pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """The fit, its frame and weights, beta and V(beta)."""
    fit = glm.fitted
    if fit is None:
        raise ModelError.not_fitted(where="GLM.margins")
    if fit.cov_matrix is None:
        raise ValueError("Covariance matrix not available")
    data, weights = _fitted_frame(glm, fit)
    beta_vec = np.array([c.est for c in fit.coefs])
    return fit, data, weights, beta_vec, fit.cov_matrix


# =============================================================================
# The at= specification
# =============================================================================


class _AtSpec(msgspec.Struct, frozen=True):
    key: str | tuple[str, ...]
    names: tuple[str, ...]
    cells: list[tuple]

    @property
    def term(self) -> str:
        return "#".join(self.names)

    def column(self, j: int) -> np.ndarray:
        return np.asarray([cell[j] for cell in self.cells])


def _fitted_levels(fit: GLMFit, var: str, key: object) -> list:
    from svy.errors._model import ModelError

    info = (fit.term_info or {}).get(var) or {}
    if info.get("type") != "categorical":
        raise ModelError.ValueError(
            title="No levels to enumerate",
            detail=(
                f"at={{{key!r}: None}} asks for every fitted level of {var!r}, "
                "but it is not categorical in the model."
            ),
            code=Code.MARGINS_AT_LEVELS,
            where="GLM.margins",
            param="at",
            got=var,
            hint=f"List the values of {var!r} to use.",
        )
    return list(info.get("levels") or [])


def _parse_at(glm: GLM, fit: GLMFit, at: dict) -> list[_AtSpec]:
    """
    Turn `at=` into cells: a string key is one variable at each value, a
    tuple key is several variables at each listed cell; None enumerates the
    fitted levels (the full grid for a tuple key).
    """
    from svy.errors._model import ModelError

    if not isinstance(at, dict):
        raise ModelError.TypeError(
            title="at= must be a dict",
            detail=f"Got {type(at).__name__}.",
            code=Code.MARGINS_AT_TYPE,
            where="GLM.margins",
            param="at",
            got=type(at).__name__,
            hint='e.g. at={"x": [1, 2]} or at={("a", "b"): [("a1", "b1"), ("a2", "b2")]}',
        )

    specs: list[_AtSpec] = []
    for key, values in at.items():
        if isinstance(key, str):
            names: tuple[str, ...] = (key,)
        elif (
            isinstance(key, tuple)
            and key
            and all(isinstance(k, str) for k in key)
            and len(set(key)) == len(key)
        ):
            names = key
        else:
            raise ModelError.ValueError(
                title="Invalid at= key",
                detail=(
                    f"{key!r} is neither a variable name nor a tuple of distinct variable names."
                ),
                code=Code.MARGINS_AT_KEY,
                where="GLM.margins",
                param="at",
                got=key,
                hint='Use "x" for one variable or ("a", "b") for a joint margin.',
            )
        _validate_at_variables(glm, dict.fromkeys(names))

        if values is None:
            grid = [_fitted_levels(fit, var, key) for var in names]
            cells = list(itertools.product(*grid))
        elif isinstance(key, str):
            cells = [(v,) for v in values]
        else:
            cells = []
            for cell in values:
                if not isinstance(cell, (tuple, list)) or len(cell) != len(names):
                    raise ModelError.ValueError(
                        title="Cell does not match its key",
                        detail=(
                            f"at={{{key!r}: ...}} needs one value per variable in "
                            f"each cell; got {cell!r}."
                        ),
                        code=Code.MARGINS_AT_CELL,
                        where="GLM.margins",
                        param="at",
                        got=cell,
                        expected=f"a tuple of {len(names)} values",
                        hint=f"e.g. {tuple(f'<{n}>' for n in names)!r}, or None for every fitted level.",
                    )
                cells.append(tuple(cell))
        specs.append(_AtSpec(key=key, names=names, cells=cells))
    return specs


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

    For each value v, every fitted observation is set counterfactually
    to ``variable = v``, predictions are averaged with the survey
    weights, and the SE comes from the delta method over the
    design-based V(beta): se^2 = g' V g with
    g = sum_i w_i mu'(eta_i) x_i / sum_i w_i  (Stata ``margins``
    convention; covariates treated as fixed).
    """
    fit, data, weights, beta_vec, cov = _fitted_parts(glm)
    at_values = np.asarray(at_values)

    # The single-column fast path is only valid when `variable` is a plain
    # feature column that appears in no interaction term; otherwise the
    # interaction columns must be rebuilt from the counterfactual data.
    terms = fit.feature_names
    col_idx = terms.index(variable) if variable in terms else None
    in_interaction = any(":" in t and variable in t.split(":") for t in terms)
    if col_idx is not None and not in_interaction:
        X_base = glm._build_prediction_matrix(data, fit)

        def design(value: object) -> np.ndarray:
            X_cf = X_base.copy()
            X_cf[:, col_idx] = float(value)  # type: ignore[arg-type]
            return X_cf
    else:

        def design(value: object) -> np.ndarray:
            return glm._build_prediction_matrix(_assign(data, fit, {variable: value}), fit)

    margins, se = _average_predictions(fit, data, weights, beta_vec, cov, map(design, at_values))
    return _with_ci(
        variable,
        margins,
        se,
        df=_model_df(fit),
        alpha=alpha,
        margin_type="predictive",
        at={variable: at_values},
    )


def compute_joint_predictive_margins(glm: GLM, spec: _AtSpec, alpha: float = 0.05) -> GLMMargins:
    """Predictive margins with several variables set together, one row per cell."""
    fit, data, weights, beta_vec, cov = _fitted_parts(glm)
    designs = (
        glm._build_prediction_matrix(_assign(data, fit, dict(zip(spec.names, cell))), fit)
        for cell in spec.cells
    )
    margins, se = _average_predictions(fit, data, weights, beta_vec, cov, designs)
    return _with_ci(
        spec.term,
        margins,
        se,
        df=_model_df(fit),
        alpha=alpha,
        margin_type="predictive",
        at={name: spec.column(j) for j, name in enumerate(spec.names)},
    )


def _average_predictions(
    fit: GLMFit,
    data: pl.DataFrame,
    weights: np.ndarray,
    beta_vec: np.ndarray,
    cov: np.ndarray,
    designs,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean prediction and its delta-method SE for each design matrix."""
    # Counterfactual rows are the fitted rows with columns overwritten, so the
    # offset is the same for every design.
    offset = offset_values(fit, data)
    w_sum = weights.sum()
    margins, se = [], []
    for X_cf in designs:
        eta = X_cf @ beta_vec + offset
        margins.append(np.sum(weights * link_inverse(fit.link, eta)) / w_sum)
        grad = (weights * link_mu_eta(fit.link, eta)) @ X_cf / w_sum
        se.append(np.sqrt(max(float(delta_var(grad, cov)), 0.0)))
    return np.asarray(margins, dtype=float), np.asarray(se, dtype=float)


def compute_average_marginal_effects(
    glm: GLM,
    variables: list[str] | None = None,
    alpha: float = 0.05,
) -> GLMMarginsList:
    """
    Compute average marginal effects: a derivative for a continuous
    predictor, a contrast per non-reference level for a categorical one.
    SEs use the delta method over the design-based V(beta).
    """
    fit, data, weights, beta_vec, cov = _fitted_parts(glm)
    w_sum = weights.sum()
    df = _model_df(fit)
    term_info = fit.term_info or {}

    if variables is None:
        variables = [
            name
            for name, info in term_info.items()
            if info.get("type") in ("continuous", "categorical")
        ]
    else:
        variables = _validate_ame_variables(glm, variables)

    pieces = _link_pieces(glm, fit, data, beta_vec)
    kw = dict(beta_vec=beta_vec, cov=cov, weights=weights, w_sum=w_sum)

    results = []
    for var in variables:
        info = term_info.get(var) or {}
        if info.get("type") == "categorical":
            labels, m, s = _categorical_contrasts(glm, fit, data, var, info, **kw)
            contrast = np.array(labels, dtype=object)
        else:
            ame, se_val = _continuous_ame(glm, fit, data, var, pieces, **kw)
            m, s, contrast = np.array([ame]), np.array([se_val]), None
        results.append(
            _with_ci(var, m, s, df=df, alpha=alpha, margin_type="ame", contrast=contrast)
        )
    return GLMMarginsList(results)


def compute_conditional_effects(
    glm: GLM,
    variables: list[str],
    specs: list[_AtSpec],
    alpha: float = 0.05,
) -> GLMMarginsList:
    """
    Average marginal effects with the `at` variables set to each cell on
    every fitted row: one result per (variable, at key), one row per cell
    (per cell and contrast for a categorical variable).
    """
    from svy.errors._model import ModelError

    fit, data, weights, beta_vec, cov = _fitted_parts(glm)
    w_sum = weights.sum()
    df = _model_df(fit)
    term_info = fit.term_info or {}
    variables = _validate_ame_variables(glm, variables)
    kw = dict(beta_vec=beta_vec, cov=cov, weights=weights, w_sum=w_sum)

    results = []
    for var in variables:
        info = term_info.get(var) or {}
        is_cat = info.get("type") == "categorical"
        for spec in specs:
            if var in spec.names:
                raise ModelError.ValueError(
                    title="Effect variable fixed by at=",
                    detail=(
                        f"{var!r} is in variables= and in at={{{spec.key!r}: ...}}; "
                        "its effect cannot be taken with it held fixed."
                    ),
                    code=Code.MARGINS_AT_CONFLICT,
                    where="GLM.margins",
                    param="at",
                    got=var,
                    hint=(
                        f"Drop {var!r} from at=, or drop variables= for its predictive margins."
                    ),
                )
            labels: list[str] = []
            rows: list[tuple] = []
            margins: list[float] = []
            ses: list[float] = []
            for cell in spec.cells:
                cf = _assign(data, fit, dict(zip(spec.names, cell)))
                if is_cat:
                    lab, m, s = _categorical_contrasts(glm, fit, cf, var, info, **kw)
                    labels.extend(lab)
                else:
                    pieces = _link_pieces(glm, fit, cf, beta_vec)
                    ame, se_val = _continuous_ame(glm, fit, cf, var, pieces, **kw)
                    m, s = np.array([ame]), np.array([se_val])
                rows.extend([cell] * len(m))
                margins.extend(m)
                ses.extend(s)
            results.append(
                _with_ci(
                    var,
                    np.asarray(margins, dtype=float),
                    np.asarray(ses, dtype=float),
                    df=df,
                    alpha=alpha,
                    margin_type="ame",
                    contrast=np.array(labels, dtype=object) if is_cat else None,
                    at={
                        name: np.asarray([r[j] for r in rows]) for j, name in enumerate(spec.names)
                    },
                )
            )
    return GLMMarginsList(results)


# =============================================================================
# Main Entry Point
# =============================================================================


def margins(
    glm: GLM,
    at: dict[str | tuple[str, ...], list | None] | None = None,
    variables: list[str] | None = None,
    alpha: float = 0.05,
) -> GLMMargins | GLMMarginsList:
    """
    Compute marginal effects or predictive margins.

    Parameters
    ----------
    glm : GLM
        Fitted GLM model.
    at : dict, optional
        Values to set on every fitted row, one margin per key. A string key
        sets one variable to each value: ``{"meals": [20, 50, 80]}``. A tuple
        key sets several together, one row per listed cell:
        ``{("a", "b"): [("a1", "b1"), ("a2", "b2")]}``. ``None`` as the value
        means every fitted level of a categorical (every combination for a
        tuple key).
    variables : list of str, optional
        Variables to compute marginal effects for. With `at`, the effects
        are taken at each `at` cell instead of over the observed values.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    GLMMargins or GLMMarginsList
        Predictive margins: one per `at` key, a single result for one key.
        Marginal effects: always a list.

    Raises
    ------
    ValueError
        If a variable in `at` or `variables` is not in the model.
    """
    from svy.errors._model import ModelError

    if glm.fitted is None:
        raise ModelError.not_fitted(where="GLM.margins")

    if at is None:
        return compute_average_marginal_effects(glm, variables, alpha)

    specs = _parse_at(glm, glm.fitted, at)
    if variables is not None:
        return compute_conditional_effects(glm, variables, specs, alpha)

    results = [
        compute_predictive_margins(glm, spec.key, [c[0] for c in spec.cells], alpha)
        if isinstance(spec.key, str)
        else compute_joint_predictive_margins(glm, spec, alpha)
        for spec in specs
    ]
    return results[0] if len(results) == 1 else GLMMarginsList(results)
