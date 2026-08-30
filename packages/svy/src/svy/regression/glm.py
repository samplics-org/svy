# src/svy/regression/glm.py
"""
Result containers for GLM.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import msgspec
import numpy as np
import polars as pl

from msgspec import field

from svy.core.containers import FDist, TDist
from svy.errors import ModelError
from svy.ui.printing import make_panel, render_plain_table, render_rich_to_str, resolve_width
from svy.utils.formats import _fmt_fixed, _fmt_p, _fmt_smart


if TYPE_CHECKING:
    from svy.core.terms import Cat
    from svy.estimation.contrast import Contrast, ContrastExpr

log = logging.getLogger(__name__)


# =============================================================================
# Result Structs
# =============================================================================


class GLMCoef(msgspec.Struct, frozen=True):
    """A single coefficient in a regression table."""

    term: str
    est: float
    se: float
    lci: float
    uci: float
    wald: TDist | None = None
    wald_adj: TDist | None = None

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)


class GLMStats(msgspec.Struct, frozen=True):
    """Model-level goodness-of-fit statistics."""

    n: int
    wald: FDist
    wald_adj: FDist
    scale: float
    deviance: float
    aic: float | None = None
    bic: float | None = None
    r_squared: float | None = None
    r_squared_adj: float | None = None
    iterations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)


class GLMFit(msgspec.Struct, frozen=True):
    """Immutable container for fitted GLM results."""

    PRINT_WIDTH: ClassVar[int | None] = None

    y: str
    family: str
    link: str
    stats: GLMStats
    coefs: list[GLMCoef] = field(default_factory=list)
    cov_matrix: np.ndarray | None = None
    term_info: dict | None = None
    feature_names: list[str] = field(default_factory=list)

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

    def to_dict(self) -> dict[str, Any]:
        d = msgspec.to_builtins(self)
        d.pop("cov_matrix", None)
        d.pop("term_info", None)
        return d

    # --- Contrasts & term tests ---

    def keys(self) -> list[str]:
        """Contrast keys: the coefficient names, in covariance order."""
        return [c.term for c in self.coefs]

    def _design_df(self) -> float:
        # Every coefficient's Wald t is referred to the same design df.
        for c in self.coefs:
            if c.wald is not None:
                return float(c.wald.df)
        raise ModelError(
            title="No degrees of freedom on this fit",
            detail="The fit carries no Wald statistics to take the design df from.",
            code="GLM_NO_DF",
            where="GLM.contrast",
        )

    def contrast(
        self, contrasts: "Mapping[Any, Any] | ContrastExpr", *, alpha: float = 0.05
    ) -> "Contrast":
        """Estimate linear contrasts between coefficients.

        ``contrasts`` is a contrast expression over coefficient names
        (``estd("stype_H") - estd("stype_M")``), a sparse ``{name: coef}``
        dict, or several named contrasts. Inference is t-based on the same
        design df as the coefficient table.
        """
        from svy.estimation.contrast import linear_contrast

        if self.cov_matrix is None:
            raise ModelError(
                title="No covariance on this fit",
                detail="The fit carries no coefficient covariance matrix.",
                code="GLM_NO_COVARIANCE",
                where="GLM.contrast",
            )
        values = np.array([c.est for c in self.coefs], dtype=float)
        return linear_contrast(
            self.keys(),
            values,
            self.cov_matrix,
            contrasts,
            df=self._design_df(),
            alpha=alpha,
            method="GLM",
        )

    def term_test(self, term: "str | Cat") -> FDist:
        """Joint Wald test that every coefficient of a model term is zero.

        ``term`` is a feature as passed to ``fit`` — a continuous column
        name, a categorical (by name or ``Cat``), or one exact coefficient
        name. For a categorical this jointly tests all its non-reference
        dummies: ``F = β̂ᵀ V⁻¹ β̂ / q`` on ``(q, design df)``, the
        counterpart of R's ``regTermTest(method="Wald")`` (the working
        likelihood-ratio variant is not implemented).
        """
        from scipy import stats as sp_stats

        name = getattr(term, "name", term)
        if not isinstance(name, str):
            raise TypeError(f"term must be a str or Cat, got {type(term).__name__}")

        info = (self.term_info or {}).get(name)
        if info is not None and info.get("type") == "categorical":
            cols = [f"{name}_{lvl}" for lvl in info["levels"][1:]]
        else:
            cols = [name]
        names = self.keys()
        missing = [c for c in cols if c not in names]
        if missing or self.cov_matrix is None:
            raise ModelError(
                title="Unknown model term",
                detail=(
                    f"term {name!r} resolves to coefficients {cols!r}, but "
                    f"{missing!r} are not in the fit (coefficients: {names!r})."
                    if missing
                    else "The fit carries no coefficient covariance matrix."
                ),
                code="GLM_UNKNOWN_TERM",
                where="GLM.term_test",
                param="term",
                got=name,
            )

        idx = [names.index(c) for c in cols]
        beta = np.array([self.coefs[i].est for i in idx], dtype=float)
        v_sub = self.cov_matrix[np.ix_(idx, idx)]
        q = len(idx)
        f_val = float(beta @ np.linalg.solve(v_sub, beta)) / q
        df_den = self._design_df()
        p_val = float(sp_stats.f.sf(f_val, q, df_den))
        return FDist(df_num=q, df_den=df_den, value=f_val, p_value=p_val)

    def to_polars(self) -> pl.DataFrame:
        """Convert coefficients to DataFrame."""
        data = []
        for c in self.coefs:
            row: dict[str, Any] = {
                "term": c.term,
                "estimate": c.est,
                "std_err": c.se,
                "conf_low": c.lci,
                "conf_high": c.uci,
            }
            if c.wald:
                row.update(
                    {
                        "statistic": c.wald.value,
                        "p_value": c.wald.p_value,
                        "df": c.wald.df,
                    }
                )
            else:
                row.update({"statistic": None, "p_value": None, "df": None})
            if c.wald_adj:
                row["adj_statistic"] = c.wald_adj.value
                row["adj_p_value"] = c.wald_adj.p_value
                row["adj_df"] = c.wald_adj.df
            data.append(row)
        return pl.DataFrame(data)

    def __rich_console__(self, console, options):
        from rich import box
        from rich.console import Group
        from rich.table import Table as RTable
        from rich.text import Text

        if not self.coefs:
            yield Text("Empty GLM Model", style="red")
            return

        # Stats grid
        st = self.stats
        df_resid = st.wald.df_den if st.wald else "-"

        stats_rows = [
            ("Observations", str(st.n), "AIC", _fmt_smart(st.aic)),
            ("DF Residuals", str(df_resid), "BIC", _fmt_smart(st.bic)),
            ("Deviance", _fmt_smart(st.deviance), "Scale", _fmt_smart(st.scale)),
        ]
        if st.r_squared is not None:
            stats_rows.append(
                (
                    "R-squared",
                    _fmt_fixed(st.r_squared),
                    "R-sq (adj)",
                    _fmt_fixed(st.r_squared_adj),
                )
            )
        if st.iterations:
            stats_rows.append(("", "", "Iterations", str(st.iterations)))
        if st.wald_adj:
            stats_rows.append(
                (
                    "F-stat (adj)",
                    _fmt_fixed(st.wald_adj.value),
                    "Prob (F-adj)",
                    _fmt_p(st.wald_adj.p_value),
                )
            )

        stats_grid = RTable.grid(padding=(0, 2))
        stats_grid.add_column(style="bold")
        stats_grid.add_column(justify="right")
        stats_grid.add_column(style="bold")
        stats_grid.add_column(justify="right")
        for r in stats_rows:
            stats_grid.add_row(*r)

        # Coefficients table
        coef_tbl = RTable(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            expand=False,
        )
        for name, justify in [
            ("Term", "left"),
            ("Coef.", "right"),
            ("Std.Err.", "right"),
            ("t", "right"),
            ("P>|t|", "right"),
            ("[0.025", "right"),
            ("0.975]", "right"),
        ]:
            coef_tbl.add_column(name, justify=justify)  # type: ignore[arg-type]

        for row in self.coefs:
            t_val = row.wald.value if row.wald else 0.0
            p_val = row.wald.p_value if row.wald else 1.0
            p_style = "bold red" if p_val < 0.05 else ""
            coef_tbl.add_row(
                row.term,
                _fmt_fixed(row.est),
                _fmt_fixed(row.se),
                _fmt_fixed(t_val),
                Text(_fmt_p(p_val), style=p_style),
                _fmt_fixed(row.lci),
                _fmt_fixed(row.uci),
            )

        content = Group(
            Text(f"Modeling: {self.y}", style="dim"),
            Text(""),
            stats_grid,
            Text(""),
            coef_tbl,
        )
        yield make_panel(
            [content], title=f"GLM: {self.family} ({self.link})", obj=self, kind="estimate"
        )

    def __plain_str__(self) -> str:
        st = self.stats
        df_resid = str(st.wald.df_den if st.wald else "-")

        # Two-column stats grid — left and right labels share fixed widths so
        # values stay aligned regardless of their length.
        _L = 13  # left label width
        _V = 12  # left value width
        _R = 13  # right label width

        def _row(lbl, val, rbl="", rval=""):
            left = f"{lbl:<{_L}}: {str(val):<{_V}}"
            right = f"  {rbl:<{_R}}: {rval}" if rbl else ""
            return f"  {left}{right}".rstrip()

        lines = [
            f"GLM: {self.family} ({self.link})",
            f"  Modeling : {self.y}",
            _row("n", st.n, "DF Residuals", df_resid),
            _row("Deviance", _fmt_smart(st.deviance), "Scale", _fmt_smart(st.scale)),
            _row("AIC", _fmt_smart(st.aic), "BIC", _fmt_smart(st.bic)),
        ]
        if st.r_squared is not None:
            lines.append(
                _row(
                    "R-squared",
                    _fmt_fixed(st.r_squared),
                    "R-sq (adj)",
                    _fmt_fixed(st.r_squared_adj),
                )
            )
        if st.wald_adj:
            lines.append(
                _row(
                    "F-stat (adj)",
                    _fmt_fixed(st.wald_adj.value),
                    "Prob (F-adj)",
                    _fmt_p(st.wald_adj.p_value),
                )
            )
        lines += ["", "Coefficients:"]
        headers = ["Term", "Coef.", "Std.Err.", "t", "P>|t|", "[0.025", "0.975]"]
        rows = []
        for c in self.coefs:
            t_val = c.wald.value if c.wald else 0.0
            p_val = c.wald.p_value if c.wald else 1.0
            rows.append(
                [
                    c.term,
                    _fmt_fixed(c.est),
                    _fmt_fixed(c.se),
                    _fmt_fixed(t_val),
                    _fmt_p(p_val),
                    _fmt_fixed(c.lci),
                    _fmt_fixed(c.uci),
                ]
            )
        lines.append(render_plain_table(headers, rows))
        return "\n".join(lines)

    def __str__(self) -> str:
        return render_rich_to_str(self, width=resolve_width(self))

    def __repr__(self) -> str:
        return f"GLMFit(y={self.y!r}, family={self.family!r}, n={self.stats.n}, coefs={len(self.coefs)})"

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
