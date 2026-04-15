# src/svy/regression/prediction.py
"""
Prediction results for fitted GLM models.
"""

from __future__ import annotations

import logging

from typing import ClassVar

import msgspec
import numpy as np
import polars as pl

from svy.ui.printing import make_panel, render_rich_to_str, resolve_width


log = logging.getLogger(__name__)


class GLMPred(msgspec.Struct, frozen=True):
    """Prediction results from a fitted GLM."""

    PRINT_WIDTH: ClassVar[int | None] = None

    yhat: np.ndarray
    se: np.ndarray
    lci: np.ndarray
    uci: np.ndarray
    df: float
    alpha: float
    residuals: np.ndarray | None = None

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
        """Convert predictions to DataFrame."""
        data = {
            "yhat": self.yhat,
            "se": self.se,
            "lci": self.lci,
            "uci": self.uci,
        }
        if self.residuals is not None:
            data["residuals"] = self.residuals
        return pl.DataFrame(data)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {
            "yhat": self.yhat.tolist(),
            "se": self.se.tolist(),
            "lci": self.lci.tolist(),
            "uci": self.uci.tolist(),
            "df": self.df,
            "alpha": self.alpha,
        }
        if self.residuals is not None:
            d["residuals"] = self.residuals.tolist()
        return d

    def __len__(self) -> int:
        return len(self.yhat)

    def __str__(self) -> str:
        return render_rich_to_str(self, width=resolve_width(self))

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed. Never calls str(self)."""
        conf_pct = int(self.conf_level * 100)

        _L, _V, _R = 10, 10, 10

        def _row(lbl, val, rbl="", rval=""):
            l = f"{lbl:<{_L}}: {str(val):<{_V}}"
            r = f"  {rbl:<{_R}}: {rval}" if rbl else ""
            return f"  {l}{r}".rstrip()

        lines = [
            f"GLM Predictions ({conf_pct}% CI)",
            _row("n", len(self), "DF", f"{self.df:.1f}"),
            _row("Mean ŷ", f"{self.yhat.mean():.4f}", "Mean SE", f"{self.se.mean():.4f}"),
            _row("Min ŷ", f"{self.yhat.min():.4f}", "Max ŷ", f"{self.yhat.max():.4f}"),
        ]
        if self.residuals is not None:
            lines.append(
                _row(
                    "Mean resid",
                    f"{self.residuals.mean():.4f}",
                    "Std resid",
                    f"{self.residuals.std():.4f}",
                )
            )
        lines += ["", "  Use .to_polars() for full results"]
        return "\n".join(lines)

    def __repr__(self) -> str:
        conf_pct = int(self.conf_level * 100)
        res_str = ", with residuals" if self.residuals is not None else ""
        return f"GLMPred(n={len(self)}, {conf_pct}% CI, df={self.df:.1f}{res_str})"

    def __rich_console__(self, console, options):
        from rich.table import Table as RTable
        from rich.text import Text

        conf_pct = int(self.conf_level * 100)

        grid = RTable.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column(justify="right")
        grid.add_column(style="bold")
        grid.add_column(justify="right")

        grid.add_row("n", str(len(self)), "DF", f"{self.df:.1f}")
        grid.add_row("Mean ŷ", f"{self.yhat.mean():.4f}", "Mean SE", f"{self.se.mean():.4f}")
        grid.add_row("Min ŷ", f"{self.yhat.min():.4f}", "Max ŷ", f"{self.yhat.max():.4f}")

        if self.residuals is not None:
            grid.add_row(
                "Mean resid",
                f"{self.residuals.mean():.4f}",
                "Std resid",
                f"{self.residuals.std():.4f}",
            )

        children = [grid, Text(""), Text("Use .to_polars() for full results", style="dim")]
        title = f"GLM Predictions ({conf_pct}% CI)"
        yield make_panel(children, title=title, obj=self, kind="estimate")

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
