# src/svy/size/power.py
"""
Power given n: the inverse of the SampleSize goals.

Each goal method answers one question and stores it; the properties read it back.
"""

from __future__ import annotations

from typing import Any, Literal

from svy.core.types import DomainScalarMap, Number
from svy.size.power_goals import power_test_mean as _power_test_mean
from svy.size.power_goals import power_test_prop as _power_test_prop
from svy.size.types import PowerRow, TargetPower
from svy.ui.printing import make_panel, render_plain_table, render_rich_to_str, resolve_width
from svy.utils.experimental import experimental


_COLUMNS = ("n", "n_eff", "effect", "power")


def _fmt(x: float) -> str:
    return f"{x:.4g}"


class Power:
    """
    Power of a test at a given sample size, under simple or stratified designs.

    Like ``SampleSize``, ``Power`` holds no configuration at construction: every goal
    method takes all its parameters, and stratification is inferred from any parameter
    given as a dict. Each call replaces the goal held.

    Examples
    --------
    >>> svy.Power().test_mean(mu0=0, mu1=0.5, sigma=1, n=32).power
    0.8074304
    >>> svy.Power().test_prop(p0=0.5, p1=0.6, n=194, deff=1.5).power
    """

    _PANEL_TITLE: str = "Power"
    PRINT_WIDTH: int | None = None

    __slots__ = ("_target", "_rows")

    def __init__(self) -> None:
        self._target: TargetPower | None = None
        self._rows: list[PowerRow] = []

    # ---------- properties ----------

    @property
    def target(self) -> TargetPower | None:
        return self._target

    @property
    def rows(self) -> list[PowerRow]:
        """One row per stratum (one row, stratum None, when unstratified)."""
        return list(self._rows)

    def _by_stratum(self, field: str) -> Any:
        if not self._rows:
            return None
        if len(self._rows) == 1 and self._rows[0].stratum is None:
            return getattr(self._rows[0], field)
        return {r.stratum: getattr(r, field) for r in self._rows}

    @property
    def power(self) -> float | dict[object, float] | None:
        """The power: a number, or ``{stratum: power}`` when stratified."""
        return self._by_stratum("power")

    @property
    def effect(self) -> Number | dict[object, Number] | None:
        """The signed alt - null the power was computed at."""
        return self._by_stratum("effect")

    @property
    def n(self) -> Number | dict[object, Number] | None:
        """The units selected, as given."""
        return self._by_stratum("n")

    # ---------- goals ----------

    @experimental
    def test_mean(
        self,
        mu0: Number | DomainScalarMap,
        mu1: Number | DomainScalarMap,
        sigma: Number | DomainScalarMap,
        n: Number | DomainScalarMap,
        *,
        pop_size: Number | DomainScalarMap | None = None,
        two_sides: bool = True,
        delta: Number | DomainScalarMap = 0.0,
        method: Literal["wald"] = "wald",
        alpha: Number | DomainScalarMap = 0.05,
        deff: Number | DomainScalarMap = 1.0,
        resp_rate: Number | DomainScalarMap = 1.0,
    ) -> Power:
        """
        Power of the z test of H0: mu = mu0 when the true mean is mu1.

        ``n`` is the number of units selected: the ``Size.n`` that
        ``SampleSize().test_mean`` returns. Every other parameter means what it means
        there; ``deff``, ``resp_rate`` and ``pop_size`` are undone exactly (no rounding)
        to the SRS-equivalent ``n_eff`` the z formula uses.

        Power is returned whatever the effect: at ``epsilon <= delta`` it is at most
        ``alpha``. With ``two_sides`` and ``delta == 0`` both tails count; one-sided with
        ``delta == 0`` it is the power of the test on the side where ``mu1 - mu0`` lies.

        Raises
        ------
        SelectionError
            ``POWER_N_EXCEEDS_POPULATION`` when the respondents (``n * resp_rate``)
            outnumber ``pop_size``.
        """
        return _power_test_mean(
            self,
            mu0,
            mu1,
            sigma,
            n,
            pop_size=pop_size,
            two_sides=two_sides,
            delta=delta,
            method=method,
            alpha=alpha,
            deff=deff,
            resp_rate=resp_rate,
        )

    @experimental
    def test_prop(
        self,
        p0: Number | DomainScalarMap,
        p1: Number | DomainScalarMap,
        n: Number | DomainScalarMap,
        *,
        pop_size: Number | DomainScalarMap | None = None,
        two_sides: bool = True,
        delta: Number | DomainScalarMap = 0.0,
        method: Literal["wald", "score", "arcsine"] = "wald",
        alpha: Number | DomainScalarMap = 0.05,
        deff: Number | DomainScalarMap = 1.0,
        resp_rate: Number | DomainScalarMap = 1.0,
    ) -> Power:
        """
        Power of the test of H0: p = p0 when the true proportion is p1.

        ``method`` is ``'wald'`` (variance under the alternative), ``'score'`` (under
        the null) or ``'arcsine'``, as in ``SampleSize().test_prop``; the rest is as in
        ``test_mean``.

        Raises
        ------
        SelectionError
            ``SIZE_METHOD_MARGIN`` for a margin with ``'score'`` or ``'arcsine'``;
            ``POWER_N_EXCEEDS_POPULATION`` as in ``test_mean``.
        """
        return _power_test_prop(
            self,
            p0,
            p1,
            n,
            pop_size=pop_size,
            two_sides=two_sides,
            delta=delta,
            method=method,
            alpha=alpha,
            deff=deff,
            resp_rate=resp_rate,
        )

    # ---------- output ----------

    def to_polars(self):
        """One row per stratum: ``stratum`` (when stratified), ``n``, ``n_eff``, ``effect``, ``power``."""
        import polars as pl

        stratified = bool(self._rows) and self._rows[0].stratum is not None
        cols = (["stratum"] if stratified else []) + list(_COLUMNS)
        data = {
            c: [(str(r.stratum) if c == "stratum" else float(getattr(r, c))) for r in self._rows]
            for c in cols
        }
        return pl.DataFrame(
            data, schema={c: pl.String if c == "stratum" else pl.Float64 for c in cols}
        )

    def _table(self) -> tuple[list[str], list[list[str]]]:
        df = self.to_polars()
        rows = [
            [v if c == "stratum" else _fmt(v) for c, v in zip(df.columns, row)]
            for row in df.rows()
        ]
        return df.columns, rows

    def __plain_str__(self) -> str:
        if not self._rows:
            return "Power\n\n  <no power computed>"
        headers, rows = self._table()
        return f"Power\n\n{render_plain_table(tuple(headers), rows)}"

    def __str__(self) -> str:
        return render_rich_to_str(self, width=resolve_width(self))

    def __repr__(self) -> str:
        stratified = bool(self._rows) and self._rows[0].stratum is not None
        goal = self._target.goal if self._target is not None else "uncomputed"
        return f"Power(stratified={stratified}, strata={len(self._rows)}, goal={goal})"

    def __rich_console__(self, console, options):
        from rich import box
        from rich.table import Table
        from rich.text import Text

        if not self._rows:
            yield Text(self.__plain_str__())
            return
        headers, rows = self._table()
        table = Table(box=box.SIMPLE_HEAVY, header_style="bold", pad_edge=False, show_edge=True)
        for c in headers:
            table.add_column(c, justify="left" if c == "stratum" else "right")
        for r in rows:
            table.add_row(*r)
        yield make_panel([table], title=self._PANEL_TITLE, obj=self, kind="size")
