# src/svy/core/check.py
"""Reports of ``Sample.check()``."""

from __future__ import annotations

from typing import Any, ClassVar

import msgspec


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            return str(x)
        if x.is_integer() and abs(x) < 1e15:
            return f"{x:,.0f}"
        return f"{x:,.4f}" if abs(x) >= 1 else f"{x:.6g}"
    return str(x)


def _key(x: Any) -> str:
    return "(" + ", ".join(map(repr, x)) + ")" if isinstance(x, tuple) else repr(x)


class _Report(msgspec.Struct, frozen=True, kw_only=True):
    PRINT_WIDTH: ClassVar[int | None] = None

    def _title(self) -> str:
        raise NotImplementedError

    def _rows(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def __rich_console__(self, console, options):
        from rich.table import Table as RTable

        from svy.ui.printing import make_panel

        t = RTable(show_header=False, box=None, show_edge=False, pad_edge=False, padding=(0, 2))
        t.add_column("Field", justify="left", no_wrap=True, style="bold")
        t.add_column("Value", justify="left", overflow="fold", max_width=70)
        for label, value in self._rows():
            t.add_row(label, value)
        yield make_panel([t], title=self._title(), obj=self, kind="panel")

    def __plain_str__(self) -> str:
        rows = self._rows()
        width = max((len(label) for label, _ in rows), default=0)
        return "\n".join(
            [self._title(), *(f"  {label:<{width}} : {value}" for label, value in rows)]
        )

    def __str__(self) -> str:
        from svy.ui.printing import render_rich_to_str, resolve_width

        try:
            return render_rich_to_str(self, width=resolve_width(self))
        except Exception:
            return self.__plain_str__()

    def __repr__(self) -> str:
        return self.__plain_str__()


class WeightCheck(_Report, frozen=True, kw_only=True):
    """Counts and summaries of a weight column.

    ``n_null``, ``n_nonfinite`` (NaN, +inf, -inf), ``n_negative``, ``n_zero``
    and ``n_positive`` partition the ``n`` rows. The summaries are over the
    ``n_positive`` positive finite weights and are None when there are none:
    ``ratio`` is ``max / min``, ``deff`` the Kish design effect due to
    weighting ``n * sum(w^2) / sum(w)^2`` and ``ess`` the effective sample
    size ``sum(w)^2 / sum(w^2)``.
    """

    wgt: str
    n: int
    n_null: int
    n_nonfinite: int
    n_negative: int
    n_zero: int
    n_positive: int
    min: float | None
    max: float | None
    ratio: float | None
    sum: float | None
    mean: float | None
    deff: float | None
    ess: float | None

    def _title(self) -> str:
        return f"Weight check: {self.wgt}"

    def _rows(self) -> list[tuple[str, str]]:
        return [
            ("Rows", _fmt(self.n)),
            ("Null", _fmt(self.n_null)),
            ("Non-finite", _fmt(self.n_nonfinite)),
            ("Negative", _fmt(self.n_negative)),
            ("Zero", _fmt(self.n_zero)),
            ("Positive", _fmt(self.n_positive)),
            ("Min", _fmt(self.min)),
            ("Max", _fmt(self.max)),
            ("Max / min", _fmt(self.ratio)),
            ("Sum", _fmt(self.sum)),
            ("Mean", _fmt(self.mean)),
            ("Kish deff", _fmt(self.deff)),
            ("Effective n", _fmt(self.ess)),
        ]


class KeyCheck(_Report, frozen=True, kw_only=True):
    """Uniqueness of a record key.

    Rows with a null in any key column are counted in ``n_null`` and left out
    of the duplicate counts. ``n_duplicated`` is the number of key values held
    by more than one row, ``n_rows_duplicated`` the rows holding them, and
    ``examples`` the first of those values in key order: values for a
    one-column key, tuples for a key on several columns.
    """

    columns: list[str]
    n: int
    n_null: int
    n_duplicated: int
    n_rows_duplicated: int
    examples: list[Any]

    def _title(self) -> str:
        return f"Key check: {', '.join(self.columns)}"

    def _rows(self) -> list[tuple[str, str]]:
        rows = [
            ("Rows", _fmt(self.n)),
            ("Null key", _fmt(self.n_null)),
            ("Duplicated keys", _fmt(self.n_duplicated)),
            ("Rows in duplicates", _fmt(self.n_rows_duplicated)),
        ]
        if self.examples:
            rows.append(("Examples", ", ".join(_key(e) for e in self.examples)))
        return rows


class NestingCheck(_Report, frozen=True, kw_only=True):
    """PSU codes that appear in more than one stratum.

    ``n_strata`` and ``n_psus`` count distinct stratum and PSU codes.
    ``n_psus_across_strata`` counts the PSU codes found in several strata and
    ``examples`` lists the first of them as ``(psu, [strata...])``: values for
    one column, tuples for several.
    """

    stratum: list[str]
    psu: list[str]
    n: int
    n_strata: int
    n_psus: int
    n_psus_across_strata: int
    examples: list[tuple[Any, list[Any]]]

    def _title(self) -> str:
        return f"Nesting check: {', '.join(self.psu)} in {', '.join(self.stratum)}"

    def _rows(self) -> list[tuple[str, str]]:
        rows = [
            ("Rows", _fmt(self.n)),
            ("Strata", _fmt(self.n_strata)),
            ("PSU codes", _fmt(self.n_psus)),
            ("PSU codes in several strata", _fmt(self.n_psus_across_strata)),
        ]
        for i, (psu, strata) in enumerate(self.examples):
            label = "Examples" if i == 0 else ""
            rows.append((label, f"{_key(psu)} in {', '.join(_key(s) for s in strata)}"))
        return rows


class MarginCheck(_Report, frozen=True, kw_only=True):
    """Whether raking margins describe the same population total.

    ``max_rel_diff`` is ``(max - min) / max`` over the margin totals; the
    margins agree when it is at most ``rtol``.
    """

    totals: dict[str, float]
    agree: bool
    max_rel_diff: float
    rtol: float

    def _title(self) -> str:
        return "Margin check"

    def _rows(self) -> list[tuple[str, str]]:
        rows = [(f"Total {m}", _fmt(t)) for m, t in self.totals.items()]
        rows += [
            ("Agree", _fmt(self.agree)),
            ("Largest relative difference", f"{self.max_rel_diff:.3g}"),
            ("Tolerance", f"{self.rtol:.3g}"),
        ]
        return rows


class SingletonCheck(_Report, frozen=True, kw_only=True):
    """Strata with a single PSU and whether the design's rule handles them.

    ``n_unhandled`` counts those the rule (``svy.Singleton``) does not handle,
    all of them without a rule: the next Taylor analysis raises on these.
    ``examples`` lists the first singleton strata.
    """

    n_singletons: int
    n_unhandled: int
    examples: list[Any]

    def _title(self) -> str:
        return "Singleton check"

    def _rows(self) -> list[tuple[str, str]]:
        rows = [
            ("Strata with one PSU", _fmt(self.n_singletons)),
            ("Not handled by a rule", _fmt(self.n_unhandled)),
        ]
        if self.examples:
            rows.append(("Examples", ", ".join(_key(s) for s in self.examples)))
        return rows


class SampleCheck(msgspec.Struct, frozen=True, kw_only=True):
    """What ``Sample.check()`` found in the data against its design.

    One section per part of the design; a section is None when the design
    does not declare that part (no weight, no case id, no strata and PSUs).
    """

    weights: WeightCheck | None = None
    case_id: KeyCheck | None = None
    nesting: NestingCheck | None = None
    singletons: SingletonCheck | None = None

    def _sections(self) -> list[_Report]:
        return [s for s in (self.weights, self.case_id, self.nesting, self.singletons) if s]

    def __rich_console__(self, console, options):
        for section in self._sections():
            yield from section.__rich_console__(console, options)

    def __plain_str__(self) -> str:
        parts = [s.__plain_str__() for s in self._sections()]
        return "\n\n".join(parts) if parts else "Sample check: nothing declared to check"

    def __str__(self) -> str:
        from svy.ui.printing import render_rich_to_str, resolve_width

        if not self._sections():
            return self.__plain_str__()
        try:
            return render_rich_to_str(self, width=resolve_width(self))
        except Exception:
            return self.__plain_str__()

    def __repr__(self) -> str:
        return self.__plain_str__()
