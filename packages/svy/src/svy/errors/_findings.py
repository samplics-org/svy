# svy/errors/_findings.py
"""Recording and raising warnings: the codes, the store behind ``sample.warnings``, the ``on_*`` rule."""

from __future__ import annotations

import builtins
import logging
import warnings as _pywarnings

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Iterator, Sequence

import msgspec

from ._base import Severity, SvyError, SvyWarning
from ._codes import Code


log = logging.getLogger(__name__)


# ---------- Aggregation / escalation ----------
@dataclass(eq=False)
class SvyWarningsError(SvyError):
    """
    Escalation exception that keeps your formatting surfaces.
    """

    _DEFAULT_CODE: ClassVar[str] = Code.SVY_WARNINGS_ERROR

    @classmethod
    def from_warnings(
        cls,
        warnings: Sequence[SvyWarning],
        *,
        where: str | None = None,
        code: str = "SVY_WARNINGS_ERROR",
    ) -> "SvyWarningsError":
        if not warnings:
            return cls.RuntimeError(
                title="No warnings to escalate",
                detail="SvyWarningsError.from_warnings called with an empty list.",
                code=code,
                where=where,
            )
        lines = []
        for w in warnings[:5]:
            loc = f" at {w.where}" if w.where else ""
            lines.append(f"- {w.level.name} {w.code}{loc}: {w.title} — {w.detail}")
        more = "" if len(warnings) <= 5 else f"\n… and {len(warnings) - 5} more"
        detail = "Escalated warnings:\n" + "\n".join(lines) + more
        return cls.RuntimeError(
            title=f"{len(warnings)} warning(s) escalated",
            detail=detail,
            code=code,
            where=where,
            extra={"warnings": [w.to_dict() for w in warnings]},
        )


class WarningStore:
    """
    Attach this to Sample._warnings.
    - De-dupes by SvyWarning.key()
    - Optional per-code cap to avoid floods
    """

    __slots__ = ("_items", "_seen", "_per_code", "_dedupe", "_max_per_code")

    def __init__(self, *, dedupe: bool = True, max_per_code: int | None = 100) -> None:
        self._items: list[SvyWarning] = []
        self._seen: set[tuple[Any, ...]] = set()
        self._per_code: dict[str, int] = {}
        self._dedupe = dedupe
        self._max_per_code = max_per_code

    # --- mutation ---
    def add(self, w: SvyWarning) -> bool:
        """Store ``w``; False when it repeats a stored one or its code is capped."""
        if self._dedupe:
            k = w.key()
            if k in self._seen:
                return False
            self._seen.add(k)

        if self._max_per_code is not None:
            key = str(w.code)
            n = self._per_code.get(key, 0)
            if n >= self._max_per_code:
                if n == self._max_per_code:
                    log.info(f"Supressing further warnings for code: {w.code}")
                    self._per_code[key] = n + 1
                return False
            self._per_code[key] = n + 1

        self._items.append(w)

        log_msg = f"[{w.code}] {w.title} - {w.detail} (var={w.var})"

        if w.level >= Severity.WARNING:
            log.warning(log_msg, extra=w.to_dict())
        else:
            log.info(log_msg, extra=w.to_dict())
        return True

    def extend(self, warnings: Iterable[SvyWarning]) -> None:
        for w in warnings:
            self.add(w)

    def clear(self) -> None:
        self._items.clear()
        self._seen.clear()
        self._per_code.clear()

    # --- querying ---
    def list(
        self,
        *,
        min_level: Severity | None = None,
        code: str | None = None,
        where_contains: str | None = None,
    ) -> builtins.list[SvyWarning]:
        out = self._items
        if min_level is not None:
            out = [w for w in out if w.level >= min_level]
        if code is not None:
            out = [w for w in out if str(w.code) == str(code)]
        if where_contains:
            out = [w for w in out if (w.where and where_contains in w.where)]
        return out

    def counts(self) -> dict[str, int]:
        return {
            "INFO": sum(1 for w in self._items if w.level == Severity.INFO),
            "WARNING": sum(1 for w in self._items if w.level == Severity.WARNING),
            "TOTAL": len(self._items),
        }

    # --- escalation ---
    def escalate(self, *, min_level: Severity = Severity.WARNING) -> None:
        sel = [w for w in self._items if w.level >= min_level]
        if sel:
            raise SvyWarningsError.from_warnings(sel)

    # --- export ---
    def to_dicts(self) -> builtins.list[dict[str, Any]]:
        return [w.to_dict() for w in self._items]

    def to_json(self) -> bytes:
        return msgspec.json.encode(self.to_dicts())

    def to_polars(self):
        try:
            import polars as pl
        except Exception:  # pragma: no cover
            raise RuntimeError("Polars is not available")
        rows = self.to_dicts()
        return pl.from_dicts(rows)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    # --- display: one labelled block per finding; print a finding for its full panel ---
    _LABELS = ("Level", "Code", "Detail")

    def _title(self) -> str:
        c = self.counts()
        parts = [f"{c[k]} {k.lower()}" for k in ("WARNING", "INFO") if c[k]]
        return f"Findings ({', '.join(parts)})" if parts else "Findings"

    def __plain_str__(self) -> str:
        from svy.ui.printing import text_width, wrap_labelled

        if not self._items:
            return "No findings."
        label_width = max(map(len, self._LABELS))
        width = text_width(self) - 2
        blocks = []
        for w in self._items:
            lines = []
            for label, value in zip(self._LABELS, (w.level.name, str(w.code), w.detail)):
                lines += wrap_labelled(label, value, label_width=label_width, width=width)
            blocks.append("\n".join(f"  {line}" for line in lines))
        return f"{self._title()}\n\n" + "\n\n".join(blocks)

    def __str__(self) -> str:
        if not self._items:
            return "No findings."
        try:
            from svy.ui.printing import render_rich_to_str, text_width

            return render_rich_to_str(self, width=text_width(self))
        except Exception:
            return self.__plain_str__()

    # The REPL shows a bare `sample.warnings` through its repr.
    __repr__ = __str__

    def __rich_console__(self, console, options):
        from rich.constrain import Constrain
        from rich.table import Table
        from rich.text import Text

        from svy.ui.printing import make_panel, styles, text_width

        if not self._items:
            yield Text("No findings.", style="dim")
            return
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold", no_wrap=True)
        grid.add_column(overflow="fold")
        for i, w in enumerate(self._items):
            if i:
                grid.add_row("", "")
            kind = w.level.name.lower()
            level, code, detail = self._LABELS
            grid.add_row(level, Text(w.level.name, style=styles(kind=kind)["title"]))
            grid.add_row(code, str(w.code))
            grid.add_row(detail, w.detail)
        panel = make_panel([grid], title=self._title(), obj=self, kind="panel")
        yield Constrain(panel, width=text_width(self))


# ---------- One rule for findings ----------
#: The sample a finding raised deep inside a design operation belongs to, set
#: by the Sample methods that run such operations (``update_design``,
#: ``use_weight``). Without one, the finding is only raised as a warning.
_FINDINGS_SAMPLE: ContextVar[Any] = ContextVar("svy_findings_sample", default=None)


@contextmanager
def findings_to(sample: Any) -> Iterator[None]:
    token = _FINDINGS_SAMPLE.set(sample)
    try:
        yield
    finally:
        _FINDINGS_SAMPLE.reset(token)


def emit_finding(finding: SvyWarning) -> None:
    """Record ``finding`` on the sample at hand (which also raises it), or, with
    no sample, raise it at the caller's line."""
    sample = _FINDINGS_SAMPLE.get()
    if sample is not None:
        sample.warn(finding)
        return
    from svy.core.design import _user_stacklevel

    _pywarnings.warn(finding, stacklevel=_user_stacklevel())


# ---------- on_* parameters ----------
#: Every ``on_*`` parameter takes these, with one meaning: "error" raises and
#: leaves the sample as it was; "warn" records the finding and raises it once;
#: "ignore" records it at INFO, which is kept but not raised.
ON_FINDING: tuple[str, ...] = ("error", "warn", "ignore")


def check_on_finding(value: Any, *, param: str, where: str) -> None:
    if not isinstance(value, str) or value not in ON_FINDING:
        from ._api import UsageError

        raise UsageError.invalid_choice(
            where=where,
            param=param,
            got=value,
            allowed=ON_FINDING,
            hint=(
                f'{param}="error" raises, "warn" records the finding and raises it once, '
                '"ignore" records it without raising.'
            ),
        )


def finding_level(on: str) -> Severity:
    """The level a finding is recorded at under an ``on_*`` choice."""
    return Severity.WARNING if on == "warn" else Severity.INFO
