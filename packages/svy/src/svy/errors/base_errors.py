# svy/errors/base_errors.py
"""
Custom exception hierarchy for the svy library.

This module defines a structured base error with multiple renderers:
- text(): plain multi-line (default for __str__)
- ansi(): colored terminal block (optional, no extra deps)
- markdown(): for notebooks/docs
- html(): compact Tailwind-friendly fragment
- to_dict(): JSON-serializable payload
- __rich_console__: pretty panel when 'rich' is installed (optional)

Design goals:
- Short, specific `title`; details in `detail`
- Clear "expected vs got"
- Consistent fields for logging/telemetry
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import indent as _indent
from typing import Any, Mapping, Optional


@dataclass(eq=False)
class SvyError(Exception):
    """
    Base svy error with rich context + multiple renderers.
    Keep `title` short; put specifics in `detail`.
    """

    title: str
    detail: str
    code: str = "SVY_ERROR"
    where: Optional[str] = None
    param: Optional[str] = None
    got: Any = None
    expected: Any = None
    hint: Optional[str] = None
    docs_url: Optional[str] = None
    extra: Optional[Mapping[str, Any]] = None
    http_status: int = 400  # kept for potential API surfaces

    # Optional, class-level overrides (match other classes' knobs)
    _PANEL_BORDER: str | None = None
    _HEADER_STYLE: str | None = None
    _TITLE_STYLE: str | None = None

    # ---- Plain multi-line text (universal, no ANSI) ----
    def text(self, *, indent: int = 2, surround: bool = True) -> str:
        lines: list[str] = [
            f"❌ {self.title} [{self.code}]",
            f"{self.detail}",
        ]
        meta: list[str] = []
        if self.where:
            meta.append(f"- where: {self.where}")
        if self.param:
            meta.append(f"- param: {self.param}")
        if self.expected is not None:
            meta.append(f"- expected: {self.expected}")
        if self.got is not None:
            got = str(self.got)
            if len(got) > 120:
                got = got[:117] + "…"
            meta.append(f"- got: {got}")
        if meta:
            lines.append("\n".join(meta))
        if self.hint:
            lines.append(f"Hint: {self.hint}")
        if self.docs_url:
            lines.append(f"Docs: {self.docs_url}")

        body = "\n".join(lines)

        # Left padding
        if indent > 0:
            pad = " " * indent
            body = _indent(body, pad)

        # Surround with a blank line above and below
        if surround:
            return f"\n{body}\n"
        return body

    # ---- Default human string: use padded, surrounded text ----
    def __str__(self) -> str:
        return self.text(indent=2, surround=True)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        fields = []

        def add(name: str, value: Any, always: bool = False, limit: int = 120):
            if value is None and not always:
                return
            # use repr() so strings show quotes/escapes; then truncate safely
            s = repr(value)
            if len(s) > limit:
                s = s[: limit - 1] + "…"
            fields.append(f"{name}={s}")

        add("title", self.title, always=True, limit=80)
        add("detail", self.detail, always=True, limit=120)
        add("code", self.code, always=True, limit=40)
        add("where", self.where, limit=80)
        add("param", self.param, limit=60)
        add("expected", self.expected, limit=80)
        add("got", self.got, limit=80)
        add("hint", self.hint, limit=100)
        add("docs_url", self.docs_url, limit=100)
        # Keep extra/json-ish small; drop if huge
        if self.extra and len(repr(self.extra)) <= 200:
            add("extra", self.extra, limit=200)
        # http_status is useful but fixed; include only if not default
        if getattr(self, "http_status", 400) != 400:
            add("http_status", self.http_status, always=True)

        return f"{cls}({', '.join(fields)})"

    # ---- One-line summary (for compact logs if needed) ----
    def summary(self) -> str:
        parts = [f"{self.code}: {self.title} — {self.detail}"]
        if self.where:
            parts.append(f"[at {self.where}]")
        if self.param:
            parts.append(f"[param {self.param}]")
        if self.expected is not None:
            parts.append(f"[expected {self.expected}]")
        if self.got is not None:
            got = str(self.got)
            if len(got) > 120:
                got = got[:117] + "…"
            parts.append(f"[got {got}]")
        if self.hint:
            parts.append(f"→ {self.hint}")
        return " ".join(parts)

    # ---- Pretty for terminals (ANSI colors) ----
    def ansi(self) -> str:
        B, _R, Y, G, DIM, RESET = (
            "\033[1m",
            "\033[31m",
            "\033[33m",
            "\033[32m",
            "\033[2m",
            "\033[0m",
        )
        line1 = f"{B}❌ {self.title}{RESET}  {DIM}[{self.code}]{RESET}"
        line2 = f"{self.detail}"
        meta: list[str] = []
        if self.where:
            meta.append(f"{DIM}at{RESET} {self.where}")
        if self.param:
            meta.append(f"{DIM}param{RESET} {self.param}")
        if self.expected is not None:
            meta.append(f"{DIM}expected{RESET} {self.expected}")
        if self.got is not None:
            got = str(self.got)
            if len(got) > 80:
                got = got[:77] + "…"
            meta.append(f"{DIM}got{RESET} {got}")
        line3 = " · ".join(meta)
        line4 = f"{G}Hint:{RESET} {self.hint}" if self.hint else ""
        line5 = f"{Y}Docs:{RESET} {self.docs_url}" if self.docs_url else ""
        return "\n".join([line1, line2, line3, line4, line5]).strip()

    # ---- Markdown (great for README / notebooks) ----
    def markdown(self) -> str:
        # Note: Added TWO spaces between title and code to match test expectation
        out = [f"**❌ {self.title}** `[{self.code}]`", f"{self.detail}"]
        bullets: list[str] = []
        if self.where:
            bullets.append(f"- **where**: `{self.where}`")
        if self.param:
            bullets.append(f"- **param**: `{self.param}`")
        if self.expected is not None:
            bullets.append(f"- **expected**: `{self.expected}`")
        if self.got is not None:
            bullets.append(f"- **got**: `{self._short(self.got)}`")
        if bullets:
            out += bullets
        if self.hint:
            out.append(f"> 💡 **Hint:** {self.hint}")
        if self.docs_url:
            out.append(f"[Docs]({self.docs_url})")
        return "\n".join(out)

    # ---- HTML fragment (Tailwind friendly) ----
    def html(self) -> str:
        # minimal, copy-paste safe Tailwind alert
        return f"""
<div role="alert" class="rounded-xl border border-rose-200 bg-white p-4 text-rose-800 shadow-sm">
  <div class="flex items-start gap-3">
    <div class="mt-1">❌</div>
    <div class="flex-1">
      <p class="font-semibold">{self._escape(self.title)} <span class="text-slate-500">[{self.code}]</span></p>
      <p class="text-sm mt-1">{self._escape(self.detail)}</p>
      <ul class="text-sm text-slate-700 mt-2 space-y-1">
        {f'<li><span class="text-slate-500">Where:</span> {self._escape(self.where)}</li>' if self.where else ""}
        {f'<li><span class="text-slate-500">Param:</span> {self._escape(self.param)}</li>' if self.param else ""}
        {f'<li><span class="text-slate-500">Expected:</span> {self._escape(self.expected)}</li>' if self.expected is not None else ""}
        {f'<li><span class="text-slate-500">Got:</span> {self._escape(self._short(self.got))}</li>' if self.got is not None else ""}
      </ul>
      {f'<p class="text-sm mt-2"><span class="font-semibold text-teal-700">Hint:</span> {self._escape(self.hint)}</p>' if self.hint else ""}
      {f'<a href="{self._escape(self.docs_url)}" class="text-sm text-blue-700 underline mt-2 inline-block">Docs</a>' if self.docs_url else ""}
    </div>
  </div>
</div>
""".strip()

    # ---- JSON payload for programmatic use ----
    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "title": self.title,
                "detail": self.detail,
                "where": self.where,
                "param": self.param,
                "expected": self.expected,
                "got": None if self.got is None else self._short(self.got),
                "hint": self.hint,
                "docs_url": self.docs_url,
                "extra": dict(self.extra) if self.extra else None,
            }
        }

    # ---- Optional pretty rendering when Rich is installed ----
    def __rich_console__(self, console, options):
        try:
            from rich.table import Table
            from rich.text import Text

            # Try to import custom renderers from the library.
            # console.renderers is optional — fall back to plain text if absent.
            try:
                from ..console.renderers import make_panel, styles  # type: ignore[import]
            except ImportError:
                yield self.text()
                return

            s = styles(self, kind="error")  # ← use theme-driven error styles

            # Header
            header = Text("❌ ", style=s["title"])
            header.append(self.title, style=s["title"])
            header.append(f" [{self.code}]", style="dim")

            # Body
            body = Text(self.detail)

            # Meta (key/value grid)
            meta = Table.grid(padding=(0, 1))
            meta.expand = False

            def add(label: str, value: Any) -> None:
                if value is not None and str(value) != "":
                    meta.add_row(Text(label, style="dim"), Text(str(value)))

            add("where", self.where)
            add("param", self.param)
            if self.expected is not None:
                add("expected", self.expected)
            if self.got is not None:
                s_got = str(self.got)
                if len(s_got) > 120:
                    s_got = s_got[:117] + "…"
                add("got", s_got)
            if self.hint:
                add("hint", Text(self.hint, style="green"))
            if self.docs_url:
                add("docs", Text(self.docs_url, style="link " + self.docs_url))

            # Assemble inside a branded panel
            content = Table.grid()
            content.add_row(header)
            content.add_row(body)
            if len(meta.rows):
                content.add_row(meta)

            yield make_panel([content], title="", obj=self, kind="error")
        except Exception:
            # Graceful fallback if rich missing or any error
            yield self.text()

    # ---- helpers ----
    @staticmethod
    def _short(x: Any, limit: int = 120) -> Any:
        s = str(x)
        return s if len(s) <= limit else s[: limit - 1] + "…"

    @staticmethod
    def _escape(x: Any) -> str:
        if x is None:
            return ""
        return (
            str(x)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
