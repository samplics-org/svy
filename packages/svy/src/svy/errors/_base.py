# svy/errors/_base.py
"""What errors and warnings share: the fields and every renderer.

- ``SvyError`` stops the call; ``SvyWarning`` is recorded on the sample and raised as the Python warning.
- Renderers: ``text``, ``ansi``, ``markdown``, ``html``, ``to_dict`` (one shape for both), rich panel.
"""

from __future__ import annotations

import datetime as _dt

from dataclasses import dataclass, field
from enum import IntEnum
from textwrap import indent as _indent
from typing import Any, ClassVar, Mapping, Optional

from ._common import _Constructors


# svy's text form of a key over several columns (e.g. weighting cells), used
# when such a key has to become a JSON object key.
_KEY_SEP = "_&_"


def _json_key(k: Any) -> Any:
    if k is None or isinstance(k, (str, int, float, bool)):
        return k
    if isinstance(k, tuple):
        return _KEY_SEP.join(str(_jsonable(p)) for p in k)
    j = _jsonable(k)
    return j if isinstance(j, (str, int, float, bool)) else str(j)


def _jsonable(x: Any) -> Any:
    """``x`` with every value in a form ``json.dumps`` accepts, structure kept."""
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    if isinstance(x, Mapping):
        return {_json_key(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        items = list(x)
        if isinstance(x, (set, frozenset)):
            try:
                items = sorted(items)
            except TypeError:
                pass
        return [_jsonable(v) for v in items]
    if isinstance(x, (_dt.date, _dt.time)):
        return x.isoformat()
    item = getattr(x, "item", None)  # numpy scalars
    if callable(item):
        try:
            v = item()
        except (TypeError, ValueError):
            v = None
        if v is not None and v is not x:
            return _jsonable(v)
    tolist = getattr(x, "tolist", None)  # numpy arrays
    if callable(tolist):
        return _jsonable(tolist())
    return str(x)


def _rebuild(cls: type[Any], state: dict[str, Any]) -> Any:
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


class Severity(IntEnum):
    INFO = 10  # recorded, not raised
    WARNING = 20


@dataclass(eq=False)
class _Report:
    """The fields and renderers errors and warnings share.

    Keep `title` short; put specifics in `detail`.
    """

    title: str
    detail: str
    code: str = ""
    where: Optional[str] = None
    param: Optional[str] = None
    got: Any = None
    expected: Any = None
    hint: Optional[str] = None
    docs_url: Optional[str] = None
    extra: Optional[Mapping[str, Any]] = None

    # Optional, class-level overrides (match other classes' knobs)
    _PANEL_BORDER: ClassVar[str | None] = None
    _HEADER_STYLE: ClassVar[str | None] = None
    _TITLE_STYLE: ClassVar[str | None] = None

    # Marker and panel theme; SvyWarning sets them from its level.
    _MARK: ClassVar[str] = "❌"
    _MARK_RICH: ClassVar[str] = "✗"
    _KIND: ClassVar[str] = "error"

    def _level_name(self) -> str:
        return "ERROR"

    # Exceptions are rebuilt from `args` by default, which these never set.
    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (type(self), dict(self.__dict__)))

    # ---- Plain multi-line text (universal, no ANSI) ----
    def text(self, *, indent: int = 2, surround: bool = True) -> str:
        lines: list[str] = [
            f"{self._MARK} {self.title} [{self.code}]",
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
        status = getattr(self, "http_status", 400)
        if status != 400:
            add("http_status", status, always=True)

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
        line1 = f"{B}{self._MARK} {self.title}{RESET}  {DIM}[{self.code}]{RESET}"
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
        out = [f"**{self._MARK} {self.title}** `[{self.code}]`", f"{self.detail}"]
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
    <div class="mt-1">{self._MARK}</div>
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

    # ---- JSON payload for programmatic use: one shape for errors and warnings ----
    def to_dict(self) -> dict[str, Any]:
        return {
            # The class with its kind, "ModelError.ValueError", so callers can tell areas apart.
            "type": type(self).__qualname__,
            "level": self._level_name(),
            "code": str(self.code),
            "title": self.title,
            "detail": self.detail,
            "where": self.where,
            "param": self.param,
            "expected": self._payload(self.expected),
            "got": self._payload(self.got),
            "hint": self.hint,
            "docs_url": self.docs_url,
            "extra": _jsonable(dict(self.extra)) if self.extra else None,
        }

    # ---- Optional pretty rendering when Rich is installed ----
    def __rich_console__(self, console, options):
        try:
            from rich.table import Table
            from rich.text import Text

            # Try to import the shared Rich renderers from the library.
            # Optional — fall back to plain text if absent.
            try:
                from ..ui.printing import make_panel, styles  # type: ignore[import]
            except ImportError:
                yield self.text()
                return

            s = styles(self, kind=self._KIND)

            # Header. Use a width-1 marker (✗) rather than an emoji: emoji
            # occupy two terminal cells but render ~1 in HTML/browser fonts,
            # which knocks the panel's right border out of alignment in docs.
            header = Text(f"{self._MARK_RICH} ", style=s["title"])
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

            # Assemble inside a branded panel, with blank rows separating the
            # title, body, and metadata so the panel breathes.
            content = Table.grid()
            content.add_row(header)
            content.add_row(Text(""))
            content.add_row(body)
            if len(meta.rows):
                content.add_row(Text(""))
                content.add_row(meta)

            from rich.constrain import Constrain
            from rich.padding import Padding

            from ..ui.printing import text_width

            panel = make_panel([Padding(content, (1, 1))], title="", obj=self, kind=self._KIND)
            yield Constrain(panel, width=text_width(self))
        except Exception:
            # Graceful fallback if rich missing or any error
            yield self.text()

    # ---- helpers ----
    @classmethod
    def _payload(cls, x: Any) -> Any:
        # Structured values stay structured; only prose is shortened.
        if x is None:
            return None
        if isinstance(x, str):
            return cls._short(x)
        return _jsonable(x)

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


# The builtin each error also is: ``ModelError.ValueError`` is caught as a
# ModelError, as a ValueError and as a SvyError.
_KINDS: tuple[type[Exception], ...] = (
    TypeError,
    ValueError,
    KeyError,
    AttributeError,
    NotImplementedError,
    RuntimeError,
    ImportError,
    OSError,
    FileNotFoundError,
    PermissionError,
)


def _add_kinds(cls: type[SvyError]) -> None:
    for kind in _KINDS:
        name = kind.__name__
        # A real attribute path, so tracebacks read ``svy.errors.ModelError.ValueError``
        # and pickle finds the class again.
        pair = type(
            name,
            (cls, kind),
            {
                "__module__": cls.__module__,
                "__qualname__": f"{cls.__qualname__}.{name}",
                "__doc__": f"A {cls.__name__} that is also a {name}.",
                "_IS_KIND": True,
            },
        )
        setattr(cls, name, pair)


@dataclass(eq=False)
class SvyError(_Report, _Constructors, Exception):
    """Base svy error with rich context + multiple renderers.

    Raise an area's kind, ``EstimationError.ValueError(...)``, or a constructor
    that picks it, ``EstimationError.missing_columns(...)``.
    """

    code: str = "SVY_ERROR"
    http_status: int = 400  # kept for potential API surfaces

    _DEFAULT_CODE: ClassVar[str] = "SVY_ERROR"
    # True on a class that already is one builtin, so it gets no kinds of its own.
    _IS_KIND: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("_IS_KIND", False):
            _add_kinds(cls)

    def __post_init__(self) -> None:
        if self.code == "SVY_ERROR":
            self.code = self._DEFAULT_CODE


_add_kinds(SvyError)


@dataclass(eq=False)
class SvyWarning(_Report, UserWarning):
    """A finding: svy reports it without stopping.

    The same instance is recorded in ``sample.warnings`` and raised as the Python
    warning, so a caught warning carries every field. Silence them all with
    ``warnings.filterwarnings("ignore", category=svy.SvyWarning)``.
    """

    code: str = "SVY_WARNING"
    level: Severity = Severity.WARNING
    var: Optional[str] = None
    rows: Optional[tuple[int, ...]] = None
    # The sample state (its data version) the finding describes: the same
    # finding on a new state is a new event.
    state: Optional[int] = None
    ts: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc))

    @property
    def _MARK(self) -> str:
        return "⚠️" if self.level >= Severity.WARNING else "ℹ️"

    @property
    def _MARK_RICH(self) -> str:
        return "▲" if self.level >= Severity.WARNING else "ℹ"

    @property
    def _KIND(self) -> str:
        return "warning" if self.level >= Severity.WARNING else "info"

    def __post_init__(self) -> None:
        if self.rows is not None and not isinstance(self.rows, tuple):
            self.rows = tuple(self.rows)

    def _level_name(self) -> str:
        return self.level.name

    # Python prints and filters a warning by its str, matched from the start:
    # filters such as ``ignore:\[CODE\]`` rely on this line.
    def __str__(self) -> str:
        return (
            f"[{self.code}] {self.title}: {self.detail}"
            if self.detail
            else f"[{self.code}] {self.title}"
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["var"] = self.var
        d["rows"] = None if self.rows is None else list(self.rows)
        d["ts"] = self.ts.isoformat()
        return d

    def key(self) -> tuple[Any, ...]:
        """What makes two findings the same event, for de-duplication."""
        return (str(self.code), self.where, self.param, self.var, self.detail, self.state)
