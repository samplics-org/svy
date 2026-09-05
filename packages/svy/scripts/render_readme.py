"""Execute the Python fences of a README and refresh the outputs shown next to them.

A fence's last ``print(...)`` argument is rendered into whatever output slot follows
the fence: a bare (or ``text``) code fence gets the plain-text rendering, an image
line pointing at an ``.svg`` gets a Rich SVG export. Fences share one namespace, so
later snippets can reuse objects built earlier.

    uv run python scripts/render_readme.py            # rewrite outputs in place
    uv run python scripts/render_readme.py --check    # exit 1 if anything drifted
"""

from __future__ import annotations

import argparse
import html
import io
import re
import sys

from dataclasses import dataclass, field
from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parents[1]
README_FILES = (PKG_ROOT.parents[1] / "README.md", PKG_ROOT / "README.md")

_IMG_RE = re.compile(r"^!\[[^\]]*\]\((?P<path>[^)]+\.svg)\)\s*$")
_SVG_TEXT_RE = re.compile(r"<text[^>]*>(.*?)</text>", re.S)


@dataclass
class Snippet:
    code: str
    text_slot: tuple[int, int] | None = None  # line range of the output fence body
    svg_path: Path | None = None
    printed: list = field(default_factory=list)


def parse(readme: Path) -> tuple[list[str], list[Snippet]]:
    lines = readme.read_text(encoding="utf-8").splitlines()
    snippets: list[Snippet] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "```python":
            i += 1
            continue
        start = i + 1
        while lines[i].strip() != "```" or i == start - 1:
            i += 1
        snippet = Snippet(code="\n".join(lines[start:i]))
        i += 1
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            head = lines[j].strip()
            if head in ("```", "```text"):
                end = j + 1
                while lines[end].strip() != "```":
                    end += 1
                snippet.text_slot = (j + 1, end)
                i = end + 1
            elif m := _IMG_RE.match(head):
                snippet.svg_path = (readme.parent / m.group("path")).resolve()
                i = j + 1
        snippets.append(snippet)
    return lines, snippets


def execute(snippets: list[Snippet]) -> None:
    ns: dict = {}
    for snippet in snippets:
        ns["print"] = lambda *args, _s=snippet, **kw: _s.printed.extend(args)
        exec(compile(snippet.code, "<readme>", "exec"), ns)


def _console(width: int | None, *, color: bool):
    from rich.console import Console

    return Console(
        record=True,
        width=width,
        file=io.StringIO(),
        force_terminal=color,
        color_system="truecolor" if color else None,
        emoji=False,
        soft_wrap=False,
    )


def render_text(obj) -> list[str]:
    from svy.ui.printing import resolve_width

    console = _console(resolve_width(obj), color=False)
    console.print(obj)
    return [ln.rstrip() for ln in console.export_text().rstrip("\n").splitlines()]


def render_svg(obj) -> str:
    width = max(len(ln) for ln in render_text(obj))
    console = _console(width, color=True)
    console.print(obj)
    svg = console.export_svg(title="svy")
    # Rich emits only a viewBox; without intrinsic dimensions GitHub stretches the image.
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    size = f'width="{round(float(m[1]))}" height="{round(float(m[2]))}" '
    return svg.replace("viewBox=", size + "viewBox=", 1)


def svg_text(svg: str) -> str:
    chunks = (html.unescape(t) for t in _SVG_TEXT_RE.findall(svg))
    return re.sub(r"\s+", "", "".join(chunks))


def refresh(readme: Path, *, check: bool) -> list[str]:
    lines, snippets = parse(readme)
    execute(snippets)
    problems: list[str] = []
    for snippet in reversed(snippets):  # bottom-up so line indices stay valid
        if not snippet.printed or (snippet.text_slot is None and snippet.svg_path is None):
            continue
        obj = snippet.printed[-1]
        if snippet.text_slot is not None:
            a, b = snippet.text_slot
            fresh = render_text(obj)
            if lines[a:b] != fresh:
                problems.append(f"{readme}: output block after line {a} is stale")
                lines[a:b] = fresh
        if snippet.svg_path is not None:
            fresh = render_svg(obj)
            current = snippet.svg_path.read_text(encoding="utf-8")
            if svg_text(current) != svg_text(fresh):
                problems.append(f"{snippet.svg_path}: stale")
                if not check:
                    snippet.svg_path.write_text(fresh, encoding="utf-8")
    if problems and not check:
        readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report drift instead of rewriting")
    ap.add_argument("readme", nargs="*", type=Path, default=list(README_FILES))
    args = ap.parse_args(argv)
    problems = [p for readme in args.readme for p in refresh(readme, check=args.check)]
    for p in problems:
        print(p, file=sys.stderr)
    if not problems:
        print("README outputs are up to date")
    return 1 if (problems and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
