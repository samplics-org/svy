# src/svy/utils/pretty.py
from __future__ import annotations

import logging
import os
import sys

from typing import IO, Any, Optional


log = logging.getLogger(__name__)


def _env_flag(name: str) -> bool | None:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return None


def _tty(stream) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


def _should_use_rich(stream) -> bool:
    # explicit override wins
    ov = _env_flag("SVY_USE_RICH")
    if ov is not None:
        return ov
    # avoid color in CI/dumb/file
    if os.getenv("CI") or os.getenv("NO_COLOR") or os.getenv("TERM") == "dumb":
        return False
    return _tty(stream)


# Public alias used by logconfig.py
want_rich_default = _should_use_rich


def print_error(err: Any, *, stream=None, prefer_rich: bool | str = "auto") -> None:
    """
    Print an error using Rich if available/desired, else plain text.
    - prefer_rich=True: force Rich if importable
    - prefer_rich=False: force plain
    - prefer_rich="auto": TTY + env heuristics
    """
    stream = stream or sys.stderr
    use_rich = _should_use_rich(stream) if prefer_rich == "auto" else bool(prefer_rich)
    if use_rich:
        try:
            from rich.console import Console

            Console(file=stream).print(err)
            return
        except Exception:
            pass  # fall back to plain
    print(str(err), file=stream)


def should_use_rich(stream: Optional[IO[str]] = None) -> bool:
    return _should_use_rich(stream or sys.stderr)
