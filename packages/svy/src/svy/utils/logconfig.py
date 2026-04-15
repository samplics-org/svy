# svy/utils/logconfig.py
from __future__ import annotations

import logging
import sys

from contextlib import contextmanager
from typing import Literal

from .pretty import should_use_rich


def install_pretty_logging(
    level: int = logging.INFO,
    use_rich: bool | str = "auto",
    *,
    replace_handlers: bool = False,
    logger_name: str | None = None,
) -> logging.Handler:
    """
    Configure logging nicely and return the active handler.

    - If replace_handlers=True, existing handlers on the target logger are removed.
    - If handlers already exist and replace_handlers=False, they are kept and the first existing
      handler is returned (no new handler is added).
    """
    target = logging.getLogger(logger_name) if logger_name else logging.getLogger()

    if replace_handlers:
        target.handlers.clear()

    # If handlers exist and we aren’t replacing them, respect them
    if target.handlers and not replace_handlers:
        if level is not None:
            target.setLevel(level)
        return target.handlers[0]

    # We will install exactly one handler
    target.setLevel(level)

    want_rich = should_use_rich(sys.stderr) if use_rich == "auto" else bool(use_rich)

    handler: logging.Handler
    if want_rich:
        try:
            from rich.logging import RichHandler
            from rich.traceback import install as install_tb

            install_tb(show_locals=False, width=120, extra_lines=1)
            handler = RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                enable_link_path=False,
                log_time_format="[%X]",
            )
        except Exception:
            # Fall back to plain if Rich import/config fails
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    target.addHandler(handler)

    # If configuring a specific logger, avoid duplicate emission via root
    if logger_name:
        target.propagate = False

    return handler


def enable_logging(
    level: int = logging.INFO,
    *,
    use_rich: bool | Literal["auto"] = "auto",
    replace_handlers: bool = False,
    logger_name: str | None = None,
) -> logging.Handler:
    """
    Convenience entry point for apps/notebooks.
    Returns the handler that is active after the call.
    """
    return install_pretty_logging(
        level=level,
        use_rich=use_rich,
        replace_handlers=replace_handlers,
        logger_name=logger_name,
    )


def enable_debug(
    *,
    use_rich: bool | Literal["auto"] = "auto",
    replace_handlers: bool = False,
    logger_name: str | None = None,
) -> logging.Handler:
    """Shorthand: DEBUG level."""
    return enable_logging(
        level=logging.DEBUG,
        use_rich=use_rich,
        replace_handlers=replace_handlers,
        logger_name=logger_name,
    )


@contextmanager
def temporary_log_level(
    level: int,
    *,
    logger_name: str | None = None,
):
    """
    Context manager to temporarily change the log level.
    """
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    prev = logger.level
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(prev)
