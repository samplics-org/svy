from __future__ import annotations

import logging
import time

from contextlib import contextmanager
from typing import Any, Iterable


log = logging.getLogger(__name__)


def _kv(items: Iterable[tuple[str, Any]]) -> str:
    parts: list[str] = []
    for k, v in items:
        try:
            s = str(v)
        except Exception:
            s = "<unrepr>"
        if len(s) > 120:
            s = s[:117] + "…"
        parts.append(f"{k}={s}")
    return " ".join(parts)


@contextmanager
def log_step(logger: logging.Logger, msg: str, /, **fields: Any):
    """
    Context manager: logs 'msg start …' and 'msg done ms=…', and on exceptions logs
    'msg failed …' with a traceback. Very low overhead when DEBUG is off.
    """
    t0 = time.perf_counter()
    debug_on = logger.isEnabledFor(logging.DEBUG)
    if debug_on:
        logger.debug("%s start %s", msg, _kv(fields.items()))
    try:
        yield
    except Exception:
        # Always log failures with traceback regardless of level
        logger.exception("%s failed %s", msg, _kv(fields.items()))
        raise
    else:
        if debug_on:
            dt = (time.perf_counter() - t0) * 1000.0
            logger.debug("%s done ms=%.1f %s", msg, dt, _kv(fields.items()))
