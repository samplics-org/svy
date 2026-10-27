# svy/utils/experimental.py
"""Mark a function experimental: it may change in any release, without a deprecation period."""

from __future__ import annotations

import functools
import inspect
import warnings

from typing import Any, Callable, TypeVar

from svy.errors import Code, ExperimentalWarning


F = TypeVar("F", bound=Callable[..., Any])

_NOTE = "**Experimental.** It may change in any release, without a deprecation period."
_WARNED: set[str] = set()


def _marked(doc: str | None) -> str:
    if not doc:
        return _NOTE
    summary, _, rest = inspect.cleandoc(doc).partition("\n")
    return f"{summary}\n\n{_NOTE}\n{rest}" if rest else f"{summary}\n\n{_NOTE}"


def experimental(fn: F) -> F:
    """Warn on the first call, and note on every result, that ``fn`` is experimental.

    The warning is not recorded in ``sample.warnings``: it is about svy, not the data.
    """
    name = fn.__qualname__

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        note = ExperimentalWarning(
            code=Code.EXPERIMENTAL,
            title="Experimental method",
            detail=f"{name}() is experimental; it may change in any release, "
            "without a deprecation period.",
            hint="Silence it with warnings.filterwarnings('ignore', "
            "category=svy.errors.ExperimentalWarning).",
        )
        if name not in _WARNED:
            _WARNED.add(name)
            warnings.warn(note, stacklevel=2)
        result = fn(*args, **kwargs)
        add_findings = getattr(result, "_add_findings", None)
        return add_findings([note]) if callable(add_findings) else result

    wrapper.__doc__ = _marked(fn.__doc__)
    wrapper.__svy_experimental__ = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
