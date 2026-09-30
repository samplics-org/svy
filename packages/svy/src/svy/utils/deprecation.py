# src/svy/utils/deprecation.py
"""Deprecations: announced with a ``DeprecationWarning`` naming the
replacement, removed no earlier than ``remove_in``. At a year bump,
``grep 'remove_in="2027.0"'`` is the removal list."""

from __future__ import annotations

import functools
import warnings

from typing import Any, Callable, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


def deprecation_message(what: str, *, since: str, remove_in: str, use: str | None) -> str:
    msg = f"{what} is deprecated since svy {since} and will be removed in {remove_in}."
    return msg if use is None else f"{msg} Use {use} instead."


def warn_deprecated(
    what: str, *, since: str, remove_in: str, use: str | None = None, stacklevel: int = 2
) -> None:
    """Emit the ``DeprecationWarning`` for ``what`` at the caller's caller."""
    warnings.warn(
        deprecation_message(what, since=since, remove_in=remove_in, use=use),
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def deprecated(
    *, since: str, remove_in: str, use: str | None = None, what: str | None = None
) -> Callable[[F], F]:
    """Mark a function or method deprecated; each call warns at the user's line.

    ``what`` names it in the message (default: its qualified name with ``()``).
    """

    def wrap(fn: F) -> F:
        name = what or f"{fn.__qualname__}()"

        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(name, since=since, remove_in=remove_in, use=use, stacklevel=2)
            return fn(*args, **kwargs)

        note = deprecation_message(name, since=since, remove_in=remove_in, use=use)
        inner.__doc__ = f".. deprecated:: {since}\n   {note}\n\n{fn.__doc__ or ''}"
        inner.__deprecated__ = note  # type: ignore[attr-defined]
        return inner  # type: ignore[return-value]

    return wrap
