# src/svy/utils/deprecation.py
"""Deprecations: announced with a ``DeprecationWarning`` naming the
replacement, removed no earlier than ``remove_in``. At a year bump,
``grep 'remove_in="2027.0"'`` is the removal list."""

from __future__ import annotations

import functools
import inspect
import warnings

from typing import Any, Callable, TypeVar

from svy.errors import Code, UsageError


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


def deprecated_param(old: str, new: str, *, since: str, remove_in: str) -> Callable[[F], F]:
    """Accept the renamed keyword ``old`` as ``new``; each use warns at the user's line.

    Stack one per renamed keyword. Passing both names raises ``PARAM_GIVEN_TWICE``.
    """

    def wrap(fn: F) -> F:
        params = inspect.signature(fn).parameters
        if new not in params or old in params:
            raise ValueError(f"{fn.__qualname__} must take {new}= and not {old}=")
        name = f"{fn.__qualname__}({old}=)"

        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if old in kwargs:
                if new in kwargs:
                    raise UsageError.TypeError(
                        title="Argument given twice",
                        detail=f"{fn.__qualname__}() got both {old}= and {new}=; "
                        f"{old}= is the deprecated name of {new}=.",
                        code=Code.PARAM_GIVEN_TWICE,
                        where=f"{fn.__qualname__}()",
                        param=new,
                        got=[old, new],
                        hint=f"Pass {new}= only.",
                    )
                warn_deprecated(
                    name, since=since, remove_in=remove_in, use=f"{new}=", stacklevel=2
                )
                kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)

        note = deprecation_message(name, since=since, remove_in=remove_in, use=f"{new}=")
        inner.__doc__ = f".. deprecated:: {since}\n   {note}\n\n{fn.__doc__ or ''}"
        renamed = dict(getattr(fn, "__deprecated_params__", {}))
        renamed[old] = new
        inner.__deprecated_params__ = renamed  # type: ignore[attr-defined]
        return inner  # type: ignore[return-value]

    return wrap
