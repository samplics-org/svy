# src/svy/weighting/_keys.py
"""Matching target keys to the levels of the data.

Every weighting method that takes keyed targets (controls, shares, a
resp_mapping) resolves its keys here. A key names a level by the level's own
value first, then by its text form -- JSON object keys are always strings, so
``"1"`` has to find the level ``1`` and ``"true"`` the level ``True``. A text key
that reads as two levels is refused rather than guessed.

Keys over several columns are tuples in column order, matched element by
element, or svy's joined text form (``"A_&_M"``).
"""

from __future__ import annotations

import datetime as _dt
import itertools

from typing import Any, Iterable, Mapping, Sequence

from svy.errors.weighting_errors import WeightingError


KEY_SEP = "_&_"

_NOT_FOUND = object()


class _Ambiguous(Exception):
    def __init__(self, matches: list[Any]) -> None:
        self.matches = matches


def native(v: Any) -> Any:
    """Python value for a numpy scalar; anything else unchanged."""
    if isinstance(v, (str, bytes, bool, int, float)) or v is None:
        return v
    item = getattr(v, "item", None)
    if callable(item) and getattr(v, "ndim", None) == 0:
        try:
            return item()
        except (TypeError, ValueError):
            return v
    return v


def text_forms(v: Any) -> tuple[str, ...]:
    """The strings that name ``v`` when it comes back as text."""
    if v is None:
        return ()
    if isinstance(v, bool):
        return (str(v), str(v).lower())
    if isinstance(v, float):
        if v.is_integer():
            return (str(v), str(int(v)))
        return (str(v),)
    if isinstance(v, _dt.datetime):
        return (str(v), v.isoformat())
    if isinstance(v, (_dt.date, _dt.time)):
        return (v.isoformat(),)
    return (str(v),)


def _text_map(values: Iterable[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for v in values:
        for form in text_forms(v):
            bucket = out.setdefault(form, [])
            if not any(v is b or (v == b and type(v) is type(b)) for b in bucket):
                bucket.append(v)
    return out


def _key_forms(k: Any) -> tuple[str, ...]:
    return (k,) if isinstance(k, str) else text_forms(k)


class LevelIndex:
    """The levels of one column (scalars) or several (tuples in column order)."""

    def __init__(self, levels: Sequence[Any], *, width: int = 1) -> None:
        self.levels = [
            tuple(native(p) for p in lv) if isinstance(lv, tuple) else native(lv) for lv in levels
        ]
        self.width = width
        self._exact: dict[Any, Any] = {}
        for lv in self.levels:
            self._exact.setdefault(lv, lv)
        if width == 1:
            self._pos = [(self._exact, _text_map(self._exact))]
            self._joined: dict[str, list[Any]] = {}
        else:
            self._pos = []
            for i in range(width):
                vals: dict[Any, Any] = {}
                for lv in self.levels:
                    vals.setdefault(lv[i], lv[i])
                self._pos.append((vals, _text_map(vals)))
            self._joined = {}
            for lv in self.levels:
                for combo in itertools.product(*(text_forms(p) for p in lv)):
                    bucket = self._joined.setdefault(KEY_SEP.join(combo), [])
                    if lv not in bucket:
                        bucket.append(lv)

    @staticmethod
    def _one(key: Any, exact: dict[Any, Any], text: dict[str, list[Any]]) -> Any:
        try:
            if key in exact:
                return exact[key]
        except TypeError:
            return _NOT_FOUND
        found: list[Any] = []
        for form in _key_forms(key):
            for v in text.get(form, ()):
                if not any(v is f or (v == f and type(v) is type(f)) for f in found):
                    found.append(v)
        if len(found) > 1:
            raise _Ambiguous(found)
        return found[0] if found else _NOT_FOUND

    def resolve(self, key: Any) -> Any:
        """The level ``key`` names, or ``_NOT_FOUND``; raises ``_Ambiguous``."""
        key = native(key)
        if self.width == 1:
            exact, text = self._pos[0]
            return self._one(key, exact, text)
        if isinstance(key, list):
            key = tuple(key)
        if isinstance(key, tuple):
            if len(key) != self.width:
                return _NOT_FOUND
            try:
                if key in self._exact:
                    return self._exact[key]
            except TypeError:
                return _NOT_FOUND
            parts = []
            for i, part in enumerate(key):
                exact, text = self._pos[i]
                got = self._one(native(part), exact, text)
                if got is _NOT_FOUND:
                    return _NOT_FOUND
                parts.append(got)
            t = tuple(parts)
            return self._exact.get(t, _NOT_FOUND)
        if isinstance(key, str):
            found = self._joined.get(key, [])
            if len(found) > 1:
                raise _Ambiguous(found)
            return found[0] if found else _NOT_FOUND
        return _NOT_FOUND


def _is_zero(v: Any) -> bool:
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return False


def match_keys(
    supplied: Mapping[Any, Any],
    index: LevelIndex,
    *,
    where: str,
    param: str,
    required: Sequence[Any] | None = None,
    cols: Sequence[str] | None = None,
    zero_extra_ok: bool = True,
) -> dict[Any, Any]:
    """``supplied`` re-keyed by the data's levels.

    Every level in ``required`` (default: all) needs a key; a key naming no
    level is an error unless its target is 0 (a level the sample lacks), and
    two keys naming one level are an error.
    """
    need = index.levels if required is None else list(required)
    out: dict[Any, Any] = {}
    source: dict[Any, Any] = {}
    extra: list[Any] = []
    for key, val in supplied.items():
        try:
            lvl = index.resolve(key)
        except _Ambiguous as a:
            raise WeightingError.key_ambiguous(
                where=where, param=param, key=native(key), matches=a.matches
            ) from None
        if lvl is _NOT_FOUND:
            if not (zero_extra_ok and _is_zero(val)):
                extra.append(native(key))
            continue
        if lvl in source:
            raise WeightingError.key_duplicate(
                where=where, param=param, level=lvl, keys=[native(source[lvl]), native(key)]
            )
        source[lvl] = key
        out[lvl] = val
    missing = [lv for lv in need if lv not in out]
    if missing or extra:
        raise WeightingError.keys_mismatch(
            where=where,
            param=param,
            levels=sort_levels(need),
            missing=missing,
            extra=extra,
            cols=cols,
        )
    return out


def _sort_key(v: Any) -> Any:
    from svy.weighting._helpers import _num_sort_key_token

    if isinstance(v, tuple):
        return (0, tuple(_sort_key(p) for p in v))
    if v is None:
        return (6, "")
    if isinstance(v, bool):
        return (2, int(v))
    if isinstance(v, (int, float)):
        return (1, v)
    if isinstance(v, str):
        return (3, _num_sort_key_token(v))
    if isinstance(v, (_dt.date, _dt.time)):
        return (4, v.isoformat())
    return (5, str(v))


def sort_levels(levels: Iterable[Any]) -> list[Any]:
    """Levels in a stable, readable order: numbers numerically, text naturally."""
    return sorted(levels, key=_sort_key)


def target_vector(
    matched: Mapping[Any, Any],
    levels: Sequence[Any],
    *,
    where: str,
    param: str,
    nonneg: bool = True,
) -> list[float]:
    """Targets aligned to ``levels``: finite numbers, >= 0 unless ``nonneg=False``."""
    out: list[float] = []
    bad: dict[Any, Any] = {}
    for lv in levels:
        v = matched[lv]
        try:
            f = float(v)
        except (TypeError, ValueError):
            bad[lv] = native(v) if isinstance(v, (int, float, str, bool)) or v is None else str(v)
            continue
        if f != f or f in (float("inf"), float("-inf")) or (nonneg and f < 0.0):
            bad[lv] = f
        out.append(f)
    if bad:
        raise WeightingError.value_invalid(where=where, param=param, bad=bad, nonneg=nonneg)
    return out


NA_OPTIONS = ("error", "level", "drop")


def check_na_option(na: Any, *, where: str) -> None:
    from svy.errors import MethodError

    if na not in NA_OPTIONS:
        raise MethodError.invalid_choice(
            where=where,
            param="na",
            got=na,
            allowed=list(NA_OPTIONS),
            hint="na='error' refuses nulls, 'level' keys them by na_label, 'drop' leaves them out.",
        )


def template_levels(
    df: Any, cols: Sequence[str], *, na: str, na_label: str, where: str, code: str
) -> list[Any]:
    """The levels a template lists for ``cols``: the columns' own values (tuples
    for several), with nulls handled by ``na``."""
    from svy.errors import DimensionError

    nulls = {c: int(n) for c, n in zip(cols, df.select(cols).null_count().row(0)) if n}
    if nulls and na == "error":
        listed = ", ".join(f"{c!r} ({n} null{'' if n == 1 else 's'})" for c, n in nulls.items())
        raise DimensionError(
            title="Missing values in template columns",
            detail=f"Nulls found in {listed}. Choose na='level' or na='drop', or fix the data.",
            code=code,
            where=where,
            param=next(iter(nulls)),
            expected=0,
            got=nulls,
            hint=f"na='level' lists them under {na_label!r}; na='drop' leaves them out.",
        )
    rows = df.select(cols).unique().rows()
    out: list[Any] = []
    has_null = False
    for row in rows:
        if any(v is None for v in row):
            has_null = True
            if na == "drop":
                continue
            row = tuple(na_label if v is None else v for v in row)
        out.append(row[0] if len(cols) == 1 else tuple(row))
    levels = sort_levels(dict.fromkeys(out))
    if len(cols) == 1 and has_null and na == "level" and na_label in levels:
        levels.remove(na_label)
        levels.append(na_label)
    return levels
