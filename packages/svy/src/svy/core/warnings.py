# src/svy/core/warnings.py
from __future__ import annotations

import builtins
import logging

from datetime import datetime, timezone
from enum import IntEnum, StrEnum
from typing import Any, Iterable, Sequence

import msgspec

from svy.errors.base_errors import SvyError


log = logging.getLogger(__name__)


# ---------- Levels & Codes ----------
class Severity(IntEnum):
    INFO = 10
    WARNING = 20
    ERROR = 30  # "soft error" – continue but may be escalated


class WarnCode(StrEnum):
    # ── Design / data quality ──────────────────────────────────────────
    MISSING_LABELS = "MISSING_LABELS"
    UNKNOWN_CATEGORY = "UNKNOWN_CATEGORY"
    ZERO_WEIGHT = "ZERO_WEIGHT"
    NEGATIVE_WEIGHT = "NEGATIVE_WEIGHT"
    SINGLETON_PSU = "SINGLETON_PSU"
    STRATUM_WITH_ONE_PSU = "STRATUM_WITH_ONE_PSU"
    COLLAPSED_STRATA = "COLLAPSED_STRATA"
    POSTSTRATA_EMPTY = "POSTSTRATA_EMPTY"
    CONTROL_MISMATCH = "CONTROL_MISMATCH"
    DESIGN_INCOMPLETE = "DESIGN_INCOMPLETE"
    # ── Weighting (generic — shared across trim, rake, calibrate, …) ──
    REPLICATE_SKIPPED = "REPLICATE_SKIPPED"  # rep weights present but not adjusted
    DOMAIN_SKIPPED = "DOMAIN_SKIPPED"  # domain skipped (e.g. below min_cell_size)
    MAX_ITER_REACHED = "MAX_ITER_REACHED"  # iterative method hit max_iter without converging
    WEIGHT_SUM_CHANGED = "WEIGHT_SUM_CHANGED"  # total weight sum changed unexpectedly
    WEIGHT_ADJ_AUDIT = "WEIGHT_ADJ_AUDIT"  # info-level audit record for any weight adjustment

    LABEL_KEY_NOT_IN_DATA = "LABEL_KEY_NOT_IN_DATA"
    DATA_VALUE_NOT_LABELED = "DATA_VALUE_NOT_LABELED"


# ---------- Warning value (immutable) ----------
class SvyWarning(msgspec.Struct, frozen=True):
    """
    Lightweight, typed, immutable warning that can render like SvyError.
    Use (title, detail) to match SvyError ergonomics.
    """

    code: WarnCode | str
    title: str
    detail: str
    where: str | None = None
    level: Severity = Severity.WARNING

    # Optional structured context matching your SvyError schema
    param: str | None = None
    expected: Any = None
    got: Any = None
    hint: str | None = None
    docs_url: str | None = None
    extra: dict[str, Any] | None = None

    # Survey-flavored context (purely additive)
    var: str | None = None
    rows: tuple[int, ...] | None = None

    ts: datetime = msgspec.field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Reuse SvyError renderers via an adapter ---
    def _as_error(self) -> SvyError:
        # Stuff warning-only fields into `extra`
        extra = dict(self.extra or {})
        extra.update(
            {
                "level": self.level.name,
                "var": self.var,
                "rows": None if self.rows is None else list(self.rows),
                "ts": self.ts.isoformat(),
            }
        )
        return SvyError(
            title=self.title,
            detail=self.detail,
            code=str(self.code),
            where=self.where,
            param=self.param,
            expected=self.expected,
            got=self.got,
            hint=self.hint,
            docs_url=self.docs_url,
            extra=extra,
        )

    # --- Renderers (identical look & feel as SvyError) ---
    def text(self, *, indent: int = 2, surround: bool = True) -> str:
        return self._as_error().text(indent=indent, surround=surround)

    def ansi(self) -> str:
        return self._as_error().ansi()

    def markdown(self) -> str:
        return self._as_error().markdown()

    def html(self) -> str:
        return self._as_error().html()

    def to_dict(self) -> dict[str, Any]:
        # Similar to SvyError.to_dict(), but tagged as a warning
        d = self._as_error().to_dict()
        payload = d["error"]
        payload["level"] = self.level.name
        payload["ts"] = self.ts.isoformat()
        payload["var"] = self.var
        payload["rows"] = None if self.rows is None else list(self.rows)
        return {"warning": payload}

    # De-dupe key (tunable)
    def key(self) -> tuple[Any, ...]:
        return (str(self.code), self.where, self.param, self.var, self.detail)


# ---------- Aggregation / escalation ----------
class SvyWarningsError(SvyError):
    """
    Escalation exception that keeps your formatting surfaces.
    """

    @classmethod
    def from_warnings(
        cls,
        warnings: Sequence[SvyWarning],
        *,
        where: str | None = None,
        code: str = "SVY_WARNINGS_ERROR",
    ) -> "SvyWarningsError":
        if not warnings:
            return cls(
                title="No warnings to escalate",
                detail="SvyWarningsError.from_warnings called with an empty list.",
                code=code,
                where=where,
            )
        lines = []
        for w in warnings[:5]:
            loc = f" at {w.where}" if w.where else ""
            lines.append(f"- {w.level.name} {w.code}{loc}: {w.title} — {w.detail}")
        more = "" if len(warnings) <= 5 else f"\n… and {len(warnings) - 5} more"
        detail = "Escalated warnings:\n" + "\n".join(lines) + more
        return cls(
            title=f"{len(warnings)} warning(s) escalated",
            detail=detail,
            code=code,
            where=where,
            extra={"warnings": [w.to_dict()["warning"] for w in warnings]},
        )


class WarningStore:
    """
    Attach this to Sample._warnings.
    - De-dupes by SvyWarning.key()
    - Optional per-code cap to avoid floods
    """

    __slots__ = ("_items", "_seen", "_per_code", "_dedupe", "_max_per_code")

    def __init__(self, *, dedupe: bool = True, max_per_code: int | None = 100) -> None:
        self._items: list[SvyWarning] = []
        self._seen: set[tuple[Any, ...]] = set()
        self._per_code: dict[str, int] = {}
        self._dedupe = dedupe
        self._max_per_code = max_per_code

    # --- mutation ---
    def add(self, w: SvyWarning) -> None:
        if self._dedupe:
            k = w.key()
            if k in self._seen:
                return
            self._seen.add(k)

        if self._max_per_code is not None:
            key = str(w.code)
            n = self._per_code.get(key, 0)
            if n >= self._max_per_code:
                if n == self._max_per_code:
                    log.info(f"Supressing further warnings for code: {w.code}")
                return
            self._per_code[key] = n + 1

        self._items.append(w)

        log_msg = f"[{w.code}] {w.title} - {w.detail} (var={w.var})"

        if w.level >= Severity.ERROR:
            log.error(log_msg, extra=w.to_dict()["warning"])
        elif w.level >= Severity.WARNING:
            log.warning(log_msg, extra=w.to_dict()["warning"])
        else:
            log.info(log_msg, extra=w.to_dict()["warning"])

    def extend(self, warnings: Iterable[SvyWarning]) -> None:
        for w in warnings:
            self.add(w)

    def clear(self) -> None:
        self._items.clear()
        self._seen.clear()
        self._per_code.clear()

    # --- querying ---
    def list(
        self,
        *,
        min_level: Severity | None = None,
        code: WarnCode | str | None = None,
        where_contains: str | None = None,
    ) -> builtins.list[SvyWarning]:
        out = self._items
        if min_level is not None:
            out = [w for w in out if w.level >= min_level]
        if code is not None:
            out = [w for w in out if str(w.code) == str(code)]
        if where_contains:
            out = [w for w in out if (w.where and where_contains in w.where)]
        return out

    def counts(self) -> dict[str, int]:
        return {
            "INFO": sum(1 for w in self._items if w.level == Severity.INFO),
            "WARNING": sum(1 for w in self._items if w.level == Severity.WARNING),
            "ERROR": sum(1 for w in self._items if w.level == Severity.ERROR),
            "TOTAL": len(self._items),
        }

    # --- escalation helpers ---
    def raise_if_errors(self) -> None:
        errs = [w for w in self._items if w.level >= Severity.ERROR]
        if errs:
            raise SvyWarningsError.from_warnings(errs)

    def escalate(self, *, min_level: Severity = Severity.ERROR) -> None:
        sel = [w for w in self._items if w.level >= min_level]
        if sel:
            raise SvyWarningsError.from_warnings(sel)

    # --- export ---
    def to_dicts(self) -> builtins.list[dict[str, Any]]:
        return [w.to_dict() for w in self._items]

    def to_json(self) -> bytes:
        return msgspec.json.encode(self.to_dicts())

    def to_polars(self):
        try:
            import polars as pl
        except Exception:  # pragma: no cover
            raise RuntimeError("Polars is not available")
        rows = [wd["warning"] for wd in self.to_dicts()]
        return pl.from_dicts(rows)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)
