# svy/errors/singleton_errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

import msgspec

from .base_errors import SvyError


# ---- Runtime-safe typing: no import from svy.core.singleton ----
class _SingletonInfoLike(Protocol):
    """Structural type for singleton records to avoid runtime import cycles."""

    stratum_key: str
    stratum_values: Mapping[str, Any] | None
    psu_key: str
    n_observations: int


# For type checkers only (optional, keeps IDEs happy without creating runtime deps)
if TYPE_CHECKING:
    from svy.core.singleton import SingletonInfo as _SingletonInfoConcrete

    _SingletonSeq = Sequence[_SingletonInfoConcrete]  # noqa: N816 (alias)
else:
    _SingletonSeq = Sequence[_SingletonInfoLike]  # structural at runtime


@dataclass(eq=False)
class SingletonError(SvyError):
    @classmethod
    def from_singletons(
        cls,
        singletons: _SingletonSeq,
        where: str = "singleton_handling",
    ) -> "SingletonError":
        n = len(singletons)

        lines = [f"Found {n} singleton PSU(s) in the following strata:"]
        for i, s in enumerate(singletons[:5], 1):
            if getattr(s, "stratum_values", None):
                sv = s.stratum_values
                stratum_desc = ", ".join(f"{k}={v}" for k, v in sv.items())
            else:
                stratum_desc = getattr(s, "stratum_key", "<unknown>")
            psu_key = getattr(s, "psu_key", "<unknown>")
            n_obs = getattr(s, "n_observations", 0)
            lines.append(f"  {i}. {stratum_desc} (PSU={psu_key}, n={n_obs})")

        if n > 5:
            lines.append(f"  ... and {n - 5} more")

        lines.append("")
        lines.append("Variance cannot be estimated with unhandled singleton PSUs.")
        lines.append("Inspect them with sample.singleton.summary(), then pick a strategy:")
        lines.append("  • sample.singleton.certainty()  — treat as self-representing units")
        lines.append("  • sample.singleton.skip()       — drop from variance (R 'remove')")
        lines.append("  • sample.singleton.center()     — grand-mean centering (R 'adjust')")
        lines.append("  • sample.singleton.scale()      — variance inflation (R 'average')")
        lines.append("  • sample.singleton.collapse()   — merge into nearby strata")
        lines.append("  • sample.singleton.pool()       — pool singletons into one stratum")
        lines.append("  • sample.singleton.combine(map) — manual stratum/PSU remapping")

        # Convert to plain Python types; works for msgspec.Struct, dataclasses, dicts, etc.
        payload = [msgspec.to_builtins(s) for s in singletons]

        return cls(
            title=f"{n} singleton PSU(s) detected",
            detail="\n".join(lines),
            code="SINGLETON_ERROR",
            where=where,
            extra={"singletons": payload},
        )
