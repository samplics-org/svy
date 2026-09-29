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
    def __post_init__(self) -> None:
        # If caller didn't set a specific code, use a type-specific default.
        if self.code == "SVY_ERROR":
            self.code = "SINGLETON_ERROR"

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
        lines.append("")
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

    @classmethod
    def from_domain_singletons(
        cls,
        pairs: Sequence[str],
        *,
        n_strata: int,
        n_domains: int,
        method: str | None,
        where: str = "estimation",
    ) -> "SingletonError":
        """Strata with several PSUs but one inside an estimation domain, under
        ``domains="error"``. ``pairs`` are "domain: stratum" labels."""
        strata = "1 stratum has" if n_strata == 1 else f"{n_strata} strata have"
        domains = "1 estimation domain" if n_domains == 1 else f"{n_domains} estimation domains"
        lines = [f"{strata} a single PSU within {domains}:"]
        lines += [f"  {i}. {p}" for i, p in enumerate(pairs[:5], 1)]
        if len(pairs) > 5:
            lines.append(f"  ... and {len(pairs) - 5} more")
        rule = method if method in ("center", "scale") else "center"
        lines += [
            "",
            'The singleton rule was set with domains="error". Pick how to handle them:',
            "",
            f'  • sample.singleton.{rule}(domains="apply")  — treat them as singletons '
            "(R survey.adjust.domain.lonely = TRUE)",
            f'  • sample.singleton.{method or "center"}(domains="warn")   — standard domain '
            "variance, with a note",
            f'  • sample.singleton.{method or "center"}(domains="ignore") — standard domain '
            "variance (R default)",
        ]
        return cls(
            title="Strata with a single PSU in a domain",
            detail="\n".join(lines),
            code="DOMAIN_SINGLETON",
            where=where,
            extra={"pairs": list(pairs)},
        )
