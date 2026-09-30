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


def _value(v: Any) -> str:
    """A stratum value as the column shows it: ``2002``, not ``2002.0``."""
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


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
                stratum_desc = ", ".join(f"{k}={_value(v)}" for k, v in sv.items())
            else:
                stratum_desc = getattr(s, "stratum_key", "<unknown>")
            psu_key = getattr(s, "psu_key", "<unknown>")
            n_obs = getattr(s, "n_observations", 0)
            lines.append(f"  {i}. {stratum_desc} (PSU={psu_key}, n={n_obs})")

        if n > 5:
            lines.append(f"  ... and {n - 5} more")

        lines.append("")
        lines.append("Variance cannot be estimated with unhandled singleton PSUs.")
        lines.append("Inspect them with sample.singletons, then declare a rule:")
        lines.append("")
        lines.append('  • svy.Singleton("center")     — grand-mean centering (R "adjust")')
        lines.append('  • svy.Singleton("scale")      — variance scaled up (R "average")')
        lines.append('  • svy.Singleton("skip")       — contribute nothing (R "remove")')
        lines.append('  • svy.Singleton("collapse")   — merge into another stratum')
        lines.append('  • svy.Singleton("pool")       — pool the singletons into one stratum')
        lines.append('  • svy.Singleton("self_representing") — the PSU becomes a stratum')

        # Convert to plain Python types; works for msgspec.Struct, dataclasses, dicts, etc.
        payload = [msgspec.to_builtins(s) for s in singletons]

        return cls(
            title=f"{n} singleton PSU(s) detected",
            detail="\n".join(lines),
            code="SINGLETON_ERROR",
            where=where,
            hint='sample.update_design(singleton=svy.Singleton("center")), or '
            'svy.Design(..., singleton="center") when building the sample.',
            extra={"singletons": payload},
        )

    @classmethod
    def for_taylor_on_replicates(
        cls,
        singletons: _SingletonSeq,
        *,
        rep_wgts: Any,
        use_replicates: str,
        where: str,
    ) -> "SingletonError":
        """A Taylor variance asked of a replicate design with singleton strata
        and no rule: the replicates are not used, the strata and PSUs are."""
        err = cls.from_singletons(singletons, where=where)
        err.detail += (
            f"\n\nThe design carries replicate weights ({rep_wgts.method}, "
            f"n_reps={rep_wgts.n_reps}), but this variance is Taylor linearization, "
            "computed from the stratum and PSU columns."
        )
        err.hint = f"{use_replicates} Or keep Taylor and declare a rule: {err.hint}"
        return err

    @classmethod
    def for_replicates(
        cls,
        singletons: _SingletonSeq,
        *,
        method: str | None,
        where: str,
    ) -> "SingletonError":
        """Replicates asked of a design with singleton strata and no rule that
        says how to build them."""
        err = cls.from_singletons(singletons, where=where)
        n = len(singletons)
        why = (
            "the design declares no singleton rule"
            if method is None
            else f"Singleton({method!r}) adjusts the Taylor variance and has no replicate analogue"
        )
        err.title = f"Replicates cannot be built on {n} singleton stratum(s)"
        err.detail = (
            f"{n} strata have one PSU and {why}. Declare how the replicates see them:\n\n"
            '  • svy.Singleton("collapse")  — merged into another stratum (Taylor too)\n'
            '  • svy.Singleton("pool")      — pooled into one stratum\n'
            '  • svy.Singleton("self_representing") — the PSU becomes a stratum\n'
            '  • svy.Singleton("skip")      — contribute nothing (R "remove")'
        )
        err.code = "SINGLETON_REPLICATES"
        err.hint = 'sample.update_design(singleton=svy.Singleton("collapse")), then create the replicates.'
        return err

    @classmethod
    def from_domain_singletons(
        cls,
        pairs: Sequence[str],
        *,
        by_domains: bool,
        method: str | None,
        applied: bool,
        where: str = "estimation",
    ) -> "SingletonError":
        """Strata with several PSUs but one inside an estimation domain, under
        ``on_domain_singletons="error"``. ``pairs`` are "stratum in domain" labels."""
        n = len(pairs)
        if by_domains:
            what = f"{n} domain × stratum {'pair has' if n == 1 else 'pairs have'}"
        else:
            what = f"{n} {'stratum has' if n == 1 else 'strata have'}"
        lines = [f"{what} one PSU in the domain:"]
        lines += [f"  {i}. {p}" for i, p in enumerate(pairs[:5], 1)]
        if n > 5:
            lines.append(f"  ... and {n - 5} more")
        m = getattr(method, "value", method)
        rule = m if m in ("center", "scale") else "center"
        how = 'domains="apply"' if applied else 'domains="standard"'
        lines += [
            "",
            f'The singleton rule was declared with {how} and on_domain_singletons="error".',
            "Declare how to report them instead:",
            "",
            f'  • svy.Singleton("{rule}", domains="apply")  — treat them as singletons '
            "(R survey.adjust.domain.lonely = TRUE)",
            f'  • svy.Singleton("{m or "center"}", on_domain_singletons="warn")   — note '
            "under the result",
            f'  • svy.Singleton("{m or "center"}")   — recorded only (R default)',
        ]
        return cls(
            title="Strata with a single PSU in a domain",
            detail="\n".join(lines),
            code="DOMAIN_SINGLETON",
            where=where,
            param="on_domain_singletons",
            hint="sample.domain_singletons(by=..., where=...) lists them.",
            extra={"pairs": list(pairs)},
        )


@dataclass(eq=False)
class SingletonAPIRemoved(SingletonError, AttributeError):
    """``sample.singleton`` was removed: the rule is declared on the design.

    Also an ``AttributeError``, so ``hasattr(sample, "singleton")`` is False.
    """

    @classmethod
    def accessor(cls) -> "SingletonAPIRemoved":
        return cls(
            title="sample.singleton was removed",
            detail=(
                "Singleton handling is part of the design. Declare the rule there:\n"
                '  svy.Design(..., singleton="center")  # or "scale", "skip", '
                '"self_representing", "collapse", "pool"\n'
                '  sample.update_design(singleton=svy.Singleton("collapse", '
                "using={singleton: target}))\n"
                "\n"
                "Inspect the singletons with sample.singletons, sample.n_singletons and\n"
                "sample.domain_singletons(by=..., where=...); read the rule with\n"
                "sample.design.singleton. To merge PSUs themselves, recode the PSU column:\n"
                "  sample.wrangling.recode(psu_column, {new: [old, ...]}, replace=True)"
            ),
            code="SINGLETON_API_REMOVED",
            where="Sample.singleton",
            hint='sample.singleton.center() is now svy.Design(..., singleton="center"); '
            'certainty() is "self_representing".',
        )
