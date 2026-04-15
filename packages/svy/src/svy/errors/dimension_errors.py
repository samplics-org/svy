# svy/errors/dimension_errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .base_errors import SvyError


@dataclass(eq=False)
class DimensionError(SvyError):
    """
    Raised for shape/schema/value-size problems: missing columns, incompatible
    dimensions, invalid counts (e.g., n < 0), or sampling size > available rows.
    """

    def __post_init__(self) -> None:
        # If caller didn't set a specific code, use a type-specific default.
        if self.code == "SVY_ERROR":
            self.code = "DIMENSION_ERROR"

    # ---- Convenience constructors -------------------------------------------------

    @classmethod
    def invalid_n(
        cls,
        *,
        where: Optional[str],
        got: Any,
        expected: Any = "n >= 0 or None",
        hint: Optional[str] = "Use None to select all rows.",
        docs_url: Optional[str] = None,
    ) -> "DimensionError":
        return cls(
            title="Invalid row count",
            detail="Parameter 'n' must be a non-negative integer or None.",
            code="INVALID_N",
            where=where,
            param="n",
            expected=expected,
            got=got,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def missing_columns(
        cls,
        *,
        where: Optional[str],
        param: str,
        missing: Iterable[str],
        available: Iterable[str] | None = None,
        hint: Optional[str] = "Check spelling or inspect df.columns.",
        docs_url: Optional[str] = None,
    ) -> "DimensionError":
        missing_list = list(missing)
        extra = {"available_preview": list(available)[:50]} if available is not None else None
        return cls(
            title="Column(s) not found",
            detail=f"Missing: {', '.join(map(repr, missing_list))}.",
            code="MISSING_COLUMNS",
            where=where,
            param=param,
            expected="subset of dataframe columns",
            got=missing_list,
            hint=hint,
            docs_url=docs_url,
            extra=extra,
        )

    @classmethod
    def sample_too_large(
        cls,
        *,
        where: Optional[str],
        n: int,
        available_rows: int,
        param: str = "n",
        hint: Optional[str] = "When sampling without replacement, ensure n ≤ number of rows.",
        docs_url: Optional[str] = None,
    ) -> "DimensionError":
        return cls(
            title="Sampling failed",
            detail=f"Requested n={n} exceeds available rows={available_rows} for sampling without replacement.",
            code="SAMPLE_TOO_LARGE",
            where=where,
            param=param,
            expected=f"n ≤ {available_rows}",
            got=n,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def empty_estimates(
        cls,
        *,
        where: str | None,
        param: str = "estimates",
        hint: str | None = "Run an analysis first or pass non-empty estimates.",
        docs_url: str | None = None,
    ) -> "DimensionError":
        return cls(
            title="No estimates to display",
            detail="The table/test has no estimate rows.",
            code="EMPTY_ESTIMATES",
            where=where,
            param=param,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def domain_keys_mismatch(
        cls,
        *,
        where: str | None,
        domain: str,
        expected_keys: Iterable[Any],
        got_keys: Iterable[Any],
        hint: str | None = "Align your provided levels/controls with the domain categories.",
        docs_url: str | None = None,
    ) -> "DimensionError":
        exp = set(expected_keys)
        got = set(got_keys)
        return cls(
            title="Domain keys mismatch",
            detail=f"For domain '{domain}', expected keys {sorted(exp)}, got {sorted(got)}.",
            code="DOMAIN_KEYS_MISMATCH",
            where=where,
            param=domain,
            expected=sorted(exp),
            got=sorted(got),
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def group_levels_mismatch(
        cls,
        *,
        where: str | None,
        var: str,
        expected_levels: Iterable[Any],
        got_levels: Iterable[Any],
        hint: str | None = "Ensure estimates only include the two requested levels.",
        docs_url: str | None = None,
    ) -> "DimensionError":
        return cls(
            title="Group levels mismatch",
            detail=f"Group '{var}' expected levels {tuple(expected_levels)}, but estimates include {sorted(set(got_levels))}.",
            code="GROUP_LEVELS_MISMATCH",
            where=where,
            param=var,
            expected=list(expected_levels),
            got=list(sorted(set(got_levels))),
            hint=hint,
            docs_url=docs_url,
        )
