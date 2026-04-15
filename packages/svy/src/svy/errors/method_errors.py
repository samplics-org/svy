# svy/errors/method_errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .base_errors import SvyError


@dataclass(eq=False)
class MethodError(SvyError):
    """
    Raised when a method/option is invalid or not applicable in the current context.
    Examples: bad enum value, unsupported 'how' mode, incompatible estimator, etc.
    """

    def __post_init__(self) -> None:
        # If caller didn't set a specific code, use a type-specific default.
        if self.code == "SVY_ERROR":
            self.code = "METHOD_ERROR"

    # ---- Convenience constructors -------------------------------------------------

    @classmethod
    def invalid_choice(
        cls,
        *,
        where: Optional[str],
        param: str,
        got: Any,
        allowed: Iterable[Any],
        hint: Optional[str] = None,
        docs_url: Optional[str] = None,
    ) -> "MethodError":
        allowed_list = list(allowed)
        return cls(
            title="Invalid option",
            detail=f"Parameter '{param}' must be one of {allowed_list}.",
            code="INVALID_CHOICE",
            where=where,
            param=param,
            expected=allowed_list,
            got=got,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def not_applicable(
        cls,
        *,
        where: Optional[str],
        method: str,
        reason: str,
        param: Optional[str] = None,
        hint: Optional[str] = None,
        docs_url: Optional[str] = None,
    ) -> "MethodError":
        return cls(
            title="Method not applicable",
            detail=f"'{method}' cannot be used here: {reason}.",
            code="METHOD_NOT_APPLICABLE",
            where=where,
            param=param,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def mutate_cycle(cls, cycle_list: str, *, where: str | None = None) -> "MethodError":
        return cls(
            title="Column transformation failed",
            detail=f"Dependency cycle or unresolved forward reference among: {cycle_list}",
            code="MUTATE_CYCLE",
            where=where,
            hint="Split into multiple mutate calls or remove circular references.",
            docs_url=None,
        )

    @classmethod
    def invalid_range(
        cls,
        *,
        where: str | None,
        param: str,
        got: Any,
        min_: float | int | None = None,
        max_: float | int | None = None,
        hint: str | None = None,
        docs_url: str | None = None,
    ) -> "MethodError":
        exp = (
            f"{min_} < {param} < {max_}"
            if (min_ is not None and max_ is not None)
            else "valid range"
        )
        return cls(
            title="Invalid numeric range",
            detail=f"Parameter '{param}' is out of range.",
            code="INVALID_RANGE",
            where=where,
            param=param,
            expected=exp,
            got=got,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def invalid_type(
        cls,
        *,
        where: str | None,
        param: str,
        got: Any,
        expected: str,
        hint: str | None = None,
        docs_url: str | None = None,
    ) -> "MethodError":
        return cls(
            title="Invalid type",
            detail=f"Parameter '{param}' has the wrong type.",
            code="INVALID_TYPE",
            where=where,
            param=param,
            expected=expected,
            got=type(got).__name__ if got is not None else None,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def invalid_mapping_keys(
        cls,
        *,
        where: str | None,
        param: str,
        missing: Iterable[Any] = (),
        extra: Iterable[Any] = (),
        hint: str | None = None,
        docs_url: str | None = None,
    ) -> "MethodError":
        miss = list(missing)
        ext = list(extra)
        detail_parts = []
        if miss:
            detail_parts.append(f"Missing keys: {miss}.")
        if ext:
            detail_parts.append(f"Unexpected keys: {ext}.")
        return cls(
            title="Mapping keys mismatch",
            detail=" ".join(detail_parts) or "Keys do not match the expected set.",
            code="INVALID_MAPPING_KEYS",
            where=where,
            param=param,
            expected="exact match of domain keys",
            got={"missing": miss, "extra": ext},
            hint=hint or "Ensure your mapping keys exactly match the domain categories.",
            docs_url=docs_url,
        )
