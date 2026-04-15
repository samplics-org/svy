# src/svy/errors/label_errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .base_errors import SvyError


@dataclass(eq=False)
class LabelError(SvyError):
    """Errors related to variable/value labels and label schemes.

    Use this subtype for failures in scheme creation, validation, lookup,
    locale resolution, and (de)serialization.
    """

    def __post_init__(self) -> None:
        # Adopt a stable, filterable default code namespace for label errors.
        if self.code == "SVY_ERROR":
            self.code = "LABEL_ERROR"

    # ---------------------------------------------------------------------
    # Convenience constructors
    # ---------------------------------------------------------------------

    @classmethod
    def unknown_scheme(
        cls,
        *,
        where: Optional[str],
        param: str,
        got: Any,
        hint: Optional[
            str
        ] = "Use catalog.list() or catalog.search() to discover available schemes.",
        docs_url: Optional[str] = None,
    ) -> "LabelError":
        return cls(
            title="Scheme not found",
            detail=f"No label scheme found for {param}.",
            code="LABEL_UNKNOWN_SCHEME",
            where=where,
            param=param,
            expected="existing scheme id or concept",
            got=got,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def scheme_exists(
        cls,
        *,
        where: Optional[str],
        scheme_id: str,
        hint: Optional[str] = "Pass overwrite=True or choose a different id.",
        docs_url: Optional[str] = None,
    ) -> "LabelError":
        return cls(
            title="Scheme already exists",
            detail=f"A label scheme with id {scheme_id!r} already exists.",
            code="LABEL_SCHEME_EXISTS",
            where=where,
            param="id",
            expected="unique id",
            got=scheme_id,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def invalid_missing_codes(
        cls,
        *,
        where: Optional[str],
        param: str,
        not_in_mapping: Iterable[Any],
        hint: Optional[str] = "Ensure every missing code appears as a key in mapping.",
        docs_url: Optional[str] = None,
    ) -> "LabelError":
        missing_list = list(not_in_mapping)
        return cls(
            title="Invalid missing codes",
            detail="Some missing codes are not present in the value-label mapping.",
            code="LABEL_INVALID_MISSING_CODES",
            where=where,
            param=param,
            expected="missing ⊆ mapping.keys()",
            got=missing_list,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def inconsistent_missing_kinds(
        cls,
        *,
        where: Optional[str],
        offending_keys: Iterable[Any],
        hint: Optional[str] = "Add these codes to 'missing' or remove their kind assignment.",
        docs_url: Optional[str] = None,
    ) -> "LabelError":
        keys = list(offending_keys)
        return cls(
            title="Inconsistent missing kinds",
            detail="Keys in missing_kinds must also be listed in 'missing'.",
            code="LABEL_INCONSISTENT_MISSING_KINDS",
            where=where,
            param="missing_kinds",
            expected="keys(missing_kinds) ⊆ missing",
            got=keys,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def nan_key_forbidden(
        cls,
        *,
        where: Optional[str],
        hint: Optional[
            str
        ] = "Use an explicit code (e.g., 9) for missing; do not use NaN as a dict key.",
        docs_url: Optional[str] = None,
    ) -> "LabelError":
        return cls(
            title="NaN key is not allowed",
            detail="Do not use NaN as a category key in value-label mappings.",
            code="LABEL_NAN_KEY",
            where=where,
            param="mapping",
            expected="non-NaN keys",
            got="NaN",
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def invalid_locale(
        cls,
        *,
        where: Optional[str],
        got: Any,
        hint: Optional[str] = "Use primary subtags or BCP-47 style locales (e.g., 'en', 'en-US').",
        docs_url: Optional[str] = None,
    ) -> "LabelError":
        return cls(
            title="Invalid locale",
            detail="Locale is malformed or unsupported for label scheme selection.",
            code="LABEL_INVALID_LOCALE",
            where=where,
            param="locale",
            expected="language tag, e.g., 'fr', 'fr-CA'",
            got=got,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def ambiguous_pick(
        cls,
        *,
        where: Optional[str],
        concept: str,
        candidates: Iterable[str],
        hint: Optional[str] = "Specify a concrete locale or scheme id to disambiguate.",
        docs_url: Optional[str] = None,
    ) -> "LabelError":
        ids = list(candidates)
        return cls(
            title="Ambiguous scheme selection",
            detail=f"Multiple equally suitable schemes found for concept {concept!r}.",
            code="LABEL_AMBIGUOUS_PICK",
            where=where,
            param="concept",
            expected="unique best match",
            got=ids,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def serialization_error(
        cls,
        *,
        where: Optional[str],
        reason: str,
        hint: Optional[
            str
        ] = "Ensure mapping keys are JSON-serializable and sets are encoded as lists/tuples.",
        docs_url: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> "LabelError":
        return cls(
            title="Label catalog serialization failed",
            detail=reason,
            code="LABEL_SERIALIZATION_ERROR",
            where=where,
            hint=hint,
            docs_url=docs_url,
            extra=extra,
        )

    @classmethod
    def invalid_scheme_id(
        cls,
        *,
        where: Optional[str],
        got: Any,
        expected: str = "normalized id or 'concept:locale'",
        hint: Optional[str] = "Use make_scheme() to derive an id or pass a normalized id.",
        docs_url: Optional[str] = None,
    ) -> "LabelError":
        return cls(
            title="Invalid scheme id",
            detail="The provided scheme id does not conform to the expected format.",
            code="LABEL_INVALID_ID",
            where=where,
            param="id",
            expected=expected,
            got=got,
            hint=hint,
            docs_url=docs_url,
        )
