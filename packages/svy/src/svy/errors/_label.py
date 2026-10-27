# src/svy/errors/_label.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from ._base import SvyError, SvyWarning
from ._codes import Code


@dataclass(eq=False)
class LabelError(SvyError):
    """Errors related to variable/value labels and label schemes.

    Use this subtype for failures in scheme creation, validation, lookup,
    locale resolution, and (de)serialization.
    """

    _DEFAULT_CODE: ClassVar[str] = Code.LABEL_ERROR

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
        return cls.KeyError(
            title="Scheme not found",
            detail=f"No label scheme found for {param}.",
            code=Code.LABEL_UNKNOWN_SCHEME,
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
        return cls.ValueError(
            title="Scheme already exists",
            detail=f"A label scheme with id {scheme_id!r} already exists.",
            code=Code.LABEL_SCHEME_EXISTS,
            where=where,
            param="id",
            expected="unique id",
            got=scheme_id,
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
        return cls.ValueError(
            title="NaN key is not allowed",
            detail="Do not use NaN as a category key in value-label mappings.",
            code=Code.LABEL_NAN_KEY,
            where=where,
            param="mapping",
            expected="non-NaN keys",
            got="NaN",
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def duplicate_code(
        cls,
        *,
        where: Optional[str],
        code: object,
        docs_url: Optional[str] = None,
    ) -> "LabelError":
        return cls.ValueError(
            title="Duplicate category code",
            detail=(
                f"The code {code!r} appears more than once. A scheme holds one "
                f"entry per code, so a second would silently shadow the first."
            ),
            code=Code.LABEL_DUPLICATE_CODE,
            where=where,
            param="entries",
            expected="each code once",
            got=repr(code),
            hint="Remove the duplicate, or merge the two entries.",
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
        return cls.ValueError(
            title="Label catalog serialization failed",
            detail=reason,
            code=Code.LABEL_SERIALIZATION_ERROR,
            where=where,
            hint=hint,
            docs_url=docs_url,
            extra=extra,
        )


@dataclass(eq=False)
class LabelAPIRemoved(LabelError, AttributeError):
    """A label accessor removed in 2026.0; names its replacement.

    Also an ``AttributeError``, so ``hasattr`` on the old name is False.
    """

    _IS_KIND: ClassVar[bool] = True

    @classmethod
    def replaced(cls, *, where: str, use: str) -> "LabelAPIRemoved":
        return cls(
            title=f"{where} was removed",
            detail=f"Use {use} instead.",
            code=Code.LABEL_API_REMOVED,
            where=where,
            hint="meta.var_labels[var] gives the variable label and meta.value_labels[var] "
            "the code-to-label mapping; set them with set_var_label(s) and set_value_labels.",
        )


@dataclass(eq=False)
class LabelsReadOnly(LabelError, TypeError):
    """``meta.var_labels`` / ``meta.value_labels`` are views; edits go through setters."""

    _IS_KIND: ClassVar[bool] = True

    @classmethod
    def view(cls, *, where: str, use: str) -> "LabelsReadOnly":
        return cls(
            title=f"{where} is read-only",
            detail=f"Changing this mapping would not change the metadata. Use {use}.",
            code=Code.LABEL_READ_ONLY,
            where=where,
        )


@dataclass(eq=False)
class LabelWarning(SvyWarning):
    pass
