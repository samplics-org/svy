# svy/errors/serialization_errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .base_errors import SvyError


_DOCS_URL = "https://svylab.com/docs/svy/tutorials/serialization.html"


@dataclass(eq=False)
class SerializationError(SvyError):
    """
    Raised when a svy result object cannot be serialized into a payload
    struct, or a JSON payload cannot be decoded back into one.
    """

    def __post_init__(self) -> None:
        if self.code == "SVY_ERROR":
            self.code = "SERIALIZATION_ERROR"

    # ---- Encoding (result object -> payload) --------------------------------

    @classmethod
    def unsupported_type(
        cls,
        *,
        got_type: str,
        registered: Sequence[str],
        hint: Optional[str] = "Pass a svy result object (e.g. Estimate, Table, GLMFit).",
    ) -> "SerializationError":
        return cls(
            title="No serializer registered",
            detail=f"Objects of type '{got_type}' cannot be serialized.",
            code="UNSUPPORTED_RESULT_TYPE",
            where="serialize",
            expected=list(registered),
            got=got_type,
            hint=hint,
            docs_url=_DOCS_URL,
        )

    # ---- Decoding (JSON payload -> struct) ----------------------------------

    @classmethod
    def missing_kind(
        cls,
        *,
        hint: Optional[str] = "Only decode JSON produced by svy.serialize.to_json().",
    ) -> "SerializationError":
        return cls(
            title="Payload missing 'kind'",
            detail="The JSON payload has no 'kind' discriminator, so its result type cannot be determined.",
            code="PAYLOAD_MISSING_KIND",
            where="from_json",
            param="kind",
            hint=hint,
        )

    @classmethod
    def unknown_kind(
        cls,
        *,
        kind: str,
        known: Sequence[str],
        hint: Optional[str] = "The payload may come from a newer svy; upgrade svy or re-serialize.",
    ) -> "SerializationError":
        return cls(
            title="Unknown payload kind",
            detail=f"No payload struct is registered for kind '{kind}'.",
            code="PAYLOAD_UNKNOWN_KIND",
            where="from_json",
            param="kind",
            expected=list(known),
            got=kind,
            hint=hint,
        )
