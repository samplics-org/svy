# svy/errors/_serialization.py
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence

from ._base import SvyError
from ._codes import Code


_DOCS_URL = "https://svylab.com/docs/svy/tutorials/serialization.html"


@dataclass(eq=False)
class SerializationError(SvyError):
    """
    Raised when a svy result object cannot be serialized into a payload
    struct, or a JSON payload cannot be decoded back into one.
    """

    _DEFAULT_CODE: ClassVar[str] = Code.SERIALIZATION_ERROR

    # ---- Encoding (result object -> payload) --------------------------------

    @classmethod
    def unsupported_type(
        cls,
        *,
        got_type: str,
        registered: Sequence[str],
        hint: Optional[str] = "Pass a svy result object (e.g. Estimate, Table, GLMFit).",
    ) -> "SerializationError":
        return cls.TypeError(
            title="No serializer registered",
            detail=f"Objects of type '{got_type}' cannot be serialized.",
            code=Code.UNSUPPORTED_RESULT_TYPE,
            where="serialize",
            expected=list(registered),
            got=got_type,
            hint=hint,
            docs_url=_DOCS_URL,
        )

    @classmethod
    def not_a_design(
        cls,
        *,
        got_type: str,
        hint: Optional[str] = "Pass the DesignData from serialize(design) or from_json().",
    ) -> "SerializationError":
        return cls.ValueError(
            title="Not a design payload",
            detail=f"to_design() needs a DesignData, got '{got_type}'.",
            code=Code.PAYLOAD_NOT_A_DESIGN,
            where="to_design",
            expected=["DesignData"],
            got=got_type,
            hint=hint,
            docs_url=_DOCS_URL,
        )

    # ---- Table view (payload -> DataFrame) ----------------------------------

    @classmethod
    def no_table(
        cls,
        *,
        got_type: str,
        supported: Sequence[str],
        hint: Optional[str] = "Read its fields directly; this result kind has no table.",
    ) -> "SerializationError":
        return cls.ValueError(
            title="No table for this payload",
            detail=f"'{got_type}' has no table view.",
            code=Code.PAYLOAD_NO_TABLE,
            where="to_polars",
            expected=list(supported),
            got=got_type,
            hint=hint,
            docs_url=_DOCS_URL,
        )

    # ---- Decoding (JSON payload -> struct) ----------------------------------

    @classmethod
    def missing_kind(
        cls,
        *,
        where: str = "from_json",
        hint: Optional[str] = "Only decode payloads from svy.serialize.to_json() or to_dict().",
    ) -> "SerializationError":
        return cls.ValueError(
            title="Payload missing 'kind'",
            detail="The payload has no 'kind' discriminator, so its result type cannot be determined.",
            code=Code.PAYLOAD_MISSING_KIND,
            where=where,
            param="kind",
            hint=hint,
        )

    @classmethod
    def unknown_kind(
        cls,
        *,
        kind: str,
        known: Sequence[str],
        where: str = "from_json",
        hint: Optional[
            str
        ] = "The payload may come from a newer svy; upgrade svy or re-serialize.",
    ) -> "SerializationError":
        return cls.ValueError(
            title="Unknown payload kind",
            detail=f"No payload struct is registered for kind '{kind}'.",
            code=Code.PAYLOAD_UNKNOWN_KIND,
            where=where,
            param="kind",
            expected=list(known),
            got=kind,
            hint=hint,
        )
