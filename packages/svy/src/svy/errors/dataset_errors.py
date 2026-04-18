# src/svy/errors/dataset_errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base_errors import SvyError


@dataclass(eq=False)
class DatasetError(SvyError):
    """
    Raised for dataset catalog, download, and integrity problems: catalog
    service failures, unknown slugs, hash mismatches, and related issues.
    """

    def __post_init__(self) -> None:
        # If caller didn't set a specific code, use a type-specific default.
        if self.code == "SVY_ERROR":
            self.code = "DATASET_ERROR"

    # ---- Catalog reachability ---------------------------------------------

    @classmethod
    def catalog_unreachable(
        cls,
        *,
        where: Optional[str],
        url: str,
        reason: str,
        hint: Optional[str] = "Check your network connection or SVYLAB_API_URL.",
        docs_url: Optional[str] = None,
    ) -> "DatasetError":
        return cls(
            title="Catalog unreachable",
            detail=f"Could not reach the dataset catalog at {url}: {reason}",
            code="CATALOG_UNREACHABLE",
            where=where,
            hint=hint,
            docs_url=docs_url,
            extra={"url": url},
        )

    @classmethod
    def catalog_bad_status(
        cls,
        *,
        where: Optional[str],
        url: str,
        status: int,
        hint: Optional[str] = "The catalog service may be temporarily unavailable.",
        docs_url: Optional[str] = None,
    ) -> "DatasetError":
        return cls(
            title="Catalog request failed",
            detail=f"Catalog at {url} returned HTTP {status}.",
            code="CATALOG_BAD_STATUS",
            where=where,
            hint=hint,
            docs_url=docs_url,
            extra={"url": url, "status": status},
        )

    # ---- Dataset lookup ---------------------------------------------------

    @classmethod
    def not_found(
        cls,
        *,
        where: Optional[str],
        slug: str,
        hint: Optional[str] = "Call svy.datasets.catalog() to see available slugs.",
        docs_url: Optional[str] = None,
    ) -> "DatasetError":
        return cls(
            title="Dataset not found",
            detail=f"No dataset in the catalog has slug {slug!r}.",
            code="DATASET_NOT_FOUND",
            where=where,
            param="name",
            got=slug,
            hint=hint,
            docs_url=docs_url,
        )

    # ---- Integrity --------------------------------------------------------

    @classmethod
    def sha_mismatch(
        cls,
        *,
        where: Optional[str],
        slug: str,
        expected: str,
        got: str,
        hint: Optional[str] = "Re-run with force_download=True, or check SVYLAB_API_URL.",
        docs_url: Optional[str] = None,
    ) -> "DatasetError":
        return cls(
            title="Dataset integrity check failed",
            detail=f"SHA-256 mismatch for dataset {slug!r}.",
            code="DATASET_SHA_MISMATCH",
            where=where,
            param="sha256",
            expected=expected,
            got=got,
            hint=hint,
            docs_url=docs_url,
            extra={"slug": slug},
        )
