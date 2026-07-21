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

    # ---- Download ---------------------------------------------------------

    @classmethod
    def download_failed(
        cls,
        *,
        where: Optional[str],
        slug: str,
        url: str,
        reason: str,
        hint: Optional[str] = (
            "Check your network connection, or use source='bundled' for the offline subset."
        ),
        docs_url: Optional[str] = None,
    ) -> "DatasetError":
        return cls(
            title="Dataset download failed",
            detail=f"Could not download dataset {slug!r} from {url}: {reason}",
            code="DATASET_DOWNLOAD_FAILED",
            where=where,
            hint=hint,
            docs_url=docs_url,
            extra={"slug": slug, "url": url},
        )

    # ---- Bundled (packaged) data -----------------------------------------

    @classmethod
    def not_bundled(
        cls,
        *,
        where: Optional[str],
        slug: str,
        bundled: "list[str]",
        hint: Optional[str] = None,
        docs_url: Optional[str] = None,
    ) -> "DatasetError":
        listed = ", ".join(bundled) if bundled else "(none)"
        return cls(
            title="Dataset not bundled",
            detail=(f"Dataset {slug!r} is not available offline as a bundled dataset."),
            code="DATASET_NOT_BUNDLED",
            where=where,
            param="source",
            got="bundled",
            hint=hint
            or (
                f"Use source='remote' (or the default source='auto') to download "
                f"it. Bundled datasets: {listed}."
            ),
            docs_url=docs_url,
            extra={"slug": slug, "bundled": list(bundled)},
        )

    @classmethod
    def bundled_unavailable(
        cls,
        *,
        where: Optional[str],
        slug: str,
        reason: str,
        hint: Optional[str] = (
            "The packaged dataset file is missing or unreadable; reinstalling "
            "svy should restore it."
        ),
        docs_url: Optional[str] = None,
    ) -> "DatasetError":
        return cls(
            title="Bundled dataset unavailable",
            detail=f"The bundled copy of dataset {slug!r} could not be read: {reason}",
            code="BUNDLED_UNAVAILABLE",
            where=where,
            param="source",
            got="bundled",
            hint=hint,
            docs_url=docs_url,
            extra={"slug": slug},
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
