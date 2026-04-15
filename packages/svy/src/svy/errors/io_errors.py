from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .base_errors import SvyError


@dataclass(eq=False)
class IoError(SvyError):
    """Structured I/O errors for reads/writes across SPSS/Stata/SAS."""

    def __post_init__(self) -> None:
        if self.code == "SVY_ERROR":
            self.code = "IO_ERROR"

    # ---------- Convenience constructors ----------
    @classmethod
    def not_found(cls, *, where: Optional[str], path: str | Path) -> "IoError":
        return cls(
            title="File not found",
            detail=f"No file at {Path(path)}",
            code="FILE_NOT_FOUND",
            where=where,
            extra={"path": str(path)},
        )

    @classmethod
    def not_a_file(cls, *, where: Optional[str], path: str | Path) -> "IoError":
        return cls(
            title="Not a file",
            detail=f"Expected a file, got directory: {Path(path)}",
            code="NOT_A_FILE",
            where=where,
            extra={"path": str(path)},
        )

    @classmethod
    def permission_denied(cls, *, where: Optional[str], path: str | Path) -> "IoError":
        return cls(
            title="Permission denied",
            detail=f"Access refused for path: {Path(path)}",
            code="PERMISSION_DENIED",
            where=where,
            extra={"path": str(path)},
        )

    @classmethod
    def unsupported_ext(
        cls,
        *,
        where: Optional[str],
        path: str | Path,
        expected_exts: Iterable[str],
    ) -> "IoError":
        exts = tuple(sorted(set(e.lower() for e in expected_exts)))
        return cls(
            title="Unsupported file extension",
            detail=f"Path {Path(path)} does not match expected extensions {exts}.",
            code="UNSUPPORTED_EXTENSION",
            where=where,
            extra={"path": str(path), "expected_exts": exts},
        )

    @classmethod
    def parse_failed(
        cls,
        *,
        where: Optional[str],
        fmt: str,
        path: str | Path,
        engine_msg: str | None = None,
        hint: str
        | None = "Ensure the file is a valid, uncorrupted data file of the expected format.",
    ) -> "IoError":
        return cls(
            title=f"{fmt.upper()} parse error",
            detail=engine_msg or "The underlying reader failed to parse the file.",
            code="READSTAT_PARSE_FAILED",
            where=where,
            hint=hint,
            extra={"path": str(path), "format": fmt, "engine_msg": engine_msg},
        )

    @classmethod
    def engine_contract_violation(cls, *, where: Optional[str], got: Any) -> "IoError":
        return cls(
            title="Engine contract violation",
            detail="Adapter returned an unexpected shape or payload.",
            code="ENGINE_CONTRACT_VIOLATION",
            where=where,
            got=type(got).__name__,
        )

    @classmethod
    def read_failed(
        cls, *, where: Optional[str], path: str | Path, reason: str | None = None
    ) -> "IoError":
        return cls(
            title="Unexpected read failure",
            detail=reason or "Read failed for an unknown reason.",
            code="IO_READ_FAILED",
            where=where,
            extra={"path": str(path)},
        )

    @classmethod
    def write_failed(
        cls, *, where: Optional[str], path: str | Path, reason: str | None = None
    ) -> "IoError":
        return cls(
            title="Unexpected write failure",
            detail=reason or "Write failed for an unknown reason.",
            code="IO_WRITE_FAILED",
            where=where,
            extra={"path": str(path)},
        )


# Small mapper you can use in core/io.py
def map_os_error(e: BaseException, *, where: str, path: str | Path) -> IoError:
    if isinstance(e, FileNotFoundError):
        return IoError.not_found(where=where, path=path)
    if isinstance(e, IsADirectoryError):
        return IoError.not_a_file(where=where, path=path)
    if isinstance(e, PermissionError):
        return IoError.permission_denied(where=where, path=path)
    return IoError.read_failed(where=where, path=path, reason=str(e))
