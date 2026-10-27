from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from ._base import SvyError, SvyWarning
from ._codes import Code


@dataclass(eq=False)
class IoError(SvyError):
    """Structured I/O errors for reads/writes across SPSS/Stata/SAS."""

    _DEFAULT_CODE: ClassVar[str] = Code.IO_ERROR

    # ---------- Convenience constructors ----------
    @classmethod
    def not_found(cls, *, where: Optional[str], path: str | Path) -> "IoError":
        return cls.FileNotFoundError(
            title="File not found",
            detail=f"No file at {Path(path)}",
            code=Code.FILE_NOT_FOUND,
            where=where,
            extra={"path": str(path)},
        )

    @classmethod
    def archive_member_not_found(
        cls, *, where: Optional[str], path: str | Path, engine_msg: str
    ) -> "IoError":
        return cls.FileNotFoundError(
            title="No readable file in archive",
            detail=engine_msg,
            code=Code.ARCHIVE_MEMBER_NOT_FOUND,
            where=where,
            hint="Check the archive's contents, or extract the data file and read it directly.",
            extra={"path": str(path), "engine_msg": engine_msg},
        )

    @classmethod
    def not_a_file(cls, *, where: Optional[str], path: str | Path) -> "IoError":
        return cls.OSError(
            title="Not a file",
            detail=f"Expected a file, got directory: {Path(path)}",
            code=Code.NOT_A_FILE,
            where=where,
            extra={"path": str(path)},
        )

    @classmethod
    def permission_denied(cls, *, where: Optional[str], path: str | Path) -> "IoError":
        return cls.PermissionError(
            title="Permission denied",
            detail=f"Access refused for path: {Path(path)}",
            code=Code.PERMISSION_DENIED,
            where=where,
            extra={"path": str(path)},
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
        return cls.ValueError(
            title=f"{fmt.upper()} parse error",
            detail=engine_msg or "The underlying reader failed to parse the file.",
            code=Code.READSTAT_PARSE_FAILED,
            where=where,
            hint=hint,
            extra={"path": str(path), "format": fmt, "engine_msg": engine_msg},
        )

    @classmethod
    def read_failed(
        cls, *, where: Optional[str], path: str | Path, reason: str | None = None
    ) -> "IoError":
        return cls.OSError(
            title="Unexpected read failure",
            detail=reason or "Read failed for an unknown reason.",
            code=Code.IO_READ_FAILED,
            where=where,
            extra={"path": str(path)},
        )

    @classmethod
    def write_failed(
        cls, *, where: Optional[str], path: str | Path, reason: str | None = None
    ) -> "IoError":
        return cls.OSError(
            title="Unexpected write failure",
            detail=reason or "Write failed for an unknown reason.",
            code=Code.IO_WRITE_FAILED,
            where=where,
            extra={"path": str(path)},
        )


# Small mapper you can use in core/io.py
def map_os_error(e: BaseException, *, where: str, path: str | Path) -> IoError:
    if isinstance(e, FileNotFoundError):
        p = Path(path)
        if not p.exists():
            return IoError.not_found(where=where, path=path)
        # The path exists, so something else is missing: a member of a zip
        # archive, or a companion file such as a SAS catalog.
        if p.suffix.lower() == ".zip":
            return IoError.archive_member_not_found(where=where, path=path, engine_msg=str(e))
        if e.filename:
            return IoError.not_found(where=where, path=e.filename)
        return IoError.read_failed(where=where, path=path, reason=str(e))
    if isinstance(e, IsADirectoryError):
        return IoError.not_a_file(where=where, path=path)
    if isinstance(e, PermissionError):
        return IoError.permission_denied(where=where, path=path)
    return IoError.read_failed(where=where, path=path, reason=str(e))


@dataclass(eq=False)
class IoWarning(SvyWarning):
    pass
