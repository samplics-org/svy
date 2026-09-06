# python/svy_io/helpers.py
import contextlib
import os
import tempfile

from typing import Any


# ---------------- encoding ----------------

_LOSSY_ENCODINGS = frozenset({"utf8-lossy", "utf-8-lossy"})

# Appended to the parse error for READSTAT_ERROR_CONVERT_BAD_STRING (rc=17).
BAD_STRING_HINT = (
    "pass encoding='utf8-lossy' to replace undecodable bytes with U+FFFD "
    "(flagged in meta['had_invalid_utf8']), or the file's actual encoding if "
    "the one assumed is wrong"
)


def _split_encoding(encoding: str | None) -> tuple[str | None, bool]:
    """Map the public ``encoding`` to (iconv name, lossy_utf8) for the native layer."""
    if encoding is not None and encoding.lower() in _LOSSY_ENCODINGS:
        return None, True
    return encoding, False


def _with_bad_string_hint(e: RuntimeError, hint: str = BAD_STRING_HINT) -> RuntimeError:
    """Attach a ``Hint:`` naming the encoding option to an rc=17 parse error."""
    if "(rc=17)" in str(e):
        return RuntimeError(f"{e}. Hint: {hint}.")
    return e


# ---------------- n_max normalization ----------------


def _normalize_n_max(n_max: Any) -> int | None:
    """
    Normalize/validate `n_max`:
      - None -> None (unlimited)
      - list/tuple -> must have length 1
      - negative -> None (unlimited)
      - 0 -> 0
      - int-like (including numpy integer) -> int
      - otherwise -> TypeError
    """
    if n_max is None:
        return None

    # Allow sequences but require length 1 (mirrors haven tests)
    if isinstance(n_max, (list, tuple)):
        if len(n_max) != 1:
            raise TypeError("n_max must have length 1")
        n_max = n_max[0]

    # Support numpy integer types without importing numpy unconditionally
    numpy_int = ()
    try:
        import numpy as np  # type: ignore

        numpy_int = (np.integer,)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Booleans are ints in Python; keep that behavior explicit
    if isinstance(n_max, bool):
        n_max = int(n_max)
    elif not isinstance(n_max, (int,) + numpy_int):
        raise TypeError("n_max must be an integer")

    n_max = int(n_max)
    if n_max < 0:
        return None  # unlimited
    return n_max


@contextlib.contextmanager
def _as_path(obj):
    """
    Yield a filesystem path for `obj` (path-like or file-like).
    Cleans up temp files automatically.
    """
    if isinstance(obj, (str, os.PathLike)):
        yield str(obj)
        return

    if hasattr(obj, "read"):
        tmp = tempfile.NamedTemporaryFile(suffix=".dta", delete=False)
        try:
            tmp.write(obj.read())
            tmp.flush()
            tmp.close()
            yield tmp.name
        finally:
            try:
                os.remove(tmp.name)
            except Exception:
                pass
        return

    raise TypeError("data_path must be a path or a file-like object")
