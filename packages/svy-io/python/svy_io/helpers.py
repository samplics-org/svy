# python/svy_io/helpers.py
import contextlib
import os

from typing import Any


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
