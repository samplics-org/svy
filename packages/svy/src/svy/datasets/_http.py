# src/svy/datasets/_http.py
"""
Lazy access to ``httpx``, the optional dependency behind the online catalog.

``httpx`` ships with the ``remote`` extra (``pip install "svy[remote]"``).  A
base install has no HTTP client, so importing svy never needs it; only a
catalog query or a download does.
"""

from __future__ import annotations

from types import ModuleType

from svy.errors.dataset_errors import DatasetError


def load_httpx(*, where: str) -> ModuleType:
    """Return the ``httpx`` module, or raise ``REMOTE_UNAVAILABLE``."""
    try:
        import httpx
    except ImportError as e:
        raise DatasetError.remote_unavailable(where=where) from e
    return httpx
