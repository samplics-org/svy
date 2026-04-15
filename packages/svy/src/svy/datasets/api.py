# from __future__ import annotations

# import os

# import httpx

# from msgspec import json as msgjson

# from .dataset import Dataset


# BASE_URL = os.getenv("SVY_DATASETS_BASE_URL", "https://svylab.com")


# def _fetch(path: str) -> bytes:
#     """Internal HTTP fetch helper."""
#     url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
#     with httpx.Client(follow_redirects=True, timeout=60.0) as client:
#         r = client.get(url)
#         r.raise_for_status()
#         return r.content


# def list_all() -> list[Dataset]:
#     """Return all available datasets from svylab.com."""
#     raw = _fetch("/api/datasets")
#     return msgjson.decode(raw, type=list[Dataset])


# def get(slug: str) -> Dataset:
#     """Return metadata for a single dataset."""
#     raw = _fetch(f"/api/datasets/{slug}")
#     return msgjson.decode(raw, type=Dataset)
