# src/svy/datasets/base.py
from __future__ import annotations

import json
import logging
import os
import shutil

from pathlib import Path
from typing import Any, Optional, Sequence, cast

import httpx
import polars as pl


log = logging.getLogger(__name__)

BASE_URL = os.getenv("SVYLAB_BASE_URL", "http://localhost:8080")

# 1. Define Persistent Cache Directory
# Uses user's home directory to persist files between sessions/reboots
CACHE_DIR = Path.home() / ".svy" / "datasets"


def load_dataset(
    name: str,
    *,
    limit: Optional[int] = 100,  # None => get ALL rows via parquet
    where: dict | list | None = None,  # JSON-friendly filter spec
    select: Optional[Sequence[str]] = None,  # columns to keep
    order_by: Optional[Sequence[str]] = None,  # sort columns
    descending: bool | Sequence[bool] = False,  # bool or per-col flags
    force_local: bool = False,
    force_download: bool = False,  # New: Force re-download even if cached
) -> pl.DataFrame:
    """
    Load an example dataset from svylab by short name (e.g., 'ea_listing') as a Polars DataFrame.

    - If limit is an int (default 100): use preview endpoint (prefers server-side filtering).
    - If limit is None: download parquet (or load from cache), then apply filters locally.
    """
    preview_url = f"{BASE_URL}/api/data/examples/by-name/{name}/preview"
    download_url = f"{BASE_URL}/api/data/examples/by-name/{name}/download"

    def _desc_flags(desc: bool | Sequence[bool], n: int) -> list[bool]:
        if isinstance(desc, bool):
            return [desc] * n
        vals = list(desc)
        if len(vals) != n:
            raise ValueError("Length of `descending` must match `order_by`.")
        return vals

    # ---- Case 1: preview (server-side filters if supported) ----
    if limit is not None and not force_local:
        params: dict[str, Any] = {}
        params["n"] = int(limit)
        if where is not None:
            params["where"] = json.dumps(where, separators=(",", ":"))
        if select:
            params["select"] = ",".join(select)
        if order_by:
            params["order_by"] = ",".join(order_by)
            params["desc"] = ",".join(
                "true" if d else "false" for d in _desc_flags(descending, len(order_by))
            )

        try:
            with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
                r = client.get(preview_url, params=params)
                r.raise_for_status()
                rows = r.json()
            return pl.DataFrame(rows)
        except httpx.HTTPError as e:
            # Fallback to local if server-side preview fails, unless forced otherwise
            log.warning(f"Preview failed ({e}), falling back to full download.")

    # ---- Case 2: full parquet + local (lazy) filtering ----

    # Use cached download logic
    path = _download_parquet_cached(download_url, name, force=force_download)

    # Optimization: Scan parquet allows predicate pushdown (faster than reading all)
    lf = pl.scan_parquet(path)

    if where:
        lf = lf.filter(_to_polars_expr(where))
    if select:
        lf = lf.select([pl.col(c) for c in select])
    if order_by:
        lf = lf.sort(order_by, descending=_desc_flags(descending, len(order_by)))

    return cast(pl.DataFrame, lf.collect())


def _download_parquet_cached(url: str, slug: str, force: bool = False) -> str:
    """
    Downloads file to ~/.svy/datasets/{slug}.parquet if not exists.
    Uses streaming to be memory efficient and fast.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{slug}.parquet"
    local_path = CACHE_DIR / filename

    # 1. Fast Path: Return immediately if exists
    if local_path.exists() and not force:
        log.info(f"Loading cached dataset from: {local_path}")
        return str(local_path)

    log.info(f"Downloading dataset '{slug}' to {local_path} ...")

    # 2. Optimized Streaming Download
    # We download to a temp file first, then move it, to prevent
    # corrupted cache files if the process is interrupted.
    temp_path = local_path.with_suffix(".tmp")

    try:
        with httpx.Client(timeout=None) as client:  # No timeout for large files
            with client.stream("GET", url) as r:
                r.raise_for_status()
                with open(temp_path, "wb") as f:
                    # 128KB chunks is a good balance for speed/memory
                    for chunk in r.iter_bytes(chunk_size=128 * 1024):
                        f.write(chunk)

        # Atomic move (renaming is instant on same filesystem)
        shutil.move(str(temp_path), str(local_path))
        log.info("Download complete.")

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download dataset: {e}") from e

    return str(local_path)


# ---- Local filter translation (safe subset of ops) ----
def _to_polars_expr(where: Any) -> pl.Expr:
    """
    Supported:
      {"col": value}                      # eq
      {"col": (">=", 5)}                 # ge, gt, le, lt, !=, == / "eq"
      {"col": ("in", [1,2,3])}
      {"col": ("between", a, b)}         # inclusive
      {"col": ("contains", "foo")}
      {"col": ("ilike", "%foo%")}        # crude % wildcard
      {"and": [ ... ]} / {"or": [ ... ]}
    """
    if isinstance(where, dict) and ("and" in where or "or" in where):
        keys = "and" if "and" in where else "or"
        parts = [_to_polars_expr(x) for x in where[keys]]
        expr = parts[0]
        for e in parts[1:]:
            expr = (expr & e) if keys == "and" else (expr | e)
        return expr

    if isinstance(where, dict):
        exprs = []
        for col, spec in where.items():
            c = pl.col(col)
            c_txt = c.cast(pl.Utf8, strict=False)  # handles Categorical/String seamlessly

            if not isinstance(spec, (tuple, list)):
                # equality: case-sensitive by default
                exprs.append((c_txt == spec) if isinstance(spec, str) else (c == spec))
                continue

            op = spec[0]
            if op == "in":
                vals = list(spec[1])
                exprs.append(
                    c_txt.is_in(vals) if all(isinstance(v, str) for v in vals) else c.is_in(vals)
                )
            elif op == "iin":  # case-insensitive IN
                vals = [str(v).lower() for v in spec[1]]
                exprs.append(c_txt.str.to_lowercase().is_in(vals))
            elif op in ("==", "eq"):
                rhs = spec[1]
                exprs.append((c_txt == rhs) if isinstance(rhs, str) else (c == rhs))
            elif op in ("eqi",):  # case-insensitive equality
                exprs.append(c_txt.str.to_lowercase() == str(spec[1]).lower())
            elif op == "contains":
                exprs.append(c_txt.str.contains(str(spec[1])))  # case-sensitive
            elif op == "ilike":  # case-insensitive contains with % wildcards
                pat = str(spec[1]).replace("%", ".*")
                exprs.append(c_txt.str.to_lowercase().str.contains(pat.lower()))
            elif op == "between":
                a, b = spec[1], spec[2]
                exprs.append((c >= a) & (c <= b))
            elif op in (">=", "ge"):
                exprs.append(c >= spec[1])
            elif op in (">", "gt"):
                exprs.append(c > spec[1])
            elif op in ("<=", "le"):
                exprs.append(c <= spec[1])
            elif op in ("<", "lt"):
                exprs.append(c < spec[1])
            elif op in ("!=", "ne"):
                exprs.append(c != spec[1])
            elif op in ("==", "eq"):
                exprs.append(c == spec[1])
            elif op == "contains":
                exprs.append(c.cast(pl.Utf8).str.contains(str(spec[1])))
            elif op == "ilike":
                pat = str(spec[1]).replace("%", ".*")
                exprs.append(c.cast(pl.Utf8).str.to_lowercase().str.contains(pat.lower()))
            else:
                raise ValueError(f"Unsupported operator: {op!r}")
        out = exprs[0]
        for e in exprs[1:]:
            out = out & e
        return out

    raise TypeError(f"Invalid where spec: {where!r}")
