# src/svy/datasets/base.py
"""
Dataset loader.

Consumes the catalog API (``svy.datasets.api``) to resolve a slug into a
``Dataset`` record, ensures the parquet file is cached + verified
(``svy.datasets._cache``), then builds a lazy Polars pipeline for filtering,
column selection, ordering, and slicing.

Loading is intentionally decoupled from survey design: this function returns
a bare ``DataFrame`` (or ``LazyFrame``).  To attach a design and run sampling,
pass the result to ``svy.Sample`` and use ``sample.selection.srs(...)``.
"""

from __future__ import annotations

import logging
import os
import warnings

from typing import Literal, Mapping, Sequence, overload

import numpy as np
import polars as pl

from svy.core.types import Category, RandomState, WhereArg
from svy.datasets import _bundled, _cache, api
from svy.errors.dataset_errors import DatasetError
from svy.utils.where import _compile_where


log = logging.getLogger(__name__)

Source = Literal["auto", "remote", "bundled"]

# Env values that force offline (bundled-first) resolution in "auto" mode.
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _offline_env() -> bool:
    """Whether SVYLAB_OFFLINE requests bundled-first resolution."""
    return os.getenv("SVYLAB_OFFLINE", "").strip().lower() in _TRUTHY


def _scan_remote(name: str, *, force_download: bool) -> pl.LazyFrame:
    """Resolve a slug from the remote catalog, download+cache, and scan it."""
    ds = api.describe(name)
    path = _cache.ensure_cached(
        slug=ds.slug,
        version=ds.version,
        url=ds.download_url,
        sha256=ds.sha256,
        force=force_download,
    )
    return pl.scan_parquet(path)


def _resolve_source(name: str, *, source: Source, force_download: bool) -> pl.LazyFrame:
    """Return a LazyFrame for ``name`` per the requested ``source`` policy.

    - ``"bundled"``: read the packaged subset; never touch the network.
    - ``"remote"``: require the online catalog; error if unreachable.
    - ``"auto"``: bundled-first when ``SVYLAB_OFFLINE`` is set; otherwise try
      remote and, if the catalog/download is unreachable, fall back to the
      bundled subset (with a warning) when one exists.
    """
    if source == "auto" and _offline_env():
        source = "bundled"

    if source == "bundled":
        if not _bundled.has(name):
            raise DatasetError.not_found(where="datasets.load(source='bundled')", slug=name)
        return _bundled.read_lazy(name)

    if source == "remote":
        return _scan_remote(name, force_download=force_download)

    # auto: remote, then bundled fallback on network/catalog failure.
    try:
        return _scan_remote(name, force_download=force_download)
    except DatasetError as exc:
        if exc.code in {"CATALOG_UNREACHABLE", "CATALOG_BAD_STATUS"} and _bundled.has(name):
            b = _bundled.describe(name)
            warnings.warn(
                f"Could not reach the dataset catalog; using the bundled subset of "
                f"{name!r} ({b.n_rows:,} rows). This is a reduced dataset — results "
                f"will differ from the full online data. Use source='remote' once "
                f"online, or set source='bundled' to silence this.",
                stacklevel=3,
            )
            return _bundled.read_lazy(name)
        raise


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


@overload
def load(
    name: str,
    *,
    lazy: Literal[False] = False,
    **kwargs,
) -> pl.DataFrame: ...


@overload
def load(
    name: str,
    *,
    lazy: Literal[True],
    **kwargs,
) -> pl.LazyFrame: ...


def load(
    name: str,
    *,
    n: int | Mapping[Category, int] | None = None,
    rate: float | Mapping[Category, float] | None = None,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    select: Sequence[str] | None = None,
    order_by: str | Sequence[str] | None = None,
    order_type: Literal["ascending", "descending", "random"] = "ascending",
    rstate: RandomState = None,
    lazy: bool = False,
    force_download: bool = False,
    source: Source = "auto",
) -> pl.DataFrame | pl.LazyFrame:
    """
    Load an example dataset from the svylab catalog.

    Parameters
    ----------
    name : str
        Dataset slug, e.g. ``"ea_listing"``.
    n : int or Mapping[Category, int], optional
        Number of rows to return.  With ``by`` set, an int applies per group;
        a mapping gives per-group counts explicitly.  Mutually exclusive with
        ``rate``.
    rate : float or Mapping[Category, float], optional
        Sampling fraction in ``(0, 1]``.  Scalar applies globally or per
        group (with ``by``); a mapping gives per-group fractions.  Mutually
        exclusive with ``n``.
    by : str or Sequence[str], optional
        Grouping column(s) for ``n`` / ``rate``.
    where : WhereArg, optional
        Row filter. Forms accepted:

        - ``pl.col("age") >= 18``                     — Polars expression
        - ``[pl.col("a") > 0, pl.col("b") < 10]``     — list, AND-combined
        - ``{"region": "north"}``                     — scalar equality
        - ``{"region": ["north", "south"]}``          — collection membership
        - ``{"region": "north", "age": [18, 19, 20]}`` — multiple cols, AND-combined

        For anything more complex (between, comparisons, string matching,
        OR combinations), use Polars expressions directly.
    select : Sequence[str], optional
        Columns to keep.  Applied after filtering to maximize Parquet I/O
        savings via column pruning.
    order_by : str or Sequence[str], optional
        Sort key(s) applied before slicing.
    order_type : {"ascending", "descending", "random"}, default "ascending"
        Sort direction.  ``"random"`` uses a reproducible hash-based shuffle.
    rstate : RandomState, optional
        Seed or Generator for reproducible random ordering.
    lazy : bool, default False
        Return a ``LazyFrame`` if True; collect to ``DataFrame`` otherwise.
    force_download : bool, default False
        Re-download the parquet file even if cached.  Applies to remote data
        only; ignored when the bundled subset is used.
    source : {"auto", "remote", "bundled"}, default "auto"
        Where to load the data from.

        - ``"remote"``  — the full online dataset; errors if the catalog is
          unreachable.
        - ``"bundled"`` — the small subset shipped inside the package; never
          touches the network. Deterministic and offline-safe (used by docs
          and CI). Note this is a *reduced* dataset, so results differ from
          the full data.
        - ``"auto"``    — remote first, falling back to the bundled subset
          (with a warning) if the catalog/download is unreachable. Set the
          ``SVYLAB_OFFLINE`` environment variable to make ``"auto"`` prefer
          the bundled subset without any network attempt.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame

    Notes
    -----
    For proper probability sampling (with design-consistent inclusion
    probabilities and weights), use ``svy.Sample(...).selection.srs(...)``
    or the ``pps_*`` methods.  This function only slices for convenience.
    """
    _validate_args(n=n, rate=rate, by=by, order_type=order_type)

    # 1. Resolve the slug to a LazyFrame per the source policy (remote /
    #    bundled / auto-with-offline-fallback).
    lf = _resolve_source(name, source=source, force_download=force_download)

    # Filter first so Polars can push predicates into the Parquet reader.
    if where is not None:
        pred = _compile_where(where)
        if pred is not None:
            lf = lf.filter(pred)

    # Column pruning: let Polars skip reading unneeded columns entirely.
    if select is not None:
        lf = lf.select(list(select))

    # Ordering (or shuffling) — must precede slicing so head(n) is meaningful.
    lf = _apply_order(lf, order_by=order_by, order_type=order_type, rstate=rstate)

    # Slicing — global or per-group via `by`.
    lf = _apply_slice(lf, n=n, rate=rate, by=by)

    return lf if lazy else lf.collect()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate_args(
    *,
    n: int | Mapping[Category, int] | None,
    rate: float | Mapping[Category, float] | None,
    by: str | Sequence[str] | None,
    order_type: str,
) -> None:
    if n is not None and rate is not None:
        raise ValueError("`n` and `rate` are mutually exclusive; pass only one.")
    if by is None and isinstance(n, Mapping):
        raise ValueError("`n` as a mapping requires `by` to be set.")
    if by is None and isinstance(rate, Mapping):
        raise ValueError("`rate` as a mapping requires `by` to be set.")
    if order_type not in ("ascending", "descending", "random"):
        raise ValueError(
            f"order_type must be 'ascending', 'descending', or 'random'; got {order_type!r}"
        )


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def _apply_order(
    lf: pl.LazyFrame,
    *,
    order_by: str | Sequence[str] | None,
    order_type: Literal["ascending", "descending", "random"],
    rstate: RandomState,
) -> pl.LazyFrame:
    if order_type == "random":
        seed = _seed_from_rstate(rstate)
        # Reproducible shuffle: hash(row_index + seed), sort on that.
        # Stays fully lazy and avoids collecting to shuffle.
        return (
            lf.with_row_index("_svy_shuffle_idx")
            .with_columns(
                (pl.col("_svy_shuffle_idx").cast(pl.UInt64) + pl.lit(seed))
                .hash()
                .alias("_svy_shuffle_key")
            )
            .sort("_svy_shuffle_key")
            .drop(["_svy_shuffle_idx", "_svy_shuffle_key"])
        )

    if order_by is None:
        return lf

    return lf.sort(order_by, descending=(order_type == "descending"))


# --------------------------------------------------------------------------- #
# Slicing
# --------------------------------------------------------------------------- #


def _apply_slice(
    lf: pl.LazyFrame,
    *,
    n: int | Mapping[Category, int] | None,
    rate: float | Mapping[Category, float] | None,
    by: str | Sequence[str] | None,
) -> pl.LazyFrame:
    if n is None and rate is None:
        return lf

    by_cols = _normalize_by(by)

    # --- No grouping: global head or fractional head -------------------- #
    if by_cols is None:
        if isinstance(n, int):
            return lf.head(n)
        if isinstance(rate, float):
            _validate_rate(rate)
            # Rate needs a height; collect once.  For teaching datasets this
            # is trivial; power users can apply where/select first to shrink
            # before rate-slicing.
            df = lf.collect()
            return df.head(int(round(len(df) * rate))).lazy()
        return lf  # unreachable given validation, but defensive.

    # --- Per-group slicing ---------------------------------------------- #
    # Polars' lazy API doesn't expose a clean per-group head/sample, so we
    # materialize here.  Filter/select/order have already pruned the frame.
    df = lf.collect()

    if isinstance(n, int):
        return df.group_by(by_cols, maintain_order=True).head(n).lazy()

    if isinstance(n, Mapping):
        return _slice_per_group_mapping(df, by_cols, n, kind="n").lazy()

    if isinstance(rate, float):
        _validate_rate(rate)
        parts = [
            part.head(int(round(len(part) * rate)))
            for _, part in df.group_by(by_cols, maintain_order=True)
        ]
        return (pl.concat(parts) if parts else df.head(0)).lazy()

    if isinstance(rate, Mapping):
        return _slice_per_group_mapping(df, by_cols, rate, kind="rate").lazy()

    return df.lazy()


def _slice_per_group_mapping(
    df: pl.DataFrame,
    by_cols: list[str],
    spec: Mapping[Category, int] | Mapping[Category, float],
    *,
    kind: Literal["n", "rate"],
) -> pl.DataFrame:
    """
    Apply a per-group ``n`` or ``rate`` mapping.

    Keys must match the group identifier exactly: a scalar for single-column
    ``by``, a tuple for multi-column ``by``.  Groups not in the mapping are
    dropped (explicit-is-better-than-implicit).

    For ``rate``, values must be in ``[0, 1]``.  Zero is allowed here (as
    opposed to the global-rate case) to let users express "exclude this
    group" without removing it from the mapping.
    """
    parts: list[pl.DataFrame] = []
    for group_key, part in df.group_by(by_cols, maintain_order=True):
        key = group_key[0] if len(by_cols) == 1 else group_key
        if key not in spec:
            continue
        value = spec[key]
        if kind == "n":
            parts.append(part.head(int(value)))
        else:
            r = float(value)
            if not (0.0 <= r <= 1.0):
                raise ValueError(f"per-group rate must be in [0, 1]; got {r!r} for group {key!r}")
            parts.append(part.head(int(round(len(part) * r))))
    return pl.concat(parts) if parts else df.head(0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalize_by(by: str | Sequence[str] | None) -> list[str] | None:
    if by is None:
        return None
    if isinstance(by, str):
        return [by]
    return list(by)


def _validate_rate(rate: float) -> None:
    if not (0.0 < rate <= 1.0):
        raise ValueError(f"rate must be in (0, 1]; got {rate!r}")


def _seed_from_rstate(rstate: RandomState) -> int:
    """Derive a 64-bit seed from any svy RandomState value."""
    if rstate is None:
        return int(np.random.default_rng().integers(0, 2**63 - 1))
    if isinstance(rstate, int):
        return rstate
    if isinstance(rstate, np.random.Generator):
        return int(rstate.integers(0, 2**63 - 1))
    if isinstance(rstate, np.random.RandomState):
        return int(rstate.randint(0, 2**31 - 1))
    raise TypeError(f"Unsupported rstate type: {type(rstate).__name__}")
