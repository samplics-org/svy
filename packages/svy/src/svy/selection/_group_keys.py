# src/svy/selection/_group_keys.py
"""
Group-key construction and n normalisation for stratified selection.

_build_group_keys        produces the (G, B, S) universe.
_normalize_n_for_groups  consumes it to resolve scalar/mapping n.
_compute_pop_sizes       counts frame units per group (edge-case guards).

None of these functions touch Sample or Design directly.
"""

from __future__ import annotations

import math
import numbers

from typing import Iterable, Mapping, cast

import polars as pl

from svy.core.constants import key_col
from svy.core.types import DF, Category, Number
from svy.errors import Code, SelectionError


_SEP = "__by__"  # internal composite-key separator


def _unique_as_str(a: Iterable[object]) -> list[str]:
    """De-duplicate while preserving first-seen order, coercing to str."""
    seen: set[str] = set()
    out: list[str] = []
    for v in a:
        s = str(v)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _coerce_key(k: object) -> str:
    """
    Convert any user-supplied n-dict key to the internal string form.

    Tuple keys are joined with _SEP so that users with multi-column
    strata can write natural Python tuples instead of the internal
    "__by__"-joined strings:

        {("Dakar", "Zone1"): 5}  ->  "Dakar__by__Zone1"

    All other keys are str()-coerced unchanged.
    """
    if isinstance(k, tuple):
        return _SEP.join(str(part) for part in k)
    return str(k)


def _build_group_keys(
    data: DF,
    *,
    stratum_col: str | None,
    by_cols: list[str],
    sample_ref,
) -> tuple[
    str | None,
    pl.Series | None,
    list[str],
    list[str],
    list[str],
    pl.DataFrame,
]:
    """
    Build composite group keys for stratified / cross-group selection.

    Parameters
    ----------
    data        : working DataFrame (already column-sliced)
    stratum_col : design stratum column name, or None
    by_cols     : additional grouping columns
    sample_ref  : Sample instance (only used for _concatenate_cols)

    Returns
    -------
    stratum_by_col  : name of the combined (stratum x by) column, or None
    stratum_by_arr  : combined labels per row, or None
    G               : unique combined group keys  (stratum x by)
    B               : unique by-only keys
    S               : unique stratum keys
    data            : DataFrame with synthetic group columns appended
    """
    group_parts: list[str] = []
    if stratum_col is not None:
        group_parts.append(stratum_col)
    group_parts.extend(by_cols)

    stratum_by_col: str | None = None
    stratum_by_arr: pl.Series | None = None

    if group_parts:
        stratum_by_col = key_col("stratum_by")
        if group_parts == [stratum_col]:
            # The stratum key already carries these labels, with nulls tokenized.
            data = data.with_columns(pl.col(stratum_col).alias(stratum_by_col))
        else:
            data = cast(
                pl.DataFrame,
                sample_ref._concatenate_cols(
                    data,
                    sep=_SEP,
                    null_token="__Null__",
                    categorical=True,
                    drop_original=False,
                    stratum_by=group_parts,
                ),
            )
        stratum_by_arr = data[stratum_by_col]

    G = (
        _unique_as_str(data[stratum_by_col].unique(maintain_order=True).to_list())
        if stratum_by_col
        else []
    )

    if by_cols:
        data = cast(
            pl.DataFrame,
            sample_ref._concatenate_cols(
                data,
                sep=_SEP,
                null_token="__Null__",
                categorical=True,
                drop_original=False,
                only_by=by_cols,
            ),
        )
        by_only_col = key_col("only_by")
        B = _unique_as_str(data[by_only_col].unique(maintain_order=True).to_list())
    else:
        B = []

    S = (
        _unique_as_str(data[stratum_col].unique(maintain_order=True).to_list())
        if stratum_col
        else []
    )

    return stratum_by_col, stratum_by_arr, G, B, S, data


def _check_n(n: object) -> None:
    values = list(n.values()) if isinstance(n, Mapping) else [n]
    for v in values:
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise SelectionError.invalid_type(
                where="Sample.sampling",
                param="n",
                got=v,
                expected="a whole number of units, or a mapping of them",
            )
        x = float(v)
        if not (math.isfinite(x) and x >= 0 and x.is_integer()):
            raise SelectionError.ValueError(
                title="Invalid sample size",
                detail=f"n must be a non-negative whole number; got {v!r}.",
                code=Code.SELECTION_N_INVALID,
                where="Sample.sampling",
                param="n",
                got=v,
            )


def _normalize_n_for_groups(
    n: Number | Mapping[Category, Number],
    *,
    G: list[str],
    B: list[str],
    S: list[str],
) -> int | dict[str, int]:
    """
    Normalize ``n`` to match the combined universe G.

    Resolution order
    ----------------
    scalar         -> broadcast to every group in G (plain int if ungrouped)
    keys == G      -> pass-through
    keys == B      -> broadcast within each stratum (same n per by-level in
                      every stratum).  ALL by-levels must be present.
    keys == S      -> broadcast across by (same n per stratum applied to
                      every by-level).  ALL stratum-levels must be present.
    keys == G sub  -> sublevel component matching.  Every G-cell must be
                      matched by exactly one key component.
    ungrouped      -> error if dict (keys are meaningless without grouping)

    Completeness rule
    -----------------
    When a dict is supplied, the keys must *exactly* cover the target
    universe (G, B, or S) -- partial coverage is rejected.  This prevents
    silent zero-allocation for forgotten groups.

    The one exception is zero values: passing ``{"urban": 10, "rural": 0}``
    is valid -- explicit zero is permitted, implicit zero (missing key) is not.

    Tuple key normalisation
    -----------------------
    Tuple keys are joined with "__by__" so users with multi-column strata
    can write ``{("Dakar", "Zone1"): 5}`` instead of ``"Dakar__by__Zone1"``.
    """

    def _broadcast(levels: list[str], v: Number) -> dict[str, int]:
        return {lvl: int(v) for lvl in levels}

    def _coerce(m: Mapping[Category, Number]) -> dict[str, int]:
        return {_coerce_key(k): int(v) for k, v in m.items()}

    def _match_sublevel(n_map: dict[str, int], G: list[str]) -> dict[str, int] | None:
        """
        Match n_map keys as sub-components of composite G keys.

        Rules
        -----
        - Each G cell must be matched by exactly one n_map key component.
        - Every n_map key must match at least one G cell (no orphan keys).
        - Any cell with zero matching keys -> incomplete coverage -> None.
        - Any cell with two+ matching keys -> ambiguous -> None.

        Returns None when no unambiguous, complete match exists.
        """
        n_keys = set(n_map.keys())
        result: dict[str, int] = {}
        matched_keys: set[str] = set()
        for g in G:
            parts = set(g.split(_SEP))
            matches = parts & n_keys
            if len(matches) == 1:
                key = matches.pop()
                result[g] = n_map[key]
                matched_keys.add(key)
            elif len(matches) == 0:
                # A G-cell has no matching key -> incomplete coverage
                return None
            else:
                # Ambiguous: multiple keys match the same cell
                return None
        # Reject orphan keys that matched no cell at all
        orphans = n_keys - matched_keys
        if orphans:
            return None
        return result

    _check_n(n)

    # ── scalar shortcut ─────────────────────────────────────────────────────
    if not isinstance(n, Mapping):
        count = int(n)
        return _broadcast(G, count) if G else count

    # ── dict path ───────────────────────────────────────────────────────────
    n_map = _coerce(cast(Mapping[Category, Number], n))

    # Problem 1: dict with no grouping — keys are meaningless
    if not G:
        raise SelectionError.not_applicable(
            where="Sample.sampling",
            method="srs/pps",
            reason=(
                "A dict was passed for n but the sample has no stratum "
                "and no by= grouping, so the keys have no meaning."
            ),
            hint=(
                "Pass a scalar n to select that many units overall, "
                "or add a stratum to the design or a by= argument."
            ),
        )

    n_keys = set(n_map.keys())
    G_set = set(G)
    B_set = set(B)
    S_set = set(S)

    # ── exact G match ────────────────────────────────────────────────────────
    if n_keys == G_set:
        return n_map

    # Broadcasts cover only the stratum x by cells present in the frame: an
    # absent cell (a PSU of another stratum) is not an empty group.
    # ── exact B match: broadcast within each stratum ─────────────────────────
    if B and n_keys == B_set:
        if not S:
            return n_map
        cells = {f"{s}{_SEP}{b}": n_map[b] for s in S for b in B}
        return {g: cells[g] for g in G}

    # ── exact S match: broadcast across by ───────────────────────────────────
    if S and n_keys == S_set:
        if not B:
            return n_map
        cells = {f"{s}{_SEP}{b}": n_map[s] for s in S for b in B}
        return {g: cells[g] for g in G}

    # ── sublevel component match ──────────────────────────────────────────────
    sub = _match_sublevel(n_map, G)
    if sub is not None:
        return sub

    # ── nothing matched -- build a helpful error ─────────────────────────────
    # Diagnose which universe the user was closest to targeting
    missing_from_G = sorted(G_set - n_keys)
    missing_from_B = sorted(B_set - n_keys) if B else []
    missing_from_S = sorted(S_set - n_keys) if S else []
    extra_keys = sorted(n_keys - G_set - B_set - S_set)

    if missing_from_B and not extra_keys:
        detail = (
            f"Keys look like by-level keys but are incomplete. "
            f"Missing by-levels: {missing_from_B}."
        )
    elif missing_from_S and not extra_keys:
        detail = (
            f"Keys look like stratum keys but are incomplete. Missing strata: {missing_from_S}."
        )
    elif missing_from_G and not extra_keys:
        detail = (
            f"Keys look like full group keys (stratum x by) but are incomplete. "
            f"Missing cells: {missing_from_G}."
        )
    else:
        detail = f"Keys do not match any valid universe. Unrecognised keys: {extra_keys}."

    raise SelectionError.invalid_choice(
        where="Sample.sampling",
        param="n",
        got=sorted(n_keys),
        allowed=G,
        hint=(
            f"{detail} "
            f"n keys must exactly cover one of: "
            f"all combined groups (stratum x by) = {sorted(G_set)}, "
            f"all by-levels = {sorted(B_set)}, "
            f"or all stratum levels = {sorted(S_set)}. "
            "Explicit zero (value=0) is allowed; implicit zero (missing key) is not."
        ),
    )


def _compute_pop_sizes(
    data: pl.DataFrame,
    stratum_by_col: str | None,
    G: list[str],
) -> dict[str, int]:
    """
    Return {group_key: frame_count}.

    When ungrouped, returns {"__all__": len(data)} so that
    _warn_n_exceeds_population can compare against the overall frame size.
    """
    if not G or stratum_by_col is None:
        return {"__all__": len(data)}
    agg = data.group_by(stratum_by_col).agg(pl.len().alias("__count__"))
    counts = dict(zip(agg[stratum_by_col].to_list(), agg["__count__"].to_list()))
    return {g: int(counts.get(g, 0)) for g in G}
