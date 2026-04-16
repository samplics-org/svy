# src/svy/selection/_group_keys.py
"""
Group-key construction and n normalisation for stratified selection.

_build_group_keys        produces the (G, B, S) universe.
_normalize_n_for_groups  consumes it to resolve scalar/mapping n.
_compute_pop_sizes       counts frame units per group (edge-case guards).

None of these functions touch Sample or Design directly.
"""

from __future__ import annotations

from typing import Iterable, Mapping, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from svy.core.types import DF, Category, Number
from svy.errors import MethodError


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
    suffix: str,
    sample_ref,
) -> tuple[
    str | None,
    npt.NDArray[np.object_] | None,
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
    suffix      : internal suffix used by Sample._concatenate_cols
    sample_ref  : Sample instance (only used for _concatenate_cols)

    Returns
    -------
    stratum_by_col  : name of the combined (stratum x by) column, or None
    stratum_by_arr  : numpy array of combined labels per row, or None
    G               : unique combined group keys  (stratum x by)
    B               : unique by-only keys
    S               : unique stratum keys
    data            : DataFrame with synthetic group columns appended
    """
    if isinstance(data, pl.LazyFrame):
        data = cast(pl.DataFrame, data.collect())

    group_parts: list[str] = []
    if stratum_col is not None:
        group_parts.append(stratum_col)
    group_parts.extend(by_cols)

    stratum_by_col: str | None = None
    stratum_by_arr: npt.NDArray[np.object_] | None = None

    if group_parts:
        data = cast(
            pl.DataFrame,
            sample_ref._concatenate_cols(
                data,
                sep=_SEP,
                null_token="__Null__",
                categorical=True,
                drop_original=False,
                rename_suffix=suffix,
                stratum_by=group_parts,
            ),
        )
        stratum_by_col = f"stratum_by{suffix}"
        stratum_by_arr = data[stratum_by_col].to_numpy()

    G = _unique_as_str(data[stratum_by_col].unique().to_list()) if stratum_by_col else []

    if by_cols:
        data = cast(
            pl.DataFrame,
            sample_ref._concatenate_cols(
                data,
                sep=_SEP,
                null_token="__Null__",
                categorical=True,
                drop_original=False,
                rename_suffix=suffix,
                only_by=by_cols,
            ),
        )
        by_only_col = f"only_by{suffix}"
        B = _unique_as_str(data[by_only_col].unique().to_list())
    else:
        B = []

    S = _unique_as_str(data[stratum_col].unique().to_list()) if stratum_col else []

    return stratum_by_col, stratum_by_arr, G, B, S, data


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

    # ── scalar shortcut ─────────────────────────────────────────────────────
    if isinstance(n, (int, float)):
        return _broadcast(G, n) if G else int(n)

    # ── dict path ───────────────────────────────────────────────────────────
    n_map = _coerce(n)

    # Problem 1: dict with no grouping — keys are meaningless
    if not G:
        raise MethodError.not_applicable(
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

    # ── exact B match: broadcast within each stratum ─────────────────────────
    if B and n_keys == B_set:
        return {f"{s}{_SEP}{b}": n_map[b] for s in S for b in B} if S else n_map

    # ── exact S match: broadcast across by ───────────────────────────────────
    if S and n_keys == S_set:
        return {f"{s}{_SEP}{b}": n_map[s] for s in S for b in B} if B else n_map

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

    raise MethodError.invalid_choice(
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
    counts = (
        data.group_by(stratum_by_col)
        .agg(pl.len().alias("__count__"))
        .to_pandas()
        .set_index(stratum_by_col)["__count__"]
        .to_dict()
    )
    return {g: int(counts.get(g, 0)) for g in G}
