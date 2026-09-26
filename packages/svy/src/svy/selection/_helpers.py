# src/svy/selection/_helpers.py
"""
Low-level helpers for the selection namespace.

Nothing here touches Sample or Design directly.  Imported by
srs.py, pps.py, multistage.py, and base.py.
"""

from __future__ import annotations

import logging

from typing import Any, Literal, Sequence

import numpy as np
import polars as pl

from svy.core.warnings import Severity


log = logging.getLogger(__name__)


def _psu_list(psu) -> list[str]:
    """Normalise a PSU spec (str | tuple | None) to a list of column names."""
    if psu is None:
        return []
    return [psu] if isinstance(psu, str) else list(psu)


def _apply_order(
    data: pl.DataFrame,
    *,
    order_by: str | Sequence[str] | None,
    order_type: Literal["ascending", "descending", "random"],
    seed: int | None,
) -> pl.DataFrame:
    """
    Apply frame ordering before selection.

    order_type="random" without order_by  -> full shuffle.
    order_type="random" with order_by     -> sort then within-group shuffle
                                             (implicit stratification for
                                             systematic PPS).
    """
    if order_type == "random":
        if order_by is not None:
            sort_cols = [order_by] if isinstance(order_by, str) else list(order_by)
            return (
                data.with_columns(
                    pl.arange(0, pl.len()).shuffle(seed=seed).over(sort_cols).alias("_shuffle_col")
                )
                .sort(*sort_cols, "_shuffle_col")
                .drop("_shuffle_col")
            )
        return data.sample(fraction=1.0, with_replacement=False, shuffle=True, seed=seed)
    if order_by is not None:
        sort_cols = [order_by] if isinstance(order_by, str) else list(order_by)
        return data.sort(by=sort_cols, descending=(order_type == "descending"))
    return data


def _finding(sample: Any, *, code: str, title: str, detail: str, **fields: Any) -> None:
    """Record on the sample being selected from, or warn when there is none.

    An INFO finding confirms what the caller asked for: recorded when there is
    a sample, dropped when there is none.
    """
    if sample is not None:
        sample.warn(code=code, title=title, detail=detail, where="Sample.sampling", **fields)
    elif fields.get("level") != Severity.INFO:
        from svy.core.warnings import warn_no_sample

        warn_no_sample(detail)


def _warn_n_exceeds_population(
    n_map: int | dict[str, int],
    pop_sizes: dict[str, int],
    *,
    wr: bool,
    pps: bool = False,
    sample: Any = None,
) -> None:
    """
    Warn or raise when requested n meets or exceeds the stratum population.

    WR:          UserWarning — duplicates will appear.
    WOR + SRS:   hard ValueError — the draw is geometrically impossible.
    WOR + PPS:   silent — certainty extraction handles n > pop correctly;
                 this was always valid and silent in the original code.

    n=0 for a group is silently skipped (allocation may intentionally omit
    some strata).
    """
    items = [(g, n_map) for g in pop_sizes] if isinstance(n_map, int) else list(n_map.items())
    for group, n_req in items:
        if n_req == 0:
            continue
        pop = pop_sizes.get(group, 0)
        if pop == 0:
            _finding(
                sample,
                code="SELECTION_GROUP_EMPTY",
                title="Group with no frame units",
                detail=(
                    f"Selection: group {group!r} has 0 units in the frame -- "
                    "no records will be drawn."
                ),
                got={"group": group, "n": n_req},
            )
        elif n_req > pop and not wr and not pps:
            raise ValueError(
                f"Selection: requested n={n_req} exceeds the available "
                f"population of {pop} units in group {group!r}. "
                "Use wr=True for with-replacement sampling, or reduce n."
            )
        elif n_req > pop and wr:
            _finding(
                sample,
                code="SELECTION_N_EXCEEDS_GROUP",
                title="Sample size exceeds the group",
                detail=(
                    f"Selection: requested n={n_req} exceeds the population "
                    f"({pop} units) in group {group!r}. "
                    "Sampling with replacement -- some units will appear multiple times."
                ),
                got={"group": group, "n": n_req, "units": pop},
                # What wr=True means, so recorded, not raised.
                level=Severity.INFO,
            )
        # PPS + n > pop: valid — certainty extraction handles this silently.


def _warn_zero_mos(
    mos_arr: np.ndarray,
    group_labels: np.ndarray | None,
    *,
    drop_nulls: bool,
    sample: Any = None,
) -> None:
    """
    Warn when any MOS values are zero or negative before PPS selection.

    Zero-MOS units have pi=0 and can never be drawn.  Surfaced here rather
    than silently ignored so the user knows their frame has a potential issue.
    """
    bad_mask = mos_arr <= 0
    if not bad_mask.any():
        return
    n_bad = int(bad_mask.sum())
    group_info = (
        f" in group(s) {np.unique(group_labels[bad_mask]).tolist()}"
        if group_labels is not None
        else ""
    )
    action = "dropped before selection" if drop_nulls else "present in frame"
    _finding(
        sample,
        code="SELECTION_MOS_NONPOSITIVE",
        title="Units with MOS <= 0",
        detail=(
            f"PPS selection: {n_bad} unit(s) with MOS <= 0 are {action}{group_info}. "
            "These units have zero selection probability and will never be drawn. "
            "Consider removing or imputing them before selecting."
        ),
        got=n_bad,
    )


def _warn_empty_strata(
    n_map: int | dict[str, int],
    pop_sizes: dict[str, int],
    sample: Any = None,
) -> None:
    """
    Warn when a stratum with n > 0 in the allocation has no frame units.

    Typically indicates a mismatch between a user-supplied allocation table
    and the actual data (e.g. a region that exists in controls but not in
    the current frame subset).
    """
    groups_with_n = (
        list(pop_sizes.keys())
        if isinstance(n_map, int)
        else [g for g, v in n_map.items() if v > 0]
    )
    for group in groups_with_n:
        if pop_sizes.get(group, 0) == 0:
            _finding(
                sample,
                code="SELECTION_STRATUM_EMPTY",
                title="Stratum with no frame units",
                detail=(
                    f"Selection: stratum {group!r} has n > 0 in the allocation "
                    "but 0 units in the frame. No records will be drawn for it."
                ),
                got={"stratum": group},
            )
