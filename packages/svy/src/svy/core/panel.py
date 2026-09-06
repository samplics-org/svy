# src/svy/core/panel.py
"""Panel bookkeeping over a long frame: pairing checks and the wave overlap.

A panel is a long ``Sample`` whose ``Design.case_id`` identifies the followed
entity and whose ``Design.wave`` orders its rows. Nothing here knows about a
``Sample``; the functions take a frame and two column names so
``combine_samples`` can run them on the raw per-wave frames before concat and
``Sample._validate_design`` can run them on the stacked one.
"""

from __future__ import annotations

from typing import Any, Sequence

import msgspec
import polars as pl


__all__ = ["WaveOverlap", "wave_overlap", "duplicate_case_ids", "design_varies_within_case"]


class WaveOverlap(msgspec.Struct, frozen=True):
    """Case overlap between two consecutive waves."""

    prev: Any
    wave: Any
    common: int
    lost: int
    new: int

    def __str__(self) -> str:
        return (
            f"{self.prev} -> {self.wave}: {self.common} common, {self.lost} lost, {self.new} new"
        )


def wave_overlap(data: pl.DataFrame, case_id: str, wave: str) -> list[WaveOverlap]:
    """Common / lost / new case counts for every consecutive pair of waves.

    Waves are ordered by their code, the order ``combine_samples`` assigns
    (caller order) and the order a producer's period codes carry.
    """
    ids = data.select(case_id, wave).drop_nulls().unique()
    waves = sorted(ids.get_column(wave).unique().to_list())
    out: list[WaveOverlap] = []
    for a, b in zip(waves, waves[1:]):
        prev = ids.filter(pl.col(wave) == a).select(case_id)
        cur = ids.filter(pl.col(wave) == b).select(case_id)
        common = prev.join(cur, on=case_id, how="inner").height
        out.append(
            WaveOverlap(
                prev=a, wave=b, common=common, lost=prev.height - common, new=cur.height - common
            )
        )
    return out


def duplicate_case_ids(
    data: pl.DataFrame, case_id: str, wave: str | None, *, limit: int = 10
) -> list[Any]:
    """Case ids appearing more than once within a wave (overall without a wave)."""
    keys = [case_id] if wave is None else [case_id, wave]
    dup = data.group_by(keys).len().filter(pl.col("len") > 1)
    return dup.get_column(case_id).unique().sort().head(limit).to_list()


def design_varies_within_case(
    data: pl.DataFrame, case_id: str, cols: Sequence[str], *, limit: int = 10
) -> dict[str, list[Any]]:
    """Design columns that take more than one value inside a case.

    Returns ``{column: [offending case ids...]}`` for the violators only. The
    case is nested in its PSU by construction of a panel: a mover keeps the
    base-wave stratum and PSU, because that is where the sampling variance
    comes from.
    """
    cols = [c for c in cols if c in data.columns]
    if not cols:
        return {}
    agg = data.group_by(case_id).agg([pl.col(c).n_unique().alias(c) for c in cols])
    out: dict[str, list[Any]] = {}
    for c in cols:
        bad = agg.filter(pl.col(c) > 1).get_column(case_id).sort().head(limit).to_list()
        if bad:
            out[c] = bad
    return out
