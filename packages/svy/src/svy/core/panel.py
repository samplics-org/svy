# src/svy/core/panel.py
"""Panel bookkeeping over a long frame: pairing checks and the wave overlap.

A panel is a long ``Sample`` whose ``Design.case_id`` identifies the followed
entity and whose ``Design.wave`` orders its rows. Nothing here knows about a
``Sample``; the functions take a frame and the case and wave columns so
``combine_samples`` can run them on the raw per-wave frames before concat and
``Sample._validate_design`` can run them on the stacked one.
"""

from __future__ import annotations

from typing import Any, Sequence

import msgspec
import polars as pl

from svy.checks.functions import check_key
from svy.checks.functions import key_values as _ids


__all__ = [
    "WaveOverlap",
    "case_id_cols",
    "wave_overlap",
    "duplicate_case_ids",
    "design_varies_within_case",
]

CaseId = str | Sequence[str]

_WAVE = "__svy_panel_wave"


def case_id_cols(case_id: CaseId) -> list[str]:
    """The case id's columns: one, or several that identify a record together."""
    return [case_id] if isinstance(case_id, str) else list(case_id)


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


def wave_overlap(data: pl.DataFrame, case_id: CaseId, wave: str) -> list[WaveOverlap]:
    """Common / lost / new case counts for every consecutive pair of waves.

    Waves are ordered by their code, the order ``combine_samples`` assigns
    (caller order) and the order a producer's period codes carry.
    """
    cols = case_id_cols(case_id)
    ids = data.select(*cols, wave).drop_nulls().unique()
    waves = sorted(ids.get_column(wave).unique().to_list())
    out: list[WaveOverlap] = []
    for a, b in zip(waves, waves[1:]):
        prev = ids.filter(pl.col(wave) == a).select(cols)
        cur = ids.filter(pl.col(wave) == b).select(cols)
        common = prev.join(cur, on=cols, how="inner").height
        out.append(
            WaveOverlap(
                prev=a, wave=b, common=common, lost=prev.height - common, new=cur.height - common
            )
        )
    return out


def duplicate_case_ids(
    data: pl.DataFrame, case_id: CaseId, wave: str | None, *, limit: int = 10
) -> list[Any]:
    """Case ids appearing more than once within a wave (overall without a wave).

    Values for a one-column id, tuples for an id on several columns.
    """
    cols = case_id_cols(case_id)
    if wave is None:
        return check_key(data, cols, limit=limit).examples
    # A struct is never null, so rows with a null wave still pair up as one wave.
    frame = data.select(*cols, pl.struct(wave).alias(_WAVE))
    examples = check_key(frame, [*cols, _WAVE], limit=frame.height).examples
    ids = [e[0] if len(cols) == 1 else e[:-1] for e in examples]
    return list(dict.fromkeys(ids))[:limit]


def design_varies_within_case(
    data: pl.DataFrame, case_id: CaseId, cols: Sequence[str], *, limit: int = 10
) -> dict[str, list[Any]]:
    """Design columns that take more than one value inside a case.

    Returns ``{column: [offending case ids...]}`` for the violators only. The
    case is nested in its PSU by construction of a panel: a mover keeps the
    base-wave stratum and PSU, because that is where the sampling variance
    comes from.
    """
    id_cols = case_id_cols(case_id)
    cols = [c for c in dict.fromkeys(cols) if c in data.columns and c not in id_cols]
    if not cols:
        return {}
    agg = data.group_by(id_cols).agg([pl.col(c).n_unique().alias(c) for c in cols])
    out: dict[str, list[Any]] = {}
    for c in cols:
        bad = _ids(agg.filter(pl.col(c) > 1).select(id_cols).sort(id_cols).head(limit), id_cols)
        if bad:
            out[c] = bad
    return out
