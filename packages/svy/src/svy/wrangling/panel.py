# src/svy/wrangling/panel.py
"""Panel primitives: the lag of a column within a case, across waves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import polars as pl

from svy.core.enumerations import MetadataSource
from svy.errors import DimensionError, MethodError
from svy.wrangling._helpers import _eager_df, _resolve_target


if TYPE_CHECKING:
    from svy.core.sample import Sample


def lag(
    sample: "Sample",
    cols: str | Sequence[str],
    n: int = 1,
    *,
    name: str | Sequence[str] | None = None,
    gaps: Literal["null", "skip"] = "null",
    inplace: bool = False,
) -> "Sample":
    """Add the value of ``cols`` at the case's wave ``n`` steps back.

    The lag is the previous wave in the panel's wave set, not the previous
    observed row: a case that skipped wave 2 gets null on its wave-3 row,
    the way Stata's ``L.y`` does, so a 1 -> 3 move is not counted as a
    one-step transition in a wave 2 -> 3 table. ``gaps="skip"`` opts into the
    previous-observed-row definition. Negative ``n`` is a lead. The first
    wave is null, which a ``where=`` on the later wave excludes naturally.
    """
    ctx = "Sample.wrangling.lag"
    design = sample._design
    if design.case_id is None or design.wave is None:
        raise MethodError.not_applicable(
            where=ctx,
            method="lag",
            reason="lag needs a panel: Design.case_id and Design.wave must both be set",
            hint=(
                "Stack the waves with svy.combine_samples(kind='panel', case_id=...) or "
                "declare Design(case_id=..., wave=...) on the long file."
            ),
        )
    if not isinstance(n, int) or n == 0:
        raise MethodError.invalid_choice(
            where=ctx, param="n", got=n, allowed=["a non-zero int (negative for a lead)"]
        )
    if gaps not in ("null", "skip"):
        raise MethodError.invalid_choice(
            where=ctx, param="gaps", got=gaps, allowed=["null", "skip"]
        )

    col_list = [cols] if isinstance(cols, str) else list(cols)
    if not col_list:
        raise MethodError.invalid_choice(
            where=ctx, param="cols", got=cols, allowed=["column names"]
        )
    df = _eager_df(sample)
    missing = [c for c in col_list if c not in df.columns]
    if missing:
        raise DimensionError.missing_columns(
            where=ctx, param="cols", missing=missing, available=df.columns
        )

    step = "lag" if n > 0 else "lead"
    k = abs(n)
    if name is None:
        names = [f"{c}_{step}{k}" for c in col_list]
    else:
        names = [name] if isinstance(name, str) else list(name)
        if len(names) != len(col_list):
            raise DimensionError(
                title="name/cols length mismatch",
                detail=f"{len(names)} name(s) for {len(col_list)} column(s).",
                code="LAG_NAME_MISMATCH",
                where=ctx,
                param="name",
            )
    taken = [nm for nm in names if nm in df.columns]
    if taken:
        raise MethodError.not_applicable(
            where=ctx,
            method="lag",
            reason=f"column(s) {taken} already exist",
            hint="Pass name= to write the lag under another name.",
        )

    case_id, wave = design.case_id, design.wave
    over = dict(partition_by=case_id, order_by=wave)
    if gaps == "skip":
        exprs = [pl.col(c).shift(n).over(**over).alias(nm) for c, nm in zip(col_list, names)]
        new_data = df.with_columns(exprs)
    else:
        # Rank the wave among the codes present so producer codes (2019,
        # 2021, ...) step by one, then self-join on (case, rank - n): a row
        # shift would hand a case that skipped a wave the wrong wave's value.
        rank = "__svy_wave_rank"
        tmp = df.with_columns(pl.col(wave).rank(method="dense").cast(pl.Int64).alias(rank))
        src = tmp.select(
            pl.col(case_id),
            (pl.col(rank) + n).alias(rank),
            *[pl.col(c).alias(nm) for c, nm in zip(col_list, names)],
        )
        new_data = tmp.join(src, on=[case_id, rank], how="left", maintain_order="left").drop(rank)

    target = _resolve_target(sample, new_data, inplace=inplace)
    meta = target._metadata
    for c, nm in zip(col_list, names):
        src = meta.get(c)
        if src is None:
            continue
        base = src.label or c
        meta.set(
            nm,
            src.clone(name=nm, label=f"{base} ({step} {k})", source=MetadataSource.USER),
        )
    return target
