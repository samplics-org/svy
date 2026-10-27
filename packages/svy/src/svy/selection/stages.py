# src/svy/selection/stages.py
"""Several stages drawn in one call from a frame that holds every stage's ids."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

import msgspec
import polars as pl

from svy.core.constants import SVY_PROB, SVY_WEIGHT
from svy.errors import Code, SelectionError
from svy.errors._findings import Severity
from svy.selection._group_keys import _SEP, _build_group_keys, _normalize_n_for_groups
from svy.selection._helpers import _finding
from svy.selection.combine_stages import _stage_col
from svy.utils.random_state import RandomState, resolve_random_state


if TYPE_CHECKING:
    from svy.core.sample import Sample

_PPS = ("sys", "brewer", "murphy", "rs")
_ROW = "__svy_ms_row__"
_PARENT = "__svy_ms_parent__"
_SKEY = "__svy_ms_stratum__"


class Stage(msgspec.Struct, frozen=True):
    """
    One stage of :meth:`Selection.multistage`.

    Parameters
    ----------
    unit : str, tuple of str or None
        Id column(s) of this stage's units, unique within the unit selected before; ``None``
        (last stage only) draws the frame's rows.
    n : int, mapping or None
        Units per stratum at stage 1; units per selected parent after; ``None`` takes all.
        A parent with fewer units gives all of them. At stage 1 a mapping is keyed as for
        :meth:`Selection.srs`: by stratum (a tuple for several stratum columns), or by one
        stratum variable, e.g. ``{"Urban": 6, "Rural": 4}`` for ``("region", "urbrur")``.
    mos : str, optional
        Size measure, constant within a unit; given, the stage is PPS.
    method : {"srs", "sys", "brewer", "murphy", "rs"}, optional
        ``"srs"`` without ``mos``, ``"sys"`` with it by default.
    certainty_threshold : float, optional
        PPS only: units whose inclusion probability reaches this are taken with certainty.
    """

    unit: str | tuple[str, ...] | None
    n: int | Mapping[Any, int] | None = None
    mos: str | None = None
    method: Literal["srs", "sys", "brewer", "murphy", "rs"] | None = None
    certainty_threshold: float | None = None

    def __post_init__(self) -> None:
        unit, n, mos, method = self.unit, self.n, self.mos, self.method
        if isinstance(unit, (list, tuple)):
            unit = tuple(unit)
            if not unit or not all(isinstance(u, str) for u in unit):
                raise _stage_error("unit", unit, "a column name, a tuple of names, or None")
            msgspec.structs.force_setattr(self, "unit", unit[0] if len(unit) == 1 else unit)
        elif unit is not None and not isinstance(unit, str):
            raise _stage_error("unit", unit, "a column name, a tuple of names, or None")
        if n is not None and not isinstance(n, Mapping):
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise _stage_error("n", n, "a non-negative int, a mapping, or None")
        if method is None:
            method = "sys" if mos is not None else "srs"
            msgspec.structs.force_setattr(self, "method", method)
        if method not in ("srs", *_PPS):
            raise _stage_error("method", method, "'srs', 'sys', 'brewer', 'murphy' or 'rs'")
        if method == "srs" and mos is not None:
            raise _stage_error("method", method, "a PPS method with mos=, or no mos")
        if method != "srs" and mos is None:
            raise _stage_error("mos", mos, f"a size column for method={method!r}")
        if self.certainty_threshold is not None and method == "srs":
            raise _stage_error("certainty_threshold", self.certainty_threshold, "a PPS stage")


def _stage_error(param: str, got: Any, expected: str) -> SelectionError:
    return SelectionError.ValueError(
        title="Invalid stage",
        detail=f"Stage {param}= got {got!r}; expected {expected}.",
        code=Code.INVALID_STAGE,
        where="svy.selection.Stage",
        param=param,
        got=got,
        expected=expected,
    )


def _cols(spec: str | Sequence[str] | None) -> list[str]:
    if spec is None:
        return []
    return [spec] if isinstance(spec, str) else list(spec)


def _as_spec(cols: list[str]) -> str | tuple[str, ...]:
    return cols[0] if len(cols) == 1 else tuple(cols)


def multistage(
    sample: "Sample",
    stages: Sequence[Stage],
    *,
    rstate: RandomState = None,
) -> "Sample":
    """Draw ``stages`` in order, each within the units the one before selected."""
    stages = list(stages)
    _check_stages(sample, stages)
    design = sample.design
    strata = _cols(design.stratum)
    gen = resolve_random_state(rstate)

    rows = sample._data.with_row_index(_ROW)
    n1 = stages[0].n
    n_map = _stage1_n(sample, rows, n1) if isinstance(n1, Mapping) else None
    # Units are keyed by the chain of ids above them, strata first: an id need only be
    # unique within its parent.
    chain: list[str] = list(strata)
    prob_cols: list[str] = []
    for k, stage in enumerate(stages, start=1):
        unit_cols = [_ROW] if stage.unit is None else _cols(stage.unit)
        parents = strata if k == 1 else chain
        keys = chain + [c for c in unit_cols if c not in chain]
        probs = _draw_stage(
            sample,
            rows,
            stage,
            k,
            parents=parents,
            keys=keys,
            n_map=n_map if k == 1 else None,
            gen=gen,
        )
        prob_col = _stage_col(SVY_PROB, k)
        rows = rows.join(probs.rename({SVY_PROB: prob_col}), on=keys, how="inner")
        prob_cols.append(prob_col)
        chain = keys

    product = pl.lit(1.0)
    for col in prob_cols:
        product = product * pl.col(col)
    rows = (
        rows.with_columns(
            *[
                (1.0 / pl.col(c)).alias(_stage_col(SVY_WEIGHT, k))
                for k, c in enumerate(prob_cols, 1)
            ],
            product.alias(SVY_PROB),
        )
        .with_columns((1.0 / pl.col(SVY_PROB)).alias(SVY_WEIGHT))
        .sort(_ROW)
        .drop(_ROW)
    )

    psu = _cols(stages[0].unit)
    ssu = _cols(stages[1].unit) if len(stages) > 1 else []
    sample._data = rows
    sample._design = design.update(
        psu=_as_spec(psu),
        ssu=_as_spec(ssu) if ssu else None,
        prob=SVY_PROB,
        wgt=SVY_WEIGHT,
        wr=False,
    )
    return sample


def _check_stages(sample: "Sample", stages: list[Stage]) -> None:
    if not stages or not all(isinstance(s, Stage) for s in stages):
        raise SelectionError.ValueError(
            title="multistage needs a list of stages",
            detail="Pass one svy.selection.Stage per stage, in order.",
            code=Code.INVALID_STAGE,
            where="Sample.sampling.multistage",
            param="stages",
            got=stages,
        )
    if stages[0].unit is None:
        raise _stage_error("unit", None, "an id column at stage 1 (its units are the PSUs)")
    for k, stage in enumerate(stages[:-1], start=1):
        if stage.unit is None:
            raise _stage_error(
                "unit", None, f"an id column at stage {k}; only the last stage may be None"
            )
    for k, stage in enumerate(stages[1:], start=2):
        if isinstance(stage.n, Mapping):
            raise _stage_error(
                "n", stage.n, f"an int or None at stage {k} (n per selected parent)"
            )
    design = sample.design
    if design.prob is not None or design.wgt is not None:
        raise SelectionError.ValueError(
            title="Frame already has selection weights",
            detail=(
                "multistage draws every stage from a frame; this sample's design already "
                f"has prob={design.prob!r} and wgt={design.wgt!r}."
            ),
            code=Code.MULTISTAGE_ALREADY_SELECTED,
            where="Sample.sampling.multistage",
            hint="Start from the frame, or add a stage to a drawn sample with add_stage.",
        )
    columns = set(sample._data.columns)
    needed = {c for s in stages for c in (*_cols(s.unit), *_cols(s.mos))}
    missing = sorted(needed - columns)
    if missing:
        raise SelectionError.KeyError(
            title="Stage column not in the frame",
            detail=f"The stages use {missing!r}, which the frame does not have.",
            code=Code.MULTISTAGE_COLUMN_MISSING,
            where="Sample.sampling.multistage",
            got=missing,
            expected=sorted(columns),
        )
    out = {SVY_PROB, SVY_WEIGHT}
    for k in range(1, len(stages) + 1):
        out |= {_stage_col(SVY_PROB, k), _stage_col(SVY_WEIGHT, k)}
    taken = sorted(out & columns)
    if taken:
        raise SelectionError.ValueError(
            title="Output column already in the frame",
            detail=f"multistage writes {taken!r}, which the frame already has.",
            code=Code.MULTISTAGE_COLUMN_TAKEN,
            where="Sample.sampling.multistage",
            got=taken,
            hint="Rename or drop them first.",
        )


def _draw_stage(
    sample: "Sample",
    rows: pl.DataFrame,
    stage: Stage,
    k: int,
    *,
    parents: list[str],
    keys: list[str],
    n_map: dict[str, int] | None,
    gen: Any,
) -> pl.DataFrame:
    """Each selected unit of stage ``k`` with its probability given its parent."""
    import svy

    attrs = [stage.mos] if stage.mos else []
    _check_constant(rows, keys, attrs, k)

    skey = [sample._internal_design["stratum"]] if n_map is not None else []
    units = rows.select(list(dict.fromkeys(parents + keys + attrs + skey))).unique(
        subset=keys, maintain_order=True
    )
    if parents:
        units = units.with_columns(pl.struct(parents).rank("dense").cast(pl.Int64).alias(_PARENT))
    else:
        units = units.with_columns(pl.lit(1, dtype=pl.Int64).alias(_PARENT))
    sizes = units.group_by(_PARENT, maintain_order=True).agg(
        pl.len().alias("N"),
        *[pl.first(c) for c in parents],
        *[pl.first(c).cast(pl.String).alias(_SKEY) for c in skey],
    )

    want: dict[int, int] = {}
    for rec in sizes.iter_rows(named=True):
        want[rec[_PARENT]] = _n_for(stage, rec, n_map)
    capped = [
        {c: rec[c] for c in parents}
        for rec in sizes.iter_rows(named=True)
        if stage.n is not None and want[rec[_PARENT]] > rec["N"]
    ]
    if capped:
        what = "rows" if stage.unit is None else ", ".join(_cols(stage.unit))
        named = [", ".join(f"{c}={v}" for c, v in p.items()) or "the whole frame" for p in capped]
        _finding(
            sample,
            code=Code.STAGE_TAKES_ALL,
            title="Stage takes every unit of some parents",
            detail=(
                f"Stage {k} ({what}) asks for more units than {len(capped)} parent(s) hold, "
                "so all their units are taken, with probability 1: "
                + "; ".join(named[:10])
                + (f"; and {len(named) - 10} more" if len(named) > 10 else "")
            ),
            level=Severity.INFO,
            param="n",
            got=capped,
        )

    n_by_parent = dict(zip(sizes[_PARENT].to_list(), sizes["N"].to_list()))
    take_all = [p for p, n in want.items() if n >= n_by_parent[p]]
    draw = {p: n for p, n in want.items() if 0 < n < n_by_parent[p]}

    parts = [
        units.filter(pl.col(_PARENT).is_in(take_all))
        .select(keys)
        .with_columns(pl.lit(1.0).alias(SVY_PROB))
    ]
    if draw:
        frame = units.filter(pl.col(_PARENT).is_in(list(draw))).select(
            list(dict.fromkeys([_PARENT, *keys, *attrs]))
        )
        tmp = svy.Sample(frame, svy.Design(stratum=_PARENT, mos=stage.mos))
        if stage.method == "srs":
            tmp.sampling.srs(draw, rstate=gen)
        else:
            getattr(tmp.sampling, f"pps_{stage.method}")(
                draw, rstate=gen, certainty_threshold=stage.certainty_threshold
            )
        parts.append(tmp._data.select(keys + [SVY_PROB]))
    return pl.concat(parts, how="vertical_relaxed")


def _n_for(stage: Stage, rec: dict, n_map: dict[str, int] | None) -> int:
    if stage.n is None:
        return rec["N"]
    if n_map is None:
        return cast(int, stage.n)
    return n_map[rec[_SKEY]]


def _stage1_n(sample: "Sample", rows: pl.DataFrame, n: Mapping[Any, int]) -> dict[str, int]:
    """Stage-1 ``n`` keyed by stratum key, resolved as ``srs`` and ``pps_*`` resolve it."""
    skey = sample._internal_design["stratum"]
    strata: list[str] = []
    if skey is not None:
        *_, strata, _ = _build_group_keys(
            rows.select(skey), stratum_col=skey, by_cols=[], sample_ref=sample
        )
    try:
        return cast(dict[str, int], _normalize_n_for_groups(n, G=strata, B=[], S=strata))
    except SelectionError:
        pass
    shown = [tuple(s.split(_SEP)) if _SEP in s else s for s in strata]
    raise SelectionError.ValueError(
        title="Stage 1 n= does not match the strata",
        detail=(
            f"Stage 1 n= keys must give each stratum exactly one size; strata: {shown[:10]!r}"
            + (" ..." if len(shown) > 10 else "")
            if strata
            else "The design has no stratum, so stage 1 n= cannot be a mapping."
        ),
        code=Code.MULTISTAGE_N_MISSING,
        where="Sample.sampling.multistage",
        param="n",
        got=sorted(map(repr, n)),
        hint=(
            "Key n by stratum (a tuple for several stratum columns) or by one stratum "
            "variable, and give every stratum a size (0 skips one)."
            if strata
            else "Pass an int, or add a stratum to the design."
        ),
    )


def _check_constant(rows: pl.DataFrame, keys: list[str], cols: list[str], k: int) -> None:
    """``mos`` must take one value per unit."""
    for col in dict.fromkeys(cols):
        bad = rows.group_by(keys).agg(pl.col(col).n_unique().alias("_k")).filter(pl.col("_k") > 1)
        if bad.height:
            unit = bad.select(keys).row(0)
            raise SelectionError.ValueError(
                title="Value varies within a unit",
                detail=(
                    f"At stage {k}, {col!r} takes several values within unit {unit!r} "
                    f"({bad.height} unit(s) in all); it must be one value per unit."
                ),
                code=Code.MULTISTAGE_NOT_CONSTANT,
                where="Sample.sampling.multistage",
                param=col,
                got=unit,
            )
