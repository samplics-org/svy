# src/svy/core/describe_runtime.py
from __future__ import annotations

import datetime as dt
import logging

from typing import TYPE_CHECKING, Sequence

import polars as pl

from .describe import (
    DescribeBoolean,
    DescribeContinuous,
    DescribeDatetime,
    DescribeDiscrete,
    DescribeItem,
    DescribeNominal,
    DescribeOrdinal,
    DescribeResult,
    DescribeString,
    Freq,
    Quantile,
)
from .enumerations import MeasurementType


if TYPE_CHECKING:
    from svy.metadata import MetadataStore


log = logging.getLogger(__name__)

DEFAULT_PERCENTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


def _infer_mtype_from_polars(dt: pl.DataType) -> MeasurementType:
    base = dt.base_type()
    if base.is_float():
        return MeasurementType.CONTINUOUS
    if base.is_integer():
        return MeasurementType.DISCRETE
    if base == pl.Boolean:
        return MeasurementType.BOOLEAN
    if base == pl.String:
        return MeasurementType.STRING
    if base == pl.Categorical or base == pl.Enum:
        return MeasurementType.NOMINAL
    if base.is_temporal():
        return MeasurementType.DATETIME
    return MeasurementType.STRING


def _resolve_columns(df: pl.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        return list(df.columns)
    out: list[str] = []
    names = set(df.columns)
    for c in columns:
        if c not in names:
            raise KeyError(f"Column {c!r} not found in data.")
        out.append(c)
    return out


def _series_nonnull(df: pl.DataFrame, col: str, *, drop_nulls: bool) -> pl.Series:
    s = df.get_column(col)
    return s.drop_nulls() if drop_nulls else s


def _series_missing_and_clean(s: pl.Series) -> tuple[int, pl.Series]:
    n_missing = int(s.null_count())
    clean = s.drop_nulls()
    return n_missing, clean


def _weight_series(df: pl.DataFrame, weight_col: str | None) -> pl.Series | None:
    if weight_col is None:
        return None
    if weight_col not in df.columns:
        raise KeyError(f"Weight column {weight_col!r} not found.")
    return df.get_column(weight_col).cast(pl.Float64)


def _to_float(x: object) -> float | None:
    """Safely convert a polars aggregate result to float, returning None for non-numeric."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _to_int(x: object) -> int | None:
    """Safely convert a polars aggregate result to int, returning None for non-numeric."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    return None


def _to_temporal(x: object) -> dt.datetime | dt.date | dt.time | None:
    """Cast a polars min/max result to datetime/date/time or None."""
    if x is None:
        return None
    if isinstance(x, (dt.datetime, dt.date, dt.time)):
        return x
    return None


def _desc_continuous(
    df: pl.DataFrame,
    col: str,
    mtype: MeasurementType,
    *,
    weighted: bool,
    weight_col: str | None,
    drop_nulls: bool,
    percentiles: Sequence[float],
) -> DescribeContinuous:
    s_base = df.get_column(col).cast(pl.Float64)
    n_total = df.height
    n_missing, s = _series_missing_and_clean(s_base)

    mean = _to_float(s.mean()) if s.len() else None
    std = _to_float(s.std()) if s.len() else None
    var = _to_float(s.var()) if s.len() else None
    vmin = _to_float(s.min()) if s.len() else None
    vmax = _to_float(s.max()) if s.len() else None
    p25 = _to_float(s.quantile(0.25)) if s.len() else None
    p50 = _to_float(s.quantile(0.50)) if s.len() else None
    p75 = _to_float(s.quantile(0.75)) if s.len() else None
    ssum = _to_float(s.sum()) if s.len() else None

    q_items: list[Quantile] = []
    for p in percentiles:
        try:
            qv = None if weighted else (_to_float(s.quantile(p)) if s.len() else None)
        except Exception:
            qv = None
        q_items.append(Quantile(p=float(p), value=qv))

    if weighted:
        w = _weight_series(df, weight_col)
        if w is not None:
            mask = ~s_base.is_null()
            w_nonnull = w.filter(mask)
            denom = _to_float(w_nonnull.sum()) if w_nonnull.len() else 0.0
            if denom and denom > 0 and s.len():
                ssum = _to_float((s * w_nonnull).sum())
                mean = (ssum / denom) if ssum is not None else None

    return DescribeContinuous(
        name=col,
        mtype=mtype,
        n=n_total,
        n_missing=n_missing,
        weighted=weighted,
        drop_nulls=drop_nulls,
        mean=mean,
        std=std,
        var=var,
        min=vmin,
        p25=p25,
        p50=p50,
        p75=p75,
        max=vmax,
        sum=ssum,
        percentiles=tuple(q_items),
    )


def _desc_discrete(
    df: pl.DataFrame,
    col: str,
    mtype: MeasurementType,
    *,
    weighted: bool,
    weight_col: str | None,
    drop_nulls: bool,
    top_k: int,
    percentiles: Sequence[float],
) -> DescribeDiscrete:
    cont = _desc_continuous(
        df,
        col,
        mtype,
        weighted=weighted,
        weight_col=weight_col,
        drop_nulls=drop_nulls,
        percentiles=percentiles,
    )

    s_base = df.get_column(col)
    if drop_nulls:
        mask = ~s_base.is_null()
        s = s_base.filter(mask)
        w = _weight_series(df, weight_col)
        if w is not None:
            w = w.filter(mask)
    else:
        s = s_base
        w = _weight_series(df, weight_col)

    if s.len() == 0:
        levels: tuple[Freq, ...] = ()
    else:
        if weighted and w is not None:
            tbl = (
                pl.DataFrame({col: s, "_w": w})
                .group_by(col)
                .agg(pl.col("_w").sum().alias("count"))
                .sort("count", descending=True)
                .head(top_k)
            )
            denom = _to_float(tbl["count"].sum()) if tbl.height else 0.0
            levels = tuple(
                Freq(
                    level=row[col],
                    count=float(row["count"]),
                    prop=(float(row["count"]) / denom if denom and denom > 0 else 0.0),
                )
                for row in tbl.iter_rows(named=True)
            )
        else:
            tbl = (
                pl.DataFrame({col: s})
                .group_by(col)
                .len()
                .rename({"len": "count"})
                .sort("count", descending=True)
                .head(top_k)
            )
            denom = float(s.len())
            levels = tuple(
                Freq(
                    level=row[col],
                    count=float(row["count"]),
                    prop=(float(row["count"]) / denom if denom > 0 else 0.0),
                )
                for row in tbl.iter_rows(named=True)
            )

    return DescribeDiscrete(
        name=cont.name,
        mtype=cont.mtype,
        n=cont.n,
        n_missing=cont.n_missing,
        weighted=cont.weighted,
        drop_nulls=cont.drop_nulls,
        mean=cont.mean,
        std=cont.std,
        var=cont.var,
        min=cont.min,
        p25=cont.p25,
        p50=cont.p50,
        p75=cont.p75,
        max=cont.max,
        sum=cont.sum,
        levels=levels,
    )


def _build_freqs(
    s: pl.Series,
    *,
    weighted: bool,
    w: pl.Series | None,
    top_k: int,
) -> tuple[Freq, ...]:
    if s.len() == 0:
        return ()
    if weighted and w is not None:
        tbl = (
            pl.DataFrame({"__k": s, "__w": w})
            .group_by("__k")
            .agg(pl.col("__w").sum().alias("count"))
            .sort("count", descending=True)
            .head(top_k)
        )
        denom = _to_float(tbl["count"].sum()) if tbl.height else 0.0
        return tuple(
            Freq(
                level=row["__k"],
                count=float(row["count"]),
                prop=(float(row["count"]) / denom if denom and denom > 0 else 0.0),
            )
            for row in tbl.iter_rows(named=True)
        )
    else:
        tbl = (
            pl.DataFrame({"__k": s})
            .group_by("__k")
            .len()
            .rename({"len": "count"})
            .sort("count", descending=True)
            .head(top_k)
        )
        denom = float(s.len())
        return tuple(
            Freq(
                level=row["__k"],
                count=float(row["count"]),
                prop=(float(row["count"]) / denom if denom > 0 else 0.0),
            )
            for row in tbl.iter_rows(named=True)
        )


def _desc_nominal_or_ordinal(
    df: pl.DataFrame,
    col: str,
    mtype: MeasurementType,
    *,
    weighted: bool,
    weight_col: str | None,
    drop_nulls: bool,
    top_k: int,
    ordinal: bool,
) -> DescribeNominal | DescribeOrdinal:
    s_base = df.get_column(col)
    if drop_nulls:
        mask = ~s_base.is_null()
        s = s_base.filter(mask)
        w = _weight_series(df, weight_col)
        if w is not None:
            w = w.filter(mask)
    else:
        s = s_base
        w = _weight_series(df, weight_col)

    n_total = df.height
    n_missing = int(s_base.null_count())

    levels = _build_freqs(s, weighted=weighted, w=w, top_k=top_k)
    n_levels = int(len(levels))
    mode = levels[0].level if levels else None

    if ordinal:
        return DescribeOrdinal(
            name=col,
            mtype=MeasurementType.ORDINAL,
            n=n_total,
            n_missing=n_missing,
            weighted=weighted,
            drop_nulls=drop_nulls,
            levels=levels,
            n_levels=n_levels,
            mode=mode,
            truncated=False,
        )
    return DescribeNominal(
        name=col,
        mtype=MeasurementType.NOMINAL,
        n=n_total,
        n_missing=n_missing,
        weighted=weighted,
        drop_nulls=drop_nulls,
        levels=levels,
        n_levels=n_levels,
        mode=mode,
        truncated=False,
    )


def _desc_boolean(
    df: pl.DataFrame,
    col: str,
    mtype: MeasurementType,
    *,
    weighted: bool,
    weight_col: str | None,
    drop_nulls: bool,
) -> DescribeBoolean:
    s_base = df.get_column(col).cast(pl.Boolean)
    if drop_nulls:
        mask = ~s_base.is_null()
        s = s_base.filter(mask)
        w = _weight_series(df, weight_col)
        if w is not None:
            w = w.filter(mask)
    else:
        s = s_base
        w = _weight_series(df, weight_col)

    n_total = df.height
    n_missing = int(s_base.null_count())

    if s.len() == 0:
        f_false = None
        f_true = None
    else:
        if weighted and w is not None:
            c_true = float(
                pl.DataFrame({"b": s.cast(pl.Int8), "w": w})
                .with_columns((pl.col("b") * pl.col("w")).alias("wt"))
                .select(pl.col("wt").sum())
                .item()
            )
            c_false = float(w.sum()) - c_true
            denom = c_true + c_false
        else:
            c_true = float(s.sum())
            c_false = float(s.len() - s.sum())
            denom = float(s.len())

        p_true = (c_true / denom) if denom > 0 else 0.0
        p_false = (c_false / denom) if denom > 0 else 0.0
        f_true = Freq(level=True, count=c_true, prop=p_true)
        f_false = Freq(level=False, count=c_false, prop=p_false)

    return DescribeBoolean(
        name=col,
        mtype=MeasurementType.BOOLEAN,
        n=n_total,
        n_missing=n_missing,
        weighted=weighted,
        drop_nulls=drop_nulls,
        false=f_false,
        true=f_true,
    )


def _desc_datetime(
    df: pl.DataFrame,
    col: str,
    mtype: MeasurementType,
    *,
    weighted: bool,
    drop_nulls: bool,
) -> DescribeDatetime:
    s_base = df.get_column(col)
    n_total = df.height
    n_missing = int(s_base.null_count())
    s = s_base.drop_nulls() if drop_nulls else s_base

    try:
        vmin = _to_temporal(s.min())
        vmax = _to_temporal(s.max())
    except Exception:
        vmin = None
        vmax = None

    return DescribeDatetime(
        name=col,
        mtype=MeasurementType.DATETIME,
        n=n_total,
        n_missing=n_missing,
        weighted=weighted,
        drop_nulls=drop_nulls,
        min=vmin,
        max=vmax,
        tz=None,
    )


def _desc_string(
    df: pl.DataFrame,
    col: str,
    mtype: MeasurementType,
    *,
    weighted: bool,
    weight_col: str | None,
    drop_nulls: bool,
    top_k: int,
) -> DescribeString:
    s_base = df.get_column(col).cast(pl.Utf8)
    if drop_nulls:
        mask = ~s_base.is_null()
        s = s_base.filter(mask)
        w = _weight_series(df, weight_col)
        if w is not None:
            w = w.filter(mask)
    else:
        s = s_base
        w = _weight_series(df, weight_col)

    n_total = df.height
    n_missing = int(s_base.null_count())

    top = _build_freqs(s, weighted=weighted, w=w, top_k=top_k)
    try:
        n_unique = int(s.n_unique())
    except Exception:
        n_unique = None

    s_len_base = s.drop_nulls()
    if s_len_base.len():
        ls = s_len_base.str.len_chars()
        len_min = _to_int(ls.min())
        len_max = _to_int(ls.max())
        len_mean = _to_float(ls.mean())
        len_p50 = _to_float(ls.quantile(0.5))
    else:
        len_min = len_max = None
        len_mean = len_p50 = None

    return DescribeString(
        name=col,
        mtype=MeasurementType.STRING,
        n=n_total,
        n_missing=n_missing,
        weighted=weighted,
        drop_nulls=drop_nulls,
        n_unique=n_unique,
        top=top,
        truncated=False,
        len_min=len_min,
        len_p50=len_p50,
        len_mean=len_mean,
        len_max=len_max,
    )


def run_describe(
    *,
    df: pl.DataFrame,
    metadata: "MetadataStore | None" = None,
    columns: Sequence[str] | None = None,
    weighted: bool = False,
    weight_col: str | None = None,
    drop_nulls: bool = True,
    top_k: int = 10,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> DescribeResult:
    cols = _resolve_columns(df, columns)

    items: list[DescribeItem] = []
    for col in cols:
        mtype: MeasurementType
        if metadata is not None:
            meta = metadata.get(col)
            if meta is not None:
                mtype = meta.mtype
            else:
                mtype = _infer_mtype_from_polars(df.schema[col])
        else:
            mtype = _infer_mtype_from_polars(df.schema[col])

        if mtype is MeasurementType.CONTINUOUS:
            items.append(
                _desc_continuous(
                    df,
                    col,
                    mtype,
                    weighted=weighted,
                    weight_col=weight_col,
                    drop_nulls=drop_nulls,
                    percentiles=percentiles,
                )
            )
        elif mtype is MeasurementType.DISCRETE:
            items.append(
                _desc_discrete(
                    df,
                    col,
                    mtype,
                    weighted=weighted,
                    weight_col=weight_col,
                    drop_nulls=drop_nulls,
                    top_k=top_k,
                    percentiles=percentiles,
                )
            )
        elif mtype is MeasurementType.NOMINAL:
            items.append(
                _desc_nominal_or_ordinal(
                    df,
                    col,
                    mtype,
                    weighted=weighted,
                    weight_col=weight_col,
                    drop_nulls=drop_nulls,
                    top_k=top_k,
                    ordinal=False,
                )
            )
        elif mtype is MeasurementType.ORDINAL:
            items.append(
                _desc_nominal_or_ordinal(
                    df,
                    col,
                    mtype,
                    weighted=weighted,
                    weight_col=weight_col,
                    drop_nulls=drop_nulls,
                    top_k=top_k,
                    ordinal=True,
                )
            )
        elif mtype is MeasurementType.BOOLEAN:
            items.append(
                _desc_boolean(
                    df, col, mtype, weighted=weighted, weight_col=weight_col, drop_nulls=drop_nulls
                )
            )
        elif mtype is MeasurementType.DATETIME:
            items.append(_desc_datetime(df, col, mtype, weighted=weighted, drop_nulls=drop_nulls))
        else:
            items.append(
                _desc_string(
                    df,
                    col,
                    mtype,
                    weighted=weighted,
                    weight_col=weight_col,
                    drop_nulls=drop_nulls,
                    top_k=top_k,
                )
            )

    return DescribeResult(
        items=tuple(items),
        weighted=weighted,
        weight_col=weight_col,
        drop_nulls=drop_nulls,
        top_k=int(top_k),
        percentiles=tuple(float(p) for p in percentiles),
        generated_at=dt.datetime.now(dt.timezone.utc),
        notes=None,
    )
