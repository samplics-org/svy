# python/svy_io/temporals.py
from __future__ import annotations

import datetime as _dt

import polars as pl


# Epochs
_SAS_EPOCH_DATE = _dt.date(1960, 1, 1)
_STATA_TD_EPOCH = _dt.date(1960, 1, 1)
_STATA_TC_EPOCH = _dt.datetime(1960, 1, 1)
_SPSS_EPOCH_DATE = _dt.date(1582, 10, 14)
_SPSS_EPOCH_DT = _dt.datetime(1582, 10, 14)

_NUMERIC_DTYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
)


def _is_numeric(dtype: pl.DataType) -> bool:
    """OPTIMIZED: frozenset for O(1) lookup"""
    return dtype in _NUMERIC_DTYPES


# ---------------- SAS ----------------


def coerce_sas_temporals(df: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """
    OPTIMIZED: Batch process all temporal conversions in single with_columns call.
    """
    conversions = []

    for v in meta.get("vars", []):
        name = v.get("name")
        if name not in df.columns:
            continue

        s = df[name]
        if not _is_numeric(s.dtype):
            continue

        fmt = (v.get("fmt") or v.get("format") or "").lower()
        col_ref = pl.col(name)

        if fmt.startswith(("date", "mmddyy", "ddmmyy", "yymmdd", "eur")):
            conversions.append(
                (pl.lit(_SAS_EPOCH_DATE) + pl.duration(days=col_ref.cast(pl.Int64))).alias(name)
            )
        elif fmt.startswith(("datetime", "e8601dt")):
            conversions.append(
                (pl.lit(_SAS_EPOCH_DATE) + pl.duration(seconds=col_ref.cast(pl.Int64)))
                .cast(pl.Datetime)
                .alias(name)
            )
        elif fmt.startswith("time"):
            conversions.append(pl.duration(seconds=col_ref.cast(pl.Int64)).alias(name))

    return df.with_columns(conversions) if conversions else df


# ---------------- Stata ----------------


def coerce_stata_temporals(df: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """
    OPTIMIZED: Batch process all temporal conversions.
    """
    conversions = []

    for v in meta.get("vars", []):
        name = v.get("name")
        if name not in df.columns:
            continue

        s = df[name]
        if not _is_numeric(s.dtype):
            continue

        fmt = (v.get("fmt") or v.get("format") or "").lower()
        col_ref = pl.col(name)

        # Daily date
        if fmt.startswith("%td") or fmt in ("%d", "d") or fmt.startswith(("%d", "d")):
            conversions.append(
                (pl.lit(_STATA_TD_EPOCH) + pl.duration(days=col_ref.cast(pl.Int64))).alias(name)
            )
        # Datetime (ms since 1960-01-01)
        elif fmt.startswith("%tc"):
            conversions.append(
                (pl.lit(_STATA_TC_EPOCH) + pl.duration(milliseconds=col_ref.cast(pl.Int64))).alias(
                    name
                )
            )

    return df.with_columns(conversions) if conversions else df


# ---------------- SPSS ----------------


def _infer_spss_fmt_from_name(name: str) -> str | None:
    """OPTIMIZED: Early exits, simplified checks"""
    n = (name or "").lower()

    # Check in priority order
    if "datetime" in n or "timestamp" in n or n.endswith("_dt") or "posix" in n:
        return "DATETIME"
    if n.endswith("_time") or n == "time":
        return "TIME"
    if "date" in n:  # Already excluded datetime/posix above
        return "DATE"
    return None


def _looks_like_spss_seconds(x: pl.Series) -> bool:
    """
    OPTIMIZED: Single try-except, simpler logic.
    SPSS datetime seconds range: ~5e8 to 5e10 for dates 1600-2300
    """
    try:
        s = x.drop_nulls().cast(pl.Float64)
        if s.len() == 0:
            return False

        m = s.quantile(0.5, interpolation="nearest")
        return 5e8 < m < 5e10
    except Exception:
        return False


def _looks_like_unix_milliseconds(x: pl.Series) -> bool:
    """Unix milliseconds: 1e11 to 1e13 for dates 1970-2100"""
    try:
        s = x.drop_nulls().cast(pl.Float64)
        if s.len() == 0:
            return False

        m = s.quantile(0.5, interpolation="nearest")
        return 1e11 < m < 1e13
    except Exception:
        return False


def _looks_like_unix_seconds(x: pl.Series) -> bool:
    """Unix seconds: 1e8 to 5e9 for dates 1970-2100"""
    try:
        s = x.drop_nulls().cast(pl.Float64)
        if s.len() == 0:
            return False

        m = s.quantile(0.5, interpolation="nearest")
        return 1e8 < m < 5e9
    except Exception:
        return False


def _coerce_string_iso_datetime(series: pl.Series) -> pl.Series | None:
    """Try to parse ISO datetime from string"""
    if series.dtype not in (pl.Utf8, pl.String):
        return None
    try:
        return series.str.strptime(pl.Datetime, strict=False, exact=False)
    except Exception:
        return None


def _coerce_string_iso_time(series: pl.Series) -> pl.Series | None:
    """Parse HH:MM[:SS] to Duration"""
    if series.dtype not in (pl.Utf8, pl.String):
        return None
    try:
        t = series.str.strptime(pl.Time, strict=False, exact=False)
        return (
            t.dt.hour().cast(pl.Int64) * 3600
            + t.dt.minute().cast(pl.Int64) * 60
            + t.dt.second().cast(pl.Int64)
            + (t.dt.nanosecond().cast(pl.Int64) // 1_000_000_000)
        ).cast(pl.Duration)
    except Exception:
        return None


def coerce_spss_temporals(df: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """
    OPTIMIZED: Batch process conversions, early exits, cache metadata lookup.

    Convert SPSS temporals:
    - DATE*/E8601DA  -> pl.Date
    - DATETIME*/E8601DT -> pl.Datetime (SPSS epoch)
    - POSIX datetime -> pl.Datetime (Unix epoch)
    - TIME*/E8601TM  -> pl.Duration
    """
    # Build metadata lookup once (O(1) access)
    var_meta = {v.get("name"): v for v in meta.get("vars", []) if v.get("name")}

    # Collect all conversions
    conversions = []

    for col_name in df.columns:
        s = df[col_name]
        v = var_meta.get(col_name)

        # Get format
        fmt = v.get("fmt") or v.get("format") if v else ""
        fmt_u = fmt.upper()

        # Infer from name if no format
        if not fmt_u:
            name_hint = _infer_spss_fmt_from_name(col_name)
            if name_hint:
                fmt_u = name_hint

        # Additional heuristics for numeric columns
        if not fmt_u and _is_numeric(s.dtype):
            if _looks_like_unix_milliseconds(s):
                fmt_u = "POSIX_MS"
            elif _looks_like_unix_seconds(s):
                fmt_u = "POSIX_SEC"
            elif _looks_like_spss_seconds(s):
                fmt_u = "DATETIME"

        # Skip if no format
        if not fmt_u:
            continue

        col_ref = pl.col(col_name)

        # ---- Numeric coercions ----
        if _is_numeric(s.dtype):
            # POSIX/Unix timestamps (check first)
            if fmt_u in ("POSIX", "POSIX_MS", "POSIX_SEC"):
                if _looks_like_unix_milliseconds(s):
                    conversions.append(pl.from_epoch(col_ref, time_unit="ms").alias(col_name))
                else:
                    conversions.append(pl.from_epoch(col_ref, time_unit="s").alias(col_name))
                continue

            # DATETIME (must check before DATE)
            if fmt_u.startswith(("DATETIME", "TIMESTAMP", "E8601DT")):
                conversions.append(
                    (pl.lit(_SPSS_EPOCH_DT) + pl.duration(seconds=col_ref.cast(pl.Int64)))
                    .cast(pl.Datetime)
                    .alias(col_name)
                )
                continue

            # DATE (after DATETIME check)
            if fmt_u.startswith(("DATE", "ADATE", "EDATE", "SDATE", "JDATE", "E8601DA")):
                conversions.append(
                    (
                        pl.lit(_SPSS_EPOCH_DATE)
                        + pl.duration(days=(col_ref.cast(pl.Int64) // 86_400))
                    ).alias(col_name)
                )
                continue

            # TIME
            if fmt_u.startswith(("TIME", "MTIME", "E8601TM")):
                conversions.append(pl.duration(seconds=col_ref.cast(pl.Int64)).alias(col_name))
                continue

        # ---- String fallbacks ----
        if s.dtype in (pl.Utf8, pl.String):
            parsed = None

            if fmt_u.startswith(("DATETIME", "TIMESTAMP", "E8601DT")) or fmt_u in (
                "POSIX",
                "POSIX_MS",
                "POSIX_SEC",
            ):
                parsed = _coerce_string_iso_datetime(s)
            elif fmt_u.startswith(("TIME", "MTIME", "E8601TM")):
                parsed = _coerce_string_iso_time(s)
            elif not fmt_u:
                # Generic fallback
                parsed = _coerce_string_iso_datetime(s)
                if parsed is None:
                    parsed = _coerce_string_iso_time(s)

            if parsed is not None:
                conversions.append(parsed.alias(col_name))

    # Single batch update
    return df.with_columns(conversions) if conversions else df
