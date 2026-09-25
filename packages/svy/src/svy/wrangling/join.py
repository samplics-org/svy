# src/svy/wrangling/join.py
"""
Bring variables from another sample or frame onto this sample's records.
"""

from __future__ import annotations

import re

from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import polars as pl

from svy.core import constants as K
from svy.errors import DimensionError, MethodError
from svy.wrangling._helpers import _eager_df, _guard_weight_writes, _resolve_target


if TYPE_CHECKING:
    from svy.core.sample import Sample


_WHERE = "wrangling.join"
_SHOWN = 10
# Names svy writes itself. Never a target; brought in only when asked for
# by name, under another name (a selection weight onto a respondent file).
_RESERVED = {
    v
    for k, v in vars(K).items()
    if k.startswith("SVY_") and isinstance(v, str) and v.startswith(K.SVY_PREFIX)
}


def _is_internal(name: str) -> bool:
    return (
        name in _RESERVED
        or name.startswith(K.SVY_PRIV_PREFIX)
        or K._INTERNAL_CONCAT_SUFFIX in name
    )


def _replicate_like(sample: "Sample", names: list[str]) -> list[str]:
    """Names this sample's replicate-weight pattern would read as replicates."""
    rep = getattr(sample._design, "rep_wgts", None)
    if rep is None:
        return []
    pattern = re.compile(rf"^{re.escape(rep.prefix)}\d+$", re.IGNORECASE)
    return [n for n in names if pattern.match(n)]


def _key_pairs(on: str | Sequence[str] | Mapping[str, str]) -> list[tuple[str, str]]:
    if isinstance(on, str):
        return [(on, on)]
    if isinstance(on, Mapping):
        return list(on.items())
    return [(k, k) for k in on]


def _chosen(
    cols: str | Sequence[str] | None, available: list[str], other_keys: set[str]
) -> list[str]:
    if cols is None:
        return [c for c in available if c not in other_keys and not _is_internal(c)]
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _string_like(dtype: pl.DataType) -> bool:
    return dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum))


def _not_whole(values: pl.Series) -> list:
    v = values.fill_nan(None).drop_nulls()
    return v.filter(v != v.floor()).unique(maintain_order=True).head(_SHOWN).to_list()


def _align_key(left: pl.Series, right: pl.Series) -> pl.DataType | None:
    """The type both sides of a key are joined as; None when they join as they are."""
    lt, rt = left.dtype, right.dtype
    if lt == rt:
        return None
    if lt.is_integer() and rt.is_integer():
        return None  # polars joins integer widths on their supertype
    if lt.is_numeric() and rt.is_numeric():
        if lt.is_float() and rt.is_float():
            return pl.Float64
        # An identifier stored as a decimal, as SPSS and Stata doubles are,
        # matches an integer one when every value is whole.
        fractional = _not_whole(left if lt.is_float() else right)
        if fractional:
            raise MethodError(
                title="Join key has values that are not whole numbers",
                detail=(
                    f"{lt} on this sample, {rt} on the other; "
                    f"values such as {fractional} cannot match an integer key."
                ),
                code="JOIN_KEY_TYPE_MISMATCH",
                where=_WHERE,
                param="on",
            )
        return pl.Int64
    if _string_like(lt) and _string_like(rt):
        return pl.String
    raise MethodError(
        title="Join key types do not match",
        detail=f"{lt} on this sample, {rt} on the other.",
        code="JOIN_KEY_TYPE_MISMATCH",
        where=_WHERE,
        param="on",
        hint="Cast one side first so both hold the same kind of value.",
    )


def _as(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    # NaN never matches, like a null key.
    if dtype == pl.Int64:
        return pl.when(expr.cast(pl.Float64).is_nan()).then(None).otherwise(expr).cast(pl.Int64)
    return expr.cast(dtype)


def _repeated(frame: pl.DataFrame, keys: list[str]) -> tuple[int, list[tuple]]:
    dup = frame.group_by(keys).len().filter(pl.col("len") > 1).sort(keys)
    return dup.height, [tuple(r) for r in dup.select(keys).head(_SHOWN).iter_rows()]


def join(
    sample: "Sample",
    other: "Sample | pl.DataFrame | pl.LazyFrame",
    on: str | Sequence[str] | Mapping[str, str],
    *,
    cols: str | Sequence[str] | None = None,
    into: str | Mapping[str, str] | None = None,
    suffix: str | None = None,
    validate: Literal["m:1", "1:1"] = "m:1",
    on_unmatched: Literal["ignore", "warn", "error"] = "warn",
    indicator: str | None = None,
    inplace: bool = False,
) -> "Sample":
    """Bring columns of ``other`` onto this sample's records, matched on ``on``.

    ``into`` maps a column of ``other`` to its name here; nothing on this
    sample is ever renamed or overwritten.
    """
    from svy.core.sample import Sample

    if validate not in ("m:1", "1:1"):
        raise MethodError.invalid_choice(
            where=_WHERE, param="validate", got=validate, allowed=["m:1", "1:1"]
        )
    if on_unmatched not in ("ignore", "warn", "error"):
        raise MethodError.invalid_choice(
            where=_WHERE,
            param="on_unmatched",
            got=on_unmatched,
            allowed=["ignore", "warn", "error"],
        )

    pairs = _key_pairs(on)
    if not pairs:
        raise MethodError.invalid_choice(
            where=_WHERE, param="on", got=on, allowed=["one or more key columns"]
        )
    left_keys = [a for a, _ in pairs]
    right_keys = [b for _, b in pairs]

    df = _eager_df(sample)
    if isinstance(other, Sample):
        other_meta = other._metadata
        other_df = _eager_df(other)
    elif isinstance(other, pl.LazyFrame):
        other_meta = None
        other_df = other
    elif isinstance(other, pl.DataFrame):
        other_meta = None
        other_df = other
    else:
        raise MethodError.invalid_choice(
            where=_WHERE,
            param="other",
            got=type(other).__name__,
            allowed=["Sample", "polars.DataFrame", "polars.LazyFrame"],
        )

    missing = [k for k in left_keys if k not in df.columns]
    if missing:
        raise DimensionError.missing_columns(
            where=_WHERE, param="on", missing=missing, available=df.columns
        )
    other_cols = (
        other_df.collect_schema().names()
        if isinstance(other_df, pl.LazyFrame)
        else other_df.columns
    )
    missing = [k for k in right_keys if k not in other_cols]
    if missing:
        raise DimensionError.missing_columns(
            where=_WHERE, param="on", missing=missing, available=other_cols
        )

    chosen = _chosen(cols, other_cols, set(right_keys))
    missing = [c for c in chosen if c not in other_cols]
    if missing:
        raise DimensionError.missing_columns(
            where=_WHERE, param="cols", missing=missing, available=other_cols
        )
    if not chosen:
        raise MethodError.invalid_choice(
            where=_WHERE, param="cols", got=cols, allowed=["at least one non-key column"]
        )
    as_key = [c for c in chosen if c in right_keys]
    if as_key:
        raise MethodError(
            title="A key cannot also be brought in",
            detail=", ".join(as_key),
            code="JOIN_KEY_IN_COLS",
            where=_WHERE,
            param="cols",
            hint="The key is already on this sample; leave it out of cols.",
        )

    if isinstance(into, str):
        if len(chosen) != 1:
            raise MethodError.invalid_choice(
                where=_WHERE,
                param="into",
                got=into,
                allowed=["a mapping {column of other: name here}, or a name with one column"],
            )
        into = {chosen[0]: into}
    named = dict(into or {})
    stray = [c for c in named if c not in chosen]
    if stray:
        raise MethodError(
            title="Naming a column that is not brought in",
            detail=", ".join(stray),
            code="JOIN_INTO_NOT_BROUGHT",
            where=_WHERE,
            param="into",
            hint="into names columns of the other side. Add them to cols, or leave cols out.",
        )
    brought = {c: named.get(c, c) for c in chosen}
    if suffix:
        # Only the clashes the caller did not name get the suffix; an
        # explicit name is taken as meant and checked like any other.
        brought = {
            c: f"{t}{suffix}" if c not in named and (t in df.columns or _is_internal(t)) else t
            for c, t in brought.items()
        }

    # Nothing on this sample is replaced, design columns included: a join
    # adds variables and leaves the design exactly as it was.
    targets = list(brought.values())
    if indicator is not None:
        targets.append(indicator)
    taken = sorted({t for t in targets if t in df.columns or _is_internal(t)})
    doubled = sorted({t for t in targets if targets.count(t) > 1})
    if taken or doubled:
        raise MethodError(
            title="Join would overwrite columns",
            detail=", ".join(taken + doubled),
            code="JOIN_COLUMN_EXISTS",
            where=_WHERE,
            param="into",
            hint='Name them on the way in, e.g. into={"region": "region_hh"}, or pass suffix="_hh".',
        )
    as_replicate = _replicate_like(sample, targets)
    if as_replicate:
        raise MethodError(
            title="Join would add columns that read as replicate weights",
            detail=", ".join(as_replicate),
            code="JOIN_COLUMN_EXISTS",
            where=_WHERE,
            param="into",
            hint="Name them so they do not match this sample's replicate-weight prefix.",
        )

    # Keys travel under private names so a brought column can never
    # collide with the other side's key.
    rkeys = [f"__svy_join_r{i}" for i in range(len(pairs))]
    lkeys = list(left_keys)
    left = df
    right = other_df.select(
        *[pl.col(b).alias(r) for (_, b), r in zip(pairs, rkeys)],
        *[pl.col(src).alias(dst) for src, dst in brought.items()],
    ).filter(pl.all_horizontal(pl.col(r).is_not_null() for r in rkeys))
    if isinstance(right, pl.LazyFrame):
        right = right.collect()
    for i, ((a, _), r) in enumerate(zip(pairs, rkeys)):
        cast_to = _align_key(df.get_column(a), right.get_column(r))
        if cast_to is not None:
            right = right.with_columns(_as(pl.col(r), cast_to).alias(r))
            lkeys[i] = f"__svy_join_l{i}"
            left = left.with_columns(_as(pl.col(a), cast_to).alias(lkeys[i]))

    n_rep, shown = _repeated(right, rkeys)
    if n_rep:
        raise MethodError(
            title="Join key repeats on the other side",
            detail=f"{n_rep} key value(s) match more than one record, e.g. {shown}.",
            code="JOIN_KEY_NOT_UNIQUE",
            where=_WHERE,
            param="on",
            hint="Each record here must match at most one record there. Deduplicate first.",
        )
    if validate == "1:1":
        present = df.filter(pl.all_horizontal(pl.col(k).is_not_null() for k in left_keys))
        n_rep, shown = _repeated(present, left_keys)
        if n_rep:
            raise MethodError(
                title="Join key repeats on this sample",
                detail=f"{n_rep} key value(s) appear on more than one record, e.g. {shown}.",
                code="JOIN_KEY_NOT_UNIQUE",
                where=_WHERE,
                param="on",
                hint='Use validate="m:1" when several records share one match.',
            )

    flag = "__svy_join_matched"
    joined = left.join(
        right.with_columns(pl.lit(True).alias(flag)),
        left_on=lkeys,
        right_on=rkeys,
        how="left",
        maintain_order="left",
    )
    matched = joined.get_column(flag).is_not_null()
    n_unmatched = int((~matched).sum())
    new_data = joined.drop([flag, *[k for k in lkeys if k not in left_keys]], strict=False)
    if indicator is not None:
        new_data = new_data.with_columns(matched.alias(indicator))

    if n_unmatched and on_unmatched == "error":
        raise MethodError(
            title="Records without a match",
            detail=f"{n_unmatched} of {df.height} record(s) found no match.",
            code="JOIN_UNMATCHED",
            where=_WHERE,
            param="on_unmatched",
            hint='Pass on_unmatched="warn" to keep them with nulls.',
        )

    _guard_weight_writes(sample, new_data, where="wrangling.join")
    target = _resolve_target(sample, new_data, inplace=inplace)

    if other_meta is not None:
        for src, dst in brought.items():
            meta = other_meta.get(src)
            if meta is not None:
                target._metadata.set(dst, meta.clone(name=dst))

    if n_unmatched and on_unmatched == "warn":
        target.warn(
            code="JOIN_UNMATCHED",
            title="Records without a match",
            detail=(
                f"{n_unmatched} of {df.height} record(s) found no match; "
                f"their {', '.join(brought.values())} are null."
            ),
            where=_WHERE,
            hint="Pass indicator= to mark matched records, and tell these nulls from blanks.",
            extra={"n_unmatched": n_unmatched, "n_records": df.height},
        )
    return target
