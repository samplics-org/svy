# src/svy/engine/wrangling/cleaning.py
from __future__ import annotations

import re

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import polars as pl

from svy.core.enumerations import CaseStyle, LetterCase


if TYPE_CHECKING:
    from svy.metadata import MetadataStore


# --- Core Survey Design Helper (MUST be present in this file's namespace) ---


def _map_name_in_design(design_field: str | None, renames: dict[str, str]) -> str | None:
    """Maps a single string design field to a new name, or None if not found/string."""
    return (
        renames.get(design_field, design_field) if isinstance(design_field, str) else design_field
    )


def _map_tuple_in_design(
    design_field: str | tuple[str, ...] | None, renames: dict[str, str]
) -> str | tuple[str, ...] | None:
    """Maps a string or tuple design field to new name(s)."""
    if design_field is None or isinstance(design_field, str):
        return _map_name_in_design(design_field, renames)
    return tuple(renames.get(s, s) for s in design_field)


def _design_with_renamed_columns(design: Any, renames: dict[str, str]) -> Any:
    """
    Returns a new Design object with all column references updated by `renames`.
    (Implementation based on assumed Design.update signature.)
    """
    if not renames:
        return design

    d = design

    # Map simple fields
    new_wgt = _map_name_in_design(d.wgt, renames)
    new_prob = _map_name_in_design(d.prob, renames)
    new_hit = _map_name_in_design(d.hit, renames)
    new_mos = _map_name_in_design(d.mos, renames)
    new_pop_size = _map_name_in_design(d.pop_size, renames)
    new_row_index = _map_name_in_design(d.row_index, renames)

    # Map Tuple-or-str fields
    new_stratum = _map_tuple_in_design(d.stratum, renames)
    new_psu = _map_tuple_in_design(d.psu, renames)
    new_ssu = _map_tuple_in_design(d.ssu, renames)

    # Replicate weights
    new_rep = d.rep_wgts
    if d.rep_wgts is not None:
        mapped_wgts = tuple(renames.get(s, s) for s in d.rep_wgts.wgts)
        if mapped_wgts != d.rep_wgts.wgts:
            new_rep = d.rep_wgts.clone(wgts=mapped_wgts, n_reps=len(mapped_wgts))

    return d.update(
        row_index=new_row_index,
        stratum=new_stratum,
        wgt=new_wgt,
        prob=new_prob,
        hit=new_hit,
        mos=new_mos,
        psu=new_psu,
        ssu=new_ssu,
        pop_size=new_pop_size,
        rep_wgts=new_rep,
    )


# ----------------------------
# Small helpers for naming
# ----------------------------


def _normalize_into(
    into: str | Mapping[str, str] | None, cols: Sequence[str]
) -> Mapping[str, str] | None:
    """
    Normalize `into` to a mapping:
    """
    if into is None:
        return None
    if isinstance(into, str):
        if len(cols) != 1:
            raise ValueError(
                "into as a string is only valid when a single input column is provided."
            )
        return {cols[0]: into}
    if isinstance(into, Mapping):
        # keep only truthy targets
        return {k: v for k, v in into.items() if v}
    raise TypeError("into must be a string, a mapping, or None.")


def _target_name(
    *,
    data: pl.DataFrame,
    src_col: str,
    replace: bool,
    into_map: Mapping[str, str] | None,
    auto_suffix: str,
) -> str:
    """
    Decide the output column name for a transformation on `src_col`.
    """
    if replace:
        return src_col

    if into_map and src_col in into_map and into_map[src_col]:
        target = into_map[src_col]
    else:
        target = f"svy_{src_col}_{auto_suffix}"

    if target in data.columns:
        # prevent silent overwrite of an existing column
        # Relying on the outer function (e.g., Sample.wrangling) to handle naming if the
        # collision happens, but raise here if the user specified a colliding name via `into`.
        raise ValueError(
            f"Output column {target!r} already exists. Choose a different name via into=."
        )
    return target


# ----------------------------
# Numeric capping / coding (Restored original functional logic)
# ----------------------------


def _top_code(
    data: pl.DataFrame,
    top_codes: Mapping[str, float],
    *,
    replace: bool = False,
    into: str | Mapping[str, str] | None = None,
) -> pl.DataFrame:
    """Cap values >= bound at bound."""
    if not top_codes:
        return data
    into_map = _normalize_into(into, list(top_codes.keys()))
    exprs: list[pl.Expr] = []
    for col, bound in top_codes.items():
        if col not in data.columns:
            raise KeyError(f"Column not found for top-code: {col!r}")
        target = _target_name(
            data=data, src_col=col, replace=replace, into_map=into_map, auto_suffix="top_coded"
        )
        exprs.append(
            pl.when(pl.col(col) >= bound).then(bound).otherwise(pl.col(col)).alias(target)
        )
    return data.with_columns(exprs)


def _bottom_code(
    data: pl.DataFrame,
    bottom_codes: Mapping[str, float],
    *,
    replace: bool = False,
    into: str | Mapping[str, str] | None = None,
) -> pl.DataFrame:
    """Floor values <= bound at bound."""
    if not bottom_codes:
        return data
    into_map = _normalize_into(into, list(bottom_codes.keys()))
    exprs: list[pl.Expr] = []
    for col, bound in bottom_codes.items():
        if col not in data.columns:
            raise KeyError(f"Column not found for bottom-code: {col!r}")
        target = _target_name(
            data=data, src_col=col, replace=replace, into_map=into_map, auto_suffix="bottom_coded"
        )
        exprs.append(
            pl.when(pl.col(col) <= bound).then(bound).otherwise(pl.col(col)).alias(target)
        )
    return data.with_columns(exprs)


def _bottom_and_top_code(
    data: pl.DataFrame,
    bottom_and_top_codes: Mapping[str, tuple[float, float] | list[float]],
    *,
    replace: bool = False,
    into: str | Mapping[str, str] | None = None,
) -> pl.DataFrame:
    """Clamp each column to [bottom, top]."""
    if not bottom_and_top_codes:
        return data

    out = data
    into_map = _normalize_into(into, list(bottom_and_top_codes.keys()))
    exprs: list[pl.Expr] = []
    for col, bounds in bottom_and_top_codes.items():
        if col not in out.columns:
            raise KeyError(f"Column not found for clamp: {col!r}")
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(
                f"Bounds for {col!r} must be a tuple or list of length 2 (bottom, top)."
            )

        bottom, top = bounds
        if bottom is None or top is None:
            raise ValueError(f"Both bottom and top values must be provided for {col!r}.")
        if float(bottom) > float(top):
            raise ValueError("bottom must not be > top")

        target = _target_name(
            data=out,
            src_col=col,
            replace=replace,
            into_map=into_map,
            auto_suffix="bottom_and_top_coded",
        )
        c = pl.col(col)
        clamped = pl.when(c < bottom).then(bottom).otherwise(c)
        clamped = pl.when(clamped > top).then(top).otherwise(clamped)
        exprs.append(clamped.alias(target))

    return out.with_columns(exprs)


# ----------------------------
# Recoding / categorization
# ----------------------------


def _as_iterable(x: Any) -> Iterable[Any]:
    if isinstance(x, (str, bytes, bytearray)):
        return (x,)
    if isinstance(x, Iterable):
        return x
    return (x,)


def _recode(
    data: pl.DataFrame,
    varnames: str | list[str] | None = None,
    recodes: Mapping[Any, Any] = {},
    *,
    replace: bool = False,
    into: str | Mapping[str, str] | None = None,
    cols: str | list[str] | None = None,
) -> pl.DataFrame:
    """
    Recode values across one/many columns.

    FIXED: Restored original pl.when/then logic, which handles mixed types without needing
    pl.get_auto_type or Expr.map_dict (which don't exist).
    """
    # --- normalize the column list (accept alias `cols`) ---
    if cols is not None:
        var_list = [cols] if isinstance(cols, str) else list(cols)
    elif varnames is not None:
        var_list = [varnames] if isinstance(varnames, str) else list(varnames)
    else:
        raise ValueError("You must provide either `varnames` or `cols`.")

    # Normalize mapping values to sequences (accept scalars like "pear")
    normalized: dict[Any, tuple[Any, ...]] = {}
    for new_val, old_vals in recodes.items():
        if old_vals is None:
            normalized[new_val] = (None,)
        else:
            normalized[new_val] = tuple(_as_iterable(old_vals))

    out = data
    into_map = _normalize_into(into, var_list)
    exprs: list[pl.Expr] = []

    for col in var_list:
        if col not in out.columns:
            raise KeyError(f"Column not found for recode: {col!r}")

        target = _target_name(
            data=out, src_col=col, replace=replace, into_map=into_map, auto_suffix="recoded"
        )

        # Build a single chained when/then expression
        expr = pl.col(col)
        builder = None

        for new_val, old_vals in normalized.items():
            # Build condition: col is in old_vals
            cond = pl.col(col).is_in(list(old_vals))

            # Handle None separately
            if None in old_vals:
                cond = cond | pl.col(col).is_null()

            if builder is None:
                builder = pl.when(cond).then(pl.lit(new_val))
            else:
                builder = builder.when(cond).then(pl.lit(new_val))

        # Default: keep original value
        final_expr = builder.otherwise(expr) if builder is not None else expr
        exprs.append(final_expr.alias(target))

    return out.with_columns(exprs)


def _categorize(
    data: pl.DataFrame,
    varname: str,
    bins: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    right: bool = True,
    replace: bool = False,
    into: str | Mapping[str, str] | None = None,
) -> pl.DataFrame:
    """
    Bin a numeric column into categories.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    varname : str
        Source column name.
    bins : Sequence[float]
        Bin edges (n+1 edges for n bins).
    labels : Sequence[str] | None
        Labels for bins (n labels for n bins). If None, auto-generate.
    right : bool
        If True, intervals are (left, right]. If False, [left, right).
    replace : bool
        If True, replace the source column.
    into : str | Mapping[str, str] | None
        Output column name(s).

    Returns
    -------
    pl.DataFrame
        DataFrame with new categorized column.
    """
    if varname not in data.columns:
        raise KeyError(f"Column not found: {varname!r}")

    if len(bins) < 2:
        raise ValueError("bins must have at least 2 edges")

    n_bins = len(bins) - 1
    if labels is not None and len(labels) != n_bins:
        raise ValueError(f"labels length ({len(labels)}) must equal number of bins ({n_bins})")

    # Determine destination column name
    into_map = _normalize_into(into, [varname])
    target = _target_name(
        data=data, src_col=varname, replace=replace, into_map=into_map, auto_suffix="categorized"
    )

    col_expr = pl.col(varname)

    # First bin
    lower, upper = bins[0], bins[1]
    first_label = (
        labels[0] if labels else (f"({lower}, {upper}]" if right else f"[{lower}, {upper})")
    )
    if right:
        builder = pl.when((col_expr > lower) & (col_expr <= upper)).then(pl.lit(first_label))
    else:
        builder = pl.when((col_expr >= lower) & (col_expr < upper)).then(pl.lit(first_label))

    # Remaining bins
    for i in range(2, len(bins)):
        lower, upper = bins[i - 1], bins[i]
        lab = (
            labels[i - 1]
            if labels
            else (f"({lower}, {upper}]" if right else f"[{lower}, {upper})")
        )
        if right:
            builder = builder.when((col_expr > lower) & (col_expr <= upper)).then(pl.lit(lab))
        else:
            builder = builder.when((col_expr >= lower) & (col_expr < upper)).then(pl.lit(lab))

    # Everything else -> None
    return data.with_columns(builder.otherwise(pl.lit(None)).alias(target))


# ----------------------------
# Renaming / labels
# ----------------------------


def _rename(data: pl.DataFrame, renames: Mapping[str, str]) -> pl.DataFrame:
    if not renames:
        return data

    if len(renames) != len(set(renames)):
        raise ValueError("Duplicate source column in renames mapping")

    dests = list(renames.values())
    if len(dests) != len(set(dests)):
        raise ValueError(f"Duplicate destination names in renames: {dests}")

    missing = [c for c in renames if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    return data.rename(renames)


def _update_metadata_keys(
    store: "MetadataStore | None",
    renames: Mapping[str, str],
) -> None:
    """
    Update variable names in MetadataStore when columns are renamed.

    Modifies the store in-place by renaming variable keys.

    Parameters
    ----------
    store : MetadataStore | None
        The metadata store to update.
    renames : Mapping[str, str]
        Mapping from old column names to new column names.
    """
    if store is None or not renames:
        return

    store.rename_variables(renames)


# ----------------------------
# Name cleaning helpers
# ----------------------------


def _cap_first_alpha(s: str) -> str:
    """Uppercase first alphabetic char; leave the rest unchanged."""
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1 :]
    return s


def _lower_rest_after_first_alpha(s: str) -> str:
    """Lowercase alphabetic chars after the first alphabetic char."""
    out = []
    seen_alpha = False
    for ch in s:
        if ch.isalpha():
            if not seen_alpha:
                out.append(ch)
                seen_alpha = True
            else:
                out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _to_case(words: list[str], style: CaseStyle) -> str:
    """
    Structure the words; do NOT apply overall letter casing here
    (that is the job of apply_letter_case).
    """
    words = [w for w in words if w]
    if not words:
        return ""

    if style == CaseStyle.SNAKE:
        return "_".join(words)

    if style == CaseStyle.KEBAB:
        return "-".join(words)

    if style == CaseStyle.CAMEL:
        head = words[0].lower()
        tail: list[str] = []
        for w in words[1:]:
            w = _lower_rest_after_first_alpha(w.lower())
            w = _cap_first_alpha(w)
            tail.append(w)
        return head + "".join(tail)

    if style == CaseStyle.PASCAL:
        parts = []
        for w in words:
            w = _lower_rest_after_first_alpha(w.lower())
            w = _cap_first_alpha(w)
            parts.append(w)
        return "".join(parts)

    raise ValueError(f"Unsupported case style: {style}")


def _apply_letter_case(name: str, casing: LetterCase) -> str:
    """
    Apply overall letter casing to the (already structured) name.
    Preserves '_' and '-' boundaries for TITLE.
    """
    if casing == LetterCase.LOWER:
        return name.lower()

    if casing == LetterCase.UPPER:
        return name.upper()

    if casing == LetterCase.TITLE:
        out: list[str] = []
        buf: list[str] = []

        def flush():
            if not buf:
                return
            seg = "".join(buf)
            seg = _lower_rest_after_first_alpha(seg.lower())
            seg = _cap_first_alpha(seg)
            out.append(seg)
            buf.clear()

        for ch in name:
            if ch in "_-":
                flush()
                out.append(ch)
            else:
                buf.append(ch)
        flush()
        return "".join(out)

    if casing == LetterCase.ORIGINAL:
        return name

    raise ValueError(f"Unsupported letter case: {casing}")


def _clean_names(
    data: pl.DataFrame,
    minimal: bool = False,
    remove: str | None = None,
    case_style: CaseStyle = CaseStyle.SNAKE,
    letter_case: LetterCase = LetterCase.LOWER,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """
    Clean column names and return (new_df, renames).
    """
    old_cols = list(data.columns)
    RESERVED = {"svy_row_index", "svy_weight", "svy_prob", "svy_hit"}

    def _remove_non_alnum(name: str, pat: str) -> str:
        # remove per-character only if it matches `pat` AND is not alphanumeric/underscore
        return "".join(
            ch for ch in name if not (re.fullmatch(pat, ch) and not (ch.isalnum() or ch == "_"))
        )

    def normalize(name: str) -> tuple[str, bool]:
        # preserve reserved/internal columns exactly
        if name in RESERVED:
            return name, False

        orig = name
        if remove:
            name = _remove_non_alnum(name, remove)
            # If nothing changed and it's already safe, keep verbatim
            if name == orig and re.fullmatch(r"[A-Za-z0-9_]+", name or ""):
                return name, False
        elif not minimal:
            # default strip punctuation (keep underscore)
            name = re.sub(r"[^\w\s]", "", name)

        # standardize delimiters
        name = re.sub(r"[\s\-]+", "_", name)

        # tokens -> case style
        words = re.findall(r"[A-Za-z0-9]+", name)
        new_name = _to_case(words, case_style)

        # apply casing only for snake/kebab
        if case_style in {CaseStyle.SNAKE, CaseStyle.KEBAB}:
            new_name = _apply_letter_case(new_name, letter_case)

        return new_name, True

    # first pass
    normalized: list[str] = []
    changed_flags: list[bool] = []
    for c in old_cols:
        new, changed = normalize(c)
        normalized.append(new)
        changed_flags.append(changed)

    # second pass: fill empties + ensure uniqueness
    seen: set[str] = set()
    final: list[str] = []
    blank_counter = 0

    for base in normalized:
        candidate = base
        if candidate == "":
            candidate = f"col_{blank_counter}"
            blank_counter += 1

        if candidate in seen:
            k = 1
            while True:
                c2 = f"{candidate}_{k}"
                if c2 not in seen:
                    candidate = c2
                    break
                k += 1

        seen.add(candidate)
        final.append(candidate)

    # build renames map
    renames: dict[str, str] = {old: new for old, new in zip(old_cols, final) if old != new}
    new_df = data.rename(renames) if renames else data
    return new_df, renames
