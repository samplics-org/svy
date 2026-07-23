# src/svy/core/data_prep.py
"""
Unified data preparation for all svy Rust backend calls.

Centralizes the common operations that every public API (estimation and
categorical) needs before calling Rust:

  1. Materialize LazyFrame
  2. Column selection (only needed columns for efficiency)
  3. Missing value handling (drop or assert)
  4. Concatenated design columns (stratum, psu, ssu, by)
  5. Paired difference (y - y_pair)
  6. Weight column creation (if no design weight)
  7. Type casting (y→Float64, group/strata/psu→String)
  8. Singleton filtering
  9. FPC column computation
  10. Where clause → domain column + zero weights (main AND replicate)
      for non-domain observations

Both ``estimation.base.Estimation`` and ``categorical.base.Categorical``
call :func:`prepare_data` rather than maintaining their own prep logic.
"""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import polars as pl

from svy.core.constants import _BY_SEP, _INTERNAL_CONCAT_SUFFIX
from svy.core.types import WhereArg
from svy.errors import DimensionError
from svy.utils.checks import assert_no_missing, drop_missing
from svy.utils.helpers import _colspec_to_list


if TYPE_CHECKING:
    from svy.core.sample import Sample

log = logging.getLogger(__name__)

# ── Phase C: cached integer design-code columns ─────────────────────────────
# Hidden UInt32 columns that are bijective with the string design labels. The
# Rust variance/df kernels re-densify them (first-appearance order) to exactly
# the partition the string labels would produce, so results are bit-identical
# but the per-call string hashing is replaced by cheap integer densification.
# Built once per Sample data version and reused across estimation calls.
_STRATUM_CODE = "__svy_stratum_code__"
_PSU_CODE = "__svy_psu_code__"
_SSU_CODE = "__svy_ssu_code__"


def _code_expr(cols: list[str], alias: str) -> pl.Expr:
    """Factorize one design group to UInt32 codes. Mirrors the string-concat
    null handling (null → "__Null__" token, so nulls form a group) so the code
    partition matches the string path exactly."""
    parts = [pl.col(c).cast(pl.Utf8).fill_null("__Null__") for c in cols]
    concat = pl.concat_str(parts, separator=_BY_SEP) if len(parts) > 1 else parts[0]
    return concat.cast(pl.Categorical).to_physical().cast(pl.UInt32).alias(alias)


def _get_design_codes(sample: Sample, design) -> dict[str, pl.Series] | None:
    """Full-data UInt32 design codes, cached on the Sample per data version.

    - stratum code: factorizes the stratum column(s).
    - psu code: factorizes the (stratum + psu) columns, so PSU labels reused
      across strata become distinct PSUs (matches the Rust pair-nesting).
    - ssu code: factorizes the ssu column(s).

    Returns ``None`` when the design has neither stratum nor psu (nothing to
    factorize). The codes are full-length (unfiltered) so callers can attach
    them before row filtering and let them filter along.
    """
    strat_cols = _colspec_to_list(design.stratum) if design.stratum else []
    psu_cols = _colspec_to_list(design.psu) if design.psu else []
    ssu_cols = _colspec_to_list(design.ssu) if design.ssu else []
    if not strat_cols and not psu_cols:
        return None

    version = getattr(sample, "_data_version", None)
    cached = getattr(sample, "_design_codes_cache", None)
    if cached is not None and cached[0] == version:
        return cached[1]

    data = sample._data
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    exprs: list[pl.Expr] = []
    key_to_name: dict[str, str] = {}
    if strat_cols:
        exprs.append(_code_expr(strat_cols, _STRATUM_CODE))
        key_to_name["stratum"] = _STRATUM_CODE
    if psu_cols:
        exprs.append(_code_expr(strat_cols + psu_cols, _PSU_CODE))
        key_to_name["psu"] = _PSU_CODE
    if ssu_cols:
        exprs.append(_code_expr(ssu_cols, _SSU_CODE))
        key_to_name["ssu"] = _SSU_CODE

    code_df = cast(pl.DataFrame, data).select(exprs)
    codes = {key: code_df[name] for key, name in key_to_name.items()}
    sample._design_codes_cache = (version, codes)
    return codes


# Cache for design.specified_fields() keyed on id(design).
# Stores (design_obj, data_version, fields). The held design_obj reference
# catches CPython id-reuse (a new design reusing a freed id), and the
# data_version — globally unique per Sample mutation — catches the case where
# the same design object persists while the underlying data columns change.
# Bounded: cleared when it exceeds _DESIGN_FIELDS_CACHE_MAX so long-lived
# processes creating many Samples don't pin Design objects forever.
_design_fields_cache: dict[int, tuple] = {}
_DESIGN_FIELDS_CACHE_MAX = 512

# Internal column name for the materialized where-clause boolean mask.
# Created and dropped within prepare_data; never exposed to the caller.
_MASK_COL = "__svy_where_mask__"

# Internal Float64 domain-mask column (1.0 inside the `where=` domain, 0.0
# outside). Emitted instead of zeroing R replicate columns when
# ``domain_mask_for_replication=True``; the Rust replicate kernels apply it on
# the fly. Carried through to Rust (not dropped), so it stays namespaced.
_DOMAIN_MASK_COL = "__svy_domain_mask__"


# ═══════════════════════════════════════════════════════════════════════
# Return type
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PreparedData:
    """Result of :func:`prepare_data`."""

    df: pl.DataFrame
    y_col: str
    weight_col: str
    strata_col: str | None = None
    psu_col: str | None = None
    ssu_col: str | None = None
    fpc_col: str | None = None
    fpc_ssu_col: str | None = None
    domain_col: str | None = None
    domain_val: str | None = None
    domain_mask_col: str | None = None
    by_col: str | None = None
    by_cols: list[str] = field(default_factory=list)
    singleton_method: str | None = None


# ═══════════════════════════════════════════════════════════════════════
# Where-clause column extraction
# ═══════════════════════════════════════════════════════════════════════


def extract_where_cols(where: WhereArg) -> list[str]:
    """Extract column names referenced by a where clause.

    Handles all WhereArg forms:
    - None → []
    - Mapping[str, Any] (dict) → list of keys
    - pl.Expr → expr.meta.root_names()
    - HasExpr (svy.col wrapper) → unwrap ._e then root_names()
    - Sequence[ExprLike] → union of all root_names()
    """
    if where is None:
        return []

    # Dict form: {"sex": 2, "region": "North"}
    if isinstance(where, Mapping):
        return list(where.keys())

    # svy Expr wrapper (HasExpr protocol): unwrap to pl.Expr
    if hasattr(where, "_e"):
        where = where._e

    # Single Polars expression
    if isinstance(where, pl.Expr):
        try:
            return where.meta.root_names()
        except Exception:
            return []

    # Sequence of expressions (combined by AND)
    if isinstance(where, Sequence) and not isinstance(where, (str, bytes)):
        cols: list[str] = []
        for w in where:
            if hasattr(w, "_e"):
                w = w._e
            if isinstance(w, pl.Expr):
                try:
                    cols.extend(w.meta.root_names())
                except Exception:
                    pass
            elif isinstance(w, Mapping):
                cols.extend(w.keys())
        return cols

    return []


# ═══════════════════════════════════════════════════════════════════════
# Replicate-weight column resolution
# ═══════════════════════════════════════════════════════════════════════


def _natural_keys(text: str) -> list:
    """Natural sort key for replicate weight column names like 'repwtp1', 'repwtp2', ..."""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


def _resolve_rep_weight_cols(data: pl.DataFrame, design) -> list[str]:
    """Resolve replicate weight column names from the design.

    Operates on raw data + design with no dependency on an Estimation
    instance. Caches the result on ``rep_wgts._cached_cols`` so repeated
    calls within the same Sample lifetime are O(1).

    Mirrors the logic in ``svy.estimation.replication.get_rep_weight_cols``;
    that function may delegate to this helper to avoid duplication.
    """
    rw = getattr(design, "rep_wgts", None)
    if rw is None:
        return []

    cached = getattr(rw, "_cached_cols", None)
    if cached is not None:
        return cached

    data_cols = data.columns

    if rw.prefix:
        # Strict `^prefix\d+$` matching (the same rule as
        # RepWeights.columns_from_data). A loose startswith() absorbed any
        # column sharing the prefix (e.g. 'repwt_flag' for prefix 'repwt')
        # into the replicate set, corrupting the variance.
        pattern = re.compile(rf"^{re.escape(rw.prefix)}\d+$", re.IGNORECASE)
        cols = sorted(
            [c for c in data_cols if pattern.match(c)],
            key=lambda c: _natural_keys(c.lower()),
        )
        if cols and len(cols) != rw.n_reps:
            raise DimensionError(
                title="Replicate weight column count mismatch",
                detail=f"Found {len(cols)} columns matching '{rw.prefix}<number>' "
                f"but RepWeights declares n_reps={rw.n_reps}.",
                code="REP_WEIGHT_COUNT_MISMATCH",
                where="core.data_prep",
                param="rep_wgts",
                expected=rw.n_reps,
                got=len(cols),
                hint="Check the replicate weight prefix and n_reps, or drop "
                "stray columns that match the prefix-number pattern.",
            )
    elif hasattr(rw, "wgts") and rw.wgts:
        # First-occurrence wins on case-insensitive collision.
        lower_index: dict[str, str] = {}
        for c in data_cols:
            lower_index.setdefault(c.lower(), c)

        cols = []
        missing = []
        for name in list(rw.wgts):
            actual = lower_index.get(name.lower())
            if actual is None:
                missing.append(name)
            else:
                cols.append(actual)
        if missing:
            raise ValueError(
                f"Replicate weight columns not found (case-insensitive match): "
                f"{missing}. Available columns: {data_cols}"
            )
    else:
        cols = []

    try:
        rw._cached_cols = cols
    except Exception:
        # rep_wgts may be frozen or otherwise non-mutable; cache miss is fine.
        pass
    return cols


# ═══════════════════════════════════════════════════════════════════════
# Main data preparation function
# ═══════════════════════════════════════════════════════════════════════


def prepare_data(
    sample: Sample,
    *,
    y: str,
    x: str | None = None,
    group: str | None = None,
    y_pair: str | None = None,
    by: str | Sequence[str] | None = None,
    where: WhereArg = None,
    extra_cols: list[str] | None = None,
    drop_nulls: bool,
    cast_y_float: bool,
    select_columns: bool,
    apply_singleton_filter: bool,
    domain_mask_for_replication: bool = False,
) -> PreparedData:
    """
    Unified data preparation for all Rust backend calls.

    Parameters
    ----------
    sample : Sample
        The survey sample object.
    y : str
        Primary variable name.
    x : str | None
        Secondary variable (e.g. denominator for ratio estimation).
    group : str | None
        Grouping variable (two-sample tests).
    y_pair : str | None
        Second variable for paired tests.
    by : str | Sequence[str] | None
        Domain / stratification variable(s).
    where : WhereArg
        Subpopulation filter.
    extra_cols : list[str] | None
        Additional columns to include (e.g. colvar for tabulate).
    drop_nulls : bool
        If True, drop rows with missing values. Required — must be explicit.
    cast_y_float : bool
        If True, cast y to Float64. Set False for categorical y (tabulate). Required — must be explicit.
    select_columns : bool
        If True, select only needed columns for efficiency. Required — must be explicit.
    apply_singleton_filter : bool
        If True, apply singleton stratum exclusion filter. Required — must be explicit.

    Returns
    -------
    PreparedData
        Dataclass with all prepared data and column names.
    """
    # ── Materialize ──────────────────────────────────────────────────────
    _raw = sample._data
    local_data: pl.DataFrame = (
        cast(pl.DataFrame, _raw)
        if not isinstance(_raw, pl.LazyFrame)
        else cast(pl.DataFrame, _raw.collect())
    )
    design = sample._design

    # ── Phase C: attach cached integer design codes ─────────────────────
    # Attach the (cached, full-data) UInt32 stratum/psu/ssu codes BEFORE any
    # row filtering so they filter along with the data; the Rust kernel then
    # densifies the filtered subset, matching the string path bit-for-bit.
    # Skipped when singleton variance columns take over the design (that path
    # keeps its own string columns).
    _sr_early = getattr(sample, "_singleton_result", None)
    _use_codes = not (_sr_early and _sr_early.config and _sr_early.config.var_stratum_col)
    _design_codes = _get_design_codes(sample, design) if _use_codes else None
    if _design_codes:
        local_data = local_data.with_columns(list(_design_codes.values()))

    # ── Resolve replicate weight columns (cached on design.rep_wgts) ─────
    # These must be carried through `needed` so they survive select_columns,
    # and they must be zeroed alongside the main weight when `where` is set.
    # Empty list for Taylor-linearization designs.
    rep_weight_cols: list[str] = _resolve_rep_weight_cols(local_data, design)

    # ── Build analysis-relevant column list ─────────────────────────────
    # This list is ALWAYS built, regardless of select_columns, because:
    #   1. drop_nulls/assert_no_missing should only check columns involved
    #      in the analysis, not every column in the dataset.
    #   2. select_columns controls whether the dataframe is subsetted for
    #      efficiency — it should never change analytical results.
    needed = [y]
    if x:
        needed.append(x)
    if y_pair:
        needed.append(y_pair)
    if group:
        needed.append(group)
    if isinstance(by, str):
        needed.append(by)
    elif isinstance(by, (list, tuple)):
        needed.extend(by)
    if extra_cols:
        needed.extend(extra_cols)
    where_cols = extract_where_cols(where)
    needed.extend(where_cols)
    key = id(design)
    data_version = getattr(sample, "_data_version", None)
    cached = _design_fields_cache.get(key)
    if cached is None or cached[0] is not design or cached[1] != data_version:
        fields = design.specified_fields(data_columns=local_data.columns)
        if len(_design_fields_cache) >= _DESIGN_FIELDS_CACHE_MAX:
            _design_fields_cache.clear()
        _design_fields_cache[key] = (design, data_version, fields)
    else:
        fields = cached[2]
    needed.extend(fields)
    # Include singleton variance/exclude columns if present in data
    _sr_pre = getattr(sample, "_singleton_result", None)
    _sc_pre = _sr_pre.config if _sr_pre else None
    _singleton_check_cols: list[str] = []
    if _sc_pre:
        for _col in [_sc_pre.var_stratum_col, _sc_pre.var_psu_col, _sc_pre.var_exclude_col]:
            if _col and _col in local_data.columns:
                needed.append(_col)
                _singleton_check_cols.append(_col)
    # Carry replicate weights through column selection.
    # Without this, ``select_columns=True`` would drop them before the where
    # block can zero them out, leaving downstream Rust calls with full-sample
    # replicate weights and the wrong variance.
    needed.extend(rep_weight_cols)
    # Carry the Phase C design-code columns through column selection.
    if _design_codes:
        needed.extend(s.name for s in _design_codes.values())
    needed = sample._dedup_preserve_order(needed)

    # ── Missing handling ─────────────────────────────────────────────────
    # Applied BEFORE column selection so that the same rows are affected
    # regardless of select_columns. Only analysis-relevant columns are
    # checked — replicate weights are excluded from the null check because
    # a rare NA in one of N replicate columns should not affect rows for
    # the entire analysis.
    #
    # With drop_nulls=True, missing values (null/NaN/±inf) fall in two
    # classes:
    #   * ANALYSIS values — numeric y/x/y_pair (cast_y_float paths), group,
    #     and by labels: the row is KEPT and treated like an out-of-domain
    #     observation — its weights (main + replicate) are zeroed in the
    #     domain block below. Physically dropping such rows deleted whole
    #     PSUs/strata under standard skip patterns (y null outside the
    #     domain), silently understating domain SEs and df; keeping them
    #     preserves the design structure, matching R's na.rm=TRUE /
    #     subset() domain semantics.
    #   * Everything else (design fields, singleton helpers, extra_cols —
    #     regression covariates and multi-y batches — and categorical y
    #     when cast_y_float=False): rows are physically dropped, as before.
    # Columns referenced only by `where` are excluded from the drop: a null
    # predicate value makes the row out-of-domain via Kleene logic in the
    # domain block.
    if rep_weight_cols:
        rep_set = set(rep_weight_cols)
        null_check_cols = [c for c in needed if c not in rep_set]
    else:
        null_check_cols = needed

    _domain_missing_cols: list[str] = []
    if drop_nulls:
        _structural = set(fields)
        _structural.update(_singleton_check_cols)
        if _design_codes:
            _structural.update(s.name for s in _design_codes.values())

        _domain_roles: list[str] = []
        if cast_y_float:
            _domain_roles.append(y)
            if x:
                _domain_roles.append(x)
            if y_pair:
                _domain_roles.append(y_pair)
        if group:
            _domain_roles.append(group)
        if isinstance(by, str):
            _domain_roles.append(by)
        elif isinstance(by, (list, tuple)):
            _domain_roles.extend(by)

        _domain_missing_cols = [
            c
            for c in sample._dedup_preserve_order(_domain_roles)
            if c in local_data.columns and c not in _structural
        ]
        _dm_set = set(_domain_missing_cols)
        _where_only = (set(where_cols) - _structural) - _dm_set
        drop_cols = [c for c in null_check_cols if c not in _dm_set and c not in _where_only]
        if drop_cols:
            local_data = drop_missing(
                df=local_data,
                cols=drop_cols,
                treat_infinite_as_missing=True,
            )
    else:
        assert_no_missing(df=local_data, subset=null_check_cols)

    # ── Column selection (optional optimization) ─────────────────────────
    if select_columns:
        local_data = local_data.select(needed)

    # ── Singleton filter ─────────────────────────────────────────────────
    if apply_singleton_filter:
        from svy.core.singleton import _VAR_EXCLUDE_COL

        if _VAR_EXCLUDE_COL in local_data.columns:
            local_data = local_data.filter(~pl.col(_VAR_EXCLUDE_COL))

    # ── Concatenated design columns ──────────────────────────────────────
    # When Phase C codes are active, skip the stratum/psu/ssu string concats
    # (the kernel uses the integer codes instead) and build only the ``by``
    # concat. This removes the per-call concat_str work from the hot path.
    _cc, _ = sample._create_concatenated_cols_from_lists(
        data=local_data,
        design=design,
        by=by,
        null_token="__Null__",
        suffix=_INTERNAL_CONCAT_SUFFIX,
        categorical=True,
        drop_original=False,
        include_design=_design_codes is None,
    )
    df: pl.DataFrame = cast(pl.DataFrame, _cc)

    # ── Design column names ──────────────────────────────────────────────
    # Use singleton variance columns if present (scale/center/collapse methods
    # create __svy_var_stratum__ / __svy_var_psu__ as the correct variance cols).
    # Otherwise use the freshly created concatenated columns from the current
    # design — NOT the stale _internal_design from Sample.__init__, which may
    # reference pre-rename column names.
    _singleton_result = getattr(sample, "_singleton_result", None)
    _singleton_config = _singleton_result.config if _singleton_result else None

    suffix = _INTERNAL_CONCAT_SUFFIX

    if _singleton_config and _singleton_config.var_stratum_col:
        strata_col = _singleton_config.var_stratum_col
        psu_col = _singleton_config.var_psu_col
        ssu_col = f"ssu{suffix}" if design.ssu else None
    elif _design_codes:
        # Phase C: point the kernel at the integer code columns (dtype-dispatch
        # picks the fast path). PSU codes are already stratum-nested.
        strata_col = _STRATUM_CODE if "stratum" in _design_codes else None
        psu_col = _PSU_CODE if "psu" in _design_codes else None
        ssu_col = _SSU_CODE if "ssu" in _design_codes else None
    else:
        strata_col = f"stratum{suffix}" if design.stratum else None
        psu_col = f"psu{suffix}" if design.psu else None
        ssu_col = f"ssu{suffix}" if design.ssu else None

    # ── By column resolution ─────────────────────────────────────────────
    if isinstance(by, str):
        by_cols_list = [by]
        by_col = f"by{_INTERNAL_CONCAT_SUFFIX}"
    elif isinstance(by, (list, tuple)) and len(by) > 0:
        by_cols_list = list(by)
        by_col = f"by{_INTERNAL_CONCAT_SUFFIX}"
    else:
        by_cols_list = []
        by_col = None

    # A variable may appear in BOTH `by` and `where` (issue #9). The two are
    # orthogonal: `where` is domain estimation (out-of-domain rows keep their
    # place but have their weight zeroed and are flagged via `__svy_domain__`),
    # while `by` groups on the original values. When a `where` predicate
    # excludes an entire `by` level, that level is dropped from the results by
    # blanking its grouping label in the `where` block below, matching R's
    # `design %>% filter(...) %>% group_by(...)`.

    # ── Paired difference ────────────────────────────────────────────────
    y_col = y
    if y_pair:
        diff_name = f"__svy_{y}_minus_{y_pair}__"
        df = df.with_columns(
            (pl.col(y).cast(pl.Float64) - pl.col(y_pair).cast(pl.Float64)).alias(diff_name)
        )
        y_col = diff_name

    # ── Weight column ────────────────────────────────────────────────────
    weight_col = design.wgt if design.wgt else "__svy_ones__"

    # Rep weight columns that actually exist in df after select/singleton.
    rep_weight_cols_in_df = [c for c in rep_weight_cols if c in df.columns]

    # ── Where clause → domain column + zero weights (main + replicate) ───
    # Must happen BEFORE the categorical/by/strata type casting below so
    # where expressions reference original column types (e.g. pl.col("sex")
    # == 2 where sex is still Int32).
    #
    # For replication designs: zero the main weight AND every replicate
    # weight, so that downstream Rust variance computation sees only
    # domain-active observations across all replicates. Without zeroing
    # the replicate weights, the point estimate (which uses the main weight)
    # correctly reflects the domain but the replicate-variance computation
    # sees the full sample, producing SEs identical to the unfiltered
    # estimate.
    #
    # Performance: the where_expr is materialized once as a Boolean column
    # (__svy_where_mask__) and reused across all downstream zeroing
    # operations. This guarantees a single evaluation of the predicate
    # regardless of Polars' CSE behavior, then folds the (cast + zero)
    # for the main weight and every replicate weight into one parallelized
    # ``with_columns`` call. The mask column is dropped before returning.
    domain_col = None
    domain_val = None
    domain_mask_col = None

    # Missing analysis values → out-of-domain (see the missing-handling
    # block above). Only engaged when something is actually missing, so the
    # no-missing fast path is unchanged.
    _miss_checks: list[pl.Expr] = []
    if _domain_missing_cols:
        _dm_schema = df.schema
        for c in _domain_missing_cols:
            e = pl.col(c).is_null()
            if _dm_schema[c] in (pl.Float32, pl.Float64):
                e = e | pl.col(c).is_nan() | pl.col(c).is_infinite()
            _miss_checks.append(e)
        if not df.select(pl.any_horizontal(_miss_checks).any()).item():
            _miss_checks = []

    if where is not None or _miss_checks:
        _keep_exprs: list[pl.Expr] = []
        if where is not None:
            _keep_exprs.append(sample.estimation._compile_where_expr(where))
        if _miss_checks:
            _keep_exprs.append(~pl.any_horizontal(_miss_checks))
        where_expr = pl.all_horizontal(_keep_exprs) if len(_keep_exprs) > 1 else _keep_exprs[0]

        # Step 1: materialize the boolean mask once.
        df = df.with_columns(where_expr.alias(_MASK_COL))
        mask = pl.col(_MASK_COL)

        # Step 2: build the fused expression list for the domain flag and
        # all weight zeroing in a single pass over the frame.
        exprs: list[pl.Expr] = [mask.cast(pl.String).alias("__svy_domain__")]

        # Main weight: zero on the non-domain branch. If design has no wgt
        # the column doesn't exist yet — synthesize it inline. This one column
        # is what the Taylor path and the replicate full-sample estimate read,
        # so it is always zeroed (cheap).
        if design.wgt:
            exprs.append(
                pl.when(mask)
                .then(pl.col(weight_col).cast(pl.Float64))
                .otherwise(0.0)
                .alias(weight_col)
            )
        else:
            exprs.append(pl.when(mask).then(pl.lit(1.0)).otherwise(0.0).alias(weight_col))

        if domain_mask_for_replication and rep_weight_cols_in_df:
            # Fast path (ungrouped replication): emit ONE Float64 domain mask
            # (1.0/0.0) and cast the replicate weights to Float64 WITHOUT
            # zeroing them. The Rust replicate kernels multiply the mask in on
            # the fly, avoiding the R materialised zeroed columns this loop
            # would otherwise write (R×n float stores per `where=` call).
            exprs.append(mask.cast(pl.Float64).alias(_DOMAIN_MASK_COL))
            for c in rep_weight_cols_in_df:
                exprs.append(pl.col(c).cast(pl.Float64).alias(c))
            domain_mask_col = _DOMAIN_MASK_COL
        else:
            # Default: cast + zero every replicate weight in one pass per
            # column (grouped/median replication and non-mask-aware callers).
            for c in rep_weight_cols_in_df:
                exprs.append(pl.when(mask).then(pl.col(c).cast(pl.Float64)).otherwise(0.0).alias(c))

        # Fill missing numeric analysis values with 0.0 — every such row is
        # out-of-domain with zero weight, but the Rust kernels require
        # non-null y/x. Non-missing values on out-of-domain rows are kept
        # untouched (rank tests read them).
        if _miss_checks:
            _fill_targets = [y_col] if cast_y_float else []
            if x:
                _fill_targets.append(x)
            for c in _fill_targets:
                if c not in df.columns:
                    continue
                _miss_c = pl.col(c).is_null()
                if df.schema[c] in (pl.Float32, pl.Float64):
                    _miss_c = _miss_c | pl.col(c).is_nan() | pl.col(c).is_infinite()
                exprs.append(
                    pl.when(_miss_c)
                    .then(0.0)
                    .otherwise(pl.col(c).cast(pl.Float64))
                    .alias(c)
                )

        df = df.with_columns(exprs).drop(_MASK_COL)
        domain_col = "__svy_domain__"
        domain_val = "true"

        # A variable may appear in BOTH `by` and `where` (issue #9). When a
        # `where` predicate excludes an entire `by` level (every one of its rows
        # out of the domain, e.g. a "don't know"/missing code), that level has
        # no domain estimate to report and must simply be absent from the
        # results — matching R's `design %>% filter(...) %>% group_by(...)`.
        #
        # We do NOT drop those rows: domain estimation requires the full sample
        # with out-of-domain weights zeroed so the design-based (Taylor /
        # replicate) variance of the surviving groups stays correct. Instead we
        # blank the grouping *label* of a fully-excluded level, so it forms no
        # group in the kernel (which iterates `by.unique()` and skips nulls) and
        # produces no row — while every row still flows to Rust and continues to
        # contribute to the shared design/df exactly as before. Partially
        # excluded levels keep their label (they have in-domain rows) and are
        # domain-estimated over their surviving observations as usual.
        if by_col is not None and by_col in df.columns:
            live_levels = (
                df.filter(pl.col(domain_col) == domain_val)
                .get_column(by_col)
                .unique()
                .to_list()
            )
            df = df.with_columns(
                pl.when(pl.col(by_col).is_in(live_levels))
                .then(pl.col(by_col))
                .otherwise(None)
                .alias(by_col)
            )
    else:
        # No where → still need (a) the main weight column to exist and be
        # Float64 for Rust, and (b) replicate weights to be Float64. Bundle
        # both into a single ``with_columns`` call to minimize passes.
        no_where_exprs: list[pl.Expr] = []

        if design.wgt:
            if df[weight_col].dtype != pl.Float64:
                no_where_exprs.append(pl.col(weight_col).cast(pl.Float64))
        else:
            no_where_exprs.append(pl.lit(1.0).alias(weight_col))

        for c in rep_weight_cols_in_df:
            if df[c].dtype != pl.Float64:
                no_where_exprs.append(pl.col(c).cast(pl.Float64))

        if no_where_exprs:
            df = df.with_columns(no_where_exprs)

    # ── Type casting (y, x, group, strata, psu, ssu, by) ─────────────────
    # Fused into a single with_columns call. Each cast checks the current
    # dtype and is skipped when already correct (Polars treats same-type
    # cast as a no-op anyway, but the explicit guard keeps the expression
    # list short).
    casts: list[pl.Expr] = []
    if cast_y_float:
        if df[y_col].dtype != pl.Float64:
            casts.append(pl.col(y_col).cast(pl.Float64))
    if x and df[x].dtype != pl.Float64:
        casts.append(pl.col(x).cast(pl.Float64))
    if group and df[group].dtype != pl.String:
        casts.append(pl.col(group).cast(pl.String))
    # Design columns → String, EXCEPT the Phase C integer code columns, which
    # must stay UInt32 so the Rust kernel takes its integer fast path.
    _code_cols = {_STRATUM_CODE, _PSU_CODE, _SSU_CODE}
    if (
        strata_col
        and strata_col not in _code_cols
        and strata_col in df.columns
        and df[strata_col].dtype != pl.String
    ):
        casts.append(pl.col(strata_col).cast(pl.String))
    if (
        psu_col
        and psu_col not in _code_cols
        and psu_col in df.columns
        and df[psu_col].dtype != pl.String
    ):
        casts.append(pl.col(psu_col).cast(pl.String))
    if (
        ssu_col
        and ssu_col not in _code_cols
        and ssu_col in df.columns
        and df[ssu_col].dtype != pl.String
    ):
        casts.append(pl.col(ssu_col).cast(pl.String))
    if by_col and by_col in df.columns and df[by_col].dtype != pl.String:
        casts.append(pl.col(by_col).cast(pl.String))
    if casts:
        df = df.with_columns(casts)

    # ── Singleton method ─────────────────────────────────────────────────
    singleton_method = getattr(design, "singleton_method", None)

    return PreparedData(
        df=df,
        y_col=y_col,
        weight_col=weight_col,
        strata_col=strata_col,
        psu_col=psu_col,
        ssu_col=ssu_col,
        fpc_col=None,  # FPC computed separately by estimation facade
        fpc_ssu_col=None,  # when needed (not all APIs use FPC)
        domain_col=domain_col,
        domain_val=domain_val,
        domain_mask_col=domain_mask_col,
        by_col=by_col,
        by_cols=by_cols_list,
        singleton_method=singleton_method,
    )
