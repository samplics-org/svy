# src/svy/weighting/replication.py
"""
Replicate weight creation: BRR, Jackknife, Bootstrap, SDR.

Each function takes a Sample and returns a Sample (for chaining).
The Weighting class in base.py delegates to these functions directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import polars as pl


try:
    from svy_rs._internal import (
        brr_hadamard_size as rust_brr_hadamard_size,  # type: ignore[import-untyped]
    )
    from svy_rs._internal import (
        create_bootstrap_wgts as rust_create_bootstrap_wgts,  # type: ignore[import-untyped]
    )
    from svy_rs._internal import (
        create_brr_wgts as rust_create_brr_wgts,  # type: ignore[import-untyped]
    )
    from svy_rs._internal import (
        create_jk_wgts as rust_create_jk_wgts,  # type: ignore[import-untyped]
    )
    from svy_rs._internal import (
        create_poisson_bootstrap_wgts as rust_create_poisson_bs_wgts,  # type: ignore[import-untyped]
    )
    from svy_rs._internal import (
        create_sdr_wgts as rust_create_sdr_wgts,  # type: ignore[import-untyped]
    )
except ImportError:  # pragma: no cover
    rust_create_bootstrap_wgts = None
    rust_brr_hadamard_size = None
    rust_create_brr_wgts = None
    rust_create_jk_wgts = None
    rust_create_poisson_bs_wgts = None
    rust_create_sdr_wgts = None

from svy.core.repwgts import (
    BootstrapKind,
    BootstrapWgts,
    BrrWgts,
    JackknifeWgts,
    SdrWgts,
    normalize_bootstrap_kind,
)
from svy.errors import DimensionError, MethodError
from svy.utils.checks import drop_missing
from svy.utils.random_state import RandomState, resolve_random_state
from svy.weighting._helpers import (
    _name_rep_cols,
    _to_float_array,
    _to_int_array,
)


if TYPE_CHECKING:
    from svy.core.design import Design
    from svy.core.sample import Sample


def _recorded_units(design: "Design") -> tuple[str | None, str | None]:
    """The user-facing columns to record as the units these replicates used.

    Provenance, so that a generated design says what it was built from rather
    than leaving a later reader to assume it matches the Design -- which is the
    very assumption the ``stratum``/``psu`` fields exist to stop.

    Multi-column units are recorded as the tuple of their source columns, the
    same shape ``Design`` holds -- never the internal concatenated name, which
    is an implementation detail and would not resolve against a frame rebuilt
    from source.
    """
    return _as_recorded(design.stratum), _as_recorded(design.psu)


def _as_recorded(col: str | tuple[str, ...] | None) -> str | tuple[str, ...] | None:
    """The user-facing reference for a unit, str or tuple alike."""
    return col if isinstance(col, (str, tuple)) else None


def _n_strata(n: int) -> str:
    """``n stratum`` / ``n strata``.

    The package spells this ``singleton PSU(s)`` elsewhere, but ``stratum`` is
    irregular so the ``(s)`` suffix does not apply.
    """
    return f"{n} stratum" if n == 1 else f"{n} strata"


def _fmt_strata(items: list, limit: int = 5) -> str:
    """The offending strata, saying how many were withheld rather than ``...``.

    Every one of these has to be fixed individually, so a bare ellipsis hides
    the part of the job the reader still has to go and find.
    """
    shown = ", ".join(f"{name}={count}" for name, count in items[:limit])
    extra = len(items) - limit
    return shown if extra <= 0 else f"{shown} (+{extra} more)"


def _pair_variance_strata(
    sample: Sample,
    *,
    method: Literal["brr", "jk2"],
    psu_col: str | tuple[str, ...],
    stratum_col: str | tuple[str, ...] | None,
    where: str,
    order_by: str | Sequence[str] | None = None,
    shuffle: bool = False,
    into: str = "svy_var_stratum",
    rstate: int | None = None,
) -> str:
    """Pair PSUs within each stratum into variance strata; return the column.

    Writes the paired identifier into ``sample._data`` and returns its name.
    Deliberately does **not** touch ``Design.stratum``: variance strata are how
    the replicates were drawn, the design stratum is what Taylor linearizes
    over, and the public two-step this replaced conflated them -- it overwrote
    the design, so every later Taylor estimate silently linearized over
    collapsed pseudo-strata. The caller records the returned name on the
    replicate weights instead.

    Private because pairing is a step in building BRR/JK2 weights rather than a
    thing to do on its own: its only output has nowhere to live until there are
    replicate weights to attach it to.
    """
    df = sample._data

    _method = method.lower()
    if _method not in ("brr", "jk2"):
        raise MethodError.invalid_choice(
            where=where,
            param="method",
            got=_method,
            allowed=["brr", "jk2"],
        )

    orig_stratum_col = stratum_col

    # When stratum is a tuple (multi-column), use the internal concatenated
    # column which already exists in the data as a single string column.
    if isinstance(orig_stratum_col, (tuple, list)):
        _internal = getattr(sample, "_internal_design", None) or {}
        stratum_col_for_grouping = _internal.get("stratum")
        if stratum_col_for_grouping is None or stratum_col_for_grouping not in df.columns:
            raise MethodError.not_applicable(
                where=where,
                method="variance strata pairing",
                reason=(
                    f"Multi-column stratum {orig_stratum_col} requires an internal "
                    f"concatenated column, but it was not found in the data."
                ),
                hint="Ensure the Sample was constructed with the stratum columns present.",
            )
        # Use individual source columns for select, but the concat column for grouping
        stratum_source_cols = list(orig_stratum_col)
    elif orig_stratum_col is not None:
        stratum_col_for_grouping = orig_stratum_col
        stratum_source_cols = [orig_stratum_col]
    else:
        stratum_col_for_grouping = None
        stratum_source_cols = []

    if order_by is None:
        order_cols = []
    elif isinstance(order_by, str):
        order_cols = [order_by]
    else:
        order_cols = list(order_by)

    missing_cols = [col for col in order_cols if col not in df.columns]
    if missing_cols:
        raise MethodError.invalid_choice(
            where=where,
            param="order_by",
            got=missing_cols,
            allowed=df.columns,
            hint="Check column names.",
        )

    select_cols = [psu_col]
    # Include stratum source columns AND the concat column (if different)
    for c in stratum_source_cols:
        if c not in select_cols:
            select_cols.append(c)
    if stratum_col_for_grouping and stratum_col_for_grouping not in select_cols:
        select_cols.append(stratum_col_for_grouping)
    select_cols.extend([c for c in order_cols if c not in select_cols])

    # One row per PSU. `order_by` is a PSU-level attribute, but nothing stops a
    # caller naming a row-level column; uniquing on the whole row then left the
    # same PSU present several times, and it would be assigned to several
    # variance strata, leaving some holding a single PSU. First occurrence in
    # the frame wins, which is what "order by this column" means once the column
    # varies within a PSU.
    _psu_keys = [psu_col] if isinstance(psu_col, str) else list(psu_col)
    psu_df = df.select(select_cols).unique(subset=_psu_keys, keep="first", maintain_order=True)

    sort_cols = []
    if stratum_col_for_grouping:
        sort_cols.append(stratum_col_for_grouping)
    sort_cols.extend(order_cols)
    if psu_col not in sort_cols:
        sort_cols.append(psu_col)

    psu_df = psu_df.sort(sort_cols)

    psu_list = psu_df[psu_col].to_list()
    n_psus = len(psu_list)

    if stratum_col_for_grouping:
        orig_strata = psu_df[stratum_col_for_grouping].to_numpy()
    else:
        orig_strata = np.zeros(n_psus, dtype=np.int64)

    unique_orig, counts = np.unique(orig_strata, return_counts=True)
    stratum_counts = dict(zip(unique_orig.tolist(), counts.tolist()))

    if _method == "brr":
        # BRR's unit is the variance stratum, and it must hold exactly 2 PSUs --
        # that is what a balanced half-sample selects from. So a stratum is
        # usable iff its PSU count is a multiple of 2, and a stratum of 1 fails
        # for the same reason a stratum of 3 does: a PSU with no partner.
        #
        # They used to be two guards, the `< 2` one raising first, so a frame
        # with both reported only the singleton and revealed the odd strata on
        # the next run. Treating 1 as a different kind of problem also borrowed
        # a question that is not BRR's: a lone PSU in a stratum is a singleton,
        # which matters to Taylor linearization and belongs to
        # ``Sample.singleton``. Here it is simply not 2.
        unpairable = [(s, c) for s, c in stratum_counts.items() if c % 2 != 0]
        if unpairable:
            raise DimensionError(
                title="BRR needs 2 PSUs per variance stratum",
                detail=(
                    f"BRR pairs PSUs into variance strata of exactly 2. "
                    f"Found {_n_strata(len(unpairable))} whose PSU count is not "
                    f"a multiple of 2, leaving a PSU with no partner."
                ),
                code="ODD_PSU_COUNT",
                where=where,
                param="stratum",
                expected="A multiple of 2 PSUs per stratum",
                got=_fmt_strata(unpairable),
                hint=(
                    "create_jk_wgts(paired=True) pairs these strata itself and "
                    "absorbs an odd count into a triplet. BRR cannot: its "
                    "balanced half-samples need exactly 2 PSUs per variance "
                    "stratum. If BRR is required, fix every stratum listed, not "
                    "just one -- drop or combine a PSU so each count is even."
                ),
            )
    else:  # jk2: absorbs an odd count into a triplet, but cannot pair a lone PSU
        small_strata = [(s, c) for s, c in stratum_counts.items() if c < 2]
        if small_strata:
            raise DimensionError(
                title="Insufficient PSUs per stratum",
                detail=(
                    f"All strata must have at least 2 PSUs. "
                    f"Found {len(small_strata)} strata with fewer."
                ),
                code="INSUFFICIENT_PSU",
                where=where,
                param="stratum",
                expected="≥2 PSUs per stratum",
                got=f"{small_strata[:5]}{'...' if len(small_strata) > 5 else ''}",
                hint="Combine small strata or check design specification.",
            )

    var_strata = np.empty(n_psus, dtype=np.int64)
    rng = np.random.default_rng(rstate) if shuffle and not order_cols else None
    var_stratum_counter = 0

    for orig_str in unique_orig:
        mask = orig_strata == orig_str
        indices = np.where(mask)[0]
        n_in_stratum = len(indices)

        if rng is not None:
            indices = indices.copy()
            rng.shuffle(indices)

        if _method == "brr":
            for i in range(0, n_in_stratum, 2):
                var_strata[indices[i]] = var_stratum_counter
                var_strata[indices[i + 1]] = var_stratum_counter
                var_stratum_counter += 1
        else:  # jk2
            if n_in_stratum % 2 == 1:
                for i in range(0, n_in_stratum - 3, 2):
                    var_strata[indices[i]] = var_stratum_counter
                    var_strata[indices[i + 1]] = var_stratum_counter
                    var_stratum_counter += 1
                var_strata[indices[-3]] = var_stratum_counter
                var_strata[indices[-2]] = var_stratum_counter
                var_strata[indices[-1]] = var_stratum_counter
                var_stratum_counter += 1
            else:
                for i in range(0, n_in_stratum, 2):
                    var_strata[indices[i]] = var_stratum_counter
                    var_strata[indices[i + 1]] = var_stratum_counter
                    var_stratum_counter += 1

    # psu_list/var_strata are PSU-level (one per PSU, in sort order).
    # Build the mapping then expand to observation level via a left join.
    # unique() on psu_col guards against psu_list having duplicate entries
    # when order_by columns cause psu_df to contain repeated PSU rows.
    mapping_df = pl.DataFrame({psu_col: psu_list, "__vs__": var_strata.tolist()}).unique(
        subset=[psu_col], keep="first"
    )
    obs_var_strata = (
        df.select(psu_col)
        .join(mapping_df, on=psu_col, how="left")
        .get_column("__vs__")
        .to_numpy()
        .astype(np.int64)
    )

    sample._data = df.with_columns(pl.Series(name=into, values=obs_var_strata))
    return into


def _resolve_build_units(
    sample: Sample,
    *,
    where: str,
    pair_method: Literal["brr", "jk2"] | None,
    stratum: str | None,
    psu: str | None,
    stratum_name: str = "svy_var_stratum",
    order_by: str | Sequence[str] | None = None,
    shuffle: bool = False,
    rstate: int | None = None,
) -> tuple[str | tuple[str, ...] | None, str | tuple[str, ...] | None]:
    """Resolve which units to build replicates from, pairing them if required.

    An explicit ``stratum``/``psu`` wins over the Design's. The Design describes
    the analysis design; a producer's variance units need not be the same
    columns, and the generator should be able to say so without the caller
    having to mutate the design to get there.

    ``pair_method`` asks for BRR/JK2 pairing, which is needed only when a
    stratum carries more than two PSUs. Pairing used to be a separate public
    step whose sole way of handing its result along was to overwrite
    ``Design.stratum`` -- so the workflow that built the weights also destroyed
    the design they were meant to be compared against.
    """
    design = sample._design
    for name, col in (("stratum", stratum), ("psu", psu)):
        if col is not None and col not in sample._data.columns:
            raise MethodError.invalid_choice(
                where=where,
                param=name,
                got=col,
                allowed=list(sample._data.columns),
                hint=(
                    f"'{name}' names the column these replicates are built from. "
                    f"Leave it unset to use the Design's."
                ),
            )
    psu_col = psu if psu is not None else design.psu
    stratum_col = stratum if stratum is not None else design.stratum
    if pair_method is None or psu_col is None:
        return stratum_col, psu_col

    # Pair only when something needs pairing: a design already at two PSUs per
    # stratum *is* its own variance-stratum scheme, and re-deriving one would
    # rename the column for no gain. With no strata at all there is one implicit
    # group holding every PSU, so there is always something to pair.
    if stratum_col is not None:
        group_cols = [stratum_col] if isinstance(stratum_col, str) else list(stratum_col)
        psu_cols = [psu_col] if isinstance(psu_col, str) else list(psu_col)
        per_stratum = (
            sample._data.select(list(dict.fromkeys(group_cols + psu_cols)))
            .unique()
            .group_by(group_cols)
            .len()
            .get_column("len")
        )
        # min() matters as much as max(): a stratum with a single PSU is an
        # error the pairing helper reports precisely, and returning early here
        # would hand it to the kernel to fail on obscurely instead.
        if per_stratum.max() <= 2 and per_stratum.min() >= 2:
            return stratum_col, psu_col

    paired = _pair_variance_strata(
        sample,
        method=pair_method,
        psu_col=psu_col,
        stratum_col=stratum_col,
        where=where,
        order_by=order_by,
        shuffle=shuffle,
        into=stratum_name,
        rstate=rstate,
    )
    return paired, psu_col


def create_brr_wgts(
    sample: Sample,
    n_reps: int | None = None,
    *,
    stratum: str | None = None,
    psu: str | None = None,
    stratum_name: str = "svy_var_stratum",
    order_by: str | Sequence[str] | None = None,
    shuffle: bool = False,
    rep_prefix: str | None = None,
    fay_coef: float = 0.0,
    rstate: int | None = None,
    drop_nulls: bool = False,
) -> Sample:
    """Create balanced repeated replication weights.

    BRR needs two PSUs per stratum. Strata carrying more are paired into
    variance strata first -- previously a separate public call whose only way
    of handing the result along was to overwrite ``Design.stratum``. The paired
    column is written as ``stratum_name`` and recorded on the replicate
    weights; the Design is left alone, so Taylor keeps the true strata.

    ``stratum``/``psu`` name the units to build from when they are not the
    Design's. ``order_by`` pairs adjacent PSUs in that order, which is what a
    systematically-sampled frame wants; ``shuffle`` pairs at random instead.
    """
    where = "Sample.weighting.create_brr_wgts"
    design = sample._design

    if (psu if psu is not None else design.psu) is None:
        raise MethodError.not_applicable(
            where=where,
            method="create_brr_wgts",
            reason="BRR requires psu (got psu=None)",
            hint="Pass psu=, or set it on the Design.",
        )

    strat_col, psu_col = _resolve_build_units(
        sample,
        where=where,
        pair_method="brr",
        stratum=stratum,
        psu=psu,
        stratum_name=stratum_name,
        order_by=order_by,
        shuffle=shuffle,
        rstate=rstate,
    )
    df = sample._data  # refreshed: pairing may have just added a column

    if drop_nulls:
        needed = list({c for c in [design.wgt, strat_col, psu_col] if isinstance(c, str)})
        data = drop_missing(df=df, cols=needed, treat_infinite_as_missing=True)
    else:
        data = df

    main_weights = _to_float_array(data, design.wgt, len(data))
    stratum_int = _to_int_array(data, strat_col)
    psu_int = _to_int_array(data, psu_col)

    # BRR replicate count is bounded by the Hadamard matrix size for the
    # design's strata: fewer requested reps are rounded UP to it (balance
    # needs the full Hadamard set); more cannot exist without silently
    # duplicating replicate columns, so that is an error.
    if n_reps is not None and rust_brr_hadamard_size is not None:
        h_size = rust_brr_hadamard_size(len(set(stratum_int.tolist())))
        if n_reps > h_size:
            raise MethodError.invalid_range(
                where="Sample.weighting.create_brr_wgts",
                param="n_reps",
                got=n_reps,
                min_=1,
                max_=h_size,
                hint=f"BRR with this design supports at most {h_size} distinct "
                f"replicates (the Hadamard matrix size). Pass n_reps<={h_size} "
                "or omit n_reps to use the full set.",
            )

    assert rust_create_brr_wgts is not None  # noqa: S101
    rep_mat, df_val = rust_create_brr_wgts(
        main_weights,
        stratum_int,
        psu_int,
        n_reps,
        fay_coef,
        rstate,
    )
    n_reps_actual = rep_mat.shape[1]

    rep_prefix = rep_prefix or design.wgt
    rep_cols = _name_rep_cols(rep_prefix, n_reps_actual)
    rep_dicts = {col: rep_mat[:, i] for i, col in enumerate(rep_cols)}
    sample._data = data.with_columns(
        [pl.Series(name=col, values=vals) for col, vals in rep_dicts.items()]
    )

    _rec_stratum, _rec_psu = _as_recorded(strat_col), _as_recorded(psu_col)
    # `update`, not `fill_missing`: these columns were just written, so the
    # design has to describe *them*. fill_missing is a no-op once rep_wgts is
    # set, which left a second generator's columns in the frame under the first
    # one's metadata -- and estimation reads the metadata.
    sample._design = sample._design.update(
        rep_wgts=BrrWgts(
            prefix=rep_prefix,
            n_reps=n_reps_actual,
            fay_coef=fay_coef,
            df=df_val,
            stratum=_rec_stratum,
            psu=_rec_psu,
        )
    )

    return sample


def create_jk_wgts(
    sample: Sample,
    *,
    paired: bool = False,
    stratum: str | None = None,
    psu: str | None = None,
    stratum_name: str = "svy_var_stratum",
    order_by: str | Sequence[str] | None = None,
    shuffle: bool = False,
    rep_prefix: str | None = None,
    rstate: int | None = None,
    drop_nulls: bool = False,
) -> Sample:
    """Create delete-one-PSU jackknife replicate weights.

    ``paired=True`` builds JK2: one replicate per stratum, each deleting a PSU
    from a two-PSU variance stratum. Strata carrying more than two PSUs are
    paired first -- which used to be a separate public call, and skipping it
    produced one replicate per *original* stratum in silence rather than an
    error. ``paired=False`` builds JK1/JKn and uses the strata exactly as given.

    ``stratum``/``psu`` name the units to build from when they are not the
    Design's. The units used are recorded on the resulting weights, so a later
    reader does not have to assume they match the Design.
    """
    df = sample._data
    design = sample._design

    where = "Sample.weighting.create_jk_wgts"

    if (psu if psu is not None else design.psu) is None:
        raise MethodError.not_applicable(
            where=where,
            method="create_jk_wgts",
            reason="Jackknife requires psu (got psu=None)",
            hint="Pass psu=, or set it on the Design.",
        )

    # Only the paired scheme pairs. jk1 and jkn delete one PSU at a time and
    # want the strata exactly as given.
    strat_col, psu_col = _resolve_build_units(
        sample,
        where=where,
        pair_method="jk2" if paired else None,
        stratum=stratum,
        psu=psu,
        stratum_name=stratum_name,
        order_by=order_by,
        shuffle=shuffle,
        rstate=rstate,
    )
    df = sample._data  # refreshed: pairing may have just added a column

    if drop_nulls:
        needed = list({c for c in [design.wgt, strat_col, psu_col] if isinstance(c, str)})
        data = drop_missing(df=df, cols=needed, treat_infinite_as_missing=True)
    else:
        data = df

    main_weights = _to_float_array(data, design.wgt, len(data))
    psu_int = _to_int_array(data, psu_col)
    stratum_int = _to_int_array(data, strat_col)

    assert rust_create_jk_wgts is not None  # noqa: S101
    rep_mat, df_val, rep_coefs = rust_create_jk_wgts(
        main_weights,
        psu_int,
        stratum_int,
        paired,
        rstate,
    )
    n_reps = rep_mat.shape[1]

    rep_prefix = rep_prefix or design.wgt
    rep_cols = _name_rep_cols(rep_prefix, n_reps)
    rep_dicts = {col: rep_mat[:, i] for i, col in enumerate(rep_cols)}
    sample._data = data.with_columns(
        [pl.Series(name=col, values=vals) for col, vals in rep_dicts.items()]
    )

    # svy generated these, so the family is a fact rather than an assumption.
    # A single stratum is not merely "close to" JK1: JKn's (n_h-1)/n_h with one
    # stratum of n PSUs gives R = n replicates each at (n-1)/n, which is exactly
    # the JK1 global. The collapse is identical, not approximate.
    if paired:
        jk_kind = "jk2"
    elif strat_col is None or len(np.unique(stratum_int)) <= 1:
        jk_kind = "jk1"
    else:
        jk_kind = "jkn"

    _rec_stratum, _rec_psu = _as_recorded(strat_col), _as_recorded(psu_col)
    # `update`, not `fill_missing` -- see the note in create_brr_wgts.
    sample._design = sample._design.update(
        rep_wgts=JackknifeWgts(
            prefix=rep_prefix,
            n_reps=n_reps,
            df=df_val,
            kind=jk_kind,
            stratum=_rec_stratum,
            psu=_rec_psu,
            # Per-replicate (n_h-1)/n_h coefficients: exact stratified-JKn
            # variance instead of the global (R-1)/R approximation. The computed
            # channel, not `scale` -- svy derived these, the user did not assert
            # them, and only svy can (it has the strata; coefficients() does not).
            rep_coefs=tuple(rep_coefs),
        )
    )

    return sample


def create_bs_wgts(
    sample: Sample,
    n_reps: int = 500,
    *,
    kind: BootstrapKind = "rao-wu",
    stratum: str | None = None,
    psu: str | None = None,
    rep_prefix: str | None = None,
    drop_nulls: bool = False,
    rstate: RandomState = None,
) -> Sample:
    """Create bootstrap replicate weights.

    Parameters
    ----------
    kind : {"rao-wu", "poisson"}, default "rao-wu"
        ``"rao-wu"`` is the stratified Rao-Wu-Yue rescaling bootstrap: PSUs are
        resampled within strata, so it requires ``psu`` on the design. Where the
        design is available this is the better choice, because resampling PSUs
        carries the stratification and clustering with it.

        ``"poisson"`` is the Beaumont-Patak generalized bootstrap with
        independent per-unit adjustment factors. It requires only a weight
        column, which is why producers document it for public use files where
        stratum and PSU identifiers are suppressed. The draws being independent
        across units is also what makes the weights non-disclosive. It cannot
        recover clustering, so use it when the design is genuinely unavailable
        rather than as a default.

    Notes
    -----
    Calibrating the replicates is a separate step and is not performed here.
    Producers sometimes document one -- Statistics Canada's LFS guide describes
    "a calibrated version" of the Poisson bootstrap that "can" bring the
    variance closer to a master-file estimate -- but it is ordinary
    post-stratification, it applies to any replicate method, and it belongs with
    the other weighting adjustments:

    >>> s = sample.weighting.create_bs_wgts(n_reps=1000, kind="poisson")
    >>> s = s.weighting.poststratify(controls=totals, by=["prov", "sex", "age"])

    :meth:`Sample.weighting.poststratify`, :meth:`~rake` and :meth:`~calibrate`
    all adjust the replicate columns alongside the main weight unless
    ``ignore_reps=True``.

    References for ``"poisson"``: Beaumont, J.-F. and Patak, Z. (2012). On the
    generalized bootstrap for sample surveys with special attention to Poisson
    sampling. *International Statistical Review*, 80(1), 127-148.
    """
    df = sample._data
    design = sample._design
    kind = normalize_bootstrap_kind(kind)

    if n_reps is None:
        raise MethodError.not_applicable(
            where="Sample.weighting.create_bs_wgts",
            method="create_bs_wgts",
            reason="n_reps must be specified for Bootstrap.",
        )
    # The Rao-Wu guard is deliberately not shared: the Poisson bootstrap exists
    # precisely for files that have no psu, so requiring one would reject the
    # only case it serves.
    if kind == "rao-wu" and (psu if psu is not None else design.psu) is None:
        raise MethodError.not_applicable(
            where="Sample.weighting.create_bs_wgts",
            method="create_bs_wgts",
            reason="Bootstrap requires psu in Design (got psu=None).",
        )

    strat_col, psu_col = _resolve_build_units(
        sample,
        where="Sample.weighting.create_bs_wgts",
        pair_method=None,  # resampling PSUs needs no pairing
        stratum=stratum,
        psu=psu,
    )

    if drop_nulls:
        if kind == "poisson":
            candidates: list[str | tuple[str, ...] | None] = [design.wgt]
        else:
            candidates = [design.wgt, strat_col, psu_col]
        needed = list({c for c in candidates if isinstance(c, str)})
        data = drop_missing(df=df, cols=needed, treat_infinite_as_missing=True)
    else:
        data = df

    main_weights = _to_float_array(data, design.wgt, len(data))

    rng = resolve_random_state(rstate)
    seed = (
        int(rng.integers(0, 2**63 - 1))
        if hasattr(rng, "integers")
        else int(rng.randint(0, 2**31 - 1))
    )

    if kind == "poisson":
        assert rust_create_poisson_bs_wgts is not None  # noqa: S101
        rep_mat, df_val = rust_create_poisson_bs_wgts(main_weights, n_reps, seed)
    else:
        psu_int = _to_int_array(data, psu_col)
        stratum_int = _to_int_array(data, strat_col)
        assert rust_create_bootstrap_wgts is not None  # noqa: S101
        rep_mat, df_val = rust_create_bootstrap_wgts(
            main_weights,
            psu_int,
            n_reps,
            stratum_int,
            seed,
        )

    rep_prefix = rep_prefix or design.wgt
    rep_cols = _name_rep_cols(rep_prefix, n_reps)
    rep_dicts = {col: rep_mat[:, i] for i, col in enumerate(rep_cols)}
    sample._data = data.with_columns(
        [pl.Series(name=col, values=vals) for col, vals in rep_dicts.items()]
    )

    _rec_stratum, _rec_psu = _as_recorded(strat_col), _as_recorded(psu_col)
    sample._design = sample._design.update(
        rep_wgts=BootstrapWgts(
            prefix=rep_prefix,
            n_reps=n_reps,
            df=df_val,
            kind=kind,
            # The Poisson bootstrap draws independent per-unit factors, so it
            # genuinely has no units to record -- that is why it works on files
            # with no psu at all.
            stratum=None if kind == "poisson" else _rec_stratum,
            psu=None if kind == "poisson" else _rec_psu,
        )
    )

    return sample


def create_sdr_wgts(
    sample: Sample,
    n_reps: int = 4,
    *,
    psu: str | None = None,
    rep_prefix: str | None = None,
    order_col: str | None = None,
    drop_nulls: bool = False,
) -> Sample:
    df = sample._data
    design = sample._design

    if n_reps < 2:
        raise MethodError.invalid_range(
            where="Sample.weighting.create_sdr_wgts",
            param="n_reps",
            got=n_reps,
            min_=2,
            hint="SDR requires at least 2 replicates.",
        )

    if drop_nulls:
        needed = list({c for c in [design.wgt, design.stratum] if isinstance(c, str)})
        if order_col:
            needed.append(order_col)
        data = drop_missing(df=df, cols=needed, treat_infinite_as_missing=True)
    else:
        data = df

    main_weights = _to_float_array(data, design.wgt, len(data))
    stratum_int = _to_int_array(data, design.stratum)

    order_int: np.ndarray | None = None
    if order_col and order_col in data.columns:
        order_int = _to_int_array(data, order_col)

    assert rust_create_sdr_wgts is not None  # noqa: S101
    rep_mat, df_val = rust_create_sdr_wgts(
        main_weights,
        n_reps,
        stratum_int,
        order_int,
    )

    rep_prefix = rep_prefix or design.wgt
    rep_cols = _name_rep_cols(rep_prefix, n_reps)
    rep_dicts = {col: rep_mat[:, i] for i, col in enumerate(rep_cols)}
    sample._data = data.with_columns(
        [pl.Series(name=col, values=vals) for col, vals in rep_dicts.items()]
    )

    _rec_stratum, _rec_psu = _recorded_units(design)
    if psu is not None:
        _rec_psu = psu
    sample._design = sample._design.update(
        rep_wgts=SdrWgts(
            prefix=rep_prefix, n_reps=n_reps, df=df_val, stratum=_rec_stratum, psu=_rec_psu
        )
    )

    return sample
