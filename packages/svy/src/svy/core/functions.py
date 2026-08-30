# src/svy/core/functions.py
"""Public free functions over core objects.

A method says "this operates on your survey"; a free function says "this is a
computation over things you supply" — see FREE_FUNCTIONS.md at the repo root
for the method-vs-function test and the per-subpackage functions.py convention.
"""

from __future__ import annotations

import logging
import warnings

from typing import Literal, Sequence

import msgspec
import polars as pl

from svy.core.constants import SVY_ROW_INDEX
from svy.core.design import Design, PopSize
from svy.core.enumerations import MeasurementType, MetadataSource
from svy.core.sample import Sample
from svy.core.types import Category
from svy.errors import MethodError
from svy.metadata.variable_meta import VariableMeta


__all__ = ["combine_samples"]

log = logging.getLogger(__name__)

_CTX = "svy.combine_samples"

_SINGLE_ROLES = ("wgt", "prob", "hit", "mos")
_MULTI_ROLES = ("stratum", "psu", "ssu")


def _as_tuple(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _role_columns(design: Design) -> dict[str, tuple[str, ...]]:
    """Every design role as a (possibly empty) tuple of column names."""
    out: dict[str, tuple[str, ...]] = {}
    for role in _MULTI_ROLES:
        out[role] = _as_tuple(getattr(design, role))
    for role in _SINGLE_ROLES:
        val = getattr(design, role)
        out[role] = (val,) if val is not None else ()
    ps = design.pop_size
    if ps is None:
        out["pop_size"] = ()
    elif isinstance(ps, PopSize):
        out["pop_size"] = tuple(c for c in (ps.psu, ps.ssu) if c is not None)
    else:
        out["pop_size"] = (ps,)
    return out


# Presence mismatches relaxable via on_mixed_design: for each, the wave lacking
# the role has a COMPLETE design whose translation into the stacked encoding is
# exact (verified: no-stratum SE == single-stratum SE, no-psu SE ==
# element-as-PSU SE). ssu and pop_size are NOT here — their translations are
# unestablished, so they always error.
_MIXABLE_ROLES = {
    "stratum": "treated as a single stratum",
    "psu": "treated as element-sampled (each row its own PSU)",
}


def _check_design_alignment(
    samples: Sequence[Sample], units: str, allow_mixed: bool
) -> tuple[dict[str, tuple[str, ...]], dict[str, list[bool]], list[str]]:
    """Require identical design columns, role by role; no implicit renaming.

    Column names carry meaning in the output, so a name mismatch is the user's
    to resolve — the hint spells out the one-line fix, which updates data,
    design and metadata together.

    Opt-in exception, independent mode only: waves may declare structurally
    DIFFERENT but individually complete designs (stratified vs not, clustered
    vs not) — each wave's variance structure is self-contained after wave
    qualification, so combining them is valid, and the translations are exact
    (see _MIXABLE_ROLES). It stays an error by default because a missing
    declaration is usually an oversight that silently understates variance;
    ``on_mixed_design`` is the user's explicit assertion. Returns per-role
    flags for the waves needing materialized columns and the notes describing
    what was mixed.
    """
    role_maps = [_role_columns(s._design) for s in samples]
    canonical = dict(role_maps[0])

    mixed_flags: dict[str, list[bool]] = {}
    mixed_notes: list[str] = []
    if units == "independent" and allow_mixed:
        for role, translation in _MIXABLE_ROLES.items():
            declared = [i for i, rm in enumerate(role_maps) if rm[role]]
            if declared and len(declared) < len(samples):
                canonical[role] = role_maps[declared[0]][role]
                lacking = [i for i, rm in enumerate(role_maps) if not rm[role]]
                mixed_flags[role] = [i in lacking for i in range(len(samples))]
                mixed_notes.append(
                    f"sample(s) {[i + 1 for i in lacking]} declare no {role}, {translation}"
                )
                role_maps = [
                    dict(rm, **{role: canonical[role]}) if not rm[role] else rm for rm in role_maps
                ]

    for role, cols in canonical.items():
        for j, rm in enumerate(role_maps[1:], start=2):
            if rm[role] == cols:
                continue
            if not cols or not rm[role]:
                with_role, without = (1, j) if cols else (j, 1)
                mixed_hint = (
                    " If the designs genuinely differ (stratified vs not, clustered vs "
                    "not), pass on_mixed_design='warn' (or 'ignore') to combine them "
                    "as declared."
                    if role in _MIXABLE_ROLES and units == "independent"
                    else ""
                )
                raise MethodError.not_applicable(
                    where=_CTX,
                    method="combine_samples",
                    reason=(
                        f"design role '{role}' is declared on sample {with_role} but not "
                        f"on sample {without}"
                    ),
                    hint=(
                        "One combined design describes every wave, so each input must "
                        "declare the same roles. Add the missing declaration (or drop "
                        "it everywhere) via update_design()." + mixed_hint
                    ),
                )
            if len(rm[role]) != len(cols):
                raise MethodError.not_applicable(
                    where=_CTX,
                    method="combine_samples",
                    reason=(
                        f"design role '{role}' differs in arity across inputs: sample 1 "
                        f"has {len(cols)} column(s) {list(cols)}, sample {j} has "
                        f"{len(rm[role])} column(s) {list(rm[role])}"
                    ),
                    hint="Declare the same design roles (with matching arity) on every input.",
                )
            fix = ", ".join(f"{o!r}: {n!r}" for o, n in zip(rm[role], cols) if o != n)
            raise MethodError.not_applicable(
                where=_CTX,
                method="combine_samples",
                reason=(
                    f"design role '{role}' is named differently across inputs: sample 1 "
                    f"uses {list(cols)}, sample {j} uses {list(rm[role])}"
                ),
                hint=(
                    f"Rename upfront so the combined columns keep one meaning, e.g. "
                    f"sample.wrangling.rename_columns({{{fix}}}) on sample {j}."
                ),
            )
    return canonical, mixed_flags, mixed_notes


def _resolve_wave_codes(
    frames: list[pl.DataFrame], wave_name: str
) -> tuple[list[pl.DataFrame], list[Category], bool]:
    """Reuse an existing wave column or create one. Returns (frames, codes, created)."""
    present = [wave_name in f.columns for f in frames]
    if any(present) and not all(present):
        missing = [j for j, p in enumerate(present, start=1) if not p]
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason=(
                f"wave column '{wave_name}' exists in some inputs but not in sample(s) {missing}"
            ),
            hint="Add the column to every input, drop it everywhere, or pick another wave_name.",
        )

    if all(present):
        codes: list[Category] = []
        for j, f in enumerate(frames, start=1):
            vals = f.get_column(wave_name).drop_nulls().unique().to_list()
            if len(vals) != 1 or f.get_column(wave_name).null_count() > 0:
                raise MethodError.not_applicable(
                    where=_CTX,
                    method="combine_samples",
                    reason=(
                        f"reused wave column '{wave_name}' must hold one non-null value "
                        f"per sample; sample {j} has {sorted(map(str, vals))[:5]}"
                        f"{' plus nulls' if f.get_column(wave_name).null_count() > 0 else ''}"
                    ),
                )
            codes.append(vals[0])
        if len(set(codes)) != len(codes):
            raise MethodError.not_applicable(
                where=_CTX,
                method="combine_samples",
                reason=f"reused wave column '{wave_name}' repeats a value across samples: {codes}",
            )
        if any(codes[i] >= codes[i + 1] for i in range(len(codes) - 1)):  # type: ignore[operator]
            raise MethodError.not_applicable(
                where=_CTX,
                method="combine_samples",
                reason=(
                    f"reused wave column '{wave_name}' is not increasing in caller order: "
                    f"{codes}. Caller order is the time order."
                ),
                hint="Pass the samples in increasing wave order, or check your inputs.",
            )
        return frames, codes, False

    codes = list(range(1, len(frames) + 1))
    frames = [
        f.with_columns(pl.lit(code, dtype=pl.Int64).alias(wave_name))
        for f, code in zip(frames, codes)
    ]
    return frames, codes, True


def _merge_metadata(combined: Sample, samples: Sequence[Sample]) -> None:
    """Carry the inputs' variable metadata onto the combined sample.

    Field-wise merge with first-sample-wins: iterating the inputs in reverse
    with ``overwrite=True`` lets every input override the combined store's
    dtype-inferred defaults while earlier samples override later ones. Value
    labels are compared RESOLVED (catalog scheme references included):
    identical across inputs → kept; conflicting → dropped with a loud warning
    — NHANES recodes categories between cycles, so a conflict is a
    semantic-drift detector, not a display nuisance. Explicit measurement-type
    disagreements warn, first wins. A single shared catalog travels; differing
    catalogs cannot both travel, so scheme references are materialized into
    the already-verified resolved labels instead.
    """
    resolved: dict[str, dict[Category, str]] = {}
    conflicted: list[str] = []
    explicit_mtypes: dict[str, MeasurementType] = {}
    mtype_conflicts: list[str] = []

    for s in samples:
        for name in s.meta.variables:
            labels = s.meta.resolve_labels(name).labels
            if labels:
                if name in resolved:
                    if resolved[name] != labels and name not in conflicted:
                        conflicted.append(name)
                else:
                    resolved[name] = labels
            meta = s.meta.get(name)
            if meta is not None and meta.source != MetadataSource.INFERRED:
                prev = explicit_mtypes.get(name)
                if prev is None:
                    explicit_mtypes[name] = meta.mtype
                elif prev != meta.mtype and name not in mtype_conflicts:
                    mtype_conflicts.append(name)

    # First sample wins per field; the label pair (value_labels, scheme_ref) is
    # ONE logical field merged atomically — a VariableMeta cannot hold both.
    fields = [f for f in msgspec.structs.fields(VariableMeta) if f.name != "name"]
    patches: dict[str, dict] = {}
    for s in samples:
        for name in s.meta.variables:
            meta = s.meta.get(name)
            if meta is None:
                continue
            patch = patches.setdefault(name, {})
            for field in fields:
                default = None if field.default is msgspec.NODEFAULT else field.default
                val = getattr(meta, field.name)
                if val == default:
                    continue
                if field.name in ("value_labels", "scheme_ref"):
                    if name in conflicted:
                        continue
                    if "value_labels" in patch or "scheme_ref" in patch:
                        continue
                    patch[field.name] = val
                else:
                    patch.setdefault(field.name, val)
    for name, patch in patches.items():
        existing = combined.meta.get(name)
        if existing is not None and patch:
            combined.meta.set(name, existing.clone(**patch))

    catalogs: list = []
    for s in samples:
        cat = s.meta.catalog
        if cat is not None and all(cat is not c for c in catalogs):
            catalogs.append(cat)
    if len(catalogs) == 1:
        combined.meta.catalog = catalogs[0]
    elif len(catalogs) > 1:
        for name in combined.meta.variables:
            meta = combined.meta.get(name)
            if meta is not None and meta.scheme_ref is not None:
                known = resolved.get(name)
                if known and name not in conflicted:
                    combined.meta.set(name, meta.with_value_labels(known))
                else:
                    combined.meta.set(name, meta.clone(scheme_ref=None))
        warnings.warn(
            "Inputs carry different labelling catalogs; scheme references were "
            "resolved to direct value labels on the combined sample.",
            UserWarning,
            stacklevel=3,
        )

    if conflicted:
        warnings.warn(
            f"Value labels conflict across inputs for {sorted(conflicted)}; the coding "
            "changed between waves. Labels for these variables were DROPPED — recode "
            "before combining.",
            UserWarning,
            stacklevel=3,
        )
    if mtype_conflicts:
        warnings.warn(
            f"Measurement types disagree across inputs for {sorted(mtype_conflicts)}; "
            "the first sample's type was kept.",
            UserWarning,
            stacklevel=3,
        )


def combine_samples(
    samples: Sequence[Sample],
    *,
    adjust: Literal["average", "none"] | None = None,
    wave_name: str = "wave",
    wave_labels: Sequence[str] | None = None,
    units: Literal["independent", "shared"] = "independent",
    on_mixed_design: Literal["error", "warn", "ignore"] = "error",
    wgt_name: str = "combined_wgt",
) -> Sample:
    """Combine repeated cross-sections (or panel waves) into one Sample.

    Stacks the data files and analyzes them as ONE stratified design — this is
    data pooling, not estimate pooling. Each independent wave contributes its
    own strata (the design nests wave → stratum → PSU), so Taylor variance
    treats waves as independent automatically. The estimand under
    ``adjust="average"`` is the PERIOD-AVERAGE population: weights are divided
    by k, which matters only for totals — means, proportions and ratios are
    invariant to it.

    Caller order of ``samples`` IS the time order; the wave column gets ordinal
    codes 1..k in that order (or reuses an existing wave column present in all
    inputs, e.g. NHANES SDDSRVYR, validating it increases in caller order).

    For bespoke factors (e.g. the NCHS 1999–2004 recipe of 4/6 and 2/6 on the
    4-yr and 2-yr weight files), pre-adjust each input with
    ``sample.weighting.normalize(factor=...)`` and pass ``adjust="none"``.

    Never combine for trend questions — use ``by=wave_name`` on the combined
    sample instead. Designs where the same physical PSUs appear in several
    samples (rotating panels, ACS overlap) fit neither mode and are out of
    scope. ``design_history`` of the inputs is not carried: the combined Sample
    is a new object built from k parents.

    Parameters
    ----------
    samples : Sequence[Sample]
        Two or more samples, in time order.
    adjust : {"average", "none"} | None
        "average" multiplies every weight by 1/k into ``wgt_name``. None picks
        the mode default: "average" for independent units, "none" for shared.
        Explicit "average" with ``units="shared"`` errors — a person is not
        half a person for appearing in two waves.
    wave_name : str
        Wave-id column name; reused if already present in all inputs.
    wave_labels : Sequence[str] | None
        Value labels for the wave codes, in caller order. Defaults to
        "s1".."sk" when the column is created.
    units : {"independent", "shared"}
        "independent" for repeated cross-sections; "shared" for panel waves
        observing the same units, which requires identical design columns
        across waves.
    on_mixed_design : {"error", "warn", "ignore"}
        What to do when waves declare structurally different designs — some
        stratified and some not, some clustered and some not (independent mode
        only). A wave without strata or without a PSU is a complete design
        (one stratum; element sampling), and combining mixed designs is valid
        under independent stacking — but a missing declaration is usually an
        oversight that silently understates variance, so the default errors.
        Passing "warn" or "ignore" is your assertion that the designs really
        differ: the combined design gets NEW String ``combined_<col>``
        variables (the ``combined_wgt`` pattern; originals untouched) holding
        the declared codes as strings, ``"__single__"`` for an unstratified
        wave's one stratum and ``"__element_<i>"`` pseudo-PSUs for an
        unclustered wave's rows — self-describing values that cannot be
        mistaken for real codes. Emitted with a warning or quietly (a log
        line) respectively. Same vocabulary as ``on_singletons`` in wrangling.
    wgt_name : str
        Name of the combined-weight column (``adjust="average"`` only).
    """
    samples = list(samples)
    k = len(samples)
    if k < 2:
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason=f"at least 2 samples are required, got {k}",
        )
    for j, s in enumerate(samples, start=1):
        if not isinstance(s, Sample):
            raise MethodError.not_applicable(
                where=_CTX,
                method="combine_samples",
                reason=f"item {j} is not a Sample (got {type(s).__name__})",
            )

    if units not in ("independent", "shared"):
        raise MethodError.invalid_choice(
            where=_CTX, param="units", got=units, allowed=["independent", "shared"]
        )
    if adjust not in (None, "average", "none"):
        raise MethodError.invalid_choice(
            where=_CTX, param="adjust", got=adjust, allowed=["average", "none", None]
        )
    if on_mixed_design not in ("error", "warn", "ignore"):
        raise MethodError.invalid_choice(
            where=_CTX,
            param="on_mixed_design",
            got=on_mixed_design,
            allowed=["error", "warn", "ignore"],
        )
    if units == "shared" and adjust == "average":
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason=(
                "adjust='average' with units='shared' divides longitudinal weights by k, "
                "but a person is not half a person for appearing in two waves"
            ),
            hint="Use adjust='none' (the shared-mode default) with longitudinal weights.",
        )
    resolved_adjust = (
        adjust if adjust is not None else ("average" if units == "independent" else "none")
    )

    for j, s in enumerate(samples, start=1):
        if s._design.rep_wgts is not None:
            raise MethodError.not_applicable(
                where=_CTX,
                method="combine_samples",
                reason=f"sample {j} carries replicate weights; combining replicate designs is not supported",
                hint="Combine the Taylor designs, then create replicate weights on the result.",
            )

    wr_values = {s._design.wr for s in samples}
    if len(wr_values) > 1:
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason="inputs disagree on with-replacement (wr) status",
        )

    canonical, mixed_flags, mixed_notes = _check_design_alignment(
        samples, units, allow_mixed=on_mixed_design != "error"
    )

    if resolved_adjust == "average" and not canonical["wgt"]:
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason="adjust='average' requires a weight declared on every input design",
            param="adjust",
        )

    frames: list[pl.DataFrame] = []
    for s in samples:
        f = s.data
        if SVY_ROW_INDEX in f.columns:
            f = f.drop(SVY_ROW_INDEX)
        frames.append(f)

    # A column name must play ONE design role across the waves: mixing roles
    # per wave (e.g. 'psu' as the PSU in one wave and a stratum component in
    # another) would give the stacked column two meanings — and collide the
    # combined_<col> names below.
    for role in mixed_flags:
        for c in canonical[role]:
            other = next((r for r, cols in canonical.items() if r != role and c in cols), None)
            if other is not None:
                raise MethodError.not_applicable(
                    where=_CTX,
                    method="combine_samples",
                    reason=(
                        f"column '{c}' plays different design roles across the inputs: "
                        f"'{other}' on some waves and '{role}' on others"
                    ),
                    hint=(
                        "One name, one role — these are different variables sharing a "
                        "name. Rename one via wrangling.rename_columns before combining."
                    ),
                )

    # Mixed designs never touch user columns: the combined encoding goes into
    # NEW `combined_<col>` variables (the combined_wgt pattern) — declaring
    # waves' values copied, lacking waves' materialized as what their design
    # already meant: element-id PSUs (unique within the wave suffices — strata
    # are wave-qualified) or a single stratum (Sample validation rejects nulls
    # in design columns, so null-fill cannot carry it). The new columns are
    # String with self-describing fills — a numeric constant like 1 would read
    # as (or collide with) a real code.
    for role, flags in mixed_flags.items():
        src_cols = canonical[role]
        new_cols = tuple(f"combined_{c}" for c in src_cols)
        for i, f in enumerate(frames):
            taken = [c for c in new_cols if c in f.columns]
            if taken:
                raise MethodError.not_applicable(
                    where=_CTX,
                    method="combine_samples",
                    reason=(
                        f"combining mixed designs writes new column(s) {list(new_cols)}, "
                        f"but sample {i + 1} already has {taken}"
                    ),
                    hint="Rename or drop the colliding column(s) before combining.",
                )
            if not flags[i]:
                frames[i] = f.with_columns(
                    pl.col(src).cast(pl.String).alias(new) for src, new in zip(src_cols, new_cols)
                )
            elif role == "psu":
                frames[i] = (
                    f.with_row_index("__svy_element_id")
                    .with_columns(
                        pl.concat_str(
                            pl.lit("__element_"), pl.col("__svy_element_id").cast(pl.String)
                        ).alias(new)
                        for new in new_cols
                    )
                    .drop("__svy_element_id")
                )
            else:
                frames[i] = f.with_columns(pl.lit("__single__").alias(new) for new in new_cols)
        canonical[role] = new_cols

    if mixed_notes:
        new_design_cols = [c for role in mixed_flags for c in canonical[role]]
        message = (
            "Waves declare different designs: "
            + "; ".join(mixed_notes)
            + f". The combined design uses new column(s) {new_design_cols}; "
            "the original columns are untouched."
        )
        if on_mixed_design == "warn":
            warnings.warn(message, UserWarning, stacklevel=2)
        else:
            log.info(message)

    if wave_labels is not None and len(wave_labels) != k:
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason=f"wave_labels has {len(wave_labels)} entries for {k} samples",
            param="wave_labels",
        )

    frames, codes, created = _resolve_wave_codes(frames, wave_name)

    if resolved_adjust == "average":
        for j, f in enumerate(frames, start=1):
            if wgt_name in f.columns:
                raise MethodError.not_applicable(
                    where=_CTX,
                    method="combine_samples",
                    reason=f"column '{wgt_name}' already exists in sample {j}",
                    param="wgt_name",
                    hint="Choose a different wgt_name.",
                )
        wgt_col = canonical["wgt"][0]
        frames = [f.with_columns((pl.col(wgt_col) / k).alias(wgt_name)) for f in frames]

    # Dtype conflicts on shared columns error before concat: silent upcasting of
    # coded variables is how category codes get corrupted.
    dtypes: dict[str, object] = {}
    conflicts: dict[str, list[str]] = {}
    for f in frames:
        for col, dt in f.schema.items():
            if col in dtypes and dtypes[col] != dt:
                conflicts.setdefault(col, [str(dtypes[col])]).append(str(dt))
            else:
                dtypes.setdefault(col, dt)
    if conflicts:
        detail = ", ".join(f"{c} ({' vs '.join(dts)})" for c, dts in sorted(conflicts.items()))
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason=f"shared column(s) have conflicting dtypes: {detail}",
            hint="Cast the columns to a common dtype before combining.",
        )

    all_cols = list(dtypes)
    partial = sorted(c for c in all_cols if any(c not in f.columns for f in frames))
    if partial:
        warnings.warn(
            f"{len(partial)} column(s) are missing from some inputs and were "
            f"null-filled: {partial[:10]}{'...' if len(partial) > 10 else ''}",
            UserWarning,
            stacklevel=2,
        )

    if units == "shared":
        unit_cols = [*canonical["stratum"], *canonical["psu"]]
        if unit_cols:
            first_units = set(frames[0].select(unit_cols).unique().iter_rows())
            for j, f in enumerate(frames[1:], start=2):
                if set(f.select(unit_cols).unique().iter_rows()) != first_units:
                    raise MethodError.not_applicable(
                        where=_CTX,
                        method="combine_samples",
                        reason=(
                            f"units='shared' requires identical design units across waves, "
                            f"but sample {j} differs from sample 1 on {unit_cols}"
                        ),
                        hint=(
                            "Shared mode is for the SAME units observed repeatedly. For "
                            "independent samples use units='independent'."
                        ),
                    )

    stacked = pl.concat(frames, how="diagonal")

    if units == "independent":
        new_stratum: tuple[str, ...] | None = (wave_name, *canonical["stratum"])
    else:
        new_stratum = canonical["stratum"] or None

    first = samples[0]._design
    design = Design(
        stratum=new_stratum,
        psu=canonical["psu"] or None,
        ssu=canonical["ssu"] or None,
        wgt=wgt_name
        if resolved_adjust == "average"
        else (canonical["wgt"][0] if canonical["wgt"] else None),
        prob=canonical["prob"][0] if canonical["prob"] else None,
        hit=canonical["hit"][0] if canonical["hit"] else None,
        mos=canonical["mos"][0] if canonical["mos"] else None,
        pop_size=first.pop_size,
        wr=first.wr,
    )

    combined = Sample(data=stacked, design=design)
    input_names = [getattr(s, "name", None) for s in samples]
    if any(input_names):
        combined.name = " + ".join(
            str(n) if n else f"s{i}" for i, n in enumerate(input_names, start=1)
        )
    _merge_metadata(combined, samples)

    combined.meta.set_type(wave_name, MeasurementType.ORDINAL)
    existing_wave_labels = combined.meta.get(wave_name)
    existing_map = existing_wave_labels.labels if existing_wave_labels is not None else {}
    if wave_labels is not None:
        label_map = dict(zip(codes, wave_labels))
        if existing_map and not created and existing_map != label_map:
            raise MethodError.not_applicable(
                where=_CTX,
                method="combine_samples",
                reason=(
                    f"wave_labels conflict with the value labels already carried by the "
                    f"reused column '{wave_name}'"
                ),
                hint="Drop wave_labels to keep the existing labels, or relabel before combining.",
            )
        combined.meta.set_value_labels(wave_name, label_map)
        _warn_if_numeric_labels_unordered(wave_labels)
    elif created:
        combined.meta.set_value_labels(
            wave_name, {c: f"s{i}" for i, c in enumerate(codes, start=1)}
        )

    return combined


def _warn_if_numeric_labels_unordered(wave_labels: Sequence[str]) -> None:
    try:
        nums = [float(x) for x in wave_labels]
    except (TypeError, ValueError):
        return
    if any(nums[i] >= nums[i + 1] for i in range(len(nums) - 1)):
        warnings.warn(
            f"wave_labels look numeric but are not increasing: {list(wave_labels)}. "
            "Caller order of `samples` is the time order — check the order of your inputs.",
            UserWarning,
            stacklevel=3,
        )
