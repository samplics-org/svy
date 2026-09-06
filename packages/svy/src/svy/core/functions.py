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
from svy.core.panel import design_varies_within_case, duplicate_case_ids, wave_overlap
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
    samples: Sequence[Sample], kind: str, allow_mixed: bool
) -> tuple[dict[str, tuple[str, ...]], dict[str, list[bool]], list[str]]:
    """Require identical design columns, role by role; no implicit renaming.

    Column names carry meaning in the output, so a name mismatch is the user's
    to resolve — the hint spells out the one-line fix, which updates data,
    design and metadata together.

    Opt-in exception, cross-sectional mode only: waves may declare structurally
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
    if kind == "cross_sectional" and allow_mixed:
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
                    if role in _MIXABLE_ROLES and kind == "cross_sectional"
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


def _resolve_case_id(samples: Sequence[Sample], kind: str, case_id: str | None) -> str | None:
    """The case column: the explicit one, else the one every input declares."""
    declared = {s._design.case_id for s in samples}
    shared = next(iter(declared)) if len(declared) == 1 else None
    if case_id is None:
        case_id = shared
    if kind == "panel" and case_id is None:
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason="kind='panel' needs the column identifying the followed case",
            param="case_id",
            hint=(
                "Pass case_id= or declare Design(case_id=...) on each wave. If the "
                "waves name it differently, rename upfront with "
                "sample.wrangling.rename_columns on one wave."
            ),
        )
    if kind != "panel" and case_id != shared:
        # A cross-sectional stack keeps an id only when every input carries
        # that very column; an explicit name is not enough to make it one.
        return None
    return case_id


def _resolve_rep_wgts(samples: Sequence[Sample], kind: str):
    """Replicate weights carried by the inputs: rejected, except producer
    longitudinal replicates identical on every wave of a panel."""
    reps = [s._design.rep_wgts for s in samples]
    if all(r is None for r in reps):
        return None
    if kind == "panel" and all(r is not None and r == reps[0] for r in reps):
        return reps[0]
    j = next(i for i, r in enumerate(reps, start=1) if r is not None)
    reason = (
        f"sample {j} carries replicate weights; combining replicate designs is not supported"
        if kind != "panel"
        else "the waves carry different replicate-weight designs"
    )
    raise MethodError.not_applicable(
        where=_CTX,
        method="combine_samples",
        reason=reason,
        hint=(
            "Combine the Taylor designs, then create replicate weights on the result. "
            "A panel accepts producer replicates only when every wave declares the "
            "same RepWgts and the columns are constant within a case."
        ),
    )


def _check_panel_ids(frames: Sequence[pl.DataFrame], case_id: str) -> None:
    missing = [j for j, f in enumerate(frames, start=1) if case_id not in f.columns]
    if missing:
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason=f"case_id column '{case_id}' is missing from sample(s) {missing}",
            param="case_id",
            hint="Every wave must carry the case id under this one name.",
        )
    for j, f in enumerate(frames, start=1):
        if f.get_column(case_id).null_count() > 0:
            raise MethodError.not_applicable(
                where=_CTX,
                method="combine_samples",
                reason=f"case_id column '{case_id}' has nulls in sample {j}",
                param="case_id",
            )
        dups = duplicate_case_ids(f, case_id, None)
        if dups:
            raise MethodError.not_applicable(
                where=_CTX,
                method="combine_samples",
                reason=f"case_id column '{case_id}' is not unique in sample {j}: {dups}",
                param="case_id",
                hint=(
                    "A case is one row per wave. If these are household rows, the "
                    "person-level file needs a person id."
                ),
            )


def _check_panel_units(
    frames: Sequence[pl.DataFrame], canonical: dict[str, tuple[str, ...]]
) -> None:
    """Later waves' (stratum, psu) set must be a subset of wave 1's; a PSU
    entirely lost is a real panel event and warns."""
    unit_cols = [*canonical["stratum"], *canonical["psu"]]
    if not unit_cols:
        return
    first_set = set(frames[0].select(unit_cols).unique().iter_rows())
    for j, f in enumerate(frames[1:], start=2):
        units = set(f.select(unit_cols).unique().iter_rows())
        extra = sorted(map(str, units - first_set))
        if extra:
            raise MethodError.not_applicable(
                where=_CTX,
                method="combine_samples",
                reason=(
                    f"sample {j} has design units {unit_cols} absent from sample 1: {extra[:10]}"
                ),
                hint=(
                    "A panel keeps the base-wave design: every case carries its wave-1 "
                    "stratum and PSU on later waves. Movers keep their base-wave PSU."
                ),
            )
        lost = sorted(map(str, first_set - units))
        if lost:
            warnings.warn(
                f"{len(lost)} design unit(s) {unit_cols} of sample 1 have no row in "
                f"sample {j}: {lost[:10]}{'...' if len(lost) > 10 else ''}",
                UserWarning,
                stacklevel=3,
            )


def _check_constant_within_case(
    stacked: pl.DataFrame, case_id: str, cols: Sequence[str], *, what: str = "design columns"
) -> None:
    varies = design_varies_within_case(stacked, case_id, cols)
    if varies:
        detail = "; ".join(f"{c}: {ids}" for c, ids in varies.items())
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason=f"{what} vary within case_id '{case_id}' across waves: {detail}",
            hint=(
                "The case is nested in its PSU: movers keep their base-wave stratum "
                "and PSU (and the base-wave replicate columns)."
            ),
        )


def _report_overlap(stacked: pl.DataFrame, case_id: str, wave_name: str) -> None:
    for ov in wave_overlap(stacked, case_id, wave_name):
        log.info("panel overlap %s", ov)
        if ov.common == 0:
            raise MethodError.not_applicable(
                where=_CTX,
                method="combine_samples",
                reason=(
                    f"no case of wave {ov.prev!r} appears in wave {ov.wave!r} "
                    f"({ov.lost} lost, {ov.new} new)"
                ),
                hint=(
                    "These look like two unrelated cross-sections. Check that case_id "
                    "names the same identifier on every wave, or stack with "
                    "kind='cross_sectional'."
                ),
            )
        n_prev = ov.common + ov.lost
        if ov.common < 0.5 * n_prev:
            warnings.warn(
                f"Small panel overlap between wave {ov.prev!r} and wave {ov.wave!r}: "
                f"{ov.common} of {n_prev} cases continue ({ov.lost} lost, {ov.new} new).",
                UserWarning,
                stacklevel=3,
            )


def combine_samples(
    samples: Sequence[Sample],
    *,
    adjust: Literal["average", "none"] | None = None,
    wave_name: str = "wave",
    wave_labels: Sequence[str] | None = None,
    kind: Literal["cross_sectional", "cs", "panel"] = "cross_sectional",
    case_id: str | None = None,
    on_mixed_design: Literal["error", "warn", "ignore"] = "error",
    wgt_name: str = "combined_wgt",
) -> Sample:
    """Combine repeated cross-sections or panel waves into one long Sample.

    Stacks the data files and analyzes them as ONE design — this is data
    pooling, not estimate pooling. With ``kind="cross_sectional"`` each wave
    contributes its own strata (the design nests wave → stratum → PSU), so
    Taylor variance treats waves as independent automatically. The estimand
    under ``adjust="average"`` is the PERIOD-AVERAGE population: weights are
    divided by k, which matters only for totals — means, proportions and
    ratios are invariant to it.

    With ``kind="panel"`` the same cases are observed repeatedly: the stacked
    sample keeps the base-wave strata and PSUs, ``Design.case_id`` identifies
    the followed entity and ``Design.wave`` orders its rows. When no PSU is
    declared the case is the variance PSU, so ``by=wave_name`` estimates and
    their contrasts carry the between-wave covariance without being told to.

    Caller order of ``samples`` IS the time order; the wave column gets ordinal
    codes 1..k in that order (or reuses an existing wave column present in all
    inputs, e.g. NHANES SDDSRVYR, validating it increases in caller order).

    For bespoke factors (e.g. the NCHS 1999–2004 recipe of 4/6 and 2/6 on the
    4-yr and 2-yr weight files), pre-adjust each input with
    ``sample.weighting.normalize(factor=...)`` and pass ``adjust="none"``.

    Never combine for trend questions — use ``by=wave_name`` on the combined
    sample instead. ``design_history`` of the inputs is not carried: the
    combined Sample is a new object built from k parents.

    Parameters
    ----------
    samples : Sequence[Sample]
        Two or more samples, in time order.
    adjust : {"average", "none"} | None
        "average" multiplies every weight by 1/k into ``wgt_name``. None picks
        the mode default: "average" for cross-sections, "none" for a panel.
        Explicit "average" with ``kind="panel"`` errors — a person is not
        half a person for appearing in two waves.
    wave_name : str
        Wave-id column name; reused if already present in all inputs. Fills
        ``Design.wave`` on the result.
    wave_labels : Sequence[str] | None
        Value labels for the wave codes, in caller order. Defaults to
        "wave 1".."wave k" when the column is created.
    kind : {"cross_sectional", "cs", "panel"}
        "cross_sectional" (alias "cs") for repeated cross-sections;
        "panel" for waves observing the same cases, which requires
        ``case_id`` and validates the pairing: the id is unique within each
        wave, consecutive waves overlap (an empty overlap errors, a small one
        warns), design columns are constant within a case, and a later
        wave's (stratum, PSU) set is a subset of wave 1's.
    case_id : str | None
        Column identifying the followed case, present in every input under
        this one name (rename upfront if the waves differ). Required for
        ``kind="panel"`` unless every input declares the same
        ``Design.case_id``. On a cross-sectional stack it is kept only when
        every input declares it and it stays unique on the stacked frame.
    on_mixed_design : {"error", "warn", "ignore"}
        What to do when waves declare structurally different designs — some
        stratified and some not, some clustered and some not (cross-sectional
        mode only). A wave without strata or without a PSU is a complete design
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

    if kind not in ("cross_sectional", "cs", "panel"):
        raise MethodError.invalid_choice(
            where=_CTX, param="kind", got=kind, allowed=["cross_sectional", "cs", "panel"]
        )
    if kind == "cs":
        kind = "cross_sectional"
    if case_id is not None and (not isinstance(case_id, str) or not case_id):
        raise MethodError.invalid_choice(
            where=_CTX, param="case_id", got=case_id, allowed=["<column name>"]
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
    if kind == "panel" and adjust == "average":
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason=(
                "adjust='average' with kind='panel' divides longitudinal weights by k, "
                "but a person is not half a person for appearing in two waves"
            ),
            hint="Use adjust='none' (the panel default) with longitudinal weights.",
        )
    resolved_adjust = (
        adjust if adjust is not None else ("average" if kind == "cross_sectional" else "none")
    )

    case_id = _resolve_case_id(samples, kind, case_id)
    rep_wgts = _resolve_rep_wgts(samples, kind)

    wr_values = {s._design.wr for s in samples}
    if len(wr_values) > 1:
        raise MethodError.not_applicable(
            where=_CTX,
            method="combine_samples",
            reason="inputs disagree on with-replacement (wr) status",
        )

    canonical, mixed_flags, mixed_notes = _check_design_alignment(
        samples, kind, allow_mixed=on_mixed_design != "error"
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

    if kind == "panel":
        assert case_id is not None  # noqa: S101 — _resolve_case_id guarantees it
        _check_panel_ids(frames, case_id)
        _check_panel_units(frames, canonical)

    stacked = pl.concat(frames, how="diagonal")

    if kind == "panel":
        assert case_id is not None  # noqa: S101
        design_cols = [
            *canonical["stratum"],
            *canonical["psu"],
            *canonical["ssu"],
            *canonical["pop_size"],
        ]
        _check_constant_within_case(stacked, case_id, design_cols)
        if rep_wgts is not None:
            _check_constant_within_case(
                stacked, case_id, rep_wgts.columns, what="replicate weight columns"
            )
        _report_overlap(stacked, case_id, wave_name)
        new_stratum: tuple[str, ...] | None = canonical["stratum"] or None
    else:
        new_stratum = (wave_name, *canonical["stratum"])
        if case_id is not None and duplicate_case_ids(stacked, case_id, None):
            log.info(
                "case_id %r is not unique on the stacked cross-sections; not kept on the design",
                case_id,
            )
            case_id = None

    first = samples[0]._design
    design = Design(
        case_id=case_id,
        wave=wave_name,
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
        rep_wgts=rep_wgts,
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
            wave_name, {c: f"wave {i}" for i, c in enumerate(codes, start=1)}
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
