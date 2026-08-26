# svy/core/repwgts.py
"""Replicate-weight designs as a tagged union.

One struct per variance-estimation method, each carrying exactly the parameters
its algorithm has.

The rule for adding a variant: **a type exists exactly when the data differs.**
Behaviour differences are handled by a method on the variant, not a new type.
That is why Fay's BRR is a ``fay_coef`` value rather than a type —
plain BRR *is* Fay's BRR at 0.0, so splitting it would make
``FayWgts(fay_coef=0.0)`` constructible and self-contradictory.

The names mirror ``RepMethod`` in ``svy-rs/src/estimation/replication.rs``, but
the kernel no longer branches on them: Python resolves the per-replicate
coefficients here and passes the resulting vector across the FFI. Nothing in
``svy`` reads the ``method`` label to choose a code path — copies go through
``msgspec.structs.replace``, which preserves the type, and variance goes through
``coefficients()``.

``RepWeights`` is a function, not a class. For annotations and ``isinstance``
use ``RepWgts``, the union of the four variants.
"""

from __future__ import annotations

import re

from typing import Literal, Sequence, Union, cast, get_args

import msgspec

from svy.errors import MethodError


# =============================================================================
# Bootstrap kinds
# =============================================================================
#
# The two kinds differ in how the replicates are drawn, not in what they carry
# or how the variance is scaled -- both use 1/R -- so this is a field rather
# than a nested union, and it is provenance rather than a coefficient input.
# A kind whose scale *does* differ (svrep's generalized bootstrap, tau^2/B)
# would branch in ``BootstrapWgts._default_coefficients`` and nowhere else.
# Calibrating the replicates afterwards is a separate weighting step (see
# ``Sample.weighting.poststratify``) and is not recorded here.
#: The canonical values. Aliases normalize onto these, and the field stores
#: only these, so the Literal describes what is actually held. Authored code
#: should use a canonical name and gets checked for it; a name arriving as data
#: goes through ``RepWeights``, where no static type applies anyway.
BootstrapKind = Literal["rao-wu", "poisson"]
BOOTSTRAP_KINDS: tuple[str, ...] = get_args(BootstrapKind)

_BS_KIND_ALIASES: dict[str, str] = {
    "rao-wu": "rao-wu",
    "rao_wu": "rao-wu",
    "raowu": "rao-wu",
    "rao wu": "rao-wu",
    "rw": "rao-wu",
    "rao-wu-yue": "rao-wu",
    "poisson": "poisson",
}


# =============================================================================
# Jackknife kinds
# =============================================================================
#
# Three families, and unlike the bootstrap kinds they carry different variance
# coefficients -- which is why this replaced a ``paired: bool``. The boolean
# folded JK1 and JKn together into False, but those two have different
# coefficients, so it could not express the distinction the coefficient logic
# depends on. The names are R's ``svrepdesign(type=)`` strings.
#
# "jk1"  unstratified delete-one-PSU          (R-1)/R
# "jkn"  stratified delete-one-PSU            (n_h-1)/n_h, per replicate
# "jk2"  one paired replicate per stratum     1.0
JackknifeKind = Literal["jk1", "jkn", "jk2"]
JACKKNIFE_KINDS: tuple[str, ...] = get_args(JackknifeKind)

_JK_KIND_ALIASES: dict[str, str] = {
    "jk1": "jk1",
    "jk-1": "jk1",
    "jk_1": "jk1",
    "jkn": "jkn",
    "jk-n": "jkn",
    "jk_n": "jkn",
    "jk2": "jk2",
    "jk-2": "jk2",
    "jk_2": "jk2",
    # "paired" describes the design (two PSUs per stratum), not the replication
    # scheme -- a JKn on a paired design is still JKn. Accepted because people
    # reach for it, normalized to the scheme's actual name.
    "paired": "jk2",
}


def normalize_jackknife_kind(kind: str) -> JackknifeKind:
    """Normalize a jackknife ``kind``. Case- and separator-insensitive."""
    if not isinstance(kind, str):
        raise TypeError(f"'kind' must be a string, got {type(kind).__name__}.")
    resolved = _JK_KIND_ALIASES.get(kind.strip().lower())
    if resolved is None:
        raise MethodError.invalid_choice(
            where="svy.RepWeights",
            param="kind",
            got=kind,
            allowed=list(JACKKNIFE_KINDS),
            docs_url=None,
            hint=(
                "'jk1' is the unstratified delete-one-PSU jackknife, 'jkn' its "
                "stratified form, and 'jk2' the paired one-replicate-per-stratum "
                "scheme. Leave it unset if you do not know which the weights are."
            ),
        )
    return cast(JackknifeKind, resolved)


# =============================================================================
# Shared fields and behaviour
# =============================================================================


def _fmt_coefs(values: Sequence[float]) -> str:
    """A uniform coefficient prints as the scalar it is.

    A varying one prints its distinct values with counts. Showing first and
    last instead said nothing useful: which two numbers happened to land at the
    ends is an artefact of the producer's replicate order, while "three
    replicates at 2/3 and four at 1/2" is the design.
    """
    counts = _coef_counts(values)
    if len(counts) == 1:
        return repr(next(iter(counts)))
    shown = ", ".join(f"{v!r} x{n}" for v, n in list(counts.items())[:4])
    more = "" if len(counts) <= 4 else f", ... ({len(counts)} distinct)"
    return f"{shown}{more}"


def _coef_counts(values: Sequence[float]) -> dict[float, int]:
    """Distinct coefficients and how many replicates carry each, in first-seen
    order -- which for a jackknife is stratum order."""
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def _normalize_scale(value: float | Sequence[float], n_reps: int, param: str) -> tuple[float, ...]:
    """Broadcast a scalar, or validate a sequence against ``n_reps``.

    Stored normalized so that every reader downstream sees one shape and the
    length is wrong at construction rather than at estimation.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value),) * n_reps
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(
            f"'{param}' must be a float or a sequence of floats, got {type(value).__name__}."
        )
    values = tuple(float(v) for v in value)
    if len(values) != n_reps:
        raise ValueError(
            f"'{param}' has {len(values)} entries but n_reps is {n_reps}. "
            f"Pass a scalar to use the same coefficient for every replicate."
        )
    return values


def _normalize_unit(
    value: str | Sequence[str] | None, param: str
) -> str | tuple[str, ...] | None:
    """Validate a unit reference, storing a multi-column one as a tuple.

    Mirrors ``Design``'s own normalization so that the same spellings mean the
    same thing on both objects: a bare name stays a string, a sequence becomes a
    tuple, and a single-element sequence stays a one-tuple rather than being
    unwrapped -- ``("region",)`` and ``"region"`` resolve to the same column, and
    collapsing one into the other would make a round trip lossy.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"RepWeights {param!r} must not be an empty column name.")
        return value
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(
            f"RepWeights {param!r} must be str | Sequence[str] | None, "
            f"got {type(value).__name__}."
        )
    items = tuple(value)
    if not items:
        raise ValueError(f"RepWeights {param!r} sequence must not be empty.")
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise TypeError(
                f"RepWeights {param!r} items must be str; "
                f"got {type(item).__name__} at index {i}."
            )
        if not item.strip():
            raise ValueError(f"RepWeights {param!r} items must not be empty.")
    return items


def unit_columns(value: str | tuple[str, ...] | None) -> list[str]:
    """The column names a unit reference resolves to; empty when unset.

    Multi-column units are grouped on directly rather than through the internal
    concatenated column ``Design`` builds: polars groups by several keys just as
    well, and the concat name is an implementation detail that would leak into
    anything reading provenance back.
    """
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


class _RepWgtsBase(msgspec.Struct, frozen=True, kw_only=True):
    """Fields every replicate design has, whatever produced it.

    ``kw_only`` is required: without it msgspec orders base fields first and a
    variant could never carry a required field of its own.
    """

    prefix: str
    n_reps: int
    # Design df for the t-quantile in CIs. None = n_reps - 1, a property of the
    # weight set and therefore the same for every domain.
    #
    # Deliberately float, not int: it feeds a t-quantile, which is defined for
    # fractional df, and Satterthwaite-style effective df is fractional by
    # construction. The kernels also hand it back as f64. An int is accepted and
    # stored unchanged -- widening the annotation is what makes the stored value
    # honest, rather than coercing a legitimate fraction away.
    df: float | None = None
    padding: int | None = None  # None = auto-detect, 0 = none, >0 = zero-pad width
    # Per-replicate variance coefficients, split by who supplied them -- the one
    # axis that is verifiable. "Is this a tweak or the standard value?" is not:
    # a user declaring stratified JKn supplies the *standard* coefficient,
    # because svy cannot derive it from replicate weights alone.
    #
    # scale: the user said so. R's svrepdesign takes scale/rscales for every
    # type=, and svy folds the pair into this one field (svy's scale is R's
    # scale * rscales). A scalar is broadcast to every replicate. Replaces,
    # rather than multiplies, the method default.
    scale: float | Sequence[float] | None = None
    # rep_coefs: svy computed these and cannot recompute them later. Exists for
    # one case -- JKn's (n_h-1)/n_h is the only standard coefficient that is not
    # closed-form in n_reps: it needs per-stratum PSU counts, coefficients() has
    # no frame, and the stratum column may be gone by estimation time. Filled by
    # create_jk_wgts; users want ``scale``.
    rep_coefs: tuple[float, ...] | None = None
    # The units the replicates were built from -- provenance, and for JKn the
    # only thing (n_h-1)/n_h can be counted from.
    #
    # Deliberately NOT the Design's stratum/psu. Those describe the analysis
    # design and are what Taylor linearizes over; these describe how the
    # replicates were drawn. They coincide often enough to hide the difference,
    # and public files are exactly where they do not: producers collapse strata
    # and suppress PSUs for disclosure, publishing a separate pair (VARSTRAT /
    # VARUNIT and its many spellings) alongside -- or instead of -- the design
    # variables. Borrowing the Design's would count the wrong n_h and return a
    # plausible wrong coefficient, so a declared JKn must name these and there
    # is no fallback. Generated weights fill them in from whatever they used.
    #
    # Same shape as Design's: ``str`` for the single collapsed identifier
    # producers usually ship, or a tuple when the unit is several columns
    # together. A str-only field could not express a tuple-stratum design at
    # all -- it recorded None, so those weights could be generated but never
    # re-declared by hand, and a JKn among them fell through to the branch that
    # has no stratum to count.
    stratum: str | tuple[str, ...] | None = None
    psu: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.prefix or not self.prefix.strip():
            raise ValueError("RepWeights 'prefix' cannot be empty or whitespace.")
        if self.n_reps < 2:
            raise ValueError(f"n_reps must be >= 2. Got {self.n_reps}.")
        for _unit in ("stratum", "psu"):
            msgspec.structs.force_setattr(
                self, _unit, _normalize_unit(getattr(self, _unit), _unit)
            )
        if self.df is not None and self.df <= 0:
            raise ValueError(f"df must be > 0. Got {self.df}.")
        if self.padding is not None and self.padding < 0:
            raise ValueError(f"padding must be >= 0. Got {self.padding}.")
        if self.scale is not None:
            msgspec.structs.force_setattr(
                self, "scale", _normalize_scale(self.scale, self.n_reps, "scale")
            )
        if self.rep_coefs is not None:
            msgspec.structs.force_setattr(
                self, "rep_coefs", _normalize_scale(self.rep_coefs, self.n_reps, "rep_coefs")
            )

    # ---- back-compat read surface ------------------------------------------

    @property
    def method(self) -> str:
        """The coarse method family, as a display label.

        One of ``"Bootstrap"``, ``"Jackknife"``, ``"BRR"``, ``"SDR"``. For
        display and reporting only -- nothing in ``svy`` reads this to choose a
        code path. Internal code that needs to copy a design uses
        ``msgspec.structs.replace``, which preserves the type; code that needs
        the coefficients calls ``coefficients()``. Both used to go through this
        label and back via the string factory, which is how a bootstrap and a
        jackknife could end up scaled by different rules.
        """
        return self.__struct_config__.tag

    @property
    def fay_coef(self) -> float:
        """0.0 for every method that has no Fay coefficient."""
        return 0.0

    # ---- column naming ------------------------------------------------------

    def _detect_padding(self, data_columns: Sequence[str]) -> int:
        pattern = re.compile(rf"^{re.escape(self.prefix)}(\d+)$", re.IGNORECASE)
        max_padding = 0
        for col in data_columns:
            match = pattern.match(col)
            if match:
                num_str = match.group(1)
                if len(num_str) > 1 and num_str[0] == "0":
                    max_padding = max(max_padding, len(num_str))
        return max_padding

    def _generate_columns(self, padding: int) -> list[str]:
        if padding > 0:
            return [f"{self.prefix}{i:0{padding}d}" for i in range(1, self.n_reps + 1)]
        return [f"{self.prefix}{i}" for i in range(1, self.n_reps + 1)]

    def columns_from_data(self, data_columns: Sequence[str]) -> list[str]:
        """Generate column names, auto-detecting padding and casing from data."""
        padding = self.padding if self.padding is not None else self._detect_padding(data_columns)
        pattern = re.compile(rf"^{re.escape(self.prefix)}\d+$", re.IGNORECASE)
        resolved_prefix = self.prefix
        for col in data_columns:
            if pattern.match(col):
                resolved_prefix = col[: len(self.prefix)]
                break
        if padding > 0:
            return [f"{resolved_prefix}{i:0{padding}d}" for i in range(1, self.n_reps + 1)]
        return [f"{resolved_prefix}{i}" for i in range(1, self.n_reps + 1)]

    @property
    def columns(self) -> list[str]:
        """Expected column names using explicit padding, else none.

        For validation against actual data use ``columns_from_data`` instead.
        """
        return self._generate_columns(self.padding if self.padding is not None else 0)

    # ---- variance ------------------------------------------------------------

    def coefficients(self) -> list[float]:
        """Per-replicate variance coefficients, in precedence order.

        The override is resolved *here*, at the single point of use, and the
        variants never see it -- they implement only their own default. Before
        the tagged union this lived at one site in the kernel
        (``rscales.unwrap_or_else(|| replicate_coefficients(...))``); scattering
        it into per-variant methods gave every method its own chance to forget,
        and three of four did. A new variant cannot regress it.
        """
        if self.scale is not None:
            return list(self.scale)  # the user asserted these
        if self.rep_coefs is not None:
            return list(self.rep_coefs)  # svy computed them and cannot redo it
        return self._default_coefficients()

    @property
    def coef_source(self) -> str:
        """Where the applied coefficients came from: ``"scale"`` (the user
        asserted them), ``"derived"`` (svy computed them and cannot redo it) or
        ``"default"`` (the method's standard value, closed-form in n_reps)."""
        if self.scale is not None:
            return "scale"
        if self.rep_coefs is not None:
            return "derived"
        return "default"

    def _default_coefficients(self) -> list[float]:
        """The method's standard coefficients, closed-form in ``n_reps``.

        The only thing a variant implements. Adding a method, or a bootstrap
        kind whose scale differs, stays a local change.
        """
        raise NotImplementedError

    # ---- display -------------------------------------------------------------

    def _variant_parts(self) -> list[str]:
        """Variant-specific fragments for repr. Overridden where there are any."""
        return []

    def _unit_parts(self) -> list[str]:
        """The units the replicates were built from, when they are recorded.

        Shown for the same reason ``scale`` is: which columns produced a set of
        replicates is what a reviewer needs to tell a design-strata JKn from a
        collapsed-variance-strata one, and the two give different coefficients.
        """
        return [
            f"{n}={v!r}" for n, v in (("stratum", self.stratum), ("psu", self.psu)) if v
        ]

    def _coef_parts(self) -> list[str]:
        """Where the coefficients came from, when it is not the method default.

        A non-standard variance coefficient is exactly what a reviewer or a
        replicator needs to see, and it is set rarely enough that showing it
        costs nothing.
        """
        if self.scale is not None:
            return [f"scale={_fmt_coefs(self.scale)}"]
        if self.rep_coefs is not None:
            return [f"rep_coefs={_fmt_coefs(self.rep_coefs)} (derived)"]
        return []

    def __repr__(self) -> str:
        parts = [f"method={self.method}", f"prefix='{self.prefix}'", f"n_reps={self.n_reps}"]
        if self.df is not None:
            parts.append(f"df={self.df}")
        parts.extend(self._variant_parts())
        parts.extend(self._unit_parts())
        parts.extend(self._coef_parts())
        if self.padding is not None:
            parts.append(f"padding={self.padding}")
        return f"RepWeights({', '.join(parts)})"

    def __plain_str__(self) -> str:
        method_name = self.method
        lines = [
            method_name,
            f"Method   : {method_name}",
            f"Prefix   : {self.prefix}",
            f"N reps   : {self.n_reps}",
            f"DF       : {self.df if self.df is not None else 'auto'}",
        ]
        lines.extend(self._plain_variant_lines())
        if self.stratum is not None:
            lines.append(f"Stratum  : {self.stratum}")
        if self.psu is not None:
            lines.append(f"PSU      : {self.psu}")
        if self.scale is not None:
            lines.append(f"Scale    : {_fmt_coefs(self.scale)}")
        elif self.rep_coefs is not None:
            lines.append(f"Coefs    : {_fmt_coefs(self.rep_coefs)} (derived)")
        return "\n".join(lines)

    def _plain_variant_lines(self) -> list[str]:
        return []


# =============================================================================
# The four dispatch units, mirroring Rust's RepMethod
# =============================================================================


class BootstrapWgts(_RepWgtsBase, frozen=True, kw_only=True, tag="Bootstrap", tag_field="method"):
    kind: BootstrapKind = "rao-wu"

    def __post_init__(self) -> None:
        super().__post_init__()
        # An explicit None means the same as omitting it: this field's default
        # is a value, not an absence, because both kinds share the 1/R
        # coefficient and rao-wu is the dominant convention.
        resolved = "rao-wu" if self.kind is None else normalize_bootstrap_kind(self.kind)
        msgspec.structs.force_setattr(self, "kind", resolved)

    def _default_coefficients(self) -> list[float]:
        # Both kinds: the kind decides how the replicates are drawn, not how the
        # variance is scaled. A kind whose scale differs (svrep's generalized
        # bootstrap, tau^2/B) branches here.
        return [1.0 / self.n_reps] * self.n_reps

    def _variant_parts(self) -> list[str]:
        return [f"kind={self.kind}"]

    def _plain_variant_lines(self) -> list[str]:
        return [f"Kind     : {self.kind}"]


class JackknifeWgts(_RepWgtsBase, frozen=True, kw_only=True, tag="Jackknife", tag_field="method"):
    # None = unspecified: nobody has said which family these weights are. That
    # is a different statement from "jk1", even though the two produce the same
    # number -- svy only claims what it knows or what it was told.
    kind: JackknifeKind | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not None:
            msgspec.structs.force_setattr(self, "kind", normalize_jackknife_kind(self.kind))

    def _default_coefficients(self) -> list[float]:
        if self.kind == "jk2":
            # One delete-one replicate per stratum. A global (R-1)/R would
            # understate the variance by exactly that factor.
            return [1.0] * self.n_reps
        if self.kind == "jkn":
            # (n_h-1)/n_h is the only standard coefficient that is not
            # closed-form in n_reps, and nothing supplied it. An unmet claim
            # fails; an absent one (kind=None) falls back to the JK1 global.
            raise MethodError.not_applicable(
                where="RepWeights.coefficients",
                method="jackknife",
                reason=(
                    "kind='jkn' needs the per-stratum (n_h-1)/n_h coefficients, "
                    "which cannot be derived from replicate weights alone. Falling "
                    "back to the JK1 global (R-1)/R would overstate the standard "
                    "errors"
                ),
                # Two fixes, and `scale` used to be the only one named -- which
                # sent anyone whose file *does* carry psu off to hand-compute
                # coefficients svy would have worked out for them.
                hint=(
                    "Either name the units these replicates were built from -- "
                    "JackknifeWgts(..., stratum='VARSTRAT', psu='VARUNIT') -- and svy "
                    "derives (n_h-1)/n_h at Sample construction when every stratum has "
                    "the same number of PSUs; or pass scale= with the per-replicate "
                    "coefficients your file documents, which is what unbalanced strata "
                    "need. The Design's stratum/psu are deliberately not used: they "
                    "describe the analysis design, not how the replicates were drawn."
                ),
            )
        # kind=None (unspecified) and "jk1" alike: the unstratified global.
        return [(self.n_reps - 1) / self.n_reps] * self.n_reps

    def _variant_parts(self) -> list[str]:
        return [] if self.kind is None else [f"kind={self.kind}"]

    def _plain_variant_lines(self) -> list[str]:
        return [] if self.kind is None else [f"Kind     : {self.kind}"]


class BrrWgts(_RepWgtsBase, frozen=True, kw_only=True, tag="BRR", tag_field="method"):
    # Fay's BRR is this at a non-zero coefficient, not a separate type: the
    # scale 1/(B(1-f)^2) collapses to 1/B at f=0.
    fay_coef: float = 0.0  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fay_coef < 0:
            raise ValueError(f"fay_coef cannot be negative. Got {self.fay_coef}.")

    def _default_coefficients(self) -> list[float]:
        coef = 1.0 / (self.n_reps * (1.0 - self.fay_coef) ** 2)
        return [coef] * self.n_reps

    def _variant_parts(self) -> list[str]:
        return [f"fay={self.fay_coef}"]

    def _plain_variant_lines(self) -> list[str]:
        return [f"Fay coef : {self.fay_coef}"]


class SdrWgts(_RepWgtsBase, frozen=True, kw_only=True, tag="SDR", tag_field="method"):
    def _default_coefficients(self) -> list[float]:
        return [4.0 / self.n_reps] * self.n_reps


RepWgts = Union[BootstrapWgts, JackknifeWgts, BrrWgts, SdrWgts]

# =============================================================================
# Normalization + the RepWeights factory
# =============================================================================

# A method name resolves straight to its variant. There is no enum in the
# middle. "taylor" is absent, and therefore rejected: linearization carries no
# replicate weights, so there is no variant to resolve it to.
_REP_METHOD_ALIASES: dict[str, type] = {
    "brr": BrrWgts,
    "bootstrap": BootstrapWgts,
    "bs": BootstrapWgts,
    "jackknife": JackknifeWgts,
    "jk": JackknifeWgts,
    "jkn": JackknifeWgts,
    "sdr": SdrWgts,
}

_REPLICATE_METHOD_NAMES = ("Bootstrap", "Jackknife", "BRR", "SDR")

# Recognised estimation methods that carry no replicate weights, so that
# `method="taylor"` earns a useful message rather than being read as a typo.
_NON_REPLICATE_METHODS = frozenset({"taylor"})


def resolve_rep_variant(
    method: Literal["brr", "bootstrap", "jackknife", "sdr"] | str,
) -> type:
    """Resolve a method name to its RepWeights variant.

    Accepts (case-insensitive): "brr", "bootstrap"/"bs",
    "jackknife"/"jk"/"jkn", "sdr".
    """
    if not isinstance(method, str):
        raise TypeError(
            f"'method' must be a string, got {type(method).__name__}. "
            f"Use 'brr', 'bootstrap', 'jackknife', or 'sdr'."
        )
    normalized = method.strip().lower()
    variant = _REP_METHOD_ALIASES.get(normalized)
    if variant is not None:
        return variant

    # A real estimation method that simply has no replicate weights (Taylor) is
    # a different mistake from a typo, and gets a different message. Compared
    # case-insensitively so the string and the enum member behave alike.
    if normalized in _NON_REPLICATE_METHODS:
        raise ValueError(
            f"Method '{method}' is not a valid replication method "
            f"(expected one of: {list(_REPLICATE_METHOD_NAMES)}). "
            f"Taylor linearization carries no replicate weights: leave "
            f"Design.rep_wgts as None and pass method='taylor' at estimation time."
        )
    raise ValueError(
        f"Unknown replication method {method!r}. Use 'brr', 'bootstrap', 'jackknife', or 'sdr'."
    )


def normalize_bootstrap_kind(kind: str) -> BootstrapKind:
    """Normalize a bootstrap ``kind``.

    Case-insensitive and tolerant of hyphen, underscore or space separators,
    matching :func:`resolve_rep_variant`.
    """
    if not isinstance(kind, str):
        raise TypeError(f"'kind' must be a string, got {type(kind).__name__}.")
    resolved = _BS_KIND_ALIASES.get(kind.strip().lower())
    if resolved is None:
        raise MethodError.invalid_choice(
            where="svy.RepWeights",
            param="kind",
            got=kind,
            allowed=list(BOOTSTRAP_KINDS),
            docs_url=None,
            hint=(
                "'rao-wu' is the stratified Rao-Wu-Yue rescaling bootstrap and needs "
                "psu on the design; 'poisson' is the Beaumont-Patak generalized "
                "bootstrap and needs only a weight."
            ),
        )
    return cast(BootstrapKind, resolved)


_MISSING_ARG: object = object()

# Parameters that only some variants carry, for turning msgspec's accurate but
# terse "Unexpected keyword argument" into one that names the owner.
_PARAM_OWNER = {
    "fay_coef": ("BrrWgts", "BRR"),
    # kind belongs to bootstrap and to jackknife, with different vocabularies;
    # only BRR and SDR have none.
    "kind": ("BootstrapWgts or JackknifeWgts", "bootstrap or jackknife"),
}


def RepWeights(  # noqa: N802 - a factory that replaced a class of this name
    method: str = _MISSING_ARG,  # type: ignore[assignment]
    prefix: str = _MISSING_ARG,  # type: ignore[assignment]
    n_reps: int = _MISSING_ARG,  # type: ignore[assignment]
    **kwargs: object,
) -> RepWgts:
    """Build the replicate-weight variant for ``method``.

    The door for a method name that arrives as a string -- from a codebook, a
    config file, or a producer's documentation. New code that knows the method
    at authoring time should construct the variant directly, which is typed and
    autocompletes:

    >>> BootstrapWgts(prefix="bsw", n_reps=1000, kind="poisson")
    >>> BrrWgts(prefix="brr_", n_reps=32, fay_coef=0.5)

    This is a function, not a class: ``isinstance(x, svy.RepWeights)`` and the
    annotation ``x: svy.RepWeights`` do not work. Use ``svy.RepWgts``, the union
    of the four variants, for both.

    Parameters that do not belong to ``method`` are refused rather than stored --
    by the variant itself, since a struct has no field to put them in. This
    wrapper only improves the message.
    """
    # Sentinels rather than bare required parameters so the message matches the
    # struct this factory replaced. Only these three need it: everything else
    # forwards to msgspec, whose own message is already verbatim
    # "Missing required argument 'prefix'".
    for _name, _val in (("method", method), ("prefix", prefix), ("n_reps", n_reps)):
        if _val is _MISSING_ARG:
            raise TypeError(f"Missing required argument {_name!r}")

    variant = resolve_rep_variant(method)
    try:
        return cast(RepWgts, variant(prefix=prefix, n_reps=n_reps, **kwargs))
    except TypeError as exc:
        name = _unexpected_param(exc)
        if name is None:
            raise
        raise _foreign_param_error(variant, name, kwargs.get(name)) from exc


def _unexpected_param(exc: TypeError) -> str | None:
    """The parameter name msgspec refused, or None if this is another TypeError."""
    match = re.search(r"Unexpected keyword argument '([^']+)'", str(exc))
    return match.group(1) if match else None


def _foreign_param_error(variant: type, name: str, value: object = None) -> Exception:
    """Name the variant that owns a parameter msgspec has just refused.

    msgspec already rejects a foreign keyword on every path, including direct
    construction -- which the hand-rolled guard this replaced never covered. All
    that is added here is the hint.
    """
    owned = _PARAM_OWNER.get(name)
    if owned is None:
        return TypeError(f"Unexpected keyword argument {name!r}")
    owner, owner_method = owned
    return MethodError.invalid_choice(
        where="svy.RepWeights",
        param=name,
        got=value,
        allowed=[None],
        hint=(
            f"'{name}' is a {owner_method} parameter and is not stored on a "
            f"{variant.__name__} design. Each method carries only its own "
            f"parameters: {owner} has '{name}'."
        ),
    )
