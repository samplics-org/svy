# svy/core/repwgts.py
"""Replicate-weight designs as a tagged union.

One struct per variance-estimation method, each carrying exactly the parameters
its algorithm has. See ``docs/design/rep-weights-tagged-union.md``.

The rule for adding a variant: **a type exists exactly when the data differs.**
Behaviour differences are handled by a method on the variant, not a new type.
That is why Rao-Wu and Poisson are separate kinds (Poisson carries
``calib_domains``) while Fay's BRR is a ``fay_coef`` value rather than a type —
plain BRR *is* Fay's BRR at 0.0, so splitting it would make
``FayWgts(fay_coef=0.0)`` constructible and self-contradictory.

The top level mirrors ``RepMethod`` in ``svy-rs/src/estimation/replication.rs``,
which is what the variance kernel actually branches on.
"""

from __future__ import annotations

import re

from typing import Literal, Sequence, Union

import msgspec

from svy.errors import MethodError


# =============================================================================
# Bootstrap kinds
# =============================================================================
#
# A nested union rather than a plain string, because the kinds carry different
# data. Behaviour that varies by kind (the replicate coefficient) lives here as
# a method, so adding a kind whose scale differs -- svrep's generalized
# bootstrap uses tau^2/B rather than 1/B -- needs no change anywhere else.


class RaoWu(msgspec.Struct, frozen=True, tag="rao-wu", tag_field="kind"):
    """Stratified Rao-Wu-Yue rescaling bootstrap. Resamples PSUs within strata."""

    def coefficients(self, n_reps: int) -> list[float]:
        return [1.0 / n_reps] * n_reps

    def __str__(self) -> str:
        return "rao-wu"


class Poisson(msgspec.Struct, frozen=True, tag="poisson", tag_field="kind"):
    """Beaumont-Patak generalized bootstrap with independent per-unit factors.

    Requires only a weight, which is why producers document it for public use
    files where stratum and PSU identifiers are suppressed. The draws being
    independent across units is also what makes the weights non-disclosive.

    ``calib_domains`` records the columns each replicate was post-stratified
    back to the base weight's own total within. It is stored because a consumer
    needs it: post-stratifying replicates that are already calibrated on the
    same margins is a double-calibration bug.
    """

    calib_domains: tuple[str, ...] | None = None

    def coefficients(self, n_reps: int) -> list[float]:
        return [1.0 / n_reps] * n_reps

    def __str__(self) -> str:
        if self.calib_domains:
            return f"poisson (calibrated on {', '.join(self.calib_domains)})"
        return "poisson"


BootstrapKind = Union[RaoWu, Poisson]

_BS_KIND_ALIASES: dict[str, type[RaoWu] | type[Poisson]] = {
    "rao-wu": RaoWu,
    "rao_wu": RaoWu,
    "raowu": RaoWu,
    "rao wu": RaoWu,
    "rw": RaoWu,
    "rao-wu-yue": RaoWu,
    "poisson": Poisson,
}


# =============================================================================
# Shared fields and behaviour
# =============================================================================


class _RepWgtsBase(msgspec.Struct, frozen=True, kw_only=True):
    """Fields every replicate design has, whatever produced it.

    ``kw_only`` is required: without it msgspec orders base fields first and a
    variant could never carry a required field of its own.
    """

    prefix: str
    n_reps: int
    # Design df for the t-quantile in CIs. None = n_reps - 1, a property of the
    # weight set and therefore the same for every domain.
    df: int | None = None
    padding: int | None = None  # None = auto-detect, 0 = none, >0 = zero-pad width
    # Per-replicate variance coefficients (R's scale*rscales combined). Kept
    # common rather than on JackknifeWgts: R's svrepdesign takes scale/rscales
    # for every type=, and the default is correct for BRR/bootstrap/SDR/JK1 --
    # stratified JKn merely needs a non-default.
    rscales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.prefix or not self.prefix.strip():
            raise ValueError("RepWeights 'prefix' cannot be empty or whitespace.")
        if self.n_reps < 2:
            raise ValueError(f"n_reps must be >= 2. Got {self.n_reps}.")
        if self.df is not None and self.df <= 0:
            raise ValueError(f"df must be > 0. Got {self.df}.")
        if self.padding is not None and self.padding < 0:
            raise ValueError(f"padding must be >= 0. Got {self.padding}.")

    # ---- back-compat read surface ------------------------------------------

    @property
    def method(self) -> str:
        """The coarse method family, as a display label.

        One of ``"Bootstrap"``, ``"Jackknife"``, ``"BRR"``, ``"SDR"``. A label
        for display and reporting: the variant's own type is what decides
        anything, so nothing reads this to choose a code path.
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
        """Per-replicate variance coefficients.

        Lives on the variant rather than in a distant ``match`` so that adding a
        method, or a bootstrap kind whose scale differs, stays a local change.
        """
        raise NotImplementedError

    # ---- display -------------------------------------------------------------

    def _variant_parts(self) -> list[str]:
        """Variant-specific fragments for repr. Overridden where there are any."""
        return []

    def __repr__(self) -> str:
        parts = [f"method={self.method}", f"prefix='{self.prefix}'", f"n_reps={self.n_reps}"]
        if self.df is not None:
            parts.append(f"df={self.df}")
        parts.extend(self._variant_parts())
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
        return "\n".join(lines)

    def _plain_variant_lines(self) -> list[str]:
        return []


# =============================================================================
# The four dispatch units, mirroring Rust's RepMethod
# =============================================================================


class BootstrapWgts(_RepWgtsBase, frozen=True, kw_only=True, tag="Bootstrap", tag_field="method"):
    kind: BootstrapKind = RaoWu()

    def coefficients(self) -> list[float]:
        return self.kind.coefficients(self.n_reps)

    def _variant_parts(self) -> list[str]:
        return [] if isinstance(self.kind, RaoWu) else [f"kind={self.kind}"]

    def _plain_variant_lines(self) -> list[str]:
        return [f"Kind     : {self.kind}"]


class JackknifeWgts(_RepWgtsBase, frozen=True, kw_only=True, tag="Jackknife", tag_field="method"):
    paired: bool = False  # JK2 paired vs JK1/JKn

    def coefficients(self) -> list[float]:
        if self.rscales is not None:
            return list(self.rscales)
        return [(self.n_reps - 1) / self.n_reps] * self.n_reps

    def _variant_parts(self) -> list[str]:
        return ["paired=True"] if self.paired else []


class BrrWgts(_RepWgtsBase, frozen=True, kw_only=True, tag="BRR", tag_field="method"):
    # Fay's BRR is this at a non-zero coefficient, not a separate type: the
    # scale 1/(B(1-f)^2) collapses to 1/B at f=0.
    fay_coef: float = 0.0  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fay_coef < 0:
            raise ValueError(f"fay_coef cannot be negative. Got {self.fay_coef}.")

    def coefficients(self) -> list[float]:
        scale = 1.0 / (self.n_reps * (1.0 - self.fay_coef) ** 2)
        return [scale] * self.n_reps

    def _variant_parts(self) -> list[str]:
        return [f"fay={self.fay_coef}"]

    def _plain_variant_lines(self) -> list[str]:
        return [f"Fay coef : {self.fay_coef}"]


class SdrWgts(_RepWgtsBase, frozen=True, kw_only=True, tag="SDR", tag_field="method"):
    def coefficients(self) -> list[float]:
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


def normalize_bootstrap_kind(kind: str | RaoWu | Poisson) -> BootstrapKind:
    """Normalize a bootstrap ``kind`` to its variant.

    Case-insensitive and tolerant of hyphen, underscore or space separators,
    matching :func:`_normalize_rep_method`.
    """
    if isinstance(kind, (RaoWu, Poisson)):
        return kind
    if not isinstance(kind, str):
        raise TypeError(f"'kind' must be a string, RaoWu or Poisson, got {type(kind).__name__}.")
    cls = _BS_KIND_ALIASES.get(kind.strip().lower())
    if cls is None:
        raise MethodError.invalid_choice(
            where="svy.RepWeights",
            param="kind",
            got=kind,
            allowed=["rao-wu", "poisson"],
            docs_url=None,
            hint=(
                "'rao-wu' is the stratified Rao-Wu-Yue rescaling bootstrap and needs "
                "psu on the design; 'poisson' is the Beaumont-Patak generalized "
                "bootstrap and needs only a weight. Construct the variant directly "
                "with svy.RaoWu() or svy.Poisson(calib_domains=...) for full control."
            ),
        )
    return cls()


_MISSING_ARG: object = object()


def RepWeights(  # noqa: N802 - a factory that replaced a class of this name
    method: str = _MISSING_ARG,  # type: ignore[assignment]
    prefix: str = _MISSING_ARG,  # type: ignore[assignment]
    n_reps: int = _MISSING_ARG,  # type: ignore[assignment]
    fay_coef: float = 0.0,
    df: int | None = None,
    padding: int | None = None,
    rscales: tuple[float, ...] | None = None,
    *,
    kind: str | RaoWu | Poisson | None = None,
    paired: bool = False,
) -> RepWgts:
    """Build the replicate-weight variant for ``method``.

    Kept as a factory with the pre-union signature so existing call sites keep
    working. New code can construct the variant directly:

    >>> BootstrapWgts(prefix="bsw", n_reps=1000, kind=Poisson())
    >>> BrrWgts(prefix="brr_", n_reps=32, fay_coef=0.5)

    Parameters that do not belong to ``method`` are rejected rather than stored:
    the union is what makes ``fay_coef`` on a bootstrap unrepresentable, and the
    factory is the boundary that enforces it for string callers.
    """
    # Sentinels rather than bare required parameters so the message matches the
    # struct this factory replaced.
    for _name, _val in (("method", method), ("prefix", prefix), ("n_reps", n_reps)):
        if _val is _MISSING_ARG:
            raise TypeError(f"Missing required argument {_name!r}")

    variant = resolve_rep_variant(method)

    common = dict(prefix=prefix, n_reps=n_reps, df=df, padding=padding, rscales=rscales)

    if variant is BootstrapWgts:
        _reject(variant, fay_coef=fay_coef, paired=paired)
        return BootstrapWgts(
            **common, kind=normalize_bootstrap_kind(kind) if kind is not None else RaoWu()
        )
    if variant is JackknifeWgts:
        _reject(variant, fay_coef=fay_coef, kind=kind)
        return JackknifeWgts(**common, paired=paired)
    if variant is BrrWgts:
        _reject(variant, kind=kind, paired=paired)
        return BrrWgts(**common, fay_coef=fay_coef)
    _reject(variant, fay_coef=fay_coef, kind=kind, paired=paired)
    return SdrWgts(**common)


_PARAM_OWNER = {
    "fay_coef": ("BrrWgts", "BRR"),
    "kind": ("BootstrapWgts", "bootstrap"),
    "paired": ("JackknifeWgts", "jackknife"),
}


def _reject(variant: type, **supplied) -> None:
    """Reject parameters that belong to a different method.

    The flat signature can express combinations the union cannot; this is where
    they are turned away, so nothing downstream holds a value that is wrong.
    """
    for name, value in supplied.items():
        empty = 0.0 if name == "fay_coef" else (False if name == "paired" else None)
        if value == empty or value is empty:
            continue
        owner, owner_method = _PARAM_OWNER[name]
        raise MethodError.invalid_choice(
            where="svy.RepWeights",
            param=name,
            got=value,
            allowed=[empty],
            hint=(
                f"'{name}' is a {owner_method} parameter and is not stored on a "
                f"{variant.__name__} design. Each method carries only its own "
                f"parameters: {owner} has '{name}'."
            ),
        )
