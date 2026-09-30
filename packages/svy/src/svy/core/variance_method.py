# src/svy/core/variance_method.py
"""The variance method an analysis uses: Taylor linearization or replication.

One rule for estimation, categorical analysis and regression: replication is
never selected implicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from svy.core.warnings import WarnCode


if TYPE_CHECKING:
    from svy.core.repwgts import RepWgts
    from svy.core.sample import Sample


def normalize_variance_method(method: str | None) -> Literal["taylor", "replication"] | None:
    """
    Normalize user-facing method string to canonical form.

    Accepts case-insensitive variants:
      - Taylor: "taylor", "Taylor", "TAYLOR", "linearization", "lin"
      - Replication: "replication", "replicate", "rep",
        "bootstrap", "brr", "jackknife", "jk", "sdr"
      - None: the default (Taylor)

    Returns "taylor", "replication", or None.
    """
    if method is None:
        return None
    if not isinstance(method, str):
        raise TypeError(
            f"'method' must be a string or None, got {type(method).__name__}. "
            f"Use method='taylor' or method='replication'."
        )
    m = method.strip().lower()
    if m in ("taylor", "linearization", "lin"):
        return "taylor"
    if m in ("replication", "replicate", "rep", "bootstrap", "brr", "jackknife", "jk", "sdr"):
        return "replication"
    raise ValueError(f"Unknown estimation method {method!r}. Use 'taylor' or 'replication'.")


def resolve_variance_method(sample: Sample, method: str | None, *, where: str) -> RepWgts | None:
    """Resolve to the replicate-weight variant, or None for Taylor.

    Replication is never selected implicitly. ``method=None`` means Taylor,
    which is what the signature ``Literal["taylor", "replication"] | None``
    already implies -- ``None`` is the unstated default, not a third mode.

    It used to be a third mode: ``None`` resolved to replication whenever
    the design carried replicate weights and no ``stratum``/``psu``. That
    made the *estimator* depend on inputs the estimator never reads --
    replication consumes the replicate columns and ``coefficients()``,
    nothing else -- so declaring a single design column silently moved the
    standard error. It was worst for JKn, where ``stratum`` and ``psu`` are
    exactly what svy needs to derive ``(n_h-1)/n_h``: the one action that
    makes JKn usable was the action that switched JKn off.
    """
    normalized = normalize_variance_method(method)
    design = sample._design

    if normalized == "replication":
        if design.rep_wgts is None:
            raise ValueError(
                "Replication requires rep_wgts in the design. "
                "Create replicate weights first or use method='taylor'."
            )
        return design.rep_wgts

    # Taylor, for an explicit "taylor" and for the None default alike.
    #
    # Linearization needs design structure. Without stratum or psu every
    # row is its own PSU in a single stratum, so the variance is SRS-like
    # and df is n-1 rather than n_reps-1 -- on a 200-row file with 8
    # replicates, df=199 against a true 7. Replicate weights sitting on the
    # design is strong evidence that is not the intended estimator, but
    # choosing one is the caller's to make, so this warns and proceeds
    # rather than switching. Warned for the explicit spelling too: the
    # hazard is in the number, not in who asked for it.
    if design.rep_wgts is not None and design.stratum is None and design.variance_psu is None:
        sample.warn(
            code=WarnCode.TAYLOR_WITHOUT_DESIGN,
            title="Taylor variance on a design with no stratum or psu",
            detail=(
                "The design carries replicate weights "
                f"({design.rep_wgts.method}, n_reps={design.rep_wgts.n_reps}) but no "
                "stratum or psu, so linearization has no clustering or "
                "stratification to work from and the variance is SRS-like."
            ),
            where=where,
            param="method",
            hint=(
                "Pass method='replication' to use the replicate weights, or "
                "declare stratum/psu if the frame carries them. Pass "
                "method='taylor' explicitly to keep this variance."
            ),
        )
    return None
