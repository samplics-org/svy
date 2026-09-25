# svy/errors/method_errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .base_errors import SvyError


@dataclass(eq=False)
class MethodError(SvyError):
    """
    Raised when a method/option is invalid or not applicable in the current context.
    Examples: bad enum value, unsupported 'how' mode, incompatible estimator, etc.
    """

    def __post_init__(self) -> None:
        # If caller didn't set a specific code, use a type-specific default.
        if self.code == "SVY_ERROR":
            self.code = "METHOD_ERROR"

    # ---- Convenience constructors -------------------------------------------------

    @classmethod
    def invalid_choice(
        cls,
        *,
        where: Optional[str],
        param: str,
        got: Any,
        allowed: Iterable[Any],
        hint: Optional[str] = None,
        docs_url: Optional[str] = None,
    ) -> "MethodError":
        allowed_list = list(allowed)
        return cls(
            title="Invalid option",
            detail=f"Parameter '{param}' must be one of {allowed_list}.",
            code="INVALID_CHOICE",
            where=where,
            param=param,
            expected=allowed_list,
            got=got,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def not_applicable(
        cls,
        *,
        where: Optional[str],
        method: str,
        reason: str,
        param: Optional[str] = None,
        hint: Optional[str] = None,
        docs_url: Optional[str] = None,
    ) -> "MethodError":
        return cls(
            title="Method not applicable",
            # `reason` is authored prose and most call sites already end it with a
            # period; appending unconditionally produced "…(got psu=None)..".
            detail=f"'{method}' cannot be used here: {reason.rstrip('.')}.",
            code="METHOD_NOT_APPLICABLE",
            where=where,
            param=param,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def weight_overwrite(
        cls,
        *,
        where: Optional[str],
        columns: dict[str, str],
    ) -> "MethodError":
        """A wrangling step would write new values into a weight column.

        ``columns`` maps each column to what depends on it, as a phrase that
        completes "'<column>' is ...".
        """
        roles = "; ".join(f"{c!r} is {why}" for c, why in columns.items())
        col = next(iter(columns))
        return cls(
            title="Weight columns cannot be overwritten",
            detail=(
                f"{roles}. New values under the same name would be a different "
                "variable that the design, its weight-adjustment record, replicate "
                "weights or history still read as the old one."
            ),
            code="WEIGHT_OVERWRITE",
            where=where,
            param=col,
            got=list(columns),
            hint=(
                f"Write the new values under a new name, e.g. "
                f"sample.wrangling.mutate({{{col + '_new'!r}: ...}}), or rename first, "
                f"sample.wrangling.rename_columns({{{col!r}: {col + '_v1'!r}}}), which "
                f"carries the rename through the design and its history, then write {col!r}."
            ),
        )

    @classmethod
    def data_rows_changed(cls, *, where: Optional[str], n_old: int, n_new: int) -> "MethodError":
        return cls(
            title="Rows changed outside svy",
            detail=(
                f"The new frame has {n_new} rows; the sample has {n_old}. Rows added "
                "or removed outside svy cannot be checked against the design's "
                "weights, replicate weights and history."
            ),
            code="DATA_ROWS_CHANGED",
            where=where,
            expected=n_old,
            got=n_new,
            hint=(
                "Change rows through svy: sample.wrangling.filter_records to drop "
                "records, sample.wrangling.join to bring variables in, "
                "svy.combine_samples to stack samples. For an unrelated frame, build "
                "a new Sample(data, design)."
            ),
        )

    @classmethod
    def no_rep_wgts(cls, *, where: Optional[str]) -> "MethodError":
        return cls(
            title="No replicate weights to rename",
            detail="The design carries no replicate weights.",
            code="REP_WGTS_MISSING",
            where=where,
            hint=(
                "Declare them with Design(rep_wgts=...) or create them with "
                "sample.weighting.create_bs_wgts/create_jk_wgts/create_brr_wgts/"
                "create_sdr_wgts; other columns are renamed with rename_columns."
            ),
        )

    @classmethod
    def invalid_rep_prefix(cls, *, where: Optional[str], got: Any) -> "MethodError":
        return cls(
            title="Invalid replicate-weight prefix",
            detail=f"The new prefix must be a non-empty string; got {got!r}.",
            code="REP_WGTS_PREFIX_INVALID",
            where=where,
            param="prefix",
            got=got,
            hint="Map each prefix to the name its replicate numbers should follow, "
            "e.g. {'w': 'final_w'}.",
        )

    @classmethod
    def unknown_rep_prefix(
        cls, *, where: Optional[str], got: Any, known: dict[str, tuple[Any, int]]
    ) -> "MethodError":
        listed = ", ".join(f"{p!r} ({n} replicates of {w!r})" for p, (w, n) in known.items())
        return cls(
            title="Unknown replicate-weight set",
            detail=f"No replicate weights with prefix {got!r}. This sample carries: {listed}.",
            code="REP_WGTS_UNKNOWN_PREFIX",
            where=where,
            param="prefixes",
            got=got,
            expected=list(known),
            hint="Key the mapping by one of those prefixes, e.g. {"
            + f"{next(iter(known))!r}: 'new_prefix'"
            + "}.",
        )

    @classmethod
    def rep_rename_collision(
        cls, *, where: Optional[str], prefix: str, existing: list[str]
    ) -> "MethodError":
        shown = existing[:10]
        more = "" if len(existing) <= 10 else f" and {len(existing) - 10} more"
        return cls(
            title="Replicate-weight names already taken",
            detail=(
                f"Renaming to {prefix!r} would give names that are already columns, "
                f"or that two replicate sets would share: {shown}{more}."
            ),
            code="REP_WGTS_RENAME_COLLISION",
            where=where,
            param="prefix",
            got=prefix,
            hint=(
                "Pick another prefix, or rename or remove those columns first "
                "(sample.wrangling.rename_columns / remove_columns)."
            ),
        )

    @classmethod
    def mutate_cycle(cls, cycle_list: str, *, where: str | None = None) -> "MethodError":
        return cls(
            title="Column transformation failed",
            detail=f"Dependency cycle or unresolved forward reference among: {cycle_list}",
            code="MUTATE_CYCLE",
            where=where,
            hint="Split into multiple mutate calls or remove circular references.",
            docs_url=None,
        )

    @classmethod
    def invalid_range(
        cls,
        *,
        where: str | None,
        param: str,
        got: Any,
        min_: float | int | None = None,
        max_: float | int | None = None,
        hint: str | None = None,
        docs_url: str | None = None,
    ) -> "MethodError":
        exp = (
            f"{min_} < {param} < {max_}"
            if (min_ is not None and max_ is not None)
            else "valid range"
        )
        return cls(
            title="Invalid numeric range",
            detail=f"Parameter '{param}' is out of range.",
            code="INVALID_RANGE",
            where=where,
            param=param,
            expected=exp,
            got=got,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def invalid_type(
        cls,
        *,
        where: str | None,
        param: str,
        got: Any,
        expected: str,
        hint: str | None = None,
        docs_url: str | None = None,
    ) -> "MethodError":
        return cls(
            title="Invalid type",
            detail=f"Parameter '{param}' has the wrong type.",
            code="INVALID_TYPE",
            where=where,
            param=param,
            expected=expected,
            got=type(got).__name__ if got is not None else None,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def invalid_mapping_keys(
        cls,
        *,
        where: str | None,
        param: str,
        missing: Iterable[Any] = (),
        extra: Iterable[Any] = (),
        hint: str | None = None,
        docs_url: str | None = None,
    ) -> "MethodError":
        miss = list(missing)
        ext = list(extra)
        detail_parts = []
        if miss:
            detail_parts.append(f"Missing keys: {miss}.")
        if ext:
            detail_parts.append(f"Unexpected keys: {ext}.")
        return cls(
            title="Mapping keys mismatch",
            detail=" ".join(detail_parts) or "Keys do not match the expected set.",
            code="INVALID_MAPPING_KEYS",
            where=where,
            param=param,
            expected="exact match of domain keys",
            got={"missing": miss, "extra": ext},
            hint=hint or "Ensure your mapping keys exactly match the domain categories.",
            docs_url=docs_url,
        )
