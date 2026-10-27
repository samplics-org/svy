# svy/errors/_common.py
"""Constructors every error class inherits; each returns the class's kind for its problem."""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Optional, Self


class _Constructors:
    """Problems any part of svy can raise: ``EstimationError.missing_columns(...)``
    is an ``EstimationError`` and a ``KeyError``."""

    # Set on every error class by _base._add_kinds; declared so type checkers see them.
    TypeError: ClassVar[type[Self]]
    ValueError: ClassVar[type[Self]]
    KeyError: ClassVar[type[Self]]
    AttributeError: ClassVar[type[Self]]
    NotImplementedError: ClassVar[type[Self]]
    RuntimeError: ClassVar[type[Self]]
    ImportError: ClassVar[type[Self]]
    OSError: ClassVar[type[Self]]
    FileNotFoundError: ClassVar[type[Self]]
    PermissionError: ClassVar[type[Self]]

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
    ) -> Self:
        allowed_list = list(allowed)
        return cls.ValueError(
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
    ) -> Self:
        return cls.ValueError(
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
    ) -> Self:
        """A wrangling step would write new values into a weight column.

        ``columns`` maps each column to what depends on it, as a phrase that
        completes "'<column>' is ...".
        """
        roles = "; ".join(f"{c!r} is {why}" for c, why in columns.items())
        col = next(iter(columns))
        return cls.ValueError(
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
    def reserved_columns(cls, *, where: Optional[str], columns: list[str]) -> Self:
        """User columns named like svy's bookkeeping, which svy rebuilds."""
        col = columns[0]
        return cls.ValueError(
            title="Column names reserved by svy",
            detail=(
                f"{', '.join(map(repr, columns))}: svy builds columns of these names from "
                "the data and design and would overwrite them. Names starting with "
                "'__svy_' are svy's."
            ),
            code="RESERVED_COLUMN",
            where=where,
            param=col,
            got=columns,
            hint=(
                f"Use another name, e.g. {col.removeprefix('__svy_').strip('_')!r}. "
                "If the frame came from svy "
                "(sample._data, or a file an earlier version saved), drop them, svy "
                f"rebuilds them: data.drop({columns!r})."
            ),
        )

    @classmethod
    def data_rows_changed(cls, *, where: Optional[str], n_old: int, n_new: int) -> Self:
        return cls.ValueError(
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
    def no_rep_wgts(cls, *, where: Optional[str]) -> Self:
        return cls.ValueError(
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
    def invalid_rep_prefix(cls, *, where: Optional[str], got: Any) -> Self:
        return cls.ValueError(
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
    ) -> Self:
        listed = ", ".join(f"{p!r} ({n} replicates of {w!r})" for p, (w, n) in known.items())
        return cls.KeyError(
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
    ) -> Self:
        shown = existing[:10]
        more = "" if len(existing) <= 10 else f" and {len(existing) - 10} more"
        return cls.ValueError(
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
    def mutate_cycle(cls, cycle_list: str, *, where: str | None = None) -> Self:
        return cls.ValueError(
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
    ) -> Self:
        exp = (
            f"{min_} < {param} < {max_}"
            if (min_ is not None and max_ is not None)
            else "valid range"
        )
        return cls.ValueError(
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
    ) -> Self:
        return cls.TypeError(
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
    ) -> Self:
        miss = list(missing)
        ext = list(extra)
        detail_parts = []
        if miss:
            detail_parts.append(f"Missing keys: {miss}.")
        if ext:
            detail_parts.append(f"Unexpected keys: {ext}.")
        return cls.ValueError(
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

    @classmethod
    def invalid_n(
        cls,
        *,
        where: Optional[str],
        got: Any,
        expected: Any = "n >= 0 or None",
        hint: Optional[str] = "Use None to select all rows.",
        docs_url: Optional[str] = None,
    ) -> Self:
        return cls.ValueError(
            title="Invalid row count",
            detail="Parameter 'n' must be a non-negative integer or None.",
            code="INVALID_N",
            where=where,
            param="n",
            expected=expected,
            got=got,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def missing_columns(
        cls,
        *,
        where: Optional[str],
        param: str,
        missing: Iterable[str],
        available: Iterable[str] | None = None,
        hint: Optional[str] = "Check spelling or inspect df.columns.",
        docs_url: Optional[str] = None,
    ) -> Self:
        missing_list = list(missing)
        extra = {"available_preview": list(available)[:50]} if available is not None else None
        return cls.KeyError(
            title="Column(s) not found",
            detail=f"Missing: {', '.join(map(repr, missing_list))}.",
            code="MISSING_COLUMNS",
            where=where,
            param=param,
            expected="subset of dataframe columns",
            got=missing_list,
            hint=hint,
            docs_url=docs_url,
            extra=extra,
        )

    @classmethod
    def sample_too_large(
        cls,
        *,
        where: Optional[str],
        n: int,
        available_rows: int,
        param: str = "n",
        hint: Optional[str] = "When sampling without replacement, ensure n ≤ number of rows.",
        docs_url: Optional[str] = None,
    ) -> Self:
        return cls.ValueError(
            title="Sampling failed",
            detail=f"Requested n={n} exceeds available rows={available_rows} for sampling without replacement.",
            code="SAMPLE_TOO_LARGE",
            where=where,
            param=param,
            expected=f"n ≤ {available_rows}",
            got=n,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def empty_estimates(
        cls,
        *,
        where: str | None,
        param: str = "estimates",
        hint: str | None = "Run an analysis first or pass non-empty estimates.",
        docs_url: str | None = None,
    ) -> Self:
        return cls.ValueError(
            title="No estimates to display",
            detail="The table/test has no estimate rows.",
            code="EMPTY_ESTIMATES",
            where=where,
            param=param,
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def domain_keys_mismatch(
        cls,
        *,
        where: str | None,
        domain: str,
        expected_keys: Iterable[Any],
        got_keys: Iterable[Any],
        hint: str | None = "Align your provided levels/controls with the domain categories.",
        docs_url: str | None = None,
    ) -> Self:
        exp = set(expected_keys)
        got = set(got_keys)
        return cls.ValueError(
            title="Domain keys mismatch",
            detail=f"For domain '{domain}', expected keys {sorted(exp)}, got {sorted(got)}.",
            code="DOMAIN_KEYS_MISMATCH",
            where=where,
            param=domain,
            expected=sorted(exp),
            got=sorted(got),
            hint=hint,
            docs_url=docs_url,
        )

    @classmethod
    def group_levels_mismatch(
        cls,
        *,
        where: str | None,
        var: str,
        expected_levels: Iterable[Any],
        got_levels: Iterable[Any],
        hint: str | None = "Ensure estimates only include the two requested levels.",
        docs_url: str | None = None,
    ) -> Self:
        return cls.ValueError(
            title="Group levels mismatch",
            detail=f"Group '{var}' expected levels {tuple(expected_levels)}, but estimates include {sorted(set(got_levels))}.",
            code="GROUP_LEVELS_MISMATCH",
            where=where,
            param=var,
            expected=list(expected_levels),
            got=list(sorted(set(got_levels))),
            hint=hint,
            docs_url=docs_url,
        )
