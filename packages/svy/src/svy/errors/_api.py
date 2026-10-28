# src/svy/errors/_api.py
"""Guiding errors for names and arguments that do not exist in svy's API."""

from __future__ import annotations

import difflib

from dataclasses import dataclass
from functools import cache
from typing import Any, Callable, ClassVar, Iterable

from ._base import SvyError, SvyWarning
from ._codes import Code


_NAMESPACES = ("wrangling", "estimation", "categorical", "glm", "weighting", "sampling")
# Present on several objects; pointing to one namespace for them would mislead.
_NOT_INDEXED = frozenset(
    {"to_polars", "keys", "contrast", "set_print_width", "set_default_print_width"}
)

_SAMPLE_AND_DESIGN = "svy.Sample(data, svy.Design(stratum=..., psu=..., wgt=...))"
_FILTER = (
    "wrangling.filter_records(...) to drop rows; for a subpopulation estimate pass where= instead"
)
_TO_POLARS = "result.to_polars()"

# Names other software uses, or that are commonly guessed, mapped to svy's.
_ALIASES: dict[str, dict[str, str]] = {
    "svy": {
        "SvyUserWarning": "svy.SvyWarning, its name since 2026.0",
        "MethodError": "the error of the part of svy you called, in svy.errors (EstimationError, "
        "WeightingError, DataError, ...), or the builtin it also is (ValueError, TypeError)",
        "DimensionError": "the error of the part of svy you called, in svy.errors (DataError, "
        "EstimationError, ...); a missing column is also a KeyError",
        "Survey": _SAMPLE_AND_DESIGN,
        "SurveyDesign": _SAMPLE_AND_DESIGN,
        "survey_design": _SAMPLE_AND_DESIGN,
        "svydesign": _SAMPLE_AND_DESIGN,
        "design": "svy.Design(...)",
        "sample": "svy.Sample(data, design)",
        "mean": "svy.Sample(data, design).estimation.mean(...)",
        "read_excel": "polars.read_excel(...), then svy.Sample(...)",
    },
    "svy.errors": {
        "SvyUserWarning": "svy.errors.SvyWarning, its name since 2026.0",
        "WarnCode": "svy.errors.Code, which lists every error and warning code since 2026.0",
        # Split by area in 2026.0; the builtin kind (ValueError, ...) catches across areas.
        "MethodError": "the error of the part of svy you called (EstimationError, "
        "WeightingError, DataError, ...), or the builtin it also is (ValueError, TypeError)",
        "DimensionError": "the error of the part of svy you called (DataError, "
        "EstimationError, ...); a missing column is also a KeyError",
    },
    "GLMMargins": {
        "values": "at (the values each variable was set to) or contrast (effect labels)",
        "value": "at (the values each variable was set to) or contrast (effect labels)",
    },
    "Sample": {
        "regression": "sample.glm",
        "model": "sample.glm",
        "estimate": "sample.estimation",
        "tables": "sample.categorical",
        "weights": "sample.weighting",
        "selection": "sample.sampling",
        "head": "sample.show_data() or sample.data.head()",
        "filter": f"sample.{_FILTER}",
        "subset": f"sample.{_FILTER}",
    },
    "Estimation": {
        "proportion": "prop",
        "proportions": "prop",
        "percent": "prop",
        "average": "mean",
        "sum": "total",
        "quantiles": "quantile",
        "percentile": "quantile",
        "correlation": "corr",
        "covariance": "cov",
        "freq": "sample.categorical.tabulate",
        "crosstab": "sample.categorical.tabulate(rowvar, colvar)",
        "regression": "sample.glm.fit",
        "contrast": "the result's contrast: sample.estimation.mean(...).contrast(...)",
    },
    "Categorical": {
        "crosstab": "tabulate(rowvar, colvar)",
        "freq": "tabulate",
        "table": "tabulate",
        "chisq": "tabulate(rowvar, colvar); its Rao-Scott tests are in .stats",
        "chi2": "tabulate(rowvar, colvar); its Rao-Scott tests are in .stats",
        "t_test": "ttest",
        "wilcoxon": "ranktest",
        "kruskal": "ranktest",
    },
    "Weighting": {
        "calibration": "calibrate",
        "raking": "rake",
        "poststratification": "poststratify",
        "nonresponse": "adjust",
        "jackknife": "create_jk_wgts",
        "bootstrap": "create_bs_wgts",
        "brr": "create_brr_wgts",
    },
    "Wrangling": {
        "filter": _FILTER,
        "subset": _FILTER,
        "query": _FILTER,
        "rename": "rename_columns",
        "with_columns": "mutate",
        "merge": "join",
        "cut": "categorize",
        "replace": "recode",
    },
    "Selection": {
        "pps": "pps_sys (or pps_brewer, pps_rs, pps_wr)",
        "sample": "srs",
    },
    "GLM": {
        "summary": "print(model) or model.to_polars()",
        "params": "model.to_polars()",
    },
}

# Owners that are results: their columns come from to_polars().
_RESULTS = frozenset(
    {
        "Estimate",
        "EstimateList",
        "Table",
        "TTestOneGroup",
        "TTestTwoGroups",
        "TTestByResult",
        "Contrast",
    }
)


@cache
def _namespace_index() -> dict[str, tuple[str, ...]]:
    from svy.categorical.base import Categorical
    from svy.estimation.base import Estimation
    from svy.regression.base import GLM
    from svy.selection.base import Selection
    from svy.weighting.base import Weighting
    from svy.wrangling.base import Wrangling

    classes = dict(
        zip(_NAMESPACES, (Wrangling, Estimation, Categorical, GLM, Weighting, Selection))
    )
    index: dict[str, list[str]] = {}
    for ns, cls in classes.items():
        for name in public_names(cls):
            if name not in _NOT_INDEXED and callable(getattr(cls, name, None)):
                index.setdefault(name, []).append(ns)
    return {k: tuple(v) for k, v in index.items()}


def public_names(obj: Any) -> list[str]:
    return [n for n in dir(obj) if not n.startswith("_")]


@dataclass(eq=False)
class UsageError(SvyError):
    """svy's API used in a way it does not take: an unknown or removed name or
    argument, data of the wrong kind. Says what to use instead."""

    _DEFAULT_CODE: ClassVar[str] = Code.USAGE_ERROR

    @classmethod
    def data_in_design(cls, *, got: Any) -> "UsageError":
        return cls.TypeError(
            title="Design does not take data",
            detail="A Design names the design columns; the data goes to Sample.",
            code=Code.DESIGN_GIVEN_DATA,
            where="svy.Design",
            got=type(got).__name__,
            hint=f"Use {_SAMPLE_AND_DESIGN}.",
        )

    @classmethod
    def unknown_arguments(
        cls, *, where: str, unknown: Iterable[str], known: Iterable[str]
    ) -> "UsageError":
        unknown, known = list(unknown), sorted(known)
        fixes = []
        for name in unknown:
            if name in ("data", "df"):
                fixes.append(f"{name}: the data goes to svy.Sample(data, design)")
            elif name == "nest":
                fixes.append("nest: PSU codes are always taken within stratum; drop it")
            elif name in _DESIGN_ARGS:
                fixes.append(f"{name}: use {_DESIGN_ARGS[name]}=")
            elif close := difflib.get_close_matches(name, known, n=1, cutoff=0.7):
                fixes.append(f"{name}: did you mean {close[0]}=?")
        return cls.TypeError(
            title="Unknown argument",
            detail=f"{where} got unexpected keyword argument(s): {', '.join(unknown)}"
            + ("; " + "; ".join(fixes) if fixes else "")
            + ".",
            code=Code.UNKNOWN_ARGUMENT,
            where=where,
            got=unknown,
            expected=known,
        )

    @classmethod
    def not_polars(cls, *, where: str, got: Any) -> "UsageError":
        is_pandas = type(got).__module__.split(".")[0] == "pandas"
        return cls.TypeError(
            title="Data must be a polars DataFrame",
            detail=f"{where} takes a polars DataFrame or LazyFrame.",
            code=Code.DATA_NOT_POLARS,
            where=where,
            got=f"{type(got).__module__}.{type(got).__name__}",
            hint="Convert with polars.from_pandas(df), or read with svy.io.read_csv and the other svy readers."
            if is_pandas
            else "Read files with svy.io.read_csv, svy.io.read_parquet, svy.io.read_stata, ...",
        )


@dataclass(eq=False)
class UnknownNameError(UsageError, AttributeError):
    """A name that does not exist; says what to use instead.

    Also an ``AttributeError``, so ``hasattr`` stays False.
    """

    _IS_KIND: ClassVar[bool] = True

    @classmethod
    def unknown(
        cls, *, owner: str, label: str, name: str, available: Iterable[str]
    ) -> "UnknownNameError":
        available = sorted(set(available))
        alias = _ALIASES.get(owner, {}).get(name)
        homes = [] if owner in _RESULTS else _namespace_index().get(name, ())
        close = difflib.get_close_matches(name, available, n=3, cutoff=0.7)
        did_you_mean = "Did you mean " + ", ".join(close) + "?" if close else ""
        if owner in _RESULTS:
            detail = (
                f"Results are not DataFrames: {_TO_POLARS} returns every column "
                f"(est, se, lci, uci, df, n, ...) at full precision. {did_you_mean}"
            ).strip()
        elif alias:
            detail = f"Use {alias}."
        elif homes:
            detail = "It is " + " or ".join(f"sample.{ns}.{name}" for ns in homes) + "."
        else:
            detail = did_you_mean or f"{label} has no {name!r}."
        if owner == "svy":
            hint = "help(svy) shows where to start; data and design meet in svy.Sample."
        elif owner == "Sample":
            hint = "Analyses are in the namespaces: " + ", ".join(
                f"sample.{ns}" for ns in _NAMESPACES
            )
        else:
            hint = "Available: " + ", ".join(available) if len(available) <= 40 else None
        return cls(
            title=f"{label}.{name} does not exist",
            detail=detail,
            code=Code.UNKNOWN_NAME,
            where=label,
            got=name,
            hint=hint,
        )

    @classmethod
    def moved(cls, *, label: str, name: str, home: str, since: str) -> "UnknownNameError":
        return cls(
            title=f"{label}.{name} moved",
            detail=f"{label}.{name} moved to {home}.{name} in {since}.",
            code=Code.NAME_MOVED,
            where=label,
            got=name,
            expected=f"{home}.{name}",
            hint=f"Use {home}.{name}, or `from {home} import {name}`.",
        )

    @classmethod
    def renamed(cls, *, label: str, name: str, new: str, since: str) -> "UnknownNameError":
        return cls(
            title=f"{label}.{name} moved",
            detail=f"{label}.{name} moved to {new} in {since}.",
            code=Code.NAME_MOVED,
            where=label,
            got=name,
            expected=new,
            hint=f"Use {new}.",
        )


def guided_getattr(owner: str, label: str) -> Callable[[Any, str], Any]:
    """A ``__getattr__`` raising UnknownNameError for public names a class lacks."""

    def __getattr__(self: Any, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        raise UnknownNameError.unknown(
            owner=owner, label=label, name=name, available=public_names(type(self))
        )

    return __getattr__


_DESIGN_ARGS = {
    "strata": "stratum",
    "weights": "wgt",
    "weight": "wgt",
    "cluster": "psu",
    "clusters": "psu",
    "ids": "psu",
    "id": "psu",
    "fpc": "pop_size",
    "repweights": "rep_wgts",
    "rep_weights": "rep_wgts",
    "probs": "prob",
}


@dataclass(eq=False)
class UsageWarning(SvyWarning):
    pass


@dataclass(eq=False)
class ExperimentalWarning(SvyWarning):
    """A feature that may change in any release, without a deprecation period."""
