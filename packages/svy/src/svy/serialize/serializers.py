# src/svy/serialize/serializers.py
"""
Serializer functions that translate svy result objects into the stable
``Data`` structs defined in ``structs.py``.

No existing svy classes are modified — these functions read public
attributes and convert types (numpy → list, StrEnum → str, etc.).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable

import msgspec
import numpy as np

from svy.core.containers import ChiSquare
from svy.core.describe import DescribeResult
from svy.categorical.table import Table
from svy.categorical.ttest import TTestOneGroup, TTestTwoGroups
from svy.estimation.estimate import Estimate
from svy.regression.glm import GLMFit
from svy.regression.prediction import GLMPred

from svy.serialize.structs import (
    CellEstData,
    ChiSquareData,
    DescribeResultData,
    DiffEstData,
    EstimateData,
    FDistData,
    GLMCoefData,
    GLMFitData,
    GLMPredData,
    GLMStatsData,
    GroupLevelsData,
    ParamEstData,
    ResultData,
    TableData,
    TableStatsData,
    TDistData,
    TTestOneGroupData,
    TTestStatsData,
    TTestTwoGroupsData,
    TtestEstData,
    _KIND_TO_STRUCT,
)

# ---------------------------------------------------------------------------
# Type-conversion helpers
# ---------------------------------------------------------------------------


def _f(v: Any) -> float:
    """Coerce a Number (int, float, np.floating) to Python float."""
    return float(v)


def _i(v: Any) -> int:
    """Coerce to Python int."""
    return int(v)


def _enum(v: Any) -> str:
    """Convert a StrEnum (or any Enum) to its string value."""
    return v.value if isinstance(v, Enum) else str(v)


def _list_or_none(v: Any) -> list | None:
    """Convert a tuple/sequence/ndarray to list, preserving None."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.tolist()
    return list(v)


def _arr(v: Any) -> list[float]:
    """Convert a numpy array (or sequence) to list[float]."""
    if v is None:
        return None  # type: ignore[return-value]
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.tolist()]
    return [float(x) for x in v]


def _enc_hook(obj: Any) -> Any:
    """msgspec enc_hook for types it doesn't handle natively (numpy)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Enum):
        return obj.value
    raise NotImplementedError(f"Cannot serialize {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Sub-struct converters
# ---------------------------------------------------------------------------


def _param_est_to_data(p: Any) -> ParamEstData:
    return ParamEstData(
        y=p.y,
        est=_f(p.est),
        se=_f(p.se),
        cv=_f(p.cv),
        lci=_f(p.lci),
        uci=_f(p.uci),
        by=list(p.by) if p.by else None,
        by_level=list(p.by_level) if p.by_level else None,
        y_level=p.y_level,
        x=p.x,
        x_level=p.x_level,
        deff=_f(p.deff) if p.deff is not None else None,
        df=int(p.df) if p.df is not None else None,
    )


def _diff_est_to_data(d: Any) -> DiffEstData:
    return DiffEstData(
        y=d.y,
        diff=_f(d.diff),
        se=_f(d.se),
        lci=_f(d.lci),
        uci=_f(d.uci),
        by=d.by,
        by_level=d.by_level,
    )


def _ttest_est_to_data(e: Any) -> TtestEstData:
    return TtestEstData(
        by=e.by,
        by_level=e.by_level,
        group=e.group,
        group_level=e.group_level,
        y=e.y,
        y_level=e.y_level,
        est=_f(e.est),
        se=_f(e.se),
        cv=_f(e.cv),
        lci=_f(e.lci),
        uci=_f(e.uci),
    )


def _ttest_stats_to_data(s: Any) -> TTestStatsData:
    return TTestStatsData(
        t=_f(s.t),
        df=_f(s.df),
        p_value=_f(s.p_value),
    )


def _group_levels_to_data(g: Any) -> GroupLevelsData:
    return GroupLevelsData(
        var=g.var,
        levels=list(g.levels),
    )


def _cell_est_to_data(c: Any) -> CellEstData:
    return CellEstData(
        rowvar=c.rowvar,
        colvar=c.colvar,
        est=_f(c.est),
        se=_f(c.se),
        cv=_f(c.cv),
        lci=_f(c.lci),
        uci=_f(c.uci),
    )


def _chi_square_to_data(c: Any) -> ChiSquareData:
    return ChiSquareData(
        df=_f(c.df),
        value=_f(c.value),
        p_value=_f(c.p_value),
    )


def _fdist_to_data(f: Any) -> FDistData:
    return FDistData(
        df_num=_f(f.df_num),
        df_den=_f(f.df_den),
        value=_f(f.value),
        p_value=_f(f.p_value),
    )


def _tdist_to_data(t: Any) -> TDistData:
    return TDistData(
        df=_f(t.df),
        value=_f(t.value),
        p_value=_f(t.p_value),
    )


def _table_stats_to_data(s: Any) -> TableStatsData:
    return TableStatsData(
        chisq=_chi_square_to_data(s.chisq),
        f=_fdist_to_data(s.f) if s.f is not None else None,
    )


def _glm_coef_to_data(c: Any) -> GLMCoefData:
    return GLMCoefData(
        term=c.term,
        est=_f(c.est),
        se=_f(c.se),
        lci=_f(c.lci),
        uci=_f(c.uci),
        wald=_tdist_to_data(c.wald) if c.wald is not None else None,
        wald_adj=_tdist_to_data(c.wald_adj) if c.wald_adj is not None else None,
    )


def _glm_stats_to_data(s: Any) -> GLMStatsData:
    return GLMStatsData(
        n=_i(s.n),
        wald=_fdist_to_data(s.wald),
        wald_adj=_fdist_to_data(s.wald_adj),
        scale=_f(s.scale),
        deviance=_f(s.deviance),
        aic=_f(s.aic) if s.aic is not None else None,
        bic=_f(s.bic) if s.bic is not None else None,
        r_squared=_f(s.r_squared) if s.r_squared is not None else None,
        r_squared_adj=_f(s.r_squared_adj) if s.r_squared_adj is not None else None,
        iterations=_i(s.iterations) if s.iterations is not None else None,
    )


# ---------------------------------------------------------------------------
# Dispatch registry
# ---------------------------------------------------------------------------

_SERIALIZERS: dict[type, Callable[[Any], ResultData]] = {}


def _register(cls: type) -> Callable[[Callable[[Any], ResultData]], Callable[[Any], ResultData]]:
    """Decorator: register a serializer function for a result class."""

    def decorator(fn: Callable[[Any], ResultData]) -> Callable[[Any], ResultData]:
        _SERIALIZERS[cls] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Serializer functions
# ---------------------------------------------------------------------------


@_register(Estimate)
def _serialize_estimate(result: Estimate) -> EstimateData:
    """Serialize ``svy.estimation.estimate.Estimate``."""
    return EstimateData(
        param=_enum(result.param),
        method=_enum(result.method),
        alpha=_f(result.alpha),
        estimates=[_param_est_to_data(p) for p in result.estimates],
        n_strata=_i(result.n_strata),
        n_psus=_i(result.n_psus),
        where_clause=result.where_clause,
        q_method=_enum(result.q_method),
    )


@_register(TTestOneGroup)
def _serialize_ttest_one_group(result: TTestOneGroup) -> TTestOneGroupData:
    """Serialize ``svy.categorical.ttest.TTestOneGroup``."""
    return TTestOneGroupData(
        y=result.y,
        mean_h0=_f(result.mean_h0),
        alternative=result.alternative,
        alpha=_f(result.alpha),
        diff=[_diff_est_to_data(d) for d in result.diff],
        estimates=[_ttest_est_to_data(e) for e in result.estimates],
        stats=_ttest_stats_to_data(result.stats) if result.stats is not None else None,
    )


@_register(TTestTwoGroups)
def _serialize_ttest_two_groups(result: TTestTwoGroups) -> TTestTwoGroupsData:
    """Serialize ``svy.categorical.ttest.TTestTwoGroups``."""
    return TTestTwoGroupsData(
        y=result.y,
        groups=_group_levels_to_data(result.groups),
        paired=result.paired,
        alternative=result.alternative,
        alpha=_f(result.alpha),
        diff=[_diff_est_to_data(d) for d in result.diff],
        estimates=[_ttest_est_to_data(e) for e in result.estimates],
        stats=_ttest_stats_to_data(result.stats) if result.stats is not None else None,
    )


@_register(ChiSquare)
def _serialize_chi_square(result: ChiSquare) -> ChiSquareData:
    """Serialize ``svy.core.containers.ChiSquare``."""
    return _chi_square_to_data(result)


@_register(Table)
def _serialize_table(result: Table) -> TableData:
    """Serialize ``svy.categorical.table.Table``."""
    return TableData(
        type=_enum(result.type),
        rowvar=result.rowvar,
        colvar=result.colvar,
        alpha=_f(result.alpha),
        estimates=[_cell_est_to_data(c) for c in (result.estimates or [])],
        stats=_table_stats_to_data(result.stats) if result.stats is not None else None,
        rowvals=_list_or_none(result.rowvals),
        colvals=_list_or_none(result.colvals),
    )


@_register(GLMFit)
def _serialize_glm_fit(result: GLMFit) -> GLMFitData:
    """Serialize ``svy.regression.glm.GLMFit``."""
    return GLMFitData(
        y=result.y,
        family=result.family,
        link=result.link,
        stats=_glm_stats_to_data(result.stats),
        coefs=[_glm_coef_to_data(c) for c in result.coefs],
        feature_names=list(result.feature_names),
    )


@_register(GLMPred)
def _serialize_glm_pred(result: GLMPred) -> GLMPredData:
    """Serialize ``svy.regression.prediction.GLMPred``."""
    return GLMPredData(
        df=_f(result.df),
        alpha=_f(result.alpha),
        yhat=_arr(result.yhat),
        se=_arr(result.se),
        lci=_arr(result.lci),
        uci=_arr(result.uci),
        residuals=_arr(result.residuals) if result.residuals is not None else None,
    )


@_register(DescribeResult)
def _serialize_describe_result(result: DescribeResult) -> DescribeResultData:
    """Serialize ``svy.core.describe.DescribeResult``."""
    return DescribeResultData(
        weighted=result.weighted,
        weight_col=result.weight_col,
        drop_nulls=result.drop_nulls,
        top_k=_i(result.top_k),
        percentiles=list(result.percentiles),
        generated_at=result.generated_at.isoformat(),
        notes=result.notes,
        items=[msgspec.to_builtins(item, enc_hook=_enc_hook) for item in result.items],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serialize(result: Any) -> ResultData:
    """
    Serialize a svy result object to a stable ``Data`` struct.

    Parameters
    ----------
    result
        A svy result object: ``Estimate``, ``TTestOneGroup``,
        ``TTestTwoGroups``, ``ChiSquare``, ``Table``, ``GLMFit``,
        ``GLMPred``, ``DescribeResult``, or a fitted ``GLM`` wrapper.

    Returns
    -------
    ResultData
        A kind-tagged struct from the discriminated union.

    Raises
    ------
    TypeError
        If no serializer is registered for the result's type.
    """
    # Handle the GLM wrapper — GLM.fit() returns GLM, not GLMFit.
    from svy.regression.base import GLM

    if isinstance(result, GLM):
        if result.fitted is None:
            raise ValueError("Cannot serialize an unfitted GLM model (call .fit() first).")
        result = result.fitted

    cls = type(result)
    serializer = _SERIALIZERS.get(cls)

    if serializer is None:
        raise TypeError(
            f"No serializer registered for {cls.__name__}. "
            f"Registered types: {[c.__name__ for c in _SERIALIZERS]}"
        )

    return serializer(result)


def to_json(result: Any) -> bytes:
    """Serialize a svy result object to JSON bytes."""
    return msgspec.json.encode(serialize(result))


def to_dict(result: Any) -> dict[str, Any]:
    """Serialize a svy result object to a JSON-safe dict."""
    return msgspec.to_builtins(serialize(result))


def from_json(data: bytes) -> ResultData:
    """
    Decode JSON bytes produced by ``to_json`` back into a ``Data`` struct.

    The ``kind`` field in the JSON determines which struct type is returned.
    Discrimination is manual (not via msgspec's ``tag_field``) so that
    ``kind`` remains a real, accessible attribute on the decoded struct.
    """
    raw = msgspec.json.decode(data)
    kind = raw.get("kind")
    if kind is None:
        raise ValueError("JSON payload is missing the 'kind' discriminator.")
    cls = _KIND_TO_STRUCT.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown kind {kind!r}. Known kinds: {sorted(_KIND_TO_STRUCT)}"
        )
    return msgspec.json.decode(data, type=cls)
