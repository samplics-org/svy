# src/svy/serialize/serializers.py
"""
Serializer functions that translate svy result objects into the stable
``Data`` structs defined in ``structs.py``.

No existing svy classes are modified — these functions read public
attributes and convert types (numpy → list, StrEnum → str, etc.).
"""

from __future__ import annotations

import datetime as dt
import math

from enum import Enum
from typing import Any, Callable, cast

import msgspec
import msgspec.inspect as mi
import numpy as np

from svy.categorical.table import Table
from svy.categorical.ttest import TTestOneGroup, TTestTwoGroups
from svy.core import design_parts as _design_parts
from svy.core.containers import ChiSquare
from svy.core.describe import DescribeResult
from svy.core.design import Design, PopSize
from svy.errors.model_errors import ModelError
from svy.errors.serialization_errors import SerializationError
from svy.estimation.estimate import Estimate, EstimateList
from svy.regression.glm import GLMFit
from svy.regression.prediction import GLMPred
from svy.serialize.structs import (
    _KIND_TO_STRUCT,
    CellEstData,
    ChiSquareData,
    DescribeResultData,
    DesignData,
    DiffEstData,
    EstimateData,
    EstimateListData,
    FDistData,
    GLMCoefData,
    GLMFitData,
    GLMPredData,
    GLMStatsData,
    GroupLevelsData,
    LevelLabelData,
    ParamEstData,
    PopSizeData,
    ResultData,
    TableData,
    TableStatsData,
    TDistData,
    TtestEstData,
    TTestOneGroupData,
    TTestStatsData,
    TTestTwoGroupsData,
    VarLabelsData,
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
        prob=_f(p.prob) if getattr(p, "prob", None) is not None else None,
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
        df=int(t.df) if isinstance(t.df, (int, np.integer)) else _f(t.df),
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
        deff_ref=result.deff_ref,
        as_factor=bool(result.as_factor),
        labels=[
            VarLabelsData(
                var=var,
                var_label=lab.var_label,
                values=[LevelLabelData(code=c, label=v) for c, v in lab.values.items()],
            )
            for var, lab in result._labels().items()
            if lab.var_label or lab.values
        ]
        or None,
    )


@_register(EstimateList)
def _serialize_estimate_list(result: EstimateList) -> EstimateListData:
    """Serialize ``svy.estimation.estimate.EstimateList``."""
    return EstimateListData(estimates=[_serialize_estimate(e) for e in result])


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


def serialize(result: Any) -> ResultData | DesignData:
    """
    Serialize a svy result object, or a ``Design``, to a stable ``Data`` struct.

    Parameters
    ----------
    result
        A ``Design`` (gives a ``DesignData``; see ``to_design``), or a svy
        result object: ``Estimate``, ``TTestOneGroup``,
        ``TTestTwoGroups``, ``ChiSquare``, ``Table``, ``GLMFit``,
        ``GLMPred``, ``DescribeResult``, or a fitted ``GLM`` wrapper.

    Returns
    -------
    ResultData
        A kind-tagged struct from the discriminated union.

    Raises
    ------
    SerializationError
        If no serializer is registered for the result's type
        (code ``UNSUPPORTED_RESULT_TYPE``).
    ModelError
        If ``result`` is an unfitted ``GLM`` (code ``MODEL_NOT_FITTED``).
    """
    if isinstance(result, Design):
        return _serialize_design(result)

    # Handle the GLM wrapper — GLM.fit() returns GLM, not GLMFit.
    from svy.regression.base import GLM

    if isinstance(result, GLM):
        if result.fitted is None:
            raise ModelError.not_fitted(where="serialize", method="serialize")
        result = result.fitted

    cls = type(result)
    serializer = _SERIALIZERS.get(cls)

    if serializer is None:
        raise SerializationError.unsupported_type(
            got_type=cls.__name__,
            registered=[c.__name__ for c in _SERIALIZERS],
        )

    return serializer(result)


#: JSON field mapping a JSON Pointer to the non-finite value written as ``null`` there.
NONFINITE_FIELD = "nonfinite"
#: JSON field mapping a JSON Pointer to the temporal type of the ISO string there.
TEMPORAL_FIELD = "temporal"

# datetime before date: a datetime is also a date.
_TEMPORAL: dict[str, type] = {
    "datetime": dt.datetime,
    "date": dt.date,
    "time": dt.time,
    "duration": dt.timedelta,
}


# ---------------------------------------------------------------------------
# Design: both ways
# ---------------------------------------------------------------------------

# Not in _SERIALIZERS: a design is an input, not a result, and has no table.


def _as_list(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def _as_tuple(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v


def _serialize_design(design: Design) -> DesignData:
    """Serialize ``svy.core.design.Design``: every field, lossless.

    Each design part saves itself (``DesignPart.to_data``) into its own field,
    or into ``parts`` when ``DesignData`` has none for it.
    """
    pop = design.pop_size
    saved: dict[str, Any] = {}
    others: dict[str, Any] = {}
    for part, value in design._part_items():
        data = part.to_data(value)
        if part.name in DesignData.__struct_fields__:
            saved[part.name] = data
        else:
            others[part.name] = msgspec.to_builtins(data)
    return DesignData(
        case_id=design.case_id,
        wave=design.wave,
        stratum=_as_list(design.stratum),
        wgt=design.wgt,
        prob=design.prob,
        hit=design.hit,
        mos=design.mos,
        psu=_as_list(design.psu),
        ssu=_as_list(design.ssu),
        pop_size=PopSizeData(psu=pop.psu, ssu=pop.ssu) if isinstance(pop, PopSize) else pop,
        wr=bool(design.wr),
        parts=others or None,
        **saved,
    )


def to_design(data: DesignData) -> Design:
    """
    Rebuild the live ``Design`` a ``DesignData`` was serialized from.

    ``to_design(from_json(to_json(design))) == design``. Pair it with the data
    version it was saved with: ``Sample(data, design)`` checks that the frame
    holds ``design.columns()``.

    Raises
    ------
    SerializationError
        If ``data`` is not a ``DesignData`` (code ``PAYLOAD_NOT_A_DESIGN``).
    """
    if not isinstance(data, DesignData):
        raise SerializationError.not_a_design(got_type=type(data).__name__)
    pop = data.pop_size
    others = data.parts or {}
    parts: dict[str, Any] = {}
    for part in _design_parts.registered():
        if part.name in DesignData.__struct_fields__:
            raw = getattr(data, part.name)
        else:
            raw = others.get(part.name)
            if raw is not None and part.data_type is not None:
                raw = msgspec.convert(raw, type=part.data_type)
        parts[part.name] = None if raw is None else part.from_data(raw)
    return Design(
        case_id=data.case_id,
        wave=data.wave,
        stratum=_as_tuple(data.stratum),
        wgt=data.wgt,
        prob=data.prob,
        hit=data.hit,
        mos=data.mos,
        psu=_as_tuple(data.psu),
        ssu=_as_tuple(data.ssu),
        pop_size=PopSize(psu=pop.psu, ssu=pop.ssu) if isinstance(pop, PopSizeData) else pop,
        wr=data.wr,
        **parts,
    )


def to_json(result: Any) -> bytes:
    """Serialize a svy result object to JSON bytes.

    JSON has no NaN or infinity, so each one is written as ``null`` and its
    value recorded under ``"nonfinite"``, keyed by JSON Pointer (RFC 6901),
    e.g. ``{"/estimates/3/cv": "inf"}``. Consumers that ignore the field see
    ``null``; ``from_json`` restores the exact value.

    A date, datetime, time or duration (a level of a temporal column) is
    written as its ISO 8601 string and its type recorded under
    ``"temporal"``, e.g. ``{"/estimates/0/by_level/0": "date"}``.
    """
    nonfinite: dict[str, str] = {}
    temporal: dict[str, str] = {}
    builtins = msgspec.to_builtins(serialize(result), builtin_types=tuple(_TEMPORAL.values()))
    raw = _pull_special(builtins, "", nonfinite, temporal)
    if nonfinite:
        raw[NONFINITE_FIELD] = nonfinite
    if temporal:
        raw[TEMPORAL_FIELD] = temporal
    return msgspec.json.encode(raw)


def _pull_special(obj: Any, path: str, nonfinite: dict[str, str], temporal: dict[str, str]) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        nonfinite[path] = "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
        return None
    for kind, cls in _TEMPORAL.items():
        if isinstance(obj, cls):
            temporal[path] = kind
            return msgspec.to_builtins(obj)
    if isinstance(obj, dict):
        return {
            k: _pull_special(v, f"{path}/{_escape(k)}", nonfinite, temporal)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_pull_special(v, f"{path}/{i}", nonfinite, temporal) for i, v in enumerate(obj)]
    return obj


def _escape(key: str) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def _put_special(root: Any, found: dict[str, str], restore: Callable[[str, Any], Any]) -> None:
    """Replace the value at each pointer in ``found`` (dicts, lists or structs) by ``restore``."""
    for pointer, tag in found.items():
        parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]]
        node = root
        for p in parts[:-1]:
            node = _child(node, p)
        last = parts[-1]
        value = restore(tag, _child(node, last))
        if isinstance(node, list):
            node[int(last)] = value
        elif isinstance(node, dict):
            node[last] = value
        else:
            msgspec.structs.force_setattr(node, last, value)


def _child(node: Any, key: str) -> Any:
    if isinstance(node, list):
        return node[int(key)]
    return node[key] if isinstance(node, dict) else getattr(node, key)


def to_dict(result: Any) -> dict[str, Any]:
    """Serialize a svy result object to a JSON-safe dict."""
    return msgspec.to_builtins(serialize(result))


def from_json(data: bytes) -> ResultData | DesignData:
    """
    Decode JSON bytes produced by ``to_json`` back into a ``Data`` struct.

    The ``kind`` field in the JSON determines which struct type is returned.
    Discrimination is manual (not via msgspec's ``tag_field``) so that
    ``kind`` remains a real, accessible attribute on the decoded struct.

    Raises
    ------
    SerializationError
        If the payload has no ``kind`` field (code ``PAYLOAD_MISSING_KIND``)
        or an unrecognized one (code ``PAYLOAD_UNKNOWN_KIND``).
    """
    raw = msgspec.json.decode(data)
    kind = raw.get("kind")
    if kind is None:
        raise SerializationError.missing_kind()
    cls = _KIND_TO_STRUCT.get(kind)
    if cls is None:
        raise SerializationError.unknown_kind(kind=kind, known=sorted(_KIND_TO_STRUCT))
    _put_special(raw, raw.pop(NONFINITE_FIELD, None) or {}, lambda tag, _: float(tag))
    temporal = raw.pop(TEMPORAL_FIELD, None) or {}
    data = msgspec.convert(_null_to_nan(raw, mi.type_info(cls)), type=cls)
    # After decoding: level fields are typed str | int | float | bool, and
    # msgspec allows no date beside str in a union, so the ISO strings decode
    # as str and are converted here.
    _put_special(data, temporal, lambda tag, s: msgspec.convert(s, type=_TEMPORAL[tag]))
    return cast(ResultData | DesignData, data)


def _is_float(t: mi.Type) -> bool:
    """A plain ``float``, or a union with one that does not admit ``None``."""
    if isinstance(t, mi.UnionType):
        members = t.types
        return any(isinstance(u, mi.FloatType) for u in members) and not any(
            isinstance(u, mi.NoneType) for u in members
        )
    return isinstance(t, mi.FloatType)


def _null_to_nan(obj: Any, t: mi.Type) -> Any:
    """Read a ``null`` in a plain ``float`` field as NaN.

    For payloads written before ``"nonfinite"`` existed, whose NaNs and
    infinities are bare ``null``s that would otherwise not decode. Optional
    fields keep ``None``.
    """
    if obj is None:
        return math.nan if _is_float(t) else None
    if isinstance(t, mi.UnionType):
        match = [u for u in t.types if isinstance(u, (mi.StructType, mi.ListType))]
        return _null_to_nan(obj, match[0]) if len(match) == 1 else obj
    if isinstance(t, mi.StructType) and isinstance(obj, dict):
        for f in t.fields:
            if f.encode_name in obj:
                obj[f.encode_name] = _null_to_nan(obj[f.encode_name], f.type)
    elif isinstance(t, mi.ListType) and isinstance(obj, list):
        return [_null_to_nan(v, t.item_type) for v in obj]
    return obj
