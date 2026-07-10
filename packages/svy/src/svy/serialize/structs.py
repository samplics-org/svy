# src/svy/serialize/structs.py
"""
Stable, versioned, JSON-safe structs for serializing svy result objects.

These structs are the cross-package contract consumed by svy-agents and
svyLab.  They are intentionally decoupled from svy's internal result classes:
if an internal class changes, only the serializer function (in
``serializers.py``) updates — this contract stays stable.

See ``DESIGN.md`` in this directory for the full design rationale.
"""
from __future__ import annotations

from typing import Any, Literal

import msgspec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "svy-result/0.1"

# JSON-safe alias for svy's Category type (str | int | float | bool).
CatValue = str | int | float | bool

# Mapping from kind tag to struct class, populated below.
_KIND_TO_STRUCT: dict[str, type] = {}


def _kinded(tag: str):
    """
    Class decorator: registers the struct in ``_KIND_TO_STRUCT`` and adds a
    ``kind`` default field.

    We use a regular ``kind: str`` field (not msgspec's ``tag_field``) so the
    value is accessible as an instance attribute.  Discriminated-union decoding
    is handled manually in ``serializers.from_json``.
    """

    def decorator(cls: type) -> type:
        _KIND_TO_STRUCT[tag] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Basic sub-structs (untagged, no kind/schema_version)
# ---------------------------------------------------------------------------


class FDistData(msgspec.Struct, kw_only=True, frozen=True):
    """F-distribution test result (mirrors ``svy.core.containers.FDist``)."""

    df_num: float
    df_den: float
    value: float
    p_value: float


class TDistData(msgspec.Struct, kw_only=True, frozen=True):
    """T-distribution test result (mirrors ``svy.core.containers.TDist``)."""

    df: float
    value: float
    p_value: float


# ---------------------------------------------------------------------------
# ChiSquareData — top-level (kind-tagged) but also reused as a sub-struct
# inside TableStatsData.  The kind/schema_version fields are harmless when
# nested; see DESIGN.md §6.
# ---------------------------------------------------------------------------


@_kinded("chi_square")
class ChiSquareData(msgspec.Struct, kw_only=True, frozen=True):
    """Chi-square test result (mirrors ``svy.core.containers.ChiSquare``)."""

    kind: Literal["chi_square"] = "chi_square"
    schema_version: str = SCHEMA_VERSION
    df: float
    value: float
    p_value: float


# ---------------------------------------------------------------------------
# Estimation sub-structs
# ---------------------------------------------------------------------------


class ParamEstData(msgspec.Struct, kw_only=True, frozen=True):
    """A single parameter estimate (mirrors ``svy.estimation.estimate.ParamEst``)."""

    y: str
    est: float
    se: float
    cv: float
    lci: float
    uci: float
    by: list[str] | None = None
    by_level: list[CatValue] | None = None
    y_level: CatValue | None = None
    x: str | None = None
    x_level: CatValue | None = None
    deff: float | None = None


# ---------------------------------------------------------------------------
# T-test sub-structs
# ---------------------------------------------------------------------------


class DiffEstData(msgspec.Struct, kw_only=True, frozen=True):
    """Difference estimate for a t-test (mirrors ``svy.categorical.ttest.DiffEst``)."""

    y: str
    diff: float
    se: float
    lci: float
    uci: float
    by: str | None = None
    by_level: CatValue | None = None


class TtestEstData(msgspec.Struct, kw_only=True, frozen=True):
    """Estimate for a (y, group) cell in a t-test (mirrors ``svy.categorical.ttest.TtestEst``)."""

    by: str | None
    by_level: CatValue | None
    group: str | None
    group_level: CatValue | None
    y: str
    y_level: CatValue | None
    est: float
    se: float
    cv: float
    lci: float
    uci: float


class TTestStatsData(msgspec.Struct, kw_only=True, frozen=True):
    """Core t-test statistics (mirrors ``svy.categorical.ttest.TTestStats``)."""

    t: float
    df: float
    p_value: float


class GroupLevelsData(msgspec.Struct, kw_only=True, frozen=True):
    """Two-group comparison specification (mirrors ``svy.categorical.ttest.GroupLevels``)."""

    var: str
    levels: list[CatValue]


# ---------------------------------------------------------------------------
# Table sub-structs
# ---------------------------------------------------------------------------


class CellEstData(msgspec.Struct, kw_only=True, frozen=True):
    """A single cell estimate in a table (mirrors ``svy.categorical.table.CellEst``)."""

    rowvar: str
    colvar: str
    est: float
    se: float
    cv: float
    lci: float
    uci: float


class TableStatsData(msgspec.Struct, kw_only=True, frozen=True):
    """Rao-Scott test statistics for a two-way table (mirrors ``svy.categorical.table.TableStats``)."""

    chisq: ChiSquareData
    f: FDistData | None = None


# ---------------------------------------------------------------------------
# GLM sub-structs
# ---------------------------------------------------------------------------


class GLMCoefData(msgspec.Struct, kw_only=True, frozen=True):
    """A single coefficient in a regression table (mirrors ``svy.regression.glm.GLMCoef``)."""

    term: str
    est: float
    se: float
    lci: float
    uci: float
    wald: TDistData | None = None
    wald_adj: TDistData | None = None


class GLMStatsData(msgspec.Struct, kw_only=True, frozen=True):
    """Model-level goodness-of-fit statistics (mirrors ``svy.regression.glm.GLMStats``)."""

    n: int
    wald: FDistData
    wald_adj: FDistData
    scale: float
    deviance: float
    aic: float | None = None
    bic: float | None = None
    r_squared: float | None = None
    r_squared_adj: float | None = None
    iterations: int | None = None


# ---------------------------------------------------------------------------
# Top-level structs (kind-tagged, versioned)
# ---------------------------------------------------------------------------


@_kinded("estimate")
class EstimateData(msgspec.Struct, kw_only=True, frozen=True):
    """
    Serialization struct for ``svy.estimation.estimate.Estimate``.

    Excludes: covariance, strata, singletons, domains, as_factor.
    """

    kind: Literal["estimate"] = "estimate"
    schema_version: str = SCHEMA_VERSION
    param: str
    method: str
    alpha: float
    estimates: list[ParamEstData]
    n_strata: int
    n_psus: int
    degrees_freedom: int
    where_clause: str | None = None
    q_method: str = "Linear"


@_kinded("ttest_one_group")
class TTestOneGroupData(msgspec.Struct, kw_only=True, frozen=True):
    """Serialization struct for ``svy.categorical.ttest.TTestOneGroup``."""

    kind: Literal["ttest_one_group"] = "ttest_one_group"
    schema_version: str = SCHEMA_VERSION
    y: str
    mean_h0: float = 0.0
    alternative: str = "two-sided"
    alpha: float = 0.05
    diff: list[DiffEstData] = []
    estimates: list[TtestEstData] = []
    stats: TTestStatsData | None = None


@_kinded("ttest_two_groups")
class TTestTwoGroupsData(msgspec.Struct, kw_only=True, frozen=True):
    """Serialization struct for ``svy.categorical.ttest.TTestTwoGroups``."""

    kind: Literal["ttest_two_groups"] = "ttest_two_groups"
    schema_version: str = SCHEMA_VERSION
    y: str
    groups: GroupLevelsData
    paired: bool = False
    alternative: str = "two-sided"
    alpha: float = 0.05
    diff: list[DiffEstData] = []
    estimates: list[TtestEstData] = []
    stats: TTestStatsData | None = None


@_kinded("table")
class TableData(msgspec.Struct, kw_only=True, frozen=True):
    """Serialization struct for ``svy.categorical.table.Table``."""

    kind: Literal["table"] = "table"
    schema_version: str = SCHEMA_VERSION
    type: str
    rowvar: str
    colvar: str | None = None
    alpha: float = 0.05
    estimates: list[CellEstData] = []
    stats: TableStatsData | None = None
    rowvals: list[CatValue] | None = None
    colvals: list[CatValue] | None = None


@_kinded("glm_fit")
class GLMFitData(msgspec.Struct, kw_only=True, frozen=True):
    """
    Serialization struct for ``svy.regression.glm.GLMFit``.

    Excludes: cov_matrix, term_info.
    """

    kind: Literal["glm_fit"] = "glm_fit"
    schema_version: str = SCHEMA_VERSION
    y: str
    family: str
    link: str
    stats: GLMStatsData
    coefs: list[GLMCoefData] = []
    feature_names: list[str] = []


@_kinded("glm_pred")
class GLMPredData(msgspec.Struct, kw_only=True, frozen=True):
    """Serialization struct for ``svy.regression.prediction.GLMPred``."""

    kind: Literal["glm_pred"] = "glm_pred"
    schema_version: str = SCHEMA_VERSION
    df: float
    alpha: float
    yhat: list[float]
    se: list[float]
    lci: list[float]
    uci: list[float]
    residuals: list[float] | None = None


@_kinded("describe")
class DescribeResultData(msgspec.Struct, kw_only=True, frozen=True):
    """
    Serialization struct for ``svy.core.describe.DescribeResult``.

    ``items`` stays as ``list[dict[str, Any]]`` — the DescribeItem union has
    7 variants; full typed sub-structs are deferred.  Each dict carries
    ``mtype`` (a StrEnum value) as an implicit discriminator.
    """

    kind: Literal["describe"] = "describe"
    schema_version: str = SCHEMA_VERSION
    weighted: bool
    weight_col: str | None = None
    drop_nulls: bool
    top_k: int
    percentiles: list[float]
    generated_at: str
    notes: str | None = None
    items: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

ResultData = (
    EstimateData
    | TTestOneGroupData
    | TTestTwoGroupsData
    | ChiSquareData
    | TableData
    | GLMFitData
    | GLMPredData
    | DescribeResultData
)
