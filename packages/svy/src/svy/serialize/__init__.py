# src/svy/serialize/__init__.py
"""
svy.serialize — stable, versioned serialization for svy result objects.

Public API::

    from svy.serialize import serialize, to_json, to_dict, from_json, to_design, to_polars

    data = serialize(result)       # -> ResultData (kind-tagged struct)
    js   = to_json(result)         # -> bytes
    d    = to_dict(result)         # -> dict[str, Any]
    data = from_json(js)           # -> ResultData
    df   = to_polars(data)         # -> pl.DataFrame, as result.to_polars()

    js     = to_json(sample.design)        # a design, too
    design = to_design(from_json(js))      # -> the live Design back

See ``DESIGN.md`` in this directory for the full design rationale and
struct reference.
"""

from svy.serialize.serializers import from_json, serialize, to_design, to_dict, to_json
from svy.serialize.structs import (
    DESIGN_SCHEMA_VERSION,
    SCHEMA_VERSION,
    BootstrapWgtsData,
    BrrWgtsData,
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
    JackknifeWgtsData,
    LevelLabelData,
    ParamEstData,
    PopSizeData,
    RepWgtsData,
    ResultData,
    SdrWgtsData,
    SingletonSpecData,
    TableData,
    TableStatsData,
    TDistData,
    TtestEstData,
    TTestOneGroupData,
    TTestStatsData,
    TTestTwoGroupsData,
    VarLabelsData,
    WgtAdjustmentData,
)
from svy.serialize.tables import to_polars


__all__ = [
    "DESIGN_SCHEMA_VERSION",
    "SdrWgtsData",
    "BrrWgtsData",
    "JackknifeWgtsData",
    "BootstrapWgtsData",
    "RepWgtsData",
    "WgtAdjustmentData",
    "SingletonSpecData",
    "PopSizeData",
    "DesignData",
    # Public API
    "serialize",
    "to_json",
    "to_dict",
    "from_json",
    "to_polars",
    "to_design",
    # Constants
    "SCHEMA_VERSION",
    # Top-level structs (kind-tagged)
    "ResultData",
    "EstimateData",
    "EstimateListData",
    "TTestOneGroupData",
    "TTestTwoGroupsData",
    "ChiSquareData",
    "TableData",
    "GLMFitData",
    "GLMPredData",
    "DescribeResultData",
    # Sub-structs
    "ParamEstData",
    "DiffEstData",
    "TtestEstData",
    "TTestStatsData",
    "GroupLevelsData",
    "CellEstData",
    "TableStatsData",
    "FDistData",
    "TDistData",
    "GLMCoefData",
    "GLMStatsData",
    "VarLabelsData",
    "LevelLabelData",
]
