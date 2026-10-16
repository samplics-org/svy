# src/svy/serialize/__init__.py
"""
svy.serialize — stable, versioned serialization for svy result objects.

Public API::

    from svy.serialize import serialize, to_json, to_dict, from_json, from_dict, to_design, to_polars

    data = serialize(result)       # -> ResultData (kind-tagged struct)
    js   = to_json(result)         # -> bytes
    d    = to_dict(result)         # -> dict[str, Any]
    data = from_json(js)           # -> ResultData
    data = from_dict(d)            # -> ResultData
    df   = to_polars(data)         # -> pl.DataFrame, as result.to_polars()

    js     = to_json(sample.design)        # a design, too
    design = to_design(from_json(js))      # -> the live Design back

See ``DESIGN.md`` in this directory for the full design rationale and
struct reference.
"""

from svy.serialize.serializers import (
    from_dict,
    from_json,
    serialize,
    to_design,
    to_dict,
    to_json,
)
from svy.serialize.structs import (
    DESIGN_SCHEMA_VERSION,
    SCHEMA_VERSION,
    BootstrapWgtsData,
    BrrWgtsData,
    CellEstData,
    ChiSquareData,
    CoefPartData,
    ContrastData,
    ContrastEstData,
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
    SingletonData,
    TableData,
    TableStatsData,
    TDistData,
    TermData,
    TtestEstData,
    TTestOneGroupByData,
    TTestOneGroupData,
    TTestStatsData,
    TTestTwoGroupsByData,
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
    "SingletonData",
    "PopSizeData",
    "DesignData",
    # Public API
    "serialize",
    "to_json",
    "to_dict",
    "from_json",
    "from_dict",
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
    "TTestOneGroupByData",
    "TTestTwoGroupsByData",
    "ChiSquareData",
    "TableData",
    "GLMFitData",
    "GLMPredData",
    "ContrastData",
    "ContrastEstData",
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
    "CoefPartData",
    "TermData",
    "GLMStatsData",
    "VarLabelsData",
    "LevelLabelData",
]
