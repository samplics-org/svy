# src/svy/serialize/__init__.py
"""
svy.serialize — stable, versioned serialization for svy result objects.

Public API::

    from svy.serialize import serialize, to_json, to_dict, from_json

    data = serialize(result)       # -> ResultData (kind-tagged struct)
    js   = to_json(result)         # -> bytes
    d    = to_dict(result)         # -> dict[str, Any]
    data = from_json(js)           # -> ResultData

See ``DESIGN.md`` in this directory for the full design rationale and
struct reference.
"""
from svy.serialize.serializers import from_json, serialize, to_dict, to_json
from svy.serialize.structs import (
    SCHEMA_VERSION,
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
)

__all__ = [
    # Public API
    "serialize",
    "to_json",
    "to_dict",
    "from_json",
    # Constants
    "SCHEMA_VERSION",
    # Top-level structs (kind-tagged)
    "ResultData",
    "EstimateData",
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
]
