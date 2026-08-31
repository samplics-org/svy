# src/svy/core/__init__.py


from svy.core.containers import ChiSquare, FDist
from svy.core.describe import DescribeResult
from svy.core.design import Design, PopSize, RepWeights
from svy.core.enumerations import (
    CaseStyle,
    DistFamily,
    LetterCase,
    LinkFunction,
    MeasurementType,
    OnePropSizeMethod,
    PopParam,
    PPSMethod,
    QuantileMethod,
    RankScoreMethod,
    SingletonHandling,
    TableType,
    TableUnits,
)
from svy.core.expr import (
    Expr,
    all_horizontal,
    any_horizontal,
    coalesce,
    col,
    cols,
    concat_str,
    lit,
    max_horizontal,
    min_horizontal,
    sum_horizontal,
    when,
)
from svy.core.functions import combine_samples
from svy.core.repwgts import (
    BootstrapWgts,
    BrrWgts,
    JackknifeWgts,
    RepWgts,
    SdrWgts,
)
from svy.core.sample import Sample
from svy.core.singleton import (
    Singleton,
    SingletonInfo,
    SingletonResult,
    SingletonSummary,
    StratumInfo,
)
from svy.core.terms import RE, Cap, Cat, Cross, Feature
from svy.core.types import DF, DT, Category, Number


__all__ = [
    # Enums
    "CaseStyle",
    "DistFamily",
    "LetterCase",
    "LinkFunction",
    "MeasurementType",
    "OnePropSizeMethod",
    "PopParam",
    "PPSMethod",
    "QuantileMethod",
    "SingletonHandling",
    "RankScoreMethod",
    "TableType",
    "TableUnits",
    # Types & Containers
    "Category",
    "ChiSquare",
    "DescribeResult",
    "Design",
    "DF",
    "DT",
    "FDist",
    "Number",
    "PopSize",
    "BootstrapWgts",
    "BrrWgts",
    "JackknifeWgts",
    "RepWeights",
    "RepWgts",
    "SdrWgts",
    "Sample",
    # Free functions
    "combine_samples",
    "Singleton",
    "SingletonInfo",
    "SingletonResult",
    "SingletonSummary",
    "StratumInfo",
    # Expressions
    "Expr",
    "col",
    "cols",
    "lit",
    "when",
    "coalesce",
    "concat_str",
    "all_horizontal",
    "any_horizontal",
    "sum_horizontal",
    "min_horizontal",
    "max_horizontal",
    # Terms
    "Cap",
    "Cat",
    "Cross",
    "Feature",
    "RE",
]
