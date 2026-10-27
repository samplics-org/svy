# tests/svy/test_top_level.py
"""The top level holds what users write; everything else lives in its subpackage and says so."""

from __future__ import annotations

import importlib

import pytest

import svy

from svy.errors import Code, UnknownNameError


KEEP = [
    "datasets",
    "io",
    "serialize",
    "Sample",
    "Design",
    "PopSize",
    "Singleton",
    "combine_samples",
    "RepWeights",
    "BootstrapWgts",
    "BrrWgts",
    "JackknifeWgts",
    "SdrWgts",
    "col",
    "cols",
    "lit",
    "when",
    "coalesce",
    "concat_str",
    "date",
    "all_horizontal",
    "any_horizontal",
    "sum_horizontal",
    "min_horizontal",
    "max_horizontal",
    "Cat",
    "Cross",
    "Cap",
    "RE",
    "estd",
    "SampleSize",
    "TargetMean",
    "TargetProp",
    "TargetTwoMeans",
    "TargetTwoProps",
    "Threshold",
    "TrimConfig",
    "SvyError",
    "SvyWarning",
]

IO = [
    f"{verb}_{fmt}"
    for verb in ("read", "write")
    for fmt in ("csv", "parquet", "sas", "xpt", "spss", "sav", "stata", "dta")
] + [
    f"create_from_{fmt}" for fmt in ("csv", "parquet", "sas", "xpt", "spss", "sav", "stata", "dta")
]
IO += ["scan_csv", "scan_parquet"]

MOVED = {
    **dict.fromkeys(IO, "svy.io"),
    **dict.fromkeys(
        ["Table", "TableStats", "CellEst", "TTestOneGroup", "TTestTwoGroups", "Categorical"],
        "svy.categorical",
    ),
    **dict.fromkeys(
        [
            "Estimate",
            "EstimateList",
            "ParamEst",
            "Contrast",
            "ContrastEst",
            "ContrastExpr",
            "Estimation",
        ],
        "svy.estimation",
    ),
    **dict.fromkeys(["GLM", "GLMFit", "GLMPred"], "svy.regression"),
    "Selection": "svy.selection",
    "TrimResult": "svy.weighting",
    **dict.fromkeys(
        [
            "CaseStyle",
            "DistFamily",
            "LetterCase",
            "LinkFunction",
            "MeasurementType",
            "OnePropSizeMethod",
            "PopParam",
            "PPSMethod",
            "QuantileMethod",
            "RankScoreMethod",
            "SingletonMethod",
            "TableType",
            "TableUnits",
        ],
        "svy.enums",
    ),
    **dict.fromkeys(
        ["Category", "DF", "DT", "Number", "Target", "RepWgts", "Feature", "RandomState"],
        "svy.typing",
    ),
    **dict.fromkeys(
        ["DatasetError", "LabelError", "ModelError", "SerializationError"], "svy.errors"
    ),
    **dict.fromkeys(
        ["enable_logging", "enable_debug", "temporary_log_level", "seed_from_random_state"],
        "svy.utils",
    ),
    "register_sample_accessor": "svy.extensions",
    **dict.fromkeys(
        ["ChiSquare", "FDist", "DescribeResult", "SingletonInfo", "StratumInfo", "WgtAdjustment"],
        "svy.core",
    ),
}


def test_top_level_is_the_keep_list():
    assert sorted(svy.__all__) == sorted(KEEP)
    assert len(svy.__all__) == len(set(svy.__all__))


@pytest.mark.parametrize("name", KEEP)
def test_kept_name_resolves(name):
    assert getattr(svy, name) is not None


@pytest.mark.parametrize("name, home", sorted(MOVED.items()))
def test_moved_name_says_where_it_went(name, home):
    with pytest.raises(UnknownNameError) as err:
        getattr(svy, name)
    assert isinstance(err.value, AttributeError)
    assert err.value.code == Code.NAME_MOVED
    assert err.value.detail == f"svy.{name} moved to {home}.{name} in 2026.0."
    assert err.value.expected == f"{home}.{name}"
    assert not hasattr(svy, name)
    assert name not in dir(svy)


@pytest.mark.parametrize("name, home", sorted(MOVED.items()))
def test_moved_name_is_exported_from_its_home(name, home):
    module = importlib.import_module(home)
    assert name in module.__all__
    assert getattr(module, name) is not None


def test_from_import_of_a_moved_name_fails():
    with pytest.raises(ImportError, match="read_sas"):
        from svy import read_sas  # noqa: F401


def test_enums_and_typing_reexport_the_definitions():
    from svy.core import enumerations, repwgts, terms, types
    from svy.size import types as size_types

    for name in svy.enums.__all__:
        assert getattr(svy.enums, name) is getattr(enumerations, name)
    assert svy.typing.Number is types.Number and svy.typing.Category is types.Category
    assert svy.typing.RepWgts is repwgts.RepWgts and svy.typing.Feature is terms.Feature
    assert svy.typing.Target is size_types.Target


def test_svy_typing_does_not_shadow_the_standard_library():
    import typing

    assert typing.__name__ == "typing" and hasattr(typing, "Any")
    assert svy.typing.__name__ == "svy.typing"


def test_subpackages_are_reachable():
    for name in ("categorical", "estimation", "regression", "weighting", "selection", "errors"):
        assert importlib.import_module(f"svy.{name}") is getattr(svy, name)


def test_unknown_names_still_guide():
    with pytest.raises(UnknownNameError) as err:
        svy.Survey
    assert err.value.code == Code.UNKNOWN_NAME and "svy.Sample" in err.value.detail


def test_to_code_writes_the_io_spelling():
    import polars as pl

    sample = svy.Sample(pl.DataFrame({"y": [1.0, 2.0], "w": [1.0, 1.0]}), svy.Design(wgt="w"))
    code = sample.to_code(data="survey.parquet")
    assert "svy.io.read_parquet('survey.parquet')" in code
    assert "svy.read_parquet" not in code
