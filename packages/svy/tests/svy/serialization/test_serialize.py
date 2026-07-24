# tests/svy/test_serialize.py
"""
Tests for svy.serialize — typed result serialization.

Covers:
- Round-trip (serialize → to_json → from_json) for every result kind
- Exhaustiveness (every public result type has a registered serializer)
- GLM wrapper delegation (GLM → GLMFit via .fitted)
- Type conversion (numpy → list, StrEnum → str, tuple → list)
- Golden-file fixtures (stable JSON output per result kind)
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import msgspec
import numpy as np
import pytest

from svy import (
    CellEst,
    ChiSquare,
    DescribeResult,
    Design,
    Estimate,
    FDist,
    GLMFit,
    GLMPred,
    ParamEst,
    PopParam,
    Sample,
    Table,
    TableType,
    TTestOneGroup,
    TTestTwoGroups,
)
from svy.core.containers import TDist
from svy.core.describe import DescribeContinuous
from svy.core.enumerations import EstimationMethod, MeasurementType, QuantileMethod
from svy.categorical.ttest import DiffEst, GroupLevels, TtestEst, TTestStats
from svy.regression.glm import GLMCoef, GLMStats
from svy.serialize import from_json, serialize, to_dict, to_json
from svy.serialize.serializers import _SERIALIZERS
from svy.serialize.structs import SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers — construct minimal result objects
# ---------------------------------------------------------------------------


def _make_estimate() -> Estimate:
    est = Estimate(PopParam.MEAN, alpha=0.05)
    est.method = EstimationMethod.TAYLOR
    est.n_strata = 10
    est.n_psus = 20
    est.q_method = QuantileMethod.LINEAR
    est.where_clause = None
    est.estimates = [
        ParamEst(
            y="income",
            est=50000.0,
            se=1000.0,
            cv=0.02,
            lci=48000.0,
            uci=52000.0,
            by=("region",),
            by_level=("North",),
            deff=1.5,
        ),
    ]
    return est


def _make_ttest_one_group() -> TTestOneGroup:
    return TTestOneGroup(
        y="income",
        mean_h0=45000.0,
        alternative="two-sided",
        alpha=0.05,
        diff=[
            DiffEst(y="income", diff=5000.0, se=1000.0, lci=3000.0, uci=7000.0),
        ],
        estimates=[
            TtestEst(
                by=None,
                by_level=None,
                group=None,
                group_level=None,
                y="income",
                y_level=None,
                est=50000.0,
                se=1000.0,
                cv=0.02,
                lci=48000.0,
                uci=52000.0,
            ),
        ],
        stats=TTestStats(t=5.0, df=9.0, p_value=0.001),
    )


def _make_ttest_two_groups() -> TTestTwoGroups:
    return TTestTwoGroups(
        y="income",
        groups=GroupLevels(var="urban", levels=("Rural", "Urban")),
        paired=False,
        alternative="two-sided",
        alpha=0.05,
        diff=[
            DiffEst(y="income", diff=3000.0, se=800.0, lci=1400.0, uci=4600.0),
        ],
        estimates=[
            TtestEst(
                by=None,
                by_level=None,
                group="urban",
                group_level="Rural",
                y="income",
                y_level=None,
                est=48000.0,
                se=900.0,
                cv=0.01875,
                lci=46200.0,
                uci=49800.0,
            ),
        ],
        stats=TTestStats(t=3.75, df=18.0, p_value=0.0015),
    )


def _make_chi_square() -> ChiSquare:
    return ChiSquare(df=4, value=9.488, p_value=0.050)


def _make_table() -> Table:
    return Table(
        type=TableType.TWO_WAY,
        rowvar="education",
        colvar="region",
        alpha=0.05,
        estimates=[
            CellEst(rowvar="education", colvar="region", est=0.35, se=0.02, cv=0.057, lci=0.31, uci=0.39),
        ],
        stats=None,
        rowvals=["Low", "Med", "High"],
        colvals=["North", "South"],
    )


def _make_glm_fit() -> GLMFit:
    return GLMFit(
        y="income",
        family="gaussian",
        link="identity",
        stats=GLMStats(
            n=500,
            wald=FDist(df_num=3, df_den=496, value=45.2, p_value=0.001),
            wald_adj=FDist(df_num=3, df_den=450, value=42.1, p_value=0.001),
            scale=1.2e8,
            deviance=5.9e10,
            aic=6020.5,
            bic=6040.3,
            r_squared=0.215,
            r_squared_adj=0.210,
            iterations=4,
        ),
        coefs=[
            GLMCoef(
                term="intercept",
                est=32000.0,
                se=2000.0,
                lci=28000.0,
                uci=36000.0,
                wald=TDist(df=496, value=16.0, p_value=0.0001),
                wald_adj=TDist(df=450, value=15.5, p_value=0.0001),
            ),
        ],
        feature_names=["age", "education"],
    )


def _make_glm_pred() -> GLMPred:
    return GLMPred(
        yhat=np.array([50000.0, 52000.0, 48000.0]),
        se=np.array([1000.0, 1100.0, 900.0]),
        lci=np.array([48000.0, 49800.0, 46200.0]),
        uci=np.array([52000.0, 54200.0, 49800.0]),
        df=496.0,
        alpha=0.05,
        residuals=np.array([200.0, -300.0, 150.0]),
    )


def _make_describe_result() -> DescribeResult:
    return DescribeResult(
        items=(
            DescribeContinuous(
                name="income",
                mtype=MeasurementType.CONTINUOUS,
                n=500,
                n_missing=5,
                weighted=True,
                drop_nulls=False,
                mean=50000.0,
                std=10000.0,
                var=1e8,
                min=20000.0,
                p25=43000.0,
                p50=50000.0,
                p75=57000.0,
                max=90000.0,
                sum=2.5e7,
            ),
        ),
        weighted=True,
        weight_col="wgt",
        drop_nulls=False,
        top_k=10,
        percentiles=(0.25, 0.5, 0.75),
        generated_at=dt.datetime(2026, 7, 10, 12, 0, 0),
        notes="test describe",
    )


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expected_kind",
    [
        (_make_estimate, "estimate"),
        (_make_ttest_one_group, "ttest_one_group"),
        (_make_ttest_two_groups, "ttest_two_groups"),
        (_make_chi_square, "chi_square"),
        (_make_table, "table"),
        (_make_glm_fit, "glm_fit"),
        (_make_glm_pred, "glm_pred"),
        (_make_describe_result, "describe"),
    ],
    ids=[
        "estimate",
        "ttest_one_group",
        "ttest_two_groups",
        "chi_square",
        "table",
        "glm_fit",
        "glm_pred",
        "describe",
    ],
)
def test_roundtrip_kind_and_schema_version(factory, expected_kind):
    """serialize() produces the correct kind tag and schema_version."""
    result = factory()
    data = serialize(result)
    assert data.kind == expected_kind
    assert data.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize(
    "factory",
    [
        _make_estimate,
        _make_ttest_one_group,
        _make_ttest_two_groups,
        _make_chi_square,
        _make_table,
        _make_glm_fit,
        _make_glm_pred,
        _make_describe_result,
    ],
    ids=[
        "estimate",
        "ttest_one_group",
        "ttest_two_groups",
        "chi_square",
        "table",
        "glm_fit",
        "glm_pred",
        "describe",
    ],
)
def test_roundtrip_json(factory):
    """to_json → from_json preserves all data."""
    result = factory()
    js = to_json(result)
    data = from_json(js)
    # Re-serialize the decoded struct and compare JSON — confirms structural equality
    js2 = msgspec.json.encode(data)
    assert json.loads(js) == json.loads(js2)


def test_estimate_fields():
    """Estimate serialization captures the right fields with correct conversions."""
    result = _make_estimate()
    data = serialize(result)
    assert data.param == "Mean"
    assert data.method == "Taylor"
    assert data.q_method == "Linear"
    assert data.alpha == 0.05
    assert data.n_strata == 10
    assert data.n_psus == 20
    assert data.where_clause is None
    assert len(data.estimates) == 1
    pe = data.estimates[0]
    assert pe.y == "income"
    assert pe.est == 50000.0
    assert pe.deff == 1.5
    assert pe.by == ["region"]
    assert pe.by_level == ["North"]


def test_ttest_one_group_fields():
    """TTestOneGroup serialization captures stats and estimates."""
    result = _make_ttest_one_group()
    data = serialize(result)
    assert data.y == "income"
    assert data.mean_h0 == 45000.0
    assert data.stats is not None
    assert data.stats.t == 5.0
    assert data.stats.p_value == 0.001
    assert len(data.diff) == 1
    assert data.diff[0].diff == 5000.0


def test_ttest_two_groups_fields():
    """TTestTwoGroups serialization captures groups and paired flag."""
    result = _make_ttest_two_groups()
    data = serialize(result)
    assert data.y == "income"
    assert data.groups.var == "urban"
    assert data.groups.levels == ["Rural", "Urban"]
    assert data.paired is False
    assert data.stats is not None
    assert data.stats.t == 3.75


def test_table_fields():
    """Table serialization captures type, estimates, rowvals/colvals."""
    result = _make_table()
    data = serialize(result)
    assert data.type == "Two-Way"
    assert data.rowvar == "education"
    assert data.colvar == "region"
    assert data.alpha == 0.05
    assert len(data.estimates) == 1
    assert data.estimates[0].est == 0.35
    assert data.rowvals == ["Low", "Med", "High"]
    assert data.colvals == ["North", "South"]


def test_glm_fit_fields():
    """GLMFit serialization excludes cov_matrix and term_info."""
    result = _make_glm_fit()
    data = serialize(result)
    assert data.y == "income"
    assert data.family == "gaussian"
    assert data.link == "identity"
    assert data.stats.n == 500
    assert data.stats.aic == 6020.5
    assert data.stats.r_squared == 0.215
    assert len(data.coefs) == 1
    assert data.coefs[0].term == "intercept"
    assert data.coefs[0].wald is not None
    assert data.coefs[0].wald.p_value == 0.0001
    assert data.feature_names == ["age", "education"]


def test_glm_pred_fields():
    """GLMPred serialization converts numpy arrays to lists."""
    result = _make_glm_pred()
    data = serialize(result)
    assert data.df == 496.0
    assert data.alpha == 0.05
    assert data.yhat == [50000.0, 52000.0, 48000.0]
    assert data.se == [1000.0, 1100.0, 900.0]
    assert data.residuals == [200.0, -300.0, 150.0]


def test_describe_result_fields():
    """DescribeResult serialization converts datetime to ISO string."""
    result = _make_describe_result()
    data = serialize(result)
    assert data.weighted is True
    assert data.weight_col == "wgt"
    assert data.drop_nulls is False
    assert data.top_k == 10
    assert data.percentiles == [0.25, 0.5, 0.75]
    assert data.generated_at == "2026-07-10T12:00:00"
    assert data.notes == "test describe"
    assert len(data.items) == 1
    assert data.items[0]["name"] == "income"
    assert data.items[0]["mtype"] == "Numerical Continuous"


# ---------------------------------------------------------------------------
# Exhaustiveness test
# ---------------------------------------------------------------------------


def test_all_result_types_have_serializers():
    """Every public result type must have a registered serializer."""
    public_result_types = {
        Estimate,
        TTestOneGroup,
        TTestTwoGroups,
        ChiSquare,
        Table,
        GLMFit,
        GLMPred,
        DescribeResult,
    }
    registered = set(_SERIALIZERS.keys())
    missing = public_result_types - registered
    assert not missing, f"No serializer registered for: {sorted(t.__name__ for t in missing)}"


# ---------------------------------------------------------------------------
# GLM wrapper test
# ---------------------------------------------------------------------------


def test_serialize_glm_wrapper_delegates_to_fitted():
    """serialize() on a GLM object delegates to GLMFit via .fitted."""
    from svy.regression.base import GLM

    glm_fit = _make_glm_fit()
    glm = GLM.__new__(GLM)
    glm.fitted = glm_fit

    data = serialize(glm)
    assert data.kind == "glm_fit"
    assert data.y == "income"
    assert data.family == "gaussian"


def test_serialize_unfitted_glm_raises():
    """serialize() on an unfitted GLM raises ValueError."""
    from svy.regression.base import GLM

    glm = GLM.__new__(GLM)
    glm.fitted = None

    with pytest.raises(ValueError, match="unfitted"):
        serialize(glm)


# ---------------------------------------------------------------------------
# Type conversion tests
# ---------------------------------------------------------------------------


def test_numpy_array_to_list():
    """numpy arrays are converted to Python lists."""
    result = _make_glm_pred()
    data = serialize(result)
    assert isinstance(data.yhat, list)
    assert all(isinstance(x, float) for x in data.yhat)


def test_strenum_to_string():
    """StrEnum values are converted to their string .value."""
    result = _make_estimate()
    data = serialize(result)
    assert isinstance(data.param, str)
    assert data.param == "Mean"
    assert isinstance(data.method, str)
    assert data.method == "Taylor"


def test_tuple_to_list():
    """Tuples are converted to lists."""
    result = _make_estimate()
    data = serialize(result)
    assert isinstance(data.estimates[0].by, list)
    assert data.estimates[0].by == ["region"]


def test_to_dict_returns_json_safe():
    """to_dict() produces a dict with no numpy types."""
    result = _make_glm_pred()
    d = to_dict(result)
    # The dict should be JSON-serializable
    json.dumps(d)
    assert d["kind"] == "glm_pred"
    assert isinstance(d["yhat"], list)


def test_unregistered_type_raises():
    """serialize() on an unregistered type raises TypeError."""
    with pytest.raises(TypeError, match="No serializer"):
        serialize("not a result object")


# ---------------------------------------------------------------------------
# Golden-file tests
# ---------------------------------------------------------------------------

GOLDEN_DIR = Path(__file__).parent / "golden"

GOLDEN_FACTORIES = {
    "estimate.json": _make_estimate,
    "ttest_one_group.json": _make_ttest_one_group,
    "ttest_two_groups.json": _make_ttest_two_groups,
    "chi_square.json": _make_chi_square,
    "table.json": _make_table,
    "glm_fit.json": _make_glm_fit,
    "glm_pred.json": _make_glm_pred,
    "describe.json": _make_describe_result,
}


@pytest.mark.parametrize("filename", sorted(GOLDEN_FACTORIES.keys()))
def test_golden_file(filename):
    """Serialized JSON matches the golden fixture."""
    factory = GOLDEN_FACTORIES[filename]
    result = factory()
    js = to_json(result)
    actual = json.loads(js)

    golden_path = GOLDEN_DIR / filename

    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"Updated golden file: {filename}")

    if not golden_path.exists():
        pytest.skip(f"Golden file not found: {filename} (run with UPDATE_GOLDEN=1 to create)")

    expected = json.loads(golden_path.read_text())
    assert actual == expected, f"Serialized output does not match {filename}"
