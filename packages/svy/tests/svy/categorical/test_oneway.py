from pathlib import Path

import polars as pl
import pytest

from svy import Design, Sample, TableUnits


REL = 1e-7


@pytest.fixture
def synthetic_sample_df():
    base = Path(__file__).parent.parent.parent
    return pl.read_csv(base / "test_data/svy_synthetic_sample_07082025.csv")


@pytest.mark.parametrize(
    "design_kwargs,expected_rows",
    [
        pytest.param(
            {"wgt": "samp_wgt"},
            {
                "High School": dict(
                    est=0.306046216,
                    se=0.017194565,
                    cv=0.056182904,
                    lci=0.273379479,
                    uci=0.340785668,
                ),
                "Less than High School": dict(
                    est=0.113266841,
                    se=0.011604887,
                    cv=0.102456177,
                    lci=0.092411801,
                    uci=0.138112137,
                ),
                "None": dict(
                    est=0.012087066,
                    se=0.004812568,
                    cv=0.398158507,
                    lci=0.005517209,
                    uci=0.026273579,
                ),
                "Other Training": dict(
                    est=0.094513559,
                    se=0.011305318,
                    cv=0.119615827,
                    lci=0.074539670,
                    uci=0.119150625,
                ),
                "Postgraduate": dict(
                    est=0.184651017,
                    se=0.014353164,
                    cv=0.077731301,
                    lci=0.158126995,
                    uci=0.214490611,
                ),
                "Undergraduate": dict(
                    est=0.289435301,
                    se=0.016745896,
                    cv=0.057857130,
                    lci=0.257708644,
                    uci=0.323366314,
                ),
            },
            id="not stratification, no clustering",
        ),
        pytest.param(
            {"wgt": "samp_wgt", "stratum": "region"},
            {
                "High School": dict(
                    est=0.306046216,
                    se=0.017220171,
                    cv=0.056266570,
                    lci=0.273332366,
                    uci=0.340838954,
                ),
                "Less than High School": dict(
                    est=0.113266841,
                    se=0.011614954,
                    cv=0.102545050,
                    lci=0.092395236,
                    uci=0.138135651,
                ),
                "None": dict(
                    est=0.012087066,
                    se=0.004803467,
                    cv=0.397405538,
                    lci=0.005525405,
                    uci=0.026235417,
                ),
                "Other Training": dict(
                    est=0.094513559,
                    se=0.011274294,
                    cv=0.119287584,
                    lci=0.074588691,
                    uci=0.119076086,
                ),
                "Postgraduate": dict(
                    est=0.184651017,
                    se=0.014358616,
                    cv=0.077760830,
                    lci=0.158117441,
                    uci=0.214502703,
                ),
                "Undergraduate": dict(
                    est=0.289435301,
                    se=0.016766921,
                    cv=0.057929773,
                    lci=0.257670154,
                    uci=0.323410342,
                ),
            },
            id="not stratification, no clustering",
        ),
        pytest.param(
            {"wgt": "samp_wgt", "psu": "cluster"},
            {
                "High School": dict(
                    est=0.306046216,
                    se=0.017662760,
                    cv=0.057712723,
                    lci=0.271884532,
                    uci=0.342481266,
                ),
                "Less than High School": dict(
                    est=0.113266841,
                    se=0.010668952,
                    cv=0.094193072,
                    lci=0.093608255,
                    uci=0.136432478,
                ),
                "None": dict(
                    est=0.012087066,
                    se=0.004696838,
                    cv=0.388583774,
                    lci=0.005538232,
                    uci=0.026175915,
                ),
                "Other Training": dict(
                    est=0.094513559,
                    se=0.012045763,
                    cv=0.127450109,
                    lci=0.073008029,
                    uci=0.121523393,
                ),
                "Postgraduate": dict(
                    est=0.184651017,
                    se=0.014931911,
                    cv=0.080865578,
                    lci=0.156618632,
                    uci=0.216413309,
                ),
                "Undergraduate": dict(
                    est=0.289435301,
                    se=0.015525598,
                    cv=0.053640996,
                    lci=0.259381697,
                    uci=0.321459684,
                ),
            },
            id="stratified, no clustering",
        ),
        pytest.param(
            {"wgt": "samp_wgt", "stratum": "region", "psu": "cluster"},
            {
                "High School": dict(
                    est=0.306046216,
                    se=0.018118334,
                    cv=0.059201299,
                    lci=0.270997809,
                    uci=0.343491680,
                ),
                "Less than High School": dict(
                    est=0.113266841,
                    se=0.010807117,
                    cv=0.095412893,
                    lci=0.093354516,
                    uci=0.136785639,
                ),
                "None": dict(
                    est=0.012087066,
                    se=0.004519132,
                    cv=0.373881652,
                    lci=0.005699861,
                    uci=0.025448513,
                ),
                "Other Training": dict(
                    est=0.094513559,
                    se=0.011585436,
                    cv=0.122579620,
                    lci=0.073719049,
                    uci=0.120411237,
                ),
                "Postgraduate": dict(
                    est=0.184651017,
                    se=0.015059173,
                    cv=0.081554782,
                    lci=0.156365715,
                    uci=0.216738399,
                ),
                "Undergraduate": dict(
                    est=0.289435301,
                    se=0.015850130,
                    cv=0.054762256,
                    lci=0.258742395,
                    uci=0.322186594,
                ),
            },
            id="stratified and clustered",
        ),
    ],
)
def test_oneway_props(synthetic_sample_df, design_kwargs, expected_rows):
    design = Design(**design_kwargs)
    sample = Sample(data=synthetic_sample_df, design=design)

    # One-way proportions
    tbl = sample.categorical.tabulate("educ", units=TableUnits.PROPORTION, drop_nulls=True)

    # sanity: categories match
    actual_keys = {c.rowvar for c in (tbl.estimates or [])}
    expected_keys = set(expected_rows.keys())
    assert actual_keys == expected_keys, (
        f"Categories differ.\nActual: {sorted(actual_keys)}\nExpected: {sorted(expected_keys)}"
    )

    # per-category checks
    cell_by_row = {c.rowvar: c for c in (tbl.estimates or [])}
    for row, exp in expected_rows.items():
        cell = cell_by_row[row]
        assert cell.est == pytest.approx(exp["est"], rel=REL)
        assert cell.se == pytest.approx(exp["se"], rel=REL)
        assert cell.cv == pytest.approx(exp["cv"], rel=REL)
        assert cell.lci == pytest.approx(exp["lci"], rel=REL)
        assert cell.uci == pytest.approx(exp["uci"], rel=REL)

    # proportions should sum to ~1
    total_est = sum(c.est for c in (tbl.estimates or []))
    assert total_est == pytest.approx(1.0, rel=REL)
