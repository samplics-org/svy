# tests/svy/estimation/test_replication.py
import pytest

from svy import EstimationMethod, Sample

from . import data_golden as golden


# Tolerance for floating point comparisons
TOL = 1e-4

# SCENARIOS Configuration:
# (Method, CSV_Filename, Prefix, N_Reps, DF, PSU_Column, Golden_Data_Dict)
SCENARIOS = [
    (EstimationMethod.BRR, "fake_survey_brr_24122025.csv", "brr_", 8, 7, "psu", golden.BRR),
    (
        EstimationMethod.BOOTSTRAP,
        "fake_survey_bootstrap_25122025.csv",
        "bs_",
        20,
        None,
        "psu",
        golden.BOOTSTRAP,
    ),
    (
        EstimationMethod.JACKKNIFE,
        "fake_survey_jackknife_25122025.csv",
        "jk_",
        8,
        7,
        "psu",
        golden.JACKKNIFE,
    ),
]


def assert_est(result, expected):
    """Helper to check est, se, lci, uci against expected dict."""
    assert result.est == pytest.approx(expected["est"], rel=TOL)
    assert result.se == pytest.approx(expected["se"], rel=TOL)
    assert result.lci == pytest.approx(expected["lci"], rel=TOL)
    assert result.uci == pytest.approx(expected["uci"], rel=TOL)


# ==============================================================================
# TESTS
# ==============================================================================


@pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", SCENARIOS)
def test_mean_overall(load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold):
    data = load_survey_data(csv)
    design = make_design(method, prefix, reps, df, psu_col)
    sample = Sample(data, design)

    est = sample.estimation.mean(y="income", method="replication", drop_nulls=True)
    assert_est(est.estimates[0], gold["mean_overall"])


@pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", SCENARIOS)
def test_mean_domain(load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold):
    data = load_survey_data(csv)
    design = make_design(method, prefix, reps, df, psu_col)
    sample = Sample(data, design)

    est = sample.estimation.mean(y="income", by="educ", method="replication", drop_nulls=True)

    for res in est.estimates:
        domain = res.by_level[0]
        assert_est(res, gold["mean_educ"][domain])


@pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", SCENARIOS)
def test_total_overall(
    load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold
):
    data = load_survey_data(csv)
    design = make_design(method, prefix, reps, df, psu_col)
    sample = Sample(data, design)

    est = sample.estimation.total(y="low_income", method="replication", drop_nulls=True)
    assert_est(est.estimates[0], gold["total_overall"])


@pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", SCENARIOS)
def test_total_domain(load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold):
    data = load_survey_data(csv)
    design = make_design(method, prefix, reps, df, psu_col)
    sample = Sample(data, design)

    est = sample.estimation.total(y="low_income", by="educ", method="replication", drop_nulls=True)

    for res in est.estimates:
        domain = res.by_level[0]
        assert_est(res, gold["total_educ"][domain])


@pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", SCENARIOS)
def test_prop_overall(load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold):
    data = load_survey_data(csv)
    design = make_design(method, prefix, reps, df, psu_col)
    sample = Sample(data, design)

    est = sample.estimation.prop(y="low_income", method="replication", drop_nulls=True)

    for res in est.estimates:
        level = res.y_level
        assert_est(res, gold["prop_overall"][level])


@pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", SCENARIOS)
def test_prop_domain(load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold):
    data = load_survey_data(csv)
    design = make_design(method, prefix, reps, df, psu_col)
    sample = Sample(data, design)

    est = sample.estimation.prop(y="low_income", by="educ", method="replication", drop_nulls=True)

    for res in est.estimates:
        domain = res.by_level[0]
        level = res.y_level
        assert_est(res, gold["prop_educ"][domain][level])


@pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", SCENARIOS)
def test_ratio_overall(
    load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold
):
    data = load_survey_data(csv)
    design = make_design(method, prefix, reps, df, psu_col)
    sample = Sample(data, design)

    est = sample.estimation.ratio(y="income", x="hh_size", method="replication", drop_nulls=True)
    assert_est(est.estimates[0], gold["ratio_overall"])


@pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", SCENARIOS)
def test_ratio_domain(load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold):
    data = load_survey_data(csv)
    design = make_design(method, prefix, reps, df, psu_col)
    sample = Sample(data, design)

    est = sample.estimation.ratio(
        y="income", x="hh_size", by="educ", method="replication", drop_nulls=True
    )

    for res in est.estimates:
        domain = res.by_level[0]
        assert_est(res, gold["ratio_educ"][domain])
