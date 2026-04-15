from pathlib import Path

import polars as pl
import pytest

from svy import Design, Sample


# Existing fixture (kept here for completeness)
@pytest.fixture
def synthetic_sample_df():
    """Load and prepare the synthetic sample dataset."""
    BASE_DIR = Path(__file__).parent.parent.parent
    df = pl.read_csv(BASE_DIR / "test_data/svy_synthetic_sample_07082025.csv")
    return df


# --- LIST INPUTS: STRATUM / BY / PSU / SSU ----------------------------------


def test_select_unstratified_srs(synthetic_sample_df):
    sample = Sample(data=synthetic_sample_df)
    samp = sample.sampling.srs(n=10, by=["region", "sex"], drop_nulls=True)
    assert samp is not None


def test_select_stratified_srs(synthetic_sample_df):
    design = Design(stratum=("region", "sex"))
    sample = Sample(data=synthetic_sample_df, design=design)
    samp = sample.sampling.srs(n=10, drop_nulls=True)
    assert samp is not None


def test_select_stratified_srs_by(synthetic_sample_df):
    design = Design(stratum="region")
    sample = Sample(data=synthetic_sample_df, design=design)
    samp = sample.sampling.srs(n=4, by="sex", drop_nulls=True)
    assert samp is not None


def test_select_stratified_pps(synthetic_sample_df):
    cluster_sample = synthetic_sample_df.group_by(["region", "educ", "sex", "cluster"]).agg(
        mos=pl.sum("samp_wgt")
    )
    design = Design(stratum=["region", "sex"], mos="mos")
    sample = Sample(data=cluster_sample, design=design)
    samp = sample.sampling.pps_sys(n=10, drop_nulls=True)
    assert samp is not None


# def test_mean_estimation_stratum_list_multi(synthetic_sample_df):
#     """stratum accepts a list of multiple columns (existing test pattern)."""
#     design = Design(stratum=["region", "sex"])
#     sample = Sample(data=synthetic_sample_df, design=design)

#     est = sample.estimation.mean(y="income", deff=True, drop_nulls=True)
#     assert est is not None


# def test_mean_estimation_psu_list(synthetic_sample_df):
#     """psu accepts a list (len 1)."""
#     design = Design(psu=["cluster"])  # list form instead of "cluster"
#     sample = Sample(data=synthetic_sample_df, design=design)

#     est = sample.estimation.mean(y="income", deff=True, drop_nulls=True)
#     assert est is not None


# def test_mean_estimation_psu_list_multi(synthetic_sample_df):
#     """psu accepts a multi-column list (hierarchical psu)."""
#     # Both columns exist in the data
#     design = Design(psu=["region", "cluster"])
#     sample = Sample(data=synthetic_sample_df, design=design)

#     est = sample.estimation.mean(y="income", deff=True, drop_nulls=True)
#     assert est is not None


# def test_mean_estimation_ssu_list(synthetic_sample_df):
#     """ssu accepts a list (len 1)."""
#     design = Design(ssu=["id"])  # list form instead of "id"
#     sample = Sample(data=synthetic_sample_df, design=design)

#     est = sample.estimation.mean(y="income", deff=True, drop_nulls=True)
#     assert est is not None


# def test_mean_estimation_combined_lists(synthetic_sample_df):
#     """All of by/stratum/psu/ssu can be passed as lists together."""
#     design = Design(
#         stratum=["region", "sex"],
#         psu=["cluster"],
#         ssu=["id"],
#     )
#     sample = Sample(data=synthetic_sample_df, design=design)

#     est = sample.estimation.mean(
#         y="income",
#         by=["region", "educ"],
#         deff=True,
#         drop_nulls=True,
#     )
#     assert est is not None
