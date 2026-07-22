from pathlib import Path

import polars as pl
import pytest

from svy import Design, Sample


REL = 1e-7


@pytest.fixture
def synthetic_sample_df():
    base = Path(__file__).parent.parent.parent
    return pl.read_csv(base / "test_data/svy_synthetic_sample_07082025.csv")


# =============================================================================
# One-sample t-tests (validated against R survey::svyttest)
# R code (domain semantics — drop_nulls=True zero-weights null-y rows,
# matching R's subset()/na.rm rather than complete-case filtering):
#   d <- d[!is.na(d$samp_wgt), ]
#   svyttest(I(income - 49000) ~ 0, subset(design, !is.na(income)))
# =============================================================================


@pytest.mark.parametrize(
    "design_kwargs, expected",
    [
        # Case 1: Weights Only (Simple Random Sample assumption)
        pytest.param(
            {"wgt": "samp_wgt"},
            {
                "y": "income",
                "mean_h0": 49000.0,
                "diff": {
                    "diff": 1074.67752319406,
                    "lci": -23.6138630618943,
                    "uci": 2172.96890945002,
                },
                "stats": {
                    "df": 978,
                    "t": 1.92020006411387,
                    "p_value": 0.0551232181108886,
                },
            },
            id="wgt_only",
        ),
        # Case 2: Stratified (Weights + Stratum)
        pytest.param(
            {"wgt": "samp_wgt", "stratum": "region"},
            {
                "y": "income",
                "mean_h0": 49000.0,
                "diff": {
                    "diff": 1074.67752319406,
                    "lci": -22.9479087984971,
                    "uci": 2172.30295518662,
                },
                "stats": {
                    "df": 975,
                    "t": 1.92137241836393,
                    "p_value": 0.0549760544033834,
                },
            },
            id="stratified",
        ),
        # Case 3: Clustered (Weights + PSU)
        pytest.param(
            {"wgt": "samp_wgt", "psu": "cluster"},
            {
                "y": "income",
                "mean_h0": 49000.0,
                "diff": {
                    "diff": 1074.67752319406,
                    "lci": 64.8139187911888,
                    "uci": 2084.54112759693,
                },
                "stats": {
                    "df": 58,
                    "t": 2.13018944203603,
                    "p_value": 0.0374092167685355,
                },
            },
            id="clustered",
        ),
        # Case 4: Stratified & Clustered
        pytest.param(
            {"wgt": "samp_wgt", "stratum": "region", "psu": "cluster"},
            {
                "y": "income",
                "mean_h0": 49000.0,
                "diff": {
                    "diff": 1074.67752319406,
                    "lci": 82.1126829647371,
                    "uci": 2067.24236342339,
                },
                "stats": {
                    "df": 55,
                    "t": 2.1698349536314,
                    "p_value": 0.0343564602837252,
                },
            },
            id="stratified_clustered",
        ),
    ],
)
def test_ttest_one_sample(synthetic_sample_df, design_kwargs, expected):
    """Test one-sample t-test against R survey::svyttest results."""
    y = "income"
    mean_h0 = 49000.0
    alpha = 0.05

    design = Design(**design_kwargs)
    sample = Sample(data=synthetic_sample_df, design=design)

    result = sample.categorical.ttest(y, mean_h0=mean_h0, drop_nulls=True, alpha=alpha)

    # Basic assertions
    assert result.y == y
    assert result.mean_h0 == mean_h0
    assert result.alpha == alpha
    assert result.alternative == "two-sided"
    assert result.diff and len(result.diff) >= 1
    assert result.estimates and len(result.estimates) >= 1
    assert result.stats is not None

    # Diff estimates (what we're testing)
    diff0 = result.diff[0]
    exp_diff = expected["diff"]
    assert diff0.diff == pytest.approx(exp_diff["diff"], rel=REL)
    assert diff0.lci == pytest.approx(exp_diff["lci"], rel=REL)
    assert diff0.uci == pytest.approx(exp_diff["uci"], rel=REL)

    # Test statistics
    exp_stats = expected["stats"]
    assert result.stats.df == pytest.approx(exp_stats["df"], abs=0)
    assert result.stats.t == pytest.approx(exp_stats["t"], rel=REL)
    assert result.stats.p_value == pytest.approx(exp_stats["p_value"], rel=1e-3)


# =============================================================================
# Two-sample t-tests (validated against R survey::svyttest)
# R code (domain semantics, as above):
#   svyttest(income ~ sex, subset(design, !is.na(income) & !is.na(sex)))
# =============================================================================


@pytest.mark.parametrize(
    "design_kwargs, expected",
    [
        # Case 1: Weights Only
        pytest.param(
            {"wgt": "samp_wgt"},
            {
                "y": "income",
                "group": "sex",
                "diff": {
                    "diff": 613.853666827849,
                    "lci": -1602.57536792248,
                    "uci": 2830.28270157818,
                },
                "stats": {
                    "df": 968,
                    "t": 0.543503661878178,
                    "p_value": 0.586908209384493,
                },
            },
            id="wgt_only",
        ),
        # Case 2: Stratified (Weights + Stratum)
        pytest.param(
            {"wgt": "samp_wgt", "stratum": "region"},
            {
                "y": "income",
                "group": "sex",
                "diff": {
                    "diff": 613.853666827849,
                    "lci": -1605.50597009859,
                    "uci": 2833.21330375429,
                },
                "stats": {
                    "df": 965,
                    "t": 0.542788092996043,
                    "p_value": 0.587401048592969,
                },
            },
            id="stratified",
        ),
        # Case 4: Stratified & Clustered (skipping Case 3 - R code had error)
        pytest.param(
            {"wgt": "samp_wgt", "stratum": "region", "psu": "cluster"},
            {
                "y": "income",
                "group": "sex",
                "diff": {
                    "diff": 613.853666827849,
                    "lci": -1944.02476681743,
                    "uci": 3171.73210047313,
                },
                "stats": {
                    "df": 55,
                    "t": 0.480941636055822,
                    "p_value": 0.632465755313356,
                },
            },
            id="stratified_clustered",
        ),
    ],
)
def test_ttest_two_sample(synthetic_sample_df, design_kwargs, expected):
    """Test two-sample t-test against R survey::svyttest results."""
    y = "income"
    group = "sex"
    alpha = 0.05

    design = Design(**design_kwargs)
    sample = Sample(data=synthetic_sample_df, design=design)

    result = sample.categorical.ttest(y, group=group, drop_nulls=True, alpha=alpha)

    # Basic assertions
    assert result.y == y
    assert result.groups.var == group
    assert result.alpha == alpha
    assert result.alternative == "two-sided"
    assert result.paired is False
    assert result.diff and len(result.diff) >= 1
    assert result.estimates and len(result.estimates) == 2  # Two groups
    assert result.stats is not None

    # Diff estimates
    diff0 = result.diff[0]
    exp_diff = expected["diff"]
    # Note: sign may differ based on group ordering, so check absolute values
    assert abs(diff0.diff) == pytest.approx(abs(exp_diff["diff"]), rel=REL)

    # Test statistics
    exp_stats = expected["stats"]
    assert result.stats.df == pytest.approx(exp_stats["df"], abs=0)
    # t-statistic sign may differ based on group ordering
    assert abs(result.stats.t) == pytest.approx(abs(exp_stats["t"]), rel=REL)
    # p-value should match regardless of sign
    assert result.stats.p_value == pytest.approx(exp_stats["p_value"], rel=1e-3)


# =============================================================================
# Test to_polars() output format
# =============================================================================


def test_ttest_one_sample_to_polars(synthetic_sample_df):
    """Test to_polars() output for one-sample t-test."""
    design = Design(wgt="samp_wgt", stratum="region", psu="cluster")
    sample = Sample(data=synthetic_sample_df, design=design)

    result = sample.categorical.ttest("income", mean_h0=49000, drop_nulls=True)

    # Test output (combined diff + stats) - default
    df = result.to_polars()
    assert df.shape[0] == 1
    assert set(df.columns) == {"y", "diff", "se", "lci", "uci", "df", "t", "p_value"}

    # Explicit "test" component - same as default
    df_test = result.to_polars("test")
    assert df_test.shape[0] == 1
    assert set(df_test.columns) == {"y", "diff", "se", "lci", "uci", "df", "t", "p_value"}

    # Estimates output (raw)
    df_est = result.to_polars("estimates")
    assert df_est.shape[0] == 1
    assert "est" in df_est.columns
    assert "se" in df_est.columns
    assert "cv" in df_est.columns


def test_ttest_two_sample_to_polars(synthetic_sample_df):
    """Test to_polars() output for two-sample t-test."""
    design = Design(wgt="samp_wgt", stratum="region", psu="cluster")
    sample = Sample(data=synthetic_sample_df, design=design)

    result = sample.categorical.ttest("income", group="sex", drop_nulls=True)

    # Test output (combined diff + stats) - default
    df = result.to_polars()
    assert df.shape[0] == 1
    assert "y" in df.columns
    assert "group_var" in df.columns
    assert "paired" in df.columns
    assert "diff" in df.columns
    assert "t" in df.columns
    assert "p_value" in df.columns

    # Explicit "test" component - same as default
    df_test = result.to_polars("test")
    assert df_test.shape[0] == 1

    # Estimates output (tidy by default)
    df_est = result.to_polars("estimates")
    assert df_est.shape[0] == 2  # Two groups
    assert "sex" in df_est.columns  # tidy: actual variable name
    assert "group" not in df_est.columns
    assert "group_level" not in df_est.columns
    assert "est" in df_est.columns
    assert "se" in df_est.columns

    # Raw format
    df_est_raw = result.to_polars("estimates", tidy=False)
    assert df_est_raw.shape[0] == 2
    assert "group" in df_est_raw.columns
    assert "group_level" in df_est_raw.columns
    assert "sex" not in df_est_raw.columns


# =============================================================================
# Test alternative hypothesis
# =============================================================================


def test_ttest_alternative_less(synthetic_sample_df):
    """Test one-sided t-test with alternative='less'."""
    design = Design(wgt="samp_wgt")
    sample = Sample(data=synthetic_sample_df, design=design)

    result = sample.categorical.ttest("income", mean_h0=49000, alternative="less", drop_nulls=True)

    assert result.alternative == "less"
    # For positive t-statistic, p_value for "less" should be close to 1
    # (we're testing if mean < 49000, but mean > 49000)
    assert result.stats.p_value > 0.9


def test_ttest_alternative_greater(synthetic_sample_df):
    """Test one-sided t-test with alternative='greater'."""
    design = Design(wgt="samp_wgt")
    sample = Sample(data=synthetic_sample_df, design=design)

    result = sample.categorical.ttest(
        "income", mean_h0=49000, alternative="greater", drop_nulls=True
    )

    assert result.alternative == "greater"
    # For positive t-statistic, p_value for "greater" should be small
    # (we're testing if mean > 49000, and mean is indeed > 49000)
    assert result.stats.p_value < 0.05
