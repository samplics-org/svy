"""
Tests for svy-rs survey estimation functions, exercised through the current
``_internal`` API re-exported at the ``svy_rs`` top level:
``taylor_mean`` / ``taylor_total`` / ``taylor_ratio`` / ``taylor_prop``.

Expected values from the R survey package / the svy library. These are golden
tests on an independent synthetic dataset (packages/svy-rs/data/), complementing
the svy-package suite which drives the same kernels through the Python layer.

Note: the ``_internal`` functions do NOT auto-cast — the svy Python layer owns
that. So the caller must hand the kernel Float64 value/weight columns and String
strata/psu/by columns (see ``_cast_for_kernel``). The removed ``df.svy`` polars
accessor used to do this implicitly.
"""

from pathlib import Path

import polars as pl
import pytest
import svy_rs as ps

TOL = 1e-7  # For overall estimates


@pytest.fixture
def synthetic_sample_df():
    """Load and prepare the synthetic sample dataset."""
    BASE_DIR = Path(__file__).parent.parent
    df = pl.read_csv(BASE_DIR / "data/svy_synthetic_sample_07082025.csv")

    # Convert NaN to null in float columns (so is_not_null() filtering works)
    float_cols = [
        col for col, dtype in zip(df.columns, df.dtypes, strict=False) if dtype == pl.Float64
    ]
    for col in float_cols:
        df = df.with_columns(
            pl.when(pl.col(col).is_nan()).then(None).otherwise(pl.col(col)).alias(col)
        )

    return df.with_columns(
        pl.when(pl.col("resp2") == 1)
        .then(1.0)
        .when(pl.col("resp2") == 2)
        .then(0.0)
        .otherwise(None)
        .alias("resp2_new")
    )


def _cast_for_kernel(df, design_kwargs, *, by_col=None, float_cols=()):
    """Cast columns to the dtypes the ``_internal`` kernels require.

    - strata/psu (and any ``by`` column) → String
    - the given value/denominator columns → Float64

    The removed ``df.svy.design`` accessor did this implicitly; the direct
    kernel API expects the columns already in the right dtype.
    """
    casts = []
    for key in ("strata", "psu"):
        col = design_kwargs.get(key)
        if col:
            casts.append(pl.col(col).cast(pl.String))
    if by_col:
        casts.append(pl.col(by_col).cast(pl.String))
    for col in float_cols:
        casts.append(pl.col(col).cast(pl.Float64))
    return df.with_columns(casts) if casts else df


# =============================================================================
# MEAN ESTIMATION TESTS
# =============================================================================


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(weight="samp_wgt"),
            dict(
                est=50074.677523194,
                se=559.6724463009815,
            ),
        ),
        (
            dict(weight="samp_wgt", strata="region"),
            dict(
                est=50074.677523194,
                se=559.3284439465469,
            ),
        ),
        (
            dict(weight="samp_wgt", psu="cluster"),
            dict(
                est=50074.677523194,
                se=504.498568055472,
            ),
        ),
        (
            dict(weight="samp_wgt", strata="region", psu="cluster"),
            dict(
                est=50074.677523194,
                se=495.280768426878,
            ),
        ),
    ],
    ids=[
        "no_strat_no_cluster",
        "strat_only",
        "cluster_only",
        "strat_and_cluster",
    ],
)
def test_mean_estimation_variants(synthetic_sample_df, design_kwargs, expected):
    """Test mean estimation under different survey design configurations."""
    # Drop nulls from y and weight
    df = synthetic_sample_df.filter(
        pl.col("income").is_not_null() & pl.col("samp_wgt").is_not_null()
    )
    df = _cast_for_kernel(df, design_kwargs)

    result = ps.taylor_mean(
        df,
        value_col="income",
        weight_col=design_kwargs["weight"],
        strata_col=design_kwargs.get("strata"),
        psu_col=design_kwargs.get("psu"),
    )

    assert result["est"][0] == pytest.approx(expected["est"], rel=TOL)
    assert result["se"][0] == pytest.approx(expected["se"], rel=TOL)


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(weight="samp_wgt", strata="region", psu="cluster"),
            {
                "High School": {
                    "est": 49668.86660548509,
                    "se": 954.9981380890723,
                },
                "Less than High School": {
                    "est": 49045.36478627392,
                    "se": 1766.9673064945998,
                },
                "None": {
                    "est": 48783.45141575658,
                    "se": 3569.0910333184715,
                },
                "Other Training": {
                    "est": 48590.49772572005,
                    "se": 2362.0782172028103,
                },
                "Postgraduate": {
                    "est": 51736.21867040831,
                    "se": 1340.8678476746595,
                },
                "Undergraduate": {
                    "est": 50396.049897858386,
                    "se": 937.773912184483,
                },
            },
        ),
    ],
    ids=["mean_by_education"],
)
def test_mean_domain_estimates(synthetic_sample_df, design_kwargs, expected):
    """Test mean estimation by education domain."""
    # Drop nulls from y, weight, and domain
    df = synthetic_sample_df.filter(
        pl.col("income").is_not_null()
        & pl.col("samp_wgt").is_not_null()
        & pl.col("educ").is_not_null()
    )
    df = _cast_for_kernel(df, design_kwargs, by_col="educ")

    result = ps.taylor_mean(
        df,
        value_col="income",
        weight_col=design_kwargs["weight"],
        strata_col=design_kwargs.get("strata"),
        psu_col=design_kwargs.get("psu"),
        by_col="educ",
    )

    for row in result.iter_rows(named=True):
        domain = row["educ"]
        if domain not in expected:
            continue

        exp = expected[domain]
        assert row["est"] == pytest.approx(exp["est"], rel=TOL), f"Failed for domain {domain}"
        assert row["se"] == pytest.approx(exp["se"], rel=TOL), f"Failed for domain {domain}"


# =============================================================================
# TOTAL ESTIMATION TESTS
# =============================================================================


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(weight="samp_wgt"),
            dict(
                est=430882.7569272894,
                se=18155.244686691767,
            ),
        ),
        (
            dict(weight="samp_wgt", strata="region"),
            dict(
                est=430882.7569272894,
                se=18179.23128608681,
            ),
        ),
        (
            dict(weight="samp_wgt", psu="cluster"),
            dict(
                est=430882.7569272894,
                se=22066.51218304301,
            ),
        ),
        (
            dict(weight="samp_wgt", strata="region", psu="cluster"),
            dict(
                est=430882.7569272894,
                se=22282.627163143527,
            ),
        ),
    ],
    ids=[
        "no_strat_no_cluster",
        "strat_only",
        "cluster_only",
        "strat_and_cluster",
    ],
)
def test_total_estimation_variants(synthetic_sample_df, design_kwargs, expected):
    """Test total estimation under different survey design configurations."""
    # Drop nulls from y and weight
    df = synthetic_sample_df.filter(
        pl.col("resp2_new").is_not_null() & pl.col("samp_wgt").is_not_null()
    )
    df = _cast_for_kernel(df, design_kwargs)

    result = ps.taylor_total(
        df,
        value_col="resp2_new",
        weight_col=design_kwargs["weight"],
        strata_col=design_kwargs.get("strata"),
        psu_col=design_kwargs.get("psu"),
    )

    assert result["est"][0] == pytest.approx(expected["est"], rel=TOL)
    assert result["se"][0] == pytest.approx(expected["se"], rel=TOL)


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(weight="samp_wgt", strata="region", psu="cluster"),
            {
                "High School": {
                    "est": 126011.02459107367,
                    "se": 12278.350285279997,
                },
                "Less than High School": {
                    "est": 50085.11096341969,
                    "se": 7671.249179217489,
                },
                "None": {
                    "est": 4031.5713053827712,
                    "se": 2400.7352274310388,
                },
                "Other Training": {
                    "est": 38881.681425075614,
                    "se": 7166.346673501455,
                },
                "Postgraduate": {
                    "est": 79852.96630730784,
                    "se": 10548.404342620915,
                },
                "Undergraduate": {
                    "est": 132020.40233502985,
                    "se": 12297.752124110806,
                },
            },
        ),
    ],
    ids=["total_by_education"],
)
def test_total_domain_estimates(synthetic_sample_df, design_kwargs, expected):
    """Test total estimation by education domain."""
    # Drop nulls from y, weight, and domain
    df = synthetic_sample_df.filter(
        pl.col("resp2_new").is_not_null()
        & pl.col("samp_wgt").is_not_null()
        & pl.col("educ").is_not_null()
    )
    df = _cast_for_kernel(df, design_kwargs, by_col="educ")

    result = ps.taylor_total(
        df,
        value_col="resp2_new",
        weight_col=design_kwargs["weight"],
        strata_col=design_kwargs.get("strata"),
        psu_col=design_kwargs.get("psu"),
        by_col="educ",
    )

    for row in result.iter_rows(named=True):
        domain = row["educ"]
        if domain not in expected:
            continue

        exp = expected[domain]
        assert row["est"] == pytest.approx(exp["est"], rel=TOL), f"Failed for domain {domain}"
        assert row["se"] == pytest.approx(exp["se"], rel=TOL), f"Failed for domain {domain}"


# =============================================================================
# RATIO ESTIMATION TESTS
# =============================================================================


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(weight="samp_wgt"),
            dict(
                est=16672.1357678339,
                se=459.843564734807,
            ),
        ),
        (
            dict(weight="samp_wgt", strata="region"),
            dict(
                est=16672.1357678339,
                se=430.583615215761,
            ),
        ),
        (
            dict(weight="samp_wgt", psu="cluster"),
            dict(
                est=16672.1357678339,
                se=755.063112238264,
            ),
        ),
        (
            dict(weight="samp_wgt", strata="region", psu="cluster"),
            dict(
                est=16672.1357678339,
                se=388.084327446482,
            ),
        ),
    ],
    ids=[
        "no_strat_no_cluster",
        "strat_only",
        "cluster_only",
        "strat_and_cluster",
    ],
)
def test_ratio_estimation_variants(synthetic_sample_df, design_kwargs, expected):
    """Test ratio estimation under different survey design configurations."""
    # Drop nulls from y, x, and weight
    df = synthetic_sample_df.filter(
        pl.col("income").is_not_null()
        & pl.col("fam_size").is_not_null()
        & pl.col("samp_wgt").is_not_null()
    )
    # fam_size is Int64 in the CSV; the kernel needs Float64.
    df = _cast_for_kernel(df, design_kwargs, float_cols=("fam_size",))

    result = ps.taylor_ratio(
        df,
        numerator_col="income",
        denominator_col="fam_size",
        weight_col=design_kwargs["weight"],
        strata_col=design_kwargs.get("strata"),
        psu_col=design_kwargs.get("psu"),
    )

    assert result["est"][0] == pytest.approx(expected["est"], rel=TOL)
    assert result["se"][0] == pytest.approx(expected["se"], rel=TOL)


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(weight="samp_wgt", strata="region", psu="cluster"),
            {
                "High School": {
                    "est": 15449.60988187744,
                    "se": 664.9105168839737,
                },
                "Less than High School": {
                    "est": 17164.92902701146,
                    "se": 1242.1601118033134,
                },
                "None": {
                    "est": 17595.50594658943,
                    "se": 5795.347009380558,
                },
                "Other Training": {
                    "est": 16594.93903142071,
                    "se": 1442.3602663088564,
                },
                "Postgraduate": {
                    "est": 18242.280746003547,
                    "se": 944.7243230547681,
                },
                "Undergraduate": {
                    "est": 16923.420157731118,
                    "se": 927.1297397098011,
                },
            },
        ),
    ],
    ids=["ratio_by_education"],
)
def test_ratio_domain_estimates(synthetic_sample_df, design_kwargs, expected):
    """Test ratio estimation by education domain."""
    # Drop nulls from y, x, weight, and domain
    df = synthetic_sample_df.filter(
        pl.col("income").is_not_null()
        & pl.col("fam_size").is_not_null()
        & pl.col("samp_wgt").is_not_null()
        & pl.col("educ").is_not_null()
    )
    df = _cast_for_kernel(df, design_kwargs, by_col="educ", float_cols=("fam_size",))

    result = ps.taylor_ratio(
        df,
        numerator_col="income",
        denominator_col="fam_size",
        weight_col=design_kwargs["weight"],
        strata_col=design_kwargs.get("strata"),
        psu_col=design_kwargs.get("psu"),
        by_col="educ",
    )

    for row in result.iter_rows(named=True):
        domain = row["educ"]
        if domain not in expected:
            continue

        exp = expected[domain]
        assert row["est"] == pytest.approx(exp["est"], rel=TOL), f"Failed for domain {domain}"
        assert row["se"] == pytest.approx(exp["se"], rel=TOL), f"Failed for domain {domain}"


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================


def test_simple_mean():
    """Basic test that the API works."""
    df = pl.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    result = ps.taylor_mean(df, value_col="value", weight_col="weight")

    assert result["est"][0] == pytest.approx(3.0, rel=1e-10)
    assert result["n"][0] == 5


def test_weighted_mean():
    """Test that weights are applied correctly."""
    df = pl.DataFrame(
        {
            "value": [1.0, 2.0],
            "weight": [1.0, 3.0],  # Second value weighted 3x
        }
    )

    result = ps.taylor_mean(df, value_col="value", weight_col="weight")

    # Weighted mean: (1*1 + 2*3) / (1+3) = 7/4 = 1.75
    assert result["est"][0] == pytest.approx(1.75, rel=1e-10)


def test_group_by():
    """Test domain estimation with a by column."""
    df = pl.DataFrame(
        {
            "value": [1.0, 2.0, 10.0, 20.0],
            "group": ["A", "A", "B", "B"],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = ps.taylor_mean(df, value_col="value", weight_col="weight", by_col="group")

    result_dict = {row["group"]: row["est"] for row in result.iter_rows(named=True)}

    assert result_dict["A"] == pytest.approx(1.5, rel=1e-10)
    assert result_dict["B"] == pytest.approx(15.0, rel=1e-10)


def test_total():
    """Test total estimation."""
    df = pl.DataFrame(
        {
            "value": [10.0, 20.0, 30.0],
            "weight": [2.0, 2.0, 2.0],
        }
    )

    result = ps.taylor_total(df, value_col="value", weight_col="weight")

    # Total: 10*2 + 20*2 + 30*2 = 120
    assert result["est"][0] == pytest.approx(120.0, rel=1e-10)


def test_ratio():
    """Test ratio estimation."""
    df = pl.DataFrame(
        {
            "num": [100.0, 200.0, 300.0],
            "denom": [10.0, 20.0, 30.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )

    result = ps.taylor_ratio(df, numerator_col="num", denominator_col="denom", weight_col="weight")

    # Ratio: (100+200+300) / (10+20+30) = 600/60 = 10
    assert result["est"][0] == pytest.approx(10.0, rel=1e-10)


# =============================================================================
# PROPORTION ESTIMATION TESTS
# =============================================================================


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(weight="samp_wgt"),
            {
                "0": {"est": 0.4919919717111689, "se": 0.01881570868400254},
                "1": {"est": 0.5080080282888311, "se": 0.01881570868400254},
            },
        ),
        (
            dict(weight="samp_wgt", strata="region"),
            {
                "0": {"est": 0.4919919717111689, "se": 0.01883380889343362},
                "1": {"est": 0.5080080282888311, "se": 0.01883380889343362},
            },
        ),
        (
            dict(weight="samp_wgt", psu="cluster"),
            {
                "0": {"est": 0.4919919717111689, "se": 0.016460751406429497},
                "1": {"est": 0.5080080282888311, "se": 0.016460751406429497},
            },
        ),
        (
            dict(weight="samp_wgt", strata="region", psu="cluster"),
            {
                "0": {"est": 0.4919919717111689, "se": 0.016684696308476837},
                "1": {"est": 0.5080080282888311, "se": 0.016684696308476837},
            },
        ),
    ],
    ids=[
        "no_strat_no_cluster",
        "strat_only",
        "cluster_only",
        "strat_and_cluster",
    ],
)
def test_prop_estimation_variants(synthetic_sample_df, design_kwargs, expected):
    """Test proportion estimation for both levels under different survey design configurations."""
    # Drop nulls from y and weight
    df = synthetic_sample_df.filter(
        pl.col("resp2_new").is_not_null() & pl.col("samp_wgt").is_not_null()
    )

    # Cast columns
    df = _cast_for_kernel(df, design_kwargs)

    result = ps.taylor_prop(
        df,
        value_col="resp2_new",
        weight_col=design_kwargs["weight"],
        strata_col=design_kwargs.get("strata"),
        psu_col=design_kwargs.get("psu"),
    )

    for row in result.iter_rows(named=True):
        level = row["level"]
        if level not in expected:
            continue

        exp = expected[level]
        assert row["est"] == pytest.approx(exp["est"], rel=TOL), f"Failed est for level {level}"
        assert row["se"] == pytest.approx(exp["se"], rel=TOL), f"Failed se for level {level}"


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(weight="samp_wgt", strata="region", psu="cluster"),
            {
                "High School": {
                    "0": {"est": 0.517386057398944, "se": 0.034773319122857166},
                    "1": {"est": 0.4826139426010561, "se": 0.03477331912285716},
                },
                "Less than High School": {
                    "0": {"est": 0.47352270517357975, "se": 0.053246062258294605},
                    "1": {"est": 0.5264772948264204, "se": 0.0532460622582946},
                },
                "None": {
                    "0": {"est": 0.6156279866192711, "se": 0.19552519488724335},
                    "1": {"est": 0.38437201338072857, "se": 0.19552519488724335},
                },
                "Other Training": {
                    "0": {"est": 0.5197329291594812, "se": 0.05605898285334822},
                    "1": {"est": 0.4802670708405189, "se": 0.05605898285334822},
                },
                "Postgraduate": {
                    "0": {"est": 0.48954863443781704, "se": 0.04238496022685627},
                    "1": {"est": 0.5104513655621832, "se": 0.04238496022685627},
                },
                "Undergraduate": {
                    "0": {"est": 0.45907514946963485, "se": 0.031146967724424917},
                    "1": {"est": 0.5409248505303652, "se": 0.031146967724424917},
                },
            },
        ),
    ],
    ids=["proportion_by_education"],
)
def test_prop_domain_estimates(synthetic_sample_df, design_kwargs, expected):
    """Test proportion estimates by education domain and response level."""
    # Drop nulls from y, weight, and domain
    df = synthetic_sample_df.filter(
        pl.col("resp2_new").is_not_null()
        & pl.col("samp_wgt").is_not_null()
        & pl.col("educ").is_not_null()
    )

    # Cast columns
    df = _cast_for_kernel(df, design_kwargs, by_col="educ")

    result = ps.taylor_prop(
        df,
        value_col="resp2_new",
        weight_col=design_kwargs["weight"],
        strata_col=design_kwargs.get("strata"),
        psu_col=design_kwargs.get("psu"),
        by_col="educ",
    )

    for row in result.iter_rows(named=True):
        domain = row["educ"]
        level = row["level"]

        if domain not in expected:
            continue
        if level not in expected[domain]:
            continue

        exp = expected[domain][level]
        assert row["est"] == pytest.approx(exp["est"], rel=TOL), f"Failed est for {domain}/{level}"
        assert row["se"] == pytest.approx(exp["se"], rel=TOL), f"Failed se for {domain}/{level}"
