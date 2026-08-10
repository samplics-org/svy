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

import numpy as np
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


# =============================================================================
# ASSOCIATION (COVARIANCE / CORRELATION) TESTS
#
# Golden values from R survey 4.5 on a single-stage cluster design (8 PSUs of
# 3 rows, unequal weights):
#
#   d  <- svydesign(id=~psu, weights=~w, data=dat)
#   v  <- svyvar(~y+x, d)                       # covariance + its SE
#   ct <- svycontrast(svymean(~y+x+yy+xx+yx, d),
#                     quote((yx - y*x)/sqrt((yy-y^2)*(xx-x^2))))
#
# The correlation targets come from svycontrast rather than svyvar because R
# offers no correlation SE: svycontrast applies its own delta method over the
# moment means, so it is an independent check of our linearization.
# =============================================================================

_ASSOC_Y = [12, 15, 11, 22, 25, 19, 31, 29, 35, 41, 44, 39,
            18, 16, 21, 27, 30, 24, 36, 33, 38, 45, 49, 43]
_ASSOC_X = [9, 4, 14, 6, 17, 8, 21, 11, 13, 12, 26, 7,
            19, 5, 23, 10, 28, 15, 8, 22, 16, 18, 31, 20]
_ASSOC_W = [2, 2, 2, 3, 3, 3, 1.5, 1.5, 1.5, 4, 4, 4,
            2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 1, 1, 1, 5, 5, 5]


@pytest.fixture
def assoc_df():
    """Clustered fixture matching the R golden design."""
    return pl.DataFrame(
        {
            "y": [float(v) for v in _ASSOC_Y],
            "x": [float(v) for v in _ASSOC_X],
            "w": [float(v) for v in _ASSOC_W],
            "psu": [str(i // 3 + 1) for i in range(24)],
            "stratum": ["1"] * 12 + ["2"] * 12,
            "grp": ["A"] * 12 + ["B"] * 12,
        }
    )


@pytest.mark.parametrize(
    "kind,strata,expected",
    [
        ("cov", None, {"est": 50.138658078368209, "se": 21.498361515848689}),
        ("corr", None, {"est": 0.52860852253793722, "se": 0.15921590459764234}),
        # svyvar under svydesign(id=~psu, strata=~stratum, ...)
        ("cov", "stratum", {"est": 50.138658078368209, "se": 23.178556453025269}),
    ],
)
def test_assoc_matches_r(assoc_df, kind, strata, expected):
    """Point estimate and SE both match R; the SE is what validates the score."""
    result = ps.taylor_assoc(
        assoc_df, ["y"], ["x"], kind, "w", strata_col=strata, psu_col="psu"
    )
    assert result["est"][0] == pytest.approx(expected["est"], rel=1e-10)
    assert result["se"][0] == pytest.approx(expected["se"], rel=1e-9)
    assert result["kind"][0] == kind


def test_assoc_is_symmetric(assoc_df):
    """Both statistics are symmetric, so argument order cannot matter."""
    fwd = ps.taylor_assoc(assoc_df, ["y"], ["x"], "corr", "w", psu_col="psu")
    rev = ps.taylor_assoc(assoc_df, ["x"], ["y"], "corr", "w", psu_col="psu")
    assert fwd["est"][0] == pytest.approx(rev["est"][0], rel=1e-14)
    assert fwd["se"][0] == pytest.approx(rev["se"][0], rel=1e-14)


def test_assoc_batches_pairs(assoc_df):
    """One row per pair, in the order requested; corr(y,y) is 1 by definition."""
    result = ps.taylor_assoc(
        assoc_df, ["y", "y"], ["x", "y"], "corr", "w", psu_col="psu"
    )
    assert result.height == 2
    assert list(result["y"]) == ["y", "y"]
    assert list(result["x"]) == ["x", "y"]
    assert result["est"][1] == pytest.approx(1.0, abs=1e-12)


def test_assoc_by_group(assoc_df):
    """`by` yields one row per (group, pair) with the group column first."""
    result = ps.taylor_assoc(
        assoc_df, ["y"], ["x"], "corr", "w", psu_col="psu", by_col="grp"
    )
    assert result.height == 2
    assert result.columns[0] == "grp"
    assert sorted(result["grp"]) == ["A", "B"]


def test_assoc_reports_deff(assoc_df):
    """The clustered fixture must show a design effect above 1."""
    result = ps.taylor_assoc(assoc_df, ["y"], ["x"], "corr", "w", psu_col="psu")
    assert result["deff"][0] > 1.0


@pytest.mark.parametrize(
    "kind,expected_est",
    [("cov", 50.138658078368209), ("corr", 0.52860852253793722)],
)
def test_replicate_assoc_matches_r_jk1(assoc_df, kind, expected_est):
    """JK1 replicate weights: full-sample estimate and SE both match R.

    R: svyvar(~y+x, as.svrepdesign(d, type="JK1")) gives cov SE
    29.751032168261617.
    """
    reps = {
        f"rep{r}": [
            0.0 if i // 3 == r else _ASSOC_W[i] * 8 / 7 for i in range(24)
        ]
        for r in range(8)
    }
    df = assoc_df.with_columns([pl.Series(k, v) for k, v in reps.items()])
    result = ps.replicate_assoc(
        df, ["y"], ["x"], kind, "w", list(reps), "Jackknife"
    )
    assert result["est"][0] == pytest.approx(expected_est, rel=1e-10)
    if kind == "cov":
        assert result["se"][0] == pytest.approx(29.751032168261617, rel=1e-9)


def test_assoc_rejects_unknown_kind(assoc_df):
    """An unimplemented coefficient must say so, not fall back silently."""
    with pytest.raises(Exception, match="unknown association kind"):
        ps.taylor_assoc(assoc_df, ["y"], ["x"], "spearman", "w")


def test_assoc_rejects_mismatched_pairs(assoc_df):
    """Pair columns are positional, so unequal lengths are a caller error."""
    with pytest.raises(Exception, match="equal length"):
        ps.taylor_assoc(assoc_df, ["y", "x"], ["x"], "corr", "w")


# =============================================================================
# SRS REFERENCE FOR THE DESIGN EFFECT (deff_ref)
#
# deff compares the design variance to a hypothetical SRS of the same size. The
# two available references differ by exactly the finite-population correction
# 1 - n/N, which is also the only scale-dependent part of the calculation --
# so "wr" is invariant to the weight scale and "wor" is not. These tests drive
# every entry point that exposes the knob, because the parameter was threaded
# mechanically and a mis-wired estimator would otherwise go unnoticed.
# =============================================================================


@pytest.fixture
def srs_ref_df():
    rng = np.random.default_rng(11)
    n = 1200
    psu = rng.integers(0, 30, n)
    return pl.DataFrame(
        {
            "stratum": (psu % 5).astype(str),
            "psu": psu.astype(str),
            "w": rng.uniform(20.0, 200.0, n),
            "y": rng.normal(100.0, 20.0, n),
            "x": np.clip(rng.normal(50.0, 8.0, n), 1.0, None),
            "flag": (rng.random(n) < 0.4).astype(float),
        }
    )


def _call(fn, df, **kw):
    """Invoke an estimator with the design columns it needs."""
    base = dict(weight_col="w", strata_col="stratum", psu_col="psu")
    return fn(df, **base, **kw)


_ESTIMATORS = [
    ("mean", lambda df, **kw: _call(ps.taylor_mean, df, value_col="y", **kw)),
    ("total", lambda df, **kw: _call(ps.taylor_total, df, value_col="y", **kw)),
    ("ratio", lambda df, **kw: _call(ps.taylor_ratio, df, numerator_col="y", denominator_col="x", **kw)),
    ("prop", lambda df, **kw: _call(ps.taylor_prop, df, value_col="flag", **kw)),
    ("assoc", lambda df, **kw: ps.taylor_assoc(df, ["y"], ["x"], "corr", "w", strata_col="stratum", psu_col="psu", **kw)),
    # Batched entry points are a separate plumbing route from the single-variable
    # ones -- they build the design once and fan out over columns -- so the
    # reference has to reach them independently. Row 0 is the first variable.
    ("mean_multi", lambda df, **kw: _call(ps.taylor_mean_multi, df, value_cols=["y", "x"], **kw)),
    ("total_multi", lambda df, **kw: _call(ps.taylor_total_multi, df, value_cols=["y", "x"], **kw)),
    ("ratio_multi", lambda df, **kw: _call(ps.taylor_ratio_multi, df, numerator_cols=["y"], denominator_cols=["x"], **kw)),
    ("prop_multi", lambda df, **kw: _call(ps.taylor_prop_multi, df, value_cols=["flag"], **kw)),
    ("assoc_multi", lambda df, **kw: ps.taylor_assoc(df, ["y", "y"], ["x", "x"], "corr", "w", strata_col="stratum", psu_col="psu", **kw)),
]


@pytest.mark.parametrize("name,fn", _ESTIMATORS, ids=[e[0] for e in _ESTIMATORS])
def test_wr_reference_is_scale_invariant(srs_ref_df, name, fn):
    """Every estimator must honour the reference it is handed.

    A mis-threaded argument shows up here: the with-replacement reference has
    no N in it, so scaling the weights cannot move the design effect.
    """
    base = fn(srs_ref_df, deff_ref="wr")["deff"][0]
    for scale in (0.5, 1000.0):
        scaled = srs_ref_df.with_columns(pl.col("w") * scale)
        assert fn(scaled, deff_ref="wr")["deff"][0] == pytest.approx(base, rel=1e-12), name


@pytest.mark.parametrize("name,fn", _ESTIMATORS, ids=[e[0] for e in _ESTIMATORS])
def test_wor_reference_moves_with_the_weight_scale(srs_ref_df, name, fn):
    """The mirror image: the without-replacement reference carries the FPC, so
    it does depend on the weight scale. This is the behaviour that made
    rescaled weights shift deff silently."""
    base = fn(srs_ref_df, deff_ref="wor")["deff"][0]
    halved = srs_ref_df.with_columns(pl.col("w") * 0.5)
    assert fn(halved, deff_ref="wor")["deff"][0] != pytest.approx(base, rel=1e-9), name


@pytest.mark.parametrize("name,fn", _ESTIMATORS, ids=[e[0] for e in _ESTIMATORS])
def test_default_reference_is_without_replacement(srs_ref_df, name, fn):
    """Omitting the argument must reproduce the historical behaviour exactly."""
    assert fn(srs_ref_df)["deff"][0] == pytest.approx(
        fn(srs_ref_df, deff_ref="wor")["deff"][0], rel=1e-15
    ), name


@pytest.mark.parametrize("name,fn", _ESTIMATORS, ids=[e[0] for e in _ESTIMATORS])
def test_references_differ_by_the_finite_population_correction(srs_ref_df, name, fn):
    """The whole basis for saying the choice is immaterial at small sampling
    fractions: the two differ by 1 - n/N and nothing else."""
    n = srs_ref_df.height
    fpc = 1.0 - n / srs_ref_df["w"].sum()
    wor = fn(srs_ref_df, deff_ref="wor")["deff"][0]
    wr = fn(srs_ref_df, deff_ref="wr")["deff"][0]
    assert wor / wr == pytest.approx(1.0 / fpc, rel=1e-10), name


@pytest.mark.parametrize("name,fn", _ESTIMATORS, ids=[e[0] for e in _ESTIMATORS])
def test_declared_pop_total_survives_normalized_weights(srs_ref_df, name, fn):
    """Weights normalized to sum to n destroy the inferred N, but a design that
    declares its own population size still gets the correct reference."""
    n = srs_ref_df.height
    true_n = srs_ref_df["w"].sum()
    normalized = srs_ref_df.with_columns(pl.col("w") / true_n * n)

    recovered = fn(normalized, deff_ref="wor", deff_pop_total=float(true_n))["deff"][0]
    assert recovered == pytest.approx(fn(srs_ref_df, deff_ref="wor")["deff"][0], rel=1e-9), name

    # Without it the correction is degenerate, reported as NaN for the Python
    # layer to diagnose rather than as a failure here.
    assert np.isnan(fn(normalized, deff_ref="wor")["deff"][0]), name


@pytest.mark.parametrize("name,fn", _ESTIMATORS, ids=[e[0] for e in _ESTIMATORS])
def test_unknown_reference_is_rejected(srs_ref_df, name, fn):
    with pytest.raises(Exception, match="unknown deff reference"):
        fn(srs_ref_df, deff_ref="srswor")


# =============================================================================
# DESIGN FPC x SRS REFERENCE x DESIGN SHAPE  (R golden, survey 4.5)
#
# Two finite-population corrections are in play and they are independent:
# the design's own fpc corrects the variance (numerator), while deff_ref
# decides whether the SRS reference carries one (denominator). Crossing them
# across design shapes is the only way to show neither leaks into the other.
#
#   st  <- svydesign(id=~1,    strata=~stype, weights=~pw,           data=apistrat)
#   stf <- svydesign(id=~1,    strata=~stype, weights=~pw, fpc=~fpc, data=apistrat)
#   cl  <- svydesign(id=~dnum,                weights=~pw,           data=apiclus1)
#   clf <- svydesign(id=~dnum,                weights=~pw, fpc=~fpc, data=apiclus1)
#   sc  <- svydesign(id=~dnum, strata=~stype, weights=~pw, data=apiclus1, nest=TRUE)
#   svymean(~api00, d, deff=TRUE)       # -> "wor"
#   svymean(~api00, d, deff="replace")  # -> "wr"
#
# Domain rows come from subset(d, sch.wide == "Yes"), where R uses that
# subset's own n and sum of weights in the reference -- so each domain gets a
# different correction. That is the case most likely to drift, which is why it
# is pinned rather than assumed.
# =============================================================================

# (shape, design fpc applied, domain) -> (se, deff_wor, deff_wr)
_R_DEFF_MATRIX = {
    ("stratSRS", False, False): (9.5361322969, 1.237241445022, 1.197291769419),
    ("stratSRS", True, False): (9.4089408028, 1.204457268538, 1.165566171454),
    ("cluster", False, False): (23.7790107209, 9.534802121425, 9.253099070589),
    ("cluster", True, False): (23.5422406938, 9.345869450591, 9.069748362453),
    ("strat+cluster", False, False): (18.3076135248, 5.651810485034, 5.484829331560),
    ("stratSRS", False, True): (10.6540738828, 1.228801107524, 1.192380187135),
    ("stratSRS", True, True): (10.5203892001, 1.198157193016, 1.162644539689),
    ("cluster", False, True): (23.6621771021, 8.044653756919, 7.806976721006),
}

# shape -> (csv, psu column, stratum column)
_DEFF_SHAPES = {
    "stratSRS": ("apistrat_deff.csv", None, "stype"),
    "cluster": ("apiclus1_deff.csv", "dnum", None),
    "strat+cluster": ("apiclus1_deff.csv", "dnum", "stype"),
}


def _deff_frame(csv: str, psu: str | None, stratum: str | None) -> pl.DataFrame:
    """Load an api dataset with the FPC expressed the way the kernel wants it.

    The kernel takes the correction factor ``(N_h - n_h) / N_h``, not the raw
    stratum population that R's ``fpc=`` accepts. ``n_h`` counts PSUs, so for a
    design without clusters every row is its own PSU.
    """
    df = pl.read_csv(
        Path(__file__).parent.parent / "data" / csv, infer_schema_length=10000
    ).with_columns(
        pl.col("pw").cast(pl.Float64),
        pl.col("api00").cast(pl.Float64),
        pl.col("fpc").cast(pl.Float64),
        pl.col("stype").cast(pl.String),
        pl.col("dnum").cast(pl.String),
        pl.col("sch.wide").cast(pl.String),
    )
    unit = psu
    if unit is None:
        df = df.with_columns(pl.int_range(pl.len()).cast(pl.String).alias("__unit"))
        unit = "__unit"
    n_h = pl.col(unit).n_unique().over(stratum) if stratum else pl.col(unit).n_unique()
    return df.with_columns(((pl.col("fpc") - n_h) / pl.col("fpc")).alias("fpcf"))


@pytest.mark.parametrize("key", sorted(_R_DEFF_MATRIX))
@pytest.mark.parametrize("ref", ["wor", "wr"])
def test_deff_matches_r_across_designs(key, ref):
    shape, use_fpc, domain = key
    se, deff_wor, deff_wr = _R_DEFF_MATRIX[key]
    csv, psu, stratum = _DEFF_SHAPES[shape]

    out = ps.taylor_mean(
        _deff_frame(csv, psu, stratum),
        value_col="api00",
        weight_col="pw",
        strata_col=stratum,
        psu_col=psu,
        fpc_col="fpcf" if use_fpc else None,
        by_col="sch.wide" if domain else None,
        deff_ref=ref,
    )
    row = (
        [r for r in out.to_dicts() if r["sch.wide"] == "Yes"][0]
        if domain
        else out.to_dicts()[0]
    )
    assert row["se"] == pytest.approx(se, rel=1e-9)
    assert row["deff"] == pytest.approx(deff_wor if ref == "wor" else deff_wr, rel=1e-10)


@pytest.mark.parametrize("key", sorted(_R_DEFF_MATRIX))
def test_the_two_corrections_stay_independent(key):
    """The reference must never touch the standard error, and the wor/wr ratio
    must be 1/(1 - n/N) whether or not the design carries its own fpc. If
    either correction leaked into the other, one of these would fail."""
    shape, use_fpc, domain = key
    csv, psu, stratum = _DEFF_SHAPES[shape]
    df = _deff_frame(csv, psu, stratum)

    got = {}
    for ref in ("wor", "wr"):
        out = ps.taylor_mean(
            df,
            value_col="api00",
            weight_col="pw",
            strata_col=stratum,
            psu_col=psu,
            fpc_col="fpcf" if use_fpc else None,
            by_col="sch.wide" if domain else None,
            deff_ref=ref,
        )
        got[ref] = (
            [r for r in out.to_dicts() if r["sch.wide"] == "Yes"][0]
            if domain
            else out.to_dicts()[0]
        )

    assert got["wor"]["se"] == pytest.approx(got["wr"]["se"], rel=1e-15)

    # n and sum(w) are the domain's own when a domain is in play, which is what
    # makes each domain carry a different correction.
    sub = df.filter(pl.col("sch.wide") == "Yes") if domain else df
    expected = 1.0 / (1.0 - sub.height / sub["pw"].sum())
    assert got["wor"]["deff"] / got["wr"]["deff"] == pytest.approx(expected, rel=1e-9)
