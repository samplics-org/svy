# tests/svy/weighting/test_wgt_standardization.py
"""Direct standardization.

The R-parity values come from ``survey::svystandardize`` (survey 4.5) on the
fixture built below; R's ``by=`` is svy's ``cells=`` (the composition axis) and
R's ``over=`` is svy's ``by=`` (the domains).
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import Design, Sample, col


STD_WGT = "std_wgt"

# survey::svystandardize(des, by=~age, over=~grp, population=c(m=30, o=50, y=20))
R_WEIGHTS = np.array(
    [
        5.418750000000,
        9.736363636364,
        17.850000000000,
        8.160000000000,
        13.005000000000,
        24.121621621622,
        11.921250000000,
        19.472727272727,
        33.150000000000,
        14.280000000000,
        21.675000000000,
        38.594594594595,
        18.423750000000,
        29.209090909091,
        48.450000000000,
        20.400000000000,
        30.345000000000,
        53.067567567568,
        24.926250000000,
        38.945454545455,
        63.750000000000,
        26.520000000000,
        39.015000000000,
        67.540540540541,
        31.428750000000,
        48.681818181818,
        79.050000000000,
        32.640000000000,
        47.685000000000,
        82.013513513514,
        37.931250000000,
        58.418181818182,
        94.350000000000,
        38.760000000000,
        56.355000000000,
        96.486486486487,
        44.433750000000,
        68.154545454545,
        109.650000000000,
        44.880000000000,
        65.025000000000,
        110.959459459459,
        50.936250000000,
        77.890909090909,
        124.950000000000,
        51.000000000000,
        73.695000000000,
        125.432432432432,
        57.438750000000,
        87.627272727273,
        140.250000000000,
        57.120000000000,
        82.365000000000,
        139.905405405405,
        63.941250000000,
        97.363636363636,
        155.550000000000,
        63.240000000000,
        91.035000000000,
        154.378378378378,
    ]
)
R_MEANS = {"A": 10.324062617647, "B": 10.190306880870}


# set.seed(7); round(rnorm(60, 10, 3), 4)
_Y = [
    16.8617,
    6.4097,
    7.9171,
    8.7631,
    7.088,
    7.1582,
    12.2444,
    9.6491,
    10.458,
    16.5699,
    11.071,
    18.1503,
    16.8444,
    10.9721,
    15.6882,
    11.403,
    7.3186,
    9.078,
    9.9855,
    12.9645,
    12.5193,
    12.116,
    13.9179,
    5.836,
    13.8188,
    10.5526,
    12.2568,
    11.7752,
    7.0508,
    9.1718,
    7.3874,
    12.1561,
    10.332,
    9.7646,
    8.7385,
    8.3136,
    12.9925,
    6.6846,
    9.5731,
    10.945,
    13.6557,
    7.902,
    9.1437,
    6.0653,
    8.827,
    8.7954,
    14.0516,
    11.7736,
    10.3016,
    12.7932,
    9.2118,
    9.977,
    11.1015,
    15.1215,
    12.1712,
    11.4431,
    5.2964,
    10.9548,
    10.498,
    7.3003,
]


@pytest.fixture
def parity_df():
    n = 60
    return pl.DataFrame(
        {
            "strat": ["S1"] * 30 + ["S2"] * 30,
            "psu": [f"P{i // 10 + 1}" for i in range(n)],
            "age": [["y", "m", "o"][i % 3] for i in range(n)],
            "grp": [["A", "B"][i % 2] for i in range(n)],
            "w": [round((5 + i) * 1.7, 3) for i in range(n)],
            "y": _Y,
        }
    )


@pytest.fixture
def parity_sample(parity_df):
    return Sample(parity_df, Design(wgt="w", stratum="strat", psu="psu"))


POP = {"m": 30, "o": 50, "y": 20}


# ---------------------------------------------------------------------------
# R parity
# ---------------------------------------------------------------------------


def test_standardize_matches_r_weights(parity_sample):
    out = parity_sample.weighting.standardize("age", shares=POP, by="grp")
    assert_allclose(out.data.get_column(STD_WGT).to_numpy(), R_WEIGHTS, rtol=1e-12)


def test_standardize_matches_r_means(parity_sample):
    out = parity_sample.weighting.standardize("age", shares=POP, by="grp")
    got = out.estimation.mean("y", by="grp").to_polars().sort("grp")
    for row in got.iter_rows(named=True):
        assert_allclose(row["est"], R_MEANS[row["grp"]], rtol=1e-10)


# ---------------------------------------------------------------------------
# Defining properties
# ---------------------------------------------------------------------------


def test_standardize_preserves_domain_totals(parity_sample):
    """Only within-domain composition is reshaped; domain totals are untouched."""
    before = parity_sample.data.group_by("grp").agg(pl.col("w").sum()).sort("grp")["w"].to_numpy()
    out = parity_sample.weighting.standardize("age", shares=POP, by="grp")
    after = out.data.group_by("grp").agg(pl.col(STD_WGT).sum()).sort("grp")[STD_WGT].to_numpy()
    assert_allclose(after, before)


def test_standardize_imposes_share_composition(parity_sample):
    """Every domain ends with the same age composition, in the given ratios."""
    out = parity_sample.weighting.standardize("age", shares=POP, by="grp")
    tab = out.data.group_by(["grp", "age"]).agg(pl.col(STD_WGT).sum()).sort(["grp", "age"])
    for grp in ("A", "B"):
        sub = tab.filter(pl.col("grp") == grp).sort("age")
        got = sub[STD_WGT].to_numpy()
        want = np.array([POP[a] for a in sub["age"].to_list()], dtype=float)
        assert_allclose(got / got.sum(), want / want.sum())


def test_standardize_shares_are_normalized(parity_sample):
    """Counts and proportions describing the same composition agree."""
    counts = parity_sample.weighting.standardize("age", shares=POP, by="grp")
    total = sum(POP.values())
    props = parity_sample.weighting.standardize(
        "age", shares={k: v / total for k, v in POP.items()}, by="grp"
    )
    assert_allclose(
        counts.data.get_column(STD_WGT).to_numpy(),
        props.data.get_column(STD_WGT).to_numpy(),
    )


def test_standardize_by_none_equals_poststratify_shares(parity_sample):
    """The documented degenerate overlap: no domains means plain poststratification."""
    std = parity_sample.weighting.standardize("age", shares=POP)
    ps = parity_sample.weighting.poststratify(shares=POP, cells="age")
    assert_allclose(
        std.data.get_column(STD_WGT).to_numpy(), ps.data.get_column("ps_wgt").to_numpy()
    )


def test_standardize_where_scopes_the_adjustment(parity_sample):
    """Out-of-scope rows keep their weight, so the column stays complete."""
    out = parity_sample.weighting.standardize(
        "age", shares=POP, by="grp", where=col("strat") == "S1"
    )
    d = out.data
    excluded = d.filter(pl.col("strat") == "S2")
    assert_allclose(excluded[STD_WGT].to_numpy(), excluded["w"].to_numpy())
    included = d.filter(pl.col("strat") == "S1")
    assert_allclose(included[STD_WGT].sum(), included["w"].sum())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


# Simulated design where domain A never observes age level "o". Passed through
# R survey 4.5 and svy identically; the point is agreement on one input, not
# that the input is any particular real population.
PARTIAL_DF = pl.DataFrame(
    {
        "strat": ["S1"] * 12 + ["S2"] * 12,
        "psu": [f"P{i // 4 + 1}" for i in range(24)],
        "grp": ["A"] * 12 + ["B"] * 12,
        "age": ["y", "m", "y", "m"] * 3 + ["y", "m", "o", "y"] * 3,
        "w": [round((3 + i) * 2.5, 3) for i in range(24)],
    }
)
_PARTIAL_V = [
    42.3045,
    47.6598,
    52.0703,
    40.7829,
    51.5663,
    50.241,
    50.6833,
    58.9329,
    40.2491,
    60.1389,
    44.0417,
    40.9503,
    44.2691,
    52.0212,
    51.2164,
    47.5387,
    42.3759,
    44.8141,
    59.7945,
    51.5985,
    45.3721,
    42.4616,
    48.3702,
    36.6682,
]
PARTIAL_POP = {"m": 250, "o": 400, "y": 350}
# svyby(~v, ~grp, svymean, design=svystandardize(des, by=~age, over=~grp, ...))
R_PARTIAL_MEANS = {"A": 47.974773726852, "B": 48.155251165312}


@pytest.fixture
def partial_sample():
    return Sample(
        PARTIAL_DF.with_columns(pl.Series("v", _PARTIAL_V)),
        Design(wgt="w", stratum="strat", psu="psu"),
    )


def test_standardize_partial_domain_warns(partial_sample):
    with pytest.warns(UserWarning, match="do not observe every level"):
        partial_sample.weighting.standardize("age", shares=PARTIAL_POP, by="grp")


def test_standardize_partial_domain_matches_r_estimates(partial_sample):
    """Composition within an incomplete domain agrees with R exactly."""
    with pytest.warns(UserWarning):
        std = partial_sample.weighting.standardize("age", shares=PARTIAL_POP, by="grp")
    got = std.estimation.mean("v", by="grp").to_polars().sort("grp")
    for row in got.iter_rows(named=True):
        assert_allclose(row["est"], R_PARTIAL_MEANS[row["grp"]], rtol=1e-10)


def test_standardize_partial_domain_preserves_the_domain_total(partial_sample):
    """A DOCUMENTED divergence from R, and the reason to prefer svy's choice.

    R drops the absent cell and leaves the rest at their unrenormalized shares,
    so an incomplete domain's total silently falls -- here to (250+350)/1000 =
    60% of its original 255. That breaks the invariant svystandardize is
    defined by, namely that domain totals stay at their current estimates.
    svy renormalizes over the levels present instead, so the total holds and
    the warning names the real consequence: that domain is standardized to a
    different population than the others.

    Within-domain means are identical either way, so this shows up only in
    totals or when pooling domains -- where R would understate A by 40%.
    """
    with pytest.warns(UserWarning):
        std = partial_sample.weighting.standardize("age", shares=PARTIAL_POP, by="grp")
    totals = (
        std.data.group_by("grp")
        .agg([pl.col("w").sum().alias("before"), pl.col(STD_WGT).sum().alias("after")])
        .sort("grp")
    )
    assert_allclose(totals["after"].to_numpy(), totals["before"].to_numpy())
    a = totals.filter(pl.col("grp") == "A")
    assert_allclose(a["after"][0], 255.0)  # R gives 153.0


def test_standardize_rejects_unknown_share_keys(parity_sample):
    with pytest.raises(Exception, match="Mapping keys|Unexpected keys"):
        parity_sample.weighting.standardize("age", shares={"m": 1, "o": 1, "z": 1}, by="grp")


def test_standardize_rejects_shared_column(parity_sample):
    with pytest.raises(Exception, match="both by= and cells="):
        parity_sample.weighting.standardize("age", shares=POP, by="age")


def test_standardize_refuses_to_overwrite(parity_sample):
    with pytest.raises(Exception, match="already exists"):
        parity_sample.weighting.standardize("age", shares=POP, by="grp", wgt_name="w")


# ---------------------------------------------------------------------------
# NHANES: the canonical age-standardization case
#
# Data: `nhanes` from R survey 4.5 -- an NHANES 2009-2010 extract published by
# NCHS (US public domain), redistributed there as the svystandardize example.
# The repo already ships the api* fixtures from the same package.
# ---------------------------------------------------------------------------

NHANES_PATH = Path(__file__).resolve().parents[2] / "test_data" / "nhanes.csv"

# Standard population for the four age groups, from the R help example.
POPAGE = {"(0,19]": 55901, "(19,39]": 77670, "(39,59]": 72816, "(59,Inf]": 45364}

# svyby(~HI_CHOL, ~race+RIAGENDR, svymean,
#       design=subset(stdes, agecat != "(0,19]"))
# "matches http://www.cdc.gov/nchs/data/databriefs/db92_fig1.png"
R_DB92 = {
    (1, 1): 0.154378607135,
    (2, 1): 0.114294593734,
    (3, 1): 0.102077638149,
    (4, 1): 0.135831229210,
    (1, 2): 0.131643567446,
    (2, 2): 0.154324740861,
    (3, 2): 0.102541052423,
    (4, 2): 0.119743388293,
}


@pytest.fixture
def nhanes():
    df = pl.read_csv(NHANES_PATH, null_values=["NA"])
    return Sample(
        df.with_columns(pl.lit("1").alias("all_adults")),
        Design(wgt="WTMEC2YR", stratum="SDMVSTRA", psu="SDMVPSU"),
    )


def _standardized(nhanes, **kwargs):
    return nhanes.weighting.standardize(
        "agecat", shares=POPAGE, where=col("HI_CHOL").is_not_null(), **kwargs
    )


def test_standardize_matches_the_nchs_databrief(nhanes):
    """Age-standardized high cholesterol by race and sex, NCHS databrief 92.

    Note the order: the weights are standardized over all four age groups and
    the estimate is then restricted to adults. That is what NCHS does, and it
    is the analysis-specific-weights caveat in action -- the same standardized
    sample would be wrong for a different breakdown.
    """
    std = _standardized(nhanes, by=["race", "RIAGENDR"])
    got = (
        std.estimation.mean(
            "HI_CHOL",
            by=["race", "RIAGENDR"],
            where=col("agecat") != "(0,19]",
            drop_nulls=True,
        )
        .to_polars()
        .sort(["race", "RIAGENDR"])
    )
    for row in got.iter_rows(named=True):
        key = (int(row["race"]), int(row["RIAGENDR"]))
        assert_allclose(row["est"], R_DB92[key], rtol=1e-9)


def test_standardize_by_none_equals_a_constant_domain(nhanes):
    """R's `over = ~1` and `over = ~<constant>` are the same standardization."""
    by_const = _standardized(nhanes, by="all_adults")
    by_none = _standardized(nhanes)
    assert_allclose(
        by_const.data.get_column(STD_WGT).to_numpy(),
        by_none.data.get_column(STD_WGT).to_numpy(),
        rtol=0,
        atol=0,
    )
    a = by_const.estimation.mean("HI_CHOL", drop_nulls=True).to_polars()["est"][0]
    b = by_none.estimation.mean("HI_CHOL", drop_nulls=True).to_polars()["est"][0]
    assert a == b
    assert_allclose(a, 0.105873359319, rtol=1e-9)
