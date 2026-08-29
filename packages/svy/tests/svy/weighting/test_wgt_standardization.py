# tests/svy/weighting/test_wgt_standardization.py
"""Direct standardization.

The R-parity values come from ``survey::svystandardize`` (survey 4.5) on the
fixture built below; R's ``by=`` is svy's ``cells=`` (the composition axis) and
R's ``over=`` is svy's ``by=`` (the domains).
"""

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


def test_standardize_partial_domain_warns(parity_df):
    """A domain missing a level is renormalized over what it has, with a warning."""
    df = parity_df.with_columns(
        pl.when((pl.col("grp") == "A") & (pl.col("age") == "o"))
        .then(pl.lit("m"))
        .otherwise(pl.col("age"))
        .alias("age")
    )
    sample = Sample(df, Design(wgt="w", stratum="strat", psu="psu"))
    with pytest.warns(UserWarning, match="do not observe every level"):
        sample.weighting.standardize("age", shares=POP, by="grp")


def test_standardize_rejects_unknown_share_keys(parity_sample):
    with pytest.raises(Exception, match="Mapping keys|Unexpected keys"):
        parity_sample.weighting.standardize("age", shares={"m": 1, "o": 1, "z": 1}, by="grp")


def test_standardize_rejects_shared_column(parity_sample):
    with pytest.raises(Exception, match="both by= and cells="):
        parity_sample.weighting.standardize("age", shares=POP, by="age")


def test_standardize_refuses_to_overwrite(parity_sample):
    with pytest.raises(Exception, match="already exists"):
        parity_sample.weighting.standardize("age", shares=POP, by="grp", wgt_name="w")
