from pathlib import Path

import polars as pl
import pytest

from svy import Design, ParamEst, Sample


TOL = 1e-7


@pytest.fixture
def synthetic_sample_df():
    """Load and prepare the synthetic sample dataset."""
    BASE_DIR = Path(__file__).parent.parent.parent
    df = pl.read_csv(BASE_DIR / "test_data/svy_synthetic_sample_07082025.csv")
    return df.with_columns(
        pl.when(pl.col("resp2") == 1)
        .then(1)
        .when(pl.col("resp2") == 2)
        .then(0)
        .otherwise(None)
        .alias("resp2_new")
    )


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(wr=False, wgt="samp_wgt", stratum="region", psu="cluster"),
            {
                "High School": {
                    "est": 49668.86660548509,
                    "se": 954.9981380890723,
                    "cv": 0.019227298776002447,
                    "lci": 47755.77544884116,
                    "uci": 51581.95776212902,
                    "deff": 1.3863404765297218,
                },
                "Less than High School": {
                    "est": 49045.36478627392,
                    "se": 1766.9673064945998,
                    "cv": 0.03602720285993493,
                    "lci": 45505.70392903099,
                    "uci": 52585.025643516856,
                    "deff": 1.771416232394358,
                },
                "None": {
                    "est": 48783.45141575658,
                    "se": 3569.0910333184715,
                    "cv": 0.07316192130197843,
                    "lci": 41633.70292853819,
                    "uci": 55933.19990297497,
                    "deff": 0.8165799731143403,
                },
                "Other Training": {
                    "est": 48590.49772572005,
                    "se": 2362.0782172028103,
                    "cv": 0.04861193706094739,
                    "lci": 43858.68645991579,
                    "uci": 53322.30899152431,
                    "deff": 1.7371259442510605,
                },
                "Postgraduate": {
                    "est": 51736.21867040831,
                    "se": 1340.8678476746595,
                    "cv": 0.025917391764110488,
                    "lci": 49050.13759935252,
                    "uci": 54422.29974146409,
                    "deff": 1.377820170623369,
                },
                "Undergraduate": {
                    "est": 50396.049897858386,
                    "se": 937.773912184483,
                    "cv": 0.01860808365110247,
                    "lci": 48517.46301189716,
                    "uci": 52274.63678381961,
                    "deff": 1.083934675784568,
                },
            },
        ),
    ],
    ids=["mean_by_education"],
)
def test_mean_domain_estimates(synthetic_sample_df, design_kwargs, expected):
    """Test mean estimation by education domain."""
    design = Design(**design_kwargs)
    sample = Sample(data=synthetic_sample_df, design=design)
    est = sample.estimation.mean(y="income", by="educ", deff=True, drop_nulls=True)

    estimates = est.estimates if isinstance(est.estimates, list) else [est.estimates]

    estimated_domains = {res.by_level[0] for res in estimates}
    expected_domains = set(expected.keys())
    assert expected_domains <= estimated_domains, (
        f"Missing domains: {expected_domains - estimated_domains}"
    )

    for result in estimates:
        domain = result.by_level
        if domain not in expected:
            continue  # Ignore domains not part of test expectations

        exp = expected[domain]
        assert result.est == pytest.approx(exp["est"], rel=TOL)
        assert result.se == pytest.approx(exp["se"], rel=TOL)
        assert result.cv == pytest.approx(exp["cv"], rel=TOL)
        assert result.lci == pytest.approx(exp["lci"], rel=TOL)
        assert result.uci == pytest.approx(exp["uci"], rel=TOL)
        assert result.deff == pytest.approx(exp["deff"], rel=TOL)


## PROPORTION
@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(wr=False, wgt="samp_wgt"),
            {
                0: {
                    "est": 0.4919919717111689,
                    "se": 0.01881570868400254,
                    "cv": 0.03824393438486549,
                    "lci": 0.45517822526111335,
                    "uci": 0.5288927689616367,
                    "deff": 1.3741473350016822,
                },
                1: {
                    "est": 0.5080080282888311,
                    "se": 0.01881570868400254,
                    "cv": 0.0370382112805996,
                    "lci": 0.47110723103836327,
                    "uci": 0.5448217747388867,
                    "deff": 1.3741473350016822,
                },
            },
        ),
        (
            dict(wr=False, wgt="samp_wgt", stratum="region"),
            {
                0: {
                    "est": 0.4919919717111689,
                    "se": 0.01883380889343362,
                    "cv": 0.0382807240287455,
                    "lci": 0.4551428393311164,
                    "uci": 0.5289283225193957,
                    "deff": 1.376792392728833,
                },
                1: {
                    "est": 0.5080080282888311,
                    "se": 0.01883380889343362,
                    "cv": 0.037073841051042486,
                    "lci": 0.4710716774806044,
                    "uci": 0.5448571606688836,
                    "deff": 1.376792392728833,
                },
            },
        ),
        (
            dict(wr=False, wgt="samp_wgt", psu="cluster"),
            {
                0: {
                    "est": 0.4919919717111689,
                    "se": 0.016460751406429497,
                    "cv": 0.03345735774748175,
                    "lci": 0.4591362958221535,
                    "uci": 0.524916968333348,
                    "deff": 1.0516989767558689,
                },
                1: {
                    "est": 0.5080080282888311,
                    "se": 0.016460751406429497,
                    "cv": 0.03240254186902463,
                    "lci": 0.475083031666652,
                    "uci": 0.5408637041778465,
                    "deff": 1.0516989767558689,
                },
            },
        ),
        (
            dict(wr=False, wgt="samp_wgt", stratum="region", psu="cluster"),
            {
                0: {
                    "est": 0.4919919717111689,
                    "se": 0.016684696308476837,
                    "cv": 0.03391253774009108,
                    "lci": 0.45865387831202487,
                    "uci": 0.5254014386619028,
                    "deff": 1.080509902238721,
                },
                1: {
                    "est": 0.5080080282888311,
                    "se": 0.016684696308476837,
                    "cv": 0.0328433713236332,
                    "lci": 0.4745985613380972,
                    "uci": 0.5413461216879751,
                    "deff": 1.080509902238721,
                },
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
    design = Design(**design_kwargs)
    sample = Sample(data=synthetic_sample_df, design=design)
    est = sample.estimation.prop(y="resp2_new", deff=True, drop_nulls=True)
    estimates = [est.estimates] if isinstance(est.estimates, ParamEst) else est.estimates

    for result in estimates:
        level = result.y_level
        assert level in expected, (
            f"Unexpected level: {level}. Expected keys: {list(expected.keys())}"
        )
        assert result.est == pytest.approx(expected[level]["est"], rel=TOL)
        assert result.se == pytest.approx(expected[level]["se"], rel=TOL)
        assert result.cv == pytest.approx(expected[level]["cv"], rel=TOL)
        assert result.lci == pytest.approx(expected[level]["lci"], rel=TOL)
        assert result.uci == pytest.approx(expected[level]["uci"], rel=TOL)
        assert result.deff == pytest.approx(expected[level]["deff"], rel=TOL)


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(wr=False, wgt="samp_wgt", stratum="region", psu="cluster"),
            {
                "High School": {
                    0: {
                        "est": 0.517386057398944,
                        "se": 0.034773319122857166,
                        "cv": 0.06720961770340922,
                        "lci": 0.44783987358531746,
                        "uci": 0.5862651621388839,
                        "deff": 1.3913580464080977,
                    },
                    1: {
                        "est": 0.4826139426010561,
                        "se": 0.03477331912285716,
                        "cv": 0.0720520400538902,
                        "lci": 0.4137348378611163,
                        "uci": 0.5521601264146826,
                        "deff": 1.3913580464080977,
                    },
                },
                "Less than High School": {
                    0: {
                        "est": 0.47352270517357975,
                        "se": 0.053246062258294605,
                        "cv": 0.11244669300234746,
                        "lci": 0.3696154570874929,
                        "uci": 0.5797750873029738,
                        "deff": 1.3208301619875387,
                    },
                    1: {
                        "est": 0.5264772948264204,
                        "se": 0.0532460622582946,
                        "cv": 0.10113648353220975,
                        "lci": 0.4202249126970264,
                        "uci": 0.6303845429125072,
                        "deff": 1.3208301619875387,
                    },
                },
                "None": {
                    0: {
                        "est": 0.6156279866192711,
                        "se": 0.19552519488724335,
                        "cv": 0.31760283667572103,
                        "lci": 0.2342931606675291,
                        "uci": 0.8934323111298527,
                        "deff": 1.4554326287881612,
                    },
                    1: {
                        "est": 0.38437201338072857,
                        "se": 0.19552519488724335,
                        "cv": 0.5086873863877585,
                        "lci": 0.10656768887014711,
                        "uci": 0.7657068393324706,
                        "deff": 1.4554326287881612,
                    },
                },
                "Other Training": {
                    0: {
                        "est": 0.5197329291594812,
                        "se": 0.05605898285334822,
                        "cv": 0.10786113349409576,
                        "lci": 0.40831715201922747,
                        "uci": 0.6292199915330341,
                        "deff": 1.0965260673251391,
                    },
                    1: {
                        "est": 0.4802670708405189,
                        "se": 0.05605898285334822,
                        "cv": 0.11672460232436711,
                        "lci": 0.3707800084669661,
                        "uci": 0.5916828479807725,
                        "deff": 1.0965260673251391,
                    },
                },
                "Postgraduate": {
                    0: {
                        "est": 0.48954863443781704,
                        "se": 0.04238496022685627,
                        "cv": 0.08657967205961035,
                        "lci": 0.40574367734836486,
                        "uci": 0.5739452177209954,
                        "deff": 1.3099450339498542,
                    },
                    1: {
                        "est": 0.5104513655621832,
                        "se": 0.04238496022685627,
                        "cv": 0.0830342772815894,
                        "lci": 0.42605478227900473,
                        "uci": 0.5942563226516353,
                        "deff": 1.3099450339498542,
                    },
                },
                "Undergraduate": {
                    0: {
                        "est": 0.45907514946963485,
                        "se": 0.031146967724424917,
                        "cv": 0.06784720924321151,
                        "lci": 0.39763496062769343,
                        "uci": 0.5217852884945804,
                        "deff": 1.1068860515288161,
                    },
                    1: {
                        "est": 0.5409248505303652,
                        "se": 0.031146967724424917,
                        "cv": 0.05758095176046355,
                        "lci": 0.4782147115054196,
                        "uci": 0.6023650393723067,
                        "deff": 1.1068860515288161,
                    },
                },
            },
        ),
    ],
    ids=["proportion_by_education"],
)
def test_proportion_domain_estimates(synthetic_sample_df, design_kwargs, expected):
    """Test proportion estimates by education domain and response level."""
    design = Design(**design_kwargs)
    sample = Sample(data=synthetic_sample_df, design=design)
    est = sample.estimation.prop(y="resp2_new", by="educ", deff=True, drop_nulls=True)

    estimates = est.estimates if isinstance(est.estimates, list) else [est.estimates]

    for result in estimates:
        by_level = result.by_level[0]
        y_level = result.y_level
        assert by_level in expected, f"Unexpected domain: {by_level}"
        assert y_level in expected[by_level], (
            f"Unexpected level '{y_level}' in domain '{by_level}'"
        )

        exp = expected[by_level][y_level]
        assert result.est == pytest.approx(exp["est"], rel=TOL)
        assert result.se == pytest.approx(exp["se"], rel=TOL)
        assert result.cv == pytest.approx(exp["cv"], rel=TOL)
        assert result.lci == pytest.approx(exp["lci"], rel=TOL)
        assert result.uci == pytest.approx(exp["uci"], rel=TOL)
        assert result.deff == pytest.approx(exp["deff"], rel=TOL)


## RATIO
@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(wr=False, wgt="samp_wgt"),
            {
                "est": 16672.1357678339,
                "se": 459.843564734807,
                "cv": 0.027581563102551,
                "lci": 15769.7317862477,
                "uci": 17574.53974942,
                "deff": 1.511649361341728,
            },
        ),
        (
            dict(wr=False, wgt="samp_wgt", stratum="region"),
            {
                "est": 16672.1357678339,
                "se": 430.583615215761,
                "cv": 0.0258265420346745,
                "lci": 15827.1486729081,
                "uci": 17517.1228627597,
                "deff": 1.3253965574884292,
            },
        ),
        (
            dict(wr=False, wgt="samp_wgt", psu="cluster"),
            {
                "est": 16672.1357678339,
                "se": 755.063112238264,
                "cv": 0.0452889253514258,
                "lci": 15161.2579700802,
                "uci": 18183.0135655876,
                "deff": 4.075654361358684,
            },
        ),
        (
            dict(wr=False, wgt="samp_wgt", stratum="region", psu="cluster"),
            {
                "est": 16672.1357678339,
                "se": 388.084327446482,
                "cv": 0.0232774212524845,
                "lci": 15894.7094407464,
                "uci": 17449.5620949213,
                "deff": 1.0766710788965628,
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
def test_ratio_estimation_variants(synthetic_sample_df, design_kwargs, expected):
    """Test total estimation under different survey design configurations."""
    design = Design(**design_kwargs)
    sample = Sample(data=synthetic_sample_df, design=design)
    est = sample.estimation.ratio(y="income", x="fam_size", deff=True, drop_nulls=True)

    if isinstance(est.estimates, ParamEst):
        result = est.estimates
    else:
        assert len(est.estimates) == 1
        result = est.estimates[0]

    assert result.est == pytest.approx(expected["est"], rel=TOL)
    assert result.se == pytest.approx(expected["se"], rel=TOL)
    assert result.cv == pytest.approx(expected["cv"], rel=TOL)
    assert result.lci == pytest.approx(expected["lci"], rel=TOL)
    assert result.uci == pytest.approx(expected["uci"], rel=TOL)
    assert result.deff == pytest.approx(expected["deff"], rel=TOL)


@pytest.mark.parametrize(
    "design_kwargs,expected",
    [
        (
            dict(wr=False, wgt="samp_wgt", stratum="region", psu="cluster"),
            {
                "High School": {
                    "est": 15449.60988187744,
                    "se": 664.9105168839737,
                    "cv": 0.04303736611912259,
                    "lci": 14117.63406006528,
                    "uci": 16781.5857036896,
                    "deff": 1.052572305208382,
                },
                "Less than High School": {
                    "est": 17164.92902701146,
                    "se": 1242.1601118033134,
                    "cv": 0.07236616649265473,
                    "lci": 14676.58331171844,
                    "uci": 19653.274742304486,
                    "deff": 1.2860185794693484,
                },
                "None": {
                    "est": 17595.50594658943,
                    "se": 5795.347009380558,
                    "cv": 0.32936518148282523,
                    "lci": 5986.030837545057,
                    "uci": 29204.981055633805,
                    "deff": 1.5821305101688423,
                },
                "Other Training": {
                    "est": 16594.93903142071,
                    "se": 1442.3602663088564,
                    "cv": 0.08691567131267577,
                    "lci": 13705.544214702548,
                    "uci": 19484.33384813887,
                    "deff": 1.579688169877227,
                },
                "Postgraduate": {
                    "est": 18242.280746003547,
                    "se": 944.7243230547681,
                    "cv": 0.051787621087990045,
                    "lci": 16349.770513974245,
                    "uci": 20134.79097803285,
                    "deff": 1.386752219142544,
                },
                "Undergraduate": {
                    "est": 16923.420157731118,
                    "se": 927.1297397098011,
                    "cv": 0.05478382803645402,
                    "lci": 15066.156111489616,
                    "uci": 18780.68420397262,
                    "deff": 1.5369780879315589,
                },
            },
        ),
    ],
    ids=["ratio_by_education"],
)
def test_ratio_domain_estimates(synthetic_sample_df, design_kwargs, expected):
    """Test ratio estimation by education domain."""
    design = Design(**design_kwargs)
    sample = Sample(data=synthetic_sample_df, design=design)
    est = sample.estimation.ratio(y="income", x="fam_size", by="educ", deff=True, drop_nulls=True)

    estimates = est.estimates if isinstance(est.estimates, list) else [est.estimates]

    estimated_domains = {res.by_level[0] for res in estimates}
    expected_domains = set(expected.keys())
    assert expected_domains <= estimated_domains, (
        f"Missing domains: {expected_domains - estimated_domains}"
    )

    for result in estimates:
        domain = result.by_level
        if domain not in expected:
            continue  # Ignore domains not part of test expectations

        exp = expected[domain]
        assert result.est == pytest.approx(exp["est"], rel=TOL)
        assert result.se == pytest.approx(exp["se"], rel=TOL)
        assert result.cv == pytest.approx(exp["cv"], rel=TOL)
        assert result.lci == pytest.approx(exp["lci"], rel=TOL)
        assert result.uci == pytest.approx(exp["uci"], rel=TOL)
        assert result.deff == pytest.approx(exp["deff"], rel=TOL)
