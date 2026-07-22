from pathlib import Path

import polars as pl
import pytest

from svy import Design, Sample, TableUnits


REL = 1e-7


@pytest.fixture
def synthetic_sample_df():
    BASE_DIR = Path(__file__).parent.parent.parent
    return (
        pl.read_csv(BASE_DIR / "test_data/svy_synthetic_sample_07082025.csv")
        .fill_nan(None)
        .drop_nulls()
    )


def _assert_chisq(stat, exp):
    assert hasattr(stat, "df")
    assert stat.df == pytest.approx(exp["df"], rel=0, abs=0)  # integer df
    assert stat.value == pytest.approx(exp["value"], rel=REL)
    assert stat.p_value == pytest.approx(exp["p_value"], rel=REL)


def _assert_fdist(stat, exp):
    assert hasattr(stat, "df_num") and hasattr(stat, "df_den")
    assert stat.df_num == pytest.approx(exp["df_num"], rel=REL)
    assert stat.df_den == pytest.approx(exp["df_den"], rel=REL)
    assert stat.value == pytest.approx(exp["value"], rel=REL)
    assert stat.p_value == pytest.approx(exp["p_value"], rel=REL)


## COUNT — 922 rows (complete.cases on all columns)
## Reference values from R survey::svychisq and survey::svytotal.
## est/se/cv are R's svytotal output verbatim. CIs use the design-df t
## quantile (Stata convention, consistent with svy's estimation.total);
## in R that is confint(svytotal(...), df=degf(design)) — NOT the plain
## confint default, which uses the normal critical value.


@pytest.mark.parametrize(
    "design_kwargs,expected_cells,expected_stats",
    [
        # baseline (no strat, no cluster)
        pytest.param(
            {"wgt": "samp_wgt"},
            {  # (row, col) → est, se, cv, lci, uci
                ("High School", "1"): dict(
                    est=127661.024644880555570,
                    se=12070.237294672462667,
                    cv=0.094549118090260,
                    lci=103972.664074038,
                    uci=151349.385215723,
                ),
                ("High School", "2"): dict(
                    est=121830.249133094810531,
                    se=11090.166894325338944,
                    cv=0.091029666057809,
                    lci=100065.318935778,
                    uci=143595.179330412,
                ),
                ("Less than High School", "1"): dict(
                    est=46329.578227587066067,
                    se=7151.489656875141009,
                    cv=0.154361207903610,
                    lci=32294.471752466,
                    uci=60364.684702709,
                ),
                ("Less than High School", "2"): dict(
                    est=46537.876849557665992,
                    se=6998.416642889843388,
                    cv=0.150381089913352,
                    lci=32803.182757426,
                    uci=60272.570941689,
                ),
                ("None", "1"): dict(
                    est=4141.041708652899615,
                    se=2513.430402934253379,
                    cv=0.606956070421199,
                    lci=-791.673713661,
                    uci=9073.757130967,
                ),
                ("None", "2"): dict(
                    est=2989.694273470061489,
                    se=1864.726089498146166,
                    cv=0.623717985496158,
                    lci=-669.910993229,
                    uci=6649.299540169,
                ),
                ("Other Training", "1"): dict(
                    est=46256.275074622659304,
                    se=7140.526795077517818,
                    cv=0.154368824198622,
                    lci=32242.683687963,
                    uci=60269.866461282,
                ),
                ("Other Training", "2"): dict(
                    est=31702.182803883362794,
                    se=6565.851701422656333,
                    cv=0.207110398108561,
                    lci=18816.416040488,
                    uci=44587.949567278,
                ),
                ("Postgraduate", "1"): dict(
                    est=60910.058609322375560,
                    se=7814.309845837641660,
                    cv=0.128292600996474,
                    lci=45574.138964877,
                    uci=76245.978253768,
                ),
                ("Postgraduate", "2"): dict(
                    est=82489.070594564793282,
                    se=9977.672788393299925,
                    cv=0.120957512510157,
                    lci=62907.457996636,
                    uci=102070.683192493,
                ),
                ("Undergraduate", "1"): dict(
                    est=107111.157788566415547,
                    se=10822.057187495287508,
                    cv=0.101035759587789,
                    lci=85872.404438720,
                    uci=128349.911138413,
                ),
                ("Undergraduate", "2"): dict(
                    est=124622.459083789202850,
                    se=10969.919556172326338,
                    cv=0.088025221431369,
                    lci=103093.519466799,
                    uci=146151.398700780,
                ),
            },
            {
                "chisq": {
                    "df": 5,
                    "value": 8.295092928876,
                    "p_value": 0.313318996377,
                },
                "f": {
                    "df_num": 4.98303175853,
                    "df_den": 4589.37224960653,
                    "value": 1.185537436983,
                    "p_value": 0.313544398219,
                },
            },
            id="no_strat_no_cluster",
        ),
        # stratified design
        pytest.param(
            {"wgt": "samp_wgt", "stratum": "region"},
            {
                ("High School", "1"): dict(
                    est=127661.024644880555570,
                    se=12029.656338844542915,
                    cv=0.094231237547308,
                    lci=104052.204425614,
                    uci=151269.844864147,
                ),
                ("High School", "2"): dict(
                    est=121830.249133094810531,
                    se=11044.665946991401142,
                    cv=0.090656187815273,
                    lci=100154.523295243,
                    uci=143505.974970947,
                ),
                ("Less than High School", "1"): dict(
                    est=46329.578227587066067,
                    se=7147.113413344021865,
                    cv=0.154266748948909,
                    lci=32303.000002115,
                    uci=60356.156453059,
                ),
                ("Less than High School", "2"): dict(
                    est=46537.876849557665992,
                    se=6998.070168970679333,
                    cv=0.150373644925686,
                    lci=32803.803668469,
                    uci=60271.950030646,
                ),
                ("None", "1"): dict(
                    est=4141.041708652899615,
                    se=2515.766198194109165,
                    cv=0.607520130245802,
                    lci=-796.279043818,
                    uci=9078.362461124,
                ),
                ("None", "2"): dict(
                    est=2989.694273470061489,
                    se=1863.732647367465916,
                    cv=0.623385696626525,
                    lci=-667.977048893,
                    uci=6647.365595833,
                ),
                ("Other Training", "1"): dict(
                    est=46256.275074622659304,
                    se=7119.925549224405586,
                    cv=0.153923452282706,
                    lci=32283.054432948,
                    uci=60229.495716298,
                ),
                ("Other Training", "2"): dict(
                    est=31702.182803883362794,
                    se=6546.575874803590523,
                    cv=0.206502369735931,
                    lci=18854.190431820,
                    uci=44550.175175947,
                ),
                ("Postgraduate", "1"): dict(
                    est=60910.058609322375560,
                    se=7792.854214479550137,
                    cv=0.127940349958666,
                    lci=45616.180799155,
                    uci=76203.936419490,
                ),
                ("Postgraduate", "2"): dict(
                    est=82489.070594564793282,
                    se=9986.924256339181738,
                    cv=0.121069666373441,
                    lci=62889.217309480,
                    uci=102088.923879649,
                ),
                ("Undergraduate", "1"): dict(
                    est=107111.157788566415547,
                    se=10820.182740405944060,
                    cv=0.101018259570722,
                    lci=85875.991806893,
                    uci=128346.323770240,
                ),
                ("Undergraduate", "2"): dict(
                    est=124622.459083789202850,
                    se=10960.294272417164393,
                    cv=0.087947985884696,
                    lci=103112.317003321,
                    uci=146132.601164257,
                ),
            },
            {
                "chisq": {
                    "df": 5,
                    "value": 8.295092928876,
                    "p_value": 0.310842870463,
                },
                "f": {
                    "df_num": 4.981130916447,
                    "df_den": 4572.678181298437,
                    "value": 1.190551763478,
                    "p_value": 0.311079053715,
                },
            },
            id="stratified",
        ),
        # clustered design
        pytest.param(
            {"wgt": "samp_wgt", "psu": "cluster"},
            {
                ("High School", "1"): dict(
                    est=127661.024644880555570,
                    se=11403.874056590208056,
                    cv=0.089329332020582,
                    lci=104841.925365343,
                    uci=150480.123924418,
                ),
                ("High School", "2"): dict(
                    est=121830.249133094810531,
                    se=13842.130897577828364,
                    cv=0.113618177719195,
                    lci=94132.209184149,
                    uci=149528.289082041,
                ),
                ("Less than High School", "1"): dict(
                    est=46329.578227587066067,
                    se=6497.215219970128601,
                    cv=0.140239032353231,
                    lci=33328.680601982,
                    uci=59330.475853192,
                ),
                ("Less than High School", "2"): dict(
                    est=46537.876849557665992,
                    se=7262.821750355683434,
                    cv=0.156062593354529,
                    lci=32005.004095217,
                    uci=61070.749603898,
                ),
                ("None", "1"): dict(
                    est=4141.041708652899615,
                    se=2479.053084005657638,
                    cv=0.598654459052069,
                    lci=-819.532054478,
                    uci=9101.615471784,
                ),
                ("None", "2"): dict(
                    est=2989.694273470061489,
                    se=1841.376215010249098,
                    cv=0.615907864342601,
                    lci=-694.891022087,
                    uci=6674.279569027,
                ),
                ("Other Training", "1"): dict(
                    est=46256.275074622659304,
                    se=7651.091102341752958,
                    cv=0.165406554894415,
                    lci=30946.477141505,
                    uci=61566.073007741,
                ),
                ("Other Training", "2"): dict(
                    est=31702.182803883362794,
                    se=6857.602896314178906,
                    cv=0.216313272140812,
                    lci=17980.151103594,
                    uci=45424.214504173,
                ),
                ("Postgraduate", "1"): dict(
                    est=60910.058609322375560,
                    se=8951.552883210910295,
                    cv=0.146963458705995,
                    lci=42998.042663305,
                    uci=78822.074555340,
                ),
                ("Postgraduate", "2"): dict(
                    est=82489.070594564793282,
                    se=8506.473451676938566,
                    cv=0.103122430527632,
                    lci=65467.656533929,
                    uci=99510.484655201,
                ),
                ("Undergraduate", "1"): dict(
                    est=107111.157788566415547,
                    se=11534.624936415819320,
                    cv=0.107688360153708,
                    lci=84030.426602817,
                    uci=130191.888974316,
                ),
                ("Undergraduate", "2"): dict(
                    est=124622.459083789202850,
                    se=11214.804549780757952,
                    cv=0.089990236368555,
                    lci=102181.687013515,
                    uci=147063.231154064,
                ),
            },
            {
                "chisq": {
                    "df": 5,
                    "value": 8.295092928876,
                    "p_value": 0.299732253148,
                },
                "f": {
                    "df_num": 4.446995293734,
                    "df_den": 262.372722330293,
                    "value": 1.213445507426,
                    "p_value": 0.304592543287,
                },
            },
            id="clustered",
        ),
        # stratified and clustered design
        pytest.param(
            {"wgt": "samp_wgt", "stratum": "region", "psu": "cluster"},
            {
                ("High School", "1"): dict(
                    est=127661.024644880555570,
                    se=10974.242602067039115,
                    cv=0.085963923856905,
                    lci=105676.975005905,
                    uci=149645.074283856,
                ),
                ("High School", "2"): dict(
                    est=121830.249133094810531,
                    se=13092.428572235354295,
                    cv=0.107464514481394,
                    lci=95602.963108586,
                    uci=148057.535157604,
                ),
                ("Less than High School", "1"): dict(
                    est=46329.578227587066067,
                    se=6409.448130264851898,
                    cv=0.138344625085499,
                    lci=33489.910747697,
                    uci=59169.245707477,
                ),
                ("Less than High School", "2"): dict(
                    est=46537.876849557665992,
                    se=7264.825350853714554,
                    cv=0.156105646468115,
                    lci=31984.682891409,
                    uci=61091.070807706,
                ),
                ("None", "1"): dict(
                    est=4141.041708652899615,
                    se=2515.766198194107801,
                    cv=0.607520130245802,
                    lci=-898.643578671,
                    uci=9180.726995976,
                ),
                ("None", "2"): dict(
                    est=2989.694273470061489,
                    se=1824.527072914022710,
                    cv=0.610272123509251,
                    lci=-665.272651632,
                    uci=6644.661198572,
                ),
                ("Other Training", "1"): dict(
                    est=46256.275074622659304,
                    se=7240.635175298892136,
                    cv=0.156533036082520,
                    lci=31751.539861142,
                    uci=60761.010288104,
                ),
                ("Other Training", "2"): dict(
                    est=31702.182803883362794,
                    se=6546.322955459528202,
                    cv=0.206494391757076,
                    lci=18588.322100778,
                    uci=44816.043506988,
                ),
                ("Postgraduate", "1"): dict(
                    est=60910.058609322375560,
                    se=8628.998904595204294,
                    cv=0.141667880504625,
                    lci=43624.096640744,
                    uci=78196.020577901,
                ),
                ("Postgraduate", "2"): dict(
                    est=82489.070594564793282,
                    se=8659.548499461527172,
                    cv=0.104978131491181,
                    lci=65141.910433605,
                    uci=99836.230755524,
                ),
                ("Undergraduate", "1"): dict(
                    est=107111.157788566415547,
                    se=11629.940869799867869,
                    cv=0.108578238812029,
                    lci=83813.586680390,
                    uci=130408.728896743,
                ),
                ("Undergraduate", "2"): dict(
                    est=124622.459083789202850,
                    se=10986.334814286796245,
                    cv=0.088156941333506,
                    lci=102614.185832914,
                    uci=146630.732334664,
                ),
            },
            {
                "chisq": {
                    "df": 5,
                    "value": 8.295092928876,
                    "p_value": 0.256620282049,
                },
                "f": {
                    "df_num": 4.445276458131,
                    "df_den": 248.935481655313,
                    "value": 1.309275495602,
                    "p_value": 0.264252247329,
                },
            },
            id="stratified-clustered",
        ),
    ],
)
def test_count_crosstab_variants(
    synthetic_sample_df, design_kwargs, expected_cells, expected_stats
):
    design = Design(**design_kwargs)
    sample = Sample(data=synthetic_sample_df, design=design)
    table = sample.categorical.tabulate(
        rowvar="educ",
        colvar="sex",
        units=TableUnits.COUNT,
        drop_nulls=True,
    )

    # sanity: same set of cells
    actual_keys = {(c.rowvar, c.colvar) for c in table.estimates}
    expected_keys = set(expected_cells.keys())
    assert actual_keys == expected_keys, (
        f"Cell keys mismatch.\nActual: {sorted(actual_keys)}\nExpected: {sorted(expected_keys)}"
    )

    # cell-level checks
    for cell in table.estimates:
        exp = expected_cells[(cell.rowvar, cell.colvar)]
        assert cell.est == pytest.approx(exp["est"], rel=REL)
        assert cell.se == pytest.approx(exp["se"], rel=REL)
        assert cell.cv == pytest.approx(exp["cv"], rel=REL)
        assert cell.lci == pytest.approx(exp["lci"], rel=REL)
        assert cell.uci == pytest.approx(exp["uci"], rel=REL)

    # Rao-Scott statistics
    stats = table.stats
    _assert_chisq(stats.chisq, expected_stats["chisq"])
    _assert_fdist(stats.f, expected_stats["f"])
