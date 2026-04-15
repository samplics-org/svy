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
## Reference values from R survey::svychisq and survey::svytotal


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
                    lci=104003.794262470357353,
                    uci=151318.255027290753787,
                ),
                ("High School", "2"): dict(
                    est=121830.249133094810531,
                    se=11090.166894325338944,
                    cv=0.091029666057809,
                    lci=100093.921437678727671,
                    uci=143566.576828510878840,
                ),
                ("Less than High School", "1"): dict(
                    est=46329.578227587066067,
                    se=7151.489656875141009,
                    cv=0.154361207903610,
                    lci=32312.916064301083679,
                    uci=60346.240390873048455,
                ),
                ("Less than High School", "2"): dict(
                    est=46537.876849557665992,
                    se=6998.416642889843388,
                    cv=0.150381089913352,
                    lci=32821.232280687865568,
                    uci=60254.521418427466415,
                ),
                ("None", "1"): dict(
                    est=4141.041708652899615,
                    se=2513.430402934253379,
                    cv=0.606956070421199,
                    lci=-785.191358746231344,
                    uci=9067.274776052030575,
                ),
                ("None", "2"): dict(
                    est=2989.694273470061489,
                    se=1864.726089498146166,
                    cv=0.623717985496158,
                    lci=-665.101702978517096,
                    uci=6644.490249918640075,
                ),
                ("Other Training", "1"): dict(
                    est=46256.275074622659304,
                    se=7140.526795077517818,
                    cv=0.154368824198622,
                    lci=32261.099725627507723,
                    uci=60251.450423617810884,
                ),
                ("Other Training", "2"): dict(
                    est=31702.182803883362794,
                    se=6565.851701422656333,
                    cv=0.207110398108561,
                    lci=18833.349941263924848,
                    uci=44571.015666502804379,
                ),
                ("Postgraduate", "1"): dict(
                    est=60910.058609322375560,
                    se=7814.309845837641660,
                    cv=0.128292600996474,
                    lci=45594.292747443862027,
                    uci=76225.824471200889093,
                ),
                ("Postgraduate", "2"): dict(
                    est=82489.070594564793282,
                    se=9977.672788393299925,
                    cv=0.120957512510157,
                    lci=62933.191279788596148,
                    uci=102044.949909340997692,
                ),
                ("Undergraduate", "1"): dict(
                    est=107111.157788566415547,
                    se=10822.057187495287508,
                    cv=0.101035759587789,
                    lci=85900.315462442827993,
                    uci=128322.000114690003102,
                ),
                ("Undergraduate", "2"): dict(
                    est=124622.459083789202850,
                    se=10969.919556172326338,
                    cv=0.088025221431369,
                    lci=103121.811840389840654,
                    uci=146123.106327188579598,
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
                    lci=104083.331474351289216,
                    uci=151238.717815409821924,
                ),
                ("High School", "2"): dict(
                    est=121830.249133094810531,
                    se=11044.665946991401142,
                    cv=0.090656187815273,
                    lci=100183.101655715698143,
                    uci=143477.396610473922919,
                ),
                ("Less than High School", "1"): dict(
                    est=46329.578227587066067,
                    se=7147.113413344021865,
                    cv=0.154266748948909,
                    lci=32321.493344009657449,
                    uci=60337.663111164474685,
                ),
                ("Less than High School", "2"): dict(
                    est=46537.876849557665992,
                    se=6998.070168970679333,
                    cv=0.150373644925686,
                    lci=32821.911357091012178,
                    uci=60253.842342024319805,
                ),
                ("None", "1"): dict(
                    est=4141.041708652899615,
                    se=2515.766198194109165,
                    cv=0.607520130245802,
                    lci=-789.769433330808170,
                    uci=9071.852850636607400,
                ),
                ("None", "2"): dict(
                    est=2989.694273470061489,
                    se=1863.732647367465916,
                    cv=0.623385696626525,
                    lci=-663.154592181659154,
                    uci=6642.543139121782588,
                ),
                ("Other Training", "1"): dict(
                    est=46256.275074622659304,
                    se=7119.925549224405586,
                    cv=0.153923452282706,
                    lci=32301.477425536264491,
                    uci=60211.072723709054117,
                ),
                ("Other Training", "2"): dict(
                    est=31702.182803883362794,
                    se=6546.575874803590523,
                    cv=0.206502369735931,
                    lci=18871.129867209532676,
                    uci=44533.235740557196550,
                ),
                ("Postgraduate", "1"): dict(
                    est=60910.058609322375560,
                    se=7792.854214479550137,
                    cv=0.127940349958666,
                    lci=45636.345012171288545,
                    uci=76183.772206473455299,
                ),
                ("Postgraduate", "2"): dict(
                    est=82489.070594564793282,
                    se=9986.924256339181738,
                    cv=0.121069666373441,
                    lci=62915.058735810540384,
                    uci=102063.082453319046181,
                ),
                ("Undergraduate", "1"): dict(
                    est=107111.157788566415547,
                    se=10820.182740405944060,
                    cv=0.101018259570722,
                    lci=85903.989311228870065,
                    uci=128318.326265903961030,
                ),
                ("Undergraduate", "2"): dict(
                    est=124622.459083789202850,
                    se=10960.294272417164393,
                    cv=0.087947985884696,
                    lci=103140.677049890931812,
                    uci=146104.241117687488440,
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
                    lci=105309.842209733062191,
                    uci=150012.207080028048949,
                ),
                ("High School", "2"): dict(
                    est=121830.249133094810531,
                    se=13842.130897577828364,
                    cv=0.113618177719195,
                    lci=94700.171104553184705,
                    uci=148960.327161636436358,
                ),
                ("Less than High School", "1"): dict(
                    est=46329.578227587066067,
                    se=6497.215219970128601,
                    cv=0.140239032353231,
                    lci=33595.270396640131366,
                    uci=59063.886058534000767,
                ),
                ("Less than High School", "2"): dict(
                    est=46537.876849557665992,
                    se=7262.821750355683434,
                    cv=0.156062593354529,
                    lci=32303.007792726377374,
                    uci=60772.745906388954609,
                ),
                ("None", "1"): dict(
                    est=4141.041708652899615,
                    se=2479.053084005657638,
                    cv=0.598654459052069,
                    lci=-717.813051761137103,
                    uci=8999.896469066936334,
                ),
                ("None", "2"): dict(
                    est=2989.694273470061489,
                    se=1841.376215010249098,
                    cv=0.615907864342601,
                    lci=-619.336789938708534,
                    uci=6598.725336878831513,
                ),
                ("Other Training", "1"): dict(
                    est=46256.275074622659304,
                    se=7651.091102341752958,
                    cv=0.165406554894415,
                    lci=31260.412071597966133,
                    uci=61252.138077647352475,
                ),
                ("Other Training", "2"): dict(
                    est=31702.182803883362794,
                    se=6857.602896314178906,
                    cv=0.216313272140812,
                    lci=18261.528106830013712,
                    uci=45142.837500936715514,
                ),
                ("Postgraduate", "1"): dict(
                    est=60910.058609322375560,
                    se=8951.552883210910295,
                    cv=0.146963458705995,
                    lci=43365.337352523318259,
                    uci=78454.779866121432860,
                ),
                ("Postgraduate", "2"): dict(
                    est=82489.070594564793282,
                    se=8506.473451676938566,
                    cv=0.103122430527632,
                    lci=65816.688993831878179,
                    uci=99161.452195297708386,
                ),
                ("Undergraduate", "1"): dict(
                    est=107111.157788566415547,
                    se=11534.624936415819320,
                    cv=0.107688360153708,
                    lci=84503.708338013806497,
                    uci=129718.607239119024598,
                ),
                ("Undergraduate", "2"): dict(
                    est=124622.459083789202850,
                    se=11214.804549780757952,
                    cv=0.089990236368555,
                    lci=102641.846072562984773,
                    uci=146603.072095015406376,
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
                    lci=106151.904387224029051,
                    uci=149170.144902537082089,
                ),
                ("High School", "2"): dict(
                    est=121830.249133094810531,
                    se=13092.428572235354295,
                    cv=0.107464514481394,
                    lci=96169.560661350362352,
                    uci=147490.937604839273263,
                ),
                ("Less than High School", "1"): dict(
                    est=46329.578227587066067,
                    se=6409.448130264851898,
                    cv=0.138344625085499,
                    lci=33767.290731490371400,
                    uci=58891.865723683760734,
                ),
                ("Less than High School", "2"): dict(
                    est=46537.876849557665992,
                    se=7264.825350853714554,
                    cv=0.156105646468115,
                    lci=32299.080807910828298,
                    uci=60776.672891204507323,
                ),
                ("None", "1"): dict(
                    est=4141.041708652899615,
                    se=2515.766198194107801,
                    cv=0.607520130245802,
                    lci=-789.769433330805441,
                    uci=9071.852850636605581,
                ),
                ("None", "2"): dict(
                    est=2989.694273470061489,
                    se=1824.527072914022710,
                    cv=0.610272123509251,
                    lci=-586.313078259706799,
                    uci=6565.701625199830232,
                ),
                ("Other Training", "1"): dict(
                    est=46256.275074622659304,
                    se=7240.635175298892136,
                    cv=0.156533036082520,
                    lci=32064.890905842974462,
                    uci=60447.659243402347784,
                ),
                ("Other Training", "2"): dict(
                    est=31702.182803883362794,
                    se=6546.322955459528202,
                    cv=0.206494391757076,
                    lci=18871.625580014886509,
                    uci=44532.740027751839079,
                ),
                ("Postgraduate", "1"): dict(
                    est=60910.058609322375560,
                    se=8628.998904595204294,
                    cv=0.141667880504625,
                    lci=43997.531533680201392,
                    uci=77822.585684964549728,
                ),
                ("Postgraduate", "2"): dict(
                    est=82489.070594564793282,
                    se=8659.548499461527172,
                    cv=0.104978131491181,
                    lci=65516.667413242335897,
                    uci=99461.473775887250667,
                ),
                ("Undergraduate", "1"): dict(
                    est=107111.157788566415547,
                    se=11629.940869799867869,
                    cv=0.108578238812029,
                    lci=84316.892541428256663,
                    uci=129905.423035704574431,
                ),
                ("Undergraduate", "2"): dict(
                    est=124622.459083789202850,
                    se=10986.334814286796245,
                    cv=0.088156941333506,
                    lci=103089.638525688555092,
                    uci=146155.279641889850609,
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
