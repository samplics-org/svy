# tests/svy/categorical/test_tabulate_where.py
"""tabulate(where=) is R's subset() on a survey design, not a row subset.

R survey 4.5 reference (full double precision):

```r
api <- read.csv("tests/test_data/apiclus1.csv")
d <- svydesign(id = ~dnum, weights = ~pw, data = api)
s <- subset(d, comp.imp == "Yes")             # 133 rows in domain, degf(s) == 13
svymean(~stype, s); svytotal(~stype, s)
svymean(~interaction(stype, awards), s); svytotal(~interaction(stype, awards), s)
svychisq(~stype + awards, s); svychisq(~stype + awards, s, statistic = "Chisq")

apistrat <- read.csv("tests/test_data/apistrat.csv")
ds <- svydesign(id = ~1, strata = ~stype, weights = ~pw, fpc = ~fpc, data = apistrat)
ss <- subset(ds, awards == "Yes")             # 113 rows, degf(ss) == 110
svymean(~interaction(stype, sch.wide), ss); svytotal(~interaction(stype, sch.wide), ss)
svymean(~awards, ds); svytotal(~awards, ds)   # full design, fpc applied
svymean(~interaction(awards, sch.wide), ds); svychisq(~awards + sch.wide, ds)
```

One district has no school with comp.imp == "Yes": R keeps it in the design
with zero weight (the SEs are those of the full 15-PSU design) but drops it
from degf. A physical row subset gets both wrong.
"""

import polars as pl
import pytest

import svy

from svy import Design, Sample, col


TOL = 1e-9


@pytest.fixture(scope="module")
def dclus1():
    api = svy.io.read_csv("tests/test_data/apiclus1.csv")
    return Sample(api, Design(psu="dnum", wgt="pw"))


@pytest.fixture(scope="module")
def dstrat():
    api = svy.io.read_csv("tests/test_data/apistrat.csv")
    return Sample(api, Design(stratum="stype", wgt="pw", pop_size="fpc"))


def _cells(table):
    return {(c.rowvar, c.colvar): c for c in table.estimates}


R_ONE_WAY_PROP = {
    "E": (0.842105263157892, 0.0409492585246707),
    "H": (0.0526315789473684, 0.0227253550689293),
    "M": (0.105263157894737, 0.029876669708234),
}
R_ONE_WAY_TOTAL = {
    "E": (3790.86358642578, 1004.30863761453),
    "H": (236.928974151611, 83.8890626604538),
    "M": (473.857948303222, 167.778125320908),
}
R_TWO_WAY_PROP = {
    ("E", "No"): (0.00751879699248119, 0.00801740093761738),
    ("H", "No"): (0.00751879699248119, 0.00665134671002414),
    ("M", "No"): (0.00751879699248119, 0.00665134671002413),
    ("E", "Yes"): (0.834586466165411, 0.0443304091088959),
    ("H", "Yes"): (0.0451127819548872, 0.0236209584895551),
    ("M", "Yes"): (0.0977443609022555, 0.0278011877422042),
}
R_TWO_WAY_TOTAL = {
    ("E", "No"): (33.846996307373, 33.846996307373),
    ("H", "No"): (33.846996307373, 33.846996307373),
    ("M", "No"): (33.846996307373, 33.846996307373),
    ("E", "Yes"): (3757.01659011841, 1011.534264815),
    ("H", "Yes"): (203.081977844238, 82.9078702789303),
    ("M", "Yes"): (440.010951995849, 147.535636446233),
}


def test_one_way_proportions_and_totals_match_r_subset(dclus1):
    prop = dclus1.categorical.tabulate("stype", where=col("comp.imp") == "Yes")
    for k, (est, se) in R_ONE_WAY_PROP.items():
        c = _cells(prop)[(k, "")]
        assert c.est == pytest.approx(est, rel=TOL)
        assert c.se == pytest.approx(se, rel=TOL)
    tot = dclus1.categorical.tabulate("stype", units="count", where=col("comp.imp") == "Yes")
    for k, (est, se) in R_ONE_WAY_TOTAL.items():
        c = _cells(tot)[(k, "")]
        assert c.est == pytest.approx(est, rel=TOL)
        assert c.se == pytest.approx(se, rel=TOL)


def test_two_way_cells_and_rao_scott_match_r_subset(dclus1):
    prop = dclus1.categorical.tabulate("stype", "awards", where=col("comp.imp") == "Yes")
    for k, (est, se) in R_TWO_WAY_PROP.items():
        c = _cells(prop)[k]
        assert c.est == pytest.approx(est, rel=TOL)
        assert c.se == pytest.approx(se, rel=TOL)
    assert prop.stats is not None
    assert prop.stats.f.value == pytest.approx(3.85040085659189, rel=TOL)
    assert prop.stats.f.df_num == pytest.approx(1.12071788197409, rel=TOL)
    assert prop.stats.f.df_den == pytest.approx(14.5693324656632, rel=TOL)
    assert prop.stats.f.p_value == pytest.approx(0.0655069168762211, rel=TOL)
    assert prop.stats.chisq.value == pytest.approx(7.05496794871795, rel=TOL)
    assert prop.stats.chisq.df == 2
    assert prop.stats.chisq.p_value == pytest.approx(0.0212712080252016, rel=TOL)

    tot = dclus1.categorical.tabulate(
        "stype", "awards", units="count", where=col("comp.imp") == "Yes"
    )
    for k, (est, se) in R_TWO_WAY_TOTAL.items():
        c = _cells(tot)[k]
        assert c.est == pytest.approx(est, rel=TOL)
        assert c.se == pytest.approx(se, rel=TOL)


def test_domain_df_matches_r_degf_not_a_row_subset(dclus1):
    from scipy.stats import t as t_dist

    prop = dclus1.categorical.tabulate("stype", where=col("comp.imp") == "Yes")
    c = _cells(prop)[("E", "")]
    # logit CI on 13 df (R degf on the subset design), not 14 and not 12
    p, se = c.est, c.se
    scale = se / (p * (1 - p))
    logit = pl.Series([p / (1 - p)]).log().item()
    lci13 = 1 / (1 + pl.Series([-(logit - t_dist.ppf(0.975, 13) * scale)]).exp().item())
    assert c.lci == pytest.approx(lci13, rel=1e-9)
    # a physical subset gives a different SE (R: 0.04105 vs 0.04095)
    physical = Sample(dclus1.data.filter(pl.col("comp.imp") == "Yes"), dclus1.design)
    assert _cells(physical.categorical.tabulate("stype"))[("E", "")].se != pytest.approx(
        se, rel=1e-6
    )


def test_stratified_fpc_two_way_matches_r_subset(dstrat):
    r_prop = {
        ("E", "Yes"): (0.815482725699157, 0.0199039427230178),
        ("H", "Yes"): (0.0610475642671339, 0.0119707320727657),
        ("M", "Yes"): (0.123469710033711, 0.0168983274596488),
    }
    r_total = {
        ("E", "Yes"): (3227.3299331665, 195.019525962437),
        ("H", "Yes"): (241.600006103515, 48.6182643626208),
        ("M", "Yes"): (488.640014648439, 70.8493436299405),
    }
    prop = dstrat.categorical.tabulate("stype", "sch.wide", where=col("awards") == "Yes")
    assert set(_cells(prop)) == set(r_prop)  # sch.wide == "No" never occurs in the domain
    for k, (est, se) in r_prop.items():
        assert _cells(prop)[k].est == pytest.approx(est, rel=TOL)
        assert _cells(prop)[k].se == pytest.approx(se, rel=TOL)
    tot = dstrat.categorical.tabulate(
        "stype", "sch.wide", units="count", where=col("awards") == "Yes"
    )
    for k, (est, se) in r_total.items():
        assert _cells(tot)[k].est == pytest.approx(est, rel=TOL)
        assert _cells(tot)[k].se == pytest.approx(se, rel=TOL)


def test_percent_and_count_total_scale_within_domain(dclus1):
    prop = dclus1.categorical.tabulate("stype", where=col("comp.imp") == "Yes")
    pct = dclus1.categorical.tabulate("stype", units="percent", where=col("comp.imp") == "Yes")
    n = dclus1.categorical.tabulate(
        "stype", units="count", count_total=1000, where=col("comp.imp") == "Yes"
    )
    for k in ("E", "H", "M"):
        assert _cells(pct)[(k, "")].est == pytest.approx(100 * _cells(prop)[(k, "")].est)
        assert _cells(n)[(k, "")].est == pytest.approx(1000 * _cells(prop)[(k, "")].est)
    assert sum(c.est for c in prop.estimates) == pytest.approx(1.0)


def test_nulls_outside_the_domain_are_not_missing_data(dclus1):
    df = dclus1.data.with_columns(
        pl.when(pl.col("comp.imp") == "Yes").then(pl.col("stype")).otherwise(None).alias("st")
    )
    s = Sample(df, dclus1.design)
    ok = s.categorical.tabulate("st", where=col("comp.imp") == "Yes")
    assert _cells(ok)[("E", "")].est == pytest.approx(R_ONE_WAY_PROP["E"][0], rel=TOL)
    with pytest.raises(ValueError, match="Missing"):
        s.categorical.tabulate("st")
    # with drop_nulls a null key inside the domain leaves the domain, not the frame
    df2 = df.with_columns(
        pl.when(pl.col("cds") == df["cds"][0]).then(None).otherwise(pl.col("st")).alias("st")
    )
    s2 = Sample(df2, dclus1.design)
    with pytest.raises(ValueError, match="Missing"):
        s2.categorical.tabulate("st", where=col("comp.imp") == "Yes")
    dropped = s2.categorical.tabulate("st", where=col("comp.imp") == "Yes", drop_nulls=True)
    assert sum(c.est for c in dropped.estimates) == pytest.approx(1.0)


def test_where_forms_and_empty_domain(dclus1):
    a = dclus1.categorical.tabulate("stype", where={"comp.imp": "Yes"})
    b = dclus1.categorical.tabulate("stype", where=col("comp.imp") == "Yes")
    assert [c.est for c in a.estimates] == [c.est for c in b.estimates]
    assert a.rowvals == ["E", "H", "M"]
    with pytest.raises(Exception, match="in scope"):
        dclus1.categorical.tabulate("stype", where=col("api00") > 10_000)


def test_no_where_is_unchanged(dclus1):
    full = dclus1.categorical.tabulate("stype", "awards")
    assert full.stats is not None and len(full.estimates) == 6


def test_panel_transition_table_on_the_target_wave():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4] * 2,
            "wave": [1] * 4 + [2] * 4,
            "emp": [1, 1, 0, 0, 1, 0, 0, 1],
            "w": [1.0] * 8,
        }
    )
    s = Sample(df, Design(case_id="id", wave="wave", wgt="w")).wrangling.lag("emp")
    # the lag is null at wave 1, which is outside the domain: no drop_nulls needed
    tab = s.categorical.tabulate("emp_lag1", "emp", units="count", where=col("wave") == 2)
    assert {k: c.est for k, c in _cells(tab).items()} == {
        ("0", "0"): 1.0,
        ("0", "1"): 1.0,
        ("1", "0"): 1.0,
        ("1", "1"): 1.0,
    }


def test_full_design_applies_the_fpc(dstrat):
    # tabulate used to ignore pop_size altogether (the ttest facade had the
    # same gap); these are the R values WITH the fpc.
    prop = dstrat.categorical.tabulate("awards")
    assert _cells(prop)[("No", "")].est == pytest.approx(0.361063935949422, rel=TOL)
    assert _cells(prop)[("No", "")].se == pytest.approx(0.034405917997269, rel=TOL)
    tot = dstrat.categorical.tabulate("awards", units="count")
    assert _cells(tot)[("Yes", "")].est == pytest.approx(3957.56995391846, rel=TOL)
    assert _cells(tot)[("Yes", "")].se == pytest.approx(213.110254631354, rel=TOL)

    two = dstrat.categorical.tabulate("awards", "sch.wide")
    r = {
        ("No", "No"): (0.172051988584176, 0.0243447801130748),
        ("No", "Yes"): (0.189011947365245, 0.0296244902573179),
        ("Yes", "Yes"): (0.638936064050579, 0.034405917997269),
    }
    for k, (est, se) in r.items():
        assert _cells(two)[k].est == pytest.approx(est, rel=TOL)
        assert _cells(two)[k].se == pytest.approx(se, rel=TOL)
    assert two.stats.f.value == pytest.approx(77.2768990382019, rel=1e-6)
    assert two.stats.f.df_num == pytest.approx(1.0, rel=TOL)
    assert two.stats.f.df_den == pytest.approx(197.0, rel=TOL)
    assert two.stats.chisq.value == pytest.approx(73.54614520966, rel=1e-6)
