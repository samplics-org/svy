# tests/svy/estimation/test_prop_drop_nulls.py
"""``prop(drop_nulls=True)``: a missing y is out of the domain, not dropped.

Every 10th row of apiclus1 has a missing y, so PSU 413 has no observed y. The
PSU must stay in the design (R ``na.rm = TRUE``) or the SE is wrong.

R reference values from survey 4.5 (sprintf %.15g):

```r
library(survey)
apiclus1 <- read.csv("tests/test_data/apiclus1.csv")
na <- (seq_len(nrow(apiclus1)) - 1) %% 10 == 0
apiclus1$st <- ifelse(na, NA, apiclus1$stype)
apiclus1$ti <- ifelse(na, NA, apiclus1$api00 %/% 200)
dna <- svydesign(id = ~dnum, weights = ~pw, data = apiclus1)
m <- svymean(~factor(st), dna, na.rm = TRUE); coef(m); SE(m); vcov(m)
obs <- subset(dna, !is.na(st)); degf(obs)
confint(svyciprop(~I(st == "E"), obs, method = "logit"))  # also H, M
svymean(~factor(ti), dna, na.rm = TRUE)
svyby(~factor(st), ~both, dna, svymean, na.rm = TRUE)

r1 <- as.svrepdesign(dna, type = "JK1")
svymean(~factor(st), r1, na.rm = TRUE); svymean(~factor(ti), r1, na.rm = TRUE)
svyby(~factor(st), ~both, r1, svymean, na.rm = TRUE)
```
"""

from pathlib import Path

import polars as pl
import pytest

import svy

from svy import Design, Sample


BASE_DIR = Path(__file__).parents[2]

TOL = 1e-10

ST = {
    "E": (0.810975609756098, 0.0463941779262182),
    "H": (0.0670731707317072, 0.0287011472557465),
    "M": (0.121951219512195, 0.0279104585073063),
}
ST_LOGIT = {
    "E": (0.690515543875448, 0.891889966529345),
    "H": (0.0259967110599201, 0.162241942794033),
    "M": (0.0732920117301412, 0.196080292706089),
}
TI = {
    2: (0.347560975609756, 0.108656064412381),
    3: (0.579268292682927, 0.0930614672242163),
    4: (0.073170731707317, 0.029889908256352),
}
ST_BY = {
    ("No", "E"): (0.695652173913043, 0.0925669556286492),
    ("No", "H"): (0.108695652173913, 0.043097568392548),
    ("No", "M"): (0.195652173913043, 0.0751774356307807),
    ("Yes", "E"): (0.855932203389828, 0.0396088874992838),
    ("Yes", "H"): (0.0508474576271186, 0.0244258134010057),
    ("Yes", "M"): (0.0932203389830507, 0.0305324929991536),
}

JK_ST = {
    "E": (0.810975609756098, 0.0507845427464525),
    "H": (0.0670731707317072, 0.0292981792711033),
    "M": (0.121951219512195, 0.0312644361813962),
}
JK_TI = {
    2: (0.347560975609756, 0.122558754225973),
    3: (0.579268292682927, 0.104851348530524),
    4: (0.073170731707317, 0.0316325710211068),
}
JK_ST_BY = {
    ("No", "E"): (0.695652173913043, 0.101568507822372),
    ("No", "H"): (0.108695652173913, 0.0456171350865523),
    ("No", "M"): (0.195652173913043, 0.0818245383339884),
    ("Yes", "E"): (0.85593220338983, 0.0430723958065414),
    ("Yes", "H"): (0.0508474576271186, 0.0246909664499916),
    ("Yes", "M"): (0.0932203389830507, 0.0331109955567604),
}


@pytest.fixture(scope="module")
def dna():
    data = svy.io.read_csv(BASE_DIR / "test_data/apiclus1.csv")
    na = pl.int_range(pl.len()) % 10 == 0
    data = data.with_columns(
        st=pl.when(na).then(None).otherwise(pl.col("stype")),
        ti=pl.when(na).then(None).otherwise(pl.col("api00") // 200),
        tf=pl.when(na).then(float("nan")).otherwise((pl.col("api00") // 200).cast(pl.Float64)),
    )
    return Sample(data, Design(psu="dnum", wgt="pw"))


@pytest.fixture(scope="module")
def jk1(dna):
    return dna.weighting.create_jk_wgts()


def _rows(r):
    return {r._row_key(p): p for p in r.estimates}


def _cov(r, a, b):
    keys = r.keys()
    return r.covariance[keys.index(a), keys.index(b)]


def _check(r, want):
    rows = _rows(r)
    assert sorted(rows) == sorted(want)
    for k, (est, se) in want.items():
        assert rows[k].est == pytest.approx(est, rel=TOL)
        assert rows[k].se == pytest.approx(se, rel=TOL)


class TestTaylor:
    def test_string_levels(self, dna):
        _check(dna.estimation.prop("st", drop_nulls=True), ST)

    def test_covariance(self, dna):
        r = dna.estimation.prop("st", drop_nulls=True)
        assert _cov(r, "E", "E") == pytest.approx(0.0021524197454496, rel=TOL)
        assert _cov(r, "E", "H") == pytest.approx(-0.00109859095257879, rel=TOL)
        assert _cov(r, "E", "M") == pytest.approx(-0.00105382879287081, rel=TOL)
        assert _cov(r, "H", "M") == pytest.approx(0.000274835098782743, rel=TOL)

    def test_logit_interval_and_df(self, dna):
        rows = _rows(dna.estimation.prop("st", drop_nulls=True))
        for lv, (lci, uci) in ST_LOGIT.items():
            assert rows[lv].df == 13
            assert rows[lv].lci == pytest.approx(lci, rel=TOL)
            assert rows[lv].uci == pytest.approx(uci, rel=TOL)

    def test_integer_levels(self, dna):
        _check(dna.estimation.prop("ti", drop_nulls=True), TI)

    def test_float_nan_is_missing(self, dna):
        _check(dna.estimation.prop("tf", drop_nulls=True), TI)

    def test_by(self, dna):
        _check(dna.estimation.prop("st", by="both", drop_nulls=True), ST_BY)


class TestReplication:
    def test_string_levels(self, jk1):
        _check(jk1.estimation.prop("st", drop_nulls=True, method="replication"), JK_ST)

    def test_integer_levels(self, jk1):
        _check(jk1.estimation.prop("ti", drop_nulls=True, method="replication"), JK_TI)

    def test_by(self, jk1):
        r = jk1.estimation.prop("st", by="both", drop_nulls=True, method="replication")
        _check(r, JK_ST_BY)
