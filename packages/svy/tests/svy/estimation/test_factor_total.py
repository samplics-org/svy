# tests/svy/estimation/test_factor_total.py
"""``total(as_factor=True)`` and replication ``mean(as_factor=True)``.

R reference values from survey 4.5 (sprintf %.15g):

```r
library(survey)
apiclus1 <- read.csv("tests/test_data/apiclus1.csv")
dclus1 <- svydesign(id = ~dnum, weights = ~pw, data = apiclus1)
t1 <- svytotal(~factor(stype), dclus1, deff = TRUE)
coef(t1); vcov(t1); deff(t1); confint(t1, df = degf(dclus1))
deff(svytotal(~factor(stype), dclus1, deff = "replace"))
svycontrast(t1, c("factor(stype)E" = 1, "factor(stype)M" = -1))
vcov(svyby(~factor(stype), ~both, dclus1, svytotal, covmat = TRUE))

apistrat <- read.csv("tests/test_data/apistrat.csv")
apistrat$tier <- ifelse(apistrat$api00 < 600, 1L, ifelse(apistrat$api00 < 750, 2L, 10L))
dstrat <- svydesign(id = ~1, strata = ~stype, weights = ~pw, fpc = ~fpc, data = apistrat)
svytotal(~factor(tier), dstrat, deff = TRUE)
svyby(~factor(tier), ~stype, dstrat, svytotal)

r1 <- as.svrepdesign(dclus1, type = "JK1")
svytotal(~factor(stype), r1); svymean(~factor(stype), r1)
svyby(~factor(stype), ~both, r1, svytotal, covmat = TRUE)

# every 10th row missing; PSU 413 then has no observed y
na <- (seq_len(nrow(apiclus1)) - 1) %% 10 == 0
apiclus1$st <- ifelse(na, NA, apiclus1$stype)
apiclus1$ti <- ifelse(na, NA, apiclus1$api00 %/% 200)
dna <- svydesign(id = ~dnum, weights = ~pw, data = apiclus1)
svytotal(~factor(st), dna, na.rm = TRUE); svymean(~factor(ti), dna, na.rm = TRUE)
svymean(~factor(st), as.svrepdesign(dna, type = "JK1"), na.rm = TRUE)
```
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import svy

from svy import Design, Sample, estd


BASE_DIR = Path(__file__).parents[2]

TOL = 1e-10


@pytest.fixture(scope="module")
def apiclus1():
    return svy.io.read_csv(BASE_DIR / "test_data/apiclus1.csv")


@pytest.fixture(scope="module")
def dclus1(apiclus1):
    return Sample(apiclus1, Design(psu="dnum", wgt="pw"))


@pytest.fixture(scope="module")
def dstrat():
    data = svy.io.read_csv(BASE_DIR / "test_data/apistrat.csv").with_columns(
        tier=pl.when(pl.col("api00") < 600)
        .then(1)
        .when(pl.col("api00") < 750)
        .then(2)
        .otherwise(10)
    )
    return Sample(data, Design(stratum="stype", wgt="pw", pop_size="fpc"))


@pytest.fixture(scope="module")
def jk1(dclus1):
    return dclus1.weighting.create_jk_wgts()


def _by_key(r):
    return {r._row_key(p): p for p in r.estimates}


def _cov(r, a, b):
    keys = r.keys()
    return r.covariance[keys.index(a), keys.index(b)]


def _assert_cov(r, keys, flat):
    want = np.array(flat).reshape(len(keys), len(keys))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            assert _cov(r, a, b) == pytest.approx(want[i, j], rel=TOL)


CLUS1_KEYS = ["E", "H", "M"]
CLUS1_EST = [4873.96746826172, 473.857948303222, 846.174907684325]
CLUS1_SE = [1346.72892173089, 160.295355947307, 169.234981536865]
CLUS1_VCOV = [
    1813678.78862645, 32650.1460323927, 88376.3351252734,
    32650.1460323927, 25694.6011382739, 6955.54489411874,
    88376.3351252734, 6955.54489411874, 28640.478975783,
]  # fmt: skip

BY_KEYS = [("No", "E"), ("Yes", "E"), ("No", "H"), ("Yes", "H"), ("No", "M"), ("Yes", "M")]
BY_EST = [
    1083.10388183594, 3790.86358642578, 236.928974151611,
    236.928974151611, 372.316959381103, 473.857948303222,
]  # fmt: skip
BY_SE = [
    376.904199658061, 1004.30863761453, 109.303116286308,
    83.8890626604538, 115.845199074682, 167.778125320908,
]  # fmt: skip
BY_VCOV = [
    142056.775719884, 331493.086659706, 20948.464622287, 81.8299399308086, -12847.300569137, 34532.2346508013,
    331493.086659706, 1008635.83958715, 21766.7640215951, -10146.9125514203, -13665.5999684451, 80357.0010120541,
    20948.464622287, 21766.7640215951, 11947.1712298981, 3355.02753716315, -163.659879861617, 6710.05507432631,
    81.8299399308086, -10146.9125514203, 3355.02753716315, 7037.37483404954, -1391.10897882375, 1800.25867847779,
    -12847.300569137, -13665.5999684451, -163.659879861617, -1391.10897882375, 13420.1101486526, -6464.56525453388,
    34532.2346508013, 80357.0010120541, 6710.05507432631, 1800.25867847779, -6464.56525453388, 28149.4993361982,
]  # fmt: skip


# =============================================================================
# Taylor
# =============================================================================


class TestTaylorFactorTotal:
    def test_levels_estimates_and_se(self, dclus1):
        r = dclus1.estimation.total("stype", as_factor=True)
        rows = _by_key(r)
        assert sorted(rows) == CLUS1_KEYS
        for k, est, se in zip(CLUS1_KEYS, CLUS1_EST, CLUS1_SE):
            assert rows[k].est == pytest.approx(est, rel=TOL)
            assert rows[k].se == pytest.approx(se, rel=TOL)

    def test_covariance(self, dclus1):
        r = dclus1.estimation.total("stype", as_factor=True)
        _assert_cov(r, CLUS1_KEYS, CLUS1_VCOV)

    def test_wald_t_interval(self, dclus1):
        # counts are not shares: no logit interval, no [0, 1] bounds
        r = dclus1.estimation.total("stype", as_factor=True)
        e = _by_key(r)["E"]
        assert e.df == 14
        assert e.lci == pytest.approx(1985.52120469941, rel=TOL)
        assert e.uci == pytest.approx(7762.41373182403, rel=TOL)

    @pytest.mark.parametrize(
        "ref, want",
        [
            ("wor", [52.8675113300029, 1.77779930649152, 1.1869673512099]),
            ("wr", [51.3055555555555, 1.72527472527472, 1.15189873417722]),
        ],
    )
    def test_deff(self, dclus1, ref, want):
        rows = _by_key(dclus1.estimation.total("stype", as_factor=True, deff=ref))
        for k, d in zip(CLUS1_KEYS, want):
            assert rows[k].deff == pytest.approx(d, rel=TOL)

    def test_contrast_between_levels(self, dclus1):
        r = dclus1.estimation.total("stype", as_factor=True)
        c = r.contrast(estd("E") - estd("M")).estimates[0]
        assert c.est == pytest.approx(4027.79256057739, rel=TOL)
        assert c.se == pytest.approx(1290.56832339543, rel=TOL)

    def test_by(self, dclus1):
        r = dclus1.estimation.total("stype", by="both", as_factor=True)
        rows = _by_key(r)
        assert sorted(rows) == sorted(BY_KEYS)
        for k, est, se in zip(BY_KEYS, BY_EST, BY_SE):
            assert rows[k].est == pytest.approx(est, rel=TOL)
            assert rows[k].se == pytest.approx(se, rel=TOL)
        _assert_cov(r, BY_KEYS, BY_VCOV)

    def test_stratified_fpc_integer_levels(self, dstrat):
        r = dstrat.estimation.total("tier", as_factor=True, deff="wor")
        rows = _by_key(r)
        assert sorted(rows) == [1, 2, 10]
        assert all(isinstance(k, int) for k in rows)
        want = {
            1: (2023.88998985291, 220.152515066912, 1.18091987367229),
            2: (2525.75998878479, 230.592280981214, 1.1801772855841),
            10: (1644.34997940064, 213.048835018579, 1.24765244829902),
        }
        for k, (est, se, deff) in want.items():
            assert rows[k].est == pytest.approx(est, rel=TOL)
            assert rows[k].se == pytest.approx(se, rel=TOL)
            assert rows[k].deff == pytest.approx(deff, rel=TOL)
        assert _cov(r, 1, 2) == pytest.approx(-28125.0619178162, rel=TOL)
        assert _cov(r, 2, 10) == pytest.approx(-25047.7381303027, rel=TOL)

    def test_stratified_by_domain(self, dstrat):
        rows = _by_key(dstrat.estimation.total("tier", by="stype", as_factor=True))
        want = {
            ("E", 1): (1370.50997161865, 203.160922698593),
            ("H", 2): (377.500009536743, 52.112272189212),
            ("M", 10): (183.240005493164, 54.4824613457954),
        }
        for k, (est, se) in want.items():
            assert rows[k].est == pytest.approx(est, rel=TOL)
            assert rows[k].se == pytest.approx(se, rel=TOL)

    def test_numeric_total_unchanged(self, dclus1):
        r = dclus1.estimation.total("api00")
        assert len(r.estimates) == 1
        assert r.estimates[0].y_level is None


# =============================================================================
# Replication
# =============================================================================


class TestReplicationFactor:
    def test_total(self, jk1):
        r = jk1.estimation.total("stype", as_factor=True, method="replication")
        rows = _by_key(r)
        assert sorted(rows) == CLUS1_KEYS
        for k, est, se in zip(CLUS1_KEYS, CLUS1_EST, CLUS1_SE):
            assert rows[k].est == pytest.approx(est, rel=TOL)
            assert rows[k].se == pytest.approx(se, rel=TOL)
        _assert_cov(r, CLUS1_KEYS, CLUS1_VCOV)
        assert rows["E"].lci == pytest.approx(1985.52120469941, rel=TOL)

    def test_total_by(self, jk1):
        r = jk1.estimation.total("stype", by="both", as_factor=True, method="replication")
        rows = _by_key(r)
        for k, est, se in zip(BY_KEYS, BY_EST, BY_SE):
            assert rows[k].est == pytest.approx(est, rel=TOL)
            assert rows[k].se == pytest.approx(se, rel=TOL)
        _assert_cov(r, BY_KEYS, BY_VCOV)

    def test_mean(self, jk1):
        r = jk1.estimation.mean("stype", as_factor=True, method="replication")
        rows = _by_key(r)
        want = {
            "E": (0.786885245901639, 0.0519586712418233),
            "H": (0.076502732240437, 0.028059292178515),
            "M": (0.136612021857923, 0.0335079420245988),
        }
        assert sorted(rows) == CLUS1_KEYS
        for k, (est, se) in want.items():
            assert rows[k].est == pytest.approx(est, rel=TOL)
            assert rows[k].se == pytest.approx(se, rel=TOL)
        assert _cov(r, "E", "H") == pytest.approx(-0.00118212260802563, rel=TOL)
        assert _cov(r, "H", "M") == pytest.approx(0.00039479873046636, rel=TOL)

    def test_integer_levels_keep_their_type(self, apiclus1):
        data = apiclus1.with_columns(
            tier=pl.when(pl.col("api00") < 600)
            .then(1)
            .when(pl.col("api00") < 750)
            .then(2)
            .otherwise(10)
        )
        rep = Sample(data, Design(psu="dnum", wgt="pw")).weighting.create_bs_wgts(
            n_reps=20, rstate=7
        )
        for fn in ("mean", "total"):
            r = getattr(rep.estimation, fn)("tier", as_factor=True, method="replication")
            assert sorted(_by_key(r)) == [1, 2, 10]


# =============================================================================
# Missing y: out of the domain, not dropped (R na.rm = TRUE)
# =============================================================================


@pytest.fixture(scope="module")
def dna(apiclus1):
    na = pl.int_range(pl.len()) % 10 == 0
    data = apiclus1.with_columns(
        st=pl.when(na).then(None).otherwise(pl.col("stype")),
        ti=pl.when(na).then(None).otherwise(pl.col("api00") // 200),
    )
    return Sample(data, Design(psu="dnum", wgt="pw"))


class TestMissingY:
    def _check(self, r, want):
        rows = _by_key(r)
        assert sorted(rows) == sorted(want)
        for k, (est, se) in want.items():
            assert rows[k].est == pytest.approx(est, rel=TOL)
            assert rows[k].se == pytest.approx(se, rel=TOL)

    ST_TOTAL = {
        "E": (4501.65050888061, 1253.31860550772),
        "H": (372.316959381103, 144.169361316562),
        "M": (676.93992614746, 137.189234942418),
    }

    def test_taylor_total(self, dna):
        self._check(dna.estimation.total("st", as_factor=True, drop_nulls=True), self.ST_TOTAL)

    def test_taylor_mean_integer(self, dna):
        want = {
            2: (0.347560975609756, 0.108656064412381),
            3: (0.579268292682927, 0.0930614672242163),
            4: (0.073170731707317, 0.029889908256352),
        }
        self._check(dna.estimation.mean("ti", as_factor=True, drop_nulls=True), want)

    def test_replication(self, dna):
        rep = dna.weighting.create_jk_wgts()
        kw = {"as_factor": True, "drop_nulls": True, "method": "replication"}
        self._check(rep.estimation.total("st", **kw), self.ST_TOTAL)
        want = {
            "E": (0.810975609756098, 0.0507845427464525),
            "H": (0.0670731707317072, 0.0292981792711033),
            "M": (0.121951219512195, 0.0312644361813962),
        }
        self._check(rep.estimation.mean("st", **kw), want)
