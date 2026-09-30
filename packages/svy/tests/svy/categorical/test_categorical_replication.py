# tests/svy/categorical/test_categorical_replication.py
"""tabulate, ttest and ranktest with method="replication".

Reference values: R survey 4.5 on svrepdesign(type="other") with svy's
replicate weights, per-replicate coefficients as rscales, and svy's design df
as degf (golden/categorical_replication_r.R). Cells are svymean/svytotal of
the cell indicators, the tests svychisq(statistic="F"/"Chisq"), svyttest and
svyranktest; `where=` is `subset()` and `by=` a loop over subsets.

The frame (tests/test_data/categorical_rep_30092026.csv) has 6 strata of 2
PSUs and svy-generated BRR, Fay-BRR (0.3), JKn, JK1, bootstrap (50) and SDR
(8) replicate weights.
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.repwgts import BootstrapWgts, BrrWgts, JackknifeWgts, SdrWgts
from svy.errors import MethodError
from svy.errors.singleton_errors import SingletonError


HERE = Path(__file__).parent
DATA = HERE.parents[1] / "test_data" / "categorical_rep_30092026.csv"
R = json.loads((HERE / "golden" / "categorical_replication_r.json").read_text())
REL = 1e-8

REPS = {
    "brr": BrrWgts(prefix="brr_", n_reps=8, df=6),
    "fay": BrrWgts(prefix="fay_", n_reps=8, fay_coef=0.3, df=6),
    "jkn": JackknifeWgts(prefix="jkn_", n_reps=12, kind="jkn", stratum="st", psu="psu", df=6),
    "jk1": JackknifeWgts(prefix="jk1_", n_reps=12, kind="jk1", df=11),
    "bs": BootstrapWgts(prefix="bs_", n_reps=50, df=49),
    "sdr": SdrWgts(prefix="sdr_", n_reps=8, df=8),
}
METHODS = list(REPS)
REGIONS = ["north", "south"]
W = svy.col("dom") == 1


def _squared(r, n):
    return (r / n) ** 2


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return pl.read_csv(DATA)


def _sample(frame: pl.DataFrame, rep: str, **design) -> svy.Sample:
    return svy.Sample(
        frame, svy.Design(stratum="st", psu="psu", wgt="w", rep_wgts=REPS[rep], **design)
    )


def _flat(values) -> np.ndarray:
    return np.hstack([np.ravel(v) for v in values]).astype(float)


def _close(got, want) -> None:
    np.testing.assert_allclose(_flat(got), _flat(want), rtol=REL, atol=0)


# ════════════════════════════════════════════════════════════════════════
# R parity
# ════════════════════════════════════════════════════════════════════════


TABLES = {
    "oneway": (("a",), {}),
    "oneway_where": (("a",), {"where": W}),
    "twoway": (("a", "b"), {}),
    "twoway_where": (("a", "b"), {"where": W}),
}


def _r_cell_keys(names: list[str], two_way: bool) -> list[tuple[str, ...]]:
    # R names one-way cells "ax" and two-way cells "interaction(a, b)x.0".
    if not two_way:
        return [(n[1:], "") for n in names]
    return [tuple(n.split(")")[1].split(".")) for n in names]


@pytest.mark.parametrize("key", list(TABLES))
@pytest.mark.parametrize("rep", METHODS)
def test_tabulate_matches_r(frame, rep, key):
    args, kw = TABLES[key]
    cat = _sample(frame, rep).categorical
    props = cat.tabulate(*args, method="replication", **kw)
    counts = cat.tabulate(*args, units="count", method="replication", **kw)
    r = R[rep][key]

    pos = {(c.rowvar, c.colvar): i for i, c in enumerate(props.estimates)}
    order = [pos[k] for k in _r_cell_keys(r["names"], len(args) == 2)]
    assert len(order) == len(props.estimates)
    _close([props.estimates[i].est for i in order], r["prop"])
    _close([props.estimates[i].se for i in order], r["prop_se"])
    _close([counts.estimates[i].est for i in order], r["total"])
    _close([counts.estimates[i].se for i in order], r["total_se"])

    if len(args) == 2:
        rc = R[rep]["chisq_where" if kw else "chisq"]
        st = props.stats
        _close(
            [st.f.value, st.f.df_num, st.f.df_den, st.f.p_value],
            [rc["f"], rc["ndf"], rc["ddf"], rc["f_p"]],
        )
        _close([st.chisq.value, st.chisq.p_value], [rc["chisq"], rc["chisq_p"]])


def _one(res, r) -> None:
    est = res.estimates[0]
    _close(
        [est.est, est.se, res.stats.t, res.stats.df, res.stats.p_value],
        [r["est"], r["se"], r["t"], r["df"], r["p"]],
    )


def _two(res, r) -> None:
    d = res.diff[0]
    _close(
        [d.diff, d.se, res.stats.t, res.stats.df, res.stats.p_value],
        [r["diff"], r["se"], r["t"], r["df"], r["p"]],
    )
    _close([e.est for e in res.estimates], r["means"])
    _close([e.se for e in res.estimates], r["mean_se"])


@pytest.mark.parametrize("rep", METHODS)
class TestTTest:
    def test_one_sample(self, frame, rep):
        _one(_sample(frame, rep).categorical.ttest("y", method="replication"), R[rep]["tt1"])

    def test_one_sample_mean_h0(self, frame, rep):
        res = _sample(frame, rep).categorical.ttest("y", mean_h0=11, method="replication")
        _one(res, R[rep]["tt1_h0"])

    def test_paired(self, frame, rep):
        res = _sample(frame, rep).categorical.ttest("y", y_pair="y2", method="replication")
        r = R[rep]["tt_paired"]
        _close([res.estimates[0].est, res.stats.t, res.stats.p_value], [r["est"], r["t"], r["p"]])

    def test_two_sample(self, frame, rep):
        res = _sample(frame, rep).categorical.ttest("y", group="grp", method="replication")
        _two(res, R[rep]["tt2"])

    def test_two_sample_where(self, frame, rep):
        cat = _sample(frame, rep).categorical
        _two(cat.ttest("y", group="grp", where=W, method="replication"), R[rep]["tt2_where"])

    def test_by(self, frame, rep):
        cat = _sample(frame, rep).categorical
        one = cat.ttest("y", by="region", method="replication")
        two = cat.ttest("y", group="grp", by="region", method="replication")
        two_where = cat.ttest("y", group="grp", by="region", where=W, method="replication")
        for i, reg in enumerate(REGIONS):
            _one(one[i], R[rep][f"tt1_by_{reg}"])
            _two(two[i], R[rep][f"tt2_by_{reg}"])
            _two(two_where[i], R[rep][f"tt2_where_by_{reg}"])


def _rank2(res, r) -> None:
    _close(
        [res.diff[0].diff, res.stats.value, res.stats.df, res.stats.p_value],
        [r["delta"], r["t"], r["df"], r["p"]],
    )


def _rankk(res, r) -> None:
    st = res.stats
    _close(
        [st.value * st.df_num, st.df_num, st.df_den, st.p_value],
        [r["chisq"], r["ndf"], r["ddf"], r["p"]],
    )


@pytest.mark.parametrize("rep", METHODS)
class TestRankTest:
    @pytest.mark.parametrize(
        "score, key",
        [("kruskal-wallis", "rk_kw"), ("vander-waerden", "rk_vdw"), ("median", "rk_median")],
    )
    def test_two_sample(self, frame, rep, score, key):
        cat = _sample(frame, rep).categorical
        _rank2(cat.ranktest("y", group="grp", score=score, method="replication"), R[rep][key])

    def test_custom_score(self, frame, rep):
        cat = _sample(frame, rep).categorical
        res = cat.ranktest("y", group="grp", score_fn=_squared, method="replication")
        _rank2(res, R[rep]["rk_custom"])

    def test_where(self, frame, rep):
        cat = _sample(frame, rep).categorical
        res = cat.ranktest("y", group="grp", score="kruskal-wallis", where=W, method="replication")
        _rank2(res, R[rep]["rk_where"])

    def test_k_sample(self, frame, rep):
        cat = _sample(frame, rep).categorical
        _rankk(
            cat.ranktest("y", group="k3", score="kruskal-wallis", method="replication"),
            R[rep]["rkk_kw"],
        )
        _rankk(
            cat.ranktest("y", group="k3", score="kruskal-wallis", where=W, method="replication"),
            R[rep]["rkk_where"],
        )

    def test_by(self, frame, rep):
        cat = _sample(frame, rep).categorical
        two = cat.ranktest(
            "y", group="grp", score="kruskal-wallis", by="region", method="replication"
        )
        k = cat.ranktest(
            "y", group="k3", score="kruskal-wallis", by="region", method="replication"
        )
        custom = cat.ranktest(
            "y", group="grp", score_fn=_squared, by="region", method="replication"
        )
        for i, reg in enumerate(REGIONS):
            _rank2(two[i], R[rep][f"rk_by_{reg}"])
            _rankk(k[i], R[rep][f"rkk_by_{reg}"])
            _rank2(custom[i], R[rep][f"rkc_by_{reg}"])


# ════════════════════════════════════════════════════════════════════════
# Method resolution
# ════════════════════════════════════════════════════════════════════════


CALLS = {
    "tabulate": lambda cat, **kw: cat.tabulate("a", "b", **kw),
    "ttest": lambda cat, **kw: cat.ttest("y", group="grp", **kw),
    "ranktest": lambda cat, **kw: cat.ranktest("y", group="grp", score="kruskal-wallis", **kw),
}


def _values(res) -> pl.DataFrame:
    return res.to_polars()


@pytest.mark.parametrize("name", list(CALLS))
def test_default_is_taylor_on_a_replicate_design(frame, name):
    rep = _sample(frame, "brr").categorical
    taylor = svy.Sample(frame, svy.Design(stratum="st", psu="psu", wgt="w")).categorical
    default = _values(CALLS[name](rep))
    assert default.equals(_values(CALLS[name](rep, method="taylor")))
    assert default.equals(_values(CALLS[name](taylor)))
    assert not default.equals(_values(CALLS[name](rep, method="replication")))


@pytest.mark.parametrize("name", list(CALLS))
def test_replication_without_replicate_weights_raises(frame, name):
    cat = svy.Sample(frame, svy.Design(stratum="st", psu="psu", wgt="w")).categorical
    with pytest.raises(ValueError, match="Replication requires rep_wgts"):
        CALLS[name](cat, method="replication")


@pytest.mark.parametrize("name", list(CALLS))
def test_unknown_method_raises(frame, name):
    with pytest.raises(ValueError, match="Unknown estimation method"):
        CALLS[name](_sample(frame, "brr").categorical, method="delta")


@pytest.mark.parametrize("name", list(CALLS))
def test_replication_needs_no_singleton_rule(frame, name):
    # PSU 12 leaves stratum 6 with one PSU. The replicate weights carry the
    # design, so replication runs where Taylor raises for the missing rule.
    s = _sample(frame.filter(pl.col("psu") != 12), "bs")
    assert s.n_singletons == 1
    with pytest.raises(SingletonError):
        CALLS[name](s.categorical)
    res = CALLS[name](s.categorical, method="replication")
    assert not res.findings


def test_replication_has_no_domain_singleton_findings(frame):
    # Taylor flags the strata where only one PSU has domain rows.
    s = _sample(frame, "brr", singleton="center")
    where = svy.col("psu") % 2 == 1
    for name, call in CALLS.items():
        assert call(s.categorical, where=where).findings, name
        assert not call(s.categorical, where=where, method="replication").findings, name


@pytest.mark.parametrize("name", list(CALLS))
def test_taylor_without_design_warns(frame, name):
    s = svy.Sample(frame, svy.Design(wgt="w", rep_wgts=REPS["bs"]))
    with pytest.warns(Warning, match="Taylor variance on a design with no stratum or psu"):
        CALLS[name](s.categorical)


def test_replicate_df_does_not_shrink_on_a_domain(frame):
    cat = _sample(frame, "jk1").categorical
    where = svy.col("region") == "north"
    tbl = cat.tabulate("a", "b", where=where, method="replication")
    assert tbl.stats.f.df_den == pytest.approx(11 * tbl.stats.f.df_num)
    assert cat.ttest("y", where=where, method="replication").stats.df == 10
    k = cat.ranktest("y", group="k3", score="median", where=where, method="replication")
    assert k.stats.df_den == 9


def test_percent_and_count_total_scale_the_proportions(frame):
    cat = _sample(frame, "sdr").categorical
    prop = cat.tabulate("a", method="replication")
    pct = cat.tabulate("a", units="percent", method="replication")
    scaled = cat.tabulate("a", count_total=5000, method="replication")
    for p, q, n in zip(prop.estimates, pct.estimates, scaled.estimates):
        assert q.est == pytest.approx(100 * p.est) and q.se == pytest.approx(100 * p.se)
        assert n.est == pytest.approx(5000 * p.est) and n.se == pytest.approx(5000 * p.se)


def test_ci_uses_the_replicate_df(frame):
    from scipy.stats import t

    res = _sample(frame, "bs").categorical.ttest("y", method="replication")
    est = res.estimates[0]
    assert est.uci - est.est == pytest.approx(t.ppf(0.975, 48) * est.se)


# ════════════════════════════════════════════════════════════════════════
# ranktest: score= and method=
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("old", ["kruskal-wallis", "median", "vdw"])
def test_ranktest_rank_score_passed_as_method_points_to_score(frame, old):
    cat = _sample(frame, "brr").categorical
    with pytest.raises(MethodError, match="score=") as err:
        cat.ranktest("y", group="grp", method=old)
    assert err.value.code == "INVALID_CHOICE"
    assert err.value.param == "method"
