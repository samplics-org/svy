# tests/svy/categorical/test_singleton_replicates.py
"""Singleton strata on a design with replicate weights.

tabulate, ttest and ranktest compute a Taylor variance from the stratum and PSU
columns whether or not the design carries replicate weights, so a singleton
stratum without a rule raises SINGLETON_ERROR on a replicate design too. The
error says the replicates are not used and where a replication variance is.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.repwgts import BrrWgts
from svy.errors.singleton_errors import SingletonError


N_REPS = 4


def _data() -> pl.DataFrame:
    rng = np.random.default_rng(20260930)
    # Stratum 3 has one PSU.
    st = [1] * 12 + [2] * 12 + [3] * 6
    psu = [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6
    n = len(st)
    w = rng.uniform(1, 3, n)
    df = pl.DataFrame(
        {
            "st": st,
            "psu": psu,
            "w": w,
            "y": rng.normal(size=n),
            "g": [0, 1] * (n // 2),
            "c": rng.integers(0, 3, n),
            "d": [0, 0, 1] * (n // 3),
        }
    )
    return df.with_columns(
        pl.Series(f"r{i + 1}", w * rng.uniform(0.5, 1.5, n)) for i in range(N_REPS)
    )


def _sample(*, replicates: bool, singleton=None) -> svy.Sample:
    design = svy.Design(
        stratum="st",
        psu="psu",
        wgt="w",
        rep_wgts=BrrWgts(prefix="r", n_reps=N_REPS) if replicates else None,
        singleton=singleton,
    )
    return svy.Sample(_data(), design)


CALLS = {
    "tabulate": lambda s: s.categorical.tabulate("c"),
    "tabulate-two-way": lambda s: s.categorical.tabulate("c", "g"),
    "tabulate-where": lambda s: s.categorical.tabulate("c", where=svy.col("d") == 1),
    "ttest-one-sample": lambda s: s.categorical.ttest("y"),
    "ttest-two-group": lambda s: s.categorical.ttest("y", group="g"),
    "ttest-by": lambda s: s.categorical.ttest("y", by="d"),
    "ttest-where": lambda s: s.categorical.ttest("y", group="g", where=svy.col("d") == 1),
    "ranktest": lambda s: s.categorical.ranktest("y", group="g", method="kruskal-wallis"),
    "ranktest-by": lambda s: s.categorical.ranktest(
        "y", group="g", method="kruskal-wallis", by="d"
    ),
}

WHERE = {name: f"Sample.categorical.{name.split('-')[0]}" for name in CALLS}

USE_REPLICATES = {
    "tabulate": "sample.estimation.prop(..., method='replication')",
    "ttest": "sample.estimation.mean(..., method='replication')",
    "ranktest": "ranktest has no replication variance yet.",
}


def _raised(call, sample) -> SingletonError:
    with pytest.raises(SingletonError) as err:
        call(sample)
    return err.value


@pytest.mark.parametrize("name", list(CALLS))
def test_replicate_design_raises_and_says_the_replicates_are_not_used(name):
    err = _raised(CALLS[name], _sample(replicates=True))
    assert err.code == "SINGLETON_ERROR"
    assert err.where == WHERE[name]
    assert f"replicate weights (BRR, n_reps={N_REPS})" in err.detail
    assert "Taylor linearization" in err.detail
    assert "st=3" in err.detail
    assert USE_REPLICATES[name.split("-")[0]] in err.hint
    assert 'svy.Singleton("center")' in err.hint


@pytest.mark.parametrize("name", list(CALLS))
def test_taylor_design_error_is_unchanged(name):
    err = _raised(CALLS[name], _sample(replicates=False))
    plain = SingletonError.from_singletons([], where=err.where)
    assert err.code == "SINGLETON_ERROR"
    assert "replicate" not in err.detail
    assert err.hint == plain.hint


@pytest.mark.parametrize("name", list(CALLS))
def test_with_a_rule_the_replicates_do_not_change_the_result(name):
    rep = CALLS[name](_sample(replicates=True, singleton="center"))
    taylor = CALLS[name](_sample(replicates=False, singleton="center"))
    assert rep.to_polars().equals(taylor.to_polars())
    assert [f.code for f in rep.findings] == [f.code for f in taylor.findings]


def test_domain_singleton_findings_are_the_taylor_ones():
    # Stratum 2 keeps both PSUs, but only PSU 3 has rows in d == 1 once
    # PSU 4's are moved out of it.
    def domain_singleton(replicates):
        s = _sample(replicates=replicates, singleton="center")
        where = (svy.col("d") == 1) & (svy.col("psu") != 4)
        return {
            name: [f.code for f in call(s).findings]
            for name, call in {
                "tabulate": lambda s: s.categorical.tabulate("c", where=where),
                "ttest": lambda s: s.categorical.ttest("y", where=where),
                "ranktest": lambda s: s.categorical.ranktest(
                    "y", group="g", method="kruskal-wallis", where=where
                ),
            }.items()
        }

    rep = domain_singleton(True)
    assert rep == domain_singleton(False)
    assert all("DOMAIN_SINGLETON_PSU" in codes for codes in rep.values())


def test_estimation_taylor_points_to_method_replication():
    s = _sample(replicates=True)
    with pytest.raises(SingletonError) as err:
        s.estimation.mean("y")
    assert err.value.where == "estimation"
    assert "Pass method='replication'" in err.value.hint
    assert s.estimation.mean("y", method="replication").estimates[0].se > 0


def test_singleton_introduced_by_a_filter_after_creating_replicates():
    df = _data().filter(pl.col("st") != 3)
    s = svy.Sample(df, svy.Design(stratum="st", psu="psu", wgt="w"))
    s = s.weighting.create_brr_wgts(rep_prefix="brr")
    s = s.wrangling.filter_records(svy.col("psu") != 4)
    assert s.design.rep_wgts is not None and s.n_singletons == 1
    err = _raised(CALLS["ttest-two-group"], s)
    assert "replicate weights (BRR" in err.detail
    assert "sample.estimation.mean(..., method='replication')" in err.hint
