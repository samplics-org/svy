"""SampleSize records every goal's inputs on .target, and keeps stratum keys as given."""

import math

import pytest

import svy

from svy.errors import MethodError
from svy.size import (
    TargetAllocation,
    TargetMean,
    TargetProp,
    TargetTwoMeans,
    TargetTwoProps,
)


# =============================================================================
# .target: every argument of the goal
# =============================================================================


def test_estimate_prop_records_every_argument():
    ss = svy.SampleSize().estimate_prop(
        p=0.3, moe=0.05, pop_size=5000, method="fleiss", alpha=0.1, deff=1.5, resp_rate=0.8
    )
    assert ss.target == TargetProp(
        p=0.3, moe=0.05, alpha=0.1, method="fleiss", pop_size=5000, deff=1.5, resp_rate=0.8
    )


def test_estimate_prop_defaults_are_recorded():
    assert svy.SampleSize().estimate_prop(p=0.3, moe=0.05).target == TargetProp(p=0.3, moe=0.05)


def test_estimate_mean_records_every_argument():
    ss = svy.SampleSize().estimate_mean(
        sigma=12, moe=1.5, pop_size=10_000, alpha=0.01, deff=2.0, resp_rate=0.9
    )
    assert ss.target == TargetMean(
        sigma=12, moe=1.5, alpha=0.01, pop_size=10_000, deff=2.0, resp_rate=0.9
    )


def test_compare_props_records_every_argument():
    ss = svy.SampleSize().compare_props(
        p1=0.3,
        p2=0.4,
        pop_size=20_000,
        two_sides=False,
        delta=0.01,
        alloc_ratio=2.0,
        method="wald",
        alpha=0.1,
        power=0.9,
        var_mode="pooled-prop",
        deff=1.2,
        resp_rate=0.85,
    )
    assert ss.target == TargetTwoProps(
        p1=0.3,
        p2=0.4,
        alloc_ratio=2.0,
        alpha=0.1,
        power=0.9,
        method="wald",
        var_mode="pooled-prop",
        two_sides=False,
        delta=0.01,
        pop_size=20_000,
        deff=1.2,
        resp_rate=0.85,
    )


def test_compare_means_records_every_argument_and_a_missing_sigma2():
    ss = svy.SampleSize().compare_means(mu1=10, mu2=12, sigma1=3)
    assert ss.target == TargetTwoMeans(mu1=10, mu2=12, sigma1=3, sigma2=None)
    ss = svy.SampleSize().compare_means(
        mu1=10, mu2=12, sigma1=3, sigma2=4, two_sides=False, delta=0.5, deff=1.3
    )
    assert (ss.target.sigma2, ss.target.two_sides, ss.target.delta, ss.target.deff) == (
        4,
        False,
        0.5,
        1.3,
    )


def test_stratified_inputs_are_recorded_per_stratum():
    p = {"a": 0.3, "b": 0.5}
    ss = svy.SampleSize().estimate_prop(p=p, moe=0.05, pop_size={"a": 1000, "b": 2000})
    assert ss.target.p == p
    assert ss.target.pop_size == {"a": 1000, "b": 2000}
    assert ss.target.moe == 0.05  # a scalar broadcast across strata stays a scalar


def test_recorded_inputs_are_copies():
    p = {"a": 0.3, "b": 0.5}
    ss = svy.SampleSize().estimate_prop(p=p, moe=0.05)
    p["a"] = 0.9
    assert ss.target.p == {"a": 0.3, "b": 0.5}


def test_allocate_records_its_target():
    ss = svy.SampleSize().allocate(30, pop_size={"a": 100, "b": 200}, power=0.5)
    assert ss.target == TargetAllocation(n=30, method="proportional", power=0.5)
    assert ss.param is None


def test_the_last_goal_owns_target_size_and_allocation():
    ss = svy.SampleSize().allocate(30, pop_size={"a": 100, "b": 200})
    ss.estimate_mean(sigma=12, moe=1.5)
    assert isinstance(ss.target, TargetMean)
    assert ss.allocation is None
    ss.allocate(pop_size={"a": 100, "b": 200})
    assert isinstance(ss.target, TargetAllocation)
    assert ss.target.from_goal == "mean"


def test_empty_sample_size():
    ss = svy.SampleSize()
    assert (ss.target, ss.size, ss.allocation, ss.n, ss.param) == (None, None, None, None, None)
    assert ss.to_polars().height == 0


# =============================================================================
# Stratum keys: kept as given, in the given order
# =============================================================================


def test_tuple_keys_are_kept():
    pop = {("N", "u"): 1000, ("S", "r"): 2000}
    ss = svy.SampleSize().estimate_prop(p={k: 0.3 for k in pop}, moe=0.05, pop_size=pop)
    assert list(ss.n) == list(pop)
    assert [s.stratum for s in ss.size] == list(pop)
    assert ss.to_polars()["stratum"].to_list() == ["N, u", "S, r"]


def test_input_order_is_kept():
    pop = {"z": 1000, "a": 2000, "m": 1500}
    ss = svy.SampleSize().estimate_mean(sigma=10, moe=1, pop_size=pop)
    assert list(ss.n) == ["z", "a", "m"]


def test_integer_keys_including_zero():
    pop = {0: 1000, 1: 2000}
    ss = svy.SampleSize().estimate_prop(p=0.3, moe=0.05, pop_size=pop)
    assert list(ss.n) == [0, 1]
    assert ss.to_polars()["stratum"].to_list() == ["0", "1"]
    assert "overall" not in repr(ss.to_polars())
    alloc = svy.SampleSize().allocate(30, pop_size=pop)
    assert alloc.n == {0: 10, 1: 20}
    assert alloc.to_polars()["stratum"].to_list() == ["0", "1"]


def test_unstratified_n_is_a_number_and_comparisons_a_pair():
    assert isinstance(svy.SampleSize().estimate_prop(p=0.3, moe=0.05).n, float)
    n = svy.SampleSize().compare_props(p1=0.3, p2=0.4).n
    assert isinstance(n, tuple) and len(n) == 2


def test_stratified_comparison_keys_are_kept():
    ss = svy.SampleSize().compare_props(p1={("a", 1): 0.3, ("b", 2): 0.2}, p2=0.4)
    assert list(ss.n) == [("a", 1), ("b", 2)]
    assert all(isinstance(v, tuple) and len(v) == 2 for v in ss.n.values())
    assert set(ss.to_polars()["stratum"].to_list()) == {"a, 1", "b, 2"}


def test_size_n_feeds_selection_with_tuple_keys():
    import polars as pl

    df = pl.DataFrame({"reg": ["N"] * 60 + ["S"] * 60, "urb": ["u", "r"] * 60})
    frame = svy.Sample(df, svy.Design(stratum=["reg", "urb"]))
    pop = {("N", "u"): 30, ("N", "r"): 30, ("S", "u"): 30, ("S", "r"): 30}
    n = svy.SampleSize().estimate_prop(p=0.5, moe=0.4, pop_size=pop).n
    n = {k: math.ceil(v) for k, v in n.items()}
    drawn = frame.sampling.srs(n, rstate=1)
    assert drawn.data.height == sum(n.values())


# =============================================================================
# allocate: keys and inputs at the edges
# =============================================================================


def test_keys_that_print_alike_are_refused():
    with pytest.raises(MethodError, match="distinct"):
        svy.SampleSize().allocate(10, pop_size={1: 100, "1": 100})


def test_a_single_stratum_takes_all():
    assert svy.SampleSize().allocate(10, pop_size={"only": 100}).n == {"only": 10}


def test_an_empty_stratum_gets_zero():
    assert svy.SampleSize().allocate(10, pop_size={"a": 100, "b": 0}).n == {"a": 10, "b": 0}


def test_every_stratum_empty_is_refused():
    with pytest.raises(MethodError):
        svy.SampleSize().allocate(10, pop_size={"a": 0, "b": 0})


def test_n_equal_to_the_population():
    assert svy.SampleSize().allocate(300, pop_size={"a": 100, "b": 200}).n == {
        "a": 100,
        "b": 200,
    }


def test_size_totals_can_be_fractional():
    out = svy.SampleSize().allocate(10, pop_size={"a": 2.5, "b": 7.5}, cap_at_population=False)
    assert out.n == {"a": 3, "b": 7} or out.n == {"a": 2, "b": 8}
    assert sum(out.n.values()) == 10
