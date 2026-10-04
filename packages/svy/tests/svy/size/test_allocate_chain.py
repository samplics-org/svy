"""SampleSize.allocate() without n splits the overall n of the goal before it."""

import math

import pytest

import svy

from svy.errors import MethodError


POP = {"North": 52_000, "South": 31_000}


def test_estimate_mean_then_allocate():
    ss = svy.SampleSize().estimate_mean(sigma=12, moe=1.5, deff=1.8, resp_rate=0.9)
    total = ss.n
    ss.allocate(pop_size=POP)
    assert sum(ss.n.values()) == math.ceil(total)
    assert ss.target.n == math.ceil(total)
    assert (ss.target.from_goal, ss.target.from_n) == ("mean", total)
    assert ss.n == svy.SampleSize().allocate(math.ceil(total), pop_size=POP).n


def test_estimate_prop_then_allocate_rounds_up():
    ss = svy.SampleSize().estimate_prop(p=0.3, moe=0.05)
    total = ss.n
    ss.allocate(pop_size=POP, method="equal")
    assert sum(ss.n.values()) == math.ceil(total)
    assert ss.target.from_goal == "prop"


def test_given_n_wins_and_is_not_marked():
    ss = svy.SampleSize().estimate_mean(sigma=12, moe=1.5).allocate(100, pop_size=POP)
    assert sum(ss.n.values()) == 100
    assert (ss.target.from_goal, ss.target.from_n) == (None, None)


def test_reallocating_keeps_the_total():
    ss = svy.SampleSize().allocate(90, pop_size={"a": 100, "b": 200})
    ss.allocate(pop_size={"a": 100, "b": 200}, method="equal")
    assert ss.n == {"a": 45, "b": 45}


def test_without_a_previous_goal_n_is_required():
    with pytest.raises(MethodError, match="no n to split"):
        svy.SampleSize().allocate(pop_size=POP)


def test_a_stratified_goal_cannot_be_split_again():
    ss = svy.SampleSize().estimate_prop(p={"a": 0.3, "b": 0.5}, moe=0.05)
    with pytest.raises(MethodError, match="stratified"):
        ss.allocate(pop_size={"a": 100, "b": 200})


def test_a_comparison_has_no_overall_n():
    ss = svy.SampleSize().compare_props(p1=0.3, p2=0.4)
    with pytest.raises(MethodError, match="comparison group"):
        ss.allocate(pop_size=POP)
