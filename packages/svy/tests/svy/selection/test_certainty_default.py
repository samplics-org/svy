"""certainty_threshold defaults to None: no unit is set aside unless asked."""

from __future__ import annotations

import polars as pl
import pytest

import svy

from svy.errors import SelectionError


# Shares 0.6, 0.3, 0.1 per stratum: with n=2 the first unit's n·p is 1.2.
DOMINANT = pl.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "s": ["a"] * 3 + ["b"] * 3, "m": [60.0, 30.0, 10.0] * 2}
)
BALANCED = pl.DataFrame(
    {"id": list(range(12)), "s": ["a"] * 6 + ["b"] * 6, "m": [5.0, 6.0, 7.0] * 4}
)


def _sample(frame):
    return svy.Sample(frame, svy.Design(stratum="s", mos="m"))


@pytest.mark.parametrize("method", ["pps_sys", "pps_brewer", "pps_rs"])
def test_a_unit_at_one_raises_by_default(method):
    with pytest.raises(SelectionError) as exc:
        getattr(_sample(DOMINANT).sampling, method)(2, rstate=1)
    assert "certainty_threshold=1.0" in str(exc.value)


@pytest.mark.parametrize("method", ["pps_sys", "pps_brewer", "pps_rs"])
def test_threshold_one_sets_the_unit_aside(method):
    drawn = getattr(_sample(DOMINANT).sampling, method)(2, certainty_threshold=1.0, rstate=1)
    big = drawn.data.filter(pl.col("m") == 60.0)
    assert big.height == 2
    assert big["svy_prob_selection"].to_list() == [1.0, 1.0]


@pytest.mark.parametrize("method", ["pps_sys", "pps_brewer", "pps_rs"])
def test_without_a_unit_at_one_the_default_draws_as_before(method):
    a = getattr(_sample(BALANCED).sampling, method)(2, rstate=3)
    b = getattr(_sample(BALANCED).sampling, method)(2, certainty_threshold=1.0, rstate=3)
    assert a.data.equals(b.data)


@pytest.mark.parametrize("method", ["pps_wr", "pps_murphy"])
def test_wr_and_murphy_draw_a_dominant_unit_at_random(method):
    drawn = getattr(_sample(DOMINANT).sampling, method)(2, rstate=1)
    probs = drawn.data.filter(pl.col("m") == 60.0)["svy_prob_selection"]
    assert probs.len() > 0 and (probs < 1.0).all()


def test_n_draws_raises_the_same_way():
    with pytest.raises(SelectionError):
        _sample(DOMINANT).sampling.pps_sys(2, n_draws=3, rstate=1)
