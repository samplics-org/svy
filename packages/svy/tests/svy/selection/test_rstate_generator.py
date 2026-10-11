"""A Generator passed as rstate advances, as NumPy's do: a loop of draws differs."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy


FRAME = pl.DataFrame(
    {"id": range(400), "s": [i % 4 for i in range(400)], "m": np.linspace(1.0, 9.0, 400)}
)

DRAWS = {
    "srs": lambda r: svy.Sample(FRAME, svy.Design(stratum="s")).sampling.srs(5, rstate=r),
    "srs_random_order": lambda r: svy.Sample(FRAME, svy.Design(stratum="s")).sampling.srs(
        5, order_by="m", order_type="random", rstate=r
    ),
    "pps_sys": lambda r: svy.Sample(FRAME, svy.Design(mos="m")).sampling.pps_sys(5, rstate=r),
    "pps_wr": lambda r: svy.Sample(FRAME, svy.Design(mos="m")).sampling.pps_wr(5, rstate=r),
    "pps_brewer": lambda r: svy.Sample(FRAME, svy.Design(mos="m")).sampling.pps_brewer(
        5, rstate=r
    ),
    "pps_murphy": lambda r: svy.Sample(FRAME, svy.Design(mos="m")).sampling.pps_murphy(
        2, rstate=r
    ),
    "pps_rs": lambda r: svy.Sample(FRAME, svy.Design(mos="m")).sampling.pps_rs(5, rstate=r),
}


def _ids(sample) -> tuple[int, ...]:
    return tuple(sorted(sample.data["id"].to_list()))


@pytest.mark.parametrize("method", DRAWS)
def test_reused_generator_gives_different_draws(method):
    rng = np.random.default_rng(3)
    draws = {_ids(DRAWS[method](rng)) for _ in range(6)}
    assert len(draws) > 1


@pytest.mark.parametrize("method", DRAWS)
def test_same_seed_generator_reproduces_the_draw(method):
    a = DRAWS[method](np.random.default_rng(11))
    b = DRAWS[method](np.random.default_rng(11))
    assert _ids(a) == _ids(b)


def test_show_data_random_order_advances_generator():
    sample = svy.Sample(FRAME)
    rng = np.random.default_rng(5)
    rows = {
        tuple(sample.show_data(n=5, order_type="random", rstate=rng)["id"].to_list())
        for _ in range(4)
    }
    assert len(rows) > 1
