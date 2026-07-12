# tests/svy/estimation/test_multi_variable.py
"""
Tests for the batched multi-variable estimation API: ``mean``/``total`` accept a
list of response columns and return a ``list[Estimate]``, one per variable.

The list form must match a manual per-variable loop. It is not required to be
bit-identical (the shared design build reorders some floating-point sums exactly
like the single-variable path already does run-to-run), so results are compared
with a tight relative tolerance.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import col
from svy.core.design import Design
from svy.core.sample import Sample
from svy.estimation.estimate import Estimate

RTOL = 1e-9


@pytest.fixture
def multi_sample() -> Sample:
    rng = np.random.default_rng(20240711)
    n = 3000
    s = rng.integers(0, 6, n)
    cols: dict = {
        "id": range(1, n + 1),
        "stratum": np.array(["A", "B", "C"])[s % 3],
        "psu": rng.integers(1, 25, n),
        "weight": rng.uniform(0.5, 2.5, n),
        "region": rng.integers(0, 4, n),
    }
    for k in range(4):
        cols[f"y{k}"] = rng.normal(100 + 10 * k, 20, n)
    df = pl.DataFrame(cols)
    return Sample(df, Design(stratum="stratum", psu="psu", wgt="weight"))


def _vals(est: Estimate) -> np.ndarray:
    d = est.to_polars()
    return np.column_stack([d["est"].to_numpy(), d["se"].to_numpy()])


def _assert_matches_loop(batched, loop):
    assert isinstance(batched, list)
    assert len(batched) == len(loop)
    for b, single in zip(batched, loop):
        assert isinstance(b, Estimate)
        np.testing.assert_allclose(_vals(b), _vals(single), rtol=RTOL)


YS = ["y0", "y1", "y2", "y3"]


@pytest.mark.parametrize("kind", ["mean", "total"])
def test_list_matches_loop(multi_sample, kind):
    est = multi_sample.estimation
    fn = getattr(est, kind)
    _assert_matches_loop(fn(YS), [fn(y) for y in YS])


@pytest.mark.parametrize("kind", ["mean", "total"])
def test_str_returns_single_estimate(multi_sample, kind):
    fn = getattr(multi_sample.estimation, kind)
    result = fn("y0")
    assert isinstance(result, Estimate)
    np.testing.assert_allclose(_vals(result), _vals(fn(["y0"])[0]), rtol=RTOL)


@pytest.mark.parametrize("kind", ["mean", "total"])
def test_list_preserves_order(multi_sample, kind):
    fn = getattr(multi_sample.estimation, kind)
    order = ["y2", "y0", "y3", "y1"]
    _assert_matches_loop(fn(order), [fn(y) for y in order])


@pytest.mark.parametrize("kind", ["mean", "total"])
def test_where_batched(multi_sample, kind):
    fn = getattr(multi_sample.estimation, kind)
    w = col("region") == 0
    _assert_matches_loop(fn(YS, where=w), [fn(y, where=w) for y in YS])


@pytest.mark.parametrize("kind", ["mean", "total"])
def test_deff_batched(multi_sample, kind):
    fn = getattr(multi_sample.estimation, kind)
    _assert_matches_loop(fn(YS, deff=True), [fn(y, deff=True) for y in YS])


@pytest.mark.parametrize("kind", ["mean", "total"])
def test_by_falls_back_to_loop(multi_sample, kind):
    # by= is not batched in Rust; it must still return a list matching the loop
    # of (multi-row, one-per-domain) single-variable estimates.
    fn = getattr(multi_sample.estimation, kind)
    _assert_matches_loop(fn(["y0", "y1"], by="region"), [fn(y, by="region") for y in ["y0", "y1"]])


@pytest.mark.parametrize("kind", ["mean", "total"])
def test_single_element_list(multi_sample, kind):
    fn = getattr(multi_sample.estimation, kind)
    result = fn(["y0"])
    assert isinstance(result, list) and len(result) == 1
    np.testing.assert_allclose(_vals(result[0]), _vals(fn("y0")), rtol=RTOL)


@pytest.mark.parametrize("kind", ["mean", "total"])
def test_empty_list(multi_sample, kind):
    fn = getattr(multi_sample.estimation, kind)
    assert fn([]) == []


def test_tuple_input_accepted(multi_sample):
    est = multi_sample.estimation
    _assert_matches_loop(est.mean(("y0", "y1")), [est.mean("y0"), est.mean("y1")])


def test_run_to_run_within_tolerance(multi_sample):
    # The batched path is not required to be bit-stable run-to-run (neither is
    # the single-variable path), but successive runs must agree to a tight
    # tolerance — i.e. no correctness-affecting nondeterminism.
    est = multi_sample.estimation
    first = _vals_list(est.mean(YS))
    for _ in range(4):
        np.testing.assert_allclose(_vals_list(est.mean(YS)), first, rtol=1e-9)


def _vals_list(results) -> np.ndarray:
    return np.vstack([_vals(r) for r in results])
