# tests/svy/estimation/test_multi_variable.py
"""
Tests for the batched multi-variable estimation API. ``mean``/``total``/``ratio``/
``prop``/``median`` accept a list of columns and return a ``list[Estimate]`` —
one per variable (ratio: one per numerator/denominator pair).

The list form must match a manual per-variable loop. It is not required to be
bit-identical to the loop (the shared design build reorders some floating-point
sums exactly as the single-variable path does), so it is compared with a tight
relative tolerance. Successive batched runs, however, must be bit-identical
(the Taylor design build is deterministic).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import col
from svy.core.design import Design
from svy.core.sample import Sample
from svy.errors import DimensionError
from svy.estimation.estimate import Estimate

RTOL = 1e-9
ATOL = 1e-12

YS = ["y0", "y1", "y2", "y3"]
CS = ["c0", "c1", "c2"]  # categorical columns for prop
ALL = ["mean", "total", "ratio", "prop", "median"]
DEFF = ["mean", "total", "ratio", "prop"]  # median takes no deff


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
        "denom": np.clip(rng.normal(50, 8, n), 1.0, None),
    }
    for k in range(4):
        cols[f"y{k}"] = rng.normal(100 + 10 * k, 20, n)
    for k in range(3):
        cols[f"c{k}"] = rng.integers(0, 3, n)
    df = pl.DataFrame(cols)
    return Sample(df, Design(stratum="stratum", psu="psu", wgt="weight"))


def _batched_and_loop(est, kind, ys, cs, *, where=None, by=None, deff=False):
    """Return (list_result, per-variable-loop_result) for the given estimator."""
    opts = dict(where=where, by=by)
    if kind == "ratio":
        return (
            est.ratio(ys, "denom", deff=deff, **opts),
            [est.ratio(y, "denom", deff=deff, **opts) for y in ys],
        )
    if kind == "prop":
        return est.prop(cs, **opts), [est.prop(c, **opts) for c in cs]
    if kind == "median":  # no deff
        return est.median(ys, **opts), [est.median(y, **opts) for y in ys]
    fn = getattr(est, kind)
    return fn(ys, deff=deff, **opts), [fn(y, deff=deff, **opts) for y in ys]


def _vals(est: Estimate) -> np.ndarray:
    d = est.to_polars()
    return np.column_stack([d["est"].to_numpy(), d["se"].to_numpy()])


def _assert_matches_loop(batched, loop):
    assert isinstance(batched, list)
    assert len(batched) == len(loop)
    for b, single in zip(batched, loop):
        assert isinstance(b, Estimate)
        np.testing.assert_allclose(_vals(b), _vals(single), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("kind", ALL)
def test_list_matches_loop(multi_sample, kind):
    _assert_matches_loop(*_batched_and_loop(multi_sample.estimation, kind, YS, CS))


@pytest.mark.parametrize("kind", ALL)
def test_where_domain_matches_loop(multi_sample, kind):
    _assert_matches_loop(
        *_batched_and_loop(multi_sample.estimation, kind, YS, CS, where=col("region") == 0)
    )


@pytest.mark.parametrize("kind", ALL)
def test_by_falls_back_to_loop(multi_sample, kind):
    # by= is not Rust-batched; the list must still match the loop of (multi-row,
    # one-per-domain) single-variable estimates.
    _assert_matches_loop(*_batched_and_loop(multi_sample.estimation, kind, YS, CS, by="region"))


@pytest.mark.parametrize("kind", DEFF)
def test_deff_matches_loop(multi_sample, kind):
    _assert_matches_loop(*_batched_and_loop(multi_sample.estimation, kind, YS, CS, deff=True))


@pytest.mark.parametrize("kind", ALL)
def test_preserves_order(multi_sample, kind):
    ys = ["y2", "y0", "y3", "y1"]
    cs = ["c2", "c0", "c1"]
    _assert_matches_loop(*_batched_and_loop(multi_sample.estimation, kind, ys, cs))


@pytest.mark.parametrize("kind", ALL)
def test_deterministic_run_to_run(multi_sample, kind):
    # The design build is deterministic, so repeated batched runs are bit-identical.
    est = multi_sample.estimation
    first = np.vstack([_vals(e) for e in _batched_and_loop(est, kind, YS, CS)[0]])
    for _ in range(4):
        again = np.vstack([_vals(e) for e in _batched_and_loop(est, kind, YS, CS)[0]])
        np.testing.assert_array_equal(again, first)


@pytest.mark.parametrize("kind", ALL)
def test_empty_list(multi_sample, kind):
    est = multi_sample.estimation
    if kind == "ratio":
        assert est.ratio([], []) == []
    elif kind == "prop":
        assert est.prop([]) == []
    else:
        assert getattr(est, kind)([]) == []


def test_str_returns_single(multi_sample):
    est = multi_sample.estimation
    assert isinstance(est.mean("y0"), Estimate)
    assert isinstance(est.total("y0"), Estimate)
    assert isinstance(est.ratio("y0", "denom"), Estimate)
    assert isinstance(est.prop("c0"), Estimate)
    assert isinstance(est.median("y0"), Estimate)


def test_single_element_list(multi_sample):
    est = multi_sample.estimation
    r = est.mean(["y0"])
    assert isinstance(r, list) and len(r) == 1
    np.testing.assert_allclose(_vals(r[0]), _vals(est.mean("y0")), rtol=RTOL, atol=ATOL)


def test_tuple_input_accepted(multi_sample):
    est = multi_sample.estimation
    _assert_matches_loop(est.mean(("y0", "y1")), [est.mean("y0"), est.mean("y1")])


# ── ratio pairing semantics ───────────────────────────────────────────────────
def test_ratio_broadcast_denominator(multi_sample):
    est = multi_sample.estimation
    _assert_matches_loop(est.ratio(YS, "denom"), est.ratio(YS, ["denom"] * len(YS)))


def test_ratio_paired_matches_loop(multi_sample):
    est = multi_sample.estimation
    xs = ["denom", "y3", "denom", "y2"]
    _assert_matches_loop(est.ratio(YS, xs), [est.ratio(y, x) for y, x in zip(YS, xs)])


def test_ratio_length_mismatch_raises(multi_sample):
    with pytest.raises(DimensionError, match="same length"):
        multi_sample.estimation.ratio(["y0", "y1"], ["denom", "y2", "y3"])


# ── prop returns one multi-row (per-level) Estimate per variable ──────────────
def test_prop_rows_per_variable(multi_sample):
    est = multi_sample.estimation
    batched = est.prop(CS)
    assert isinstance(batched, list) and len(batched) == len(CS)
    for b, c in zip(batched, CS):
        assert b.to_polars().height == est.prop(c).to_polars().height
