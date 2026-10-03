from __future__ import annotations

import msgspec
import numpy as np
import polars as pl
import pytest

import svy

from svy.core._check import check_weights
from svy.core.check import WeightCheck
from svy.errors import DimensionError, MethodError


def test_clean_weights():
    r = check_weights(pl.DataFrame({"w": [1.0, 2.0, 4.0]}), "w")
    assert r == WeightCheck(
        wgt="w",
        n=3,
        n_null=0,
        n_nonfinite=0,
        n_negative=0,
        n_zero=0,
        n_positive=3,
        min=1.0,
        max=4.0,
        ratio=4.0,
        sum=7.0,
        mean=7.0 / 3,
        deff=r.deff,
        ess=49.0 / 21.0,
    )
    assert r.deff == pytest.approx(3 * 21 / 49, rel=1e-15)


def test_dirty_weights_are_counted_apart():
    w = [1.0, None, float("nan"), float("inf"), float("-inf"), -2.0, 0.0, 0.0, 3.0]
    r = check_weights(pl.DataFrame({"w": w}), "w")
    assert (r.n, r.n_null, r.n_nonfinite, r.n_negative, r.n_zero, r.n_positive) == (
        9,
        1,
        3,
        1,
        2,
        2,
    )
    assert r.n_null + r.n_nonfinite + r.n_negative + r.n_zero + r.n_positive == r.n
    assert (r.min, r.max, r.ratio, r.sum, r.mean) == (1.0, 3.0, 3.0, 4.0, 2.0)
    assert r.deff == pytest.approx(2 * 10 / 16)
    assert r.ess == pytest.approx(16 / 10)


def test_no_positive_weight_leaves_the_summaries_empty():
    r = check_weights(pl.DataFrame({"w": [0.0, -1.0, None]}), "w")
    assert (r.n_zero, r.n_negative, r.n_null, r.n_positive) == (1, 1, 1, 0)
    assert all(
        getattr(r, f) is None for f in ("min", "max", "ratio", "sum", "mean", "deff", "ess")
    )


def test_implied_decimals_show_in_the_sum_not_the_shape():
    rng = np.random.default_rng(1)
    w = rng.uniform(50, 500, 200)
    base = check_weights(pl.DataFrame({"w": w}), "w")
    scaled = check_weights(pl.DataFrame({"w": w * 100}), "w")
    assert scaled.sum == pytest.approx(base.sum * 100)
    assert scaled.mean == pytest.approx(base.mean * 100)
    assert scaled.ratio == pytest.approx(base.ratio)
    assert scaled.deff == pytest.approx(base.deff)
    assert scaled.ess == pytest.approx(base.ess)


def test_deff_agrees_with_sample_deff_w():
    rng = np.random.default_rng(7)
    df = pl.DataFrame({"w": rng.lognormal(3, 0.8, 500), "y": rng.normal(size=500)})
    r = check_weights(df, "w")
    assert r.deff == svy.Sample(df, svy.Design(wgt="w")).deff_w
    assert r.ess == pytest.approx(500 / r.deff)


def test_integer_weights_and_lazy_frames():
    lf = pl.LazyFrame({"w": pl.Series([2, 2, 0], dtype=pl.Int32)})
    r = check_weights(lf, "w")
    assert (r.n, r.n_zero, r.sum, r.deff) == (3, 1, 4.0, 1.0)


def test_missing_weight_column():
    with pytest.raises(DimensionError) as exc:
        check_weights(pl.DataFrame({"w": [1.0]}), "wt")
    assert exc.value.code == "MISSING_COLUMNS"
    assert exc.value.param == "wgt"
    assert exc.value.got == ["wt"]


def test_non_numeric_weight():
    with pytest.raises(MethodError) as exc:
        check_weights(pl.DataFrame({"w": ["1", "2"]}), "w")
    assert exc.value.code == "WEIGHT_NOT_NUMERIC"
    assert exc.value.where == "checks.check_weights"
    assert "cast" in exc.value.hint


def test_bad_arguments():
    with pytest.raises(MethodError):
        check_weights(pl.DataFrame({"w": [1.0]}), ["w"])
    with pytest.raises(MethodError):
        check_weights({"w": [1.0]}, "w")


def test_report_serializes_and_prints():
    r = check_weights(pl.DataFrame({"w": [1.0, 0.0, None]}), "w")
    assert msgspec.json.decode(msgspec.json.encode(r), type=WeightCheck) == r
    with pytest.raises(AttributeError):
        r.n = 5
    text = repr(r)
    assert text.splitlines()[0] == "Weight check: w"
    assert "Kish deff" in text
    assert "Kish deff   : —" in repr(check_weights(pl.DataFrame({"w": [0.0]}), "w"))
    assert "Weight check: w" in str(r)
