# tests/svy/weighting/test_threshold.py
"""
svy.Threshold: explicit kinds of trimming bound.

A bare number is always an absolute bound; a quantile is only ever
Threshold.quantile(p). Nothing is inferred from a number's size.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

import svy

from svy import Cap, Design, Sample, Threshold, TrimConfig
from svy.weighting.types import resolve_threshold


W = np.array([0.0, 0.5, 0.8, 0.9, 1.0, 1.2, 1.5, 3.0, 9.0])
POS = W[W > 0]


# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------


def test_absolute_ignores_the_data():
    assert Threshold.absolute(0.9).compute(W) == 0.9
    assert Threshold.absolute(40).compute(np.array([])) == 40.0


@pytest.mark.parametrize("p", [0.01, 0.5, 0.9, 0.99, 1.0])
def test_quantile_of_the_positive_values(p):
    assert_allclose(Threshold.quantile(p).compute(W), np.quantile(POS, p))


def test_quantile_of_no_positive_values_is_zero():
    assert Threshold.quantile(0.9).compute(np.array([0.0, 0.0])) == 0.0


@pytest.mark.parametrize(
    "stat,expected",
    [
        ("median", np.median(POS)),
        ("mean", np.mean(POS)),
        ("sd", np.std(POS, ddof=1)),
        ("iqr", np.percentile(POS, 75) - np.percentile(POS, 25)),
    ],
)
def test_statistic_kinds_unchanged(stat, expected):
    assert_allclose(Threshold(stat, 2.5).compute(W), 2.5 * expected)


def test_composition_mixes_kinds():
    t = Threshold.quantile(0.9) + 2 * Threshold("sd") - Threshold.absolute(0.5)
    expected = np.quantile(POS, 0.9) + 2 * np.std(POS, ddof=1) - 0.5
    assert_allclose(t.compute(W), expected)


def test_scaling_a_quantile_keeps_p():
    t = 2 * Threshold.quantile(0.75)
    assert t.p == 0.75 and t.k == 2.0
    assert_allclose(t.compute(W), 2 * np.quantile(POS, 0.75))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0, -0.1, 1.5])
def test_quantile_out_of_range(p):
    with pytest.raises(ValueError, match=r"Quantile p must be in \(0, 1\]"):
        Threshold.quantile(p)


def test_quantile_given_as_a_percentage_says_how_to_fix_it():
    with pytest.raises(ValueError, match="For the 99th percentile use 0.99"):
        Threshold.quantile(99)


@pytest.mark.parametrize("v", [0, -1.0])
def test_absolute_must_be_positive(v):
    with pytest.raises(ValueError, match="must be > 0"):
        Threshold.absolute(v)


def test_quantile_needs_p_and_others_refuse_it():
    with pytest.raises(ValueError, match=r"Threshold.quantile\(p\)"):
        Threshold("quantile")
    with pytest.raises(ValueError, match="p applies only to a quantile"):
        Threshold("median", p=0.5)


def test_unknown_stat():
    with pytest.raises(ValueError, match="Unsupported stat"):
        Threshold("max")


# ---------------------------------------------------------------------------
# Names and reprs
# ---------------------------------------------------------------------------


def test_cap_is_the_same_class():
    assert Cap is Threshold is svy.Threshold is svy.Cap
    assert Cap("median", 3.5) == Threshold("median", 3.5)


@pytest.mark.parametrize(
    "t,text",
    [
        (Threshold.quantile(0.99), "Threshold.quantile(0.99)"),
        (Threshold.absolute(0.9), "Threshold.absolute(0.9)"),
        (Threshold("median", 3.5), "Threshold('median', 3.5)"),
        (Threshold("iqr"), "Threshold('iqr')"),
        (2 * Threshold.quantile(0.5), "2.0 * Threshold.quantile(0.5)"),
    ],
)
def test_reprs_are_runnable(t, text):
    assert repr(t) == text
    assert eval(text, {"Threshold": Threshold}) == t


def test_composed_repr_is_runnable():
    t = Threshold("median") + 6 * Threshold("iqr") - Threshold.quantile(0.1)
    text = repr(t)
    assert text == "Threshold('median') + Threshold('iqr', 6.0) - Threshold.quantile(0.1)"
    assert_allclose(eval(text, {"Threshold": Threshold}).compute(W), t.compute(W))


def test_equal_and_hashable():
    assert Threshold.quantile(0.9) == Threshold.quantile(0.9)
    assert Threshold.quantile(0.9) != Threshold.quantile(0.95)
    assert len({Threshold.quantile(0.9), Threshold.quantile(0.9), Threshold.absolute(0.9)}) == 2


# ---------------------------------------------------------------------------
# Bare numbers are absolute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", [0.5, 0.9, 1, 1.0, 40, np.float32(0.9), np.int64(3)])
def test_a_number_is_absolute(v):
    assert resolve_threshold(v, W) == pytest.approx(float(v))


@pytest.mark.parametrize("v", [0, -2.0])
def test_a_non_positive_number_points_to_quantile(v):
    with pytest.raises(ValueError, match=r"Threshold.quantile\(p\)"):
        resolve_threshold(v, W)


def test_bool_is_refused():
    with pytest.raises(TypeError):
        resolve_threshold(True, W)


# ---------------------------------------------------------------------------
# Through the weighting methods
# ---------------------------------------------------------------------------


@pytest.fixture
def sample() -> Sample:
    rng = np.random.default_rng(11)
    n = 200
    df = pl.DataFrame(
        {
            "psu": np.arange(n),
            "w": np.concatenate([rng.uniform(0.5, 1.5, n - 5), [4.0, 5.0, 6.0, 7.0, 8.0]]),
            "g": rng.choice(["a", "b"], n),
        }
    )
    return Sample(df, Design(psu="psu", wgt="w"))


def test_trim_with_a_number_below_one_caps_there(sample):
    out = sample.weighting.trim(upper=0.9, redistribute=False)
    assert out.data["trim_wgt"].max() == pytest.approx(0.9)


def test_trim_with_quantile_caps_at_the_quantile(sample):
    cap = np.quantile(sample.data["w"].to_numpy(), 0.9)
    out = sample.weighting.trim(upper=Threshold.quantile(0.9), redistribute=False)
    assert out.data["trim_wgt"].max() == pytest.approx(cap)


def test_mixed_kinds_per_bound(sample):
    out = sample.weighting.trim(
        upper=Threshold.quantile(0.95), lower=Threshold.absolute(0.6), redistribute=False
    )
    w = out.data["trim_wgt"].to_numpy()
    assert w.min() >= 0.6 - 1e-12
    assert w.max() <= np.quantile(sample.data["w"].to_numpy(), 0.95) + 1e-12


def test_quantile_per_domain(sample):
    out = sample.weighting.trim(upper=Threshold.quantile(0.9), by="g", redistribute=False)
    for g in ("a", "b"):
        orig = sample.data.filter(pl.col("g") == g)["w"].to_numpy()
        got = out.data.filter(pl.col("g") == g)["trim_wgt"].to_numpy()
        assert got.max() == pytest.approx(np.quantile(orig, 0.9))


def test_quantile_through_poststratify_trimming():
    """trimming= takes the same thresholds: a quantile equals the hand-written one."""
    from pathlib import Path

    data = pl.read_csv(
        Path(__file__).resolve().parents[2] / "test_data" / "apiclus1.csv", null_values=["NA"]
    )
    base = Sample(data, Design(wgt="pw", psu="dnum", pop_size="fpc"))

    def run(upper):
        cfg = TrimConfig(upper=upper, redistribute=True, min_cell_size=1, max_iter=20)
        return base.weighting.poststratify(
            {"E": 4421.0, "H": 755.0, "M": 1018.0}, cells="stype", trimming=cfg
        )

    a = run(Threshold.quantile(0.995))
    b = run(lambda w: float(np.quantile(w, 0.995)))
    assert a.data["ps_wgt"].to_list() == b.data["ps_wgt"].to_list()
