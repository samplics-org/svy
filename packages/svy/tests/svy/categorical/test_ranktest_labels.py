import numpy as np
import polars as pl
import pytest

import svy


@pytest.fixture
def sample():
    rng = np.random.default_rng(1)
    n = 400
    df = pl.DataFrame(
        {
            "str": rng.integers(1, 5, n),
            "psu": np.arange(n) // 10,
            "w": rng.uniform(1, 3, n),
            "area": rng.integers(1, 3, n).astype(float),
            "reg": rng.integers(1, 3, n),
            "y": rng.uniform(0, 1, n),
        }
    )
    s = svy.Sample(df, svy.Design(stratum="str", psu="psu", wgt="w"))
    s.set_value_labels("area", {1: "URBANA", 2: "RURAL"})
    s.set_value_labels("reg", {1: "Norte", 2: "Sul"})
    return s


def test_two_sample_header_uses_labels(sample):
    r = sample.categorical.ranktest("y", group="area", score="kruskal-wallis")
    assert "Groups: area = [URBANA vs RURAL]" in r.__plain_str__()
    assert "URBANA vs RURAL" in str(r)
    assert r.groups.levels == (1.0, 2.0)


def test_use_labels_false_prints_codes(sample):
    r = sample.categorical.ranktest("y", group="area", score="kruskal-wallis", use_labels=False)
    assert "Groups: area = [1.0 vs 2.0]" in r.__plain_str__()


def test_by_result_uses_group_and_by_labels(sample):
    r = sample.categorical.ranktest("y", group="area", by="reg", score="kruskal-wallis")
    out = r.__plain_str__()
    assert "Groups: area = [URBANA vs RURAL]" in out
    assert "── reg = Norte " in out and "── reg = Sul " in out
    assert all(x.groups.labels == ("URBANA", "RURAL") for x in r)


def test_custom_score_path_uses_labels(sample):
    r = sample.categorical.ranktest("y", group="area", by="reg", score_fn=lambda rk, n: rk / n)
    out = r.__plain_str__()
    assert "Groups: area = [URBANA vs RURAL]" in out
    assert "── reg = Norte " in out


def test_replication_path_uses_labels():
    rng = np.random.default_rng(2)
    n = 120
    df = pl.DataFrame(
        {
            "w": rng.uniform(1, 3, n),
            "area": rng.integers(1, 3, n),
            "y": rng.uniform(0, 1, n),
            **{f"r{i}": rng.uniform(1, 3, n) for i in range(1, 11)},
        }
    )
    d = svy.Design(wgt="w", rep_wgts=svy.BootstrapWgts(prefix="r", n_reps=10))
    s = svy.Sample(df, d)
    s.set_value_labels("area", {1: "URBANA", 2: "RURAL"})
    r = s.categorical.ranktest("y", group="area", score="kruskal-wallis", method="replication")
    assert "Groups: area = [URBANA vs RURAL]" in r.__plain_str__()


def test_k_sample_unchanged(sample):
    sample.set_value_labels("reg", {1: "Norte", 2: "Sul"})
    df = sample.data.with_columns(k=(pl.int_range(pl.len()) % 3) + 1)
    s = svy.Sample(df, svy.Design(stratum="str", psu="psu", wgt="w"))
    r = s.categorical.ranktest("y", group="k", score="kruskal-wallis")
    assert "Groups: k (3 levels)" in r.__plain_str__()
