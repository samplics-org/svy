# tests/svy/estimation/test_row_order.py
"""Estimate rows come back in a fixed order.

The kernel returns domains in hash order, which changed from run to run.
Rows are now sorted by domain level, then by category level, the order
``to_polars()`` lists them in, and the covariance is permuted with them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample
from svy.estimation.base import Estimation


N = 400


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(20260925)
    return pl.DataFrame(
        {
            "stratum": np.repeat(["s1", "s2"], N // 2),
            "psu": np.repeat(np.arange(40), N // 40),
            "w": rng.uniform(1, 3, N),
            "y": rng.normal(10, 2, N),
            "x": rng.uniform(1, 2, N),
            "k": rng.choice([10, 2, 1, 33], N),
            "g": rng.choice(["b", "a10", "B", "a2", "c"], N),
            "h": rng.choice(["u", "r"], N),
            "c": rng.choice(["z", "x", "y"], N),
            "n12": rng.integers(1, 13, N),
        }
    )


@pytest.fixture(scope="module")
def sample(data) -> Sample:
    return Sample(data, Design(stratum="stratum", psu="psu", wgt="w"))


@pytest.fixture(scope="module")
def rep_sample(sample) -> Sample:
    return sample.weighting.create_jk_wgts()


G_ORDER = ["a2", "a10", "B", "b", "c"]
K_ORDER = [1, 2, 10, 33]


def _by(result) -> list:
    return [p.by_level[0] if len(p.by_level) == 1 else p.by_level for p in result.estimates]


class TestDomainOrder:
    @pytest.mark.parametrize("which", ["sample", "rep_sample"])
    @pytest.mark.parametrize("stat", ["mean", "total", "median"])
    def test_string_and_numeric_domains(self, request, which, stat):
        s = request.getfixturevalue(which)
        fn = getattr(s.estimation, stat)
        assert _by(fn("y", by="g")) == G_ORDER
        assert _by(fn("y", by="k")) == K_ORDER

    @pytest.mark.parametrize("which", ["sample", "rep_sample"])
    def test_ratio(self, request, which):
        s = request.getfixturevalue(which)
        assert _by(s.estimation.ratio("y", "x", by="g")) == G_ORDER

    @pytest.mark.parametrize("which", ["sample", "rep_sample"])
    def test_quantile(self, request, which):
        s = request.getfixturevalue(which)
        for r in s.estimation.quantile("y", p=[0.25, 0.75], by="k"):
            assert _by(r) == K_ORDER

    def test_corr_keeps_pair_order_within_domain(self, sample):
        r = sample.estimation.corr([("y", "x"), ("x", "k")], by="g")
        assert [(p.by_level[0], p.y, p.x) for p in r.estimates] == [
            (g, y, x) for g in G_ORDER for y, x in [("y", "x"), ("x", "k")]
        ]

    def test_several_by(self, sample):
        r = sample.estimation.mean("y", by=["k", "h"])
        assert _by(r) == [(k, h) for k in K_ORDER for h in ["r", "u"]]

    def test_domains_follow_rows(self, sample):
        assert sample.estimation.mean("y", by="g").domains == G_ORDER


class TestCategoryOrder:
    @pytest.mark.parametrize("which", ["sample", "rep_sample"])
    def test_prop_by_domain_then_level(self, request, which):
        s = request.getfixturevalue(which)
        r = s.estimation.prop("c", by="k")
        assert [(p.by_level[0], p.y_level) for p in r.estimates] == [
            (k, c) for k in K_ORDER for c in ["x", "y", "z"]
        ]

    def test_numeric_levels_sort_numerically(self, sample):
        r = sample.estimation.prop("n12")
        assert [p.y_level for p in r.estimates] == list(range(1, 13))


class TestCovarianceFollowsRows:
    @pytest.mark.parametrize("which", ["sample", "rep_sample"])
    def test_matches_kernel_matrix_by_level(self, request, monkeypatch, which):
        s = request.getfixturevalue(which)
        seen = {}
        original = Estimation._cov_from_kernel

        def capture(result_df, cov_flat):
            cov = original(result_df, cov_flat)
            by_col = next(c for c in result_df.columns if c.startswith("by"))
            seen["pos"] = {v: i for i, v in enumerate(result_df[by_col].to_list())}
            seen["cov"] = cov
            return cov

        monkeypatch.setattr(Estimation, "_cov_from_kernel", staticmethod(capture))
        r = s.estimation.mean("y", by="g")
        idx = [seen["pos"][k] for k in r.keys()]
        np.testing.assert_array_equal(r.covariance, seen["cov"][np.ix_(idx, idx)])

        a, b = seen["pos"]["a2"], seen["pos"]["c"]
        raw = seen["cov"]
        se = np.sqrt(raw[a, a] + raw[b, b] - 2 * raw[a, b])
        assert r.contrast({"a2": 1, "c": -1}).estimates[0].se == pytest.approx(se, rel=1e-12)

    def test_prop_diagonal_matches_se(self, sample):
        r = sample.estimation.prop("c", by="k")
        np.testing.assert_allclose(
            np.diag(r.covariance), [p.se**2 for p in r.estimates], rtol=1e-12
        )


def test_to_polars_lists_rows_in_order(sample):
    r = sample.estimation.prop("c", by="g")
    df = r.to_polars(use_labels=False)
    assert df.select("g", "c").rows() == [(p.by_level[0], p.y_level) for p in r.estimates]


_RUN = textwrap.dedent(
    """
    import json, sys
    import numpy as np, polars as pl
    from svy import Design, Sample

    rng = np.random.default_rng(7)
    n = 400
    df = pl.DataFrame({
        "s": rng.choice([1, 2, 3], n),
        "w": rng.uniform(1, 3, n),
        "y": rng.normal(size=n),
        "g": rng.choice(["e", "b", "d", "a", "c", "f", "h", "g"], n),
        "c": rng.choice(["x", "z", "y"], n),
    })
    s = Sample(df, Design(stratum="s", wgt="w"))
    out = {}
    for name, r in [
        ("mean", s.estimation.mean("y", by="g")),
        ("prop", s.estimation.prop("c", by="g")),
    ]:
        out[name] = {"keys": [list(k) if isinstance(k, tuple) else k for k in r.keys()],
                     "cov": np.asarray(r.covariance).round(14).tolist()}
    json.dump(out, sys.stdout)
    """
)


def test_same_order_across_processes():
    runs = [
        json.loads(
            subprocess.run(
                [sys.executable, "-c", _RUN],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
            ).stdout
        )
        for seed in range(4)
    ]
    assert all(r == runs[0] for r in runs[1:])
    assert runs[0]["mean"]["keys"] == list("abcdefgh")
