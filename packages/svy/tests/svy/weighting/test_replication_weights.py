# tests/svy/weighting/test_replication_weights.py
"""
Tests for replicate weight creation methods.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Sample
from svy.core.design import Design
from svy.errors import DimensionError, MethodError


@pytest.fixture
def simple_stratified_sample():
    data = pl.DataFrame(
        {
            "id": list(range(1, 13)),
            "stratum": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "wgt": [1.0] * 12,
            "y": [10.0, 12.0, 15.0, 11.0, 20.0, 22.0, 18.0, 25.0, 30.0, 28.0, 32.0, 35.0],
        }
    )
    return Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))


@pytest.fixture
def multi_psu_sample():
    data = pl.DataFrame(
        {
            "id": list(range(1, 25)),
            "stratum": [1] * 8 + [2] * 8 + [3] * 8,
            "psu": [1, 1, 2, 2, 3, 3, 4, 4] * 3,
            "wgt": [1.0] * 24,
            "y": np.random.default_rng(42).normal(100, 10, 24).tolist(),
        }
    )
    data = data.with_columns((pl.col("psu") + (pl.col("stratum") - 1) * 4).alias("psu"))
    return Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))


@pytest.fixture
def odd_psu_sample():
    data = pl.DataFrame(
        {
            "id": list(range(1, 19)),
            "stratum": [1] * 6 + [2] * 6 + [3] * 6,
            "psu": [1, 1, 2, 2, 3, 3] * 3,
            "wgt": [1.0] * 18,
            "y": np.random.default_rng(42).normal(100, 10, 18).tolist(),
        }
    )
    data = data.with_columns((pl.col("psu") + (pl.col("stratum") - 1) * 3).alias("psu"))
    return Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))


@pytest.fixture
def unstratified_sample():
    data = pl.DataFrame(
        {
            "id": list(range(1, 13)),
            "psu": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "wgt": [1.0] * 12,
            "y": [10.0, 12.0, 15.0, 11.0, 20.0, 22.0, 18.0, 25.0, 30.0, 28.0, 32.0, 35.0],
        }
    )
    return Sample(data=data, design=Design(wgt="wgt", psu="psu"))


@pytest.fixture
def bootstrap_adjustment_sample():
    return pl.DataFrame(
        {
            "id": list(range(1, 11)),
            "base_wgt": [1.0] * 10,
            "strata": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "psu": ["1", "1", "2", "2", "3", "3", "1", "1", "2", "2"],
            "status": ["rr", "rr", "nr", "in", "uk", "rr", "nr", "in", "rr", "uk"],
            "resp_class": ["A", "A", "B", "A", "A", "B", "B", "A", "B", "B"],
            "some_val": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )


class TestBRRWeights:
    def test_brr_basic(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_brr_wgts()
        assert sample.design.rep_wgts is not None
        assert sample.design.rep_wgts.method.value == "BRR"
        assert sample.design.rep_wgts.n_reps >= 3
        for col in sample.design.rep_wgts.columns:
            assert col in sample.data.columns

    def test_brr_default_prefix_uses_design_wgt(self, simple_stratified_sample):
        """Without rep_prefix, columns are named {design.wgt}1, {design.wgt}2, ..."""
        sample = simple_stratified_sample.weighting.create_brr_wgts()
        assert sample.design.rep_wgts.prefix == "wgt"

    def test_brr_rstate_reproducibility(self, simple_stratified_sample):
        s1 = simple_stratified_sample.weighting.create_brr_wgts(rstate=42)
        s2 = simple_stratified_sample.weighting.create_brr_wgts(rstate=42)
        for c1, c2 in zip(s1.design.rep_wgts.columns, s2.design.rep_wgts.columns):
            np.testing.assert_array_almost_equal(s1.data[c1].to_numpy(), s2.data[c2].to_numpy())

    def test_brr_fay_coefficient(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_brr_wgts(fay_coef=0.5)
        assert sample.design.rep_wgts.fay_coef == 0.5

    def test_brr_custom_n_reps(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_brr_wgts(n_reps=8)
        assert sample.design.rep_wgts.n_reps >= 8

    def test_brr_custom_rep_prefix(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_brr_wgts(n_reps=4, rep_prefix="brr_rep")
        rep_cols = [c for c in sample.data.columns if c.startswith("brr_rep")]
        assert len(rep_cols) == 4

    def test_brr_requires_stratum(self, unstratified_sample):
        with pytest.raises(Exception, match="BRR requires.*stratum|stratum.*None"):
            unstratified_sample.weighting.create_brr_wgts()

    def test_brr_requires_psu(self, unstratified_sample):
        data = (
            unstratified_sample.data.drop("psu")
            if "psu" in unstratified_sample.data.columns
            else unstratified_sample.data
        )
        sample = Sample(data=data, design=Design(wgt="wgt", stratum=None))
        with pytest.raises(Exception):
            sample.weighting.create_brr_wgts()

    def test_brr_rejects_odd_psu_count(self, odd_psu_sample):
        with pytest.raises(Exception):
            odd_psu_sample.weighting.create_brr_wgts()

    def test_brr_string_psu_stratum(self):
        data = pl.DataFrame(
            {
                "id": list(range(1, 11)),
                "base_wgt": [1.0] * 10,
                "strata": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
                "psu": ["1", "1", "2", "2", "2", "2", "1", "1", "2", "2"],
            }
        )
        sample = Sample(data=data, design=Design(wgt="base_wgt", stratum="strata", psu="psu"))
        sample = sample.weighting.create_brr_wgts(n_reps=4)
        assert sample.design.rep_wgts.n_reps == 4
        assert sample.design.rep_wgts.prefix == "base_wgt"


class TestJackknifeWeights:
    def test_jkn_basic(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_jk_wgts(paired=False)
        assert sample.design.rep_wgts.method.value == "Jackknife"
        assert sample.design.rep_wgts.n_reps == 6

    def test_jkn_default_prefix_uses_design_wgt(self, simple_stratified_sample):
        """Without rep_prefix, prefix matches design.wgt."""
        sample = simple_stratified_sample.weighting.create_jk_wgts(paired=False)
        assert sample.design.rep_wgts.prefix == "wgt"

    def test_jkn_unstratified(self, unstratified_sample):
        sample = unstratified_sample.weighting.create_jk_wgts(paired=False)
        assert sample.design.rep_wgts.n_reps == 6

    def test_jkn_string_psu(self):
        data = pl.DataFrame(
            {
                "base_wgt": [1.0] * 10,
                "strata": ["A"] * 5 + ["B"] * 5,
                "psu": ["1", "1", "2", "2", "3", "3", "1", "1", "2", "2"],
            }
        )
        sample = Sample(data=data, design=Design(wgt="base_wgt", stratum="strata", psu="psu"))
        sample.weighting.create_jk_wgts()

    def test_jk2_basic(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_jk_wgts(paired=True)
        assert sample.design.rep_wgts.method.value == "Jackknife"
        assert sample.design.rep_wgts.n_reps == 3

    def test_jk2_triplet(self, odd_psu_sample):
        sample = odd_psu_sample.weighting.create_jk_wgts(paired=True)
        assert sample.design.rep_wgts.n_reps == 3
        rep_wgts = sample.data.select(sample.design.rep_wgts.columns).to_numpy()
        for r in range(3):
            stratum_wgts = rep_wgts[r * 6 : (r + 1) * 6, r]
            zero_count = np.sum(stratum_wgts == 0.0)
            adjusted_count = np.sum(np.isclose(stratum_wgts, 1.5))
            assert zero_count == 2
            assert adjusted_count == 4

    def test_jk2_adjustment_factors_pair(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_jk_wgts(paired=True)
        rep_wgts = sample.data.select(sample.design.rep_wgts.columns).to_numpy()
        stratum1_wgts = rep_wgts[:4, 0]
        assert np.sum(stratum1_wgts == 0.0) == 2
        assert np.sum(np.isclose(stratum1_wgts, 2.0)) == 2

    def test_jk2_rstate_reproducibility(self, odd_psu_sample):
        s1 = odd_psu_sample.weighting.create_jk_wgts(paired=True, rstate=42)
        s2 = odd_psu_sample.weighting.create_jk_wgts(paired=True, rstate=42)
        for c1, c2 in zip(s1.design.rep_wgts.columns, s2.design.rep_wgts.columns):
            np.testing.assert_array_almost_equal(s1.data[c1].to_numpy(), s2.data[c2].to_numpy())

    def test_jk2_rstate_parameter_accepted(self, odd_psu_sample):
        sample = odd_psu_sample.weighting.create_jk_wgts(paired=True, rstate=12345)
        assert sample.design.rep_wgts.n_reps == 3

    def test_jk2_rejects_singleton_strata(self):
        data = pl.DataFrame({"stratum": [1, 2, 2], "psu": [1, 2, 3], "wgt": [1.0, 1.0, 1.0]})
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))
        with pytest.raises(Exception):
            sample.weighting.create_jk_wgts(paired=True)


class TestBootstrapWeights:
    def test_bootstrap_basic(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_bs_wgts(n_reps=100, rstate=42)
        assert sample.design.rep_wgts.method.value == "Bootstrap"
        assert sample.design.rep_wgts.n_reps == 100

    def test_bootstrap_default_prefix_uses_design_wgt(self, simple_stratified_sample):
        """Without rep_prefix, columns are named {design.wgt}1, {design.wgt}2, ..."""
        sample = simple_stratified_sample.weighting.create_bs_wgts(n_reps=3, rstate=42)
        assert sample.design.rep_wgts.prefix == "wgt"
        assert "wgt1" in sample.data.columns
        assert "wgt2" in sample.data.columns
        assert "wgt3" in sample.data.columns

    def test_bootstrap_reproducibility(self, simple_stratified_sample):
        s1 = simple_stratified_sample.weighting.create_bs_wgts(n_reps=50, rstate=42)
        s2 = simple_stratified_sample.weighting.create_bs_wgts(n_reps=50, rstate=42)
        for c1, c2 in zip(s1.design.rep_wgts.columns, s2.design.rep_wgts.columns):
            np.testing.assert_array_almost_equal(s1.data[c1].to_numpy(), s2.data[c2].to_numpy())

    def test_bootstrap_unstratified(self, unstratified_sample):
        sample = unstratified_sample.weighting.create_bs_wgts(n_reps=50, rstate=42)
        assert sample.design.rep_wgts.n_reps == 50

    def test_bootstrap_rng_object_reproducibility(self, bootstrap_adjustment_sample):
        def _run(seed):
            return (
                Sample(data=bootstrap_adjustment_sample, design=Design(wgt="base_wgt", psu="psu"))
                .weighting.create_bs_wgts(n_reps=5, rstate=np.random.default_rng(seed=seed))
                .data.select([f"base_wgt{i}" for i in range(1, 6)])
                .to_numpy()
            )

        assert np.allclose(_run(19), _run(19), equal_nan=True)

    def test_bootstrap_stratified_string_columns(self, bootstrap_adjustment_sample):
        for design_kwargs in [
            dict(wgt="base_wgt", psu="psu"),
            dict(wgt="base_wgt", stratum="strata", psu="psu"),
        ]:
            sample = Sample(
                data=bootstrap_adjustment_sample, design=Design(**design_kwargs)
            ).weighting.create_bs_wgts(n_reps=500)
            assert sample.design.rep_wgts.n_reps == 500


class TestBootstrapAdjustment:
    def test_adjust_after_bootstrap_weight_sums_match(self, bootstrap_adjustment_sample):
        for design_kwargs in [
            dict(wgt="base_wgt", psu="psu"),
            dict(wgt="base_wgt", stratum="strata", psu="psu"),
        ]:
            sample = Sample(
                data=bootstrap_adjustment_sample, design=Design(**design_kwargs)
            ).weighting.adjust(
                by="resp_class", resp_status="status", wgt_name="nr_wgt", respondents_only=False
            )
            sums = sample.data.select(["base_wgt", "nr_wgt"]).sum()
            assert sums[0, 0] == sums[0, 1]

    def test_bootstrap_then_adjust_reproducibility(self, bootstrap_adjustment_sample):
        def _run():
            return (
                Sample(data=bootstrap_adjustment_sample, design=Design(wgt="base_wgt", psu="psu"))
                .weighting.create_bs_wgts(n_reps=5, rstate=np.random.default_rng(seed=19))
                .weighting.adjust(
                    by="resp_class",
                    resp_status="status",
                    wgt_name="nr_wgt",
                    respondents_only=False,
                )
                .data.select(["nr_wgt"] + [f"nr_wgt{i}" for i in range(1, 6)])
                .to_numpy()
            )

        assert np.allclose(_run(), _run(), equal_nan=True)

    def test_bootstrap_adjust_creates_rep_columns(self, bootstrap_adjustment_sample):
        sample = (
            Sample(data=bootstrap_adjustment_sample, design=Design(wgt="base_wgt", psu="psu"))
            .weighting.create_bs_wgts(n_reps=5, rstate=np.random.default_rng(seed=19))
            .weighting.adjust(
                by="resp_class",
                resp_status="status",
                wgt_name="nr_wgt",
                respondents_only=False,
            )
        )
        assert "nr_wgt" in sample.data.columns
        for i in range(1, 6):
            assert f"nr_wgt{i}" in sample.data.columns


class TestSDRWeights:
    def test_sdr_basic(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_sdr_wgts(n_reps=4)
        assert sample.design.rep_wgts.method.value == "SDR"
        assert sample.design.rep_wgts.n_reps == 4

    def test_sdr_default_prefix_uses_design_wgt(self, simple_stratified_sample):
        """Without rep_prefix, prefix matches design.wgt."""
        sample = simple_stratified_sample.weighting.create_sdr_wgts(n_reps=4)
        assert sample.design.rep_wgts.prefix == "wgt"

    def test_sdr_with_order_col(self, simple_stratified_sample):
        data = simple_stratified_sample.data.with_columns(pl.col("id").alias("sort_order"))
        sample = Sample(data=data, design=simple_stratified_sample.design)
        sample = sample.weighting.create_sdr_wgts(n_reps=4, order_col="sort_order")
        assert sample.design.rep_wgts.n_reps == 4


class TestCreateVarianceStrata:
    def test_variance_strata_brr_even(self, multi_psu_sample):
        sample = multi_psu_sample.weighting.create_variance_strata(method="brr")
        assert "svy_var_stratum" in sample.data.columns
        assert sample.data["svy_var_stratum"].n_unique() == 6
        psu_counts = (
            sample.data.select(["svy_var_stratum", "psu"])
            .unique()
            .group_by("svy_var_stratum")
            .agg(pl.n_unique("psu").alias("n_psu"))
        )
        assert (psu_counts["n_psu"] == 2).all()
        sample = sample.weighting.create_brr_wgts()
        assert sample.design.rep_wgts is not None

    def test_variance_strata_jk2_odd(self, odd_psu_sample):
        sample = odd_psu_sample.weighting.create_variance_strata(method="jk2")
        assert sample.data["svy_var_stratum"].n_unique() == 3
        psu_counts = (
            sample.data.select(["svy_var_stratum", "psu"])
            .unique()
            .group_by("svy_var_stratum")
            .agg(pl.n_unique("psu").alias("n_psu"))
        )
        assert ((psu_counts["n_psu"] >= 2) & (psu_counts["n_psu"] <= 3)).all()
        sample = sample.weighting.create_jk_wgts(paired=True)
        assert sample.design.rep_wgts is not None

    def test_variance_strata_brr_rejects_odd(self, odd_psu_sample):
        with pytest.raises(DimensionError):
            odd_psu_sample.weighting.create_variance_strata(method="brr")

    def test_variance_strata_invalid_method(self, multi_psu_sample):
        with pytest.raises(MethodError):
            multi_psu_sample.weighting.create_variance_strata(method="invalid")

    def test_variance_strata_order_by(self, multi_psu_sample):
        data = multi_psu_sample.data.with_columns(pl.col("y").alias("sort_var"))
        sample = Sample(data=data, design=multi_psu_sample.design)
        sample = sample.weighting.create_variance_strata(method="brr", order_by="sort_var")
        assert "svy_var_stratum" in sample.data.columns

    def test_variance_strata_shuffle_reproducible(self, multi_psu_sample):
        s1 = multi_psu_sample.weighting.create_variance_strata(
            method="brr", shuffle=True, rstate=42
        )
        s2 = multi_psu_sample.weighting.create_variance_strata(
            method="brr", shuffle=True, rstate=42
        )
        np.testing.assert_array_equal(
            s1.data["svy_var_stratum"].to_numpy(),
            s2.data["svy_var_stratum"].to_numpy(),
        )

    def test_variance_strata_custom_name(self, multi_psu_sample):
        sample = multi_psu_sample.weighting.create_variance_strata(
            method="brr", into="my_var_stratum"
        )
        assert "my_var_stratum" in sample.data.columns
        assert sample.design.stratum == "my_var_stratum"

    def test_variance_strata_singleton_raises(self):
        data = pl.DataFrame({"stratum": [1, 2, 2, 2, 2], "psu": [1, 2, 2, 3, 3], "wgt": [1.0] * 5})
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))
        with pytest.raises(DimensionError, match="at least 2 PSUs"):
            sample.weighting.create_variance_strata(method="brr")

    def test_variance_strata_no_psu_raises(self):
        data = pl.DataFrame({"stratum": [1, 1, 2, 2], "wgt": [1.0] * 4})
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum"))
        with pytest.raises(MethodError):
            sample.weighting.create_variance_strata(method="brr")


class TestReplicationIntegration:
    def test_brr_weight_sums(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_brr_wgts(rstate=42)
        original_sum = sample.data["wgt"].sum()
        for col in sample.design.rep_wgts.columns:
            assert abs(sample.data[col].sum() - original_sum) < 1e-10

    def test_jk2_degrees_of_freedom(self, odd_psu_sample):
        sample = odd_psu_sample.weighting.create_jk_wgts(paired=True, rstate=42)
        assert sample.design.rep_wgts.df == 3.0

    def test_variance_strata_then_brr(self, multi_psu_sample):
        sample = multi_psu_sample.weighting.create_variance_strata(
            method="brr"
        ).weighting.create_brr_wgts()
        assert sample.design.rep_wgts.method.value == "BRR"

    def test_variance_strata_then_jk2(self, multi_psu_sample):
        sample = multi_psu_sample.weighting.create_variance_strata(
            method="jk2"
        ).weighting.create_jk_wgts(paired=True)
        assert sample.design.rep_wgts.method.value == "Jackknife"


class TestEdgeCases:
    def test_single_stratum_jkn(self):
        data = pl.DataFrame({"psu": [1, 1, 2, 2, 3, 3], "wgt": [1.0] * 6})
        sample = Sample(data=data, design=Design(wgt="wgt", psu="psu"))
        sample = sample.weighting.create_jk_wgts(paired=False)
        assert sample.design.rep_wgts.n_reps == 3

    def test_large_sample_bootstrap(self):
        n = 1000
        data = pl.DataFrame(
            {
                "stratum": np.repeat([1, 2, 3, 4, 5], n // 5),
                "psu": np.tile(np.arange(1, 21), n // 20),
                "wgt": np.ones(n),
            }
        )
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))
        sample = sample.weighting.create_bs_wgts(n_reps=200, rstate=42)
        assert sample.design.rep_wgts.n_reps == 200

    def test_non_unit_weights_brr(self):
        data = pl.DataFrame(
            {
                "stratum": [1, 1, 2, 2],
                "psu": [1, 2, 3, 4],
                "wgt": [2.0, 3.0, 1.5, 2.5],
            }
        )
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))
        sample = sample.weighting.create_brr_wgts()
        for col in sample.design.rep_wgts.columns:
            ratios = sample.data[col].to_numpy() / sample.data["wgt"].to_numpy()
            assert all(r in [0.0, 2.0] for r in ratios)


# ===========================================================================
# Multi-column (tuple) strata
# ===========================================================================


class TestTupleStrata:
    """Tests for designs where stratum is a tuple of column names."""

    @pytest.fixture
    def tuple_stratum_sample(self):
        """Sample with stratum=("region", "urban") — 4 strata, 2 PSUs each."""
        data = pl.DataFrame(
            {
                "id": list(range(1, 17)),
                "region": [
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "B",
                    "B",
                    "B",
                    "B",
                    "B",
                    "B",
                    "B",
                    "B",
                ],
                "urban": [
                    "U",
                    "U",
                    "U",
                    "U",
                    "R",
                    "R",
                    "R",
                    "R",
                    "U",
                    "U",
                    "U",
                    "U",
                    "R",
                    "R",
                    "R",
                    "R",
                ],
                "psu": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
                "wgt": [1.0] * 16,
            }
        )
        return Sample(
            data=data,
            design=Design(wgt="wgt", stratum=("region", "urban"), psu="psu"),
        )

    @pytest.fixture
    def tuple_stratum_odd_psu_sample(self):
        """Sample with tuple stratum and odd PSU count per stratum."""
        data = pl.DataFrame(
            {
                "id": list(range(1, 19)),
                "region": ["A"] * 6 + ["A"] * 6 + ["B"] * 6,
                "urban": ["U"] * 6 + ["R"] * 6 + ["U"] * 6,
                "psu": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
                "wgt": [1.0] * 18,
            }
        )
        return Sample(
            data=data,
            design=Design(wgt="wgt", stratum=("region", "urban"), psu="psu"),
        )

    def test_bootstrap_with_tuple_strata(self, tuple_stratum_sample):
        sample = tuple_stratum_sample.weighting.create_bs_wgts(n_reps=50, rstate=42)
        assert sample.design.rep_wgts is not None
        assert sample.design.rep_wgts.n_reps == 50
        assert sample.design.rep_wgts.method.value == "Bootstrap"

    def test_jkn_with_tuple_strata(self, tuple_stratum_sample):
        sample = tuple_stratum_sample.weighting.create_jk_wgts(paired=False)
        assert sample.design.rep_wgts is not None
        assert sample.design.rep_wgts.n_reps == 8  # 8 PSUs

    def test_variance_strata_jk2_with_tuple_strata(self, tuple_stratum_odd_psu_sample):
        """create_variance_strata should work with tuple strata (the bug fix)."""
        sample = tuple_stratum_odd_psu_sample.weighting.create_variance_strata(
            method="jk2",
            into="var_stratum",
        )
        assert "var_stratum" in sample.data.columns
        # Should be able to create JK2 replicates after
        sample = sample.weighting.create_jk_wgts(paired=True)
        assert sample.design.rep_wgts is not None

    def test_variance_strata_brr_with_tuple_strata(self, tuple_stratum_sample):
        """create_variance_strata(method='brr') with tuple strata."""
        sample = tuple_stratum_sample.weighting.create_variance_strata(method="brr")
        assert "svy_var_stratum" in sample.data.columns
        # Each variance stratum should have exactly 2 PSUs
        psu_counts = (
            sample.data.select(["svy_var_stratum", "psu"])
            .unique()
            .group_by("svy_var_stratum")
            .agg(pl.n_unique("psu").alias("n_psu"))
        )
        assert (psu_counts["n_psu"] == 2).all()
        # BRR creation should work
        sample = sample.weighting.create_brr_wgts()
        assert sample.design.rep_wgts is not None


# ===========================================================================
# Propagation through adjustment pipeline
# ===========================================================================


class TestReplicatePropagation:
    """Tests that replicate weights propagate through adjustment methods."""

    @pytest.fixture
    def sample_with_replicates(self, bootstrap_adjustment_sample):
        """Sample with bootstrap replicates created."""
        return Sample(
            data=bootstrap_adjustment_sample,
            design=Design(wgt="base_wgt", stratum="strata", psu="psu"),
        ).weighting.create_bs_wgts(n_reps=5, rstate=np.random.default_rng(seed=42))

    def test_adjust_renames_replicates(self, sample_with_replicates):
        """After adjust(wgt_name='nr_wgt'), replicates should be nr_wgt1..nr_wgt5."""
        sample = sample_with_replicates.weighting.adjust(
            resp_status="status",
            by="resp_class",
            wgt_name="nr_wgt",
            respondents_only=False,
        )
        assert sample.design.wgt == "nr_wgt"
        assert sample.design.rep_wgts.prefix == "nr_wgt"
        for i in range(1, 6):
            assert f"nr_wgt{i}" in sample.data.columns

    def test_chained_adjustments_propagate(self, sample_with_replicates):
        """adjust → normalize chain: replicates follow both steps."""
        sample = sample_with_replicates.weighting.adjust(
            resp_status="status",
            by="resp_class",
            wgt_name="nr_wgt",
            respondents_only=False,
        ).weighting.normalize(
            controls=100,
            wgt_name="norm_wgt",
        )
        assert sample.design.wgt == "norm_wgt"
        assert sample.design.rep_wgts.prefix == "norm_wgt"
        for i in range(1, 6):
            assert f"norm_wgt{i}" in sample.data.columns
        # Normalized main weight should sum to 100
        np.testing.assert_allclose(sample.data["norm_wgt"].sum(), 100.0, rtol=1e-6)

    def test_ignore_reps_skips_replicates(self, sample_with_replicates):
        """ignore_reps=True should only adjust the main weight."""
        sample = sample_with_replicates.weighting.normalize(
            controls=100,
            wgt_name="norm_wgt",
            ignore_reps=True,
        )
        assert sample.design.wgt == "norm_wgt"
        # Replicate prefix should NOT be updated
        assert sample.design.rep_wgts.prefix == "base_wgt"
        # Original replicate columns should still exist
        for i in range(1, 6):
            assert f"base_wgt{i}" in sample.data.columns


# ===========================================================================
# Custom rep_prefix parameter
# ===========================================================================


class TestCustomRepPrefix:
    def test_bootstrap_custom_prefix(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_bs_wgts(
            n_reps=10,
            rep_prefix="my_bs",
            rstate=42,
        )
        assert sample.design.rep_wgts.prefix == "my_bs"
        for i in range(1, 11):
            assert f"my_bs{i}" in sample.data.columns

    def test_jkn_custom_prefix(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_jk_wgts(
            paired=False,
            rep_prefix="jk_rep",
        )
        assert sample.design.rep_wgts.prefix == "jk_rep"
        for col in sample.design.rep_wgts.columns:
            assert col.startswith("jk_rep")

    def test_sdr_custom_prefix(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_sdr_wgts(
            n_reps=4,
            rep_prefix="sdr_rep",
        )
        assert sample.design.rep_wgts.prefix == "sdr_rep"
        assert all(c.startswith("sdr_rep") for c in sample.design.rep_wgts.columns)
