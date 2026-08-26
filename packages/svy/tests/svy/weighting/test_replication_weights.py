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
        assert sample.design.rep_wgts.method == "BRR"
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

    def test_brr_custom_n_reps_beyond_hadamard_errors(self, simple_stratified_sample):
        """3 strata -> Hadamard order 4; requesting more replicates than the
        order used to silently duplicate columns and is now a typed error."""
        from svy.errors import MethodError

        with pytest.raises(MethodError, match="n_reps"):
            simple_stratified_sample.weighting.create_brr_wgts(n_reps=8)

    def test_brr_small_n_reps_rounds_up_to_hadamard(self, simple_stratified_sample):
        """Balance needs the full Hadamard set: n_reps below the order is
        rounded up (documented behavior)."""
        sample = simple_stratified_sample.weighting.create_brr_wgts(n_reps=2)
        assert sample.design.rep_wgts.n_reps == 4

    def test_brr_every_psu_participates(self, simple_stratified_sample):
        """Regression: the stratum on the all-ones Hadamard row had one PSU
        zeroed in EVERY replicate. Mean replicate weight must equal the base
        weight for every observation."""
        sample = simple_stratified_sample.weighting.create_brr_wgts()
        prefix = sample.design.rep_wgts.prefix
        rep_cols = [c for c in sample.data.columns if c.startswith(prefix) and c != "wgt"]
        rep_cols = [c for c in rep_cols if c[len(prefix) :].isdigit()]
        assert rep_cols
        import numpy as np

        mat = sample.data.select(rep_cols).to_numpy()
        base = sample.data.get_column("wgt").to_numpy()
        assert np.allclose(mat.mean(axis=1), base)
        assert (mat > 0).sum(axis=1).min() == len(rep_cols) // 2

    def test_brr_custom_rep_prefix(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_brr_wgts(n_reps=4, rep_prefix="brr_rep")
        rep_cols = [c for c in sample.data.columns if c.startswith("brr_rep")]
        assert len(rep_cols) == 4

    def test_brr_no_longer_requires_a_stratum(self, unstratified_sample):
        """Pairing derives variance strata from the PSUs, so an unstratified
        design is buildable rather than an error telling you to go and make
        strata by hand first."""
        s = unstratified_sample.weighting.create_brr_wgts()
        assert s.design.rep_wgts is not None
        assert s.design.rep_wgts.stratum == "svy_var_stratum"
        assert s.design.stratum is None  # the Design is left alone

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
        with pytest.raises(DimensionError) as exc:
            odd_psu_sample.weighting.create_brr_wgts()

        err = exc.value
        assert err.code == "ODD_PSU_COUNT"
        assert err.where == "Sample.weighting.create_brr_wgts"

        # The remedy has to name an API that exists. This hint used to read
        # "Use method='jk2'", which was create_variance_strata's parameter --
        # a function removed in 0.25.0, and `method` is not a parameter of
        # create_brr_wgts either.
        assert "create_jk_wgts(paired=True)" in err.hint
        assert "method=" not in err.hint
        assert "create_variance_strata" not in err.hint

        # More than one stratum can be at fault, and each needs its own fix,
        # so the remedy must not read as one PSU overall.
        assert "every stratum" in err.hint

    @pytest.mark.parametrize(
        "n_odd_strata, expected_count_phrase",
        [(1, "Found 1 stratum whose"), (3, "Found 3 strata whose")],
    )
    def test_odd_psu_message_agrees_in_number(self, n_odd_strata, expected_count_phrase):
        """`stratum`/`strata` is irregular, so the package's `(s)` suffix fails here."""
        strata, psus, psu_id = [], [], 1
        for i in range(n_odd_strata):
            for _ in range(3):  # 3 PSUs -> odd
                strata += [f"s{i}"] * 2
                psus += [psu_id, psu_id]
                psu_id += 1
        data = pl.DataFrame({"stratum": strata, "psu": psus, "wgt": [1.0] * len(psus)})
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))

        with pytest.raises(DimensionError) as exc:
            sample.weighting.create_brr_wgts()
        assert expected_count_phrase in exc.value.detail

    def test_one_error_lists_every_stratum_that_is_not_a_multiple_of_2(self):
        """1 and 3 are the same failure, so they arrive in one error, not two runs.

        These used to be separate guards with the `< 2` one raising first, so a
        frame carrying both reported only the lone-PSU stratum and revealed the
        odd ones on the next run.
        """
        counts = {"a": 1, "b": 3, "c": 5, "d": 4}  # d pairs cleanly, must not appear
        strata, psus, psu_id = [], [], 1
        for name, n in counts.items():
            for _ in range(n):
                strata += [name] * 2
                psus += [psu_id, psu_id]
                psu_id += 1
        data = pl.DataFrame({"stratum": strata, "psu": psus, "wgt": [1.0] * len(psus)})
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))

        with pytest.raises(DimensionError) as exc:
            sample.weighting.create_brr_wgts()

        err = exc.value
        assert err.code == "ODD_PSU_COUNT"
        assert "3 strata" in err.detail
        for offender in ("a=1", "b=3", "c=5"):
            assert offender in err.got
        assert "d=" not in err.got, "a stratum that pairs cleanly must not be reported"

    @pytest.mark.parametrize("n_psus", [2, 4, 6])
    def test_any_multiple_of_two_is_accepted(self, n_psus):
        """Pairing is why the test is 'multiple of 2', not 'equal to 2'."""
        strata, psus, psu_id = [], [], 1
        for name in ("a", "b", "c"):
            for _ in range(n_psus):
                strata += [name] * 2
                psus += [psu_id, psu_id]
                psu_id += 1
        data = pl.DataFrame({"stratum": strata, "psu": psus, "wgt": [1.0] * len(psus)})
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))

        out = sample.weighting.create_brr_wgts(rep_prefix="r")
        assert out._design.rep_wgts.method == "BRR"

    def test_jk2_still_rejects_a_lone_psu(self):
        """The `< 2` guard is now jk2's alone -- it can triple, but not pair one."""
        data = pl.DataFrame({"stratum": [1, 2, 2, 2, 2], "psu": [1, 2, 2, 3, 3], "wgt": [1.0] * 5})
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))
        with pytest.raises(DimensionError) as exc:
            sample.weighting.create_jk_wgts(paired=True)
        assert exc.value.code == "INSUFFICIENT_PSU"

    def test_odd_psu_got_says_how_many_it_withheld(self):
        """Each offending stratum needs fixing, so a bare `...` hides the job."""
        strata, psus, psu_id = [], [], 1
        for i in range(7):
            for _ in range(3):
                strata += [f"s{i}"] * 2
                psus += [psu_id, psu_id]
                psu_id += 1
        data = pl.DataFrame({"stratum": strata, "psu": psus, "wgt": [1.0] * len(psus)})
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))

        with pytest.raises(DimensionError) as exc:
            sample.weighting.create_brr_wgts()
        got = exc.value.got
        assert "(+2 more)" in got  # 7 offenders, 5 shown
        assert "..." not in got

    def test_the_odd_psu_hint_actually_works(self, odd_psu_sample):
        """Follow the hint and it succeeds on the frame that raised."""
        out = odd_psu_sample.weighting.create_jk_wgts(paired=True, rep_prefix="jk")
        assert out._design.rep_wgts.method == "Jackknife"
        assert out._design.rep_wgts.kind == "jk2"

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
        assert sample.design.rep_wgts.method == "Jackknife"
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
        assert sample.design.rep_wgts.method == "Jackknife"
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
        assert sample.design.rep_wgts.method == "Bootstrap"
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
        assert sample.design.rep_wgts.method == "SDR"
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


class TestVarianceStrataPairing:
    """Pairing is now a step inside the generators, not a public pre-step.

    It used to be `sample.weighting.create_variance_strata(...)`, whose only way
    of handing its result to the generator was to overwrite `Design.stratum` --
    so building BRR/JK2 weights destroyed the design they would be compared
    against. It is exercised through the generators here; `_pair_variance_strata`
    is called directly only where the algorithm itself is the subject.
    """

    def test_brr_pairs_even_psu_strata(self, multi_psu_sample):
        sample = multi_psu_sample.weighting.create_brr_wgts()
        assert "svy_var_stratum" in sample.data.columns
        assert sample.data["svy_var_stratum"].n_unique() == 6
        psu_counts = (
            sample.data.select(["svy_var_stratum", "psu"])
            .unique()
            .group_by("svy_var_stratum")
            .agg(pl.n_unique("psu").alias("n_psu"))
        )
        assert (psu_counts["n_psu"] == 2).all()
        assert sample.design.rep_wgts is not None

    def test_the_design_stratum_survives_pairing(self, multi_psu_sample):
        """The whole point: Taylor keeps the true strata."""
        before = multi_psu_sample.design.stratum
        sample = multi_psu_sample.weighting.create_brr_wgts()
        assert sample.design.stratum == before == "stratum"
        assert sample.design.rep_wgts.stratum == "svy_var_stratum"

    def test_jk2_pairs_odd_psu_strata(self, odd_psu_sample):
        sample = odd_psu_sample.weighting.create_jk_wgts(paired=True)
        assert sample.data["svy_var_stratum"].n_unique() == 3
        psu_counts = (
            sample.data.select(["svy_var_stratum", "psu"])
            .unique()
            .group_by("svy_var_stratum")
            .agg(pl.n_unique("psu").alias("n_psu"))
        )
        assert ((psu_counts["n_psu"] >= 2) & (psu_counts["n_psu"] <= 3)).all()
        assert sample.design.rep_wgts is not None

    def test_jk2_on_more_than_two_psus_no_longer_undercounts(self, multi_psu_sample):
        """Skipping the old pre-step gave one replicate per *original* stratum,
        in silence. Pairing makes it one per variance stratum."""
        sample = multi_psu_sample.weighting.create_jk_wgts(paired=True)
        assert sample.design.rep_wgts.n_reps == 6  # not 3

    def test_already_paired_strata_are_left_alone(self, simple_stratified_sample):
        """A design at two PSUs per stratum is its own variance-stratum scheme;
        re-deriving one would rename the column for nothing."""
        sample = simple_stratified_sample.weighting.create_jk_wgts(paired=True)
        assert "svy_var_stratum" not in sample.data.columns
        assert sample.design.rep_wgts.stratum == "stratum"

    def test_brr_rejects_odd_psu_counts(self, odd_psu_sample):
        with pytest.raises(DimensionError):
            odd_psu_sample.weighting.create_brr_wgts()

    def test_order_by_is_accepted_by_the_generator(self, multi_psu_sample):
        data = multi_psu_sample.data.with_columns(pl.col("y").alias("sort_var"))
        sample = Sample(data=data, design=multi_psu_sample.design)
        sample = sample.weighting.create_brr_wgts(order_by="sort_var")
        assert "svy_var_stratum" in sample.data.columns

    def test_shuffle_is_reproducible(self, multi_psu_sample):
        s1 = multi_psu_sample.weighting.create_brr_wgts(shuffle=True, rstate=42)
        s2 = multi_psu_sample.weighting.create_brr_wgts(shuffle=True, rstate=42)
        np.testing.assert_array_equal(
            s1.data["svy_var_stratum"].to_numpy(),
            s2.data["svy_var_stratum"].to_numpy(),
        )

    def test_stratum_name_names_the_created_column(self, multi_psu_sample):
        """The house convention: the caller names what gets created."""
        sample = multi_psu_sample.weighting.create_brr_wgts(stratum_name="my_var_stratum")
        assert "my_var_stratum" in sample.data.columns
        assert sample.design.rep_wgts.stratum == "my_var_stratum"
        assert sample.design.stratum == "stratum"

    def test_singleton_stratum_raises(self):
        """BRR rejects a lone PSU as "not a multiple of 2", not as a singleton.

        A stratum of 1 and a stratum of 3 fail for the same reason -- a PSU with
        no partner -- so BRR reports them the same way. Whether a lone PSU is a
        *singleton* is a Taylor question, and `Sample.singleton` owns it.
        """
        data = pl.DataFrame({"stratum": [1, 2, 2, 2, 2], "psu": [1, 2, 2, 3, 3], "wgt": [1.0] * 5})
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum", psu="psu"))
        with pytest.raises(DimensionError) as exc:
            sample.weighting.create_brr_wgts()
        assert exc.value.code == "ODD_PSU_COUNT"
        assert "1=1" in exc.value.got

    def test_no_psu_raises(self):
        data = pl.DataFrame({"stratum": [1, 1, 2, 2], "wgt": [1.0] * 4})
        sample = Sample(data=data, design=Design(wgt="wgt", stratum="stratum"))
        with pytest.raises(MethodError):
            sample.weighting.create_brr_wgts()

    def test_an_explicit_unit_column_must_exist(self, multi_psu_sample):
        with pytest.raises(MethodError):
            multi_psu_sample.weighting.create_brr_wgts(stratum="not_a_column")

    def test_explicit_units_override_the_design(self, multi_psu_sample):
        """Build from columns the Design does not name, without mutating it."""
        data = multi_psu_sample.data.with_columns(pl.col("stratum").alias("vstrat"))
        sample = Sample(data=data, design=multi_psu_sample.design)
        out = sample.weighting.create_jk_wgts(stratum="vstrat")
        assert out.design.rep_wgts.stratum == "vstrat"
        assert out.design.stratum == "stratum"

    def test_pairing_is_no_longer_public(self, multi_psu_sample):
        assert not hasattr(multi_psu_sample.weighting, "create_variance_strata")


class TestReplicationIntegration:
    def test_brr_weight_sums(self, simple_stratified_sample):
        sample = simple_stratified_sample.weighting.create_brr_wgts(rstate=42)
        original_sum = sample.data["wgt"].sum()
        for col in sample.design.rep_wgts.columns:
            assert abs(sample.data[col].sum() - original_sum) < 1e-10

    def test_jk2_degrees_of_freedom(self, odd_psu_sample):
        sample = odd_psu_sample.weighting.create_jk_wgts(paired=True, rstate=42)
        assert sample.design.rep_wgts.df == 3.0

    def test_brr_builds_in_one_step(self, multi_psu_sample):
        sample = multi_psu_sample.weighting.create_brr_wgts()
        assert sample.design.rep_wgts.method == "BRR"

    def test_jk2_builds_in_one_step(self, multi_psu_sample):
        sample = multi_psu_sample.weighting.create_jk_wgts(paired=True)
        assert sample.design.rep_wgts.method == "Jackknife"


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
        assert sample.design.rep_wgts.method == "Bootstrap"

    def test_jkn_with_tuple_strata(self, tuple_stratum_sample):
        sample = tuple_stratum_sample.weighting.create_jk_wgts(paired=False)
        assert sample.design.rep_wgts is not None
        assert sample.design.rep_wgts.n_reps == 8  # 8 PSUs

    def test_jk2_pairing_with_tuple_strata(self, tuple_stratum_odd_psu_sample):
        """Pairing resolves a multi-column stratum through the internal
        concatenated column; the pairing itself is single-column by
        construction, so the result is nameable as one."""
        sample = tuple_stratum_odd_psu_sample.weighting.create_jk_wgts(
            paired=True, stratum_name="var_stratum"
        )
        assert "var_stratum" in sample.data.columns
        assert sample.design.rep_wgts is not None
        assert sample.design.rep_wgts.stratum == "var_stratum"
        # The tuple stratum on the Design is untouched.
        assert isinstance(sample.design.stratum, tuple)

    def test_brr_on_already_paired_tuple_strata_does_not_pair(self, tuple_stratum_sample):
        """This fixture is 4 strata x 2 PSUs -- already its own variance-stratum
        scheme, so no column is created and none is recorded."""
        sample = tuple_stratum_sample.weighting.create_brr_wgts()
        assert "svy_var_stratum" not in sample.data.columns
        assert sample.design.rep_wgts is not None
        assert isinstance(sample.design.stratum, tuple)
        # A multi-column unit is recorded as the tuple of its source columns --
        # never the internal concatenated name, which would not resolve against
        # a frame rebuilt from source.
        assert sample.design.rep_wgts.stratum == ("region", "urban")
        assert sample.design.rep_wgts.psu == "psu"


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


# ===========================================================================
# Determinism stress tests
#
# These tests loop many times to surface nondeterministic iteration order
# bugs that pairwise comparisons would miss intermittently.  The existing
# `test_*_reproducibility` tests compare only two calls; if a bug
# manifests in ~35% of calls (as the HashMap iteration-order bug did),
# a pairwise test passes ~70% of the time and only flakes occasionally.
# A 50-run loop reduces the false-negative rate to effectively zero.
# ===========================================================================


class TestDeterminismStress:
    """Loop-based tests that catch intermittent nondeterminism."""

    def test_bootstrap_100_runs_identical(self, simple_stratified_sample):
        """100 bootstrap calls with the same seed must all produce identical weights."""
        baseline = simple_stratified_sample.weighting.create_bs_wgts(n_reps=20, rstate=147)
        baseline_mat = baseline.data.select(baseline.design.rep_wgts.columns).to_numpy()

        for trial in range(100):
            s = simple_stratified_sample.weighting.create_bs_wgts(n_reps=20, rstate=147)
            mat = s.data.select(s.design.rep_wgts.columns).to_numpy()
            np.testing.assert_array_equal(
                baseline_mat,
                mat,
                err_msg=f"Bootstrap weights drifted on trial {trial} with seed=147",
            )

    def test_bootstrap_100_runs_identical_generator(self, simple_stratified_sample):
        """Same determinism requirement when rstate is a fresh Generator."""
        baseline = simple_stratified_sample.weighting.create_bs_wgts(
            n_reps=20, rstate=np.random.default_rng(147)
        )
        baseline_mat = baseline.data.select(baseline.design.rep_wgts.columns).to_numpy()

        for trial in range(100):
            s = simple_stratified_sample.weighting.create_bs_wgts(
                n_reps=20, rstate=np.random.default_rng(147)
            )
            mat = s.data.select(s.design.rep_wgts.columns).to_numpy()
            np.testing.assert_array_equal(
                baseline_mat,
                mat,
                err_msg=f"Bootstrap weights drifted on trial {trial} with Generator(147)",
            )

    def test_bootstrap_many_strata_50_runs(self, multi_psu_sample):
        """Stress with more strata to increase HashMap permutation space."""
        baseline = multi_psu_sample.weighting.create_bs_wgts(n_reps=50, rstate=42)
        baseline_mat = baseline.data.select(baseline.design.rep_wgts.columns).to_numpy()

        for trial in range(50):
            s = multi_psu_sample.weighting.create_bs_wgts(n_reps=50, rstate=42)
            mat = s.data.select(s.design.rep_wgts.columns).to_numpy()
            np.testing.assert_array_equal(baseline_mat, mat, err_msg=f"Drifted on trial {trial}")

    def test_brr_seeded_50_runs_identical(self, simple_stratified_sample):
        """BRR with a seed must be deterministic across many calls."""
        baseline = simple_stratified_sample.weighting.create_brr_wgts(rstate=147)
        baseline_mat = baseline.data.select(baseline.design.rep_wgts.columns).to_numpy()

        for trial in range(50):
            s = simple_stratified_sample.weighting.create_brr_wgts(rstate=147)
            mat = s.data.select(s.design.rep_wgts.columns).to_numpy()
            np.testing.assert_array_equal(
                baseline_mat, mat, err_msg=f"BRR drifted on trial {trial}"
            )

    def test_jk2_seeded_50_runs_identical(self, odd_psu_sample):
        """JK2 with a seed must be deterministic across many calls."""
        baseline = odd_psu_sample.weighting.create_jk_wgts(paired=True, rstate=42)
        baseline_mat = baseline.data.select(baseline.design.rep_wgts.columns).to_numpy()

        for trial in range(50):
            s = odd_psu_sample.weighting.create_jk_wgts(paired=True, rstate=42)
            mat = s.data.select(s.design.rep_wgts.columns).to_numpy()
            np.testing.assert_array_equal(
                baseline_mat, mat, err_msg=f"JK2 drifted on trial {trial}"
            )


# ===========================================================================
# Poisson bootstrap (kind="poisson")
# ===========================================================================
#
# Beaumont & Patak (2012) generalized bootstrap with independent per-unit
# factors. Reference: Statistics Canada, Labour Force Survey PUMF User Guide
# (2025), section 6 and appendices A-D.


@pytest.fixture
def pumf_like_sample():
    """A weight-only file: no stratum, no psu, as a PUMF is published."""
    rng = np.random.default_rng(20260821)
    n = 4000
    data = pl.DataFrame(
        {
            "wgt": rng.uniform(50.0, 500.0, n),
            "prov": rng.integers(0, 4, n),
            "sex": rng.integers(0, 2, n),
            "y": rng.normal(100.0, 15.0, n),
            "flag": rng.random(n) < 0.3,
        }
    )
    return Sample(data=data, design=Design(wgt="wgt"))


def _rep_matrix(sample, prefix, n_reps):
    return np.column_stack([sample._data[f"{prefix}{i + 1}"].to_numpy() for i in range(n_reps)])


@pytest.mark.parametrize(
    "given",
    ["poisson", "Poisson", "POISSON", "  PoIsSoN  "],
)
def test_poisson_kind_is_case_and_whitespace_insensitive(pumf_like_sample, given):
    s = pumf_like_sample.weighting.create_bs_wgts(n_reps=8, kind=given, rep_prefix="r", rstate=1)
    assert s._design.rep_wgts.n_reps == 8


@pytest.mark.parametrize(
    "given", ["rao-wu", "RAO-WU", "Rao_Wu", "raowu", "rw", "rao wu", "rao-wu-yue"]
)
def test_rao_wu_kind_aliases_normalize(multi_psu_sample, given):
    s = multi_psu_sample.weighting.create_bs_wgts(n_reps=8, kind=given, rep_prefix="r", rstate=1)
    assert s._design.rep_wgts.n_reps == 8


def test_unknown_kind_raises_and_names_the_variants(pumf_like_sample):
    with pytest.raises(MethodError) as exc:
        pumf_like_sample.weighting.create_bs_wgts(n_reps=4, kind="rao–wu")  # en dash
    rendered = str(exc.value)
    assert "kind" in rendered
    # the hint should say what each kind needs, not just list the names
    assert "rao-wu" in rendered and "poisson" in rendered
    assert "psu" in rendered


def test_non_string_kind_raises_type_error(pumf_like_sample):
    with pytest.raises(TypeError, match="'kind' must be a string"):
        pumf_like_sample.weighting.create_bs_wgts(n_reps=4, kind=1)


def test_poisson_does_not_require_psu(pumf_like_sample):
    """The whole point of the method: it works where Rao-Wu cannot."""
    assert pumf_like_sample._design.psu is None
    s = pumf_like_sample.weighting.create_bs_wgts(
        n_reps=16, kind="poisson", rep_prefix="r", rstate=3
    )
    assert _rep_matrix(s, "r", 16).shape == (4000, 16)


def test_rao_wu_still_requires_psu(pumf_like_sample):
    with pytest.raises(MethodError, match="psu"):
        pumf_like_sample.weighting.create_bs_wgts(n_reps=8, kind="rao-wu")


def test_poisson_df_is_n_reps_minus_one(pumf_like_sample):
    s = pumf_like_sample.weighting.create_bs_wgts(
        n_reps=32, kind="poisson", rep_prefix="r", rstate=5
    )
    assert s._design.rep_wgts.df == 31.0


def test_poisson_weights_are_strictly_positive(pumf_like_sample):
    """Non-negativity is what makes these weights publishable."""
    s = pumf_like_sample.weighting.create_bs_wgts(
        n_reps=64, kind="poisson", rep_prefix="r", rstate=7
    )
    assert (_rep_matrix(s, "r", 64) > 0).all()


def test_poisson_is_deterministic_under_rstate(pumf_like_sample):
    kw = dict(n_reps=16, kind="poisson", rep_prefix="r", rstate=11)
    a = _rep_matrix(pumf_like_sample.weighting.create_bs_wgts(**kw), "r", 16)
    b = _rep_matrix(pumf_like_sample.weighting.create_bs_wgts(**kw), "r", 16)
    np.testing.assert_array_equal(a, b)


def test_poisson_replicates_differ_from_each_other(pumf_like_sample):
    s = pumf_like_sample.weighting.create_bs_wgts(
        n_reps=8, kind="poisson", rep_prefix="r", rstate=13
    )
    m = _rep_matrix(s, "r", 8)
    assert not np.allclose(m[:, 0], m[:, 1])


def test_poisson_matches_published_worked_example():
    """Appendix C.1 of the LFS PUMF user guide, worked by hand.

    A unit weighted 500 has adjustment factor 1 +/- sqrt(499/500), giving
    replicate weights of 999.500 and 0.50025. The square root is the step the
    guide's prose loses to PDF extraction, so pin it against the published
    arithmetic rather than against our own formula.
    """
    data = pl.DataFrame({"wgt": [500.0, 450.0, 150.0, 250.0]})
    s = Sample(data=data, design=Design(wgt="wgt"))
    s = s.weighting.create_bs_wgts(n_reps=200, kind="poisson", rep_prefix="r", rstate=17)
    m = _rep_matrix(s, "r", 200)

    expected = {
        500.0: (500.0 * (1 + np.sqrt(499 / 500)), 500.0 * (1 - np.sqrt(499 / 500))),
        450.0: (450.0 * (1 + np.sqrt(449 / 450)), 450.0 * (1 - np.sqrt(449 / 450))),
    }
    np.testing.assert_allclose(sorted(np.unique(m[0].round(9)))[::-1][:1], [expected[500.0][0]])
    np.testing.assert_allclose(np.unique(m[0].round(9))[:1], [expected[500.0][1]])
    # Published values, to the five decimals the guide prints.
    assert round(expected[500.0][0], 3) == 999.500
    assert round(expected[500.0][1], 5) == 0.50025


def test_poisson_rejects_weights_below_one():
    """sqrt((w - 1) / w) is not real below 1; a NaN here would be silent."""
    data = pl.DataFrame({"wgt": [10.0, 0.5, 20.0]})
    s = Sample(data=data, design=Design(wgt="wgt"))
    with pytest.raises(ValueError, match="requires weights >= 1"):
        s.weighting.create_bs_wgts(n_reps=4, kind="poisson")


def test_poisson_accepts_weight_of_exactly_one():
    """The boundary is admissible: adjustment is 0, so the replicate is w."""
    data = pl.DataFrame({"wgt": [1.0, 10.0, 20.0]})
    s = Sample(data=data, design=Design(wgt="wgt"))
    s = s.weighting.create_bs_wgts(n_reps=8, kind="poisson", rep_prefix="r", rstate=19)
    np.testing.assert_allclose(_rep_matrix(s, "r", 8)[0], 1.0)


def test_poisson_leaves_the_point_estimate_untouched(pumf_like_sample):
    """Replicate weights change the variance, never the estimate."""
    s = pumf_like_sample.weighting.create_bs_wgts(
        n_reps=64,
        kind="poisson",
        rep_prefix="r",
        rstate=37,
    )
    taylor = pumf_like_sample.estimation.mean("y")
    rep = s.estimation.mean("y", method="replication")
    np.testing.assert_allclose(taylor.to_dicts()[0]["est"], rep.to_dicts()[0]["est"], rtol=1e-12)


def test_bootstrap_kind_is_recorded_on_the_design(multi_psu_sample, pumf_like_sample):
    """create_bs_wgts used to choose the algorithm and then forget it."""
    rw = multi_psu_sample.weighting.create_bs_wgts(n_reps=8, rep_prefix="r", rstate=43)
    assert rw._design.rep_wgts.kind == "rao-wu"

    po = pumf_like_sample.weighting.create_bs_wgts(
        n_reps=8, kind="poisson", rep_prefix="r", rstate=41
    )
    assert po._design.rep_wgts.kind == "poisson"


# ===========================================================================
# Regenerating replicate weights replaces the recorded design
# ===========================================================================


class TestRegeneratedDesignIsRecorded:
    """A generator records the columns it just wrote.

    ``create_brr_wgts`` and ``create_jk_wgts`` used ``Design.fill_missing``,
    which is a no-op once ``rep_wgts`` is set, while ``create_bs_wgts`` and
    ``create_sdr_wgts`` used ``Design.update``. So building a second set of
    replicates wrote the new columns and kept the *first* method's metadata --
    and estimation reads the metadata, not the columns.
    """

    def test_bootstrap_then_jackknife(self, simple_stratified_sample):
        s = simple_stratified_sample.weighting.create_bs_wgts(n_reps=8, rep_prefix="bs", rstate=1)
        s = s.weighting.create_jk_wgts(rep_prefix="jk")

        rw = s._design.rep_wgts
        assert rw.method == "Jackknife"
        assert rw.prefix == "jk"
        assert rw.n_reps == 6  # one per PSU
        assert "jk1" in s._data.columns

    def test_jackknife_then_bootstrap(self, simple_stratified_sample):
        s = simple_stratified_sample.weighting.create_jk_wgts(rep_prefix="jk")
        s = s.weighting.create_bs_wgts(n_reps=8, rep_prefix="bs", rstate=1)

        rw = s._design.rep_wgts
        assert rw.method == "Bootstrap"
        assert rw.prefix == "bs"
        assert rw.n_reps == 8

    def test_jackknife_then_brr(self, simple_stratified_sample):
        s = simple_stratified_sample.weighting.create_jk_wgts(rep_prefix="jk")
        s = s.weighting.create_brr_wgts(rep_prefix="brr")

        rw = s._design.rep_wgts
        assert rw.method == "BRR"
        assert rw.prefix == "brr"

    def test_regenerating_the_same_method_updates_its_parameters(self, simple_stratified_sample):
        s = simple_stratified_sample.weighting.create_jk_wgts(rep_prefix="jk")
        s = s.weighting.create_jk_wgts(paired=True, rep_prefix="jk2")

        rw = s._design.rep_wgts
        assert rw.prefix == "jk2"
        assert rw.kind == "jk2"
        assert rw.n_reps == 3  # one per variance stratum, not per PSU

    def test_estimation_uses_the_regenerated_design(self, simple_stratified_sample):
        """The symptom: an estimate off the second design read the first's columns."""
        s = simple_stratified_sample.weighting.create_bs_wgts(n_reps=8, rep_prefix="bs", rstate=1)
        s = s.weighting.create_jk_wgts(rep_prefix="jk")

        est = s.estimation.mean(y="y", method="replication")
        assert "JACKKNIFE" in str(est)

    def test_a_declared_design_is_replaced_by_what_was_generated(self, simple_stratified_sample):
        """Generating weights overrides a declaration that no longer describes them."""
        from svy.core.repwgts import BootstrapWgts

        declared = simple_stratified_sample._design.update(
            rep_wgts=BootstrapWgts(prefix="declared", n_reps=4)
        )
        s = Sample(
            data=simple_stratified_sample._data.with_columns(
                [pl.lit(1.0).alias(f"declared{i}") for i in range(1, 5)]
            ),
            design=declared,
        )
        s = s.weighting.create_jk_wgts(rep_prefix="jk")

        assert s._design.rep_wgts.method == "Jackknife"
        assert s._design.rep_wgts.prefix == "jk"
