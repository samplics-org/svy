# tests/svy/core/test_sample_select_pps.py
import numpy as np
import polars as pl
import pytest

from svy import Design, Sample


# --- Tiny fixture data -------------------------------------------------------

DF = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "name": [
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Eve",
            "Frank",
            "Grace",
            "Hannah",
            "Isaac",
            "Jack",
            "Kate",
            "Liam",
            "Mia",
            "Nora",
            "Oliver",
        ],
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        "education": [
            "Less than HS",
            "HS or higher",
            "HS or higher",
            "HS or higher",
            "Less than HS",
            "Less than HS",
            "Less than HS",
            "HS or higher",
            "HS or higher",
            "Less than HS",
            "HS or higher",
            "HS or higher",
            "HS or higher",
            "Less than HS",
            "Less than HS",
        ],
        "income": [
            50000,
            60000,
            70000,
            80000,
            90000,
            100000,
            11000,
            120000,
            130000,
            140000,
            150000,
            16000,
            17000,
            18000,
            19000,
        ],
    }
)


def _with_region(df: pl.DataFrame) -> pl.DataFrame:
    """Add alternating North/South region column for 2×2 tests."""
    region = ["North" if i % 2 == 0 else "South" for i in range(df.height)]
    return df.with_columns(pl.Series("region", region))


# =============================================================================
# PPS SYS — core correctness
# =============================================================================


class TestPPSSysProbs:
    def test_unstratified_probs_and_hits(self):
        """Hits sum to n; prob_i = n * mos_i / sum(mos) for all selected."""
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        n = 3
        samp2 = samp.sampling.pps_sys(n=n, drop_nulls=True, rstate=np.random.default_rng(0))

        assert int(samp2.data["svy_number_of_hits"].sum()) == n

        incomes = DF["income"].to_numpy()
        total = incomes.sum()
        for row in samp2.data.iter_rows(named=True):
            idx = row["svy_row_index"]
            expected = n * incomes[idx] / total
            assert np.isclose(row["svy_prob_selection"], expected, atol=1e-8)

    def test_stratified_probs_and_hits(self):
        """Hits sum to n × #strata; per-stratum probs correct."""
        design = Design(mos="income", stratum="education")
        samp = Sample(DF.clone(), design)
        n = 3
        samp2 = samp.sampling.pps_sys(n=n, drop_nulls=True, rstate=np.random.default_rng(123))

        strata = DF["education"].unique().to_list()
        assert int(samp2.data["svy_number_of_hits"].sum()) == n * len(strata)

        for grp in strata:
            orig = DF.filter(pl.col("education") == grp)
            total_mos = orig["income"].sum()
            sel = samp2.data.filter(pl.col("education") == grp)
            assert int(sel["svy_number_of_hits"].sum()) == n
            for row in sel.iter_rows(named=True):
                idx = row["svy_row_index"]
                income_i = DF[idx, "income"]
                expected = n * income_i / total_mos
                assert np.isclose(row["svy_prob_selection"], expected, atol=1e-8)

    def test_chains_previous_probabilities(self):
        """When design.prob exists, chaining: pi_new = pi_prev * pi_pps."""
        DF2 = DF.with_columns(pl.lit(0.5).alias("prev_prob"))
        design = Design(mos="income", prob="prev_prob")
        samp = Sample(DF2, design)
        samp2 = samp.sampling.pps_sys(n=2, drop_nulls=True, rstate=np.random.default_rng(7))

        assert (samp2.data["svy_prob_selection"] <= 0.5 + 1e-10).all()

        total_income = DF2["income"].sum()
        inferred_pps = (samp2.data["svy_prob_selection"] / 0.5).to_numpy()
        for idx, p in zip(samp2.data["svy_row_index"].to_list(), inferred_pps):
            expected = 2 * DF2[idx, "income"] / total_income
            assert np.isclose(p, expected, atol=1e-8)

    def test_requires_mos(self):
        """Missing MOS should raise ValueError."""
        design = Design()
        samp = Sample(DF, design)
        with pytest.raises(Exception, match="MOS"):
            samp.sampling.pps_sys(n=2, drop_nulls=True)

    def test_weight_is_inverse_prob(self):
        """svy_sample_weight == 1 / svy_prob_selection for all rows."""
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        samp2 = samp.sampling.pps_sys(n=4, drop_nulls=True, rstate=np.random.default_rng(5))
        for row in samp2.data.iter_rows(named=True):
            assert np.isclose(
                row["svy_sample_weight"],
                1.0 / row["svy_prob_selection"],
                atol=1e-8,
            )

    def test_svy_certainty_column_present(self):
        """svy_certainty column must always be present and boolean."""
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        samp2 = samp.sampling.pps_sys(n=3, drop_nulls=True, rstate=np.random.default_rng(0))
        assert "svy_certainty" in samp2.data.columns
        assert samp2.data["svy_certainty"].dtype == pl.Boolean


# =============================================================================
# PPS SYS — order_by / order_type
# =============================================================================


class TestPPSSysOrdering:
    def test_order_by_ascending_does_not_change_hit_count(self):
        """Ordering should not change how many units are selected."""
        design = Design(mos="income", stratum="education")
        n = 2
        s1 = Sample(DF.clone(), design).sampling.pps_sys(
            n=n,
            order_by="age",
            order_type="ascending",
            drop_nulls=True,
            rstate=np.random.default_rng(0),
        )
        s2 = Sample(DF.clone(), design).sampling.pps_sys(
            n=n,
            order_by="age",
            order_type="descending",
            drop_nulls=True,
            rstate=np.random.default_rng(0),
        )
        assert int(s1.data["svy_number_of_hits"].sum()) == n * 2
        assert int(s2.data["svy_number_of_hits"].sum()) == n * 2

    def test_order_type_random_same_hit_count(self):
        """Random frame order should not change total number of hits."""
        design = Design(mos="income", stratum="education")
        n = 2
        s1 = Sample(DF.clone(), design).sampling.pps_sys(
            n=n,
            order_type="random",
            drop_nulls=True,
            rstate=np.random.default_rng(0),
        )
        assert int(s1.data["svy_number_of_hits"].sum()) == n * 2

    def test_order_type_random_deterministic_with_seed(self):
        """Same seed → same selected indices."""
        design = Design(mos="income")
        s1 = Sample(DF.clone(), design).sampling.pps_sys(
            n=3,
            order_type="random",
            drop_nulls=True,
            rstate=np.random.default_rng(42),
        )
        s2 = Sample(DF.clone(), design).sampling.pps_sys(
            n=3,
            order_type="random",
            drop_nulls=True,
            rstate=np.random.default_rng(42),
        )
        assert sorted(s1.data["svy_row_index"].to_list()) == sorted(
            s2.data["svy_row_index"].to_list()
        )

    def test_order_type_random_can_differ_from_natural(self):
        """With random ordering, selected units sometimes differ from natural order."""
        design = Design(mos="income")
        natural = (
            Sample(DF.clone(), design)
            .sampling.pps_sys(
                n=3,
                drop_nulls=True,
                rstate=np.random.default_rng(0),
            )
            .data["svy_row_index"]
            .to_list()
        )

        results = [
            sorted(
                Sample(DF.clone(), design)
                .sampling.pps_sys(
                    n=3,
                    order_type="random",
                    drop_nulls=True,
                    rstate=np.random.default_rng(seed),
                )
                .data["svy_row_index"]
                .to_list()
            )
            for seed in range(20)
        ]
        assert any(r != sorted(natural) for r in results)

    def test_order_by_with_order_type_ascending_default(self):
        """order_by without order_type defaults to ascending."""
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        samp2 = samp.sampling.pps_sys(
            n=3, order_by="age", drop_nulls=True, rstate=np.random.default_rng(0)
        )
        assert int(samp2.data["svy_number_of_hits"].sum()) == 3

    def test_shuffle_parameter_removed(self):
        """shuffle parameter no longer exists — should raise TypeError."""
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        with pytest.raises(TypeError):
            samp.sampling.pps_sys(n=2, shuffle=True, drop_nulls=True)

    def test_sort_by_parameter_removed(self):
        """sort_by parameter no longer exists — should raise TypeError."""
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        with pytest.raises(TypeError):
            samp.sampling.pps_sys(n=2, sort_by="age", drop_nulls=True)


# =============================================================================
# PPS SYS — certainty threshold
# =============================================================================


class TestPPSSysCertainty:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_certainty_units_get_prob_one(self):
        """Units detected as certainty must have prob=1.0."""
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        samp2 = samp.sampling.pps_sys(n=5, drop_nulls=True, rstate=np.random.default_rng(0))
        for row in samp2.data.iter_rows(named=True):
            if row["svy_certainty"]:
                assert row["svy_prob_selection"] == 1.0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_custom_certainty_threshold(self):
        """With threshold < 1.0, more units should be marked as certainty."""
        design = Design(mos="income")
        samp_default = Sample(DF.clone(), design).sampling.pps_sys(
            n=3,
            certainty_threshold=1.0,
            drop_nulls=True,
            rstate=np.random.default_rng(0),
        )
        samp_low = Sample(DF.clone(), design).sampling.pps_sys(
            n=3,
            certainty_threshold=0.5,
            drop_nulls=True,
            rstate=np.random.default_rng(0),
        )
        n_cert_default = int(samp_default.data["svy_certainty"].sum())
        n_cert_low = int(samp_low.data["svy_certainty"].sum())
        assert n_cert_low >= n_cert_default


# =============================================================================
# PPS SYS — sublevel n mapping
# =============================================================================


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sample_select_pps_sys_sublevel_mapping_broadcasts_by_component():
    """
    Sub-level n mapping broadcasts across all strata containing that level.
    Certainty units correctly get prob=1.0; non-certainty units match formula.
    """
    DF2 = _with_region(DF)
    design = Design(mos="income", stratum="region")
    samp = Sample(DF2, design)
    n_map = {"Less than HS": 2, "HS or higher": 3}
    samp2 = samp.sampling.pps_sys(
        n=n_map, by="education", drop_nulls=True, rstate=np.random.default_rng(42)
    )

    n_regions = DF2["region"].n_unique()
    assert int(samp2.data["svy_number_of_hits"].sum()) <= sum(n_map.values()) * n_regions
    assert samp2.data["svy_certainty"].dtype == pl.Boolean

    for row in samp2.data.iter_rows(named=True):
        if row["svy_certainty"]:
            assert row["svy_prob_selection"] == 1.0
        else:
            reg = row["region"]
            ed = row["education"]
            n_g = n_map[ed]
            idx = row["svy_row_index"]
            income_i = DF2[idx, "income"]
            cell = DF2.filter((pl.col("region") == reg) & (pl.col("education") == ed))
            total_mos = cell["income"].sum()
            expected = min(1.0, n_g * income_i / total_mos)
            assert np.isclose(row["svy_prob_selection"], expected, atol=1e-8), (
                f"reg={reg}, ed={ed}, idx={idx}, income={income_i}, "
                f"got={row['svy_prob_selection']:.6f}, expected={expected:.6f}"
            )


def test_sample_select_pps_sys_sublevel_unrecognized_keys_raises():
    """Keys matching no combined key component should raise ValueError."""
    DF2 = _with_region(DF)
    design = Design(mos="income", stratum="region")
    samp = Sample(DF2, design)
    with pytest.raises(Exception, match="keys"):
        samp.sampling.pps_sys(
            n={"ZZZ": 2},
            by="education",
            drop_nulls=True,
            rstate=np.random.default_rng(0),
        )


def test_sample_select_pps_sys_sublevel_ambiguous_keys_raises():
    """Keys matching multiple components of the same combined key raise ValueError."""
    DF2 = _with_region(DF)
    design = Design(mos="income", stratum="region")
    samp = Sample(DF2, design)
    with pytest.raises(Exception, match="keys"):
        samp.sampling.pps_sys(
            n={"North": 1, "Less than HS": 2},
            by="education",
            drop_nulls=True,
            rstate=np.random.default_rng(0),
        )


# =============================================================================
# Other PPS methods — smoke tests
# =============================================================================


class TestOtherPPSMethods:
    """Smoke tests: correct hit count and prob range for non-SYS methods."""

    def test_pps_brewer_hits_and_probs(self):
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        n = 4
        samp2 = samp.sampling.pps_brewer(n=n, drop_nulls=True, rstate=np.random.default_rng(1))
        assert int(samp2.data["svy_number_of_hits"].sum()) == n
        assert (samp2.data["svy_prob_selection"] > 0).all()
        assert (samp2.data["svy_prob_selection"] <= 1.0 + 1e-10).all()

    def test_pps_rs_hits_and_probs(self):
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        n = 4
        samp2 = samp.sampling.pps_rs(n=n, drop_nulls=True, rstate=np.random.default_rng(2))
        assert int(samp2.data["svy_number_of_hits"].sum()) == n
        assert (samp2.data["svy_prob_selection"] > 0).all()

    def test_pps_murphy_hits(self):
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        samp2 = samp.sampling.pps_murphy(n=2, drop_nulls=True, rstate=np.random.default_rng(3))
        assert int(samp2.data["svy_number_of_hits"].sum()) == 2

    def test_pps_wr_hits(self):
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        n = 5
        samp2 = samp.sampling.pps_wr(n=n, drop_nulls=True, rstate=np.random.default_rng(4))
        assert int(samp2.data["svy_number_of_hits"].sum()) == n

    def test_pps_brewer_no_order_params(self):
        """pps_brewer should not accept order_by or order_type."""
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        with pytest.raises(TypeError):
            samp.sampling.pps_brewer(n=2, order_by="age", drop_nulls=True)

    def test_pps_rs_no_order_params(self):
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        with pytest.raises(TypeError):
            samp.sampling.pps_rs(n=2, order_type="random", drop_nulls=True)

    def test_pps_murphy_no_order_params(self):
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        with pytest.raises(TypeError):
            samp.sampling.pps_murphy(n=2, order_type="descending", drop_nulls=True)

    def test_pps_wr_no_order_params(self):
        design = Design(mos="income")
        samp = Sample(DF.clone(), design)
        with pytest.raises(TypeError):
            samp.sampling.pps_wr(n=2, order_by="age", drop_nulls=True)
