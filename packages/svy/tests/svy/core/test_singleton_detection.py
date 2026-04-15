# tests/svy/core/test_singleton_detection.py
"""
Tests for Sample-level singleton detection (_check_for_singletons method).

Note: The _check_for_singletons method identifies strata with only one PSU,
which is important for variance estimation in complex survey designs.
"""

import polars as pl
import pytest

from svy.core.design import Design
from svy.core.sample import Sample


class TestSingletonDetectionBasic:
    """Basic singleton detection tests."""

    def test_no_singletons_with_stratum_and_psu(self):
        """Test case where no strata have singleton PSUs."""
        data = pl.DataFrame(
            {
                "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "psu": [1, 1, 2, 2, 3, 3, 4, 4],
                "value": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )
        design = Design(stratum="stratum", psu="psu")
        sample = Sample(data=data, design=design)

        assert not sample.singleton.exists
        assert sample.singleton.count == 0

    def test_singleton_psu_in_one_stratum(self):
        """Test detection of singleton PSU in one stratum."""
        data = pl.DataFrame(
            {
                "stratum": ["A", "A", "A", "A", "B", "B"],
                "psu": [1, 1, 2, 2, 3, 3],  # Stratum A has 2 PSUs, B has 1 PSU
                "value": [10, 20, 30, 40, 50, 60],
            }
        )
        design = Design(stratum="stratum", psu="psu")
        sample = Sample(data=data, design=design)

        assert sample.singleton.exists
        assert sample.singleton.count == 1
        assert "B" in str(sample.singleton.keys())

    def test_multiple_singleton_strata(self):
        """Test detection of multiple strata with singleton PSUs."""
        data = pl.DataFrame(
            {
                "stratum": ["A", "A", "B", "B", "C", "C"],
                "psu": [1, 1, 2, 2, 3, 3],  # All strata have only 1 PSU each
                "value": [10, 20, 30, 40, 50, 60],
            }
        )
        design = Design(stratum="stratum", psu="psu")
        sample = Sample(data=data, design=design)

        assert sample.singleton.exists
        assert sample.singleton.count == 3

    def test_mixed_singleton_and_non_singleton_strata(self):
        """Test realistic scenario with mix of singleton and non-singleton strata."""
        data = pl.DataFrame(
            {
                "stratum": ["A", "A", "A", "A", "B", "B", "C", "C", "C", "C"],
                "psu": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],  # A has 2 PSUs, B has 1, C has 2
                "value": range(10),
            }
        )
        design = Design(stratum="stratum", psu="psu")
        sample = Sample(data=data, design=design)

        assert sample.singleton.exists
        assert sample.singleton.count == 1


class TestSingletonDetectionColumnSpecs:
    """Tests for different column specification formats."""

    def test_string_stratum_and_psu(self):
        """Test that string column names work correctly."""
        data = pl.DataFrame(
            {
                "strat_col": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
                "psu_col": [1, 1, 2, 2, 3, 3, 4, 4],
                "value": [100, 200, 300, 400, 500, 600, 700, 800],
            }
        )
        design = Design(stratum="strat_col", psu="psu_col")
        sample = Sample(data=data, design=design)

        assert not sample.singleton.exists

    def test_list_stratum_and_psu(self):
        """Test that list column names work correctly."""
        data = pl.DataFrame(
            {
                "strat1": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "strat2": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
                "psu": [1, 1, 2, 2, 3, 3, 4, 4],
                "value": [100, 200, 300, 400, 500, 600, 700, 800],
            }
        )
        design = Design(stratum=["strat1", "strat2"], psu=["psu"])
        sample = Sample(data=data, design=design)

        assert not sample.singleton.exists

    def test_list_stratum_with_singleton(self):
        """Test list column names with actual singleton detection."""
        data = pl.DataFrame(
            {
                "region": ["North", "North", "South", "South", "West", "West"],
                "district": ["A", "A", "B", "B", "C", "C"],
                "psu": [1, 1, 2, 2, 3, 3],  # Each stratum combo has 1 PSU
                "value": [100, 200, 300, 400, 500, 600],
            }
        )
        design = Design(stratum=["region", "district"], psu=["psu"])
        sample = Sample(data=data, design=design)

        assert sample.singleton.exists
        assert sample.singleton.count == 3

    def test_tuple_stratum_and_psu(self):
        """Test that tuple column names are handled correctly."""
        data = pl.DataFrame(
            {
                "region": ["North", "North", "South", "South"],
                "district": ["A", "A", "B", "B"],
                "cluster": [1, 1, 2, 2],
                "value": [100, 200, 300, 400],
            }
        )
        design = Design(stratum=("region", "district"), psu=("cluster",))
        sample = Sample(data=data, design=design)

        assert sample is not None
        assert sample.singleton.count == 2  # Both strata are singletons


class TestSingletonDetectionPartialDesign:
    """Tests for partial design specifications."""

    def test_psu_only_no_stratum(self):
        """Test singleton detection with PSU but no stratum."""
        data = pl.DataFrame(
            {
                "psu": [1, 1, 2, 2, 3, 4],
                "value": [10, 20, 30, 40, 50, 60],
            }
        )
        design = Design(psu="psu")
        sample = Sample(data=data, design=design)

        assert sample is not None
        assert sample._data is not None

    def test_stratum_only_no_psu(self):
        """Test singleton detection with stratum but no PSU."""
        data = pl.DataFrame(
            {
                "stratum": ["A", "A", "B", "B", "C"],
                "value": [10, 20, 30, 40, 50],
            }
        )
        design = Design(stratum="stratum")
        sample = Sample(data=data, design=design)

        assert sample is not None
        assert sample._data is not None

    def test_no_design_no_singletons(self):
        """Test that no singleton detection occurs without design."""
        data = pl.DataFrame({"value": [10, 20, 30, 40]})
        sample = Sample(data=data, design=None)

        assert not sample.singleton.exists
        assert sample.singleton.count == 0


class TestSingletonDetectionEdgeCases:
    """Edge case tests."""

    def test_empty_dataframe(self):
        """Test singleton detection with empty dataframe."""
        data = pl.DataFrame({"stratum": [], "psu": [], "value": []})
        design = Design(stratum="stratum", psu="psu")
        sample = Sample(data=data, design=design)

        assert sample is not None
        assert not sample.singleton.exists

    def test_null_values_in_stratum_or_psu(self):
        """Test handling of null values in stratum or PSU columns."""
        data = pl.DataFrame(
            {
                "stratum": ["A", "A", None, None],
                "psu": [1, 1, 2, 2],
                "value": [10, 20, 30, 40],
            }
        )
        design = Design(stratum="stratum", psu="psu")
        sample = Sample(data=data, design=design)

        assert sample is not None

    def test_all_unique_psus(self):
        """Test case where each stratum has only one PSU (all strata are singletons)."""
        data = pl.DataFrame(
            {
                "stratum": ["A", "A", "B", "B", "C", "C"],
                "psu": [1, 1, 2, 2, 3, 3],
                "value": [10, 20, 30, 40, 50, 60],
            }
        )
        design = Design(stratum="stratum", psu="psu")
        sample = Sample(data=data, design=design)

        assert sample.singleton.exists
        assert sample.singleton.count == 3


class TestSingletonConcept:
    """Educational tests to clarify singleton concept."""

    def test_psu_with_single_observation_not_singleton(self):
        """
        A PSU with one observation is NOT a singleton.
        A singleton is when a STRATUM has only ONE PSU.
        """
        data = pl.DataFrame(
            {
                "stratum": ["A", "A", "A", "A"],
                "psu": [1, 1, 2, 3],  # PSU 2 and 3 have 1 obs, but stratum A has 3 PSUs
                "value": [10, 20, 30, 40],
            }
        )
        design = Design(stratum="stratum", psu="psu")
        sample = Sample(data=data, design=design)

        assert not sample.singleton.exists

    def test_singleton_is_stratum_with_one_psu(self):
        """A singleton is when a stratum has only one PSU."""
        # NOT a singleton (stratum has multiple PSUs)
        data_not_singleton = pl.DataFrame(
            {
                "stratum": ["A", "A", "A"],
                "psu": [1, 2, 3],  # Each PSU has 1 obs, but stratum has 3 PSUs
                "value": [10, 20, 30],
            }
        )
        design1 = Design(stratum="stratum", psu="psu")
        sample1 = Sample(data=data_not_singleton, design=design1)
        assert not sample1.singleton.exists

        # IS a singleton (stratum has only one PSU)
        data_singleton = pl.DataFrame(
            {
                "stratum": ["A", "A", "A"],
                "psu": [1, 1, 1],  # All observations in same PSU
                "value": [10, 20, 30],
            }
        )
        design2 = Design(stratum="stratum", psu="psu")
        sample2 = Sample(data=data_singleton, design=design2)
        assert sample2.singleton.exists
        assert sample2.singleton.count == 1


class TestSingletonDetectionPerformance:
    """Performance-related tests."""

    def test_large_dataset_performance(self):
        """Test singleton detection on a larger dataset."""
        n_strata = 100
        n_psu_per_stratum = 50
        n_obs_per_psu = 10

        strata = []
        psus = []
        values = []

        for s in range(n_strata):
            for p in range(n_psu_per_stratum):
                for o in range(n_obs_per_psu):
                    strata.append(f"Stratum_{s}")
                    psus.append(f"PSU_{s}_{p}")
                    values.append(o)

        data = pl.DataFrame({"stratum": strata, "psu": psus, "value": values})
        design = Design(stratum="stratum", psu="psu")
        sample = Sample(data=data, design=design)

        assert not sample.singleton.exists


class TestSingletonDetectionRegressions:
    """Regression tests for known bugs."""

    def test_concatenation_bug_regression(self):
        """
        Regression test for the string concatenation bug.
        Previously, stratum="strat" + psu="psu" would become "stratpsu".
        """
        data = pl.DataFrame(
            {
                "my_stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "my_psu": [1, 1, 2, 2, 3, 3, 4, 4],
                "value": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )
        design = Design(stratum="my_stratum", psu="my_psu")
        sample = Sample(data=data, design=design)

        assert sample is not None
        assert not sample.singleton.exists

    def test_string_concat_bug_would_have_failed(self):
        """Test the exact scenario where the old buggy code would have failed."""
        data = pl.DataFrame(
            {
                "region": ["North", "North", "South", "South"],
                "cluster": [1, 1, 2, 2],
                "value": [100, 200, 300, 400],
            }
        )
        design = Design(stratum="region", psu="cluster")

        try:
            sample = Sample(data=data, design=design)
            assert sample is not None
        except Exception as e:
            if "ColumnNotFoundError" in str(type(e)) or "regioncluster" in str(e).lower():
                pytest.fail(f"Bug still exists! String concatenation failed: {e}")
            raise
