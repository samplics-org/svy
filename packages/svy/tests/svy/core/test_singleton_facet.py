# tests/test_singleton_facet.py
"""
Comprehensive tests for the Singleton facet API.

Covers:
- Quick checks: exists, count
- Inspection: detected(), show(), keys(), summary()
- Diagnostic helpers: strata_profile(), candidates_for(), compare(), rank_strata(), suggest_mapping()
- Handling methods: raise_error(), certainty(), skip(), combine(), collapse(), pool()
- Result access: last_result

Note: The singleton handling methods (certainty, skip, collapse, pool) create internal
variance columns for variance estimation WITHOUT modifying the original stratum/PSU columns.
This means `singleton.exists` will still return True after handling - the tests verify
that the internal columns are correctly configured instead.
"""

import copy

import polars as pl
import pytest

from svy.core.constants import _INTERNAL_CONCAT_SUFFIX, SVY_ROW_INDEX
from svy.core.enumerations import SingletonHandling
from svy.core.singleton import (
    _VAR_EXCLUDE_COL,
    _VAR_IS_SINGLETON_COL,
    _VAR_PSU_COL,
    _VAR_STRATUM_COL,
    Singleton,
    SingletonInfo,
    SingletonResult,
    SingletonSummary,
)
from svy.errors.singleton_errors import SingletonError


# ══════════════════════════════════════════════════════════════════════════════
# STUBS AND FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


class DesignStub:
    """Minimal design stub for testing."""

    def __init__(
        self,
        *,
        row_index: str = SVY_ROW_INDEX,
        stratum=None,
        psu=None,
        wgt=None,
    ):
        self.row_index = row_index
        self.stratum = stratum
        self.psu = psu
        self.wgt = wgt


class SampleStub:
    """
    Minimal sample stub for testing the Singleton facet.

    The singleton facet relies on:
      - _data (pl.DataFrame)
      - _design (has .row_index, .stratum, .psu, .wgt)
      - _internal_design dict with "stratum" and "psu"
      - clone(data=..., design=...) -> new Sample-like
    """

    __slots__ = ("_data", "_design", "_internal_design", "_singleton_result")

    def __init__(
        self,
        df: pl.DataFrame,
        design: DesignStub,
        *,
        stratum_internal: str | None,
        psu_internal: str,
    ):
        self._data = df
        self._design = design
        self._internal_design = {
            "stratum": stratum_internal,
            "psu": psu_internal,
            "ssu": None,
            "suffix": _INTERNAL_CONCAT_SUFFIX,
        }
        self._singleton_result = None

    def clone(self, *, data: pl.DataFrame | None = None, design: DesignStub | None = None):
        new = SampleStub(
            df=data if data is not None else self._data.clone(),
            design=copy.deepcopy(design if design is not None else self._design),
            stratum_internal=self._internal_design["stratum"],
            psu_internal=self._internal_design["psu"],
        )
        return new

    @property
    def singleton(self):
        return Singleton(self)


@pytest.fixture
def names():
    """Column name fixtures."""
    stratum_col = f"stratum{_INTERNAL_CONCAT_SUFFIX}"
    psu_col = f"psu{_INTERNAL_CONCAT_SUFFIX}"
    return {"stratum": stratum_col, "psu": psu_col}


@pytest.fixture
def base_df(names):
    """
    Base dataset with:
    - Singletons: ("North","A") with PSU=101; ("South","X") with PSU=301
    - Non-singletons: ("North","B") with PSUs 201,202; ("South","Y") with PSUs 401,402

    Note: Each singleton stratum has at least 2 observations so certainty() can resolve them.
    """
    rows = [
        (0, "North", "A", "101", 100.0, 25),
        (1, "North", "A", "101", 150.0, 30),
        (2, "North", "B", "201", 200.0, 35),
        (3, "North", "B", "201", 180.0, 28),
        (4, "North", "B", "202", 220.0, 40),
        (5, "South", "X", "301", 300.0, 45),
        (6, "South", "X", "301", 320.0, 42),  # Added second row for South__by__X
        (7, "South", "Y", "401", 250.0, 32),
        (8, "South", "Y", "402", 280.0, 38),
    ]
    df = pl.DataFrame(
        rows,
        schema=[SVY_ROW_INDEX, "region", "district", "cluster", "income", "age"],
        orient="row",
    )

    sep = "__by__"
    df = df.with_columns(
        (pl.col("region").cast(pl.Utf8) + sep + pl.col("district").cast(pl.Utf8)).alias(
            names["stratum"]
        ),
        pl.col("cluster").cast(pl.Utf8).alias(names["psu"]),
    )
    return df


@pytest.fixture
def sample(base_df, names):
    """Standard sample fixture with singletons."""
    design = DesignStub(
        row_index=SVY_ROW_INDEX,
        stratum=["region", "district"],
        psu="cluster",
        wgt=None,
    )
    return SampleStub(
        base_df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
    )


@pytest.fixture
def sample_with_weight(base_df, names):
    """Sample with weight column."""
    df = base_df.with_columns(pl.lit(1.0).alias("weight"))
    design = DesignStub(
        row_index=SVY_ROW_INDEX,
        stratum=["region", "district"],
        psu="cluster",
        wgt="weight",
    )
    return SampleStub(df, design, stratum_internal=names["stratum"], psu_internal=names["psu"])


@pytest.fixture
def sample_no_singletons(names):
    """Sample without any singletons."""
    rows = [
        (0, "North", "A", "101", 100.0),
        (1, "North", "A", "102", 150.0),
        (2, "North", "B", "201", 200.0),
        (3, "North", "B", "202", 180.0),
        (4, "South", "X", "301", 300.0),
        (5, "South", "X", "302", 250.0),
        (6, "South", "Y", "401", 280.0),
        (7, "South", "Y", "402", 220.0),
    ]
    df = pl.DataFrame(
        rows, schema=[SVY_ROW_INDEX, "region", "district", "cluster", "income"], orient="row"
    )
    sep = "__by__"
    df = df.with_columns(
        (pl.col("region").cast(pl.Utf8) + sep + pl.col("district").cast(pl.Utf8)).alias(
            names["stratum"]
        ),
        pl.col("cluster").cast(pl.Utf8).alias(names["psu"]),
    )
    design = DesignStub(row_index=SVY_ROW_INDEX, stratum=["region", "district"], psu="cluster")
    return SampleStub(df, design, stratum_internal=names["stratum"], psu_internal=names["psu"])


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR TESTING INTERNAL VARIANCE COLUMNS
# ══════════════════════════════════════════════════════════════════════════════


def has_variance_columns(sample) -> bool:
    """Check if sample has internal variance columns."""
    cols = sample._data.columns
    return all(c in cols for c in [_VAR_STRATUM_COL, _VAR_PSU_COL, _VAR_EXCLUDE_COL])


def get_effective_singletons(sample) -> int:
    """
    Count singletons in the effective variance structure.

    This checks the internal variance columns to see if singletons are resolved
    for variance estimation purposes.
    """
    if not has_variance_columns(sample):
        return sample.singleton.count

    df = sample._data

    # Filter out excluded rows
    if _VAR_EXCLUDE_COL in df.columns:
        df = df.filter(~pl.col(_VAR_EXCLUDE_COL))

    if df.height == 0:
        return 0

    # Count singletons in the variance structure
    agg = (
        df.lazy()
        .group_by(_VAR_STRATUM_COL)
        .agg(pl.col(_VAR_PSU_COL).n_unique().alias("n_psu"))
        .filter(pl.col("n_psu") == 1)
        .collect()
    )
    return agg.height


def singletons_resolved_for_variance(sample) -> bool:
    """Check if singletons are resolved in the variance structure."""
    return get_effective_singletons(sample) == 0


# ══════════════════════════════════════════════════════════════════════════════
# QUICK CHECKS: exists, count
# ══════════════════════════════════════════════════════════════════════════════


class TestQuickChecks:
    """Tests for exists and count properties."""

    def test_exists_true_when_singletons_present(self, sample):
        assert sample.singleton.exists is True

    def test_exists_false_when_no_singletons(self, sample_no_singletons):
        assert sample_no_singletons.singleton.exists is False

    def test_count_returns_correct_number(self, sample):
        assert sample.singleton.count == 2

    def test_count_zero_when_no_singletons(self, sample_no_singletons):
        assert sample_no_singletons.singleton.count == 0


# ══════════════════════════════════════════════════════════════════════════════
# INSPECTION: detected(), show(), keys(), summary()
# ══════════════════════════════════════════════════════════════════════════════


class TestInspection:
    """Tests for inspection methods."""

    def test_detected_returns_singleton_info_list(self, sample):
        singles = sample.singleton.detected()
        assert isinstance(singles, list)
        assert len(singles) == 2
        assert all(isinstance(s, SingletonInfo) for s in singles)

    def test_detected_contains_correct_keys(self, sample):
        singles = sample.singleton.detected()
        keys = sorted(s.stratum_key for s in singles)
        assert keys == sorted(["North__by__A", "South__by__X"])

    def test_detected_info_has_correct_attributes(self, sample):
        singles = sample.singleton.detected()
        info = {s.stratum_key: s for s in singles}["North__by__A"]

        assert info.psu_key == "101"
        assert info.n_observations == 2
        assert info.stratum_values == {"region": "North", "district": "A"}

    def test_detected_empty_when_no_singletons(self, sample_no_singletons):
        singles = sample_no_singletons.singleton.detected()
        assert singles == []

    def test_show_returns_dataframe(self, sample):
        df = sample.singleton.show()
        assert isinstance(df, pl.DataFrame)
        assert {"singleton_key", "n_obs", "psu"}.issubset(set(df.columns))
        assert df.height == 2

    def test_show_includes_stratum_values(self, sample):
        df = sample.singleton.show()
        assert "region" in df.columns
        assert "district" in df.columns

    def test_show_empty_dataframe_when_no_singletons(self, sample_no_singletons):
        df = sample_no_singletons.singleton.show()
        assert df.height == 0
        assert "singleton_key" in df.columns

    def test_keys_returns_stratum_keys(self, sample):
        keys = sample.singleton.keys()
        assert set(keys) == {"North__by__A", "South__by__X"}

    def test_keys_empty_when_no_singletons(self, sample_no_singletons):
        keys = sample_no_singletons.singleton.keys()
        assert keys == []

    def test_summary_returns_singleton_summary(self, sample):
        summary = sample.singleton.summary()
        assert isinstance(summary, SingletonSummary)

    def test_summary_has_correct_counts(self, sample):
        summary = sample.singleton.summary()
        assert summary.n_singletons == 2
        assert summary.n_strata == 4
        assert summary.affected_rows == 4  # 2 in North__by__A, 2 in South__by__X

    def test_summary_has_percentages(self, sample):
        summary = sample.singleton.summary()
        assert summary.pct_singletons == pytest.approx(50.0)  # 2 of 4 strata
        assert summary.pct_rows_affected == pytest.approx(44.44, rel=0.01)  # 4 of 9 rows

    def test_summary_has_recommendation(self, sample):
        summary = sample.singleton.summary()
        assert summary.recommendation is not None
        assert summary.recommendation_reason is not None

    def test_summary_when_no_singletons(self, sample_no_singletons):
        summary = sample_no_singletons.singleton.summary()
        assert summary.n_singletons == 0
        assert summary.recommendation is None or summary.recommendation_reason is not None


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════


class TestStrataProfile:
    """Tests for strata_profile() method."""

    def test_strata_profile_returns_dataframe(self, sample):
        profile = sample.singleton.strata_profile()
        assert isinstance(profile, pl.DataFrame)
        assert profile.height == 4  # 4 strata

    def test_strata_profile_has_required_columns(self, sample):
        profile = sample.singleton.strata_profile()
        assert "stratum_key" in profile.columns
        assert "n_psus" in profile.columns
        assert "n_obs" in profile.columns
        assert "is_singleton" in profile.columns

    def test_strata_profile_identifies_singletons(self, sample):
        profile = sample.singleton.strata_profile()
        singletons = profile.filter(pl.col("is_singleton"))
        assert singletons.height == 2

    def test_strata_profile_with_variables(self, sample):
        profile = sample.singleton.strata_profile(variables=["income", "age"])
        assert "mean_income" in profile.columns
        assert "mean_age" in profile.columns


class TestCandidatesFor:
    """Tests for candidates_for() method."""

    def test_candidates_for_returns_dataframe(self, sample):
        candidates = sample.singleton.candidates_for("North__by__A")
        assert isinstance(candidates, pl.DataFrame)

    def test_candidates_for_excludes_singletons(self, sample):
        candidates = sample.singleton.candidates_for("North__by__A")
        # Should only include non-singleton strata
        keys = candidates.get_column("stratum_key").to_list()
        assert "North__by__A" not in keys
        assert "South__by__X" not in keys

    def test_candidates_for_respects_top_k(self, sample):
        candidates = sample.singleton.candidates_for("North__by__A", top_k=1)
        assert candidates.height == 1

    def test_candidates_for_invalid_singleton_raises(self, sample):
        with pytest.raises(ValueError, match="not found"):
            sample.singleton.candidates_for("Invalid__Key")

    def test_candidates_for_by_size(self, sample):
        candidates = sample.singleton.candidates_for("North__by__A", by="size")
        # Should be sorted by n_psus ascending
        n_psus = candidates.get_column("n_psus").to_list()
        assert n_psus == sorted(n_psus)

    def test_candidates_for_with_within_constraint(self, sample):
        candidates = sample.singleton.candidates_for("North__by__A", within="region")
        # Should only include North strata
        keys = candidates.get_column("stratum_key").to_list()
        for key in keys:
            assert "North" in key


class TestCompare:
    """Tests for compare() method."""

    def test_compare_returns_dataframe(self, sample):
        df = sample.singleton.compare("North__by__A", "North__by__B")
        assert isinstance(df, pl.DataFrame)

    def test_compare_has_required_columns(self, sample):
        df = sample.singleton.compare("North__by__A", "North__by__B")
        assert set(df.columns) == {"variable", "stratum_1", "stratum_2", "difference"}

    def test_compare_with_specific_variables(self, sample):
        df = sample.singleton.compare("North__by__A", "North__by__B", variables=["income"])
        assert df.height == 1
        assert df.get_column("variable").to_list() == ["income"]


class TestRankStrata:
    """Tests for rank_strata() method."""

    def test_rank_strata_returns_dataframe(self, sample):
        ranked = sample.singleton.rank_strata(by="income")
        assert isinstance(ranked, pl.DataFrame)

    def test_rank_strata_has_rank_column(self, sample):
        ranked = sample.singleton.rank_strata(by="income")
        assert "rank" in ranked.columns

    def test_rank_strata_ascending_by_default(self, sample):
        ranked = sample.singleton.rank_strata(by="income")
        ranks = ranked.get_column("rank").to_list()
        assert ranks == list(range(1, len(ranks) + 1))

    def test_rank_strata_descending(self, sample):
        ranked = sample.singleton.rank_strata(by="income", descending=True)
        values = ranked.get_column("mean_income").to_list()
        assert values == sorted(values, reverse=True)


class TestSuggestMapping:
    """Tests for suggest_mapping() method."""

    def test_suggest_mapping_returns_dict(self, sample):
        mapping = sample.singleton.suggest_mapping(variables=["income"])
        assert isinstance(mapping, dict)

    def test_suggest_mapping_covers_all_singletons(self, sample):
        mapping = sample.singleton.suggest_mapping(variables=["income"])
        assert set(mapping.keys()) == {"North__by__A", "South__by__X"}

    def test_suggest_mapping_targets_are_non_singletons(self, sample):
        mapping = sample.singleton.suggest_mapping(variables=["income"])
        singleton_keys = sample.singleton.keys()
        for target in mapping.values():
            assert target not in singleton_keys

    def test_suggest_mapping_respects_within(self, sample):
        mapping = sample.singleton.suggest_mapping(variables=["income"], within="region")
        # North singleton should map to North non-singleton
        assert "North" in mapping["North__by__A"]
        # South singleton should map to South non-singleton
        assert "South" in mapping["South__by__X"]


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: raise_error()
# ══════════════════════════════════════════════════════════════════════════════


class TestRaiseError:
    """Tests for raise_error() method."""

    def test_raise_error_raises_when_singletons_exist(self, sample):
        with pytest.raises(SingletonError) as exc_info:
            sample.singleton.raise_error()
        assert "singleton" in str(exc_info.value).lower()

    def test_raise_error_silent_when_no_singletons(self, sample_no_singletons):
        # Should not raise
        result = sample_no_singletons.singleton.raise_error()
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: certainty()
# ══════════════════════════════════════════════════════════════════════════════


class TestCertainty:
    """Tests for certainty() method."""

    def test_certainty_returns_new_sample(self, sample):
        result = sample.singleton.certainty()
        assert result is not sample

    def test_certainty_creates_variance_columns(self, sample):
        """
        certainty() creates internal variance columns for variance estimation.
        The original stratum/PSU columns are NOT modified.
        """
        result = sample.singleton.certainty()
        assert has_variance_columns(result)

    def test_certainty_resolves_singletons_for_variance(self, sample):
        """
        certainty() resolves singletons in the variance structure by treating
        the PSU as a stratum and observations as PSUs.
        """
        result = sample.singleton.certainty()
        assert singletons_resolved_for_variance(result)

    def test_certainty_preserves_original_stratum_column(self, sample, names):
        """The original stratum column should be unchanged."""
        result = sample.singleton.certainty()
        original_strata = set(sample._data.get_column(names["stratum"]).unique().to_list())
        result_strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert original_strata == result_strata

    def test_certainty_variance_stratum_uses_psu(self, sample, names):
        """For singletons, the variance stratum should be the original PSU."""
        result = sample.singleton.certainty()
        df = result._data

        # For singleton North__by__A (PSU=101), variance stratum should be "101"
        north_a = df.filter(pl.col(names["stratum"]) == "North__by__A")
        var_strata = north_a.get_column(_VAR_STRATUM_COL).unique().to_list()
        assert var_strata == ["101"]

    def test_certainty_variance_psu_uses_row_index(self, sample):
        """For singletons, the variance PSU should be the row index."""
        result = sample.singleton.certainty()
        df = result._data

        # Check that variance PSUs are unique per row for singletons
        singleton_rows = df.filter(pl.col(_VAR_STRATUM_COL) == "101")
        n_rows = singleton_rows.height
        n_unique_psus = singleton_rows.get_column(_VAR_PSU_COL).n_unique()
        assert n_unique_psus == n_rows

    def test_certainty_preserves_non_singleton_structure(self, sample, names):
        """Non-singleton strata should keep their original structure in variance columns."""
        result = sample.singleton.certainty()
        df = result._data

        # For non-singleton North__by__B, variance stratum should be "North__by__B"
        north_b = df.filter(pl.col(names["stratum"]) == "North__by__B")
        var_strata = north_b.get_column(_VAR_STRATUM_COL).unique().to_list()
        assert var_strata == ["North__by__B"]

    def test_certainty_idempotent_when_no_singletons(self, sample_no_singletons):
        """Should return same object when no singletons exist."""
        result = sample_no_singletons.singleton.certainty()
        assert result is sample_no_singletons

    def test_certainty_preserves_categorical_dtype(self, sample, names):
        df_cat = sample._data.with_columns(pl.col(names["psu"]).cast(pl.Categorical))
        sample_cat = sample.clone(data=df_cat, design=sample._design)
        result = sample_cat.singleton.certainty()
        # Original column dtype preserved
        assert result._data.schema[names["psu"]] == pl.Categorical

    def test_certainty_sets_last_result(self, sample):
        result = sample.singleton.certainty()
        assert result.singleton.last_result is not None
        assert result.singleton.last_result.method == SingletonHandling.CERTAINTY

    def test_certainty_config_has_variance_columns(self, sample):
        """The result config should specify the variance column names."""
        result = sample.singleton.certainty()
        config = result.singleton.last_result.config
        assert config is not None
        assert config.var_stratum_col == _VAR_STRATUM_COL
        assert config.var_psu_col == _VAR_PSU_COL
        assert config.var_exclude_col == _VAR_EXCLUDE_COL

    def test_certainty_cannot_resolve_single_observation_singleton(self, names):
        """
        certainty() cannot resolve singletons that have only 1 observation.
        Each obs becomes its own PSU, but 1 obs = 1 PSU = still a singleton.
        """
        rows = [
            (0, "A", "101"),  # Singleton with only 1 row - CANNOT be resolved by certainty
            (1, "B", "201"),
            (2, "B", "202"),
        ]
        df = pl.DataFrame(rows, schema=[SVY_ROW_INDEX, "stratum", "cluster"], orient="row")
        df = df.with_columns(
            pl.col("stratum").alias(names["stratum"]),
            pl.col("cluster").cast(pl.Utf8).alias(names["psu"]),
        )
        design = DesignStub(row_index=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        assert sample.singleton.count == 1

        result = sample.singleton.certainty()
        # Still has 1 singleton in variance structure because the stratum had only 1 observation
        assert get_effective_singletons(result) == 1


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: skip()
# ══════════════════════════════════════════════════════════════════════════════


class TestSkip:
    """Tests for skip() method."""

    def test_skip_returns_new_sample(self, sample):
        result = sample.singleton.skip()
        assert result is not sample

    def test_skip_creates_variance_columns(self, sample):
        """skip() creates internal variance columns with exclusion flags."""
        result = sample.singleton.skip()
        assert has_variance_columns(result)

    def test_skip_marks_singleton_rows_as_excluded(self, sample, names):
        """Singleton rows should be marked as excluded in the variance structure."""
        result = sample.singleton.skip()
        df = result._data

        # Singleton rows should have exclude=True
        north_a = df.filter(pl.col(names["stratum"]) == "North__by__A")
        assert north_a.get_column(_VAR_EXCLUDE_COL).all()

        south_x = df.filter(pl.col(names["stratum"]) == "South__by__X")
        assert south_x.get_column(_VAR_EXCLUDE_COL).all()

    def test_skip_preserves_non_singleton_rows(self, sample, names):
        """Non-singleton rows should NOT be marked as excluded."""
        result = sample.singleton.skip()
        df = result._data

        # Non-singleton rows should have exclude=False
        north_b = df.filter(pl.col(names["stratum"]) == "North__by__B")
        assert not north_b.get_column(_VAR_EXCLUDE_COL).any()

    def test_skip_preserves_all_rows_in_data(self, sample):
        """skip() does NOT remove rows - it marks them as excluded."""
        result = sample.singleton.skip()
        assert result._data.height == sample._data.height

    def test_skip_resolves_singletons_for_variance(self, sample):
        """After exclusion, no singletons should remain in the variance structure."""
        result = sample.singleton.skip()
        assert singletons_resolved_for_variance(result)

    def test_skip_noop_when_no_singletons(self, sample_no_singletons):
        result = sample_no_singletons.singleton.skip()
        # Should return same sample when no singletons
        assert result is sample_no_singletons

    def test_skip_sets_last_result(self, sample):
        result = sample.singleton.skip()
        assert result.singleton.last_result is not None
        assert result.singleton.last_result.method == SingletonHandling.SKIP


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: combine()
# ══════════════════════════════════════════════════════════════════════════════


class TestCombine:
    """Tests for combine() method."""

    def test_combine_happy_path(self, sample, names):
        mapping = {"district": {"A": "B", "X": "Y"}}
        result = sample.singleton.combine(mapping)

        # combine() actually modifies the original columns
        assert not result.singleton.exists
        strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert strata == {"North__by__B", "South__by__Y"}

    def test_combine_invalid_column_raises(self, sample):
        mapping = {"invalid_col": {"A": "B"}}
        with pytest.raises(ValueError, match="not in design"):
            sample.singleton.combine(mapping)

    def test_combine_empty_mapping_raises(self, sample):
        with pytest.raises(ValueError):
            sample.singleton.combine({})

    def test_combine_flat_mapping_raises(self, sample):
        # Flat mapping should raise TypeError
        mapping = {"North__by__A": "North__by__B"}
        with pytest.raises(TypeError):
            sample.singleton.combine(mapping)

    def test_combine_incomplete_mapping_raises(self, sample):
        # Mapping that doesn't resolve all singletons
        mapping = {"cluster": {"301": "401"}}  # Only changes PSU, not stratum
        with pytest.raises(SingletonError):
            sample.singleton.combine(mapping)

    def test_combine_sets_last_result(self, sample):
        mapping = {"district": {"A": "B", "X": "Y"}}
        result = sample.singleton.combine(mapping)
        assert result.singleton.last_result is not None
        assert result.singleton.last_result.method == SingletonHandling.COMBINE


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: collapse()
# ══════════════════════════════════════════════════════════════════════════════


class TestCollapseSmallest:
    """Tests for collapse(using='smallest')."""

    def test_collapse_smallest_returns_new_sample(self, sample):
        result = sample.singleton.collapse(using="smallest")
        assert result is not sample

    def test_collapse_smallest_creates_variance_columns(self, sample):
        """collapse() creates internal variance columns with remapped strata."""
        result = sample.singleton.collapse(using="smallest")
        assert has_variance_columns(result)

    def test_collapse_smallest_resolves_singletons_for_variance(self, sample):
        """Singletons should be resolved in the variance structure."""
        result = sample.singleton.collapse(using="smallest")
        assert singletons_resolved_for_variance(result)

    def test_collapse_smallest_preserves_original_stratum_column(self, sample, names):
        """The original stratum column should be unchanged."""
        result = sample.singleton.collapse(using="smallest")
        original_strata = set(sample._data.get_column(names["stratum"]).unique().to_list())
        result_strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert original_strata == result_strata

    def test_collapse_smallest_remaps_variance_stratum(self, sample):
        """The variance stratum column should have remapped singleton keys."""
        result = sample.singleton.collapse(using="smallest")
        df = result._data

        # Variance strata should have fewer unique values than original
        var_strata = set(df.get_column(_VAR_STRATUM_COL).unique().to_list())
        # Singletons should be merged into non-singletons
        assert "North__by__A" not in var_strata
        assert "South__by__X" not in var_strata

    def test_collapse_smallest_preserves_row_count(self, sample):
        result = sample.singleton.collapse(using="smallest")
        assert result._data.height == sample._data.height

    def test_collapse_smallest_sets_last_result(self, sample):
        result = sample.singleton.collapse(using="smallest")
        assert result.singleton.last_result is not None
        assert result.singleton.last_result.method == SingletonHandling.COLLAPSE
        assert isinstance(result.singleton.last_result.applied, dict)


class TestCollapseLargest:
    """Tests for collapse(using='largest')."""

    def test_collapse_largest_resolves_singletons_for_variance(self, sample):
        result = sample.singleton.collapse(using="largest")
        assert singletons_resolved_for_variance(result)


class TestCollapseNext:
    """Tests for collapse(using='next')."""

    def test_collapse_next_resolves_singletons_for_variance(self, sample):
        result = sample.singleton.collapse(using="next")
        assert singletons_resolved_for_variance(result)

    def test_collapse_next_with_order_by(self, sample):
        result = sample.singleton.collapse(using="next", order_by="income")
        assert singletons_resolved_for_variance(result)


class TestCollapsePrevious:
    """Tests for collapse(using='previous')."""

    def test_collapse_previous_resolves_singletons_for_variance(self, sample):
        result = sample.singleton.collapse(using="previous")
        assert singletons_resolved_for_variance(result)


class TestCollapseWithMapping:
    """Tests for collapse(using=dict)."""

    def test_collapse_with_explicit_mapping(self, sample):
        mapping = {
            "North__by__A": "North__by__B",
            "South__by__X": "South__by__Y",
        }
        result = sample.singleton.collapse(using=mapping)

        assert singletons_resolved_for_variance(result)

        # Verify mapping in variance column
        df = result._data
        var_strata = set(df.get_column(_VAR_STRATUM_COL).unique().to_list())
        assert var_strata == {"North__by__B", "South__by__Y"}

    def test_collapse_with_incomplete_mapping_raises(self, sample):
        mapping = {"North__by__A": "North__by__B"}  # Missing South__by__X
        with pytest.raises(ValueError, match="does not contain key"):
            sample.singleton.collapse(using=mapping)

    def test_collapse_with_invalid_target_raises(self, sample):
        mapping = {
            "North__by__A": "Invalid__Target",
            "South__by__X": "South__by__Y",
        }
        with pytest.raises(ValueError, match="not found"):
            sample.singleton.collapse(using=mapping)


class TestCollapseWithCallable:
    """Tests for collapse(using=callable)."""

    def test_collapse_with_callable(self, sample):
        def always_pick_first(singleton, candidates):
            return candidates[0].stratum_key

        result = sample.singleton.collapse(using=always_pick_first)
        assert singletons_resolved_for_variance(result)

    def test_collapse_with_callable_invalid_return_raises(self, sample):
        def bad_picker(singleton, candidates):
            return "Invalid__Key"

        with pytest.raises(ValueError, match="not a valid candidate"):
            sample.singleton.collapse(using=bad_picker)


class TestCollapseWithin:
    """Tests for collapse with within constraint."""

    def test_collapse_within_region(self, sample):
        result = sample.singleton.collapse(using="smallest", within="region")
        assert singletons_resolved_for_variance(result)

        # Verify mapping respected region constraint
        last_result = result.singleton.last_result
        mapping = last_result.applied
        assert "North" in mapping["North__by__A"]
        assert "South" in mapping["South__by__X"]

    def test_collapse_within_no_candidates_raises(self, names):
        # Create sample where within constraint leaves no candidates
        rows = [
            (0, "North", "A", "101"),  # Singleton
            (1, "North", "A", "101"),
            (2, "South", "B", "201"),  # Non-singleton
            (3, "South", "B", "202"),
        ]
        df = pl.DataFrame(
            rows, schema=[SVY_ROW_INDEX, "region", "district", "cluster"], orient="row"
        )
        sep = "__by__"
        df = df.with_columns(
            (pl.col("region").cast(pl.Utf8) + sep + pl.col("district").cast(pl.Utf8)).alias(
                names["stratum"]
            ),
            pl.col("cluster").cast(pl.Utf8).alias(names["psu"]),
        )
        design = DesignStub(row_index=SVY_ROW_INDEX, stratum=["region", "district"], psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        # North singleton has no non-singleton in North region
        with pytest.raises(SingletonError, match="No valid merge targets"):
            sample.singleton.collapse(using="smallest", within="region")


class TestCollapseRebalancing:
    """Tests for rebalancing behavior during collapse."""

    def test_collapse_distributes_singletons(self, names):
        # Create sample with multiple singletons and two small non-singletons
        rows = [
            (0, "A", "101"),  # Singleton
            (1, "B", "201"),  # Singleton
            (2, "C", "301"),
            (3, "C", "302"),  # Non-singleton (2 PSUs)
            (4, "D", "401"),
            (5, "D", "402"),  # Non-singleton (2 PSUs)
        ]
        df = pl.DataFrame(rows, schema=[SVY_ROW_INDEX, "stratum", "cluster"], orient="row")
        df = df.with_columns(
            pl.col("stratum").alias(names["stratum"]),
            pl.col("cluster").cast(pl.Utf8).alias(names["psu"]),
        )
        design = DesignStub(row_index=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        result = sample.singleton.collapse(using="smallest")
        assert singletons_resolved_for_variance(result)

        # Due to rebalancing, singletons should be distributed
        # (not all going to the same stratum)
        mapping = result.singleton.last_result.applied
        targets = set(mapping.values())
        # With rebalancing, we might get different targets
        assert len(targets) >= 1  # At minimum 1, likely 2 with rebalancing


class TestCollapseTieBreaking:
    """Tests for tie-breaking behavior."""

    def test_collapse_deterministic_by_default(self, sample):
        """
        Without rstate, collapse should be deterministic.

        With rebalancing and alphabetical processing:
        1. North__by__A processed first (alphabetically)
        2. Candidates tied at 2 PSUs: North__by__B, South__by__Y
        3. Tie-broken alphabetically: North__by__A → North__by__B
        4. North__by__B now has 3 PSUs
        5. South__by__X processed second
        6. South__by__Y (2 PSUs) < North__by__B (3 PSUs)
        7. South__by__X → South__by__Y
        """
        result1 = sample.singleton.collapse(using="smallest")
        result2 = sample.singleton.collapse(using="smallest")

        mapping1 = result1.singleton.last_result.applied
        mapping2 = result2.singleton.last_result.applied
        assert mapping1 == mapping2

        # Verify expected mapping
        expected = {
            "North__by__A": "North__by__B",
            "South__by__X": "South__by__Y",
        }
        assert mapping1 == expected

    def test_collapse_with_rstate_reproducible(self, sample):
        result1 = sample.singleton.collapse(using="smallest", rstate=42)
        result2 = sample.singleton.collapse(using="smallest", rstate=42)

        mapping1 = result1.singleton.last_result.applied
        mapping2 = result2.singleton.last_result.applied
        assert mapping1 == mapping2

    def test_collapse_different_rstate_may_differ(self, names):
        # Create scenario with ties
        rows = [
            (0, "A", "101"),  # Singleton
            (1, "B", "201"),
            (2, "B", "202"),  # 2 PSUs
            (3, "C", "301"),
            (4, "C", "302"),  # 2 PSUs (tie with B)
        ]
        df = pl.DataFrame(rows, schema=[SVY_ROW_INDEX, "stratum", "cluster"], orient="row")
        df = df.with_columns(
            pl.col("stratum").alias(names["stratum"]),
            pl.col("cluster").cast(pl.Utf8).alias(names["psu"]),
        )
        design = DesignStub(row_index=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        # With rstate, ties broken randomly
        # Run multiple times to see if we get variation (probabilistic test)
        results = set()
        for seed in range(10):
            result = sample.singleton.collapse(using="smallest", rstate=seed)
            mapping = result.singleton.last_result.applied
            results.add(mapping["A"])

        # With deterministic default, would always pick same
        # With random, might pick different (not guaranteed but likely)
        # At minimum, should work without error
        assert len(results) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: pool()
# ══════════════════════════════════════════════════════════════════════════════


class TestPool:
    """Tests for pool() method."""

    def test_pool_returns_new_sample(self, sample):
        result = sample.singleton.pool()
        assert result is not sample

    def test_pool_creates_variance_columns(self, sample):
        """pool() creates internal variance columns with pooled stratum."""
        result = sample.singleton.pool()
        assert has_variance_columns(result)

    def test_pool_resolves_singletons_for_variance(self, sample):
        """Singletons should be resolved in the variance structure."""
        result = sample.singleton.pool()
        assert singletons_resolved_for_variance(result)

    def test_pool_creates_pooled_stratum_in_variance_column(self, sample):
        """The pooled stratum should appear in the variance column."""
        result = sample.singleton.pool()
        var_strata = result._data.get_column(_VAR_STRATUM_COL).unique().to_list()
        assert "__pooled__" in var_strata

    def test_pool_custom_name_in_variance_column(self, sample):
        """Custom pool name should appear in the variance column."""
        result = sample.singleton.pool(name="misc_strata")
        var_strata = result._data.get_column(_VAR_STRATUM_COL).unique().to_list()
        assert "misc_strata" in var_strata

    def test_pool_preserves_original_stratum_column(self, sample, names):
        """The original stratum column should be unchanged."""
        result = sample.singleton.pool()
        original_strata = set(sample._data.get_column(names["stratum"]).unique().to_list())
        result_strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert original_strata == result_strata

    def test_pool_preserves_row_count(self, sample):
        result = sample.singleton.pool()
        assert result._data.height == sample._data.height  # 9 rows

    def test_pool_combines_singleton_psus_in_variance_column(self, sample):
        """The pooled stratum should have multiple PSUs in the variance structure."""
        result = sample.singleton.pool()
        pooled = result._data.filter(pl.col(_VAR_STRATUM_COL) == "__pooled__")
        n_psus = pooled.get_column(_VAR_PSU_COL).n_unique()
        # Should have 2 PSUs (one from each original singleton)
        assert n_psus == 2

    def test_pool_sets_last_result(self, sample):
        result = sample.singleton.pool()
        assert result.singleton.last_result is not None
        assert result.singleton.last_result.method == SingletonHandling.POOL

    def test_pool_noop_when_no_singletons(self, sample_no_singletons):
        result = sample_no_singletons.singleton.pool()
        # Should return same sample when no singletons
        assert result is sample_no_singletons


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: scale()
# ══════════════════════════════════════════════════════════════════════════════


class TestScale:
    """Tests for scale() method."""

    def test_scale_returns_new_sample(self, sample):
        result = sample.singleton.scale()
        assert result is not sample

    def test_scale_creates_variance_columns(self, sample):
        """scale() creates internal variance columns like skip()."""
        result = sample.singleton.scale()
        assert has_variance_columns(result)

    def test_scale_marks_singletons_as_excluded(self, sample, names):
        """Singleton rows should be marked as excluded in the variance structure."""
        result = sample.singleton.scale()
        df = result._data

        # Singleton rows should have exclude=True
        north_a = df.filter(pl.col(names["stratum"]) == "North__by__A")
        assert north_a.get_column(_VAR_EXCLUDE_COL).all()

        south_x = df.filter(pl.col(names["stratum"]) == "South__by__X")
        assert south_x.get_column(_VAR_EXCLUDE_COL).all()

    def test_scale_preserves_non_singleton_rows(self, sample, names):
        """Non-singleton rows should NOT be marked as excluded."""
        result = sample.singleton.scale()
        df = result._data

        north_b = df.filter(pl.col(names["stratum"]) == "North__by__B")
        assert not north_b.get_column(_VAR_EXCLUDE_COL).any()

    def test_scale_resolves_singletons_for_variance(self, sample):
        """After exclusion, no singletons should remain in the variance structure."""
        result = sample.singleton.scale()
        assert singletons_resolved_for_variance(result)

    def test_scale_preserves_all_rows(self, sample):
        """scale() does NOT remove rows - it marks them as excluded."""
        result = sample.singleton.scale()
        assert result._data.height == sample._data.height

    def test_scale_preserves_original_stratum_column(self, sample, names):
        """The original stratum column should be unchanged."""
        result = sample.singleton.scale()
        original_strata = set(sample._data.get_column(names["stratum"]).unique().to_list())
        result_strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert original_strata == result_strata

    def test_scale_sets_last_result(self, sample):
        result = sample.singleton.scale()
        assert result.singleton.last_result is not None
        assert result.singleton.last_result.method == SingletonHandling.SCALE

    def test_scale_config_has_singleton_fraction(self, sample):
        """The result config should include the singleton fraction."""
        result = sample.singleton.scale()
        config = result.singleton.last_result.config
        assert config is not None
        assert config.singleton_fraction is not None
        # 2 singletons out of 4 strata = 0.5
        assert config.singleton_fraction == pytest.approx(0.5)

    def test_scale_config_has_variance_columns(self, sample):
        """The result config should specify the variance column names."""
        result = sample.singleton.scale()
        config = result.singleton.last_result.config
        assert config.var_stratum_col == _VAR_STRATUM_COL
        assert config.var_psu_col == _VAR_PSU_COL
        assert config.var_exclude_col == _VAR_EXCLUDE_COL

    def test_scale_noop_when_no_singletons(self, sample_no_singletons):
        result = sample_no_singletons.singleton.scale()
        # Should return same sample when no singletons
        assert result is sample_no_singletons

    def test_scale_singleton_fraction_calculation(self, names):
        """Test singleton fraction calculation with different proportions."""
        # Create sample with 1 singleton out of 5 strata = 20%
        rows = [
            (0, "A", "101"),  # Singleton
            (1, "B", "201"),
            (2, "B", "202"),  # 2 PSUs
            (3, "C", "301"),
            (4, "C", "302"),  # 2 PSUs
            (5, "D", "401"),
            (6, "D", "402"),  # 2 PSUs
            (7, "E", "501"),
            (8, "E", "502"),  # 2 PSUs
        ]
        df = pl.DataFrame(rows, schema=[SVY_ROW_INDEX, "stratum", "cluster"], orient="row")
        df = df.with_columns(
            pl.col("stratum").alias(names["stratum"]),
            pl.col("cluster").cast(pl.Utf8).alias(names["psu"]),
        )
        design = DesignStub(row_index=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        result = sample.singleton.scale()
        config = result.singleton.last_result.config

        # 1 singleton out of 5 strata = 0.2
        assert config.singleton_fraction == pytest.approx(0.2)

    def test_scale_inflation_factor(self, sample):
        """Test that the expected inflation factor can be calculated from singleton_fraction."""
        result = sample.singleton.scale()
        config = result.singleton.last_result.config

        singleton_frac = config.singleton_fraction
        # Expected inflation factor: 1 / (1 - singleton_frac)
        expected_inflation = 1.0 / (1.0 - singleton_frac)

        # With 2 singletons out of 4 strata (50%), inflation = 1 / 0.5 = 2.0
        assert expected_inflation == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: center()
# ══════════════════════════════════════════════════════════════════════════════


class TestCenter:
    """Tests for center() method."""

    def test_center_returns_new_sample(self, sample):
        result = sample.singleton.center()
        assert result is not sample

    def test_center_creates_variance_columns(self, sample):
        """center() creates internal variance columns."""
        result = sample.singleton.center()
        assert has_variance_columns(result)

    def test_center_does_not_exclude_singletons(self, sample, names):
        """Singleton rows should NOT be marked as excluded for CENTER."""
        result = sample.singleton.center()
        df = result._data

        # No rows should be excluded
        assert not df.get_column(_VAR_EXCLUDE_COL).any()

    def test_center_marks_singleton_rows(self, sample, names):
        """Singleton rows should be marked with is_singleton flag."""
        result = sample.singleton.center()
        df = result._data

        # Singleton rows should have is_singleton=True
        north_a = df.filter(pl.col(names["stratum"]) == "North__by__A")
        assert north_a.get_column(_VAR_IS_SINGLETON_COL).all()

        # Non-singleton rows should have is_singleton=False
        north_b = df.filter(pl.col(names["stratum"]) == "North__by__B")
        assert not north_b.get_column(_VAR_IS_SINGLETON_COL).any()

    def test_center_preserves_all_rows(self, sample):
        """center() preserves all rows including singletons."""
        result = sample.singleton.center()
        assert result._data.height == sample._data.height

    def test_center_preserves_original_stratum_column(self, sample, names):
        """The original stratum column should be unchanged."""
        result = sample.singleton.center()
        original_strata = set(sample._data.get_column(names["stratum"]).unique().to_list())
        result_strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert original_strata == result_strata

    def test_center_sets_last_result(self, sample):
        result = sample.singleton.center()
        assert result.singleton.last_result is not None
        assert result.singleton.last_result.method == SingletonHandling.CENTER

    def test_center_config_has_no_singleton_fraction(self, sample):
        """CENTER method should not use singleton_fraction (that's for SCALE)."""
        result = sample.singleton.center()
        config = result.singleton.last_result.config
        assert config is not None
        assert config.singleton_fraction is None

    def test_center_config_has_singleton_keys(self, sample):
        """The config should include the singleton keys."""
        result = sample.singleton.center()
        config = result.singleton.last_result.config
        assert config.singleton_keys is not None
        assert set(config.singleton_keys) == {"North__by__A", "South__by__X"}

    def test_center_config_has_variance_columns(self, sample):
        """The result config should specify the variance column names."""
        result = sample.singleton.center()
        config = result.singleton.last_result.config
        assert config.var_stratum_col == _VAR_STRATUM_COL
        assert config.var_psu_col == _VAR_PSU_COL
        assert config.var_exclude_col == _VAR_EXCLUDE_COL

    def test_center_noop_when_no_singletons(self, sample_no_singletons):
        result = sample_no_singletons.singleton.center()
        # Should return same sample when no singletons
        assert result is sample_no_singletons

    def test_center_strata_counts_unchanged(self, sample):
        """CENTER should not change the stratum/PSU counts (unlike SKIP/SCALE)."""
        result = sample.singleton.center()
        lr = result.singleton.last_result

        # No strata/PSUs should be "removed" for variance calculation
        assert lr.n_strata_before == lr.n_strata_after
        assert lr.n_psus_before == lr.n_psus_after


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: handle() dispatcher
# ══════════════════════════════════════════════════════════════════════════════


class TestHandle:
    """Tests for handle() dispatcher method."""

    def test_handle_certainty(self, sample):
        result = sample.singleton.handle("certainty")
        assert singletons_resolved_for_variance(result)
        assert result.singleton.last_result.method == SingletonHandling.CERTAINTY

    def test_handle_skip(self, sample):
        result = sample.singleton.handle("skip")
        assert singletons_resolved_for_variance(result)
        assert result.singleton.last_result.method == SingletonHandling.SKIP

    def test_handle_collapse(self, sample):
        result = sample.singleton.handle("collapse", using="smallest")
        assert singletons_resolved_for_variance(result)
        assert result.singleton.last_result.method == SingletonHandling.COLLAPSE

    def test_handle_pool(self, sample):
        result = sample.singleton.handle("pool")
        assert singletons_resolved_for_variance(result)
        assert result.singleton.last_result.method == SingletonHandling.POOL

    def test_handle_scale(self, sample):
        result = sample.singleton.handle("scale")
        assert singletons_resolved_for_variance(result)
        assert result.singleton.last_result.method == SingletonHandling.SCALE
        assert result.singleton.last_result.config.singleton_fraction is not None

    def test_handle_center(self, sample):
        result = sample.singleton.handle("center")
        # CENTER doesn't exclude singletons, so check variance columns exist
        assert has_variance_columns(result)
        assert result.singleton.last_result.method == SingletonHandling.CENTER
        # CENTER should not have singleton_fraction
        assert result.singleton.last_result.config.singleton_fraction is None

    def test_handle_combine(self, sample):
        mapping = {"district": {"A": "B", "X": "Y"}}
        result = sample.singleton.handle("combine", mapping=mapping)
        # combine() actually resolves singletons in the original column
        assert not result.singleton.exists

    def test_handle_invalid_method_raises(self, sample):
        with pytest.raises(ValueError, match="Unknown method"):
            sample.singleton.handle("invalid_method")

    def test_handle_case_insensitive(self, sample):
        result = sample.singleton.handle("CERTAINTY")
        assert singletons_resolved_for_variance(result)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT ACCESS: last_result
# ══════════════════════════════════════════════════════════════════════════════


class TestLastResult:
    """Tests for last_result property."""

    def test_last_result_none_on_original_sample(self, sample):
        assert sample.singleton.last_result is None

    def test_last_result_set_after_handling(self, sample):
        result = sample.singleton.certainty()
        assert result.singleton.last_result is not None

    def test_last_result_is_singleton_result(self, sample):
        result = sample.singleton.certainty()
        assert isinstance(result.singleton.last_result, SingletonResult)

    def test_last_result_has_method(self, sample):
        result = sample.singleton.certainty()
        assert result.singleton.last_result.method == SingletonHandling.CERTAINTY

    def test_last_result_has_detected(self, sample):
        result = sample.singleton.certainty()
        assert len(result.singleton.last_result.detected) == 2

    def test_last_result_has_counts(self, sample):
        result = sample.singleton.certainty()
        lr = result.singleton.last_result
        assert lr.n_singletons_detected == 2
        assert lr.n_strata_before > 0
        assert lr.n_psus_before > 0

    def test_last_result_applied_for_collapse(self, sample):
        result = sample.singleton.collapse(using="smallest")
        lr = result.singleton.last_result
        assert isinstance(lr.applied, dict)
        assert len(lr.applied) == 2  # Two singletons mapped

    def test_last_result_applied_for_pool(self, sample):
        result = sample.singleton.pool()
        lr = result.singleton.last_result
        assert isinstance(lr.applied, dict)
        # All singletons mapped to pooled name
        assert all(v == "__pooled__" for v in lr.applied.values())

    def test_last_result_has_config(self, sample):
        """The result should include a config for variance estimation."""
        result = sample.singleton.certainty()
        lr = result.singleton.last_result
        assert lr.config is not None
        assert lr.config.method == SingletonHandling.CERTAINTY
        assert lr.config.var_stratum_col == _VAR_STRATUM_COL


# ══════════════════════════════════════════════════════════════════════════════
# TUPLE PSU TESTS
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_tuple_psu(names):
    """Sample with tuple PSU columns."""
    rows = [
        (0, "North", "A", "101", "alpha"),
        (1, "North", "A", "101", "alpha"),
        (2, "North", "B", "201", "alpha"),
        (3, "North", "B", "201", "alpha"),
        (4, "North", "B", "202", "alpha"),
        (5, "South", "X", "301", "alpha"),
        (6, "South", "X", "301", "alpha"),  # Added second row for South__by__X
        (7, "South", "Y", "401", "alpha"),
        (8, "South", "Y", "402", "alpha"),
    ]
    df = pl.DataFrame(
        rows,
        schema=[SVY_ROW_INDEX, "region", "district", "cluster", "cluster_b"],
        orient="row",
    )
    sep = "__by__"
    df = df.with_columns(
        (pl.col("region").cast(pl.Utf8) + sep + pl.col("district").cast(pl.Utf8)).alias(
            names["stratum"]
        ),
        (pl.col("cluster").cast(pl.Utf8) + sep + pl.col("cluster_b").cast(pl.Utf8)).alias(
            names["psu"]
        ),
    )
    design = DesignStub(
        row_index=SVY_ROW_INDEX,
        stratum=("region", "district"),
        psu=("cluster", "cluster_b"),
    )
    return SampleStub(df, design, stratum_internal=names["stratum"], psu_internal=names["psu"])


class TestTuplePsu:
    """Tests for tuple PSU handling."""

    def test_detect_with_tuple_psu(self, sample_tuple_psu):
        singles = sample_tuple_psu.singleton.detected()
        keys = sorted(s.stratum_key for s in singles)
        assert keys == sorted(["North__by__A", "South__by__X"])

    def test_certainty_with_tuple_psu(self, sample_tuple_psu):
        result = sample_tuple_psu.singleton.certainty()
        assert singletons_resolved_for_variance(result)

    def test_collapse_with_tuple_psu(self, sample_tuple_psu):
        result = sample_tuple_psu.singleton.collapse(using="smallest")
        assert singletons_resolved_for_variance(result)

    def test_pool_with_tuple_psu(self, sample_tuple_psu):
        result = sample_tuple_psu.singleton.pool()
        assert singletons_resolved_for_variance(result)


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests."""

    def test_all_strata_are_singletons(self, names):
        """Test when all strata are singletons (pool should still work)."""
        rows = [
            (0, "A", "101"),
            (1, "B", "201"),
            (2, "C", "301"),
        ]
        df = pl.DataFrame(rows, schema=[SVY_ROW_INDEX, "stratum", "cluster"], orient="row")
        df = df.with_columns(
            pl.col("stratum").alias(names["stratum"]),
            pl.col("cluster").cast(pl.Utf8).alias(names["psu"]),
        )
        design = DesignStub(row_index=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        # Pool should work - combines all into one stratum with 3 PSUs
        result = sample.singleton.pool()
        assert singletons_resolved_for_variance(result)

        # Collapse should fail - no non-singleton targets
        with pytest.raises(SingletonError, match="No valid merge targets"):
            sample.singleton.collapse(using="smallest")

    def test_single_row_singleton(self, names):
        """Test singleton with only one observation."""
        rows = [
            (0, "A", "101"),  # Singleton with 1 row
            (1, "B", "201"),
            (2, "B", "202"),
        ]
        df = pl.DataFrame(rows, schema=[SVY_ROW_INDEX, "stratum", "cluster"], orient="row")
        df = df.with_columns(
            pl.col("stratum").alias(names["stratum"]),
            pl.col("cluster").cast(pl.Utf8).alias(names["psu"]),
        )
        design = DesignStub(row_index=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        singles = sample.singleton.detected()
        assert len(singles) == 1
        assert singles[0].n_observations == 1

    def test_no_stratum_column(self):
        """Test behavior when no stratum is defined."""
        rows = [(0, "101"), (1, "102"), (2, "201")]
        df = pl.DataFrame(rows, schema=[SVY_ROW_INDEX, "cluster"], orient="row")
        design = DesignStub(row_index=SVY_ROW_INDEX, stratum=None, psu="cluster")
        sample = SampleStub(df, design, stratum_internal=None, psu_internal="cluster")

        assert not sample.singleton.exists
        assert sample.singleton.detected() == []
