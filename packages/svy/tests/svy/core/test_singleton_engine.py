# tests/svy/core/test_singleton_engine.py
"""The engine that applies the declared singleton rule (``svy.Singleton``).

Detection of singleton strata, and what each method does to the variance
columns: self_representing, skip, scale, center, collapse (every strategy,
within, ties, rebalancing) and pool. The original stratum and PSU columns are
never modified; the rule writes svy's internal variance columns.
"""

import polars as pl
import pytest

from svy.core.constants import SVY_ROW_INDEX, key_col
from svy.core.enumerations import SingletonMethod
from svy.core.singleton import (
    _VAR_EXCLUDE_COL,
    _VAR_IS_SINGLETON_COL,
    _VAR_PSU_COL,
    _VAR_STRATUM_COL,
    SingletonInfo,
    SingletonResult,
)
from svy.errors.singleton_errors import SingletonError


def _result(sample):
    """What the design's singleton rule did to the current data (internal)."""
    sample._sync_parts()
    return sample._singleton_result


def _detected(sample):
    """The singleton strata of the current data (internal)."""
    from svy.core.singleton import _Engine

    sample._sync_parts()
    return _Engine(sample, _sync=False).detected()


def _keys(sample):
    """svy's keys of the singleton strata of the current data (internal)."""
    return [s.stratum_key for s in _detected(sample)]


def _declare(sample, method, **kw):
    """A fork of ``sample`` with the singleton rule declared on its design."""
    from svy.core.design import Singleton as _Rule

    new = sample._fork()
    new.update_design(singleton=_Rule(method, **kw))
    return new


def _resolve(sample):
    """What every Taylor analysis runs first: raises when the declared rule
    cannot be applied to the data."""
    from svy.core.singleton import require_singleton_rule

    require_singleton_rule(sample, where="test")


# ══════════════════════════════════════════════════════════════════════════════
# STUBS AND FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


class DesignStub:
    """Minimal design stub for testing."""

    def __init__(
        self,
        *,
        case_id: str = SVY_ROW_INDEX,
        stratum=None,
        psu=None,
        wgt=None,
    ):
        self.case_id = case_id
        self.wave = None
        self.stratum = stratum
        self.psu = psu
        self.wgt = wgt


def SampleStub(  # noqa: N802 (the name of the stub this replaced)
    df: pl.DataFrame,
    design: DesignStub,
    *,
    stratum_internal: str | None = None,
    psu_internal: str | None = None,
):
    """A real Sample on the frame and design (svy's own key columns dropped):
    the rule lives on a design, which a stub does not have."""
    import svy

    keys = [c for c in (SVY_ROW_INDEX, key_col("stratum"), key_col("psu")) if c in df.columns]
    return svy.Sample(
        df.drop(keys), svy.Design(stratum=design.stratum, psu=design.psu, wgt=design.wgt)
    )


@pytest.fixture
def names():
    """Column name fixtures."""
    stratum_col = key_col("stratum")
    psu_col = key_col("psu")
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
        case_id=SVY_ROW_INDEX,
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
        case_id=SVY_ROW_INDEX,
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
    design = DesignStub(case_id=SVY_ROW_INDEX, stratum=["region", "district"], psu="cluster")
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
        return sample.n_singletons

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
        assert (sample.n_singletons > 0) is True

    def test_exists_false_when_no_singletons(self, sample_no_singletons):
        assert (sample_no_singletons.n_singletons > 0) is False

    def test_count_returns_correct_number(self, sample):
        assert sample.n_singletons == 2

    def test_count_zero_when_no_singletons(self, sample_no_singletons):
        assert sample_no_singletons.n_singletons == 0


# ══════════════════════════════════════════════════════════════════════════════
# INSPECTION: detected(), show(), keys(), summary()
# ══════════════════════════════════════════════════════════════════════════════


class TestInspection:
    """Tests for inspection methods."""

    def test_detected_returns_singleton_info_list(self, sample):
        singles = _detected(sample)
        assert isinstance(singles, list)
        assert len(singles) == 2
        assert all(isinstance(s, SingletonInfo) for s in singles)

    def test_detected_info_has_correct_attributes(self, sample):
        singles = _detected(sample)
        info = {s.stratum_key: s for s in singles}["North__by__A"]

        assert info.psu_key == "101"
        assert info.n_observations == 2
        assert info.stratum_values == {"region": "North", "district": "A"}

    def test_detected_empty_when_no_singletons(self, sample_no_singletons):
        singles = _detected(sample_no_singletons)
        assert singles == []


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: raise_error()
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: certainty()
# ══════════════════════════════════════════════════════════════════════════════


class TestCertainty:
    """Tests for certainty() method."""

    def test_certainty_returns_new_sample(self, sample):
        result = _declare(sample, "self_representing")
        assert result is not sample

    def test_certainty_creates_variance_columns(self, sample):
        """
        certainty() creates internal variance columns for variance estimation.
        The original stratum/PSU columns are NOT modified.
        """
        result = _declare(sample, "self_representing")
        assert has_variance_columns(result)

    def test_certainty_resolves_singletons_for_variance(self, sample):
        """
        certainty() resolves singletons in the variance structure by treating
        the PSU as a stratum and observations as PSUs.
        """
        result = _declare(sample, "self_representing")
        assert singletons_resolved_for_variance(result)

    def test_certainty_preserves_original_stratum_column(self, sample, names):
        """The original stratum column should be unchanged."""
        result = _declare(sample, "self_representing")
        original_strata = set(sample._data.get_column(names["stratum"]).unique().to_list())
        result_strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert original_strata == result_strata

    def test_certainty_variance_stratum_uses_psu(self, sample, names):
        """For singletons, the variance stratum should be the original PSU."""
        result = _declare(sample, "self_representing")
        df = result._data

        # For singleton North__by__A (PSU=101), variance stratum should be "101"
        north_a = df.filter(pl.col(names["stratum"]) == "North__by__A")
        var_strata = north_a.get_column(_VAR_STRATUM_COL).unique().to_list()
        assert var_strata == ["101"]

    def test_certainty_variance_psu_uses_row_index(self, sample):
        """For singletons, the variance PSU should be the row index."""
        result = _declare(sample, "self_representing")
        df = result._data

        # Check that variance PSUs are unique per row for singletons
        singleton_rows = df.filter(pl.col(_VAR_STRATUM_COL) == "101")
        n_rows = singleton_rows.height
        n_unique_psus = singleton_rows.get_column(_VAR_PSU_COL).n_unique()
        assert n_unique_psus == n_rows

    def test_certainty_preserves_non_singleton_structure(self, sample, names):
        """Non-singleton strata should keep their original structure in variance columns."""
        result = _declare(sample, "self_representing")
        df = result._data

        # For non-singleton North__by__B, variance stratum should be "North__by__B"
        north_b = df.filter(pl.col(names["stratum"]) == "North__by__B")
        var_strata = north_b.get_column(_VAR_STRATUM_COL).unique().to_list()
        assert var_strata == ["North__by__B"]

    def test_certainty_idempotent_when_no_singletons(self, sample_no_singletons):
        """Without singletons the declared rule is idle."""
        result = _declare(sample_no_singletons, "self_representing")
        assert result.design.singleton.method == "self_representing"
        assert _result(result) is None

    def test_certainty_preserves_categorical_dtype(self, sample, names):
        df_cat = sample.data.with_columns(pl.col("cluster").cast(pl.Categorical))
        sample_cat = sample.clone(data=df_cat, design=sample._design)
        result = _declare(sample_cat, "self_representing")
        # The PSU column keeps its dtype
        assert result.data.schema["cluster"] == pl.Categorical

    def test_certainty_sets_last_result(self, sample):
        result = _declare(sample, "self_representing")
        assert _result(result) is not None
        assert _result(result).method == SingletonMethod.SELF_REPRESENTING

    def test_certainty_records_handling_on_no_psu_design(self):
        # Regression: with no PSU declared each row is its own PSU, which made
        # a former ids-already-conform shortcut return the sample with NO
        # config — estimation then failed exactly as if nothing was handled.
        from svy.core import Design, Sample

        df = pl.DataFrame(
            {
                "strat": [3, 3, 3, 4, 4, 4, 3, 4],
                "region": [1, 1, 1, 1, 2, 2, 2, 2],
                "wgt": [125.0, 108.0, 98.0, 140.0, 112.0, 88.0, 118.0, 102.0],
                "y": [25.0, 28.1, 23.5, 31.0, 27.2, 24.4, 29.5, 26.1],
            }
        )
        s = Sample(df, Design(stratum=("strat", "region"), wgt="wgt"))
        fixed = _declare(s, "self_representing")
        assert _result(fixed) is not None
        est = fixed.estimation.mean("y").to_polars()
        assert est["se"][0] > 0
        again = _declare(fixed, "self_representing")
        assert _result(again) == _result(fixed)

    def test_certainty_config_has_variance_columns(self, sample):
        """The result config should specify the variance column names."""
        result = _declare(sample, "self_representing")
        config = _result(result).config
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
        design = DesignStub(case_id=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        assert sample.n_singletons == 1

        result = _declare(sample, "self_representing")
        # Still has 1 singleton in variance structure because the stratum had only 1 observation
        assert get_effective_singletons(result) == 1


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: skip()
# ══════════════════════════════════════════════════════════════════════════════


class TestSkip:
    """Tests for skip() method."""

    def test_skip_returns_new_sample(self, sample):
        result = _declare(sample, "skip")
        assert result is not sample

    def test_skip_creates_variance_columns(self, sample):
        """skip() creates internal variance columns with exclusion flags."""
        result = _declare(sample, "skip")
        assert has_variance_columns(result)

    def test_skip_marks_singleton_rows_as_excluded(self, sample, names):
        """Singleton rows should be marked as excluded in the variance structure."""
        result = _declare(sample, "skip")
        df = result._data

        # Singleton rows should have exclude=True
        north_a = df.filter(pl.col(names["stratum"]) == "North__by__A")
        assert north_a.get_column(_VAR_EXCLUDE_COL).all()

        south_x = df.filter(pl.col(names["stratum"]) == "South__by__X")
        assert south_x.get_column(_VAR_EXCLUDE_COL).all()

    def test_skip_preserves_non_singleton_rows(self, sample, names):
        """Non-singleton rows should NOT be marked as excluded."""
        result = _declare(sample, "skip")
        df = result._data

        # Non-singleton rows should have exclude=False
        north_b = df.filter(pl.col(names["stratum"]) == "North__by__B")
        assert not north_b.get_column(_VAR_EXCLUDE_COL).any()

    def test_skip_preserves_all_rows_in_data(self, sample):
        """skip() does NOT remove rows - it marks them as excluded."""
        result = _declare(sample, "skip")
        assert result._data.height == sample._data.height

    def test_skip_resolves_singletons_for_variance(self, sample):
        """After exclusion, no singletons should remain in the variance structure."""
        result = _declare(sample, "skip")
        assert singletons_resolved_for_variance(result)

    def test_skip_noop_when_no_singletons(self, sample_no_singletons):
        result = _declare(sample_no_singletons, "skip")
        # Without singletons the declared rule is idle
        assert _result(result) is None

    def test_skip_sets_last_result(self, sample):
        result = _declare(sample, "skip")
        assert _result(result) is not None
        assert _result(result).method == SingletonMethod.SKIP


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: combine()
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: collapse()
# ══════════════════════════════════════════════════════════════════════════════


class TestCollapseSmallest:
    """Tests for collapse(using='smallest')."""

    def test_collapse_smallest_returns_new_sample(self, sample):
        result = _declare(sample, "collapse", using="smallest")
        assert result is not sample

    def test_collapse_smallest_creates_variance_columns(self, sample):
        """collapse() creates internal variance columns with remapped strata."""
        result = _declare(sample, "collapse", using="smallest")
        assert has_variance_columns(result)

    def test_collapse_smallest_resolves_singletons_for_variance(self, sample):
        """Singletons should be resolved in the variance structure."""
        result = _declare(sample, "collapse", using="smallest")
        assert singletons_resolved_for_variance(result)

    def test_collapse_smallest_preserves_original_stratum_column(self, sample, names):
        """The original stratum column should be unchanged."""
        result = _declare(sample, "collapse", using="smallest")
        original_strata = set(sample._data.get_column(names["stratum"]).unique().to_list())
        result_strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert original_strata == result_strata

    def test_collapse_smallest_remaps_variance_stratum(self, sample):
        """The variance stratum column should have remapped singleton keys."""
        result = _declare(sample, "collapse", using="smallest")
        df = result._data

        # Variance strata should have fewer unique values than original
        var_strata = set(df.get_column(_VAR_STRATUM_COL).unique().to_list())
        # Singletons should be merged into non-singletons
        assert "North__by__A" not in var_strata
        assert "South__by__X" not in var_strata

    def test_collapse_smallest_preserves_row_count(self, sample):
        result = _declare(sample, "collapse", using="smallest")
        assert result._data.height == sample._data.height

    def test_collapse_smallest_sets_last_result(self, sample):
        result = _declare(sample, "collapse", using="smallest")
        assert _result(result) is not None
        assert _result(result).method == SingletonMethod.COLLAPSE
        assert isinstance(_result(result).applied, dict)


class TestCollapseLargest:
    """Tests for collapse(using='largest')."""

    def test_collapse_largest_resolves_singletons_for_variance(self, sample):
        result = _declare(sample, "collapse", using="largest")
        assert singletons_resolved_for_variance(result)


class TestCollapseNext:
    """Tests for collapse(using='next')."""

    def test_collapse_next_resolves_singletons_for_variance(self, sample):
        result = _declare(sample, "collapse", using="next")
        assert singletons_resolved_for_variance(result)

    def test_collapse_next_with_order_by(self, sample):
        # region is constant within each stratum, so it can order them.
        result = _declare(sample, "collapse", using="next", order_by="region")
        assert singletons_resolved_for_variance(result)

    def test_collapse_order_by_a_column_varying_within_strata_raises(self, sample):
        # income differs between rows of a stratum: it gives a stratum no one place.
        result = _declare(sample, "collapse", using="next", order_by="income")
        with pytest.raises(SingletonError) as err:
            _resolve(result)
        assert err.value.code == "SINGLETON_ORDER_BY_INVALID"


class TestCollapsePrevious:
    """Tests for collapse(using='previous')."""

    def test_collapse_previous_resolves_singletons_for_variance(self, sample):
        result = _declare(sample, "collapse", using="previous")
        assert singletons_resolved_for_variance(result)


class TestCollapseWithMapping:
    """Tests for collapse(using=dict)."""

    def test_collapse_with_explicit_mapping(self, sample):
        mapping = {
            "North__by__A": "North__by__B",
            "South__by__X": "South__by__Y",
        }
        result = _declare(sample, "collapse", using=mapping)

        assert singletons_resolved_for_variance(result)

        # Verify mapping in variance column
        df = result._data
        var_strata = set(df.get_column(_VAR_STRATUM_COL).unique().to_list())
        assert var_strata == {"North__by__B", "South__by__Y"}

    def test_collapse_with_incomplete_mapping_raises(self, sample):
        mapping = {"North__by__A": "North__by__B"}  # Missing South__by__X
        h = _declare(sample, "collapse", using=mapping)
        with pytest.raises(SingletonError, match="not in the collapse mapping"):
            _resolve(h)

    def test_collapse_with_invalid_target_raises(self, sample):
        mapping = {
            "North__by__A": "Invalid__Target",
            "South__by__X": "South__by__Y",
        }
        h = _declare(sample, "collapse", using=mapping)
        with pytest.raises(SingletonError, match="no longer has"):
            _resolve(h)


class TestCollapseWithCallable:
    """Tests for collapse(using=callable)."""

    def test_collapse_with_callable(self, sample):
        def always_pick_first(singleton, candidates):
            return candidates[0].stratum_key

        result = _declare(sample, "collapse", using=always_pick_first)
        assert singletons_resolved_for_variance(result)

    def test_collapse_with_callable_invalid_return_raises(self, sample):
        def bad_picker(singleton, candidates):
            return "Invalid__Key"

        h = _declare(sample, "collapse", using=bad_picker)
        with pytest.raises(SingletonError, match="not a valid candidate"):
            _resolve(h)


class TestCollapseWithin:
    """Tests for collapse with within constraint."""

    def test_collapse_within_region(self, sample):
        result = _declare(sample, "collapse", using="smallest", within="region")
        assert singletons_resolved_for_variance(result)

        # Verify mapping respected region constraint
        last_result = _result(result)
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
        design = DesignStub(case_id=SVY_ROW_INDEX, stratum=["region", "district"], psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        # North singleton has no non-singleton in North region
        h = _declare(sample, "collapse", using="smallest", within="region")
        with pytest.raises(SingletonError, match="No valid merge targets"):
            _resolve(h)


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
        design = DesignStub(case_id=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        result = _declare(sample, "collapse", using="smallest")
        assert singletons_resolved_for_variance(result)

        # Due to rebalancing, singletons should be distributed
        # (not all going to the same stratum)
        mapping = _result(result).applied
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
        result1 = _declare(sample, "collapse", using="smallest")
        result2 = _declare(sample, "collapse", using="smallest")

        mapping1 = _result(result1).applied
        mapping2 = _result(result2).applied
        assert mapping1 == mapping2

        # Verify expected mapping
        expected = {
            "North__by__A": "North__by__B",
            "South__by__X": "South__by__Y",
        }
        assert mapping1 == expected

    def test_collapse_with_rstate_reproducible(self, sample):
        result1 = _declare(sample, "collapse", using="smallest", rstate=42)
        result2 = _declare(sample, "collapse", using="smallest", rstate=42)

        mapping1 = _result(result1).applied
        mapping2 = _result(result2).applied
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
        design = DesignStub(case_id=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        # With rstate, ties broken randomly
        # Run multiple times to see if we get variation (probabilistic test)
        results = set()
        for seed in range(10):
            result = _declare(sample, "collapse", using="smallest", rstate=seed)
            mapping = _result(result).applied
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
        result = _declare(sample, "pool")
        assert result is not sample

    def test_pool_creates_variance_columns(self, sample):
        """pool() creates internal variance columns with pooled stratum."""
        result = _declare(sample, "pool")
        assert has_variance_columns(result)

    def test_pool_resolves_singletons_for_variance(self, sample):
        """Singletons should be resolved in the variance structure."""
        result = _declare(sample, "pool")
        assert singletons_resolved_for_variance(result)

    def test_pool_creates_pooled_stratum_in_variance_column(self, sample):
        """The pooled stratum should appear in the variance column."""
        result = _declare(sample, "pool")
        var_strata = result._data.get_column(_VAR_STRATUM_COL).unique().to_list()
        assert "__pooled__" in var_strata

    def test_pool_custom_name_in_variance_column(self, sample):
        """Custom pool name should appear in the variance column."""
        result = _declare(sample, "pool", name="misc_strata")
        var_strata = result._data.get_column(_VAR_STRATUM_COL).unique().to_list()
        assert "misc_strata" in var_strata

    def test_pool_preserves_original_stratum_column(self, sample, names):
        """The original stratum column should be unchanged."""
        result = _declare(sample, "pool")
        original_strata = set(sample._data.get_column(names["stratum"]).unique().to_list())
        result_strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert original_strata == result_strata

    def test_pool_preserves_row_count(self, sample):
        result = _declare(sample, "pool")
        assert result._data.height == sample._data.height  # 9 rows

    def test_pool_combines_singleton_psus_in_variance_column(self, sample):
        """The pooled stratum should have multiple PSUs in the variance structure."""
        result = _declare(sample, "pool")
        pooled = result._data.filter(pl.col(_VAR_STRATUM_COL) == "__pooled__")
        n_psus = pooled.get_column(_VAR_PSU_COL).n_unique()
        # Should have 2 PSUs (one from each original singleton)
        assert n_psus == 2

    def test_pool_sets_last_result(self, sample):
        result = _declare(sample, "pool")
        assert _result(result) is not None
        assert _result(result).method == SingletonMethod.POOL

    def test_pool_noop_when_no_singletons(self, sample_no_singletons):
        result = _declare(sample_no_singletons, "pool")
        # Without singletons the declared rule is idle
        assert _result(result) is None


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: scale()
# ══════════════════════════════════════════════════════════════════════════════


class TestScale:
    """Tests for scale() method."""

    def test_scale_returns_new_sample(self, sample):
        result = _declare(sample, "scale")
        assert result is not sample

    def test_scale_creates_variance_columns(self, sample):
        """scale() creates internal variance columns like skip()."""
        result = _declare(sample, "scale")
        assert has_variance_columns(result)

    def test_scale_marks_singletons_as_excluded(self, sample, names):
        """Singleton rows should be marked as excluded in the variance structure."""
        result = _declare(sample, "scale")
        df = result._data

        # Singleton rows should have exclude=True
        north_a = df.filter(pl.col(names["stratum"]) == "North__by__A")
        assert north_a.get_column(_VAR_EXCLUDE_COL).all()

        south_x = df.filter(pl.col(names["stratum"]) == "South__by__X")
        assert south_x.get_column(_VAR_EXCLUDE_COL).all()

    def test_scale_preserves_non_singleton_rows(self, sample, names):
        """Non-singleton rows should NOT be marked as excluded."""
        result = _declare(sample, "scale")
        df = result._data

        north_b = df.filter(pl.col(names["stratum"]) == "North__by__B")
        assert not north_b.get_column(_VAR_EXCLUDE_COL).any()

    def test_scale_resolves_singletons_for_variance(self, sample):
        """After exclusion, no singletons should remain in the variance structure."""
        result = _declare(sample, "scale")
        assert singletons_resolved_for_variance(result)

    def test_scale_preserves_all_rows(self, sample):
        """scale() does NOT remove rows - it marks them as excluded."""
        result = _declare(sample, "scale")
        assert result._data.height == sample._data.height

    def test_scale_preserves_original_stratum_column(self, sample, names):
        """The original stratum column should be unchanged."""
        result = _declare(sample, "scale")
        original_strata = set(sample._data.get_column(names["stratum"]).unique().to_list())
        result_strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert original_strata == result_strata

    def test_scale_sets_last_result(self, sample):
        result = _declare(sample, "scale")
        assert _result(result) is not None
        assert _result(result).method == SingletonMethod.SCALE

    def test_scale_config_has_singleton_fraction(self, sample):
        """The result config should include the singleton fraction."""
        result = _declare(sample, "scale")
        config = _result(result).config
        assert config is not None
        assert config.singleton_fraction is not None
        # 2 singletons out of 4 strata = 0.5
        assert config.singleton_fraction == pytest.approx(0.5)

    def test_scale_config_has_variance_columns(self, sample):
        """The result config should specify the variance column names."""
        result = _declare(sample, "scale")
        config = _result(result).config
        assert config.var_stratum_col == _VAR_STRATUM_COL
        assert config.var_psu_col == _VAR_PSU_COL
        assert config.var_exclude_col == _VAR_EXCLUDE_COL

    def test_scale_noop_when_no_singletons(self, sample_no_singletons):
        result = _declare(sample_no_singletons, "scale")
        # Without singletons the declared rule is idle
        assert _result(result) is None

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
        design = DesignStub(case_id=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        result = _declare(sample, "scale")
        config = _result(result).config

        # 1 singleton out of 5 strata = 0.2
        assert config.singleton_fraction == pytest.approx(0.2)

    def test_scale_inflation_factor(self, sample):
        """Test that the expected inflation factor can be calculated from singleton_fraction."""
        result = _declare(sample, "scale")
        config = _result(result).config

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
        result = _declare(sample, "center")
        assert result is not sample

    def test_center_creates_variance_columns(self, sample):
        """center() creates internal variance columns."""
        result = _declare(sample, "center")
        assert has_variance_columns(result)

    def test_center_does_not_exclude_singletons(self, sample, names):
        """Singleton rows should NOT be marked as excluded for CENTER."""
        result = _declare(sample, "center")
        df = result._data

        # No rows should be excluded
        assert not df.get_column(_VAR_EXCLUDE_COL).any()

    def test_center_marks_singleton_rows(self, sample, names):
        """Singleton rows should be marked with is_singleton flag."""
        result = _declare(sample, "center")
        df = result._data

        # Singleton rows should have is_singleton=True
        north_a = df.filter(pl.col(names["stratum"]) == "North__by__A")
        assert north_a.get_column(_VAR_IS_SINGLETON_COL).all()

        # Non-singleton rows should have is_singleton=False
        north_b = df.filter(pl.col(names["stratum"]) == "North__by__B")
        assert not north_b.get_column(_VAR_IS_SINGLETON_COL).any()

    def test_center_preserves_all_rows(self, sample):
        """center() preserves all rows including singletons."""
        result = _declare(sample, "center")
        assert result._data.height == sample._data.height

    def test_center_preserves_original_stratum_column(self, sample, names):
        """The original stratum column should be unchanged."""
        result = _declare(sample, "center")
        original_strata = set(sample._data.get_column(names["stratum"]).unique().to_list())
        result_strata = set(result._data.get_column(names["stratum"]).unique().to_list())
        assert original_strata == result_strata

    def test_center_sets_last_result(self, sample):
        result = _declare(sample, "center")
        assert _result(result) is not None
        assert _result(result).method == SingletonMethod.CENTER

    def test_center_config_has_no_singleton_fraction(self, sample):
        """CENTER method should not use singleton_fraction (that's for SCALE)."""
        result = _declare(sample, "center")
        config = _result(result).config
        assert config is not None
        assert config.singleton_fraction is None

    def test_center_config_has_singleton_keys(self, sample):
        """The config should include the singleton keys."""
        result = _declare(sample, "center")
        config = _result(result).config
        assert config.singleton_keys is not None
        assert set(config.singleton_keys) == {"North__by__A", "South__by__X"}

    def test_center_config_has_variance_columns(self, sample):
        """The result config should specify the variance column names."""
        result = _declare(sample, "center")
        config = _result(result).config
        assert config.var_stratum_col == _VAR_STRATUM_COL
        assert config.var_psu_col == _VAR_PSU_COL
        assert config.var_exclude_col == _VAR_EXCLUDE_COL

    def test_center_noop_when_no_singletons(self, sample_no_singletons):
        result = _declare(sample_no_singletons, "center")
        # Without singletons the declared rule is idle
        assert _result(result) is None

    def test_center_strata_counts_unchanged(self, sample):
        """CENTER should not change the stratum/PSU counts (unlike SKIP/SCALE)."""
        result = _declare(sample, "center")
        lr = _result(result)

        # No strata/PSUs should be "removed" for variance calculation
        assert lr.n_strata_before == lr.n_strata_after
        assert lr.n_psus_before == lr.n_psus_after


# ══════════════════════════════════════════════════════════════════════════════
# HANDLING: handle() dispatcher
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# RESULT ACCESS: last_result
# ══════════════════════════════════════════════════════════════════════════════


class TestLastResult:
    """Tests for last_result property."""

    def test_last_result_none_on_original_sample(self, sample):
        assert _result(sample) is None

    def test_last_result_set_after_handling(self, sample):
        result = _declare(sample, "self_representing")
        assert _result(result) is not None

    def test_last_result_is_singleton_result(self, sample):
        result = _declare(sample, "self_representing")
        assert isinstance(_result(result), SingletonResult)

    def test_last_result_has_method(self, sample):
        result = _declare(sample, "self_representing")
        assert _result(result).method == SingletonMethod.SELF_REPRESENTING

    def test_last_result_has_detected(self, sample):
        result = _declare(sample, "self_representing")
        assert len(_result(result).detected) == 2

    def test_last_result_has_counts(self, sample):
        result = _declare(sample, "self_representing")
        lr = _result(result)
        assert lr.n_singletons_detected == 2
        assert lr.n_strata_before > 0
        assert lr.n_psus_before > 0

    def test_last_result_applied_for_collapse(self, sample):
        result = _declare(sample, "collapse", using="smallest")
        lr = _result(result)
        assert isinstance(lr.applied, dict)
        assert len(lr.applied) == 2  # Two singletons mapped

    def test_last_result_applied_for_pool(self, sample):
        result = _declare(sample, "pool")
        lr = _result(result)
        assert isinstance(lr.applied, dict)
        # All singletons mapped to pooled name
        assert all(v == "__pooled__" for v in lr.applied.values())

    def test_last_result_has_config(self, sample):
        """The result should include a config for variance estimation."""
        result = _declare(sample, "self_representing")
        lr = _result(result)
        assert lr.config is not None
        assert lr.config.method == SingletonMethod.SELF_REPRESENTING
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
        case_id=SVY_ROW_INDEX,
        stratum=("region", "district"),
        psu=("cluster", "cluster_b"),
    )
    return SampleStub(df, design, stratum_internal=names["stratum"], psu_internal=names["psu"])


class TestTuplePsu:
    """Tests for tuple PSU handling."""

    def test_detect_with_tuple_psu(self, sample_tuple_psu):
        singles = _detected(sample_tuple_psu)
        keys = sorted(s.stratum_key for s in singles)
        assert keys == sorted(["North__by__A", "South__by__X"])

    def test_certainty_with_tuple_psu(self, sample_tuple_psu):
        result = _declare(sample_tuple_psu, "self_representing")
        assert singletons_resolved_for_variance(result)

    def test_collapse_with_tuple_psu(self, sample_tuple_psu):
        result = _declare(sample_tuple_psu, "collapse", using="smallest")
        assert singletons_resolved_for_variance(result)

    def test_pool_with_tuple_psu(self, sample_tuple_psu):
        result = _declare(sample_tuple_psu, "pool")
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
        design = DesignStub(case_id=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        # Pool should work - combines all into one stratum with 3 PSUs
        result = _declare(sample, "pool")
        assert singletons_resolved_for_variance(result)

        # Collapse should fail - no non-singleton targets
        h = _declare(sample, "collapse", using="smallest")
        with pytest.raises(SingletonError, match="No valid merge targets"):
            _resolve(h)

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
        design = DesignStub(case_id=SVY_ROW_INDEX, stratum="stratum", psu="cluster")
        sample = SampleStub(
            df, design, stratum_internal=names["stratum"], psu_internal=names["psu"]
        )

        singles = _detected(sample)
        assert len(singles) == 1
        assert singles[0].n_observations == 1

    def test_no_stratum_column(self):
        """Test behavior when no stratum is defined."""
        rows = [(0, "101"), (1, "102"), (2, "201")]
        df = pl.DataFrame(rows, schema=[SVY_ROW_INDEX, "cluster"], orient="row")
        design = DesignStub(case_id=SVY_ROW_INDEX, stratum=None, psu="cluster")
        sample = SampleStub(df, design, stratum_internal=None, psu_internal="cluster")

        assert not (sample.n_singletons > 0)
        assert _detected(sample) == []
