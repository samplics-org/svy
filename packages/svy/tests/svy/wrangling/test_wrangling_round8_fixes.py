# tests/svy/wrangling/test_wrangling_round8_fixes.py
"""
Round 8 wrangling edge-case fixes (findings WR1-WR8).

WR1  categorize() includes values equal to the outermost bin edge
WR2  remove_columns(force=True) cleans design.pop_size
WR3  partial replicate rename raises instead of corrupting the prefix
WR4  mutate() dependents always see same-call (re)defined values
WR5  clean_names(upper) leaves internal concat columns alone
WR6  filter_records warns about rows dropped via null predicates
WR7  fill_null(strategy="mean") casts integer columns to Float64
WR8  cast(strict=True) raises on lossy float->int truncation
"""

import polars as pl
import pytest

import svy

from svy.core.design import make_rep_weights
from svy.core.sample import Design, Sample
from svy.errors import MethodError


# ---------------------------------------------------------------------------
# WR1 — categorize outer edge
# ---------------------------------------------------------------------------


class TestCategorizeOuterEdge:
    def test_right_true_includes_lowest_edge(self):
        s = Sample(pl.DataFrame({"x": [0, 5, 10, 15, 20]}))
        out = s.wrangling.categorize("x", bins=[0, 10, 20])
        cats = out._data["svy_x_categorized"].to_list()
        assert cats == ["[0, 10]", "[0, 10]", "[0, 10]", "(10, 20]", "(10, 20]"]
        assert None not in cats

    def test_right_false_includes_highest_edge(self):
        s = Sample(pl.DataFrame({"x": [0, 5, 10, 15, 20]}))
        out = s.wrangling.categorize("x", bins=[0, 10, 20], right=False)
        cats = out._data["svy_x_categorized"].to_list()
        assert cats == ["[0, 10)", "[0, 10)", "[10, 20]", "[10, 20]", "[10, 20]"]
        assert None not in cats

    def test_out_of_range_still_null(self):
        s = Sample(pl.DataFrame({"x": [-1, 5, 21]}))
        out = s.wrangling.categorize("x", bins=[0, 10, 20])
        cats = out._data["svy_x_categorized"].to_list()
        assert cats[0] is None and cats[2] is None

    def test_custom_labels_cover_edge(self):
        s = Sample(pl.DataFrame({"x": [0, 10, 20]}))
        out = s.wrangling.categorize("x", bins=[0, 10, 20], labels=["lo", "hi"])
        assert out._data["svy_x_categorized"].to_list() == ["lo", "lo", "hi"]


# ---------------------------------------------------------------------------
# WR2 — pop_size cleaned on column removal
# ---------------------------------------------------------------------------


def test_remove_columns_cleans_pop_size():
    s = Sample(
        pl.DataFrame({"y": [1.0, 2.0], "w": [1.0, 1.0], "N": [100.0, 100.0]}),
        Design(wgt="w", pop_size="N"),
    )
    out = s.wrangling.remove_columns("N", force=True)
    assert "N" not in out._data.columns
    # Previously stayed "N", crashing later estimation with a bare KeyError
    assert out.design.pop_size is None


# ---------------------------------------------------------------------------
# WR3 — partial replicate rename raises
# ---------------------------------------------------------------------------


@pytest.fixture
def rep_sample():
    df = pl.DataFrame(
        {
            "strat": ["A", "A", "B", "B"],
            "psu": ["1", "2", "3", "4"],
            "w": [1.0, 2.0, 3.0, 4.0],
            "rw1": [1.0] * 4,
            "rw2": [1.0] * 4,
            "rw3": [1.0] * 4,
        }
    )
    rw = make_rep_weights("jackknife", prefix="rw", n_reps=3)
    return Sample(df, Design(stratum="strat", psu="psu", wgt="w", rep_wgts=rw))


def test_partial_replicate_rename_raises(rep_sample):
    """rw1 -> boot1 with rw2/rw3 left behind used to rewrite the prefix for
    ALL replicates, surfacing later as REP_WEIGHT_COUNT_MISMATCH."""
    with pytest.raises(Exception, match="[Pp]artial"):
        rep_sample.wrangling.rename_columns({"rw1": "boot1"})


def test_full_replicate_rename_still_works(rep_sample):
    out = rep_sample.wrangling.rename_columns(
        {"rw1": "boot1", "rw2": "boot2", "rw3": "boot3"}
    )
    assert out.design.rep_wgts.prefix == "boot"
    assert set(out.design.rep_wgts.columns) <= set(out._data.columns)


# ---------------------------------------------------------------------------
# WR4 — mutate ordering consistency
# ---------------------------------------------------------------------------


class TestMutateOrdering:
    def test_dependent_sees_redefined_value(self):
        """{"x": x+100, "y": x*1000}: y must use the NEW x (it previously
        used the old value, unlike the new-column path)."""
        s = Sample(pl.DataFrame({"x": [1.0, 2.0]}))
        out = s.wrangling.mutate({"x": pl.col("x") + 100, "y": pl.col("x") * 1000})
        assert out._data["x"].to_list() == [101.0, 102.0]
        assert out._data["y"].to_list() == [101000.0, 102000.0]

    def test_matches_new_column_path(self):
        s = Sample(pl.DataFrame({"x": [1.0, 2.0]}))
        out = s.wrangling.mutate({"x2": pl.col("x") + 100, "y2": pl.col("x2") * 1000})
        assert out._data["y2"].to_list() == [101000.0, 102000.0]

    def test_self_reference_uses_old_value(self):
        s = Sample(pl.DataFrame({"x": [1.0, 2.0]}))
        out = s.wrangling.mutate({"x": pl.col("x") + 1})
        assert out._data["x"].to_list() == [2.0, 3.0]

    def test_mutual_redefinition_raises(self):
        """Circular same-call redefinitions are ambiguous — error, not
        silent old-value evaluation."""
        s = Sample(pl.DataFrame({"a": [1.0], "b": [2.0]}))
        with pytest.raises(MethodError, match="[Cc]ircular"):
            s.wrangling.mutate({"a": pl.col("b") + 1, "b": pl.col("a") + 1})


# ---------------------------------------------------------------------------
# WR5 — clean_names leaves internal concat columns alone
# ---------------------------------------------------------------------------


def test_clean_names_upper_no_orphan_concat_columns():
    from svy.core.constants import _INTERNAL_CONCAT_SUFFIX

    df = pl.DataFrame(
        {
            "s1": ["A", "A", "B", "B"],
            "s2": ["x", "y", "x", "y"],
            "p1": ["1", "2", "3", "4"],
            "p2": ["a", "b", "c", "d"],
            "w": [1.0] * 4,
        }
    )
    # Multi-column stratum/psu force internal concat columns to exist
    s = Sample(df, Design(stratum=("s1", "s2"), psu=("p1", "p2"), wgt="w"))
    concat_before = [c for c in s._data.columns if _INTERNAL_CONCAT_SUFFIX in c]
    assert concat_before  # sanity: the fixture exercises the concat path

    out = s.wrangling.clean_names(letter_case="upper")
    concat_after = [c for c in out._data.columns if _INTERNAL_CONCAT_SUFFIX.upper() in c]
    # No uppercased orphans left behind
    assert concat_after == []
    # Exactly one copy of each concat column (no junk duplicates)
    concat_now = [c for c in out._data.columns if _INTERNAL_CONCAT_SUFFIX in c]
    assert sorted(concat_now) == sorted(concat_before)


# ---------------------------------------------------------------------------
# WR6 — null predicates warn
# ---------------------------------------------------------------------------


class TestFilterNullPredicates:
    def test_filter_warns_on_null_predicate_rows(self):
        s = Sample(pl.DataFrame({"x": [1.0, None, 3.0, None]}))
        out = s.wrangling.filter_records(svy.col("x") > 2)
        assert out._data.height == 1
        warns = [w for w in out._warnings.list() if w.code == "NULL_PREDICATE_ROWS_DROPPED"]
        assert warns and "2 row(s)" in warns[0].detail

    def test_negated_filter_also_warns(self):
        s = Sample(pl.DataFrame({"x": [1.0, None, 3.0]}))
        out = s.wrangling.filter_records(svy.col("x") > 2, negate=True)
        assert out._data.height == 1
        assert any(
            w.code == "NULL_PREDICATE_ROWS_DROPPED" for w in out._warnings.list()
        )

    def test_no_warning_without_nulls(self):
        s = Sample(pl.DataFrame({"x": [1.0, 3.0]}))
        out = s.wrangling.filter_records(svy.col("x") > 2)
        assert not any(
            w.code == "NULL_PREDICATE_ROWS_DROPPED" for w in out._warnings.list()
        )


# ---------------------------------------------------------------------------
# WR7 — mean fill on integer columns
# ---------------------------------------------------------------------------


class TestMeanFillIntegers:
    def test_mean_fill_int_casts_to_float(self):
        s = Sample(pl.DataFrame({"x": [1, None, 2]}, schema={"x": pl.Int64}))
        out = s.wrangling.fill_null("x", strategy="mean")
        assert out._data["x"].dtype == pl.Float64
        assert out._data["x"].to_list() == [1.0, 1.5, 2.0]

    def test_mean_fill_int_warns(self):
        s = Sample(pl.DataFrame({"x": [1, None, 2]}, schema={"x": pl.Int64}))
        out = s.wrangling.fill_null("x", strategy="mean")
        assert any(w.code == "MEAN_FILL_INT_CAST" for w in out._warnings.list())

    def test_mean_fill_float_unchanged(self):
        s = Sample(pl.DataFrame({"x": [1.0, None, 2.0]}))
        out = s.wrangling.fill_null("x", strategy="mean")
        assert out._data["x"].dtype == pl.Float64
        assert out._data["x"].to_list() == [1.0, 1.5, 2.0]
        assert not any(w.code == "MEAN_FILL_INT_CAST" for w in out._warnings.list())

    def test_other_strategies_keep_int(self):
        s = Sample(pl.DataFrame({"x": [1, None, 2]}, schema={"x": pl.Int64}))
        out = s.wrangling.fill_null("x", strategy="max")
        assert out._data["x"].dtype == pl.Int64
        assert out._data["x"].to_list() == [1, 2, 2]


# ---------------------------------------------------------------------------
# WR8 — strict cast truncation guard
# ---------------------------------------------------------------------------


class TestCastTruncationGuard:
    def test_strict_float_to_int_truncation_raises(self):
        s = Sample(pl.DataFrame({"x": [1.7, 2.0, 3.5]}))
        with pytest.raises(MethodError, match="[Tt]runcat|[Ll]ossy"):
            s.wrangling.cast("x", pl.Int64)

    def test_strict_whole_floats_ok(self):
        s = Sample(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
        out = s.wrangling.cast("x", pl.Int64)
        assert out._data["x"].dtype == pl.Int64
        assert out._data["x"].to_list() == [1, 2, 3]

    def test_non_strict_still_truncates(self):
        s = Sample(pl.DataFrame({"x": [1.7, 2.2]}))
        out = s.wrangling.cast("x", pl.Int64, strict=False)
        assert out._data["x"].to_list() == [1, 2]

    def test_mapping_form_guarded(self):
        s = Sample(pl.DataFrame({"x": [1.7], "y": [2.0]}))
        with pytest.raises(MethodError):
            s.wrangling.cast({"x": pl.Int64, "y": pl.Int64})

    def test_int_to_int_unaffected(self):
        s = Sample(pl.DataFrame({"x": [1, 2]}, schema={"x": pl.Int64}))
        out = s.wrangling.cast("x", pl.Int32)
        assert out._data["x"].dtype == pl.Int32
