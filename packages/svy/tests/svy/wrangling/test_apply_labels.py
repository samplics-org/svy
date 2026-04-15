# tests/svy/wrangling/test_apply_labels.py
"""Tests for apply_labels functionality."""

import math

import polars as pl
import pytest

from svy.core.sample import Sample
from svy.errors import DimensionError, LabelError, MethodError
from svy.metadata import VariableMeta


# ---------- Fixtures ----------


@pytest.fixture
def sample_basic():
    df = pl.DataFrame(
        {
            "sex": [1, 2, 1, 2],
            "region": ["N", "S", "E", "W"],
            "age": [25, 30, 35, 40],
        }
    )
    return Sample(df)


@pytest.fixture
def sample_categorical():
    df = pl.DataFrame(
        {
            "status": pl.Series(["active", "inactive", "active"], dtype=pl.Categorical),
            "grade": pl.Series(["A", "B", "C"], dtype=pl.Categorical),
        }
    )
    return Sample(df)


# ==================== Basic label application ====================


def test_apply_labels_variable_label_only(sample_basic: Sample):
    out = sample_basic.wrangling.apply_labels(labels={"age": "Age in years"})
    meta = out.meta.get("age")
    assert meta is not None
    assert meta.label == "Age in years"
    assert meta.value_labels is None


def test_apply_labels_categories_only(sample_basic: Sample):
    out = sample_basic.wrangling.apply_labels(categories={"sex": {1: "Male", 2: "Female"}})
    meta = out.meta.get("sex")
    assert meta is not None
    assert meta.value_labels == {1: "Male", 2: "Female"}


def test_apply_labels_both(sample_basic: Sample):
    out = sample_basic.wrangling.apply_labels(
        labels={"sex": "Sex of respondent"},
        categories={"sex": {1: "Male", 2: "Female"}},
    )
    meta = out.meta.get("sex")
    assert meta is not None
    assert meta.label == "Sex of respondent"
    assert meta.value_labels == {1: "Male", 2: "Female"}


def test_apply_labels_multiple_variables(sample_basic: Sample):
    out = sample_basic.wrangling.apply_labels(
        labels={"sex": "Sex", "region": "Region"},
        categories={
            "sex": {1: "Male", 2: "Female"},
            "region": {"N": "North", "S": "South", "E": "East", "W": "West"},
        },
    )
    assert out.meta.get("sex").label == "Sex"
    assert out.meta.get("sex").value_labels == {1: "Male", 2: "Female"}
    assert out.meta.get("region").label == "Region"
    assert out.meta.get("region").value_labels == {
        "N": "North",
        "S": "South",
        "E": "East",
        "W": "West",
    }


def test_apply_labels_independent_keys(sample_basic: Sample):
    """Variables in labels and categories don't have to overlap."""
    out = sample_basic.wrangling.apply_labels(
        labels={"age": "Age in years"},
        categories={"sex": {1: "Male", 2: "Female"}},
    )
    assert out.meta.get("age").label == "Age in years"
    assert out.meta.get("sex").value_labels == {1: "Male", 2: "Female"}


def test_apply_labels_returns_new_instance_by_default(sample_basic: Sample):
    out = sample_basic.wrangling.apply_labels(labels={"age": "Age"})
    assert out is not sample_basic


def test_apply_labels_inplace_returns_same_instance(sample_basic: Sample):
    out = sample_basic.wrangling.apply_labels(labels={"age": "Age"}, inplace=True)
    assert out is sample_basic
    assert out.meta.get("age").label == "Age"


# ==================== No labels provided ====================


def test_apply_labels_no_args_raises():
    df = pl.DataFrame({"a": [1]})
    s = Sample(df)
    with pytest.raises(MethodError) as exc_info:
        s.wrangling.apply_labels()
    assert exc_info.value.code == "MISSING_LABELS"


def test_apply_labels_both_none_raises():
    df = pl.DataFrame({"a": [1]})
    s = Sample(df)
    with pytest.raises(MethodError) as exc_info:
        s.wrangling.apply_labels(labels=None, categories=None)
    assert exc_info.value.code == "MISSING_LABELS"


# ==================== Type validation ====================


def test_apply_labels_labels_must_be_dict():
    df = pl.DataFrame({"a": [1]})
    s = Sample(df)
    with pytest.raises(MethodError) as exc_info:
        s.wrangling.apply_labels(labels=["not", "a", "dict"])
    assert exc_info.value.code == "INVALID_TYPE"


def test_apply_labels_labels_values_must_be_strings():
    df = pl.DataFrame({"a": [1]})
    s = Sample(df)
    with pytest.raises(MethodError) as exc_info:
        s.wrangling.apply_labels(labels={"a": 123})
    assert exc_info.value.code == "INVALID_TYPE"


def test_apply_labels_categories_must_be_dict():
    df = pl.DataFrame({"a": [1, 2]})
    s = Sample(df)
    with pytest.raises(MethodError) as exc_info:
        s.wrangling.apply_labels(categories=["not", "a", "dict"])
    assert exc_info.value.code == "INVALID_TYPE"


def test_apply_labels_categories_inner_must_be_dict():
    df = pl.DataFrame({"a": [1, 2]})
    s = Sample(df)
    with pytest.raises(MethodError) as exc_info:
        s.wrangling.apply_labels(categories={"a": "not a dict"})
    assert exc_info.value.code == "INVALID_TYPE"


def test_apply_labels_category_values_must_be_strings():
    df = pl.DataFrame({"a": [1, 2]})
    s = Sample(df)
    with pytest.raises(MethodError):
        s.wrangling.apply_labels(categories={"a": {1: 123}})


def test_apply_labels_nan_key_forbidden():
    df = pl.DataFrame({"a": [1.0, 2.0]})
    s = Sample(df)
    with pytest.raises(LabelError):
        s.wrangling.apply_labels(categories={"a": {math.nan: "Invalid"}})


# ==================== Missing column handling ====================


def test_apply_labels_missing_column_raises_strict(sample_basic: Sample):
    with pytest.raises(DimensionError):
        sample_basic.wrangling.apply_labels(labels={"nonexistent": "Label"}, strict=True)


def test_apply_labels_missing_column_in_categories_raises_strict(sample_basic: Sample):
    with pytest.raises(DimensionError):
        sample_basic.wrangling.apply_labels(categories={"nonexistent": {1: "X"}}, strict=True)


def test_apply_labels_missing_column_ignored_non_strict(sample_basic: Sample):
    out = sample_basic.wrangling.apply_labels(labels={"nonexistent": "Label"}, strict=False)
    meta = out.meta.get("nonexistent")
    assert meta is None or meta.label is None


def test_apply_labels_partial_missing_non_strict(sample_basic: Sample):
    out = sample_basic.wrangling.apply_labels(
        labels={"age": "Age", "nonexistent": "Missing"}, strict=False
    )
    assert out.meta.get("age").label == "Age"
    meta = out.meta.get("nonexistent")
    assert meta is None or meta.label is None


def test_apply_labels_missing_in_labels_and_categories_combined(sample_basic: Sample):
    """Missing variable referenced in both labels and categories."""
    with pytest.raises(DimensionError):
        sample_basic.wrangling.apply_labels(
            labels={"nonexistent": "Label"},
            categories={"nonexistent": {1: "X"}},
            strict=True,
        )


# ==================== Overwrite behavior ====================


def test_apply_labels_overwrite_true_replaces(sample_basic: Sample):
    s = sample_basic.wrangling.apply_labels(labels={"age": "Original label"})
    out = s.wrangling.apply_labels(labels={"age": "New label"}, overwrite=True)
    assert out.meta.get("age").label == "New label"


def test_apply_labels_overwrite_false_raises_for_labels():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    s = s.wrangling.apply_labels(labels={"a": "First label"})
    with pytest.raises(MethodError) as exc_info:
        s.wrangling.apply_labels(labels={"a": "Second label"}, overwrite=False)
    assert "overwrite=True" in str(exc_info.value.hint)


def test_apply_labels_overwrite_false_raises_for_categories():
    df = pl.DataFrame({"a": [1, 2]})
    s = Sample(df)
    s = s.wrangling.apply_labels(categories={"a": {1: "One", 2: "Two"}})
    with pytest.raises(MethodError) as exc_info:
        s.wrangling.apply_labels(categories={"a": {1: "X"}}, overwrite=False)
    assert "overwrite=True" in str(exc_info.value.hint)


def test_apply_labels_overwrite_false_label_ok_when_no_existing():
    """overwrite=False should succeed if no existing label."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.apply_labels(labels={"a": "First label"}, overwrite=False)
    assert out.meta.get("a").label == "First label"


# ==================== String-keyed categories ====================


def test_apply_labels_string_categories():
    df = pl.DataFrame({"status": ["A", "B", "C"]})
    s = Sample(df)
    out = s.wrangling.apply_labels(
        labels={"status": "Status code"},
        categories={"status": {"A": "Active", "B": "Blocked", "C": "Closed"}},
    )
    assert out.meta.get("status").value_labels == {
        "A": "Active",
        "B": "Blocked",
        "C": "Closed",
    }


def test_apply_labels_partial_categories():
    """Only some values labeled — others left unlabeled (with warning)."""
    df = pl.DataFrame({"tenure": ["Owned", "Rented", "Occupied for free"]})
    s = Sample(df)
    out = s.wrangling.apply_labels(
        categories={"tenure": {"Occupied for free": "Free use"}},
    )
    assert out.meta.get("tenure").value_labels == {"Occupied for free": "Free use"}
    warns = [w for w in out.warnings if w.code == "DATA_VALUE_NOT_LABELED"]
    assert len(warns) == 1


def test_apply_labels_extra_category_keys_warns():
    """Category keys not in data should produce a warning."""
    df = pl.DataFrame({"a": [1, 2]})
    s = Sample(df)
    out = s.wrangling.apply_labels(
        categories={"a": {1: "One", 2: "Two", 99: "Unknown"}},
    )
    assert out.meta.get("a").value_labels == {1: "One", 2: "Two", 99: "Unknown"}
    warns = [w for w in out.warnings if w.code == "LABEL_KEY_NOT_IN_DATA"]
    assert len(warns) == 1


# ==================== Integration with other methods ====================


def test_labels_preserved_after_filter(sample_basic: Sample):
    from svy.core.expr import col

    labeled = sample_basic.wrangling.apply_labels(
        labels={"sex": "Sex"},
        categories={"sex": {1: "M", 2: "F"}},
    )
    filtered = labeled.wrangling.filter_records(col("age") > 25)
    meta = filtered.meta.get("sex")
    assert meta is not None
    assert meta.value_labels == {1: "M", 2: "F"}


def test_labels_updated_on_rename(sample_basic: Sample):
    labeled = sample_basic.wrangling.apply_labels(
        labels={"sex": "Sex of respondent"},
        categories={"sex": {1: "M", 2: "F"}},
    )
    renamed = labeled.wrangling.rename_columns({"sex": "gender"})
    gender_meta = renamed.meta.get("gender")
    sex_meta = renamed.meta.get("sex")
    assert gender_meta is not None
    assert sex_meta is None
    assert gender_meta.label == "Sex of respondent"


def test_labels_initialized_empty():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    meta = s.meta.get("a")
    assert meta is None or meta.label is None


def test_apply_labels_creates_metadata():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.apply_labels(labels={"a": "Column A"})
    meta = out.meta.get("a")
    assert meta is not None
    assert meta.label == "Column A"


# ==================== Label object structure ====================


def test_label_object_attributes():
    df = pl.DataFrame({"x": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.apply_labels(
        labels={"x": "X Variable"},
        categories={"x": {1: "Low", 2: "Med", 3: "High"}},
    )
    meta = out.meta.get("x")
    assert isinstance(meta, VariableMeta)
    assert meta.label == "X Variable"
    assert meta.value_labels == {1: "Low", 2: "Med", 3: "High"}


def test_label_without_categories():
    df = pl.DataFrame({"continuous_var": [1.5, 2.5, 3.5]})
    s = Sample(df)
    out = s.wrangling.apply_labels(labels={"continuous_var": "A continuous measure"})
    meta = out.meta.get("continuous_var")
    assert meta.label == "A continuous measure"
    assert meta.value_labels is None


# ==================== Empty and edge cases ====================


def test_apply_labels_empty_labels_dict():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.apply_labels(labels={}, categories={"a": {1: "One"}})
    assert out.meta.get("a").value_labels == {1: "One"}


def test_apply_labels_empty_categories_dict():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.apply_labels(labels={"a": "Label"}, categories={})
    meta = out.meta.get("a")
    assert meta is not None
    assert meta.label == "Label"


def test_apply_labels_single_row_dataframe():
    df = pl.DataFrame({"a": [1]})
    s = Sample(df)
    out = s.wrangling.apply_labels(
        labels={"a": "Single value"},
        categories={"a": {1: "One"}},
    )
    assert out.meta.get("a").value_labels == {1: "One"}


def test_apply_labels_with_all_nulls():
    df = pl.DataFrame({"a": [None, None, None]})
    s = Sample(df)
    out = s.wrangling.apply_labels(labels={"a": "All nulls"})
    meta = out.meta.get("a")
    assert meta is not None
    assert meta.label == "All nulls"
