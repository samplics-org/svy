# tests/test_utils.py
import polars as pl
import pytest

from svy_io import (
    apply_value_labels,
    as_factor,
    get_column_labels,
    get_value_labels_for_column,
)
from svy_io.helpers import (
    _normalize_n_max,
)


def test_normalize_n_max_variants():
    assert _normalize_n_max(None) is None
    assert _normalize_n_max(0) == 0
    assert _normalize_n_max(5) == 5
    assert _normalize_n_max(-1) is None
    assert _normalize_n_max([3]) == 3
    assert _normalize_n_max((7,)) == 7
    assert _normalize_n_max(True) == 1
    assert _normalize_n_max(False) == 0
    with pytest.raises(TypeError):
        _normalize_n_max("x")  # type: ignore
    with pytest.raises(TypeError):
        _normalize_n_max([1, 2])  # type: ignore


def _fake_meta():
    return {
        "vars": [
            {"name": "gender", "label": None, "label_set": "$GENDER", "fmt": None},
            {
                "name": "q1",
                "label": "The instructor was well prepared",
                "label_set": None,
                "fmt": None,
            },
            {"name": "age", "label": "Age", "label_set": None, "fmt": None},
        ],
        "value_labels": [
            {"set_name": "$GENDER", "mapping": {"f": "Female", "m": "Male"}},
        ],
    }


def test_column_label_helpers():
    meta = _fake_meta()
    labels = get_column_labels(meta)
    assert labels["gender"] is None
    assert labels["q1"] == "The instructor was well prepared"

    mapping = get_value_labels_for_column(meta, "gender")
    assert mapping == {"f": "Female", "m": "Male"}
    assert get_value_labels_for_column(meta, "q1") is None


def test_as_factor_default_and_modes():
    meta = _fake_meta()
    s = pl.Series("gender", ["f", "m", "f", None])
    mapping = get_value_labels_for_column(meta, "gender")

    # default: prefer labels where available, else raw values
    out_def = as_factor(s, labels=mapping, levels="default")
    assert out_def.dtype == pl.Categorical
    assert out_def.to_list()[:3] == ["Female", "Male", "Female"]

    out_labels = as_factor(s, labels=mapping, levels="labels")
    assert out_labels.to_list() == ["Female", "Male", "Female", None]

    out_values = as_factor(s, labels=mapping, levels="values")
    assert out_values.to_list() == ["f", "m", "f", None]

    out_both = as_factor(s, labels=mapping, levels="both")
    assert out_both.to_list()[:2] == ["[f] Female", "[m] Male"]


def test_apply_value_labels_dataframe():
    meta = _fake_meta()
    df = pl.DataFrame(
        {
            "gender": ["f", "m", "m", "f"],
            "q1": [5, 4, 5, 3],
            "age": [30, 40, 35, 50],
        }
    )
    out = apply_value_labels(df, meta, levels="labels", ordered=False)
    assert out["gender"].dtype == pl.Categorical
    assert out["gender"].to_list() == ["Female", "Male", "Male", "Female"]
    # untouched numeric columns remain non-categorical
    assert out["age"].dtype != pl.Categorical
