# tests/svy/weighting/test_wgt_aux_build.py
import polars as pl
import pytest

from svy.core.sample import Sample
from svy.core.terms import Cat, Cross
from svy.errors import MethodError


def _make_sample():
    # Categorical-like (strings); continuous numeric
    df = pl.DataFrame(
        {
            "cat1": ["A", "B", "A", None, "B"],
            "cat2": ["X", "X", "Y", "Y", None],
            "g": ["u", "u", "v", "v", "v"],  # simple domain column
            "h": [1, 1, 1, 2, 2],  # numeric-ish domain col
            "x": [1.0, 2.0, 3.0, 4.0, None],  # continuous
        }
    )
    return Sample(df)


# ----------------------------- controls_template ----------------------------- #


def test_controls_template_accepts_cat_term():
    """Verify that terms=[Cat('var')] works correctly."""
    s = _make_sample()
    # Using new terms API
    a = s.weighting.control_aux_template(x=[Cat("cat1")], na_label="__NA__")

    # Check structure
    labels = list(a.keys())
    # Cat term extracts ALL unique levels found in data, including Nulls.
    assert set(labels) == {"A", "B", None}


def test_controls_template_by_normalization_equivalence():
    """Verify different ways of passing 'by' (single term vs explicit Term object)."""
    s = _make_sample()
    # Case A: by="g" (string)
    a = s.weighting.control_aux_template(x=[Cat("cat1")], by="g")

    # Case B: by="g" semantic equivalent
    b = s.weighting.control_aux_template(x=[Cat("cat1")], by="g")

    # Case C: by=("g", "h") - Multi-column check
    c = s.weighting.control_aux_template(x=[Cat("cat1")], by=("g", "h"))

    assert set(a.keys()) == {"u", "v"}
    assert set(b.keys()) == {"u", "v"}
    # C keys will be tuples like ('u', '1')
    assert isinstance(list(c.keys())[0], tuple)


# ----------------------------- build_aux_matrix ------------------------------ #


def test_build_aux_matrix_cat_only_single_var_labels_and_matrix_shape():
    s = _make_sample()
    # Use Cat term.
    X, shape = s.weighting.build_aux_matrix(x=[Cat("cat1")], by=None)

    labels = list(shape.keys())
    assert set(labels) == {"A", "B", None}

    # X has 3 columns (A, B, None)
    assert X.ndim == 2 and X.shape[1] == 3
    # Rows match df height
    assert X.shape[0] == s.data.height


def test_build_aux_matrix_multi_categorical_produces_tuple_labels():
    """
    Test interaction via Cross(Cat, Cat).
    """
    s = _make_sample()
    # Cross terms explicitly
    term = Cross(Cat("cat1"), Cat("cat2"))

    X, shape = s.weighting.build_aux_matrix(x=[term], by=None)
    labels = list(shape.keys())

    # Expect tuple labels covering (A,X), (A,Y), (B,X), (B,Y) etc.
    assert all(isinstance(lab, tuple) for lab in labels)

    # Check observed combos exist
    assert ("A", "X") in labels
    assert ("B", "X") in labels
    assert ("A", "Y") in labels
    assert (None, "Y") in labels
    assert ("B", None) in labels

    # One-hot columns = number of labels
    assert X.shape[1] == len(labels)
    assert X.shape[0] == s.data.height


def test_build_aux_matrix_with_by_single_and_multi_keys():
    s = _make_sample()

    # 1. by="g" (Single column)
    X1, shape1 = s.weighting.build_aux_matrix(x=[Cat("cat1")], by="g", by_na="level")
    by_keys1 = set(shape1.keys())
    assert by_keys1 == {"u", "v"}
    for _, inner in shape1.items():
        assert isinstance(inner, dict)
        # Inner labels (cat1 levels)
        assert set(inner.keys()) == {"A", "B", None}

    # 2. by=("g", "h") (Multi-column tuple keys)
    X2, shape2 = s.weighting.build_aux_matrix(
        x=[Cat("cat1")],
        by=("g", "h"),
        by_na="level",
    )
    by_keys2 = set(shape2.keys())

    # Normalize to strings/tuples for comparison
    def _skey(k):
        return (str(k[0]), str(k[1])) if isinstance(k, tuple) else str(k)

    by_keys2_str = {_skey(k) for k in by_keys2}
    # Expected combos in data: (u,1), (v,1), (v,2)
    assert {("u", "1"), ("v", "1"), ("v", "2")} <= by_keys2_str

    n = s.data.height
    assert X1.shape[0] == n and X2.shape[0] == n


def test_build_aux_matrix_continuous_na_handling():
    s = _make_sample()
    # 'x' has a None. Default behavior for continuous terms is fill_null(0.0).
    X, shape = s.weighting.build_aux_matrix(x=["x"])

    # one continuous column -> 1 feature
    assert X.shape[1] == 1
    # shape contains one label (the column name)
    assert list(shape.keys()) == ["x"]

    # Verify fillna behavior: last row is None, should be 0.0
    import numpy as np

    assert np.isclose(X[-1, 0], 0.0)


def test_build_aux_matrix_requires_some_output_columns():
    s = _make_sample()

    # Case 1: Empty list
    with pytest.raises(MethodError, match="No terms specified"):
        s.weighting.build_aux_matrix(x=[])

    # Case 2: None
    # This matches the "if not terms:" check in build_aux_matrix
    with pytest.raises(MethodError, match="No terms specified"):
        s.weighting.build_aux_matrix(x=None)
