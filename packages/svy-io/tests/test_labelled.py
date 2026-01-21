from __future__ import annotations

import pytest

from svy_io.labelled import Labelled, labelled


# ---- constructors / validation ----


def test_labelled_zero_length_vector():
    x = labelled()
    assert isinstance(x, Labelled)
    assert len(x) == 0


def test_x_must_be_numeric_or_character():
    with pytest.raises(TypeError):
        _ = labelled([True, False])  # bools -> not allowed


def test_x_and_labels_must_be_compatible():
    # incompatible types
    with pytest.raises(TypeError):
        _ = labelled([1], labels={"a": "female"})  # keys must be numeric if x numeric

    # numeric with numeric labels -> ok
    _ = labelled([1], labels={2: "female", 1: "male"})
    _ = labelled([1], labels={2.0: "female", 1.0: "male"})


def test_labels_can_be_none():
    x = labelled([1, 2, 3], labels=None)
    assert x.as_list() == [1, 2, 3]


def test_labels_must_have_names():
    # In our API, names are the dict values (label strings)
    with pytest.raises(TypeError):
        _ = labelled([1], labels={1: 1})  # label must be str


def test_label_must_be_length1_string_or_missing():
    _ = labelled([1], labels={1: "female"})  # ok
    _ = labelled([1], labels={1: "female"}, label="foo")  # ok
    with pytest.raises(TypeError):
        _ = labelled([1], labels={1: "female"}, label=1)  # not str
    # multiple-length label not a concept here; any str is fine


def test_labels_must_be_unique():
    def test_labels_must_be_unique():
        # Under Python's value->label mapping, we can't create duplicate keys,
        # so we assert that duplicate *label strings* are rejected.
        with pytest.raises(ValueError):
            labelled(1, {1: "female", 2: "female"})


# ---- basic api / methods ----


def test_as_character_and_levels():
    x = labelled([1, 2, 3], labels={1: "x", 2: "y"})
    assert x.as_character() == ["1", "2", "3"]
    assert x.levels() is None


def test_arithmetic_strips_class():
    xi = labelled([1])
    xd = labelled([2.0])

    with pytest.raises(TypeError):
        _ = xi + "x"

    assert (xi + xd) == [3.0]
    assert (xi + 1) == [2.0]
    assert (1 + xi) == [2.0]

    # sum equivalent
    assert sum(xi.as_list()) == 1


# ---- “methods” parity ----


def test_median_quantile_summary_numeric():
    x = labelled([1, 2, 3])
    assert x.median() == 2.0
    assert x.quantile(0.25) == 1.5
    s = x.summary()
    assert isinstance(s, dict)
    assert s["median"] == 2.0


def test_median_quantile_error_on_character():
    x = labelled(["a", "b", "c"])
    with pytest.raises(TypeError):
        _ = x.median()
    with pytest.raises(TypeError):
        _ = x.quantile(0.25)
    # summary works but is different
    s = x.summary()
    assert s["length"] == 3


# ---- combining / casting: out of scope (vctrs) ----


@pytest.mark.xfail(reason="vctrs-style casting/combining not implemented in Python")
def test_vec_cast_vec_c_behaviors():
    pass


# ---- factor conversion integration (optional sanity check) ----


def test_as_factor_labels_roundtrip_with_existing_helper():
    # This test assumes your as_factor(series, value_labels=...) helper exists
    import polars as pl

    from svy_io import as_factor

    x = labelled([1, None, 5], labels={1: "Good", 5: "Bad"})
    s = pl.Series(x.as_list())
    cat = as_factor(s, labels={str(k): v for k, v in x.labels.items()}, levels="labels")
    # labels-only -> unlabelled values become nulls
    assert cat.dtype == pl.Categorical
    assert cat.to_list() == ["Good", None, "Bad"]
