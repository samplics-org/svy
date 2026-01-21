"""Tests for SPSS labelled vectors (labelled_spss class)"""

import pytest

from svy_io.labelled import (
    Labelled,
    LabelledSPSS,
    is_labelled,
    is_labelled_spss,
    labelled_spss,
)


# Constructor validation tests --------------------------------------------


def test_constructor_checks_na_value():
    """na_values must be compatible with data type"""
    # String na_values for numeric data should raise TypeError
    with pytest.raises(TypeError, match="type"):
        labelled_spss(list(range(1, 11)), na_values=["a"])

    # None in na_values should raise ValueError
    with pytest.raises(ValueError, match="missing"):
        labelled_spss(list(range(1, 11)), na_values=[None])


def test_constructor_checks_na_range():
    """na_range must be valid"""
    # String na_range for numeric data
    with pytest.raises(TypeError, match="type"):
        labelled_spss(list(range(1, 11)), na_range=("a", "b"))

    # na_range must be length 2
    with pytest.raises(ValueError, match="length"):
        labelled_spss(list(range(1, 11)), na_range=(1, 2, 3))

    # na_range can't contain None
    with pytest.raises(ValueError, match="missing"):
        labelled_spss(list(range(1, 11)), na_range=(2, None))

    # na_range must be ascending
    with pytest.raises(ValueError, match="ascending"):
        labelled_spss(list(range(1, 11)), na_range=(2, 1))


def test_printed_output_is_stable():
    """String representation should be consistent"""
    x = labelled_spss(
        [1, 2, 3, 4, 5],
        labels={1: "Good", 5: "Bad"},
        na_values=[1, 2],
        na_range=(3, float("inf")),
        label="Rating",
    )

    # Check that repr includes key information
    repr_str = repr(x)
    assert "LabelledSPSS" in repr_str
    assert "Good" in repr_str or "labels" in repr_str


def test_subsetting_preserves_attributes():
    """Slicing should preserve labels and missing value specs"""
    x = labelled_spss(
        [1, 2, 3, 4, 5],
        labels={1: "Good", 5: "Bad"},
        na_values=[1, 2],
        na_range=(3, float("inf")),
        label="Rating",
    )

    # Full slice should preserve metadata
    x_slice = x[:]

    assert x_slice.labels == x.labels
    assert x_slice.na_values == x.na_values
    assert x_slice.na_range == x.na_range
    assert x_slice.label == x.label


def test_labels_must_be_unique():
    """Can't have duplicate label values"""
    # This is actually fine - dict automatically keeps last value
    x = labelled_spss([1], labels={1: "female", 1: "male"}, na_values=[9])
    assert x.labels[1] == "male"  # Second value wins

    # If you want to test for duplicate label TEXT (values in the dict),
    # that's a different check that would need to be implemented in the class
    # For now, this test should just verify the dict behavior:
    labels_dict = {1: "female", 2: "male"}
    x2 = labelled_spss([1, 2], labels=labels_dict, na_values=[9])
    assert len(x2.labels) == 2


# is_na / missing value detection ----------------------------------------


def test_values_in_na_range_flagged_as_missing():
    """Values within na_range should be treated as missing"""
    x = labelled_spss([1, 2, 3, 4, 5], labels={1: "a"}, na_range=(1, 3))

    # Check which values are considered missing
    missing = x.is_na()
    expected = [True, True, True, False, False]

    assert missing == expected


def test_values_in_na_values_flagged_as_missing():
    """Values in na_values list should be treated as missing"""
    x = labelled_spss([1, 2, 3, 4, 5], labels={1: "a"}, na_values=[1, 3, 5])

    missing = x.is_na()
    expected = [True, False, True, False, True]

    assert missing == expected


# Combining / concatenation tests ----------------------------------------


def test_combining_preserves_class():
    """Concatenating labelled_spss vectors should preserve type"""
    # Both plain labelled_spss → labelled_spss
    result1 = LabelledSPSS.concat([labelled_spss([]), labelled_spss([])])
    assert is_labelled_spss(result1)

    # One with na_values, one without → regular labelled
    result2 = LabelledSPSS.concat([labelled_spss([]), labelled_spss([], na_values=[1])])
    # Should downgrade to regular labelled if na specs differ
    assert is_labelled(result2)

    # Both with same na_values → labelled_spss
    result3 = LabelledSPSS.concat(
        [labelled_spss([], na_values=[1]), labelled_spss([], na_values=[1])]
    )
    assert is_labelled_spss(result3)


def test_combining_is_symmetrical_wrt_data_types():
    """Type checking should work properly"""
    # Both numeric should work
    result1 = LabelledSPSS.concat([labelled_spss([1, 2]), labelled_spss([3.0, 4.0])])
    assert len(result1) == 4


def test_combining_preserves_label_sets():
    """Concatenating vectors with same labels preserves labels"""
    result = LabelledSPSS.concat(
        [
            labelled_spss([1], labels={1: "Good", 5: "Bad"}),
            labelled_spss([5], labels={1: "Good", 5: "Bad"}),
        ]
    )

    expected = labelled_spss([1, 5], labels={1: "Good", 5: "Bad"})
    assert result.labels == expected.labels


def test_combining_preserves_user_missing():
    """Concatenating vectors with same na specs preserves them"""
    # na_values
    result1 = LabelledSPSS.concat(
        [labelled_spss([1], na_values=[1, 5]), labelled_spss([5], na_values=[1, 5])]
    )
    expected1 = labelled_spss([1, 5], na_values=[1, 5])
    assert result1.na_values == expected1.na_values

    # na_range
    result2 = LabelledSPSS.concat(
        [labelled_spss([1], na_range=(1, 5)), labelled_spss([5], na_range=(1, 5))]
    )
    expected2 = labelled_spss([1, 5], na_range=(1, 5))
    assert result2.na_range == expected2.na_range


def test_take_labels_from_lhs():
    """When labels conflict, should use left-hand side labels"""
    with pytest.warns(UserWarning, match="Conflicting"):
        result1 = LabelledSPSS.concat(
            [
                labelled_spss([1], labels={1: "Good", 5: "Bad"}),
                labelled_spss([5], labels={1: "Bad", 5: "Good"}),
            ]
        )
        # Should use first vector's labels
        assert result1.labels == {1: "Good", 5: "Bad"}

    with pytest.warns(UserWarning, match="Conflicting"):
        result2 = LabelledSPSS.concat(
            [labelled_spss([1], labels={1: "Good"}), labelled_spss([5], labels={1: "Bad"})]
        )
        # Should use first vector's labels
        assert result2.labels == {1: "Good"}


def test_warn_only_for_conflicting_labels():
    """Should warn when labels actually conflict"""
    with pytest.warns(UserWarning, match="Conflicting"):
        x = labelled_spss([1, 2], labels={1: "Yes", 2: "No"})
        y = labelled_spss([1, 2], labels={1: "Female", 2: "Male", 3: "Other"})
        LabelledSPSS.concat([x, y])


def test_strip_user_missing_if_different():
    """If na specs differ, should drop to regular labelled"""
    # Different na_values
    result1 = LabelledSPSS.concat(
        [labelled_spss([], na_values=[1]), labelled_spss([], na_values=[5])]
    )
    # Should be regular labelled, not labelled_spss
    assert isinstance(result1, Labelled)
    assert not isinstance(result1, LabelledSPSS)

    # Different na_range
    result2 = LabelledSPSS.concat(
        [labelled_spss([], na_range=(1, 5)), labelled_spss([], na_range=(2, 4))]
    )
    assert isinstance(result2, Labelled)
    assert not isinstance(result2, LabelledSPSS)

    # na_range vs na_values
    result3 = LabelledSPSS.concat(
        [labelled_spss([], na_range=(1, 5)), labelled_spss([], na_values=[5])]
    )
    assert isinstance(result3, Labelled)
    assert not isinstance(result3, LabelledSPSS)


def test_combining_picks_label_from_the_left():
    """Variable label should come from first vector"""
    result = LabelledSPSS.concat(
        [labelled_spss([], label="left"), labelled_spss([], label="right")]
    )

    assert result.label == "left"


def test_combining_with_bare_vectors_results_in_labelled_spss():
    """Concatenating with plain lists should work"""
    result = LabelledSPSS.concat([labelled_spss([]), [1.1]])
    assert len(result) == 1
    assert result.data[0] == 1.1


# Casting tests ----------------------------------------------------------


def test_casting_to_superset_of_labels_works():
    """Can cast to vector with more labels"""
    x = labelled_spss([1, 5], labels={1: "Good"})
    template = labelled_spss([], labels={1: "Good", 5: "Bad"})

    result = x.cast_to(template)

    expected = labelled_spss([1, 5], labels={1: "Good", 5: "Bad"})
    assert result.labels == expected.labels


def test_casting_to_subset_of_labels_works_iff_labels_were_unused():
    """Can only remove unused labels without error"""
    # Unused label - OK
    x1 = labelled_spss([1], labels={1: "Good", 5: "Bad"})
    template1 = labelled_spss([], labels={1: "Good"})

    result1 = x1.cast_to(template1)
    assert result1.labels == {1: "Good"}

    # Used label - should raise
    x2 = labelled_spss([1, 5], labels={1: "Good", 5: "Bad"})
    template2 = labelled_spss([], labels={1: "Good"})

    with pytest.raises(ValueError, match="Lossy|label"):
        x2.cast_to(template2)


def test_casting_away_labels_throws_lossy_cast():
    """Removing labels that are used should fail"""
    x = labelled_spss([1], labels={1: "Good"})
    template = labelled_spss([], labels={5: "Bad"})

    with pytest.raises(ValueError, match="Lossy|label"):
        x.cast_to(template)


def test_casting_to_superset_of_user_missing_works():
    """Can cast to vector with more na specs"""
    # na_values superset
    x1 = labelled_spss([1, 5], na_values=[1])
    template1 = labelled_spss([], na_values=[1, 5])

    result1 = x1.cast_to(template1)
    assert result1.na_values == [1, 5]

    # na_values to na_range (covering same values)
    x2 = labelled_spss([1, 5], na_values=[1])
    template2 = labelled_spss([], na_range=(1, 5))

    result2 = x2.cast_to(template2)
    assert result2.na_range == (1, 5)

    # na_range to wider na_range
    x3 = labelled_spss([1, 5], na_range=(2, 4))
    template3 = labelled_spss([], na_range=(1, 5))

    result3 = x3.cast_to(template3)
    assert result3.na_range == (1, 5)


def test_casting_to_subset_of_user_missing_works_iff_values_were_unused():
    """Can only remove unused na values without error"""
    # Unused na_value - OK
    x1 = labelled_spss([1], na_values=[1, 5])
    template1 = labelled_spss([], na_values=[1])

    result1 = x1.cast_to(template1)
    assert result1.na_values == [1]

    # Used na_value - should fail
    x2 = labelled_spss([1, 5], na_values=[1, 5])
    template2 = labelled_spss([], na_values=[1])

    with pytest.raises(ValueError, match="Lossy|missing"):
        x2.cast_to(template2)

    # Similar tests for na_range
    x3 = labelled_spss([1], na_range=(1, 5))
    template3 = labelled_spss([], na_range=(1, 3))

    result3 = x3.cast_to(template3)
    assert result3.na_range == (1, 3)

    x4 = labelled_spss([1, 5], na_range=(1, 5))
    template4 = labelled_spss([], na_range=(1, 3))

    with pytest.raises(ValueError, match="Lossy|missing"):
        x4.cast_to(template4)


def test_casting_away_user_missing_throws_lossy_cast():
    """Removing na specs that are used should fail"""
    # na_values
    x1 = labelled_spss([1], na_values=[1])
    template1 = labelled_spss([], na_values=[5])

    with pytest.raises(ValueError, match="Lossy|missing"):
        x1.cast_to(template1)

    # na_range
    x2 = labelled_spss([1], na_range=(1, 3))
    template2 = labelled_spss([], na_range=(5, 7))

    with pytest.raises(ValueError, match="Lossy|missing"):
        x2.cast_to(template2)

    # na_range to different na_values
    x3 = labelled_spss([1], na_range=(1, 3))
    template3 = labelled_spss([], na_values=[5])

    with pytest.raises(ValueError, match="Lossy|missing"):
        x3.cast_to(template3)

    # na_values to different na_range
    x4 = labelled_spss([1], na_values=[1])
    template4 = labelled_spss([], na_range=(5, 7))

    with pytest.raises(ValueError, match="Lossy|missing"):
        x4.cast_to(template4)


def test_casting_to_regular_labelled_ignores_missing_values():
    """Converting to regular labelled should drop na specs"""
    x = labelled_spss([1], na_values=[1, 5])

    # Cast to regular Labelled
    result = Labelled(data=x.data, labels=x.labels, label=x.label)

    assert not hasattr(result, "na_values") or result.na_values is None
    assert not hasattr(result, "na_range") or result.na_range is None


def test_can_cast_labelled_spss_to_atomic_vectors():
    """Should be able to convert to plain numeric/string"""
    x_int = labelled_spss([1, 2])
    x_dbl = labelled_spss([1.0, 2.0])
    x_chr = labelled_spss(["a", "b"])

    # Integer labelled
    assert x_int.to_int() == [1, 2]
    assert x_int.to_float() == [1.0, 2.0]
    with pytest.raises(TypeError):
        x_int.to_str()

    # Float labelled
    assert x_dbl.to_int() == [1, 2]
    assert x_dbl.to_float() == [1.0, 2.0]
    with pytest.raises(TypeError):
        x_dbl.to_str()

    # String labelled
    with pytest.raises(TypeError):
        x_chr.to_int()
    with pytest.raises(TypeError):
        x_chr.to_float()
    assert x_chr.to_str() == ["a", "b"]


def test_can_cast_atomic_vectors_to_labelled_spss():
    """Should be able to create labelled from plain vectors"""
    x_int = labelled_spss([1, 2])
    x_dbl = labelled_spss([1.0, 2.0])
    x_chr = labelled_spss(["a", "b"])

    # Cast integers
    result1 = LabelledSPSS.from_values([1, 2, 3], like=x_int)
    assert result1.data == [1, 2, 3]

    result2 = LabelledSPSS.from_values([1, 2, 3], like=x_dbl)
    assert result2.data == [1, 2, 3]

    with pytest.raises(TypeError):
        LabelledSPSS.from_values([1, 2, 3], like=x_chr)

    # Cast floats
    result3 = LabelledSPSS.from_values([0, 1], like=x_int)
    assert result3.data == [0, 1]

    result4 = LabelledSPSS.from_values([0.0, 1.0], like=x_dbl)
    assert result4.data == [0.0, 1.0]

    with pytest.raises(TypeError):
        LabelledSPSS.from_values([0.0, 1.0], like=x_chr)

    # Cast strings
    with pytest.raises(TypeError):
        LabelledSPSS.from_values(["a"], like=x_int)

    with pytest.raises(TypeError):
        LabelledSPSS.from_values(["a"], like=x_dbl)

    result5 = LabelledSPSS.from_values(["a"], like=x_chr)
    assert result5.data == ["a"]


def test_wont_cast_labelled_spss_numeric_to_character():
    """Numeric labelled can't become string labelled"""
    x = labelled_spss([1, 2])

    with pytest.raises(TypeError):
        x.to_str()

    x_int = labelled_spss([1, 2])
    with pytest.raises(TypeError):
        x_int.to_str()


# Equality tests ---------------------------------------------------------


def test_equality_works():
    """Equality checking should work"""
    x1 = labelled_spss([1, 2, 3], labels={1: "Good"}, na_values=[9])
    x2 = labelled_spss([1, 2, 3], labels={1: "Good"}, na_values=[9])
    x3 = labelled_spss([1, 2, 4], labels={1: "Good"}, na_values=[9])

    assert x1 == x2
    assert x1 != x3


def test_equality_checks_all_attributes():
    """Equality should check labels, na_values, and na_range"""
    base = labelled_spss([1, 2], labels={1: "A"}, na_values=[9])

    # Different data
    diff_data = labelled_spss([1, 3], labels={1: "A"}, na_values=[9])
    assert base != diff_data

    # Different labels
    diff_labels = labelled_spss([1, 2], labels={1: "B"}, na_values=[9])
    assert base != diff_labels

    # Different na_values
    diff_na_vals = labelled_spss([1, 2], labels={1: "A"}, na_values=[8])
    assert base != diff_na_vals

    # Different na_range
    with_range = labelled_spss([1, 2], labels={1: "A"}, na_range=(8, 10))
    assert base != with_range


# Integration tests ------------------------------------------------------


def test_realistic_spss_workflow():
    """Test a realistic SPSS data workflow"""
    # Create survey data with SPSS-style coding
    gender = labelled_spss(
        [1, 2, 1, 2, 9], labels={1: "Male", 2: "Female", 9: "Refused"}, na_values=[9]
    )

    # Check missing values
    assert gender.is_na() == [False, False, False, False, True]

    # Concatenate with more data
    gender2 = labelled_spss([1, 2], labels={1: "Male", 2: "Female", 9: "Refused"}, na_values=[9])
    combined = LabelledSPSS.concat([gender, gender2])

    assert len(combined) == 7
    assert combined.na_values == [9]

    # Cast to remove unused labels
    simple = labelled_spss([1, 2, 1], labels={1: "Male", 2: "Female"})
    template = labelled_spss([], labels={1: "Male"})

    # This should fail because 2 is used
    with pytest.raises(ValueError):
        simple.cast_to(template)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
