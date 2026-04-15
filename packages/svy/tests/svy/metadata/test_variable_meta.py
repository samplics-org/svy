# tests/test_variable_meta.py
"""
Tests for the unified variable metadata system.
"""

import polars as pl
import pytest

from svy.core.enumerations import MeasurementType, MetadataSource, MissingKind
from svy.metadata import MetadataStore, MissingDef, ResolvedLabels, SchemeRef, VariableMeta


class TestMissingDef:
    """Tests for MissingDef class."""

    def test_create_empty(self):
        """Empty MissingDef should work."""
        m = MissingDef()
        assert m.codes == frozenset()
        assert m.kinds is None
        assert m.na_is_missing is True
        assert m.nan_is_missing is True

    def test_create_with_codes(self):
        """Create with simple codes."""
        m = MissingDef(codes=frozenset([-99, -98, -97]))
        assert -99 in m.codes
        assert -98 in m.codes
        assert -97 in m.codes
        assert len(m.codes) == 3

    def test_create_with_kinds(self):
        """Create with codes and kinds."""
        m = MissingDef(
            codes=frozenset([-99, -98]),
            kinds={
                -99: MissingKind.DONT_KNOW,
                -98: MissingKind.REFUSED,
            },
        )
        assert m.kinds[-99] == MissingKind.DONT_KNOW
        assert m.kinds[-98] == MissingKind.REFUSED

    def test_kinds_must_be_subset_of_codes(self):
        """Kinds keys must be in codes."""
        with pytest.raises(ValueError, match="not in codes"):
            MissingDef(
                codes=frozenset([-99]),
                kinds={-99: MissingKind.DONT_KNOW, -98: MissingKind.REFUSED},
            )

    def test_is_missing_with_none(self):
        """None should be missing when na_is_missing=True."""
        m = MissingDef(codes=frozenset([-99]))
        assert m.is_missing(None) is True

        m2 = MissingDef(codes=frozenset([-99]), na_is_missing=False)
        assert m2.is_missing(None) is False

    def test_is_missing_with_nan(self):
        """NaN should be missing when nan_is_missing=True."""
        m = MissingDef(codes=frozenset([-99]))
        assert m.is_missing(float("nan")) is True

        m2 = MissingDef(codes=frozenset([-99]), nan_is_missing=False)
        assert m2.is_missing(float("nan")) is False

    def test_is_missing_with_code(self):
        """Codes in the set should be missing."""
        m = MissingDef(codes=frozenset([-99, -98]))
        assert m.is_missing(-99) is True
        assert m.is_missing(-98) is True
        assert m.is_missing(1) is False
        assert m.is_missing(0) is False

    def test_is_missing_by_kind(self):
        """Test filtering by kind."""
        m = MissingDef(
            codes=frozenset([-99, -98, -97]),
            kinds={
                -99: MissingKind.DONT_KNOW,
                -98: MissingKind.REFUSED,
                -97: MissingKind.STRUCTURAL,
            },
        )
        assert m.is_missing_by_kind(-99, MissingKind.DONT_KNOW) is True
        assert m.is_missing_by_kind(-99, MissingKind.REFUSED) is False
        assert m.is_missing_by_kind(-98, MissingKind.DONT_KNOW, MissingKind.REFUSED) is True

    def test_by_kind(self):
        """Get codes by kind."""
        m = MissingDef(
            codes=frozenset([-99, -98, -97, -96]),
            kinds={
                -99: MissingKind.DONT_KNOW,
                -98: MissingKind.REFUSED,
                -97: MissingKind.STRUCTURAL,
                -96: MissingKind.SYSTEM,
            },
        )
        user_missing = m.by_kind(MissingKind.DONT_KNOW, MissingKind.REFUSED)
        assert user_missing == frozenset([-99, -98])

    def test_user_missing(self):
        """Get user-generated missing codes."""
        m = MissingDef(
            codes=frozenset([-99, -98, -97]),
            kinds={
                -99: MissingKind.DONT_KNOW,
                -98: MissingKind.REFUSED,
                -97: MissingKind.STRUCTURAL,
            },
        )
        # User missing includes DONT_KNOW, REFUSED, NO_ANSWER
        assert m.user_missing() == frozenset([-99, -98])

    def test_system_missing(self):
        """Get system-generated missing codes."""
        m = MissingDef(
            codes=frozenset([-99, -98, -97]),
            kinds={
                -99: MissingKind.DONT_KNOW,
                -98: MissingKind.REFUSED,
                -97: MissingKind.STRUCTURAL,
            },
        )
        # System missing includes SYSTEM, STRUCTURAL
        assert m.system_missing() == frozenset([-97])

    def test_from_codes_factory(self):
        """Test from_codes factory method."""
        m = MissingDef.from_codes([-99, -98])
        assert m.codes == frozenset([-99, -98])
        assert m.kinds is None

    def test_from_kinds_factory(self):
        """Test from_kinds factory method."""
        m = MissingDef.from_kinds(
            {
                -99: MissingKind.DONT_KNOW,
                -98: MissingKind.REFUSED,
            }
        )
        assert m.codes == frozenset([-99, -98])
        assert m.kinds is not None
        assert m.kinds[-99] == MissingKind.DONT_KNOW

    def test_equality(self):
        """Test equality comparison."""
        m1 = MissingDef(codes=frozenset([-99, -98]))
        m2 = MissingDef(codes=frozenset([-99, -98]))
        m3 = MissingDef(codes=frozenset([-99]))

        assert m1 == m2
        assert m1 != m3

    def test_clone(self):
        """Test cloning with overrides."""
        m1 = MissingDef(codes=frozenset([-99]))
        m2 = m1.clone(na_is_missing=False)

        assert m1.na_is_missing is True
        assert m2.na_is_missing is False
        assert m1.codes == m2.codes


class TestSchemeRef:
    """Tests for SchemeRef class."""

    def test_create(self):
        """Create a scheme reference."""
        ref = SchemeRef(concept="agreement")
        assert ref.concept == "agreement"
        assert ref.locale is None

    def test_create_with_locale(self):
        """Create with explicit locale."""
        ref = SchemeRef(concept="agreement", locale="en-US")
        assert ref.concept == "agreement"
        assert ref.locale == "en-US"

    def test_immutable(self):
        """SchemeRef should be immutable."""
        ref = SchemeRef(concept="test")
        with pytest.raises(AttributeError):
            ref.concept = "other"


class TestVariableMeta:
    """Tests for VariableMeta class."""

    def test_create_minimal(self):
        """Create with just a name."""
        meta = VariableMeta(name="q1")
        assert meta.name == "q1"
        assert meta.label is None
        assert meta.value_labels is None
        assert meta.scheme_ref is None
        assert meta.mtype == MeasurementType.STRING
        assert meta.source == MetadataSource.INFERRED

    def test_create_with_labels(self):
        """Create with direct value labels."""
        meta = VariableMeta(
            name="gender",
            label="What is your gender?",
            value_labels={1: "Male", 2: "Female", 3: "Other"},
            mtype=MeasurementType.NOMINAL,
        )
        assert meta.label == "What is your gender?"
        assert meta.value_labels[1] == "Male"
        assert meta.has_labels is True

    def test_create_with_scheme_ref(self):
        """Create with scheme reference."""
        meta = VariableMeta(
            name="q1",
            label="How satisfied are you?",
            scheme_ref=SchemeRef(concept="satisfaction"),
            mtype=MeasurementType.ORDINAL,
        )
        assert meta.scheme_ref.concept == "satisfaction"
        assert meta.has_labels is True

    def test_cannot_have_both_labels_and_scheme(self):
        """Can't specify both value_labels and scheme_ref."""
        with pytest.raises(ValueError, match="cannot specify both"):
            VariableMeta(
                name="q1",
                value_labels={1: "Yes", 0: "No"},
                scheme_ref=SchemeRef(concept="yes_no"),
            )

    def test_has_labels(self):
        """Test has_labels property."""
        m1 = VariableMeta(name="q1")
        assert m1.has_labels is False

        m2 = VariableMeta(name="q1", value_labels={1: "Yes"})
        assert m2.has_labels is True

        m3 = VariableMeta(name="q1", scheme_ref=SchemeRef(concept="yes_no"))
        assert m3.has_labels is True

    def test_has_missing(self):
        """Test has_missing property."""
        m1 = VariableMeta(name="q1")
        assert m1.has_missing is False

        m2 = VariableMeta(name="q1", missing=MissingDef(codes=frozenset([-99])))
        assert m2.has_missing is True

        m3 = VariableMeta(name="q1", missing=MissingDef())
        assert m3.has_missing is False  # Empty codes

    def test_is_categorical(self):
        """Test is_categorical property."""
        assert VariableMeta(name="q1", mtype=MeasurementType.NOMINAL).is_categorical
        assert VariableMeta(name="q1", mtype=MeasurementType.ORDINAL).is_categorical
        assert VariableMeta(name="q1", mtype=MeasurementType.BOOLEAN).is_categorical
        assert not VariableMeta(name="q1", mtype=MeasurementType.CONTINUOUS).is_categorical
        assert not VariableMeta(name="q1", mtype=MeasurementType.STRING).is_categorical

    def test_is_numeric(self):
        """Test is_numeric property."""
        assert VariableMeta(name="q1", mtype=MeasurementType.CONTINUOUS).is_numeric
        assert VariableMeta(name="q1", mtype=MeasurementType.DISCRETE).is_numeric
        assert not VariableMeta(name="q1", mtype=MeasurementType.NOMINAL).is_numeric

    def test_is_ordered(self):
        """Test is_ordered property."""
        assert VariableMeta(name="q1", mtype=MeasurementType.ORDINAL).is_ordered
        assert not VariableMeta(name="q1", mtype=MeasurementType.NOMINAL).is_ordered

    def test_with_label(self):
        """Test with_label method."""
        m1 = VariableMeta(name="q1")
        m2 = m1.with_label("New label")

        assert m1.label is None
        assert m2.label == "New label"
        assert m2.source == MetadataSource.USER

    def test_with_value_labels(self):
        """Test with_value_labels method."""
        m1 = VariableMeta(name="q1", scheme_ref=SchemeRef(concept="test"))
        m2 = m1.with_value_labels({1: "Yes", 0: "No"})

        assert m1.scheme_ref is not None
        assert m2.scheme_ref is None  # Cleared by default
        assert m2.value_labels == {1: "Yes", 0: "No"}

    def test_with_scheme_ref(self):
        """Test with_scheme_ref method."""
        m1 = VariableMeta(name="q1", value_labels={1: "Yes"})
        m2 = m1.with_scheme_ref("yes_no", locale="en")

        assert m1.value_labels is not None
        assert m2.value_labels is None  # Cleared by default
        assert m2.scheme_ref.concept == "yes_no"

    def test_with_missing(self):
        """Test with_missing method."""
        m1 = VariableMeta(name="q1")
        missing = MissingDef.from_codes([-99])
        m2 = m1.with_missing(missing)

        assert m1.missing is None
        assert m2.missing is missing

    def test_with_categories(self):
        """Test with_categories method."""
        m1 = VariableMeta(name="q1", mtype=MeasurementType.NOMINAL)
        m2 = m1.with_categories([1, 2, 3], ordered=True)

        assert m1.categories is None
        assert m2.categories == (1, 2, 3)
        assert m2.mtype == MeasurementType.ORDINAL


class TestResolvedLabels:
    """Tests for ResolvedLabels class."""

    def test_create_empty(self):
        """Create empty resolved labels."""
        r = ResolvedLabels()
        assert r.var_label == ""
        assert r.value_labels == {}
        assert r.missing_codes == frozenset()

    def test_create_with_labels(self):
        """Create with labels."""
        r = ResolvedLabels(
            var_label="How satisfied are you?",
            value_labels={1: "Very dissatisfied", 5: "Very satisfied"},
            missing_codes=frozenset([-99]),
        )
        assert r.var_label == "How satisfied are you?"
        assert r.value_labels[1] == "Very dissatisfied"
        assert -99 in r.missing_codes

    def test_display_with_label(self):
        """Display a value with a label."""
        r = ResolvedLabels(value_labels={1: "Yes", 0: "No"})
        assert r.display(1) == "Yes"
        assert r.display(0) == "No"

    def test_display_without_label(self):
        """Display a value without a label falls back to str."""
        r = ResolvedLabels(value_labels={1: "Yes"})
        assert r.display(2) == "2"
        assert r.display(999) == "999"

    def test_display_none(self):
        """Display None returns null_text."""
        r = ResolvedLabels(value_labels={1: "Yes"})
        assert r.display(None) == ""
        assert r.display(None, null_text="N/A") == "N/A"

    def test_display_nan(self):
        """Display NaN returns null_text."""
        r = ResolvedLabels(value_labels={1: "Yes"})
        assert r.display(float("nan")) == ""
        assert r.display(float("nan"), null_text="N/A") == "N/A"

    def test_has_var_label(self):
        """Test has_var_label property."""
        assert ResolvedLabels(var_label="Test").has_var_label is True
        assert ResolvedLabels(var_label="").has_var_label is False
        assert ResolvedLabels().has_var_label is False

    def test_has_value_labels(self):
        """Test has_value_labels property."""
        assert ResolvedLabels(value_labels={1: "Yes"}).has_value_labels is True
        assert ResolvedLabels(value_labels={}).has_value_labels is False
        assert ResolvedLabels().has_value_labels is False

    def test_is_missing(self):
        """Test is_missing method."""
        r = ResolvedLabels(missing_codes=frozenset([-99, -98]))
        assert r.is_missing(-99) is True
        assert r.is_missing(-98) is True
        assert r.is_missing(1) is False
        assert r.is_missing(None) is True

    def test_non_missing_labels(self):
        """Get labels excluding missing codes."""
        r = ResolvedLabels(
            value_labels={1: "Yes", 0: "No", -99: "Don't know"},
            missing_codes=frozenset([-99]),
        )
        non_missing = r.non_missing_labels()
        assert non_missing == {1: "Yes", 0: "No"}
        assert -99 not in non_missing

    def test_display_series(self):
        """Apply labels to a Polars series."""
        r = ResolvedLabels(value_labels={1: "Yes", 0: "No"})
        s = pl.Series("test", [1, 0, 1, None, 0])
        result = r.display_series(s)

        assert result.to_list() == ["Yes", "No", "Yes", "", "No"]

    def test_display_series_unmapped(self):
        """Unmapped values become strings."""
        r = ResolvedLabels(value_labels={1: "Yes", 0: "No"})
        s = pl.Series("test", [1, 0, 2, 3])
        result = r.display_series(s)

        assert result.to_list() == ["Yes", "No", "2", "3"]


class TestMetadataStore:
    """Tests for MetadataStore class."""

    def test_create_empty(self):
        """Create empty store."""
        store = MetadataStore()
        assert len(store) == 0
        assert store.catalog is None
        assert store.locale is None

    def test_set_and_get(self):
        """Set and retrieve metadata."""
        store = MetadataStore()
        meta = VariableMeta(name="q1", label="Test question")
        store.set("q1", meta)

        assert store.get("q1") is not None
        assert store.get("q1").label == "Test question"
        assert "q1" in store

    def test_get_nonexistent(self):
        """Get nonexistent returns None."""
        store = MetadataStore()
        assert store.get("q1") is None

    def test_require_raises(self):
        """Require raises for nonexistent."""
        store = MetadataStore()
        with pytest.raises(KeyError):
            store.require("q1")

    def test_remove(self):
        """Remove metadata."""
        store = MetadataStore()
        store.set("q1", VariableMeta(name="q1"))
        assert "q1" in store

        removed = store.remove("q1")
        assert removed is not None
        assert "q1" not in store

    def test_set_label(self):
        """Set variable label convenience method."""
        store = MetadataStore()
        store.set_label("q1", "What is your age?")

        assert store.get("q1") is not None
        assert store.get("q1").label == "What is your age?"
        assert store.get("q1").source == MetadataSource.USER

    def test_set_labels(self):
        """Set multiple labels."""
        store = MetadataStore()
        store.set_labels(
            q1="Question 1",
            q2="Question 2",
            q3="Question 3",
        )

        assert store.get("q1").label == "Question 1"
        assert store.get("q2").label == "Question 2"
        assert store.get("q3").label == "Question 3"

    def test_set_value_labels(self):
        """Set value labels convenience method."""
        store = MetadataStore()
        store.set_value_labels("gender", {1: "Male", 2: "Female"})

        assert store.get("gender").value_labels == {1: "Male", 2: "Female"}

    def test_set_scheme(self):
        """Set scheme reference."""
        store = MetadataStore()
        store.set_scheme("q1", "agreement", locale="en")

        meta = store.get("q1")
        assert meta.scheme_ref is not None
        assert meta.scheme_ref.concept == "agreement"
        assert meta.scheme_ref.locale == "en"

    def test_set_missing_with_user_friendly_names(self):
        """Set missing values using user-friendly parameter names."""
        store = MetadataStore()
        store.set_missing(
            "q1",
            dont_know=[-99],
            refused=[-98],
            skipped=[-97],  # maps to STRUCTURAL
            not_applicable=[-96],  # maps to STRUCTURAL
        )

        meta = store.get("q1")
        assert meta.missing is not None
        assert -99 in meta.missing.codes
        assert -98 in meta.missing.codes
        assert -97 in meta.missing.codes
        assert -96 in meta.missing.codes

        # Check mechanism mapping
        assert meta.missing.kinds[-99] == MissingKind.DONT_KNOW
        assert meta.missing.kinds[-98] == MissingKind.REFUSED
        assert meta.missing.kinds[-97] == MissingKind.STRUCTURAL  # skipped -> STRUCTURAL
        assert meta.missing.kinds[-96] == MissingKind.STRUCTURAL  # not_applicable -> STRUCTURAL

    def test_set_missing_simple_codes(self):
        """Set missing with simple codes (no kinds)."""
        store = MetadataStore()
        store.set_missing("q1", codes=[-99, -98])

        meta = store.get("q1")
        assert meta.missing.codes == frozenset([-99, -98])
        assert meta.missing.kinds is None

    def test_set_type(self):
        """Set measurement type."""
        store = MetadataStore()
        store.set_type("q1", MeasurementType.ORDINAL)

        assert store.get("q1").mtype == MeasurementType.ORDINAL

    def test_set_categories(self):
        """Set categories."""
        store = MetadataStore()
        store.set_categories("q1", [1, 2, 3, 4, 5], ordered=True)

        meta = store.get("q1")
        assert meta.categories == (1, 2, 3, 4, 5)
        assert meta.mtype == MeasurementType.ORDINAL

    def test_resolve_labels_direct(self):
        """Resolve directly specified labels."""
        store = MetadataStore()
        store.set(
            "q1",
            VariableMeta(
                name="q1",
                label="Test",
                value_labels={1: "Yes", 0: "No"},
            ),
        )

        resolved = store.resolve_labels("q1")
        assert resolved.var_label == "Test"
        assert resolved.value_labels == {1: "Yes", 0: "No"}

    def test_resolve_labels_empty(self):
        """Resolve returns empty for unknown variable."""
        store = MetadataStore()
        resolved = store.resolve_labels("unknown")

        assert resolved.var_label == ""
        assert resolved.value_labels == {}

    def test_resolve_labels_cached(self):
        """Resolved labels are cached."""
        store = MetadataStore()
        store.set("q1", VariableMeta(name="q1", label="Test"))

        r1 = store.resolve_labels("q1")
        r2 = store.resolve_labels("q1")

        # Same object from cache
        assert r1 is r2

    def test_cache_invalidation_on_set(self):
        """Cache is invalidated when metadata changes."""
        store = MetadataStore()
        store.set("q1", VariableMeta(name="q1", label="Test 1"))

        r1 = store.resolve_labels("q1")
        assert r1.var_label == "Test 1"

        store.set("q1", VariableMeta(name="q1", label="Test 2"))

        r2 = store.resolve_labels("q1")
        assert r2.var_label == "Test 2"
        assert r1 is not r2

    def test_infer_from_dataframe(self):
        """Infer metadata from DataFrame."""
        df = pl.DataFrame(
            {
                "age": [25, 30, 35],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [1.5, 2.5, 3.5],
                "active": [True, False, True],
            }
        )

        store = MetadataStore()
        store.infer_from_dataframe(df)

        assert "age" in store
        assert "name" in store
        assert "score" in store
        assert "active" in store

        assert store.get("age").mtype == MeasurementType.DISCRETE
        assert store.get("name").mtype == MeasurementType.NOMINAL
        assert store.get("score").mtype == MeasurementType.CONTINUOUS
        assert store.get("active").mtype == MeasurementType.BOOLEAN

    def test_infer_preserves_existing(self):
        """Infer doesn't overwrite existing by default."""
        store = MetadataStore()
        store.set_label("age", "What is your age?")

        df = pl.DataFrame({"age": [25, 30], "name": ["A", "B"]})
        store.infer_from_dataframe(df, overwrite=False)

        # Existing label preserved
        assert store.get("age").label == "What is your age?"
        # New variable added
        assert "name" in store

    def test_infer_overwrite(self):
        """Infer can overwrite existing."""
        store = MetadataStore()
        store.set_label("age", "What is your age?")

        df = pl.DataFrame({"age": [25, 30]})
        store.infer_from_dataframe(df, overwrite=True)

        # Label was overwritten (now None from inference)
        assert store.get("age").label is None

    def test_align_to_dataframe(self):
        """Align removes dropped columns, adds new ones."""
        store = MetadataStore()
        store.set_label("a", "Column A")
        store.set_label("b", "Column B")
        store.set_label("c", "Column C")

        # DataFrame only has 'b' and 'd'
        df = pl.DataFrame({"b": [1, 2], "d": [3, 4]})
        store.align_to_dataframe(df)

        assert "a" not in store  # removed
        assert "b" in store  # kept
        assert "c" not in store  # removed
        assert "d" in store  # added

        # Label preserved for 'b'
        assert store.get("b").label == "Column B"

    def test_method_chaining(self):
        """Convenience methods support chaining."""
        store = MetadataStore()
        result = (
            store.set_label("q1", "Question 1")
            .set_label("q2", "Question 2")
            .set_value_labels("q1", {1: "Yes", 0: "No"})
            .set_missing("q1", dont_know=[-99])
            .set_type("q2", MeasurementType.ORDINAL)
        )

        assert result is store
        assert len(store) == 2

    def test_variables_property(self):
        """Variables property returns list of names."""
        store = MetadataStore()
        store.set_label("a", "A")
        store.set_label("b", "B")
        store.set_label("c", "C")

        vars = store.variables
        assert set(vars) == {"a", "b", "c"}

    def test_iteration(self):
        """Can iterate over variable names."""
        store = MetadataStore()
        store.set_label("a", "A")
        store.set_label("b", "B")

        names = list(store)
        assert set(names) == {"a", "b"}

    def test_summary_all(self):
        """Summary returns a DataFrame for all variables."""
        store = MetadataStore()
        store.set(
            "q1",
            VariableMeta(
                name="q1",
                label="Test question",
                value_labels={1: "Yes", 0: "No"},
                mtype=MeasurementType.NOMINAL,
                categories=(0, 1),
                source=MetadataSource.USER,
            ),
        )

        summary = store.summary()
        assert isinstance(summary, pl.DataFrame)
        assert len(summary) == 1

        row = summary.row(0, named=True)
        assert row["name"] == "q1"
        assert row["label"] == "Test question"
        assert row["mtype"] == "Categorical Nominal"
        assert row["has_value_labels"] is True
        assert row["n_categories"] == 2

    def test_summary_specific_vars(self):
        """Summary can filter to specific variables."""
        store = MetadataStore()
        store.set_label("q1", "Question 1")
        store.set_label("q2", "Question 2")
        store.set_label("q3", "Question 3")

        summary = store.summary(["q1", "q3"])
        assert len(summary) == 2
        names = summary["name"].to_list()
        assert "q1" in names
        assert "q3" in names
        assert "q2" not in names

    def test_summary_unknown_var(self):
        """Summary handles unknown variables gracefully."""
        store = MetadataStore()
        store.set_label("q1", "Question 1")

        summary = store.summary(["q1", "unknown"])
        assert len(summary) == 2

        # Check unknown row has None values
        unknown_row = summary.filter(pl.col("name") == "unknown").row(0, named=True)
        assert unknown_row["label"] is None
        assert unknown_row["mtype"] is None

    def test_inspect_single(self):
        """Inspect returns detailed metadata for a single variable."""
        store = MetadataStore()
        store.set(
            "q1",
            VariableMeta(
                name="q1",
                label="How satisfied are you?",
                value_labels={1: "Very dissatisfied", 5: "Very satisfied"},
                mtype=MeasurementType.ORDINAL,
                categories=(1, 2, 3, 4, 5),
                missing=MissingDef.from_kinds(
                    {-99: MissingKind.DONT_KNOW, -98: MissingKind.REFUSED}
                ),
                unit="scale",
                notes="5-point Likert scale",
                source=MetadataSource.QUESTIONNAIRE,
            ),
        )

        result = store.inspect("q1")
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

        row = result.row(0, named=True)
        assert row["name"] == "q1"
        assert row["label"] == "How satisfied are you?"
        assert "1=Very dissatisfied" in row["value_labels"]
        assert "5=Very satisfied" in row["value_labels"]
        assert row["categories"] == "1, 2, 3, 4, 5"
        assert "-99" in row["missing_codes"]
        assert "-98" in row["missing_codes"]
        assert "dnk" in row["missing_kinds"]  # MissingKind.DONT_KNOW.value
        assert row["unit"] == "scale"
        assert row["notes"] == "5-point Likert scale"
        assert row["source"] == "questionnaire"

    def test_inspect_multiple(self):
        """Inspect returns detailed metadata for multiple variables."""
        store = MetadataStore()
        store.set_label("q1", "Question 1")
        store.set_label("q2", "Question 2")
        store.set_label("q3", "Question 3")

        result = store.inspect(["q1", "q2"])
        assert len(result) == 2
        names = result["name"].to_list()
        assert names == ["q1", "q2"]

    def test_inspect_unknown_var(self):
        """Inspect handles unknown variables gracefully."""
        store = MetadataStore()
        store.set_label("q1", "Question 1")

        result = store.inspect(["q1", "unknown"])
        assert len(result) == 2

        unknown_row = result.filter(pl.col("name") == "unknown").row(0, named=True)
        assert unknown_row["label"] is None
        assert unknown_row["value_labels"] is None

    def test_coverage(self):
        """Coverage shows metadata vs data overlap."""
        store = MetadataStore()
        store.set_label("q1", "Question 1")
        store.set_label("q2", "Question 2")
        store.set_label("orphan", "Orphaned variable")  # not in data

        df = pl.DataFrame({"q1": [1, 2], "q2": [3, 4], "unlabeled": [5, 6]})

        result = store.coverage(df)
        assert isinstance(result, pl.DataFrame)

        # Check all variables present
        names = set(result["name"].to_list())
        assert names == {"q1", "q2", "orphan", "unlabeled"}

        # Check q1 - in both
        q1_row = result.filter(pl.col("name") == "q1").row(0, named=True)
        assert q1_row["in_data"] is True
        assert q1_row["in_metadata"] is True
        assert q1_row["has_label"] is True

        # Check orphan - in metadata only
        orphan_row = result.filter(pl.col("name") == "orphan").row(0, named=True)
        assert orphan_row["in_data"] is False
        assert orphan_row["in_metadata"] is True
        assert orphan_row["has_label"] is True

        # Check unlabeled - in data only
        unlabeled_row = result.filter(pl.col("name") == "unlabeled").row(0, named=True)
        assert unlabeled_row["in_data"] is True
        assert unlabeled_row["in_metadata"] is False
        assert unlabeled_row["has_label"] is False

    def test_unlabeled(self):
        """Unlabeled returns variables in data without labels."""
        store = MetadataStore()
        store.set_label("q1", "Question 1")
        store.set("q2", VariableMeta(name="q2"))  # no label

        df = pl.DataFrame({"q1": [1], "q2": [2], "q3": [3]})

        result = store.unlabeled(df)
        assert "q2" in result  # has metadata but no label
        assert "q3" in result  # no metadata at all
        assert "q1" not in result  # has label

    def test_orphaned(self):
        """Orphaned returns variables in metadata but not in data."""
        store = MetadataStore()
        store.set_label("q1", "Question 1")
        store.set_label("old_var", "Old variable")
        store.set_label("deleted", "Deleted variable")

        df = pl.DataFrame({"q1": [1], "new_var": [2]})

        result = store.orphaned(df)
        assert "old_var" in result
        assert "deleted" in result
        assert "q1" not in result  # in data
        assert "new_var" not in result  # not in metadata


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test a complete metadata workflow."""
        # Create store
        store = MetadataStore()

        # Infer from data
        df = pl.DataFrame(
            {
                "gender": [1, 2, 1, 2, 1],
                "age": [25, 30, 35, 40, 45],
                "satisfaction": [3, 4, 5, 2, 4],
            }
        )
        store.infer_from_dataframe(df)

        # Add labels
        store.set_labels(
            gender="What is your gender?",
            age="What is your age?",
            satisfaction="How satisfied are you?",
        )

        # Add value labels
        store.set_value_labels("gender", {1: "Male", 2: "Female"})
        store.set_value_labels(
            "satisfaction",
            {
                1: "Very dissatisfied",
                2: "Dissatisfied",
                3: "Neutral",
                4: "Satisfied",
                5: "Very satisfied",
            },
        )

        # Set measurement types
        store.set_type("satisfaction", MeasurementType.ORDINAL)
        store.set_categories("satisfaction", [1, 2, 3, 4, 5], ordered=True)

        # Resolve and use
        gender_labels = store.resolve_labels("gender")
        assert gender_labels.display(1) == "Male"
        assert gender_labels.display(2) == "Female"

        sat_labels = store.resolve_labels("satisfaction")
        assert sat_labels.display(5) == "Very satisfied"

        # Apply to data
        labeled_gender = gender_labels.display_series(df["gender"])
        assert labeled_gender.to_list() == ["Male", "Female", "Male", "Female", "Male"]

    def test_missing_handling_workflow(self):
        """Test workflow with missing values using user-friendly names."""
        store = MetadataStore()

        # Set up variable with missing codes using user-friendly names
        store.set_value_labels(
            "q1",
            {
                1: "Yes",
                0: "No",
                -99: "Don't know",
                -98: "Refused",
                -97: "Skipped",
            },
        )
        store.set_missing(
            "q1",
            dont_know=[-99],
            refused=[-98],
            skipped=[-97],  # maps to STRUCTURAL
        )

        # Resolve
        resolved = store.resolve_labels("q1")
        meta = store.get("q1")

        # Check that missing codes are identified
        assert meta.missing.is_missing(-99)
        assert meta.missing.is_missing(-98)
        assert meta.missing.is_missing(-97)
        assert not meta.missing.is_missing(1)

        # Check mechanism mapping
        assert meta.missing.kinds[-99] == MissingKind.DONT_KNOW
        assert meta.missing.kinds[-98] == MissingKind.REFUSED
        assert meta.missing.kinds[-97] == MissingKind.STRUCTURAL

        # Check user vs system missing
        assert meta.missing.user_missing() == frozenset([-99, -98])
        assert meta.missing.system_missing() == frozenset([-97])

        # Non-missing labels
        non_missing = resolved.non_missing_labels()
        assert non_missing == {1: "Yes", 0: "No"}

    def test_inspect_and_coverage_workflow(self):
        """Test using inspect and coverage for metadata review."""
        store = MetadataStore()

        # Set up some metadata
        store.set_labels(
            q1="How satisfied are you?",
            q2="Would you recommend us?",
            old_q="Old question (removed from survey)",
        )
        store.set_value_labels("q1", {1: "Very dissatisfied", 5: "Very satisfied"})
        store.set_missing("q1", dont_know=[-99])

        # Simulate data that doesn't have all variables
        df = pl.DataFrame(
            {
                "q1": [1, 2, 3],
                "q2": [4, 5, 1],
                "new_col": [1, 2, 3],  # not in metadata
            }
        )

        # Check coverage
        coverage = store.coverage(df)
        assert len(coverage) == 4  # q1, q2, old_q, new_col

        # Find unlabeled columns
        unlabeled = store.unlabeled(df)
        assert "new_col" in unlabeled

        # Find orphaned metadata
        orphaned = store.orphaned(df)
        assert "old_q" in orphaned

        # Inspect specific variables
        details = store.inspect(["q1", "q2"])
        assert len(details) == 2
        assert details.filter(pl.col("name") == "q1")["label"][0] == "How satisfied are you?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
