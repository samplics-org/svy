# tests/svy/estimation/test_estimate_labels.py
"""Tests for Estimate label integration with MetadataStore."""

import pytest

from svy.core.enumerations import EstimationMethod, PopParam
from svy.estimation.estimate import Estimate, ParamEst
from svy.metadata import MetadataStore


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def metadata_store():
    """MetadataStore with labels for testing."""
    store = MetadataStore()
    store.set_label("region", "Geographic Region")
    store.set_value_labels("region", {"N": "North", "S": "South", "E": "East", "W": "West"})
    store.set_label("gender", "Sex of Respondent")
    store.set_value_labels("gender", {1: "Male", 2: "Female"})
    store.set_label("income", "Annual Income ($)")
    store.set_label("employed", "Employment Status")
    store.set_value_labels("employed", {0: "Unemployed", 1: "Employed"})
    return store


@pytest.fixture
def estimate_no_meta():
    """Basic Estimate without metadata."""
    est = Estimate(PopParam.MEAN, alpha=0.05)
    est.estimates = [
        ParamEst(
            y="income",
            est=50000,
            se=1000,
            cv=0.02,
            lci=48000,
            uci=52000,
            by=("region",),
            by_level=("N",),
        ),
        ParamEst(
            y="income",
            est=55000,
            se=1200,
            cv=0.02,
            lci=52600,
            uci=57400,
            by=("region",),
            by_level=("S",),
        ),
    ]
    est.method = EstimationMethod.TAYLOR
    return est


@pytest.fixture
def estimate_with_meta(metadata_store):
    """Estimate with metadata."""
    est = Estimate(PopParam.MEAN, alpha=0.05, metadata=metadata_store)
    est.estimates = [
        ParamEst(
            y="income",
            est=50000,
            se=1000,
            cv=0.02,
            lci=48000,
            uci=52000,
            by=("region",),
            by_level=("N",),
        ),
        ParamEst(
            y="income",
            est=55000,
            se=1200,
            cv=0.02,
            lci=52600,
            uci=57400,
            by=("region",),
            by_level=("S",),
        ),
    ]
    est.method = EstimationMethod.TAYLOR
    return est


@pytest.fixture
def prop_estimate_with_meta(metadata_store):
    """Proportion Estimate with metadata and y_level."""
    est = Estimate(PopParam.PROP, alpha=0.05, metadata=metadata_store)
    est.as_factor = True
    est.estimates = [
        ParamEst(
            y="employed",
            est=0.65,
            se=0.02,
            cv=0.03,
            lci=0.61,
            uci=0.69,
            by=("gender",),
            by_level=(1,),
            y_level=1,
        ),
        ParamEst(
            y="employed",
            est=0.55,
            se=0.03,
            cv=0.05,
            lci=0.49,
            uci=0.61,
            by=("gender",),
            by_level=(2,),
            y_level=1,
        ),
    ]
    est.method = EstimationMethod.TAYLOR
    return est


# =============================================================================
# Test Estimate construction with metadata
# =============================================================================


class TestEstimateConstruction:
    """Tests for Estimate construction with metadata parameter."""

    def test_estimate_accepts_metadata_parameter(self, metadata_store):
        """Estimate can be constructed with metadata."""
        est = Estimate(PopParam.MEAN, metadata=metadata_store)
        assert est._metadata is metadata_store

    def test_estimate_metadata_defaults_to_none(self):
        """Estimate metadata defaults to None."""
        est = Estimate(PopParam.MEAN)
        assert est._metadata is None

    def test_metadata_property_getter(self, estimate_with_meta, metadata_store):
        """metadata property returns the store."""
        assert estimate_with_meta.metadata is metadata_store

    def test_metadata_property_setter(self, estimate_no_meta, metadata_store):
        """metadata property can be set."""
        estimate_no_meta.metadata = metadata_store
        assert estimate_no_meta._metadata is metadata_store


# =============================================================================
# Test use_labels configuration
# =============================================================================


class TestUseLabelsSetting:
    """Tests for use_labels configuration."""

    def test_use_labels_defaults_to_true(self, estimate_no_meta):
        """use_labels defaults to True."""
        assert estimate_no_meta.use_labels is True

    def test_use_labels_can_be_set_false(self, estimate_no_meta):
        """use_labels can be set to False."""
        estimate_no_meta.use_labels = False
        assert estimate_no_meta.use_labels is False

    def test_use_labels_can_be_set_true(self, estimate_no_meta):
        """use_labels can be set to True."""
        estimate_no_meta.use_labels = False
        estimate_no_meta.use_labels = True
        assert estimate_no_meta.use_labels is True

    def test_use_labels_none_uses_class_default(self, estimate_no_meta):
        """use_labels=None falls back to class default."""
        estimate_no_meta.use_labels = None
        assert estimate_no_meta.use_labels is True

    def test_class_level_use_labels_default(self):
        """Class-level USE_LABELS can be changed."""
        original = Estimate.USE_LABELS
        try:
            Estimate.USE_LABELS = False
            est = Estimate(PopParam.MEAN)
            assert est.use_labels is False
        finally:
            Estimate.USE_LABELS = original

    def test_set_default_use_labels_class_method(self):
        """set_default_use_labels() changes class default."""
        original = Estimate.USE_LABELS
        try:
            Estimate.set_default_use_labels(False)
            assert Estimate.USE_LABELS is False
            Estimate.set_default_use_labels(True)
            assert Estimate.USE_LABELS is True
        finally:
            Estimate.USE_LABELS = original

    def test_style_method_sets_use_labels(self, estimate_no_meta):
        """style() method can set use_labels."""
        result = estimate_no_meta.style(use_labels=False)
        assert result is estimate_no_meta
        assert estimate_no_meta.use_labels is False


# =============================================================================
# Test to_polars_printable with labels
# =============================================================================


class TestToPolarsWithLabels:
    """Tests for to_polars_printable() with label support."""

    def test_printable_without_metadata_shows_raw(self, estimate_no_meta):
        """Without metadata, shows raw values."""
        df = estimate_no_meta.to_polars_printable()
        assert "region" in df.columns
        values = df["region"].to_list()
        assert "N" in values
        assert "S" in values

    def test_printable_with_metadata_shows_labels(self, estimate_with_meta):
        """With metadata, shows value labels."""
        df = estimate_with_meta.to_polars_printable()
        # Column should use variable label
        assert "Geographic Region" in df.columns
        values = df["Geographic Region"].to_list()
        assert "North" in values
        assert "South" in values

    def test_printable_use_labels_false(self, estimate_with_meta):
        """use_labels=False shows raw values even with metadata."""
        df = estimate_with_meta.to_polars_printable(use_labels=False)
        # Column should use raw name
        assert "region" in df.columns
        values = df["region"].to_list()
        assert "N" in values
        assert "S" in values

    def test_printable_use_labels_override(self, estimate_with_meta):
        """use_labels parameter overrides instance setting."""
        estimate_with_meta.use_labels = False
        # Override with explicit True
        df = estimate_with_meta.to_polars_printable(use_labels=True)
        assert "Geographic Region" in df.columns

    def test_printable_y_level_with_labels(self, prop_estimate_with_meta):
        """y_level values get labels for proportions."""
        df = prop_estimate_with_meta.to_polars_printable()
        # Should have y column with label
        assert "Employment Status" in df.columns
        # y_level=1 should show "Employed"
        values = df["Employment Status"].to_list()
        assert "Employed" in values

    def test_printable_by_level_integer_keys(self, prop_estimate_with_meta):
        """by_level with integer keys gets labels."""
        df = prop_estimate_with_meta.to_polars_printable()
        # gender column should use label
        assert "Sex of Respondent" in df.columns
        values = df["Sex of Respondent"].to_list()
        assert "Male" in values
        assert "Female" in values


# =============================================================================
# Test label helper methods
# =============================================================================


class TestLabelHelpers:
    """Tests for internal label helper methods."""

    def test_get_var_label_without_metadata(self, estimate_no_meta):
        """_get_var_label returns var name without metadata."""
        result = estimate_no_meta._get_var_label("region")
        assert result == "region"

    def test_get_var_label_with_metadata(self, estimate_with_meta):
        """_get_var_label returns label with metadata."""
        result = estimate_with_meta._get_var_label("region")
        assert result == "Geographic Region"

    def test_get_var_label_unlabeled_var(self, estimate_with_meta):
        """_get_var_label returns var name for unlabeled variable."""
        result = estimate_with_meta._get_var_label("unknown_var")
        assert result == "unknown_var"

    def test_get_value_label_without_metadata(self, estimate_no_meta):
        """_get_value_label returns str(value) without metadata."""
        result = estimate_no_meta._get_value_label("region", "N")
        assert result == "N"

    def test_get_value_label_with_metadata(self, estimate_with_meta):
        """_get_value_label returns label with metadata."""
        result = estimate_with_meta._get_value_label("region", "N")
        assert result == "North"

    def test_get_value_label_integer_key(self, estimate_with_meta):
        """_get_value_label handles integer keys."""
        result = estimate_with_meta._get_value_label("gender", 1)
        assert result == "Male"

    def test_get_value_label_string_to_int_conversion(self, estimate_with_meta):
        """_get_value_label converts string to int for lookup."""
        # Labels keyed by int 1, but value is string "1"
        result = estimate_with_meta._get_value_label("gender", "1")
        assert result == "Male"


# =============================================================================
# Test __str__ and display
# =============================================================================


class TestStringRepresentation:
    """Tests for string representation with labels."""

    def test_str_without_metadata(self, estimate_no_meta):
        """__str__ works without metadata."""
        result = str(estimate_no_meta)
        assert isinstance(result, str)
        assert "region" in result.lower() or "N" in result or "S" in result

    def test_str_with_metadata_shows_labels(self, estimate_with_meta):
        """__str__ shows labels when metadata is present."""
        result = str(estimate_with_meta)
        # Should contain labeled values
        assert "North" in result or "South" in result or "Geographic" in result


# =============================================================================
# Test edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in label handling."""

    def test_empty_estimate_with_metadata(self, metadata_store):
        """Empty estimate with metadata doesn't break display."""
        est = Estimate(PopParam.MEAN, metadata=metadata_store)
        df = est.to_polars_printable()
        assert df.is_empty()

    def test_estimate_no_by_cols(self, metadata_store):
        """Estimate without by columns works with labels."""
        est = Estimate(PopParam.MEAN, metadata=metadata_store)
        est.estimates = [
            ParamEst(y="income", est=50000, se=1000, cv=0.02, lci=48000, uci=52000),
        ]
        df = est.to_polars_printable()
        assert not df.is_empty()

    def test_partial_labels(self, metadata_store):
        """Estimate handles partial label coverage."""
        est = Estimate(PopParam.MEAN, metadata=metadata_store)
        est.estimates = [
            ParamEst(
                y="income",
                est=50000,
                se=1000,
                cv=0.02,
                lci=48000,
                uci=52000,
                by=("region",),
                by_level=("N",),
            ),
            ParamEst(
                y="income",
                est=45000,
                se=900,
                cv=0.02,
                lci=43200,
                uci=46800,
                by=("region",),
                by_level=("X",),
            ),  # No label for "X"
        ]
        df = est.to_polars_printable()
        values = df["Geographic Region"].to_list()
        assert "North" in values
        assert "X" in values  # Falls back to raw value

    def test_none_by_level(self, metadata_store):
        """Estimate handles None in by_level."""
        est = Estimate(PopParam.MEAN, metadata=metadata_store)
        est.estimates = [
            ParamEst(
                y="income",
                est=50000,
                se=1000,
                cv=0.02,
                lci=48000,
                uci=52000,
                by=("region",),
                by_level=None,
            ),
        ]
        # Should not raise
        df = est.to_polars_printable()
        assert not df.is_empty()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
