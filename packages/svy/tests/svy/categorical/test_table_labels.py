# tests/svy/categorical/test_table_labels.py
"""Tests for Table label integration with MetadataStore."""

import polars as pl
import pytest

from svy.categorical.table import CellEst, Table, _headers_for_display, _rows_for_display
from svy.metadata import MetadataStore


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def metadata_store():
    """MetadataStore with labels for testing."""
    store = MetadataStore()
    store.set_label("gender", "Sex of respondent")
    store.set_value_labels("gender", {1: "Male", 2: "Female"})
    store.set_label("region", "Geographic region")
    store.set_value_labels("region", {"N": "North", "S": "South", "E": "East", "W": "West"})
    store.set_label("satisfaction", "How satisfied are you?")
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
    return store


@pytest.fixture
def one_way_table():
    """Basic one-way table without metadata."""
    return Table.one_way(
        rowvar="gender",
        estimates=[
            CellEst(rowvar="1", colvar="", est=0.48, se=0.02, cv=0.04, lci=0.44, uci=0.52),
            CellEst(rowvar="2", colvar="", est=0.52, se=0.02, cv=0.04, lci=0.48, uci=0.56),
        ],
        rowvals=[1, 2],
    )


@pytest.fixture
def two_way_table():
    """Basic two-way table without metadata."""
    return Table.two_way(
        rowvar="gender",
        colvar="region",
        estimates=[
            CellEst(rowvar="1", colvar="N", est=0.25, se=0.02, cv=0.08, lci=0.21, uci=0.29),
            CellEst(rowvar="1", colvar="S", est=0.25, se=0.02, cv=0.08, lci=0.21, uci=0.29),
            CellEst(rowvar="2", colvar="N", est=0.25, se=0.02, cv=0.08, lci=0.21, uci=0.29),
            CellEst(rowvar="2", colvar="S", est=0.25, se=0.02, cv=0.08, lci=0.21, uci=0.29),
        ],
        rowvals=[1, 2],
        colvals=["N", "S"],
    )


@pytest.fixture
def one_way_table_with_meta(metadata_store):
    """One-way table with metadata."""
    return Table.one_way(
        rowvar="gender",
        estimates=[
            CellEst(rowvar="1", colvar="", est=0.48, se=0.02, cv=0.04, lci=0.44, uci=0.52),
            CellEst(rowvar="2", colvar="", est=0.52, se=0.02, cv=0.04, lci=0.48, uci=0.56),
        ],
        rowvals=[1, 2],
        metadata=metadata_store,
    )


@pytest.fixture
def two_way_table_with_meta(metadata_store):
    """Two-way table with metadata."""
    return Table.two_way(
        rowvar="gender",
        colvar="region",
        estimates=[
            CellEst(rowvar="1", colvar="N", est=0.25, se=0.02, cv=0.08, lci=0.21, uci=0.29),
            CellEst(rowvar="1", colvar="S", est=0.25, se=0.02, cv=0.08, lci=0.21, uci=0.29),
            CellEst(rowvar="2", colvar="N", est=0.25, se=0.02, cv=0.08, lci=0.21, uci=0.29),
            CellEst(rowvar="2", colvar="S", est=0.25, se=0.02, cv=0.08, lci=0.21, uci=0.29),
        ],
        rowvals=[1, 2],
        colvals=["N", "S"],
        metadata=metadata_store,
    )


# =============================================================================
# Test Table construction with metadata
# =============================================================================


class TestTableConstruction:
    """Tests for Table construction with metadata parameter."""

    def test_table_accepts_metadata_parameter(self, metadata_store):
        """Table can be constructed with metadata."""
        tbl = Table.one_way(
            rowvar="gender",
            estimates=[],
            metadata=metadata_store,
        )
        assert tbl._metadata is metadata_store

    def test_table_metadata_defaults_to_none(self):
        """Table metadata defaults to None."""
        tbl = Table.one_way(rowvar="gender", estimates=[])
        assert tbl._metadata is None

    def test_one_way_factory_accepts_metadata(self, metadata_store):
        """Table.one_way() accepts metadata parameter."""
        tbl = Table.one_way(rowvar="gender", metadata=metadata_store)
        assert tbl._metadata is metadata_store

    def test_two_way_factory_accepts_metadata(self, metadata_store):
        """Table.two_way() accepts metadata parameter."""
        tbl = Table.two_way(rowvar="gender", colvar="region", metadata=metadata_store)
        assert tbl._metadata is metadata_store


# =============================================================================
# Test use_labels property and configuration
# =============================================================================


class TestUseLabelsSetting:
    """Tests for use_labels configuration."""

    def test_use_labels_defaults_to_true(self, one_way_table):
        """use_labels defaults to True."""
        assert one_way_table.use_labels is True

    def test_use_labels_can_be_set_false(self, one_way_table):
        """use_labels can be set to False."""
        one_way_table.use_labels = False
        assert one_way_table.use_labels is False

    def test_use_labels_can_be_set_true(self, one_way_table):
        """use_labels can be set to True."""
        one_way_table.use_labels = False
        one_way_table.use_labels = True
        assert one_way_table.use_labels is True

    def test_use_labels_none_uses_class_default(self, one_way_table):
        """use_labels=None falls back to class default."""
        one_way_table.use_labels = None
        assert one_way_table.use_labels is True  # class default

    def test_class_level_use_labels_default(self):
        """Class-level USE_LABELS can be changed."""
        original = Table.USE_LABELS
        try:
            Table.USE_LABELS = False
            tbl = Table.one_way(rowvar="test")
            assert tbl.use_labels is False
        finally:
            Table.USE_LABELS = original

    def test_set_default_use_labels_class_method(self):
        """set_default_use_labels() changes class default."""
        original = Table.USE_LABELS
        try:
            Table.set_default_use_labels(False)
            assert Table.USE_LABELS is False
            Table.set_default_use_labels(True)
            assert Table.USE_LABELS is True
        finally:
            Table.USE_LABELS = original

    def test_style_method_sets_use_labels(self, one_way_table):
        """style() method can set use_labels."""
        result = one_way_table.style(use_labels=False)
        assert result is one_way_table
        assert one_way_table.use_labels is False


# =============================================================================
# Test header display with labels
# =============================================================================


class TestHeadersWithLabels:
    """Tests for header display with labels."""

    def test_headers_without_metadata_use_row_col(self, one_way_table):
        """Without metadata, headers use 'Row' and 'Col'."""
        headers = _headers_for_display(one_way_table)
        assert headers[0] == "Row"

    def test_headers_without_metadata_two_way(self, two_way_table):
        """Two-way table without metadata uses 'Row' and 'Col'."""
        headers = _headers_for_display(two_way_table)
        assert headers[0] == "Row"
        assert headers[1] == "Col"

    def test_headers_with_metadata_use_var_labels(self, one_way_table_with_meta):
        """With metadata, headers use variable labels."""
        headers = _headers_for_display(one_way_table_with_meta)
        assert headers[0] == "Sex of respondent"

    def test_headers_with_metadata_two_way(self, two_way_table_with_meta):
        """Two-way table with metadata uses variable labels."""
        headers = _headers_for_display(two_way_table_with_meta)
        assert headers[0] == "Sex of respondent"
        assert headers[1] == "Geographic region"

    def test_headers_with_use_labels_false(self, one_way_table_with_meta):
        """With use_labels=False, headers use 'Row'/'Col' even with metadata."""
        one_way_table_with_meta.use_labels = False
        headers = _headers_for_display(one_way_table_with_meta)
        assert headers[0] == "Row"

    def test_headers_always_include_stats(self, one_way_table):
        """Headers always include statistic columns."""
        headers = _headers_for_display(one_way_table)
        assert "Estimate" in headers
        assert "Std Err" in headers
        assert "CV" in headers
        assert "Lower" in headers
        assert "Upper" in headers


# =============================================================================
# Test row display with value labels
# =============================================================================


class TestRowDisplayWithLabels:
    """Tests for row value display with labels."""

    def test_rows_without_metadata_show_raw_values(self, one_way_table):
        """Without metadata, rows show raw values."""
        rows = list(_rows_for_display(one_way_table))
        row_values = [r[0] for r in rows]
        assert "1" in row_values or "2" in row_values

    def test_rows_with_metadata_show_labels(self, one_way_table_with_meta):
        """With metadata, rows show value labels."""
        rows = list(_rows_for_display(one_way_table_with_meta))
        row_values = [r[0] for r in rows]
        assert "Male" in row_values
        assert "Female" in row_values

    def test_rows_with_use_labels_false(self, one_way_table_with_meta):
        """With use_labels=False, rows show raw values."""
        one_way_table_with_meta.use_labels = False
        rows = list(_rows_for_display(one_way_table_with_meta))
        row_values = [r[0] for r in rows]
        # Should show raw values, not labels
        assert "Male" not in row_values
        assert "Female" not in row_values

    def test_two_way_rows_show_col_labels(self, two_way_table_with_meta):
        """Two-way table shows column value labels."""
        rows = list(_rows_for_display(two_way_table_with_meta))
        col_values = [r[1] for r in rows]  # column values are second element
        assert "North" in col_values or "South" in col_values


# =============================================================================
# Test __str__ and show() with labels
# =============================================================================


class TestStringRepresentationWithLabels:
    """Tests for string representation with labels."""

    def test_str_without_metadata(self, one_way_table):
        """__str__ works without metadata."""
        result = str(one_way_table)
        assert "Row" in result or "gender" in result.lower()
        assert "Estimate" in result

    def test_str_with_metadata_shows_labels(self, one_way_table_with_meta):
        """__str__ shows labels when metadata is present."""
        result = str(one_way_table_with_meta)
        # Should contain the variable label or value labels
        assert "Sex of respondent" in result or "Male" in result or "Female" in result

    def test_show_method_exists(self, one_way_table):
        """show() method exists and is callable."""
        # Should not raise
        one_way_table.show(use_rich=False)


# =============================================================================
# Test crosstab() with labels
# =============================================================================


class TestCrosstabWithLabels:
    """Tests for crosstab() method with labels."""

    def test_crosstab_without_metadata(self, one_way_table):
        """crosstab() works without metadata."""
        result = one_way_table.crosstab()
        assert isinstance(result, pl.DataFrame)

    def test_crosstab_with_metadata_applies_labels(self, one_way_table_with_meta):
        """crosstab() applies value labels when metadata present."""
        result = one_way_table_with_meta.crosstab(use_labels=True)
        assert isinstance(result, pl.DataFrame)
        # Check that labeled values appear
        row_col = result.columns[0]
        values = result[row_col].to_list()
        assert "Male" in values or "Female" in values

    def test_crosstab_use_labels_false(self, one_way_table_with_meta):
        """crosstab(use_labels=False) shows raw values."""
        result = one_way_table_with_meta.crosstab(use_labels=False)
        row_col = result.columns[0]
        values = result[row_col].to_list()
        # Should not contain labeled values
        assert "Male" not in values
        assert "Female" not in values

    def test_crosstab_two_way_with_labels(self, two_way_table_with_meta):
        """Two-way crosstab applies labels to both dimensions."""
        result = two_way_table_with_meta.crosstab(use_labels=True)
        assert isinstance(result, pl.DataFrame)


# =============================================================================
# Test update() and fill_missing() preserve metadata
# =============================================================================


class TestMetadataPreservation:
    """Tests that metadata is preserved through table operations."""

    def test_update_preserves_metadata(self, one_way_table_with_meta):
        """update() preserves metadata reference."""
        updated = one_way_table_with_meta.update(alpha=0.10)
        assert updated._metadata is one_way_table_with_meta._metadata

    def test_fill_missing_preserves_metadata(self, one_way_table_with_meta):
        """fill_missing() preserves metadata reference."""
        filled = one_way_table_with_meta.fill_missing(alpha=0.10)
        assert filled._metadata is one_way_table_with_meta._metadata

    def test_update_can_change_metadata(self, one_way_table_with_meta, metadata_store):
        """update() can change metadata."""
        new_store = MetadataStore()
        new_store.set_label("gender", "New Label")
        updated = one_way_table_with_meta.update(metadata=new_store)
        assert updated._metadata is new_store

    def test_add_estimate_preserves_metadata(self, one_way_table_with_meta):
        """add_estimate() preserves metadata."""
        new_est = CellEst(rowvar="3", colvar="", est=0.0, se=0.0, cv=0.0, lci=0.0, uci=0.0)
        updated = one_way_table_with_meta.add_estimate(new_est)
        assert updated._metadata is one_way_table_with_meta._metadata


# =============================================================================
# Test _get_var_label and _get_value_label helpers
# =============================================================================


class TestLabelHelpers:
    """Tests for internal label helper methods."""

    def test_get_var_label_without_metadata(self, one_way_table):
        """_get_var_label returns var name without metadata."""
        result = one_way_table._get_var_label("gender")
        assert result == "gender"

    def test_get_var_label_with_metadata(self, one_way_table_with_meta):
        """_get_var_label returns label with metadata."""
        result = one_way_table_with_meta._get_var_label("gender")
        assert result == "Sex of respondent"

    def test_get_var_label_unlabeled_var(self, one_way_table_with_meta):
        """_get_var_label returns var name for unlabeled variable."""
        result = one_way_table_with_meta._get_var_label("unknown_var")
        assert result == "unknown_var"

    def test_get_value_label_without_metadata(self, one_way_table):
        """_get_value_label returns str(value) without metadata."""
        result = one_way_table._get_value_label("gender", 1)
        assert result == "1"

    def test_get_value_label_with_metadata(self, one_way_table_with_meta):
        """_get_value_label returns label with metadata."""
        result = one_way_table_with_meta._get_value_label("gender", 1)
        assert result == "Male"

    def test_get_value_label_unlabeled_value(self, one_way_table_with_meta):
        """_get_value_label returns str(value) for unlabeled value."""
        result = one_way_table_with_meta._get_value_label("gender", 99)
        assert result == "99"

    def test_get_row_display(self, one_way_table_with_meta):
        """_get_row_display applies row variable labels."""
        result = one_way_table_with_meta._get_row_display(1)
        assert result == "Male"

    def test_get_col_display(self, two_way_table_with_meta):
        """_get_col_display applies column variable labels."""
        result = two_way_table_with_meta._get_col_display("N")
        assert result == "North"


# =============================================================================
# Test edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in label handling."""

    def test_empty_table_with_metadata(self, metadata_store):
        """Empty table with metadata doesn't break display."""
        tbl = Table.one_way(rowvar="gender", estimates=[], metadata=metadata_store)
        result = str(tbl)
        assert isinstance(result, str)

    def test_partial_labels(self, metadata_store):
        """Table handles partial label coverage."""
        # Add estimate with value not in labels
        tbl = Table.one_way(
            rowvar="gender",
            estimates=[
                CellEst(rowvar="1", colvar="", est=0.33, se=0.02, cv=0.06, lci=0.29, uci=0.37),
                CellEst(rowvar="2", colvar="", est=0.33, se=0.02, cv=0.06, lci=0.29, uci=0.37),
                CellEst(
                    rowvar="3", colvar="", est=0.34, se=0.02, cv=0.06, lci=0.30, uci=0.38
                ),  # No label for 3
            ],
            rowvals=[1, 2, 3],
            metadata=metadata_store,
        )
        rows = list(_rows_for_display(tbl))
        row_values = [r[0] for r in rows]
        assert "Male" in row_values
        assert "Female" in row_values
        assert "3" in row_values  # Falls back to raw value

    def test_none_value_in_estimates(self, metadata_store):
        """Table handles None values in estimates."""
        tbl = Table.one_way(
            rowvar="gender",
            estimates=[
                CellEst(rowvar="1", colvar="", est=0.5, se=None, cv=None, lci=0.4, uci=0.6),
            ],
            metadata=metadata_store,
        )
        # Should not raise
        rows = list(_rows_for_display(tbl))
        assert len(rows) == 1

    def test_metadata_with_no_labels_for_variable(self):
        """Metadata store without labels for the table variable."""
        store = MetadataStore()
        store.set_label("other_var", "Other Variable")  # Different variable

        tbl = Table.one_way(
            rowvar="gender",
            estimates=[
                CellEst(rowvar="1", colvar="", est=0.5, se=0.02, cv=0.04, lci=0.46, uci=0.54),
            ],
            metadata=store,
        )
        # Should fall back to defaults
        headers = _headers_for_display(tbl)
        assert headers[0] == "Row"  # Falls back to "Row"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
