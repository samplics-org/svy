"""Measurement type is resolved on import, not left at the storage type.

`Sample(data=df)` infers `mtype` from the polars dtype, so every numeric
column starts out CONTINUOUS. The label importer used to bring across labels
and leave that untouched, which made `is_categorical` False for every labelled
variable in a survey recode (issue #130).

SPSS declares a measurement level but defaults numeric variables to "scale",
and Stata has no such attribute at all, so the declaration alone fixes
nothing. Label coverage of the observed values carries the rest.
"""

import polars as pl
import pytest

from svy.core.enumerations import MeasurementType, MetadataSource
from svy.engine.io.core import (
    _canonical_code,
    _resolve_mtype,
    import_labels_from_svyio_meta,
)
from svy.metadata import MetadataStore, VariableMeta


def _meta(name="v106", *, measure=None, mapping=None):
    """svy-io's classic split metadata shape."""
    var = {"name": name, "label": "Educ", "kind": "double"}
    if measure is not None:
        var["measure"] = measure
    out = {"vars": [var], "value_labels": []}
    if mapping is not None:
        var["label_set"] = f"{name}_set"
        out["value_labels"] = [{"set_name": f"{name}_set", "mapping": mapping}]
    return out


# ---- code canonicalization ------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", "1"),  # code as svy-io returns it
        (1.0, "1"),  # the same value as the Float64 column holds it
        ("1.0", "1"),  # a code that spells itself as a float
        (1, "1"),
        (True, "1"),
        (-9.0, "-9"),
        (1.5, "1.5"),
        ("F", "F"),  # string codes pass through
    ],
)
def test_codes_and_values_canonicalize_to_one_form(value, expected):
    """Codes arrive as strings, columns hold floats; they must compare equal."""
    assert _canonical_code(value) == expected


# ---- the coverage rule ----------------------------------------------------


def test_full_label_coverage_is_categorical():
    series = pl.Series("x", [1.0, 2.0, 1.0, 2.0])
    got = _resolve_mtype(None, {"1": "Urban", "2": "Rural"}, series)

    assert got == MeasurementType.NOMINAL


def test_partial_label_coverage_is_left_alone():
    """
    A count variable that labels only its endpoints -- "how many TVs?" with
    0 = none and 4 = 4 or more -- is continuous, not a 5-level factor. This is
    the case the issue warned about when rejecting "has labels -> categorical".
    """
    series = pl.Series("tv", [0.0, 1.0, 2.0, 3.0, 4.0])
    got = _resolve_mtype(None, {"0": "None", "4": "4 or more"}, series)

    assert got is None


def test_nulls_do_not_count_against_coverage():
    series = pl.Series("x", [1.0, None, 2.0, None])
    got = _resolve_mtype(None, {"1": "Urban", "2": "Rural"}, series)

    assert got == MeasurementType.NOMINAL


def test_all_null_column_is_left_alone():
    """No observed values means no evidence, so do not guess."""
    series = pl.Series("x", [None, None], dtype=pl.Float64)

    assert _resolve_mtype(None, {"1": "Urban"}, series) is None


def test_no_labels_and_no_measure_leaves_mtype_alone():
    assert _resolve_mtype(None, {}, pl.Series("x", [1.0, 2.0])) is None


# ---- the declared measure -------------------------------------------------


def test_declared_nominal_wins_without_needing_labels():
    """A declaration decides on its own; no labels need exist to corroborate."""
    unlabelled = pl.Series("x", [1.0, 2.0, 3.0])

    assert _resolve_mtype("nominal", {}, unlabelled) == MeasurementType.NOMINAL


def test_declared_ordinal_is_honored():
    """Coverage cannot tell ordinal from nominal; only the file can."""
    unlabelled = pl.Series("x", [1.0, 2.0, 3.0])

    assert _resolve_mtype("ordinal", {}, unlabelled) == MeasurementType.ORDINAL


def test_declared_scale_does_not_veto_the_coverage_rule():
    """
    SPSS defaults numeric variables to "scale" whether or not anyone meant it,
    so it must not be read as a positive claim of continuity.
    """
    series = pl.Series("x", [1.0, 2.0])
    got = _resolve_mtype("scale", {"1": "Urban", "2": "Rural"}, series)

    assert got == MeasurementType.NOMINAL


# ---- the importer end to end ----------------------------------------------


def test_import_sets_categorical_for_a_fully_labelled_column():
    store = MetadataStore()
    df = pl.DataFrame({"v106": [0.0, 1.0, 2.0, 3.0]})
    mapping = {"0": "None", "1": "Primary", "2": "Secondary", "3": "Higher"}

    import_labels_from_svyio_meta(store, _meta(mapping=mapping), df)

    got = store.get("v106")
    assert got.has_labels
    assert got.mtype == MeasurementType.NOMINAL
    assert got.is_categorical


def test_import_does_not_override_a_user_set_mtype():
    store = MetadataStore()
    store.set(
        "v106",
        VariableMeta(
            name="v106",
            mtype=MeasurementType.CONTINUOUS,
            source=MetadataSource.USER,
        ),
    )
    df = pl.DataFrame({"v106": [0.0, 1.0]})

    import_labels_from_svyio_meta(store, _meta(mapping={"0": "No", "1": "Yes"}), df)

    assert store.get("v106").mtype == MeasurementType.CONTINUOUS


def test_import_leaves_a_partially_labelled_column_continuous():
    store = MetadataStore()
    store.set("tv", VariableMeta(name="tv", mtype=MeasurementType.CONTINUOUS))
    df = pl.DataFrame({"tv": [0.0, 1.0, 2.0, 3.0, 4.0]})
    mapping = {"0": "None", "4": "4 or more"}

    import_labels_from_svyio_meta(store, _meta("tv", mapping=mapping), df)

    got = store.get("tv")
    assert got.has_labels
    assert not got.is_categorical


def test_import_handles_the_variables_metadata_shape():
    """The other shape the importer accepts must resolve mtype too."""
    store = MetadataStore()
    df = pl.DataFrame({"v106": [1.0, 2.0]})
    meta = {
        "variables": {
            "v106": {"label": "Educ", "values": {"1": "Yes", "2": "No"}},
        }
    }

    import_labels_from_svyio_meta(store, meta, df)

    assert store.get("v106").mtype == MeasurementType.NOMINAL


def test_import_ignores_labels_for_a_column_not_in_the_frame():
    store = MetadataStore()
    df = pl.DataFrame({"other": [1.0]})

    import_labels_from_svyio_meta(store, _meta(mapping={"1": "Yes"}), df)

    assert store.get("v106").mtype != MeasurementType.NOMINAL
