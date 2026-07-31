"""svy has no model of missing values (design 001 §2.2).

A 99 labelled "Refusal" is the integer 99 with the label "Refusal". svy reads
it, prints it, and forms no opinion about it. Absence is a polars null, which
needs no metadata.

These tests assert the *absence* of a model, which is unusual and deliberate:
the removed API was public, and a later change that quietly reintroduced a
`missing` field or an automatic exclusion would otherwise pass unnoticed.
"""

import polars as pl
import pytest

from svy.metadata import (
    CategoryScheme,
    LabellingCatalog,
    MetadataStore,
    ResolvedLabels,
    VariableMeta,
)
from svy.metadata.variable_meta import SchemeRef


def test_a_declared_code_is_an_ordinary_labelled_value():
    store = MetadataStore()
    store.set_label("age", "Age")
    store.set_value_labels("age", {98: "Don't know", 99: "Refused"})

    resolved = store.resolve_labels("age")
    assert resolved.var_label == "Age"
    assert resolved.labels == {98: "Don't know", 99: "Refused"}
    assert resolved.display(99) == "Refused"


def test_the_same_holds_for_a_code_that_arrives_from_a_catalogue():
    catalog = LabellingCatalog().register(
        CategoryScheme(concept="yesno", entries={1: "Yes", 0: "No", 99: "Refused"})
    )
    store = MetadataStore(catalog=catalog)
    store.set("q", VariableMeta(name="q", scheme_ref=SchemeRef(concept="yesno")))

    assert store.resolve_labels("q").labels[99] == "Refused"


def test_there_is_no_missing_model_to_declare():
    for gone in ("missing", "na_as_level", "has_missing", "with_missing"):
        assert not hasattr(VariableMeta(name="q"), gone), f"VariableMeta grew {gone!r} back"
    for gone in ("missing_codes", "is_missing", "non_missing_labels"):
        assert not hasattr(ResolvedLabels(), gone), f"ResolvedLabels grew {gone!r} back"
    for gone in ("set_missing", "set_na_as_level"):
        assert not hasattr(MetadataStore(), gone), f"MetadataStore grew {gone!r} back"

    with pytest.raises(TypeError):
        VariableMeta(name="q", missing=frozenset({99}))


def test_svy_metadata_no_longer_exports_a_missing_vocabulary():
    import svy.metadata as m

    assert not hasattr(m, "MissingDef")
    # MissingKind's only consumer was MissingDef. The enum still exists in
    # svy.core.enumerations for anything else that wants it; metadata does not.
    assert "MissingKind" not in m.__all__


def test_a_declared_code_does_not_change_an_estimate():
    """The behaviour this scope makes explicit rather than accidental.

    `drop_nulls` handles values that are *absent*. A 99 is present — an
    ordinary integer — so nothing skips it, and the mean includes it. Recoding
    is the analyst's step; knowing that 99 *means* a refusal is svy-spec's job.
    """
    df = pl.DataFrame({"age": [20, 30, 40, 99, 99, None]})
    store = MetadataStore()
    store.set_value_labels("age", {99: "Refused"})

    # labelling the code changes what it is called, and nothing else
    assert df.select(pl.col("age").mean()).item() == pytest.approx(57.6)
    assert store.resolve_labels("age").labels == {99: "Refused"}


def test_the_writers_have_nothing_to_declare_as_user_missing():
    """What the SPSS/SAS writers emit is what `resolve_labels` gives them.

    They previously also wrote `meta.missing.codes` into the file's user-missing
    declaration. Nothing populated it on import — svy-io surfaces `MissingRule`
    and svy reads variable labels and value labels only — so its sole source was
    a hand-set field or svy-spec's bridge. With no field, the key stays in
    svy-io's dict shape and stays empty.

    Asserted through `resolve_labels` rather than by writing a file, because
    `svy_io` exposes no `write_spss` for the writer to call.
    """
    store = MetadataStore()
    store.set_label("q", "Q")
    store.set_value_labels("q", {1: "Yes", 99: "Refused"})

    resolved = store.resolve_labels("q")
    assert resolved.var_label == "Q"
    assert resolved.labels == {1: "Yes", 99: "Refused"}
    assert not hasattr(store.get("q"), "missing")
