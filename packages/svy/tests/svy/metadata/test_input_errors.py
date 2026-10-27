# tests/svy/metadata/test_input_errors.py
"""Bad input to labels and the metadata store raises LabelError, also the builtin it replaces."""

from __future__ import annotations

import polars as pl
import pytest

import svy

from svy.errors import Code, LabelError, SvyError
from svy.metadata import MetadataStore, VariableMeta


@pytest.fixture
def store() -> MetadataStore:
    df = pl.DataFrame({"y": [1.0, 2.0], "g": [1, 2], "w": [1.0, 1.0]})
    return svy.Sample(df, svy.Design(wgt="w")).meta


def _raises(kind, code, call, *, param=None, says=None):
    with pytest.raises(LabelError) as err:
        call()
    e = err.value
    assert type(e) is getattr(LabelError, kind.__name__)
    assert isinstance(e, kind) and isinstance(e, SvyError)
    assert e.code == code
    if param is not None:
        assert e.param == param
    if says is not None:
        assert says in e.detail
    return e


def test_require_a_variable_the_store_lacks(store):
    e = _raises(
        KeyError,
        Code.LABEL_VARIABLE_UNKNOWN,
        lambda: store.require("nope"),
        param="var",
        says="No metadata for variable: 'nope'",
    )
    assert e.got == "nope"
    assert store.get("nope") is None


def test_value_labels_and_a_scheme_together():
    _raises(
        ValueError,
        Code.LABEL_SOURCE_CONFLICT,
        lambda: VariableMeta(name="g", value_labels={1: "a"}, scheme_ref="c"),
        param="value_labels",
        says="cannot specify both value_labels and scheme_ref",
    )


@pytest.mark.parametrize(
    "call, param",
    [
        (lambda s: s.set_value_labels("g", ["a", "b"]), "labels"),
        (lambda s: VariableMeta(name="g").with_value_labels(3), "labels"),
        (lambda s: s.set_categories("g", 3), "categories"),
        (lambda s: VariableMeta(name="g").with_categories("ab"), "categories"),
        (lambda s: s.rename_variables("y"), "renames"),
        (lambda s: s.coverage(3), "data"),
        (lambda s: s.set("g", 3), "meta"),
        (lambda s: MetadataStore().update(3), "other"),
    ],
    ids=[
        "value labels list",
        "with_value_labels int",
        "categories int",
        "categories str",
        "rename str",
        "coverage int",
        "set int",
        "update int",
    ],
)
def test_argument_of_wrong_type(store, call, param):
    _raises(TypeError, Code.INVALID_TYPE, lambda: call(store), param=param)


def test_right_types_still_work(store):
    store.set_value_labels("g", {1: "one", 2: "two"})
    assert store.require("g").labels == {1: "one", 2: "two"}
    assert VariableMeta(name="g").with_categories([1, 2]).categories == (1, 2)
    assert store.coverage(pl.DataFrame({"g": [1]})).height >= 1
