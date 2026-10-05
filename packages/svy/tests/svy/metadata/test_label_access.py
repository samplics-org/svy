import copy
import pickle

import polars as pl
import pytest

from svy import Sample
from svy.errors.label_errors import LabelAPIRemoved, LabelsReadOnly
from svy.metadata import CategoryScheme, LabellingCatalog, MetadataStore, VariableMeta
from svy.metadata.variable_meta import SchemeRef


@pytest.fixture
def store():
    catalog = LabellingCatalog().register(
        CategoryScheme(concept="yesno", entries={1: "Yes", 0: "No"})
    )
    s = MetadataStore(catalog=catalog)
    s.set_var_label("sex", "Gender")
    s.set_value_labels("sex", {1: "Male", 2: "Female"})
    s.set_var_label("age", "Age in years")
    s.set("q", VariableMeta(name="q", scheme_ref=SchemeRef(concept="yesno")))
    return s


class TestVarLabels:
    def test_every_labelled_variable(self, store):
        assert store.var_labels == {"sex": "Gender", "age": "Age in years"}

    def test_follows_later_setters(self, store):
        store.set_var_label("age", "Age")
        assert store.var_labels["age"] == "Age"


class TestValueLabels:
    def test_direct_and_catalog_labels(self, store):
        assert store.value_labels == {"sex": {1: "Male", 2: "Female"}, "q": {1: "Yes", 0: "No"}}

    def test_variable_without_value_labels_is_absent(self, store):
        assert "age" not in store.value_labels

    def test_follows_later_setters(self, store):
        store.set_value_labels("age", {0: "Under 1"})
        assert store.value_labels["age"] == {0: "Under 1"}

    def test_prints_as_a_dict(self, store):
        assert repr(store.value_labels["sex"]) == "{1: 'Male', 2: 'Female'}"


class TestReadOnly:
    @pytest.mark.parametrize(
        "edit",
        [
            lambda m: m.__setitem__("x", "y"),
            lambda m: m.__delitem__("sex"),
            lambda m: m.update({"x": "y"}),
            lambda m: m.pop("sex"),
            lambda m: m.popitem(),
            lambda m: m.setdefault("x", "y"),
            lambda m: m.clear(),
        ],
    )
    def test_edits_raise(self, store, edit):
        with pytest.raises(LabelsReadOnly, match="set_var_label"):
            edit(store.var_labels)

    def test_inner_mapping_is_read_only(self, store):
        with pytest.raises(LabelsReadOnly, match="set_value_labels"):
            store.value_labels["sex"][1] = "M"

    def test_in_place_union_raises(self, store):
        view = store.var_labels
        with pytest.raises(LabelsReadOnly):
            view |= {"x": "y"}

    def test_read_only_error_is_a_type_error(self, store):
        with pytest.raises(TypeError):
            store.var_labels["x"] = "y"

    @pytest.mark.parametrize("dup", [dict, copy.copy, copy.deepcopy])
    def test_copies_are_plain_dicts(self, store, dup):
        out = dup(store.value_labels)
        out["x"] = {}
        assert type(out) is dict

    def test_pickles_as_a_plain_dict(self, store):
        back = pickle.loads(pickle.dumps(store.value_labels))
        assert type(back) is dict
        assert back["sex"] == {1: "Male", 2: "Female"}


class TestRemovedNames:
    @pytest.mark.parametrize(
        "name, use",
        [
            ("resolve_labels", "meta.var_labels"),
            ("resolve_all", "meta.var_labels"),
            ("set_label", "set_var_label"),
            ("set_labels", "set_var_labels"),
        ],
    )
    def test_store_names_point_to_replacement(self, store, name, use):
        with pytest.raises(LabelAPIRemoved, match=use):
            getattr(store, name)
        assert not hasattr(store, name)

    @pytest.mark.parametrize("name", ["labels", "resolve_labels"])
    def test_sample_names_point_to_meta(self, name):
        s = Sample(pl.DataFrame({"x": [1, 2]}))
        with pytest.raises(LabelAPIRemoved, match="sample.meta"):
            getattr(s, name)
        assert not hasattr(s, name)

    def test_unknown_names_still_plain_attribute_errors(self, store):
        with pytest.raises(AttributeError) as e:
            store.nope
        assert not isinstance(e.value, LabelAPIRemoved)


def test_sample_meta_reaches_the_same_labels():
    s = Sample(pl.DataFrame({"sex": [1, 2]}))
    s.set_var_label("sex", "Gender").set_value_labels("sex", {1: "Male", 2: "Female"})
    assert s.meta.var_labels["sex"] == "Gender"
    assert s.meta.value_labels["sex"] == {1: "Male", 2: "Female"}
