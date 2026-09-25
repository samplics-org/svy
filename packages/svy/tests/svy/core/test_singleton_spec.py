# tests/svy/core/test_singleton_spec.py
"""SingletonSpec and Design.singleton: the decision stored on the design.

The sample-level behaviour (installing, deriving, keeping or clearing a rule)
is in test_singleton_design_part.py.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import pickle
import warnings

import msgspec
import pytest

import svy

from svy.core.design import Design, SingletonSpec, WgtAdjustment
from svy.core.repwgts import BrrWgts
from svy.serialize import (
    DesignData,
    SingletonSpecData,
    from_json,
    serialize,
    to_design,
    to_json,
)


pytestmark = pytest.mark.filterwarnings("error")


def _round_trip(design: Design) -> Design:
    return to_design(from_json(to_json(design)))


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["certainty", "skip", "scale", "center"])
def test_strata_constructors(method):
    spec = getattr(SingletonSpec, method)(["d", "e"])
    assert spec.method == method
    assert spec.strata == ("d", "e")
    assert spec.handled == ("d", "e")
    assert spec.mapping == ()
    assert spec.name is None
    assert spec.targets == ()


def test_collapse_constructor():
    spec = SingletonSpec.collapse({"d": "a", "e": "a"})
    assert spec.method == "collapse"
    assert spec.mapping == (("d", "a"), ("e", "a"))
    assert spec.handled == ("d", "e")
    assert spec.targets == ("a",)
    assert spec.strata == ()


def test_pool_constructor_and_default_name():
    assert SingletonSpec.pool(["d", "e"]).name == "__pooled__"
    assert SingletonSpec.pool(["d"], name="other").name == "other"
    assert SingletonSpec(method="pool", strata=("d",)).name == "__pooled__"


def test_a_single_stratum_can_be_passed_bare():
    assert SingletonSpec.skip("d").strata == ("d",)
    assert SingletonSpec.skip(4).strata == (4,)


def test_tuple_strata_are_stored_as_tuples():
    spec = SingletonSpec.certainty([("N", 4), ["S", 5]])
    assert spec.strata == (("N", 4), ("S", 5))
    col = SingletonSpec.collapse({("N", 4): ("N", 1)})
    assert col.mapping == ((("N", 4), ("N", 1)),)


def test_native_value_types_are_kept():
    spec = SingletonSpec.skip([1, 2.5, True, None, dt.date(2020, 1, 4)])
    assert spec.strata == (1, 2.5, True, None, dt.date(2020, 1, 4))


def test_duplicates_are_dropped_in_order():
    assert SingletonSpec.skip(["e", "d", "e"]).strata == ("e", "d")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "average", "strata": ("d",)}, "Unknown singleton method"),
        ({"method": "skip"}, "non-empty strata"),
        ({"method": "skip", "strata": ("d",), "mapping": (("d", "a"),)}, "no mapping"),
        ({"method": "collapse"}, "non-empty mapping"),
        ({"method": "collapse", "mapping": (("d", "a"),), "strata": ("d",)}, "no strata"),
        ({"method": "collapse", "mapping": (("d", "a"), ("d", "b"))}, "each singleton"),
        ({"method": "collapse", "mapping": (("d", "e"), ("e", "a"))}, "not be singletons"),
        ({"method": "skip", "strata": ("d",), "name": "x"}, "pool only"),
        ({"method": "pool", "strata": ("d",), "name": ""}, "non-empty string"),
    ],
)
def test_invalid_specs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SingletonSpec(**kwargs)


def test_nested_values_are_refused():
    with pytest.raises(TypeError, match="scalar"):
        SingletonSpec.skip([("N", ("x", 1))])
    with pytest.raises(TypeError):
        SingletonSpec.skip([{"a": 1}])


def test_spec_is_frozen_and_hashable():
    spec = SingletonSpec.collapse({"d": "a"})
    with pytest.raises(AttributeError):
        spec.method = "skip"  # type: ignore[misc]
    assert hash(spec) == hash(SingletonSpec.collapse({"d": "a"}))
    assert spec != SingletonSpec.collapse({"d": "b"})


def test_exported():
    assert svy.SingletonSpec is SingletonSpec
    assert svy.core.SingletonSpec is SingletonSpec


# ---------------------------------------------------------------------------
# On the Design
# ---------------------------------------------------------------------------

SPEC = SingletonSpec.collapse({"d": "a"})


def test_design_carries_the_spec():
    d = Design(stratum="st", psu="psu", wgt="w", singleton=SPEC)
    assert d.singleton == SPEC
    assert Design(stratum="st").singleton is None


def test_design_needs_a_stratum_for_a_spec():
    with pytest.raises(ValueError, match="no stratum"):
        Design(psu="psu", singleton=SPEC)


def test_design_rejects_other_types():
    with pytest.raises(TypeError, match="SingletonSpec"):
        Design(stratum="st", singleton={"d": "a"})  # type: ignore[arg-type]


def test_equality_and_hash_include_the_spec():
    a = Design(stratum="st", psu="psu", singleton=SPEC)
    b = Design(stratum="st", psu="psu")
    assert a != b
    assert a == Design(stratum="st", psu="psu", singleton=SingletonSpec.collapse({"d": "a"}))
    assert hash(a) == hash(Design(stratum="st", psu="psu", singleton=SPEC))
    assert hash(a) != hash(b)


def test_repr_shows_the_spec_only_when_set():
    assert "singleton=SingletonSpec.collapse({'d': 'a'})" in repr(
        Design(stratum="st", singleton=SPEC)
    )
    assert "singleton" not in repr(Design(stratum="st"))


@pytest.mark.parametrize(
    "spec, shown",
    [
        (SingletonSpec.collapse({3: 1}), "SingletonSpec.collapse({3: 1})"),
        (SingletonSpec.skip([4, 7]), "SingletonSpec.skip([4, 7])"),
        (SingletonSpec.certainty([("N", 4)]), "SingletonSpec.certainty([('N', 4)])"),
        (SingletonSpec.pool(["d", "e"]), "SingletonSpec.pool(['d', 'e'])"),
        (SingletonSpec.pool(["d"], name="rest"), "SingletonSpec.pool(['d'], name='rest')"),
        (SingletonSpec.center([None, True]), "SingletonSpec.center([None, True])"),
        (
            SingletonSpec.scale([dt.date(2020, 1, 4)]),
            "SingletonSpec.scale([datetime.date(2020, 1, 4)])",
        ),
    ],
)
def test_repr_is_the_constructor_call(spec, shown):
    assert repr(spec) == shown
    assert spec._to_code() == "svy." + shown.replace("datetime.date(2020, 1, 4)", "'2020-01-04'")


def test_printed_design_shows_the_rule():
    d = Design(stratum="st", psu="psu", singleton=SingletonSpec.collapse({3: 1}))
    assert "Singleton        : SingletonSpec.collapse({3: 1})" in d.__plain_str__()
    assert "SingletonSpec.collapse({3: 1})" in str(d)
    plain = Design(stratum="st", psu="psu")
    assert "Singleton" not in plain.__plain_str__() and "Singleton" not in str(plain)


def test_printed_sample_shows_the_rule():
    import polars as pl

    df = pl.DataFrame({"st": ["a", "a", "b"], "psu": ["1", "2", "3"], "y": [1.0, 2.0, 3.0]})
    s = svy.Sample(df, Design(stratum="st", psu="psu")).singleton.skip()
    assert "SingletonSpec.skip(['b'])" in s.__plain_str__()
    assert "SingletonSpec.skip(['b'])" in str(s)


def test_copy_and_pickle_keep_the_spec():
    d = Design(stratum="st", psu="psu", wgt="w", singleton=SPEC)
    assert copy.deepcopy(d) == d
    assert copy.copy(d) == d
    assert pickle.loads(pickle.dumps(d)) == d


def test_design_stays_frozen():
    d = Design(stratum="st", singleton=SPEC)
    with pytest.raises(AttributeError, match="frozen"):
        d.singleton = None  # type: ignore[misc]


def test_unknown_keyword_is_refused():
    with pytest.raises(TypeError, match="unexpected keyword"):
        Design(stratum="st", singletons=SPEC)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword"):
        Design(stratum="st").update(singletons=SPEC)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# design.update: the rule
# ---------------------------------------------------------------------------

BASE = Design(stratum="st", psu="psu", ssu="hh", wgt="w", singleton=SPEC)


@pytest.mark.parametrize(
    "edit",
    [
        {"wgt": "w2"},
        {"prob": "p"},
        {"pop_size": "N"},
        {"wr": True},
        {"case_id": "id"},
        {"mos": "m"},
    ],
)
def test_edits_that_keep_the_strata_keep_the_spec(edit):
    assert BASE.update(**edit).singleton == SPEC


@pytest.mark.parametrize(
    "edit",
    [
        {"stratum": "st2"},
        {"stratum": ("st", "reg")},
        {"psu": "psu2"},
        {"psu": None},
        {"ssu": None},
        {"ssu": "hh2"},
    ],
)
def test_a_strata_psu_ssu_edit_clears_the_spec_with_one_warning(edit):
    with pytest.warns(UserWarning, match=r"singleton handling \(collapse\) cleared") as rec:
        new = BASE.update(**edit)
    assert len(rec) == 1
    assert new.singleton is None
    assert "stratum/psu/ssu columns changed" in str(rec[0].message)
    assert rec[0].filename == __file__


def test_removing_the_stratum_clears_the_spec():
    with pytest.warns(UserWarning, match="cleared"):
        assert BASE.update(stratum=None).singleton is None


def test_to_a_tuple_and_back_does_not_bring_it_back():
    with pytest.warns(UserWarning):
        tup = BASE.update(stratum=("st", "reg"))
    assert tup.update(stratum="st").singleton is None


def test_panel_variance_psu_change_clears_the_spec():
    d = Design(case_id="id", wave="wave", stratum="st", singleton=SPEC)
    with pytest.warns(UserWarning, match="cleared"):
        assert d.update(case_id="id2").singleton is None


def test_passing_a_spec_with_the_edit_keeps_it():
    other = SingletonSpec.skip(["x"])
    assert BASE.update(stratum="st2", singleton=other).singleton == other
    assert BASE.update(singleton=None).singleton is None


def test_setting_the_same_strata_is_not_an_edit():
    assert BASE.update(stratum="st", psu="psu", ssu="hh").singleton == SPEC


def test_fill_missing_keeps_an_existing_spec():
    assert BASE.fill_missing(singleton=SingletonSpec.skip(["x"])).singleton == SPEC
    d = Design(stratum="st").fill_missing(singleton=SPEC)
    assert d.singleton == SPEC


def test_the_spec_does_not_disturb_the_other_parts_rules():
    rec = WgtAdjustment(kind="raking", prev_wgt="w", new_wgt="rk", cells=("__svy_cells_a",))
    d = Design(
        stratum="st",
        psu="psu",
        wgt="rk",
        rep_wgts=BrrWgts(prefix="r", n_reps=4),
        wgt_adjustment=rec,
        singleton=SPEC,
    )
    with pytest.warns(UserWarning, match="Replicate weights 'r' go with weight 'rk'"):
        moved = d.update(wgt="w")
    assert moved.rep_wgts is None and moved.wgt_adjustment is None
    assert moved.singleton == SPEC


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def test_columns_are_unchanged_by_the_spec():
    plain = Design(stratum=("st", "reg"), psu="psu", ssu="hh", wgt="w")
    assert plain.update(singleton=SPEC).columns() == plain.columns()


# ---------------------------------------------------------------------------
# Saved form and code form
# ---------------------------------------------------------------------------

SPECS = {
    "certainty": SingletonSpec.certainty(["d", "e"]),
    "skip": SingletonSpec.skip(["d"]),
    "scale": SingletonSpec.scale(["d", "e"]),
    "center": SingletonSpec.center(["e"]),
    "collapse": SingletonSpec.collapse({"d": "a", "e": "c"}),
    "pool": SingletonSpec.pool(["d", "e"]),
    "pool_named": SingletonSpec.pool(["d", "e"], name="rest"),
    "ints": SingletonSpec.skip([4, 5]),
    "floats": SingletonSpec.skip([4.5]),
    "bools": SingletonSpec.skip([True]),
    "null": SingletonSpec.skip([None, "d"]),
    "tuple": SingletonSpec.certainty([("N", 4), ("S", 5)]),
    "tuple_null": SingletonSpec.skip([("N", None)]),
    "tuple_collapse": SingletonSpec.collapse({("N", 4): ("N", 1), ("S", 5): ("S", 3)}),
}


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_saved_form_round_trip(spec):
    stratum = ("reg", "code") if isinstance(spec.handled[0], tuple) else "st"
    d = Design(stratum=stratum, psu="psu", wgt="w", singleton=spec)
    back = _round_trip(d)
    assert back == d
    assert back.singleton == spec
    assert hash(back) == hash(d)
    for v in (*back.singleton.strata, *[x for pair in back.singleton.mapping for x in pair]):
        assert not isinstance(v, list)


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_code_form_round_trip(spec):
    stratum = ("reg", "code") if isinstance(spec.handled[0], tuple) else "st"
    d = Design(stratum=stratum, psu="psu", wgt="w", singleton=spec)
    code = spec._to_code()
    assert code.startswith("svy.SingletonSpec.")
    assert eval(code, {"svy": svy}) == spec
    assert eval(d._to_code(), {"svy": svy}) == d


def test_dates_are_saved_and_rendered_as_iso_strings():
    spec = SingletonSpec.skip([dt.date(2020, 1, 4)])
    raw = json.loads(to_json(Design(stratum="day", singleton=spec)))
    assert raw["singleton"] == {
        "method": "skip",
        "strata": ["2020-01-04"],
        "mapping": [],
        "name": None,
    }
    assert eval(spec._to_code(), {"svy": svy}) == SingletonSpec.skip(["2020-01-04"])


def test_payload_shape():
    raw = json.loads(to_json(Design(stratum=("reg", "code"), singleton=SPECS["tuple_collapse"])))
    assert raw["singleton"]["method"] == "collapse"
    assert raw["singleton"]["mapping"] == [[["N", 4], ["N", 1]], [["S", 5], ["S", 3]]]
    assert raw["parts"] is None
    assert json.loads(to_json(Design(stratum="st")))["singleton"] is None


def test_the_struct_mirrors_every_live_field():
    """A field added to SingletonSpec must be added to its saved form too."""
    assert set(SingletonSpec.__struct_fields__) == set(SingletonSpecData.__struct_fields__)


def test_design_data_has_every_field_and_part():
    from svy.core import design_parts
    from svy.core.design import _FIELDS

    saved = set(DesignData.__struct_fields__) - {"kind", "schema_version", "parts"}
    assert saved == {*_FIELDS, *(p.name for p in design_parts.registered())}


def test_serialize_gives_the_struct():
    data = serialize(Design(stratum="st", singleton=SPEC))
    assert isinstance(data.singleton, SingletonSpecData)
    assert msgspec.json.decode(msgspec.json.encode(data), type=DesignData) == data


def test_a_payload_whose_spec_has_no_stratum_fails_like_the_design():
    raw = json.loads(to_json(Design(stratum="st", singleton=SPEC)))
    raw["stratum"] = None
    with pytest.raises(ValueError, match="no stratum"):
        to_design(from_json(json.dumps(raw).encode()))


def test_a_payload_without_the_field_still_loads():
    """0.1 payloads written before the singleton field existed."""
    raw = json.loads(to_json(Design(stratum="st", wgt="w")))
    del raw["singleton"], raw["parts"]
    assert to_design(from_json(json.dumps(raw).encode())) == Design(stratum="st", wgt="w")


def test_no_warning_anywhere_in_a_round_trip():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for spec in SPECS.values():
            stratum = ("reg", "code") if isinstance(spec.handled[0], tuple) else "st"
            _round_trip(Design(stratum=stratum, singleton=spec))
