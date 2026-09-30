# tests/svy/core/test_singleton_rule.py
"""svy.Singleton and Design.singleton: the rule declared on the design.

What the rule does to a sample (applied to the singletons of the data, again
after every change) is in test_singleton_design_part.py.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import pickle
import warnings

import msgspec
import numpy as np
import pytest

import svy

from svy.core.design import Design, Singleton, WgtAdjustment
from svy.core.enumerations import SingletonDomains, SingletonMethod
from svy.core.repwgts import BrrWgts
from svy.errors import MethodError, SerializationError
from svy.serialize import (
    DesignData,
    SingletonData,
    from_json,
    serialize,
    to_design,
    to_json,
)


pytestmark = pytest.mark.filterwarnings("error")

METHODS = ["center", "scale", "skip", "self_representing", "collapse", "pool"]


def _round_trip(design: Design) -> Design:
    return to_design(from_json(to_json(design)))


# ---------------------------------------------------------------------------
# The type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_each_method_with_its_defaults(method):
    rule = Singleton(method)
    assert rule.method is SingletonMethod(method) and rule.method == method
    assert rule.domains is SingletonDomains.STANDARD
    assert rule.on_domain_singletons == "ignore"
    assert rule.using == ("smallest" if method == "collapse" else None)
    assert rule.name == ("__pooled__" if method == "pool" else None)
    assert (rule.within, rule.order_by, rule.descending, rule.rstate) == (None, None, False, None)


def test_method_as_the_enum_or_in_any_case():
    assert Singleton(SingletonMethod.POOL) == Singleton("pool") == Singleton(" Pool ")


def test_method_is_positional_and_the_rest_keyword_only():
    assert Singleton(method="center") == Singleton("center")
    with pytest.raises(TypeError):
        Singleton("center", "apply")  # type: ignore[misc]


def test_unknown_method_lists_the_choices():
    with pytest.raises(MethodError) as err:
        Singleton("adjust")  # type: ignore[arg-type]
    assert err.value.code == "INVALID_CHOICE"
    assert err.value.expected == METHODS


def test_certainty_points_to_self_representing_and_to_skip():
    with pytest.raises(MethodError, match="self_representing") as err:
        Singleton("certainty")  # type: ignore[arg-type]
    assert "method='skip'" in err.value.hint


@pytest.mark.parametrize(
    "kwargs, param",
    [
        ({"using": "largest"}, "using"),
        ({"within": "region"}, "within"),
        ({"order_by": "code"}, "order_by"),
        ({"descending": True}, "descending"),
        ({"rstate": 1}, "rstate"),
    ],
)
@pytest.mark.parametrize("method", ["center", "scale", "skip", "self_representing", "pool"])
def test_collapse_options_belong_to_collapse(method, kwargs, param):
    with pytest.raises(MethodError, match=f"{param} applies to method='collapse' only") as err:
        Singleton(method, **kwargs)
    assert err.value.code == "INVALID_SINGLETON_RULE" and err.value.param == param


@pytest.mark.parametrize("method", ["center", "scale", "skip", "self_representing", "collapse"])
def test_name_belongs_to_pool(method):
    with pytest.raises(MethodError, match="name applies to method='pool' only"):
        Singleton(method, name="rest")


def test_pool_name_must_be_a_non_empty_string():
    with pytest.raises(MethodError, match="non-empty string"):
        Singleton("pool", name="")


@pytest.mark.parametrize("using", ["next", "previous", "smallest", "largest"])
def test_collapse_strategies(using):
    assert Singleton("collapse", using=using).using == using


def test_unknown_strategy():
    with pytest.raises(MethodError) as err:
        Singleton("collapse", using="nearest")
    assert err.value.code == "INVALID_CHOICE"


def test_collapse_mapping_is_stored_as_pairs():
    rule = Singleton("collapse", using={"d": "a", "e": "c"})
    assert rule.using == (("d", "a"), ("e", "c"))
    assert rule.mapping == {"d": "a", "e": "c"}
    assert Singleton("collapse").mapping is None


def test_collapse_mapping_keeps_native_values_and_tuples():
    rule = Singleton("collapse", using={("N", 4): ["N", 1], 5: 3, dt.date(2020, 1, 5): None})
    assert rule.using == ((("N", 4), ("N", 1)), (5, 3), (dt.date(2020, 1, 5), None))


@pytest.mark.parametrize(
    "using, match",
    [
        ({}, "empty"),
        ({"d": "e", "e": "a"}, "must not be singleton strata"),
        ([("d", "a"), ("d", "b")], "twice"),
        ({"d": {"x": 1}}, "strategy name, a mapping"),
        (42, "strategy name, a mapping"),
    ],
)
def test_invalid_mappings(using, match):
    with pytest.raises(MethodError, match=match):
        Singleton("collapse", using=using)


def test_collapse_mapping_keeps_1_and_true_apart():
    assert len(Singleton("collapse", using=[(1, 3), (True, 3)]).using) == 2


def test_collapse_callable():
    def pick(singleton, candidates):
        return candidates[0].stratum_key

    assert Singleton("collapse", using=pick).using is pick


@pytest.mark.parametrize("param", ["within", "order_by"])
def test_columns_as_a_name_or_a_list(param):
    assert getattr(Singleton("collapse", **{param: "region"}), param) == ("region",)
    rule = Singleton("collapse", **{param: ["region", "urban", "region"]})
    assert getattr(rule, param) == ("region", "urban")
    for bad in ("", [], ["region", 3]):
        with pytest.raises(MethodError, match=f"{param} names one column"):
            Singleton("collapse", **{param: bad})


def test_rstate_is_not_a_bool():
    with pytest.raises(MethodError, match="rstate"):
        Singleton("collapse", rstate=True)


def test_rule_is_frozen_and_hashable():
    rule = Singleton("collapse", using={"d": "a"}, within="region", rstate=3)
    with pytest.raises(AttributeError):
        rule.method = "skip"  # type: ignore[misc]
    assert hash(rule) == hash(Singleton("collapse", using={"d": "a"}, within="region", rstate=3))
    assert rule != Singleton("collapse", using={"d": "b"}, within="region", rstate=3)


def test_exported():
    assert svy.Singleton is Singleton and svy.core.Singleton is Singleton
    assert svy.SingletonMethod is SingletonMethod
    assert not hasattr(svy, "SingletonSpec")


@pytest.mark.parametrize(
    "rule, shown",
    [
        (Singleton("center"), "Singleton('center')"),
        (
            Singleton("skip", on_domain_singletons="warn"),
            "Singleton('skip', on_domain_singletons='warn')",
        ),
        (Singleton("scale", domains="apply"), "Singleton('scale', domains='apply')"),
        (Singleton("collapse"), "Singleton('collapse')"),
        (Singleton("collapse", using={3: 1}), "Singleton('collapse', using={3: 1})"),
        (
            Singleton(
                "collapse",
                using="next",
                within=["r", "u"],
                order_by="c",
                descending=True,
                rstate=7,
            ),
            "Singleton('collapse', using='next', within=['r', 'u'], order_by='c', "
            "descending=True, rstate=7)",
        ),
        (Singleton("pool"), "Singleton('pool')"),
        (Singleton("pool", name="rest"), "Singleton('pool', name='rest')"),
        (Singleton("self_representing"), "Singleton('self_representing')"),
        (
            Singleton("collapse", using={dt.date(2020, 1, 5): dt.date(2020, 1, 4)}),
            "Singleton('collapse', using={datetime.date(2020, 1, 5): datetime.date(2020, 1, 4)})",
        ),
    ],
)
def test_repr_is_the_constructor_call(rule, shown):
    assert repr(rule) == shown
    assert rule._to_code() == "svy." + shown
    assert eval(rule._to_code(), {"svy": svy, "datetime": dt}) == rule


def test_a_callable_or_generator_rule_cannot_be_written_as_code():
    with pytest.raises(MethodError, match="callable"):
        Singleton("collapse", using=lambda s, c: c[0])._to_code()
    with pytest.raises(MethodError, match="Generator"):
        Singleton("collapse", rstate=np.random.default_rng(1))._to_code()


# ---------------------------------------------------------------------------
# On the Design
# ---------------------------------------------------------------------------

RULE = Singleton("collapse", using={"d": "a"})


def test_design_carries_the_rule():
    d = Design(stratum="st", psu="psu", wgt="w", singleton=RULE)
    assert d.singleton == RULE
    assert Design(stratum="st").singleton is None


@pytest.mark.parametrize("method", METHODS)
def test_a_string_is_shorthand(method):
    assert Design(stratum="st", singleton=method).singleton == Singleton(method)
    assert Design(stratum="st").update(singleton=method).singleton == Singleton(method)


def test_design_needs_a_stratum_for_a_rule():
    with pytest.raises(ValueError, match="no stratum"):
        Design(psu="psu", singleton=RULE)


def test_design_rejects_other_types():
    with pytest.raises(MethodError, match="wrong type") as err:
        Design(stratum="st", singleton={"d": "a"})  # type: ignore[arg-type]
    assert err.value.expected == "svy.Singleton | str | None"


def test_equality_and_hash_include_the_rule():
    a = Design(stratum="st", psu="psu", singleton=RULE)
    b = Design(stratum="st", psu="psu")
    assert a != b
    assert a == Design(stratum="st", psu="psu", singleton=Singleton("collapse", using={"d": "a"}))
    assert hash(a) == hash(Design(stratum="st", psu="psu", singleton=RULE))


def test_printed_design_and_sample_show_the_rule():
    import polars as pl

    d = Design(stratum="st", psu="psu", singleton="skip")
    assert "Singleton        : Singleton('skip')" in d.__plain_str__()
    assert "singleton=Singleton('skip')" in repr(d)
    assert "Singleton" not in Design(stratum="st").__plain_str__()
    df = pl.DataFrame({"st": ["a", "a", "b"], "psu": ["1", "2", "3"], "y": [1.0, 2.0, 3.0]})
    s = svy.Sample(df, d)
    assert "Singleton('skip')" in s.__plain_str__() and "Singleton('skip')" in str(s)


def test_copy_and_pickle_keep_the_rule():
    d = Design(stratum="st", psu="psu", wgt="w", singleton=RULE)
    assert copy.deepcopy(d) == d
    assert copy.copy(d) == d
    assert pickle.loads(pickle.dumps(d)) == d


def test_design_stays_frozen():
    d = Design(stratum="st", singleton=RULE)
    with pytest.raises(AttributeError, match="frozen"):
        d.singleton = None  # type: ignore[misc]


def test_unknown_keyword_is_refused():
    with pytest.raises(TypeError, match="unexpected keyword"):
        Design(stratum="st", singletons=RULE)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# design.update: the rule is intent and stays
# ---------------------------------------------------------------------------

BASE = Design(stratum="st", psu="psu", ssu="hh", wgt="w", singleton=RULE)


@pytest.mark.parametrize(
    "edit",
    [
        {"wgt": "w2"},
        {"prob": "p"},
        {"pop_size": "N"},
        {"wr": True},
        {"case_id": "id"},
        {"mos": "m"},
        {"stratum": "st2"},
        {"stratum": ("st", "reg")},
        {"psu": "psu2"},
        {"psu": None},
        {"ssu": None},
        {"ssu": "hh2"},
    ],
)
def test_every_edit_keeps_the_rule_silently(edit):
    assert BASE.update(**edit).singleton == RULE


def test_panel_variance_psu_change_keeps_the_rule():
    d = Design(case_id="id", wave="wave", stratum="st", singleton=RULE)
    assert d.update(case_id="id2").singleton == RULE


def test_removing_the_stratum_drops_the_rule_with_one_warning():
    with pytest.warns(UserWarning, match=r"singleton rule \(collapse\) removed") as rec:
        assert BASE.update(stratum=None).singleton is None
    assert len(rec) == 1 and "the design has no stratum" in str(rec[0].message)


def test_passing_a_rule_with_the_edit_replaces_it():
    assert BASE.update(stratum="st2", singleton="skip").singleton == Singleton("skip")
    assert BASE.update(singleton=None).singleton is None


def test_fill_missing_keeps_an_existing_rule():
    assert BASE.fill_missing(singleton="skip").singleton == RULE
    assert Design(stratum="st").fill_missing(singleton=RULE).singleton == RULE


def test_the_rule_does_not_disturb_the_other_parts_rules():
    rec = WgtAdjustment(kind="raking", prev_wgt="w", new_wgt="rk", cells=("__svy_cells_a",))
    d = Design(
        stratum="st",
        psu="psu",
        wgt="rk",
        rep_wgts=BrrWgts(prefix="r", n_reps=4),
        wgt_adjustment=rec,
        singleton=RULE,
    )
    with pytest.warns(UserWarning, match="Replicate weights 'r' go with weight 'rk'"):
        moved = d.update(wgt="w")
    assert moved.rep_wgts is None and moved.wgt_adjustment is None
    assert moved.singleton == RULE


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def test_columns_add_what_the_rule_reads():
    plain = Design(stratum=("st", "reg"), psu="psu", ssu="hh", wgt="w")
    assert plain.update(singleton=RULE).columns() == plain.columns()
    rule = Singleton("collapse", within="region", order_by=["code", "st"])
    assert set(plain.update(singleton=rule).columns()) == {*plain.columns(), "region", "code"}


# ---------------------------------------------------------------------------
# Saved form and code form
# ---------------------------------------------------------------------------

RULES = {
    "center": Singleton("center"),
    "center_apply_warn": Singleton("center", domains="apply", on_domain_singletons="warn"),
    "scale": Singleton("scale", on_domain_singletons="error"),
    "skip": Singleton("skip"),
    "self_representing": Singleton("self_representing"),
    "pool": Singleton("pool"),
    "pool_named": Singleton("pool", name="rest"),
    "collapse": Singleton("collapse"),
    "collapse_options": Singleton(
        "collapse", using="previous", within="region", order_by=["code"], descending=True, rstate=5
    ),
    "collapse_map": Singleton("collapse", using={"d": "a", "e": "c"}),
    "collapse_ints": Singleton("collapse", using={4: 1, 5.5: 3}),
    "collapse_bool_null": Singleton("collapse", using={True: False, None: "x"}),
    "collapse_tuples": Singleton("collapse", using={("N", 4): ("N", 1), ("S", None): ("S", 3)}),
}


def _stratum(rule):
    mapping = rule.mapping or {}
    return ("reg", "code") if any(isinstance(k, tuple) for k in mapping) else "st"


@pytest.mark.parametrize("rule", RULES.values(), ids=RULES.keys())
def test_saved_form_round_trip(rule):
    d = Design(stratum=_stratum(rule), psu="psu", wgt="w", singleton=rule)
    back = _round_trip(d)
    assert back == d and back.singleton == rule and hash(back) == hash(d)


@pytest.mark.parametrize("rule", RULES.values(), ids=RULES.keys())
def test_code_form_round_trip(rule):
    d = Design(stratum=_stratum(rule), psu="psu", wgt="w", singleton=rule)
    assert rule._to_code().startswith("svy.Singleton(")
    assert eval(rule._to_code(), {"svy": svy}) == rule
    assert eval(d._to_code(), {"svy": svy}) == d


def test_defaults_are_left_out_of_the_payload():
    raw = json.loads(to_json(Design(stratum="st", singleton="center")))
    assert raw["singleton"] == {"method": "center"}
    raw = json.loads(to_json(Design(stratum="st", singleton=RULES["collapse_options"])))
    assert raw["singleton"] == {
        "method": "collapse",
        "using": "previous",
        "within": ["region"],
        "order_by": ["code"],
        "descending": True,
        "rstate": 5,
    }


def test_dates_are_saved_as_iso_strings_and_restored_as_dates():
    rule = Singleton("collapse", using={dt.date(2020, 1, 5): dt.date(2020, 1, 4)})
    raw = json.loads(to_json(Design(stratum="day", singleton=rule)))
    assert raw["singleton"] == {"method": "collapse", "using": [["2020-01-05", "2020-01-04"]]}
    assert raw["temporal"] == {"/singleton/using/0/0": "date", "/singleton/using/0/1": "date"}
    assert _round_trip(Design(stratum="day", singleton=rule)).singleton == rule


def test_tuple_strata_payload_shape():
    raw = json.loads(to_json(Design(stratum=("reg", "code"), singleton=RULES["collapse_tuples"])))
    assert raw["singleton"]["using"] == [[["N", 4], ["N", 1]], [["S", None], ["S", 3]]]
    assert json.loads(to_json(Design(stratum="st")))["singleton"] is None


@pytest.mark.parametrize(
    "rule, match",
    [
        (Singleton("collapse", using=lambda s, c: c[0]), "callable"),
        (Singleton("collapse", rstate=np.random.default_rng(1)), "Generator"),
    ],
)
def test_what_cannot_be_saved_says_why(rule, match):
    with pytest.raises(SerializationError, match=match) as err:
        serialize(Design(stratum="st", singleton=rule))
    assert err.value.code == "SINGLETON_RULE_NOT_SAVABLE"


def test_the_struct_mirrors_every_live_field():
    """A field added to svy.Singleton must be added to its saved form too."""
    assert set(Singleton.__struct_fields__) == set(SingletonData.__struct_fields__)


def test_design_data_has_every_field_and_part():
    from svy.core import design_parts
    from svy.core.design import _FIELDS

    saved = set(DesignData.__struct_fields__) - {"kind", "schema_version", "parts"}
    assert saved == {*_FIELDS, *(p.name for p in design_parts.registered())}


def test_serialize_gives_the_struct():
    data = serialize(Design(stratum="st", singleton=RULE))
    assert isinstance(data.singleton, SingletonData)
    assert msgspec.json.decode(msgspec.json.encode(data), type=DesignData) == data


def test_a_payload_whose_rule_has_no_stratum_fails_like_the_design():
    raw = json.loads(to_json(Design(stratum="st", singleton=RULE)))
    raw["stratum"] = None
    with pytest.raises(ValueError, match="no stratum"):
        to_design(from_json(json.dumps(raw).encode()))


def test_a_payload_without_the_field_still_loads():
    raw = json.loads(to_json(Design(stratum="st", wgt="w")))
    del raw["singleton"], raw["parts"]
    assert to_design(from_json(json.dumps(raw).encode())) == Design(stratum="st", wgt="w")


def test_no_warning_anywhere_in_a_round_trip():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for rule in RULES.values():
            _round_trip(Design(stratum=_stratum(rule), singleton=rule))
