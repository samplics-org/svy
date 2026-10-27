# tests/svy/serialization/test_serialize_renderer_fields.py
"""
What a renderer needs from a payload without the live result: a table's units,
a t-test by domain, a fit's term structure, and decoding from a dict.
"""

import copy
import json
import math

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors.serialization_errors import SerializationError
from svy.serialize import (
    SCHEMA_VERSION,
    TTestOneGroupByData,
    TTestTwoGroupsByData,
    from_dict,
    from_json,
    serialize,
    to_dict,
    to_json,
    to_polars,
)


@pytest.fixture(scope="module")
def sample() -> svy.Sample:
    rng = np.random.default_rng(11)
    n = 600
    df = pl.DataFrame(
        {
            "stratum": np.repeat(["s1", "s2", "s3", "s4"], n // 4),
            "psu": np.arange(n) // 10,
            "w": rng.uniform(1, 3, n),
            "region": rng.choice(["East", "North", "West"], n),
            "sex": rng.integers(1, 3, n),
            "age": rng.integers(18, 80, n),
            "y": rng.normal(50, 10, n),
        }
    )
    df = df.with_columns(d=(pl.col("y") + (pl.col("sex") == 2) * 3 > 52).cast(int))
    s = svy.Sample(df, svy.Design(stratum="stratum", psu="psu", wgt="w"))
    s.set_value_labels("sex", {1: "Female", 2: "Male"})
    return s


def _round_trip(result):
    return from_json(to_json(result))


def test_schema_version():
    assert SCHEMA_VERSION == "svy-result/0.10"


class TestTableUnits:
    @pytest.mark.parametrize(
        "units, expected",
        [
            ("proportion", "proportion"),
            ("percent", "percent"),
            ("pct", "percent"),
            ("count", "count"),
        ],
    )
    def test_recorded(self, sample, units, expected):
        table = sample.categorical.tabulate("region", units=units)
        assert table.units == expected
        assert _round_trip(table).units == expected

    def test_default_is_proportion(self, sample):
        assert serialize(sample.categorical.tabulate("region", "sex")).units == "proportion"

    def test_count_total_is_a_count(self, sample):
        table = sample.categorical.tabulate("region", units="count", count_total=1000)
        assert serialize(table).units == "count"

    def test_survives_update(self, sample):
        table = sample.categorical.tabulate("region", units="percent")
        assert table.update(alpha=0.1).units == "percent"

    def test_older_payload_decodes_without_units(self, sample):
        raw = json.loads(to_json(sample.categorical.tabulate("region")))
        raw.pop("units")
        raw["schema_version"] = "svy-result/0.8"
        assert from_json(json.dumps(raw)).units is None


class TestTTestByDomain:
    @pytest.fixture
    def one(self, sample):
        return sample.categorical.ttest("y", mean_h0=50, by="region")

    @pytest.fixture
    def two(self, sample):
        return sample.categorical.ttest("y", group="sex", by="region")

    def test_one_group_round_trips(self, one):
        data = _round_trip(one)
        assert isinstance(data, TTestOneGroupByData)
        assert data.kind == "ttest_one_group_by"
        assert data == serialize(one)
        assert (data.y, data.by) == ("y", "region")
        assert len(data.results) == 3

    def test_two_groups_round_trips(self, two):
        data = _round_trip(two)
        assert isinstance(data, TTestTwoGroupsByData)
        assert data.kind == "ttest_two_groups_by"
        assert data == serialize(two)

    def test_members_carry_their_level(self, one):
        data = serialize(one)
        assert [r.diff[0].by_level for r in data.results] == ["East", "North", "West"]

    @pytest.mark.parametrize("fixture", ["one", "two"])
    @pytest.mark.parametrize("component", ["test", "estimates"])
    def test_table_matches_live(self, request, fixture, component):
        live = request.getfixturevalue(fixture)
        assert to_polars(_round_trip(live), component=component).equals(live.to_polars(component))

    def test_row_index_counts_through_members(self, two):
        df = to_polars(serialize(two), component="estimates", row_index="i")
        assert df["i"].to_list() == list(range(6))

    def test_where_clause_kept(self, sample):
        res = sample.categorical.ttest("y", mean_h0=50, by="region", where=pl.col("age") > 30)
        assert serialize(res).where_clause == res.where_clause is not None


class TestPairedTTest:
    @pytest.mark.parametrize("kw", [{}, {"group": "sex"}, {"by": "region"}])
    def test_names_both_variables(self, sample, kw):
        res = sample.categorical.ttest("y", y_pair="age", **kw)
        assert res.pair == ("y", "age")
        assert _round_trip(res).pair == ["y", "age"]

    def test_by_members_carry_the_pair(self, sample):
        data = _round_trip(sample.categorical.ttest("y", y_pair="age", by="region"))
        assert [r.pair for r in data.results] == [["y", "age"]] * 3

    def test_two_groups_is_marked_paired(self, sample):
        res = sample.categorical.ttest("y", y_pair="age", group="sex")
        assert res.paired is True
        assert _round_trip(res).paired is True

    @pytest.mark.parametrize("kw", [{"mean_h0": 50}, {"group": "sex"}, {"by": "region"}])
    def test_unpaired_has_no_pair(self, sample, kw):
        res = sample.categorical.ttest("y", **kw)
        assert res.pair is None
        assert getattr(res, "paired", False) is False
        assert _round_trip(res).pair is None

    def test_y_is_the_difference_in_pair_order(self, sample):
        res = sample.categorical.ttest("y", y_pair="age")
        df = sample.data
        expected = ((df["y"] - df["age"]) * df["w"]).sum() / df["w"].sum()
        assert res.y == "svy_y_minus_age"
        assert res.estimates[0].est == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("kw", [{}, {"group": "sex"}, {"by": "region"}])
    def test_header_names_both_variables(self, sample, kw):
        res = sample.categorical.ttest("y", y_pair="age", **kw)
        assert "Y = 'y' - 'age'" in res.__plain_str__()

    def test_older_payload_decodes_without_pair(self, sample):
        raw = json.loads(to_json(sample.categorical.ttest("y", y_pair="age")))
        raw.pop("pair")
        raw["schema_version"] = "svy-result/0.9"
        assert from_json(json.dumps(raw)).pair is None


class TestGLMTerms:
    @pytest.fixture
    def fit(self, sample):
        return sample.glm.fit(
            "d",
            x=[svy.Cat("sex"), svy.Cat("region", ref="North"), svy.Cross(svy.Cat("sex"), "age")],
            family="binomial",
        )

    def test_categorical_term(self, fit):
        terms = {t.name: t for t in _round_trip(fit).terms}
        region = terms["region"]
        assert region.type == "categorical"
        assert region.reference == "North"
        assert region.levels == ["North", "East", "West"]

    def test_continuous_term(self, fit):
        age = next(t for t in serialize(fit).terms if t.name == "age")
        assert (age.type, age.levels, age.reference) == ("continuous", None, None)

    def test_coefficient_names_its_variable_and_level(self, fit):
        coefs = {c.term: c for c in _round_trip(fit).coefs}
        (part,) = coefs["sex_2"].parts
        assert (part.variable, part.level) == ("sex", 2)
        assert coefs["_intercept_"].parts is None

    def test_interaction_has_one_part_per_factor(self, fit):
        coefs = {c.term: c for c in serialize(fit).coefs}
        parts = coefs["sex_2:age"].parts
        assert [(p.variable, p.level) for p in parts] == [("sex", 2), ("age", None)]

    def test_value_labels_travel(self, fit):
        data = _round_trip(fit)
        sex = next(t for t in data.terms if t.name == "sex")
        assert {(lv.code, lv.label) for lv in sex.level_labels} == {(1, "Female"), (2, "Male")}
        assert next(c for c in data.coefs if c.term == "sex_2").label == "sex_Male"

    def test_level_is_not_parsed_from_the_name(self, sample):
        """An underscore in the variable name cannot be split off the level."""
        s = sample.clone() if hasattr(sample, "clone") else sample
        data = s.data.with_columns(pl.col("region").alias("home_region"))
        s2 = svy.Sample(data, svy.Design(stratum="stratum", psu="psu", wgt="w"))
        fit = s2.glm.fit("d", x=[svy.Cat("home_region")], family="binomial")
        coefs = {c.term: c for c in serialize(fit).coefs}
        (part,) = coefs["home_region_West"].parts
        assert (part.variable, part.level) == ("home_region", "West")

    def test_older_payload_decodes_without_terms(self, fit):
        raw = json.loads(to_json(fit))
        raw.pop("terms")
        for c in raw["coefs"]:
            c.pop("parts", None)
            c.pop("label", None)
        data = from_json(json.dumps(raw))
        assert data.terms is None
        assert all(c.parts is None for c in data.coefs)


class TestFromDict:
    def test_round_trips_to_dict(self, sample):
        result = sample.estimation.mean("y", by="region")
        assert from_dict(to_dict(result)) == serialize(result)

    def test_reads_parsed_json(self, sample):
        result = sample.categorical.ttest("y", group="sex", by="region")
        assert from_dict(json.loads(to_json(result))) == serialize(result)

    def test_restores_nonfinite_from_parsed_json(self, sample):
        payload = json.loads(to_json(sample.estimation.mean("y", by="region")))
        payload["nonfinite"] = {"/estimates/0/cv": "inf"}
        assert math.isinf(from_dict(payload).estimates[0].cv)

    def test_does_not_modify_its_input(self, sample):
        payload = json.loads(to_json(sample.estimation.mean("y", by="region")))
        payload["nonfinite"] = {"/estimates/0/cv": "nan"}
        before = copy.deepcopy(payload)
        from_dict(payload)
        assert payload == before

    def test_design(self, sample):
        assert from_dict(to_dict(sample.design)) == serialize(sample.design)

    def test_missing_kind_names_from_dict(self):
        with pytest.raises(SerializationError) as exc:
            from_dict({"schema_version": SCHEMA_VERSION})
        assert exc.value.code == "PAYLOAD_MISSING_KIND"
        assert exc.value.where == "from_dict"

    def test_unknown_kind_names_from_dict(self):
        with pytest.raises(SerializationError) as exc:
            from_dict({"kind": "nope"})
        assert exc.value.code == "PAYLOAD_UNKNOWN_KIND"
        assert exc.value.where == "from_dict"
