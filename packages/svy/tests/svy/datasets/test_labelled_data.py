# tests/svy/datasets/test_labelled_data.py
"""The bundled datasets hold codes, carry their labels in the registry, and load as a ready Sample."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import svy
import svy.datasets as d

from svy.datasets import Dataset, DatasetCatalog
from svy.errors import Code, DatasetError, UnknownNameError


TEST_DATA = Path(__file__).parents[2] / "test_data"
SLUGS = sorted(d.info(source="bundled").slugs)
WB = [s for s in SLUGS if s.endswith("_wb_2023")]
# The same rows with the labels applied, written from the label-text masters.
TEXT_ORACLES = ["ea_frame_wb_2023", "hld_sample_wb_2023", "ind_sample_wb_2023"]


def _entry(slug: str) -> Dataset:
    return d.info(slug, source="bundled")


def _codes(spec: dict, dtype: pl.DataType) -> dict:
    cast = int if dtype.is_integer() else float if dtype.is_float() else str
    return {cast(code): label for code, label in spec.get("values", {}).items()}


@pytest.fixture(scope="module")
def hld() -> svy.Sample:
    return d.load("hld_sample_wb_2023", source="bundled")


# --------------------------------------------------------------------------- #
# The registry describes its own file
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", SLUGS)
def test_every_labelled_variable_is_a_column(slug):
    df = d.load(slug, source="bundled").data
    assert _entry(slug).variables
    assert set(_entry(slug).variables) <= set(df.columns)


@pytest.mark.parametrize("slug", WB)
def test_every_wb_column_has_a_label(slug):
    df = d.load(slug, source="bundled").data
    variables = _entry(slug).variables
    assert set(variables) == set(df.columns)
    assert all(spec["label"] for spec in variables.values())


@pytest.mark.parametrize("slug", SLUGS)
def test_every_observed_code_is_labelled(slug):
    df = d.load(slug, source="bundled").data
    for var, spec in _entry(slug).variables.items():
        if "values" not in spec:
            continue
        observed = set(df[var].drop_nulls().unique().to_list())
        assert observed <= set(_codes(spec, df.schema[var])), var


@pytest.mark.parametrize("slug", WB)
def test_wb_data_holds_codes_not_text(slug):
    df = d.load(slug, source="bundled").data
    for var, spec in _entry(slug).variables.items():
        if "values" in spec:
            assert df.schema[var] == pl.Int64, var


@pytest.mark.parametrize("slug", TEXT_ORACLES)
def test_codes_decode_to_the_label_text_of_the_same_rows(slug):
    coded = d.load(slug, source="bundled").data
    variables = _entry(slug).variables
    decoded = coded.with_columns(
        pl.col(var).replace_strict(_codes(spec, pl.Int64), return_dtype=pl.String)
        for var, spec in variables.items()
        if "values" in spec
    )
    text = pl.read_parquet(TEST_DATA / f"wb_text_{slug}_28102026.parquet")
    assert decoded.equals(text)


@pytest.mark.parametrize("slug", WB)
def test_wb_versions_bumped_so_text_caches_are_not_reused(slug):
    assert _entry(slug).version == "2.0.0"


def test_label_sets_are_full_instrument_lists():
    # 96 means a different thing in each list: the codes, not the text, carry it.
    variables = _entry("hld_sample_wb_2023").variables
    assert variables["water"]["values"]["96"] == "Other"
    assert variables["geo2"]["values"]["96"] == "geo_09_06"  # a district outside the bundle
    assert "96" not in variables["electricity"]["values"]
    assert variables["electricity"]["values"] == {"0": "No", "1": "Yes"}


def test_panel_labels():
    variables = _entry("panel_syn_2026").variables
    assert set(variables) == {"sex", "urban", "age_grp", "emp", "resp"}
    assert variables["urban"]["values"] == {"0": "Rural", "1": "Urban"}
    assert variables["resp"]["values"] == {"rr": "Respondent", "nr": "Nonrespondent"}


# --------------------------------------------------------------------------- #
# load gives a ready Sample
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", SLUGS)
def test_load_applies_the_entry_design(slug):
    design = _entry(slug).design
    expected = svy.Design(**design) if design else svy.Design()
    assert d.load(slug, source="bundled").design == expected


def test_frame_comes_back_ready_for_pps():
    frame = d.load("ea_frame_wb_2023", source="bundled")
    assert frame.design.mos == "n_hlds_census"
    assert frame.design.stratum == ("geo1", "urbrur")


@pytest.mark.parametrize("slug", SLUGS)
def test_load_puts_the_entry_labels_in_meta(slug):
    sample = d.load(slug, source="bundled")
    variables = _entry(slug).variables
    schema = sample.data.schema
    assert dict(sample.meta.var_labels) == {v: spec["label"] for v, spec in variables.items()}
    assert {v: dict(m) for v, m in sample.meta.value_labels.items()} == {
        v: _codes(spec, schema[v]) for v, spec in variables.items() if "values" in spec
    }


def test_value_labels_match_integer_codes(hld):
    assert hld.meta.value_labels["water"][96] == "Other"
    assert hld.meta.value_labels["urbrur"] == {1: "Rural", 2: "Urban"}


@pytest.mark.parametrize(
    "var, categorical",
    [("geo2", True), ("water", True), ("quint_nat", True), ("hhsize", False), ("ea", False)],
)
def test_measurement_type_follows_label_coverage(hld, var, categorical):
    assert hld.meta.get(var).is_categorical is categorical


def test_tabulate_prints_labels(hld):
    out = str(hld.categorical.tabulate("water"))
    assert "Main source of drinking water" in out
    assert "Piped into dwelling" in out
    assert "Other" in out


def test_cat_ref_by_label_resolves_to_its_code(hld):
    fit = hld.glm.fit(y="pc_exp", x=[svy.Cat("urbrur", ref="Urban")])
    assert [c.term for c in fit.coefs] == ["_intercept_", "urbrur_1"]
    assert [c.label for c in fit.coefs] == [None, "urbrur_Rural"]


def test_where_takes_codes(hld):
    urban = hld.estimation.mean("pc_exp", where=svy.col("urbrur") == 2)
    by_area = hld.estimation.mean("pc_exp", by="urbrur")
    assert urban.to_dicts()[0]["est"] == pytest.approx(by_area.to_dicts()[1]["est"], rel=1e-12)


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #


def test_info_lists_every_bundled_entry():
    cat = d.info(source="bundled")
    assert isinstance(cat, DatasetCatalog)
    assert set(cat.slugs) == {
        "ea_frame_wb_2023",
        "hld_pop_wb_2023",
        "hld_sample_wb_2023",
        "ind_sample_wb_2023",
        "panel_syn_2026",
    }


@pytest.mark.parametrize("slug", SLUGS)
def test_info_name_is_that_entry(slug):
    ds = d.info(slug, source="bundled")
    assert isinstance(ds, Dataset)
    assert ds == d.info(source="bundled").get(slug)
    assert slug in str(ds)
    assert slug in str(d.info(source="bundled"))


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", ["bundled", "auto"])
def test_unknown_name_raises_the_same_error_in_load_and_info(monkeypatch, source):
    monkeypatch.setenv("SVYLAB_OFFLINE", "1")
    errors = []
    for call in (d.load, d.info):
        with pytest.raises(DatasetError) as err:
            call("no_such_dataset", source=source)
        errors.append(err.value)
    assert {type(e) for e in errors} == {DatasetError.KeyError}
    assert {e.code for e in errors} == {Code.DATASET_NOT_BUNDLED}
    assert all("no_such_dataset" in e.detail for e in errors)


@pytest.mark.parametrize(
    "name, target",
    [("catalog", "svy.datasets.info()"), ("describe", "svy.datasets.info(name)")],
)
def test_removed_names_point_to_info(name, target):
    with pytest.raises(UnknownNameError) as err:
        getattr(d, name)
    e = err.value
    assert isinstance(e, AttributeError)
    assert e.code == Code.NAME_MOVED
    assert e.detail == f"svy.datasets.{name} moved to {target} in 0.36.0."
    assert e.expected == target
    assert e.hint == f"Use {target}."
    assert name not in d.__all__


def test_other_missing_names_stay_plain_attribute_errors():
    with pytest.raises(AttributeError, match="no_such_thing"):
        d.no_such_thing  # noqa: B018
