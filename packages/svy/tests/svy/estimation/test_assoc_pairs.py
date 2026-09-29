# tests/svy/estimation/test_assoc_pairs.py
"""Correlation and covariance rows name their (y, x) pair in frames and keys."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from polars.testing import assert_frame_equal

from svy import Design, Sample, estd
from svy.errors import MethodError
from svy.serialize import from_json, to_json, to_polars


DATA = Path(__file__).resolve().parents[2] / "test_data" / "apistrat.csv"
COLS = ["api00", "api99", "enroll"]
PAIRS = [("api00", "api99"), ("api00", "enroll"), ("api99", "enroll")]


@pytest.fixture
def sample() -> Sample:
    return Sample(pl.read_csv(DATA), Design(wgt="pw", stratum="stype"))


@pytest.fixture
def labelled(sample) -> Sample:
    sample.meta.set_label("api00", "API 2000")
    sample.meta.set_label("stype", "School type")
    sample.meta.set_value_labels("stype", {"E": "Elementary", "H": "High", "M": "Middle"})
    return sample


def _pairs(df: pl.DataFrame) -> list[tuple[str, str]]:
    return list(zip(df["y"], df["x"]))


@pytest.mark.parametrize("verb", ["corr", "cov"])
def test_frame_leads_with_the_pair(sample, verb):
    r = getattr(sample.estimation, verb)(COLS)
    df = r.to_polars()
    assert df.columns[:3] == ["y", "x", "est"]
    assert _pairs(df) == PAIRS
    assert df["est"].to_list() == [p.est for p in r.estimates]


def test_single_pair_is_named(sample):
    df = sample.estimation.corr(("api00", "enroll")).to_polars_printable()
    assert _pairs(df) == [("api00", "enroll")]


def test_pairs_follow_the_domain_in_request_order(sample):
    r = sample.estimation.corr(COLS, by="stype")
    for df in (r.to_polars(), r.to_polars_printable()):
        assert df.columns[:3] == ["stype", "y", "x"]
        assert list(zip(df["stype"], df["y"], df["x"])) == [
            (g, y, x) for g in "EHM" for y, x in PAIRS
        ]


def test_pairs_are_not_sorted_alphabetically(sample):
    wanted = [("enroll", "api99"), ("api00", "api99")]
    assert _pairs(sample.estimation.corr(wanted).to_polars()) == wanted


def test_printed_tables_show_the_pair(sample):
    r = sample.estimation.corr(COLS)
    for text in (r.__plain_str__(), str(r)):
        rows = [line.split() for line in text.splitlines()]
        for y, x in PAIRS:
            assert any(y in row and x in row for row in rows)


def test_display_uses_variable_labels(labelled):
    r = labelled.estimation.corr(COLS, by="stype")
    df = r.to_polars_printable()
    assert df.columns[:3] == ["School type", "y", "x"]
    assert df["y"].to_list()[:3] == ["API 2000", "API 2000", "api99"]
    assert df["School type"].unique(maintain_order=True).to_list() == [
        "Elementary",
        "High",
        "Middle",
    ]


def test_data_view_adds_pair_label_columns(labelled):
    df = labelled.estimation.corr(COLS).to_polars()
    assert df.columns[:4] == ["y", "y_label", "x", "x_label"]
    assert _pairs(df) == PAIRS
    assert df["y_label"].to_list() == ["API 2000", "API 2000", "api99"]
    assert "y_label" not in labelled.estimation.corr(COLS).to_polars(use_labels=False).columns


def test_no_pair_label_columns_without_variable_labels(sample):
    assert sample.estimation.corr(COLS).to_polars().columns[:3] == ["y", "x", "est"]


@pytest.mark.parametrize("use_labels", [True, False])
@pytest.mark.parametrize("by", [None, "stype"])
def test_saved_frame_matches_the_live_frame(labelled, use_labels, by):
    r = labelled.estimation.corr(COLS, by=by)
    back = from_json(to_json(r))
    assert_frame_equal(to_polars(back, use_labels=use_labels), r.to_polars(use_labels=use_labels))


def test_domain_named_like_a_pair_column_raises(sample):
    s = sample.wrangling.rename_columns({"stype": "x"})
    r = s.estimation.corr(COLS, by="x")
    with pytest.raises(MethodError) as exc:
        r.to_polars()
    assert exc.value.code == "PAIR_COLUMN_CLASH"


def test_keys_are_unique(sample):
    assert sample.estimation.corr(COLS).keys() == PAIRS
    keys = sample.estimation.cov(COLS, by="stype").keys()
    assert keys == [(g, y, x) for g in "EHM" for y, x in PAIRS]
    assert len(set(keys)) == len(keys)


def test_labelled_keys_keep_the_pair(labelled):
    keys = labelled.estimation.corr(COLS, by="stype").keys(labels=True)
    assert keys[0] == ("Elementary", "api00", "api99")
    assert len(set(keys)) == len(keys)


def test_covariance_frame_names_each_pair(sample):
    df = sample.estimation.cov(("api00", "api99")).covariance_to_polars()
    assert df["key_a"].to_list() == [str(("api00", "api99"))]


def test_contrast_resolves_the_pair_either_way_round(sample):
    r = sample.estimation.corr(("api00", "api99"))
    ref = r.contrast(estd("api00", "api99"))
    rev = r.contrast(estd("api99", "api00"))
    assert rev.to_polars()["est"].to_list() == ref.to_polars()["est"].to_list()


def test_multi_row_contrast_explains_the_missing_covariance(sample):
    r = sample.estimation.corr(COLS)
    with pytest.raises(MethodError) as exc:
        r.contrast(estd("api00", "api99") - estd("api00", "enroll"))
    assert exc.value.code == "CONTRAST_NO_COVARIANCE"
    assert "prop(" not in str(exc.value)
    assert "Correlation and covariance" in str(exc.value)


def test_other_params_have_no_pair_columns(sample):
    df = sample.estimation.mean("api00", by="stype").to_polars()
    assert "y" not in df.columns and "x" not in df.columns
