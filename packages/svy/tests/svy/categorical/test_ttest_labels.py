import numpy as np
import polars as pl
import pytest

import svy


@pytest.fixture
def sample():
    rng = np.random.default_rng(1)
    n = 400
    df = pl.DataFrame(
        {
            "str": rng.integers(1, 5, n),
            "psu": np.arange(n) // 10,
            "w": rng.uniform(1, 3, n),
            "area": rng.integers(1, 3, n).astype(float),
            "reg": rng.integers(1, 3, n),
            "y": rng.uniform(0, 1, n),
        }
    )
    s = svy.Sample(df, svy.Design(stratum="str", psu="psu", wgt="w"))
    s.set_value_labels("area", {1: "URBANA", 2: "RURAL"})
    return s


def test_labelled_group_prints_its_labels(sample):
    out = sample.categorical.ttest("y", group="area").__plain_str__()
    assert "Groups: area = [URBANA vs RURAL]" in out
    assert "\n  URBANA " in out and "\n  RURAL " in out


def test_rich_print_uses_labels(sample):
    out = str(sample.categorical.ttest("y", group="area"))
    assert "URBANA vs RURAL" in out


def test_result_keeps_the_codes(sample):
    t = sample.categorical.ttest("y", group="area")
    assert t.groups.levels == (1.0, 2.0)
    assert {e.group_level for e in t.estimates} == {1.0, 2.0}


def test_use_labels_false_prints_codes(sample):
    t = sample.categorical.ttest("y", group="area", use_labels=False)
    assert t.groups.labels is None
    assert "Groups: area = [1.0 vs 2.0]" in t.__plain_str__()


def test_unlabelled_group_unchanged(sample):
    t = sample.categorical.ttest("y", group="reg")
    assert t.groups.labels is None
    assert "Groups: reg = [1 vs 2]" in t.__plain_str__()


def test_to_polars_carries_a_label_column(sample):
    t = sample.categorical.ttest("y", group="area")
    tidy = t.to_polars("estimates")
    assert tidy["area"].to_list() == [1.0, 2.0]
    assert tidy["area_label"].to_list() == ["URBANA", "RURAL"]
    raw = t.to_polars("estimates", tidy=False)
    assert raw["group_level_label"].to_list() == ["URBANA", "RURAL"]


def test_no_label_column_without_labels(sample):
    t = sample.categorical.ttest("y", group="area", use_labels=False)
    assert "area_label" not in t.to_polars("estimates").columns


def test_unlabelled_code_falls_back_to_the_code(sample):
    sample.set_value_labels("area", {1: "URBANA"})
    t = sample.categorical.ttest("y", group="area")
    assert t.groups.labels == ("URBANA", "2.0")


def test_by_result_header_and_frames_use_labels(sample):
    r = sample.categorical.ttest("y", group="area", by="reg")
    out = r.__plain_str__()
    assert "Groups: area = [URBANA vs RURAL]" in out
    assert out.count("\n  RURAL ") == 2
    assert set(r.to_polars("estimates")["area_label"].to_list()) == {"URBANA", "RURAL"}


def test_saved_result_still_round_trips(sample):
    t = sample.categorical.ttest("y", group="area")
    back = svy.serialize.from_json(svy.serialize.to_json(t))
    assert back.groups.levels == [1.0, 2.0]
