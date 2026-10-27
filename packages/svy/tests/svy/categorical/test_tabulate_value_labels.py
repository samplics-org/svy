# tests/svy/categorical/test_tabulate_value_labels.py
"""tabulate keeps levels as the column holds them and shows their value labels, by either method."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import svy

from svy.serialize import from_json, to_json


N = 40

# column -> (values over 4 rows, labels keyed as set_value_labels takes them)
LEVELS = {
    "flag": ([True, False, True, True], {False: "No", True: "Yes"}),
    "code": ([1, 2, 1, 1], {1: "Urban", 2: "Rural"}),
    "fcode": ([1.0, 2.0, 1.0, 1.0], {1: "One", 2: "Two"}),
    "word": (["a", "b", "a", "a"], {"a": "Alpha", "b": "Beta"}),
}


@pytest.fixture(scope="module")
def sample() -> svy.Sample:
    cols = {name: vals * (N // 4) for name, (vals, _) in LEVELS.items()}
    df = pl.DataFrame(
        {
            **cols,
            "g": np.repeat(["x", "y"], N // 2),
            "s": np.repeat(["s1", "s2"], N // 2),
            "p": np.arange(N) // 2,
            "w": np.ones(N),
            "r1": np.ones(N),
            "r2": np.ones(N),
        }
    )
    s = svy.Sample(
        df,
        svy.Design(stratum="s", psu="p", wgt="w", rep_wgts=svy.BrrWgts(prefix="r", n_reps=2)),
    )
    for name, (_, labels) in LEVELS.items():
        s.meta.set_value_labels(name, labels)
    return s


METHODS = ["taylor", "replication"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("var", list(LEVELS))
def test_one_way_shows_labels(sample, method, var):
    t = sample.categorical.tabulate(var, method=method)
    values, labels = LEVELS[var]
    text = t.__plain_str__()
    for label in labels.values():
        assert label in text
    # 3 of every 4 rows hold the level of the first value.
    shares = {t._get_row_display(c.rowvar): c.est for c in t.estimates}
    major = labels[values[0]]
    (minor,) = set(labels.values()) - {major}
    assert shares == pytest.approx({major: 0.75, minor: 0.25})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("var", list(LEVELS))
def test_two_way_shows_labels_on_the_column(sample, method, var):
    t = sample.categorical.tabulate(rowvar="g", colvar=var, method=method)
    assert t.crosstab().columns == ["g", *LEVELS[var][1].values()]
    for label in LEVELS[var][1].values():
        assert label in t.__plain_str__()


@pytest.mark.parametrize("method", METHODS)
def test_two_way_shows_labels_on_the_row(sample, method):
    t = sample.categorical.tabulate(rowvar="flag", colvar="g", method=method)
    assert t.crosstab()["flag"].to_list() == ["No", "Yes"]


@pytest.mark.parametrize(
    "var, levels",
    [("flag", [False, True]), ("code", [1, 2]), ("fcode", [1.0, 2.0]), ("word", ["a", "b"])],
)
def test_levels_keep_the_column_s_values(sample, var, levels):
    t = sample.categorical.tabulate(rowvar="g", colvar=var)
    assert t.colvals == levels
    assert [type(v) for v in t.colvals] == [type(v) for v in levels]
    assert t.rowvals == ["x", "y"]


@pytest.mark.parametrize("method", METHODS)
def test_without_labels_levels_print_as_their_cells_spell_them(sample, method):
    t = sample.categorical.tabulate("flag", method=method)
    t.use_labels = False
    assert {t._get_row_display(c.rowvar) for c in t.estimates} == {"false", "true"}
    assert t.crosstab(use_labels=False)["flag"].to_list() == ["false", "true"]


def test_a_level_without_a_label_prints_as_its_text():
    df = pl.DataFrame(
        {
            "flag": [True, False] * 10,
            "s": ["a"] * 10 + ["b"] * 10,
            "p": list(range(20)),
            "w": [1.0] * 20,
        }
    )
    s = svy.Sample(df, svy.Design(stratum="s", psu="p", wgt="w"))
    s.meta.set_value_labels("flag", {True: "Yes"})
    t = s.categorical.tabulate("flag")
    assert {t._get_row_display(c.rowvar) for c in t.estimates} == {"false", "Yes"}


@pytest.mark.parametrize("var", list(LEVELS))
def test_saved_levels_keep_their_type(sample, var):
    t = sample.categorical.tabulate(rowvar="g", colvar=var)
    payload = json.loads(to_json(t))
    assert payload["colvals"] == t.colvals
    assert from_json(to_json(t)).colvals == t.colvals
