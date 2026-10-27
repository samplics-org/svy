# tests/svy/estimation/test_labelled_order.py
"""Printed estimates list a labelled variable's levels in code order, as to_polars() does."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample


N = 400
AREA = {1.0: "URBANA", 2.0: "RURAL"}
ANSWER = {0: "Não", 1: "Sim", 97: "Não sabe", 98: "Não respondeu"}


@pytest.fixture(scope="module")
def sample() -> Sample:
    rng = np.random.default_rng(20261027)
    df = pl.DataFrame(
        {
            "psu": np.repeat(np.arange(40), N // 40),
            "w": rng.uniform(1, 3, N),
            "y": rng.normal(10, 2, N),
            "area": rng.choice(list(AREA), N),
            "a4": rng.choice(list(ANSWER), N),
            "zone": rng.choice(["b", "a"], N),
        }
    )
    s = Sample(df, Design(psu="psu", wgt="w")).weighting.create_jk_wgts()
    s.meta.set_value_labels("area", AREA)
    s.meta.set_value_labels("a4", ANSWER)
    return s


def _shown(result, col: str) -> list:
    return list(dict.fromkeys(result.to_polars_printable()[col].to_list()))


@pytest.mark.parametrize("method", ["taylor", "replication"])
def test_prop_prints_levels_in_code_order(sample, method):
    r = sample.estimation.prop("a4", by="area", method=method)
    assert _shown(r, "area") == ["URBANA", "RURAL"]
    df = r.to_polars_printable()
    assert df.filter(pl.col("area") == "RURAL")["a4"].to_list() == [
        "Não",
        "Sim",
        "Não sabe",
        "Não respondeu",
    ]


def test_print_and_data_view_list_rows_alike(sample):
    r = sample.estimation.prop("a4", by="area")
    data = r.to_polars().select("area", "a4").rows()
    shown = r.to_polars_printable().select("area", "a4").rows()
    assert shown == [(AREA[a], ANSWER[v]) for a, v in data]


def test_str_lists_levels_in_code_order(sample):
    out = str(sample.estimation.mean("y", by="area"))
    assert out.index("URBANA") < out.index("RURAL")


def test_level_without_a_label_keeps_its_code_place(sample):
    s = sample.clone()
    s.meta.set_value_labels("a4", {0: "Não", 1: "Sim"})
    r = s.estimation.prop("a4")
    assert _shown(r, "a4") == ["Não", "Sim", "97", "98"]


def test_unlabelled_domain_sorts_on_its_text(sample):
    r = sample.estimation.prop("area", by="zone")
    assert _shown(r, "zone") == ["a", "b"]
    assert _shown(r, "area") == ["URBANA", "RURAL"]


def test_codes_shown_without_labels_keep_code_order(sample):
    r = sample.estimation.prop("a4", by="area")
    r.use_labels = False
    assert _shown(r, "area") == ["1.0", "2.0"]
    assert _shown(r, "a4") == ["0", "1", "97", "98"]
