# tests/svy/categorical/test_table_level_order.py
"""Table levels follow the column's own order.

The kernel sees levels as text, and cells came back in its order. Levels are
now ordered on the column's values, an Enum's in the Enum's order and numbers
numerically, and cells are listed by row level, then column level.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest

from polars.testing import assert_frame_equal

from svy import Design, Sample
from svy.serialize import from_json, to_json, to_polars


N = 480

Q_ORDER = ["Jul-Sep 2023", "Oct-Dec 2023", "Jan-Mar 2024", "Apr-Jun 2024"]
S_ORDER = ["low", "mid", "high"]


@pytest.fixture(scope="module")
def sample() -> Sample:
    rng = np.random.default_rng(20260928)
    df = pl.DataFrame(
        {
            "stratum": np.repeat(["s1", "s2"], N // 2),
            "psu": np.repeat(np.arange(48), N // 48),
            "w": rng.uniform(1, 3, N),
            "quarter": rng.choice(Q_ORDER, N),
            "status": rng.choice(S_ORDER, N),
            "k": rng.choice([1, 2, 10], N),
            "g": rng.choice(["b", "a"], N),
        }
    ).with_columns(
        pl.col("quarter").cast(pl.Enum(Q_ORDER)),
        pl.col("status").cast(pl.Enum(S_ORDER)),
        quarter_cat=pl.col("quarter").cast(pl.Categorical),
    )
    return Sample(df, Design(stratum="stratum", psu="psu", wgt="w"))


def _cells(t) -> list:
    return [(c.rowvar, c.colvar) if t.colvar else c.rowvar for c in t.estimates]


def test_one_way_enum(sample):
    t = sample.categorical.tabulate("quarter")
    assert t.rowvals == Q_ORDER
    assert _cells(t) == Q_ORDER
    assert t.to_polars()["quarter"].to_list() == Q_ORDER


def test_two_way_enums(sample):
    t = sample.categorical.tabulate("quarter", "status")
    assert (t.rowvals, t.colvals) == (Q_ORDER, S_ORDER)
    assert _cells(t) == list(itertools.product(Q_ORDER, S_ORDER))
    assert t.to_polars().select("quarter", "status").rows() == _cells(t)


def test_enum_beside_plain(sample):
    t = sample.categorical.tabulate("g", "quarter")
    assert _cells(t) == list(itertools.product(["a", "b"], Q_ORDER))


def test_numbers_sort_numerically(sample):
    t = sample.categorical.tabulate("k", "status")
    assert t.rowvals == ["1", "2", "10"]
    assert _cells(t) == list(itertools.product(["1", "2", "10"], S_ORDER))


def test_categorical_sorts_as_text(sample):
    t = sample.categorical.tabulate("quarter_cat")
    assert _cells(t) == sorted(Q_ORDER)


def test_where_keeps_order(sample):
    t = sample.categorical.tabulate("quarter", "status", where=pl.col("k") > 1)
    assert _cells(t) == list(itertools.product(Q_ORDER, S_ORDER))


def test_printed_rows_follow_enum_order(sample):
    from svy.categorical.table import _rows_for_display

    t = sample.categorical.tabulate("quarter", "status")
    assert [tuple(r[:2]) for r in _rows_for_display(t)] == _cells(t)


def test_crosstab_follows_enum_order(sample):
    t = sample.categorical.tabulate("quarter", "status")
    ct = t.crosstab()
    assert ct["quarter"].to_list() == Q_ORDER
    assert ct.columns[1:] == S_ORDER


def test_saved_payload_keeps_order(sample):
    t = sample.categorical.tabulate("quarter", "status")
    assert_frame_equal(to_polars(from_json(to_json(t))), t.to_polars())
