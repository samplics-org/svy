"""Sample.describe(by=...): each group described; counts and sums per group; top_k=None."""

import json

import polars as pl
import pytest

import svy

from svy.errors import DimensionError, MethodError
from svy.serialize import serialize, to_json


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "reg": ["N", "N", "S", "S", "S", None],
            "urb": [1, 2, 1, 1, 2, 1],
            "hh": [10.0, 30.0, 5.0, 6.0, 2.0, 50.0],
            "sex": ["m", "f", "f", "m", "m", "f"],
            "w": [1.0, 2.0, 1.0, 1.0, 3.0, 1.0],
        }
    )


def _sample() -> svy.Sample:
    return svy.Sample(_frame(), svy.Design(wgt="w"))


def test_items_carry_their_group_in_sorted_order_null_last():
    r = _sample().describe(["hh"], by="reg")
    assert [(it.by, it.by_level) for it in r.items] == [
        (("reg",), ("N",)),
        (("reg",), ("S",)),
        (("reg",), (None,)),
    ]


def test_counts_and_sums_per_group():
    df = _sample().describe(["hh", "sex"], by="reg").to_polars()
    hh = df.filter(pl.col("name") == "hh")
    assert hh["reg"].to_list() == ["N", "S", None]
    assert hh["n"].to_list() == [2, 3, 1]
    assert hh["sum"].to_list() == [40.0, 13.0, 50.0]
    sex = [it for it in _sample().describe(["sex"], by="reg").items if it.by_level == ("S",)][0]
    assert {f.level: f.count for f in sex.levels} == {"m": 2.0, "f": 1.0}


def test_each_group_equals_describing_its_rows():
    by = _sample().describe(["hh"], by="reg")
    alone = svy.Sample(_frame().filter(pl.col("reg") == "S"), svy.Design(wgt="w")).describe(["hh"])
    s_item = [it for it in by.items if it.by_level == ("S",)][0]
    assert s_item.mean == alone.items[0].mean
    assert s_item.percentiles == alone.items[0].percentiles


def test_several_by_columns_cross():
    df = _sample().describe(["sex"], by=["reg", "urb"]).to_polars()
    assert df.columns[:3] == ["reg", "urb", "name"]
    assert df.select("reg", "urb").rows() == [("N", 1), ("N", 2), ("S", 1), ("S", 2), (None, 1)]
    assert df["n"].to_list() == [1, 1, 2, 1, 1]


def test_by_columns_are_not_described_by_default():
    names = {it.name for it in _sample().describe(by="reg").items}
    assert "reg" not in names
    assert {"hh", "sex"} <= names


def test_weighted_counts_per_group():
    r = _sample().describe(["sex"], by="reg", weighted=True)
    s_item = [it for it in r.items if it.by_level == ("S",)][0]
    assert {f.level: f.count for f in s_item.levels} == {"m": 4.0, "f": 1.0}


def test_top_k_none_lists_every_level():
    df = pl.DataFrame({"stratum": [f"s{i:02d}" for i in range(40)] * 2})
    s = svy.Sample(df)
    capped = s.describe(["stratum"]).items[0]
    full = s.describe(["stratum"], top_k=None).items[0]
    assert (len(capped.levels), capped.truncated) == (10, True)
    assert (len(full.levels), full.truncated, full.n_levels) == (40, False, 40)
    assert s.describe(["stratum"], top_k=None).top_k is None


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "10"])
def test_bad_top_k(bad):
    with pytest.raises(MethodError):
        _sample().describe(["sex"], top_k=bad)


def test_missing_by_column():
    with pytest.raises(DimensionError):
        _sample().describe(["hh"], by="region")


def test_without_by_items_have_no_group():
    r = _sample().describe(["hh"])
    assert (r.items[0].by, r.items[0].by_level) == (None, None)
    assert r.to_polars().columns[0] == "name"  # no group columns, order as before


def test_serialized_items_keep_the_group():
    r = _sample().describe(["hh"], by="reg", top_k=None)
    data = serialize(r)
    assert data.top_k is None
    assert list(data.items[0]["by"]) == ["reg"] and list(data.items[0]["by_level"]) == ["N"]
    assert json.loads(to_json(r))["items"][2]["by_level"] == [None]


def test_printed_names_show_the_group():
    text = str(_sample().describe(["hh"], by="reg"))
    assert "hh (reg=N)" in text and "hh (reg=S)" in text
