from __future__ import annotations

import msgspec
import polars as pl
import pytest

from svy.checks import NestingCheck, check_nesting
from svy.errors import DimensionError, MethodError


def test_nested_psus():
    df = pl.DataFrame({"st": [1, 1, 1, 2, 2, 2], "psu": [1, 1, 2, 3, 4, 4]})
    assert check_nesting(df, "st", "psu") == NestingCheck(
        stratum=["st"],
        psu=["psu"],
        n=6,
        n_strata=2,
        n_psus=4,
        n_psus_across_strata=0,
        examples=[],
    )


def test_psu_codes_reused_across_strata():
    df = pl.DataFrame(
        {
            "st": ["B", "A", "A", "B", "C", "C", "A"],
            "psu": [2, 1, 2, 1, 1, 3, 4],
        }
    )
    r = check_nesting(df, "st", "psu")
    assert (r.n, r.n_strata, r.n_psus, r.n_psus_across_strata) == (7, 3, 4, 2)
    assert r.examples == [(1, ["A", "B", "C"]), (2, ["A", "B"])]


def test_several_columns():
    df = pl.DataFrame(
        {
            "region": [1, 1, 1, 2, 2],
            "urban": [0, 0, 1, 0, 0],
            "ea": [10, 10, 10, 10, 11],
            "seg": [1, 2, 1, 1, 1],
        }
    )
    r = check_nesting(df, ["region", "urban"], ["ea", "seg"])
    assert (r.stratum, r.psu) == (["region", "urban"], ["ea", "seg"])
    assert (r.n_strata, r.n_psus, r.n_psus_across_strata) == (3, 3, 1)
    assert r.examples == [((10, 1), [(1, 0), (1, 1), (2, 0)])]
    one = check_nesting(df, ["region", "urban"], "ea")
    assert one.examples == [(10, [(1, 0), (1, 1), (2, 0)])]
    assert check_nesting(df, "region", ["ea", "seg"]).examples == [((10, 1), [1, 2])]


def test_limit_caps_examples_not_counts():
    df = pl.DataFrame({"st": [1, 2] * 4, "psu": [1, 1, 2, 2, 3, 3, 4, 4]})
    r = check_nesting(df, "st", "psu", limit=2)
    assert r.n_psus_across_strata == 4
    assert r.examples == [(1, [1, 2]), (2, [1, 2])]
    assert check_nesting(df, "st", "psu", limit=0).examples == []


def test_bad_input():
    df = pl.DataFrame({"st": [1], "psu": [1]})
    with pytest.raises(DimensionError) as exc:
        check_nesting(df, "stratum", "psu")
    assert exc.value.code == "MISSING_COLUMNS" and exc.value.got == ["stratum"]
    with pytest.raises(MethodError):
        check_nesting(df, [], "psu")
    with pytest.raises(MethodError):
        check_nesting(df, "st", "psu", limit=-1)


def test_report_serializes_and_prints():
    df = pl.DataFrame({"st": [1, 2], "psu": [7, 7]})
    r = check_nesting(pl.LazyFrame(df), "st", "psu")
    back = msgspec.json.decode(msgspec.json.encode(r), type=NestingCheck)
    assert back.n_psus_across_strata == 1 and back.examples == [(7, [1, 2])]
    text = repr(r)
    assert text.splitlines()[0] == "Nesting check: psu in st"
    assert "7 in 1, 2" in text
