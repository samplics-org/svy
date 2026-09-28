# tests/svy/wrangling/test_clean_names_reserved.py
"""clean_names keeps the columns the selection methods write.

The reserved set named svy_weight/svy_prob/svy_hit, which svy never writes, so
a camel or upper clean renamed svy_sample_weight and the other outputs.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample
from svy.core.constants import (
    SELECTION_COLUMNS,
    SVY_CERTAINTY,
    SVY_HIT,
    SVY_PROB,
    SVY_PROB_STAGE1,
    SVY_ROW_INDEX,
    SVY_WEIGHT,
    SVY_WGT_STAGE1,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ea": [1, 2, 3, 4, 5, 6],
            "Region Name": ["N", "N", "N", "S", "S", "S"],
            "mos": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
            "Some Var": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _selected() -> Sample:
    return Sample(_frame(), Design(mos="mos", psu="ea")).sampling.pps_sys(n=4, rstate=2)


def _two_stage() -> Sample:
    s1 = _selected()
    eas = s1._data["ea"].to_list()
    hh = pl.DataFrame(
        {
            "ea": [e for e in eas for _ in range(4)],
            "Income Level": np.arange(4 * len(eas), dtype=float),
        }
    )
    return s1.sampling.add_stage(hh).sampling.srs(n=2, by="ea", rstate=3)


_STYLES = [
    {"case_style": "camel"},
    {"case_style": "pascal"},
    {"case_style": "kebab"},
    {"letter_case": "upper"},
    {"letter_case": "title"},
    {"minimal": True, "letter_case": "upper"},
    {"remove": "[ ]", "case_style": "camel"},
]


@pytest.mark.parametrize("kw", _STYLES)
def test_selection_columns_kept(kw):
    s = _selected()
    written = [c for c in s._data.columns if c in SELECTION_COLUMNS]
    assert set(written) == {SVY_PROB, SVY_WEIGHT, SVY_HIT, SVY_CERTAINTY}
    out = s.wrangling.clean_names(**kw)
    assert set(written) <= set(out._data.columns)
    assert SVY_ROW_INDEX in out._data.columns
    assert out.design.wgt == SVY_WEIGHT
    assert out.design.prob == SVY_PROB


@pytest.mark.parametrize("kw", _STYLES)
def test_two_stage_columns_kept(kw):
    s = _two_stage()
    written = {c for c in s._data.columns if c in SELECTION_COLUMNS}
    assert {SVY_PROB_STAGE1, SVY_WGT_STAGE1} <= written
    out = s.wrangling.clean_names(**kw)
    assert written <= set(out._data.columns)
    assert out.design.wgt == s.design.wgt
    assert out.design.prob == s.design.prob


def test_user_columns_still_cleaned():
    out = _selected().wrangling.clean_names(case_style="camel")
    assert "regionName" in out._data.columns
    assert "someVar" in out._data.columns


def test_estimation_after_clean():
    s = _selected()
    before = s.estimation.mean("Some Var").to_polars()
    out = s.wrangling.clean_names(letter_case="upper")
    after = out.estimation.mean("SOME_VAR").to_polars()
    assert after["est"][0] == pytest.approx(before["est"][0], rel=1e-12)
    assert after["se"][0] == pytest.approx(before["se"][0], rel=1e-12)


@pytest.mark.parametrize("first", [True, False])
def test_user_column_cleaned_into_reserved_name_is_suffixed(first):
    s = _selected()
    clash = pl.Series("SVY Sample Weight", [9.0] * s._data.height)
    data = s.data
    data = data.insert_column(0, clash) if first else data.with_columns(clash)
    s = Sample(data, s.design)
    out = s.wrangling.clean_names()
    assert out.design.wgt == SVY_WEIGHT
    np.testing.assert_array_equal(out._data[SVY_WEIGHT].to_numpy(), s._data[SVY_WEIGHT].to_numpy())
    assert out._data[f"{SVY_WEIGHT}_1"].to_list() == [9.0] * s._data.height


def test_absent_reserved_name_is_not_claimed():
    df = pl.DataFrame({"SVY Certainty": [1, 0, 1]})
    out = Sample(df).wrangling.clean_names()
    assert SVY_CERTAINTY in out._data.columns
    assert f"{SVY_CERTAINTY}_1" not in out._data.columns


def test_custom_output_names_are_user_columns():
    s = Sample(_frame(), Design(mos="mos", psu="ea")).sampling.pps_sys(
        n=4, rstate=2, wgt_name="Design Wgt"
    )
    out = s.wrangling.clean_names()
    assert "design_wgt" in out._data.columns
    assert out.design.wgt == "design_wgt"


def test_stale_names_no_longer_reserved():
    df = pl.DataFrame({"svy_weight": [1.0], "svy_prob": [1.0], "svy_hit": [1]})
    out = Sample(df).wrangling.clean_names(letter_case="upper")
    assert {"SVY_WEIGHT", "SVY_PROB", "SVY_HIT"} <= set(out._data.columns)


def test_reserved_set_covers_what_selection_writes():
    s = _two_stage()
    svy_written = {c for c in s._data.columns if c.startswith("svy_") and c != SVY_ROW_INDEX}
    assert svy_written <= SELECTION_COLUMNS
