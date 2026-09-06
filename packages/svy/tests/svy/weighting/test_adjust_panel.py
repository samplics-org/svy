"""adjust() on a panel: the factor applies to the case, and a case with earlier
rows but none in scope is a nonrespondent. Numbers are checked against the
same chain done by hand on the wide frame."""

import numpy as np
import polars as pl
import pytest

import svy

from svy import col


def _long():
    w1 = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "g": ["A", "A", "A", "B", "B", "B"],
            "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "resp": ["rr"] * 6,
            "wave": [1] * 6,
        }
    )
    # cases 5 and 6 attrite; case 3 is a wave-2 nonrespondent
    w2 = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "g": ["A", "A", "A", "B"],
            "w": [1.0, 2.0, 3.0, 4.0],
            "resp": ["rr", "rr", "nr", "rr"],
            "wave": [2] * 4,
        }
    )
    return svy.Sample(pl.concat([w1, w2]), svy.Design(case_id="id", wave="wave", wgt="w"))


# By hand on the wide frame: cell A at wave 2 = {1 rr, 2 rr, 3 nr}, factor 6/3;
# cell B = {4 rr, 5 attrited, 6 attrited}, factor (4+5+6)/4.
EXPECTED = {1: 2.0, 2: 4.0, 3: 0.0, 4: 15.0, 5: 0.0, 6: 0.0}


def test_factor_written_to_every_row_of_the_case():
    out = _long().weighting.adjust(
        "resp", cells="g", where=col("wave") == 2, respondents_only=False
    )
    got = out.data.select("id", "wave", "nr_wgt")
    for cid, wave, w in got.iter_rows():
        assert w == pytest.approx(EXPECTED[cid]), (cid, wave)
    # the longitudinal weight is constant within case
    assert (got.group_by("id").agg(pl.col("nr_wgt").n_unique())["nr_wgt"] == 1).all()
    # and preserves the base-wave total among survivors
    assert got.filter(pl.col("wave") == 2)["nr_wgt"].sum() == pytest.approx(21.0)
    assert out.design.wgt == "nr_wgt"
    assert out.design.wgt_adjustment.kind == "nonresponse"


def test_attriters_are_nonrespondents_and_keep_their_earlier_rows():
    out = _long().weighting.adjust("resp", cells="g", where=col("wave") == 2)
    # respondents_only drops the in-scope nonrespondent row only (case 3, wave 2)
    assert out.n_records == 9
    assert out.data.filter((pl.col("id") == 3) & (pl.col("wave") == 2)).height == 0
    assert out.data.filter(pl.col("id").is_in([5, 6]))["wave"].to_list() == [1, 1]
    assert out.data.filter(pl.col("id").is_in([5, 6]))["nr_wgt"].to_list() == [0.0, 0.0]


def test_matches_wide_computation_with_single_class():
    out = _long().weighting.adjust("resp", where=col("wave") == 2, respondents_only=False)
    # one class: respondents 1, 2, 4 share the weight of 3, 5, 6
    factor = 21.0 / 7.0
    got = dict(out.data.filter(pl.col("wave") == 1).select("id", "nr_wgt").iter_rows())
    assert got == pytest.approx({1: factor, 2: 2 * factor, 3: 0.0, 4: 4 * factor, 5: 0.0, 6: 0.0})


def test_replicate_columns_propagate_per_case():
    s = _long()
    # each case is its own PSU on this panel: JK1 replicates over cases
    s = s.weighting.create_jk_wgts()
    out = s.weighting.adjust("resp", cells="g", where=col("wave") == 2, respondents_only=False)
    rep_cols = out.design.rep_wgts.columns
    assert len(rep_cols) == 6
    rep = out.data.select("id", *rep_cols)
    within = rep.group_by("id").agg([pl.col(c).n_unique() for c in rep_cols])
    assert (within.drop("id").to_numpy() == 1).all()
    assert out.data.filter(pl.col("id") == 3).select(rep_cols).to_numpy().sum() == 0.0


def test_chained_waves():
    w3 = pl.DataFrame(
        {
            "id": [1, 4],
            "g": ["A", "B"],
            "w": [1.0, 4.0],
            "resp": ["rr", "rr"],
            "wave": [3, 3],
        }
    )
    s = svy.Sample(pl.concat([_long().data.drop("svy_row_index"), w3]), _long().design)
    s = s.weighting.adjust("resp", cells="g", where=col("wave") == 2, wgt_name="lw_12")
    s = s.weighting.adjust("resp", cells="g", where=col("wave") == 3, wgt_name="lw_123")
    lw = dict(s.data.filter(pl.col("wave") == 1).select("id", "lw_123").iter_rows())
    # wave 3: A = {1 rr, 2 attrited} on lw_12 (2, 4) -> factor 3; B = {4 rr} -> factor 1
    assert lw == pytest.approx({1: 6.0, 2: 0.0, 3: 0.0, 4: 15.0, 5: 0.0, 6: 0.0})


def test_scope_not_a_wave_set_warns_and_skips_missing_rule():
    s = _long()
    with pytest.warns(UserWarning, match="missing-in-scope rule was skipped"):
        out = s.weighting.adjust(
            "resp", cells="g", where=(col("wave") == 2) & (col("id") != 4), respondents_only=False
        )
    # cell B in scope is empty of rows, so cell B never forms; case 4 keeps its weight
    assert out.data.filter(pl.col("id") == 4)["nr_wgt"].to_list() == [4.0, 4.0]


def test_cross_section_behaviour_unchanged():
    df = _long().data.filter(pl.col("wave") == 2).drop("svy_row_index")
    s = svy.Sample(df, svy.Design(wgt="w"))
    out = s.weighting.adjust("resp", cells="g", respondents_only=False)
    assert out.data["nr_wgt"].to_list() == pytest.approx([2.0, 4.0, 0.0, 4.0])


def test_case_weights_zero_for_in_scope_nonrespondent_everywhere():
    out = _long().weighting.adjust(
        "resp", cells="g", where=col("wave") == 2, respondents_only=False
    )
    assert out.data.filter(pl.col("id") == 3)["nr_wgt"].to_list() == [0.0, 0.0]
    assert np.isfinite(out.data["nr_wgt"].to_numpy()).all()
