# tests/svy/weighting/test_replication_tuple_psu.py
"""BRR and paired-JK weights built from a multi-column PSU.

The pairing step used to put the PSU tuple into a polars select and crash. A
tuple PSU must give the same replicates as a single column holding the same
composite key.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample
from svy.errors import DimensionError


def _frame(psus_per_stratum: list[int], rows_per_psu: int = 2) -> pl.DataFrame:
    """PSU ids restart in every stratum, so only (reg, ea) identifies a PSU."""
    reg, ea = [], []
    for h, n_psu in enumerate(psus_per_stratum):
        for i in range(1, n_psu + 1):
            reg += [f"r{h}"] * rows_per_psu
            ea += [i] * rows_per_psu
    n = len(reg)
    return pl.DataFrame(
        {
            "reg": reg,
            "ea": ea,
            "y": np.random.default_rng(7).normal(50, 10, n),
            "w": np.random.default_rng(8).uniform(1, 3, n),
        }
    ).with_columns(
        pl.concat_str([pl.col("reg"), pl.col("ea").cast(pl.Utf8)], separator="|").alias("key"),
        pl.col("ea").rank("dense").alias("rank_in_frame"),
    )


def _pair(df, *, method, psu, stratum="reg", **kw):
    s = Sample(df, Design(stratum=stratum, psu=psu, wgt="w"))
    if method == "brr":
        return s.weighting.create_brr_wgts(**kw)
    return s.weighting.create_jk_wgts(paired=True, **kw)


def _reps(s: Sample) -> np.ndarray:
    return s._data.select(s.design.rep_wgts.columns).to_numpy()


def _composite_per_var_stratum(s: Sample) -> list[int]:
    return (
        s._data.select("svy_var_stratum", "reg", "ea")
        .unique()
        .group_by("svy_var_stratum")
        .len()
        .get_column("len")
        .to_list()
    )


@pytest.mark.parametrize("method", ["brr", "jk2"])
class TestTuplePsuMatchesCompositeColumn:
    def test_four_psus_per_stratum(self, method):
        df = _frame([4, 4])
        tup = _pair(df, method=method, psu=("reg", "ea"))
        one = _pair(df, method=method, psu="key")
        np.testing.assert_array_equal(_reps(tup), _reps(one))
        assert tup._data["svy_var_stratum"].to_list() == one._data["svy_var_stratum"].to_list()

    def test_psu_recorded_as_tuple(self, method):
        out = _pair(_frame([4, 4]), method=method, psu=("reg", "ea"))
        assert out.design.rep_wgts.psu == ("reg", "ea")
        assert out.design.rep_wgts.stratum == "svy_var_stratum"

    def test_every_variance_stratum_holds_two_psus(self, method):
        out = _pair(_frame([4, 6, 2]), method=method, psu=("reg", "ea"))
        assert set(_composite_per_var_stratum(out)) == {2}

    def test_repeated_ids_across_strata_stay_distinct(self, method):
        # 'ea' alone repeats across regions; pairing on it would merge PSUs
        out = _pair(_frame([4, 4]), method=method, psu=("reg", "ea"))
        assert out._data.select("reg", "ea").unique().height == 8
        assert out._data["svy_var_stratum"].n_unique() == 4

    def test_without_stratum(self, method):
        df = _frame([6])
        tup = _pair(df, method=method, psu=("reg", "ea"), stratum=None)
        one = _pair(df, method=method, psu="key", stratum=None)
        np.testing.assert_array_equal(_reps(tup), _reps(one))
        assert tup._data["svy_var_stratum"].n_unique() == 3

    def test_order_by(self, method):
        df = _frame([4, 4])
        tup = _pair(df, method=method, psu=("reg", "ea"), order_by="rank_in_frame")
        one = _pair(df, method=method, psu="key", order_by="rank_in_frame")
        np.testing.assert_array_equal(_reps(tup), _reps(one))

    def test_order_by_a_psu_column(self, method):
        out = _pair(_frame([4, 4]), method=method, psu=("reg", "ea"), order_by="ea")
        assert set(_composite_per_var_stratum(out)) == {2}

    def test_shuffle_is_reproducible(self, method):
        df = _frame([6, 4])
        a = _pair(df, method=method, psu=("reg", "ea"), shuffle=True, rstate=11)
        b = _pair(df, method=method, psu=("reg", "ea"), shuffle=True, rstate=11)
        np.testing.assert_array_equal(_reps(a), _reps(b))
        assert set(_composite_per_var_stratum(a)) == {2}

    def test_unequal_rows_per_psu(self, method):
        df = _frame([4, 4]).vstack(_frame([4, 4], rows_per_psu=1))
        tup = _pair(df, method=method, psu=("reg", "ea"))
        one = _pair(df, method=method, psu="key")
        np.testing.assert_array_equal(_reps(tup), _reps(one))

    def test_row_order_preserved(self, method):
        df = _frame([4, 4]).sample(fraction=1.0, shuffle=True, seed=3)
        out = _pair(df, method=method, psu=("reg", "ea"))
        assert out._data["key"].to_list() == df["key"].to_list()

    def test_estimate_matches(self, method):
        df = _frame([4, 4])
        tup = _pair(df, method=method, psu=("reg", "ea")).estimation.mean("y")
        one = _pair(df, method=method, psu="key").estimation.mean("y")
        a, b = tup.to_polars(), one.to_polars()
        assert a["est"][0] == pytest.approx(b["est"][0], rel=1e-12)
        assert a["se"][0] == pytest.approx(b["se"][0], rel=1e-12)


def test_brr_tuple_psu_odd_count_raises_guiding_error():
    with pytest.raises(DimensionError) as exc:
        _pair(_frame([3, 4]), method="brr", psu=("reg", "ea"))
    assert exc.value.code == "ODD_PSU_COUNT"


def test_jk2_tuple_psu_odd_count_forms_triplet():
    out = _pair(_frame([5, 4]), method="jk2", psu=("reg", "ea"))
    assert sorted(_composite_per_var_stratum(out)) == [2, 2, 2, 3]


def test_jk2_tuple_psu_single_psu_stratum_raises_guiding_error():
    with pytest.raises(DimensionError) as exc:
        _pair(_frame([1, 4]), method="jk2", psu=("reg", "ea"))
    assert exc.value.code == "INSUFFICIENT_PSU"


@pytest.mark.parametrize("method", ["brr", "jk2"])
def test_tuple_stratum_and_tuple_psu(method):
    df = _frame([4, 4]).with_columns(pl.lit("z").alias("zone"))
    tup = _pair(df, method=method, psu=("reg", "ea"), stratum=("zone", "reg"))
    one = _pair(df, method=method, psu="key", stratum=("zone", "reg"))
    np.testing.assert_array_equal(_reps(tup), _reps(one))


@pytest.mark.parametrize("method", ["brr", "jk2"])
def test_two_psus_per_stratum_needs_no_pairing(method):
    out = _pair(_frame([2, 2]), method=method, psu=("reg", "ea"))
    assert "svy_var_stratum" not in out._data.columns
    assert out.design.rep_wgts.stratum == "reg"
