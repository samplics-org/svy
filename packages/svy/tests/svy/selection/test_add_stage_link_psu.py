# tests/svy/selection/test_add_stage_link_psu.py
"""The next stage's psu links rows to stage 1; it is not the combined ssu.

Using it as the ssu made psu == ssu whenever the column kept the stage-1 name,
and estimation then crashed on a duplicate column.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample


def _ea_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ea": [1, 2, 3, 4, 5, 6],
            "region": ["N", "N", "N", "S", "S", "S"],
            "mos": [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
        }
    )


def _hh_frame() -> pl.DataFrame:
    """Listing of the EAs stage 1 selects."""
    n_per_ea = 5
    ea = [e for e in range(1, 7) for _ in range(n_per_ea)]
    frame = pl.DataFrame(
        {
            "hid": list(range(len(ea))),
            "ea": ea,
            "ea_code": ea,
            "region": ["N" if e <= 3 else "S" for e in ea],
            "y": np.random.default_rng(3).normal(20, 4, len(ea)),
            "flag": [i % 2 for i in range(len(ea))],
        }
    )
    selected = _stage1()._data["ea"].to_list()
    return frame.filter(pl.col("ea").is_in(selected))


def _stage1(psu="ea", **kw) -> Sample:
    df = _ea_frame()
    return Sample(df, Design(mos="mos", stratum="region", psu=psu, **kw)).sampling.pps_sys(
        n=2, by="region", rstate=5, certainty_threshold=1.0
    )


def _stage2(psu) -> Sample:
    return Sample(_hh_frame(), Design(stratum="region", psu=psu)).sampling.srs(
        n=3, by="ea", rstate=9
    )


def _est(s: Sample, fn="mean", y="y", **kw):
    return getattr(s.estimation, fn)(y, **kw).to_polars()


class TestSameNameLink:
    def test_ssu_is_none(self):
        out = _stage1().sampling.add_stage(_stage2("ea"))
        assert out.design.psu == "ea"
        assert out.design.ssu is None

    @pytest.mark.parametrize("fn,y", [("mean", "y"), ("total", "y"), ("prop", "flag")])
    def test_estimation_runs_and_matches_unlinked(self, fn, y):
        linked = _stage1().sampling.add_stage(_stage2("ea"))
        unlinked = _stage1().sampling.add_stage(_stage2(None))
        a, b = _est(linked, fn, y), _est(unlinked, fn, y)
        np.testing.assert_allclose(a["est"].to_numpy(), b["est"].to_numpy(), rtol=1e-12)
        np.testing.assert_allclose(a["se"].to_numpy(), b["se"].to_numpy(), rtol=1e-12)
        assert np.isfinite(a["se"].to_numpy()).all()

    def test_domain_estimation_runs(self):
        out = _stage1().sampling.add_stage(_stage2("ea"))
        res = _est(out, by="region")
        assert res.height == 2
        assert np.isfinite(res["se"].to_numpy()).all()

    def test_matches_dataframe_then_select(self):
        """Linking a selected Sample equals chaining the selection after add_stage."""
        linked = _stage1().sampling.add_stage(_stage2("ea"))
        chained = _stage1().sampling.add_stage(_hh_frame()).sampling.srs(n=3, by="ea", rstate=9)
        assert chained.design.ssu is None
        a, b = _est(linked), _est(chained)
        assert a["est"][0] == pytest.approx(b["est"][0], rel=1e-12)
        assert a["se"][0] == pytest.approx(b["se"][0], rel=1e-12)

    def test_unselected_next_stage(self):
        out = _stage1().sampling.add_stage(Sample(_hh_frame(), Design(psu="ea")))
        assert out.design.ssu is None
        selected = out.sampling.srs(n=2, by="ea", rstate=1)
        assert selected.design.ssu is None
        assert np.isfinite(_est(selected)["se"][0])

    def test_replicate_weights_build(self):
        out = _stage1().sampling.add_stage(_stage2("ea"))
        rep = out.weighting.create_jk_wgts()
        assert rep.design.rep_wgts is not None
        assert np.isfinite(_est(rep)["se"][0])


def test_renamed_link_column_is_not_the_ssu():
    out = _stage1().sampling.add_stage(_stage2("ea_code"))
    assert out.design.psu == "ea"
    assert out.design.ssu is None


def test_tuple_psu_link():
    s1 = _stage1(psu=("region", "ea"))
    out = s1.sampling.add_stage(_stage2(("region", "ea")))
    assert out.design.psu == ("region", "ea")
    assert out.design.ssu is None
    assert np.isfinite(_est(out)["se"][0])


def test_stage1_ssu_does_not_carry_over():
    s1 = Sample(
        _ea_frame().with_columns(pl.col("ea").alias("seg")),
        Design(mos="mos", stratum="region", psu="ea", ssu="seg"),
    ).sampling.pps_sys(n=2, by="region", rstate=5, certainty_threshold=1.0)
    out = s1.sampling.add_stage(_stage2("ea"))
    assert out.design.ssu is None


@pytest.mark.parametrize("by_key", ["psu", "stratum"])
def test_second_stage_by_psu_has_no_empty_cell_findings(by_key):
    psus = pl.DataFrame(
        {"stratum": [f"S{i // 4}" for i in range(12)], "psu": list(range(12)), "mos": [5.0] * 12}
    )
    units = psus.join(pl.DataFrame({"unit": list(range(10))}), how="cross")
    first = Sample(psus, Design(stratum="stratum", psu="psu", mos="mos")).sampling.pps_sys(
        2, rstate=1, certainty_threshold=1.0
    )
    sel = first.data["psu"].to_list()
    n = {p: 3 for p in sel} if by_key == "psu" else {"S0": 3, "S1": 3, "S2": 3}
    second = first.sampling.add_stage(units.filter(pl.col("psu").is_in(sel))).sampling.srs(
        n, by="psu", rstate=2
    )
    codes = {f.code for f in second.warnings}
    assert not codes & {"SELECTION_STRATUM_EMPTY", "SELECTION_GROUP_EMPTY"}
    assert second.data.height == 3 * len(sel)
