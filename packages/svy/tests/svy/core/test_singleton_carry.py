# tests/svy/core/test_singleton_carry.py
"""The declared singleton rule carried by combine_samples and add_stage."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.design import Singleton
from svy.errors import MethodError


def _result(sample):
    """What the design's singleton rule did to the current data (internal)."""
    sample._sync_parts()
    return sample._singleton_result


def _cycle(strat, psu=(1, 2, 1, 2, 3, 3), scale=1.0):
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "strat": strat,
            "psu": list(psu),
            "w": [x * scale for x in [10.0, 12.0, 8.0, 9.0, 11.0, 7.0]],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _sample(df, singleton=None):
    return svy.Sample(df, svy.Design(stratum="strat", psu="psu", wgt="w", singleton=singleton))


class TestCombineSamples:
    def test_the_same_rule_is_carried(self):
        a = _sample(_cycle([1, 1, 2, 2, 1, 2]), "center")
        b = _sample(_cycle([3, 3, 4, 4, 3, 4], scale=2.0), "center")
        combined = svy.combine_samples([a, b])
        assert combined.design.singleton == Singleton("center")

    def test_no_rule_stays_no_rule(self):
        a = _sample(_cycle([1, 1, 2, 2, 1, 2]))
        b = _sample(_cycle([3, 3, 4, 4, 3, 4]))
        assert svy.combine_samples([a, b]).design.singleton is None

    @pytest.mark.parametrize("second", [None, "skip", Singleton("center", domains="apply")])
    def test_different_rules_are_refused(self, second):
        a = _sample(_cycle([1, 1, 2, 2, 1, 2]), "center")
        b = _sample(_cycle([3, 3, 4, 4, 3, 4]), second)
        with pytest.raises(MethodError, match="different singleton rules") as err:
            svy.combine_samples([a, b])
        assert "combined.update_design(singleton=" in err.value.hint

    def test_an_explicit_mapping_is_refused_on_cross_sections(self):
        # Stratum 2 has one PSU in each cycle.
        rule = Singleton("collapse", using={2: 1})
        a = _sample(_cycle([1, 1, 2, 2, 1, 2], psu=(1, 2, 3, 3, 1, 3)), rule)
        b = _sample(_cycle([1, 1, 2, 2, 1, 2], psu=(1, 2, 3, 3, 1, 3)), rule)
        with pytest.raises(MethodError, match=r"\(wave, stratum\) pairs"):
            svy.combine_samples([a, b])

    def test_the_carried_rule_applies_to_the_combined_strata(self):
        # Stratum 2 has one PSU in each cycle: two singletons once stacked.
        df = _cycle([1, 1, 2, 2, 1, 2], psu=(1, 2, 3, 3, 1, 3))
        a, b = _sample(df, "center"), _sample(df, "center")
        combined = svy.combine_samples([a, b])
        assert combined.n_singletons == 2
        assert _result(combined).n_singletons_detected == 2
        assert combined.estimation.mean("x").estimates[0].se > 0


def _ea_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ea": [1, 2, 3, 4, 5, 6],
            "region": ["N", "N", "N", "S", "S", "S"],
            "mos": [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
        }
    )


def _stage1(**kw) -> svy.Sample:
    design = svy.Design(mos="mos", stratum="region", psu="ea", **kw)
    return svy.Sample(_ea_frame(), design).sampling.pps_sys(n=2, by="region", rstate=5)


def _stage2() -> svy.Sample:
    ea = [e for e in range(1, 7) for _ in range(5)]
    frame = pl.DataFrame(
        {
            "hid": list(range(len(ea))),
            "ea": ea,
            "region": ["N" if e <= 3 else "S" for e in ea],
            "y": np.random.default_rng(3).normal(20, 4, len(ea)),
        }
    )
    selected = _stage1()._data["ea"].to_list()
    frame = frame.filter(pl.col("ea").is_in(selected))
    return svy.Sample(frame, svy.Design(stratum="region", psu="ea")).sampling.srs(
        n=3, by="ea", rstate=9
    )


class TestAddStage:
    def test_the_stage_one_rule_is_carried(self):
        combined = _stage1(singleton="center").sampling.add_stage(_stage2())
        assert combined.design.singleton == Singleton("center")

    def test_no_rule_stays_no_rule(self):
        assert _stage1().sampling.add_stage(_stage2()).design.singleton is None
