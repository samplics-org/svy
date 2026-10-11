"""n_draws=R returns R samples, each equal to a single call with child r of the seed."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import MethodError
from svy.errors.base_errors import SvyUserWarning
from svy.selection import pps as pps_mod
from svy.selection import srs as srs_mod


N = 600
FRAME = pl.DataFrame(
    {
        "id": range(N),
        "region": [f"R{i % 3}" for i in range(N)],
        "zone": [f"Z{i % 2}" for i in range(N)],
        "m": np.linspace(1.0, 9.0, N) ** 2,
        "y": np.linspace(0.0, 1.0, N),
    }
)
R = 6

CASES = {
    "srs": ("srs", {"stratum": "region"}, {"n": 5}),
    "srs_by": ("srs", {"stratum": "region"}, {"n": 3, "by": "zone"}),
    "srs_map": ("srs", {"stratum": "region"}, {"n": {"R0": 2, "R1": 4, "R2": 0}}),
    "srs_wr": ("srs", {}, {"n": 30, "wr": True}),
    "srs_where": ("srs", {"stratum": "region"}, {"n": 4, "where": svy.col("y") > 0.5}),
    "srs_random": (
        "srs",
        {"stratum": "region"},
        {"n": 4, "order_by": "y", "order_type": "random"},
    ),
    "pps_sys": ("pps_sys", {"stratum": "region", "mos": "m"}, {"n": 5}),
    "pps_sys_cert": ("pps_sys", {"mos": "m"}, {"n": 40, "certainty_threshold": 0.5}),
    "pps_sys_random": (
        "pps_sys",
        {"mos": "m"},
        {"n": 5, "order_by": "zone", "order_type": "random"},
    ),
    "pps_wr": ("pps_wr", {"mos": "m"}, {"n": 8}),
    "pps_brewer": ("pps_brewer", {"stratum": "region", "mos": "m"}, {"n": 4}),
    "pps_murphy": ("pps_murphy", {"stratum": "region", "mos": "m"}, {"n": 2}),
    "pps_rs": ("pps_rs", {"mos": "m"}, {"n": 4}),
}


def _draw(case, **extra):
    method, design, kwargs = CASES[case]
    sample = svy.Sample(FRAME, svy.Design(**design))
    return getattr(sample.sampling, method)(**kwargs, **extra)


@pytest.mark.parametrize("case", CASES)
def test_each_draw_equals_a_single_call(case):
    drawn = _draw(case, n_draws=R, rstate=2026)
    assert isinstance(drawn, list) and len(drawn) == R
    children = np.random.SeedSequence(2026).spawn(R)
    for r, sample in enumerate(drawn):
        single = _draw(case, rstate=children[r])
        assert sample.data.equals(single.data)
        assert sample.design == single.design


@pytest.mark.parametrize("case", ["srs", "pps_sys", "pps_brewer"])
def test_draws_differ(case):
    drawn = _draw(case, n_draws=R, rstate=7)
    assert len({tuple(sorted(s.data["id"].to_list())) for s in drawn}) > 1


@pytest.mark.parametrize("case", ["srs", "pps_sys"])
def test_without_batch_kernel_the_draws_are_the_same(monkeypatch, case):
    want = [s.data for s in _draw(case, n_draws=R, rstate=11)]
    monkeypatch.setattr(srs_mod, "_select_srs_batch_rs", None)
    monkeypatch.setattr(pps_mod, "_select_pps_batch_rs", None)
    got = [s.data for s in _draw(case, n_draws=R, rstate=11)]
    assert all(a.equals(b) for a, b in zip(want, got))


def test_input_sample_is_untouched():
    sample = svy.Sample(FRAME, svy.Design(stratum="region"))
    before = sample.data
    sample.sampling.srs(5, n_draws=R, rstate=1)
    assert sample.data.equals(before)
    assert sample.design.wgt is None


def test_one_draw_is_a_list_of_one():
    drawn = _draw("srs", n_draws=1, rstate=1)
    assert isinstance(drawn, list) and len(drawn) == 1


def test_findings_are_on_every_draw():
    sample = svy.Sample(FRAME, svy.Design(stratum="region"))
    drawn = sample.sampling.srs(250, wr=True, n_draws=3, rstate=1)
    single = svy.Sample(FRAME, svy.Design(stratum="region")).sampling.srs(250, wr=True, rstate=1)
    codes = sorted(f.code for f in single.warnings)
    assert codes
    assert all(sorted(f.code for f in s.warnings) == codes for s in drawn)


def test_where_with_no_eligible_rows_gives_null_selection_columns():
    with pytest.warns(SvyUserWarning):
        drawn = _draw("srs", where=svy.col("y") > 2.0, n_draws=2, rstate=1)
    assert all(s.data["svy_prob_selection"].null_count() == N for s in drawn)


@pytest.mark.parametrize("bad", [0, -3])
def test_n_draws_below_one_raises(bad):
    with pytest.raises(MethodError) as exc:
        _draw("srs", n_draws=bad)
    assert exc.value.code == "INVALID_RANGE"


@pytest.mark.parametrize("bad", [1.5, True, "10"])
def test_n_draws_not_an_int_raises(bad):
    with pytest.raises(MethodError) as exc:
        _draw("pps_sys", n_draws=bad)
    assert exc.value.code == "INVALID_TYPE"
