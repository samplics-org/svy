# tests/svy/core/test_to_code.py
"""Design parts print as, and write out as, the code that rebuilds them."""

import datetime

import pytest

import svy

from svy.errors import DesignError


NS = {"svy": svy, "datetime": datetime}

REP_WGTS = [
    svy.BrrWgts(prefix="r", n_reps=4, fay_coef=0.3),
    svy.BrrWgts(prefix="r", n_reps=4),
    svy.JackknifeWgts(prefix="jk", n_reps=10, df=9),
    svy.JackknifeWgts(prefix="jk", n_reps=10, kind="jkn", stratum="s", psu="p"),
    svy.BootstrapWgts(prefix="bs", n_reps=50),
    svy.BootstrapWgts(prefix="bs", n_reps=50, kind="poisson"),
    svy.SdrWgts(prefix="sdr", n_reps=80, padding=2),
]

PARTS = [
    *REP_WGTS,
    svy.WgtAdjustment(kind="raking", prev_wgt="w0", new_wgt="w", aux=("a", "b")),
    svy.Singleton("collapse"),
    svy.Singleton("skip"),
    svy.PopSize(psu="N"),
    svy.PopSize(psu="N", ssu="M"),
]

DESIGNS = [
    svy.Design(wgt="w"),
    svy.Design(stratum="st", psu="psu", wgt="w", pop_size="fpc"),
    svy.Design(stratum=("a", "b"), psu="psu", ssu="ssu", wgt="w", wr=True),
    svy.Design(
        stratum="st",
        psu="psu",
        wgt="w",
        rep_wgts=REP_WGTS[0],
        singleton=svy.Singleton("collapse"),
        pop_size=svy.PopSize(psu="N"),
        wgt_adjustment=PARTS[len(REP_WGTS)],
    ),
]


@pytest.mark.parametrize("obj", PARTS + DESIGNS, ids=repr)
def test_to_code_rebuilds_the_object(obj):
    assert eval(obj.to_code(), NS) == obj


@pytest.mark.parametrize("rw", REP_WGTS, ids=repr)
def test_rep_wgts_repr_is_valid_python(rw):
    rebuilt = eval(f"svy.{rw!r}", NS)
    assert type(rebuilt) is type(rw)
    assert (rebuilt.prefix, rebuilt.n_reps) == (rw.prefix, rw.n_reps)


def test_rep_wgts_repr_names_the_fay_coefficient():
    rw = REP_WGTS[0]
    assert "fay_coef=0.3" in repr(rw)
    assert eval(f"svy.{rw!r}", NS).fay_coef == 0.3


def test_design_repr_keeps_the_full_replicate_spec():
    shown = repr(DESIGNS[-1])
    assert "fay_coef=0.3" in shown
    assert "method='BRR'" in shown


def test_design_repr_shows_the_weight_adjustment():
    assert "wgt_adjustment=WgtAdjustment(kind='raking'" in repr(DESIGNS[-1])
    assert "wgt_adjustment" not in repr(DESIGNS[0])


def test_unrepresentable_singleton_rule_still_refuses():
    rule = svy.Singleton("collapse", using=lambda s: s)
    with pytest.raises(DesignError, match="callable"):
        rule.to_code()
