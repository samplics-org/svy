# tests/svy/core/test_to_code.py
"""
to_code() on design objects: source runnable with only ``import svy`` that
rebuilds an equal object, or a guiding error when there is no source form.
"""

from __future__ import annotations

import numpy as np
import pytest

import svy

from svy.core.terms import _ComposedCap
from svy.errors import MethodError
from svy.errors.weighting_errors import WeightingError


def _rebuilt(obj):
    return eval(obj.to_code(), {"svy": svy})


def _assert_round_trip(obj):
    code = obj.to_code()
    assert code.startswith("svy.")
    back = _rebuilt(obj)
    assert type(back) is type(obj)
    assert back == obj
    return code


# ---------------------------------------------------------------------------
# PopSize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pop_size", [svy.PopSize(psu="N_psu", ssu="N_ssu"), svy.PopSize(psu="N_psu")]
)
def test_pop_size(pop_size):
    code = _assert_round_trip(pop_size)
    assert ("ssu=" in code) is (pop_size.ssu is not None)


# ---------------------------------------------------------------------------
# Replicate weights
# ---------------------------------------------------------------------------

REP_WGTS = {
    "bootstrap_default": svy.BootstrapWgts(prefix="bs", n_reps=50),
    "bootstrap_poisson": svy.BootstrapWgts(
        prefix="bs_", n_reps=200, kind="poisson", df=120.5, padding=3, wgt="w"
    ),
    "bootstrap_scale": svy.BootstrapWgts(prefix="bs", n_reps=4, scale=0.3),
    "jackknife_default": svy.JackknifeWgts(prefix="jk", n_reps=10),
    "jk1": svy.JackknifeWgts(prefix="jk", n_reps=10, kind="jk1", df=9),
    "jk2": svy.JackknifeWgts(prefix="jk", n_reps=8, kind="jk2", padding=0),
    "jkn_units": svy.JackknifeWgts(
        prefix="jkw_", n_reps=5, kind="jkn", stratum=("reg", "urb"), psu="ea"
    ),
    "jkn_rep_coefs": svy.JackknifeWgts(
        prefix="jk", n_reps=5, kind="jkn", rep_coefs=(0.5, 0.5, 2 / 3, 2 / 3, 2 / 3)
    ),
    "brr": svy.BrrWgts(prefix="brr", n_reps=16),
    "fay": svy.BrrWgts(prefix="brr_", n_reps=32, fay_coef=0.5, df=31.0),
    "brr_scale_vector": svy.BrrWgts(prefix="brr", n_reps=4, scale=(0.1, 0.2, 0.3, 0.4)),
    "sdr": svy.SdrWgts(prefix="sdr", n_reps=80, scale=0.05, psu=("a",), stratum="s"),
    "factory": svy.RepWeights(method="brr", prefix="r", n_reps=4, fay_coef=0.3),
}


@pytest.mark.parametrize("rep", list(REP_WGTS.values()), ids=list(REP_WGTS))
def test_rep_wgts(rep):
    code = _assert_round_trip(rep)
    assert code.startswith(f"svy.{type(rep).__name__}(")
    assert _rebuilt(rep).coef_source == rep.coef_source


def test_rep_wgts_code_omits_defaults():
    assert svy.BootstrapWgts(prefix="bs", n_reps=50).to_code() == (
        "svy.BootstrapWgts(prefix='bs', n_reps=50)"
    )
    assert svy.BrrWgts(prefix="b", n_reps=4, fay_coef=0.5).to_code() == (
        "svy.BrrWgts(prefix='b', n_reps=4, fay_coef=0.5)"
    )


def test_rep_wgts_repr_quotes_method_and_names_fay_coef():
    shown = repr(svy.BrrWgts(prefix="b", n_reps=4, fay_coef=0.5))
    assert shown.startswith("RepWeights(method='BRR', ")
    assert "fay_coef=0.5" in shown
    assert "fay=" not in shown
    assert repr(svy.SdrWgts(prefix="s", n_reps=4)).startswith("RepWeights(method='SDR', ")


# ---------------------------------------------------------------------------
# WgtAdjustment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adj",
    [
        svy.WgtAdjustment(kind="trimming", prev_wgt="w0", new_wgt="w1"),
        svy.WgtAdjustment(
            kind="raking", prev_wgt="w0", new_wgt="w1", cells=("c1", "c2"), pins_total=False
        ),
        svy.WgtAdjustment(
            kind="calibration", prev_wgt="w0", new_wgt="w1", cells=("c",), aux=("x1", "x2")
        ),
    ],
)
def test_wgt_adjustment(adj):
    _assert_round_trip(adj)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

SINGLETONS = [
    svy.Singleton("center"),
    svy.Singleton("scale", domains="apply", on_domain_singletons="warn"),
    svy.Singleton("skip"),
    svy.Singleton("self_representing"),
    svy.Singleton("pool"),
    svy.Singleton("pool", name="lonely"),
    svy.Singleton("collapse"),
    svy.Singleton(
        "collapse", using="next", within=["region", "urban"], order_by="size", descending=True
    ),
    svy.Singleton("collapse", using={("a", 1): ("a", 2), ("b", 1): ("b", 3)}, rstate=7),
]


@pytest.mark.parametrize("rule", SINGLETONS, ids=repr)
def test_singleton(rule):
    _assert_round_trip(rule)


def test_singleton_callable_using_is_refused():
    rule = svy.Singleton("collapse", using=lambda s, c: c[0])
    with pytest.raises(MethodError) as exc:
        rule.to_code()
    assert exc.value.param == "using"
    assert exc.value.hint


def test_singleton_generator_rstate_is_refused():
    rule = svy.Singleton("collapse", rstate=np.random.default_rng(1))
    with pytest.raises(MethodError) as exc:
        rule.to_code()
    assert exc.value.param == "rstate"
    assert "rstate=42" in exc.value.hint


# ---------------------------------------------------------------------------
# Trimming thresholds and TrimConfig
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "absolute": svy.Threshold.absolute(40),
    "quantile": svy.Threshold.quantile(0.99),
    "scaled_quantile": 1.5 * svy.Threshold.quantile(0.95),
    "median": svy.Threshold("median"),
    "median_k": svy.Threshold("median", 3.5),
    "sd_negative": svy.Threshold("sd", -2.0),
    "absolute_negative": svy.Threshold("absolute", -3.0),
    "numpy_k": svy.Threshold("mean", np.float64(2.5)),
    "cap_alias": svy.Cap("iqr", 6),
}


@pytest.mark.parametrize("threshold", list(THRESHOLDS.values()), ids=list(THRESHOLDS))
def test_threshold(threshold):
    _assert_round_trip(threshold)


def test_threshold_code_reads_like_the_api():
    assert svy.Threshold.quantile(0.99).to_code() == "svy.Threshold.quantile(0.99)"
    assert svy.Threshold.absolute(0.9).to_code() == "svy.Threshold.absolute(0.9)"
    assert svy.Threshold("median", 3.5).to_code() == "svy.Threshold('median', 3.5)"


COMPOSED = {
    "sum": svy.Threshold("median") + 6 * svy.Threshold("iqr"),
    "difference": svy.Threshold("mean") - 2 * svy.Threshold("sd"),
    "chain": svy.Threshold("median")
    + svy.Threshold("iqr")
    - svy.Threshold.absolute(2)
    - 2 * svy.Threshold.quantile(0.9)
    + (svy.Threshold("mean") - svy.Threshold("sd")),
}


@pytest.mark.parametrize("composed", list(COMPOSED.values()), ids=list(COMPOSED))
def test_composed_threshold(composed):
    assert isinstance(composed, _ComposedCap)
    _assert_round_trip(composed)


def test_composed_threshold_equality_is_by_value():
    a = svy.Threshold("median") + 6 * svy.Threshold("iqr")
    b = svy.Threshold("median") + 6 * svy.Threshold("iqr")
    assert a == b
    assert hash(a) == hash(b)
    assert a != svy.Threshold("median") + 5 * svy.Threshold("iqr")
    assert a != svy.Threshold("median")
    assert (svy.Threshold("mean") - svy.Threshold("sd")).to_code() == (
        "svy.Threshold('mean') - svy.Threshold('sd')"
    )


TRIM_CONFIGS = {
    "number": svy.TrimConfig(upper=40.0),
    "numpy_number": svy.TrimConfig(upper=np.float64(12.5)),
    "every_field": svy.TrimConfig(
        upper=svy.Threshold("median") + 6 * svy.Threshold("iqr"),
        lower=svy.Threshold.quantile(0.01),
        by=["region", "urban"],
        redistribute=False,
        min_cell_size=5,
        max_iter=50,
        tol=1e-8,
    ),
    "by_tuple": svy.TrimConfig(lower=0.5, by=("region",)),
    "by_str": svy.TrimConfig(upper=svy.Threshold.absolute(3), by="region"),
}


@pytest.mark.parametrize("config", list(TRIM_CONFIGS.values()), ids=list(TRIM_CONFIGS))
def test_trim_config(config):
    _assert_round_trip(config)


def test_trim_config_code_omits_defaults():
    assert svy.TrimConfig(upper=40.0).to_code() == "svy.TrimConfig(upper=40.0)"


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_trim_config_callable_bound_is_refused(side):
    def cutoff(w):
        return float(np.max(w))

    config = svy.TrimConfig(**{side: cutoff})
    with pytest.raises(WeightingError) as exc:
        config.to_code()
    assert exc.value.code == "TO_CODE_CALLABLE"
    assert exc.value.param == side
    assert exc.value.got == "cutoff"
    assert "svy.Threshold" in exc.value.hint


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------


def test_design_with_every_field_and_part():
    d = svy.Design(
        case_id=("clu", "hh", "ln"),
        wave="round",
        stratum=("region", "urban"),
        wgt="w",
        prob="p",
        hit="hits",
        mos="size",
        psu=("clu", "seg"),
        ssu=("hh",),
        pop_size=svy.PopSize(psu="N_psu", ssu="N_ssu"),
        wr=True,
        rep_wgts=svy.BrrWgts(prefix="brr", n_reps=8, fay_coef=0.3, scale=0.2, padding=2),
        wgt_adjustment=svy.WgtAdjustment(
            kind="poststratification", prev_wgt="w0", new_wgt="w", cells=("region",)
        ),
        singleton=svy.Singleton("collapse", using={"a": "b"}, within="region"),
    )
    code = _assert_round_trip(d)
    for name in ("rep_wgts=svy.BrrWgts(", "wgt_adjustment=svy.WgtAdjustment(", "svy.PopSize("):
        assert name in code


@pytest.mark.parametrize(
    "design",
    [
        svy.Design(),
        svy.Design(wgt="w"),
        svy.Design(stratum="s", psu="p", wgt="w", pop_size="N"),
        svy.Design(psu="p", wgt="w", pop_size=svy.PopSize(psu="N")),
        svy.Design(stratum=["s1", "s2"], psu=["p1", "p2"], case_id=["a", "b"], wgt="w"),
        svy.Design(stratum="s", psu="p", wgt="w", singleton="center"),
        svy.Design(wgt="w", rep_wgts=svy.JackknifeWgts(prefix="jk", n_reps=4, kind="jk2")),
    ],
)
def test_design(design):
    _assert_round_trip(design)


def test_design_with_a_callable_singleton_is_refused():
    d = svy.Design(stratum="s", psu="p", singleton=svy.Singleton("collapse", using=lambda s, c: 0))
    with pytest.raises(MethodError):
        d.to_code()


def test_no_private_to_code_remains():
    from svy.core.repwgts import _RepWgtsBase

    for cls in (svy.Design, svy.Singleton, svy.WgtAdjustment, _RepWgtsBase, svy.PopSize):
        assert not hasattr(cls, "_to_code")
