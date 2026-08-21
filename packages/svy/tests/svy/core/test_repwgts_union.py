# tests/svy/core/test_repwgts_union.py
"""RepWeights as a tagged union — see docs/design/rep-weights-tagged-union.md."""

from __future__ import annotations

import msgspec
import pytest

import svy

from svy.core.enumerations import EstimationMethod
from svy.core.repwgts import (
    BootstrapWgts,
    BrrWgts,
    JackknifeWgts,
    Poisson,
    RaoWu,
    RepWeights,
    RepWgts,
    SdrWgts,
)
from svy.errors import MethodError


# ---------------------------------------------------------------------------
# The variants carry only their own parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant, expected_method",
    [
        (BootstrapWgts(prefix="w", n_reps=10), EstimationMethod.BOOTSTRAP),
        (JackknifeWgts(prefix="w", n_reps=10), EstimationMethod.JACKKNIFE),
        (BrrWgts(prefix="w", n_reps=10), EstimationMethod.BRR),
        (SdrWgts(prefix="w", n_reps=10), EstimationMethod.SDR),
    ],
)
def test_method_property_returns_the_enum(variant, expected_method):
    """`.method` stays an EstimationMethod so existing comparisons hold."""
    assert variant.method == expected_method
    assert variant.method == expected_method.value  # StrEnum


@pytest.mark.parametrize(
    "ctor, kwargs",
    [
        (BootstrapWgts, {"fay_coef": 0.5}),
        (BootstrapWgts, {"paired": True}),
        (JackknifeWgts, {"kind": Poisson()}),
        (JackknifeWgts, {"fay_coef": 0.5}),
        (BrrWgts, {"kind": Poisson()}),
        (BrrWgts, {"paired": True}),
        (SdrWgts, {"fay_coef": 0.5}),
    ],
)
def test_foreign_parameters_are_unrepresentable(ctor, kwargs):
    """The flat struct accepted all of these and never read them."""
    with pytest.raises(TypeError):
        ctor(prefix="w", n_reps=10, **kwargs)


def test_rao_wu_cannot_carry_calibration_domains():
    """Calibrating replicates is not a Rao-Wu concept."""
    with pytest.raises(TypeError):
        RaoWu(calib_domains=("PROV",))


def test_fay_is_a_value_not_a_type():
    """Plain BRR is Fay's BRR at 0.0, so there is no separate FayWgts."""
    plain, fay = BrrWgts(prefix="w", n_reps=32), BrrWgts(prefix="w", n_reps=32, fay_coef=0.5)
    assert type(plain) is type(fay)
    assert plain.coefficients()[0] == pytest.approx(1 / 32)
    assert fay.coefficients()[0] == pytest.approx(1 / (32 * 0.5**2))


# ---------------------------------------------------------------------------
# Coefficients live on the variant
# ---------------------------------------------------------------------------


def test_bootstrap_kinds_share_the_coefficient():
    """Both bootstraps are 1/B; the kernel dispatches on Bootstrap as one arm."""
    rw = BootstrapWgts(prefix="w", n_reps=500)
    poisson = BootstrapWgts(prefix="w", n_reps=500, kind=Poisson())
    assert rw.coefficients() == poisson.coefficients() == [1 / 500] * 500


def test_coefficients_match_the_rust_kernel():
    """The Python-side coefficients must agree with replicate_coefficients()."""
    n = 64
    assert BootstrapWgts(prefix="w", n_reps=n).coefficients() == [1.0 / n] * n
    assert BrrWgts(prefix="w", n_reps=n, fay_coef=0.3).coefficients() == pytest.approx(
        [1.0 / (n * (1 - 0.3) ** 2)] * n
    )
    assert JackknifeWgts(prefix="w", n_reps=n).coefficients() == pytest.approx([(n - 1) / n] * n)


def test_jackknife_rscales_override_the_default():
    rs = tuple(0.5 for _ in range(10))
    assert JackknifeWgts(prefix="w", n_reps=10, rscales=rs).coefficients() == list(rs)


# ---------------------------------------------------------------------------
# The factory keeps the pre-union call surface
# ---------------------------------------------------------------------------


def test_factory_positional_call_still_works():
    rw = RepWeights("bootstrap", "rep", 10)
    assert isinstance(rw, BootstrapWgts)
    assert (rw.prefix, rw.n_reps) == ("rep", 10)


def test_factory_routes_to_the_right_variant():
    assert isinstance(RepWeights(method="bs", prefix="w", n_reps=10), BootstrapWgts)
    assert isinstance(RepWeights(method="jk", prefix="w", n_reps=10), JackknifeWgts)
    assert isinstance(RepWeights(method="brr", prefix="w", n_reps=10), BrrWgts)
    assert isinstance(RepWeights(method="sdr", prefix="w", n_reps=10), SdrWgts)


@pytest.mark.parametrize("given", ["poisson", "POISSON", " Poisson "])
def test_factory_normalizes_the_bootstrap_kind(given):
    rw = RepWeights(method="bootstrap", prefix="w", n_reps=10, kind=given)
    assert isinstance(rw.kind, Poisson)


def test_factory_accepts_a_constructed_kind():
    rw = RepWeights(
        method="bootstrap", prefix="w", n_reps=10, kind=Poisson(calib_domains=("a", "b"))
    )
    assert rw.kind.calib_domains == ("a", "b")


@pytest.mark.parametrize(
    "kwargs, param",
    [
        ({"method": "bootstrap", "fay_coef": 0.5}, "fay_coef"),
        ({"method": "bootstrap", "paired": True}, "paired"),
        ({"method": "brr", "kind": "poisson"}, "kind"),
        ({"method": "jackknife", "fay_coef": 0.5}, "fay_coef"),
        ({"method": "sdr", "kind": "poisson"}, "kind"),
    ],
)
def test_factory_rejects_foreign_parameters_and_names_the_owner(kwargs, param):
    """The wrapper is the boundary that enforces what the union cannot express."""
    with pytest.raises(MethodError) as exc:
        RepWeights(prefix="w", n_reps=10, **kwargs)
    rendered = str(exc.value)
    assert param in rendered
    assert "Wgts" in rendered  # hint names the variant that owns it


def test_missing_required_argument_message_is_preserved():
    with pytest.raises(TypeError, match="Missing required argument 'method'"):
        RepWeights(prefix="rep", n_reps=50)


def test_taylor_rejection_keeps_its_valueerror_contract():
    with pytest.raises(ValueError, match="not a valid replication method"):
        RepWeights(method=EstimationMethod.TAYLOR, prefix="w", n_reps=10)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_round_trips_through_the_tagged_union():
    rw = BootstrapWgts(prefix="bsw", n_reps=1000, kind=Poisson(calib_domains=("PROV", "SEX")))
    raw = msgspec.json.encode(rw)
    back = msgspec.json.Decoder(RepWgts).decode(raw)
    assert back == rw
    assert back.kind.calib_domains == ("PROV", "SEX")


def test_tag_is_the_method_name():
    """Keeps the wire format aligned with the pre-union `method` field."""
    raw = msgspec.json.encode(BrrWgts(prefix="w", n_reps=32, fay_coef=0.5))
    assert msgspec.json.decode(raw)["method"] == "BRR"


# ---------------------------------------------------------------------------
# Design integration
# ---------------------------------------------------------------------------


def test_design_accepts_every_variant():
    for variant in (
        BootstrapWgts(prefix="w", n_reps=10),
        JackknifeWgts(prefix="w", n_reps=10),
        BrrWgts(prefix="w", n_reps=10),
        SdrWgts(prefix="w", n_reps=10),
    ):
        assert svy.Design(wgt="wgt", rep_wgts=variant).rep_wgts is variant


def test_design_rejects_a_non_variant():
    with pytest.raises(TypeError, match="rep_wgts"):
        svy.Design(wgt="wgt", rep_wgts="not-a-repwgts")


def test_update_rep_weights_carries_kind_within_the_same_method():
    d = svy.Design(wgt="w", rep_wgts=BootstrapWgts(prefix="b", n_reps=10, kind=Poisson()))
    updated = d.update_rep_weights(n_reps=20)
    assert isinstance(updated.rep_wgts.kind, Poisson)
    assert updated.rep_wgts.n_reps == 20


def test_update_rep_weights_drops_kind_when_the_method_changes():
    """A bootstrap kind means nothing on a jackknife design."""
    d = svy.Design(wgt="w", rep_wgts=BootstrapWgts(prefix="b", n_reps=10, kind=Poisson()))
    updated = d.update_rep_weights(method="jackknife")
    assert isinstance(updated.rep_wgts, JackknifeWgts)
    assert not hasattr(updated.rep_wgts, "kind")


def test_variants_are_publicly_exported():
    """They are on the read path: rep_wgts hands one back."""
    for name in (
        "RepWgts",
        "BootstrapWgts",
        "JackknifeWgts",
        "BrrWgts",
        "SdrWgts",
        "RaoWu",
        "Poisson",
    ):
        assert hasattr(svy, name), name
