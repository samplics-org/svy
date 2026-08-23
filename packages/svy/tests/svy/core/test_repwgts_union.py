# tests/svy/core/test_repwgts_union.py
"""RepWeights as a tagged union — see docs/design/rep-weights-tagged-union.md."""

from __future__ import annotations

import msgspec
import polars as pl
import pytest

import svy

from svy.core.repwgts import (
    BootstrapWgts,
    BrrWgts,
    JackknifeWgts,
    RepWeights,
    RepWgts,
    SdrWgts,
    resolve_rep_variant,
)
from svy.core.warnings import WarnCode
from svy.errors import MethodError


# ---------------------------------------------------------------------------
# The variants carry only their own parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant, expected_method",
    [
        (BootstrapWgts(prefix="w", n_reps=10), "Bootstrap"),
        (JackknifeWgts(prefix="w", n_reps=10), "Jackknife"),
        (BrrWgts(prefix="w", n_reps=10), "BRR"),
        (SdrWgts(prefix="w", n_reps=10), "SDR"),
    ],
)
def test_method_property_is_the_display_label(variant, expected_method):
    """`.method` is a plain string naming the method family."""
    assert variant.method == expected_method
    assert isinstance(variant.method, str)


@pytest.mark.parametrize(
    "ctor, kwargs",
    [
        (BootstrapWgts, {"fay_coef": 0.5}),
        (JackknifeWgts, {"fay_coef": 0.5}),
        (BrrWgts, {"kind": "poisson"}),
        (SdrWgts, {"fay_coef": 0.5}),
    ],
)
def test_foreign_parameters_are_unrepresentable(ctor, kwargs):
    """The flat struct accepted all of these and never read them."""
    with pytest.raises(TypeError):
        ctor(prefix="w", n_reps=10, **kwargs)


def test_jackknife_rejects_a_bootstrap_kind():
    """Both variants have a ``kind``; the vocabularies are not interchangeable."""
    with pytest.raises(MethodError) as exc:
        JackknifeWgts(prefix="w", n_reps=10, kind="poisson")
    assert "jk1" in str(exc.value)


@pytest.mark.parametrize(
    "given, expected",
    [("jk1", "jk1"), ("JKN", "jkn"), ("jk_2", "jk2"), ("paired", "jk2")],
)
def test_jackknife_kind_aliases_normalize(given, expected):
    assert JackknifeWgts(prefix="w", n_reps=10, kind=given).kind == expected


def test_jackknife_kind_is_unspecified_by_default():
    """None is not JK1: same number, different statement about who said so."""
    rw = JackknifeWgts(prefix="w", n_reps=10)
    assert rw.kind is None
    assert rw.coefficients() == [0.9] * 10


def test_declared_jkn_without_coefficients_refuses_to_guess():
    """An unmet claim fails; an absent one falls back."""
    with pytest.raises(MethodError) as exc:
        JackknifeWgts(prefix="w", n_reps=10, kind="jkn").coefficients()
    assert "scale" in str(exc.value)


def test_declared_jk2_uses_one_not_the_global():
    assert JackknifeWgts(prefix="w", n_reps=10, kind="jk2").coefficients() == [1.0] * 10


def test_bootstrap_kind_is_a_value_not_a_type():
    """The two kinds differ in how replicates are drawn, not in what they carry.

    Calibration used to live on the Poisson kind, which made it carry data the
    other did not. With calibration moved to the weighting adjustments, both
    kinds are bare labels and a field is the right shape.
    """
    rw = BootstrapWgts(prefix="w", n_reps=10)
    po = BootstrapWgts(prefix="w", n_reps=10, kind="poisson")
    assert type(rw) is type(po)
    assert (rw.kind, po.kind) == ("rao-wu", "poisson")
    assert rw.coefficients() == po.coefficients()


def test_fay_is_a_value_not_a_type():
    """Plain BRR is Fay's BRR at 0.0, so there is no separate FayWgts."""
    plain, fay = BrrWgts(prefix="w", n_reps=32), BrrWgts(prefix="w", n_reps=32, fay_coef=0.5)
    assert type(plain) is type(fay)
    assert plain.coefficients()[0] == pytest.approx(1 / 32)
    assert fay.coefficients()[0] == pytest.approx(1 / (32 * 0.5**2))


# ---------------------------------------------------------------------------
# Coefficients live on the variant
# ---------------------------------------------------------------------------


def test_coefficients_match_the_rust_kernel():
    """The Python-side coefficients must agree with replicate_coefficients()."""
    n = 64
    assert BootstrapWgts(prefix="w", n_reps=n).coefficients() == [1.0 / n] * n
    assert BrrWgts(prefix="w", n_reps=n, fay_coef=0.3).coefficients() == pytest.approx(
        [1.0 / (n * (1 - 0.3) ** 2)] * n
    )
    assert JackknifeWgts(prefix="w", n_reps=n).coefficients() == pytest.approx([(n - 1) / n] * n)


def test_user_scale_overrides_the_default():
    rs = tuple(0.5 for _ in range(10))
    assert BootstrapWgts(prefix="w", n_reps=10, scale=rs).coefficients() == list(rs)


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
    assert rw.kind == "poisson"


@pytest.mark.parametrize("given", ["rao-wu", "RAO_WU", "rw", "raowu"])
def test_bootstrap_kind_aliases_normalize(given):
    assert BootstrapWgts(prefix="w", n_reps=10, kind=given).kind == "rao-wu"


@pytest.mark.parametrize(
    "kwargs, param",
    [
        ({"method": "bootstrap", "fay_coef": 0.5}, "fay_coef"),
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


@pytest.mark.parametrize(
    "given, expected",
    [
        ("bootstrap", BootstrapWgts),
        ("BOOTSTRAP", BootstrapWgts),
        ("bs", BootstrapWgts),
        ("jackknife", JackknifeWgts),
        ("jk", JackknifeWgts),
        ("jkn", JackknifeWgts),
        ("brr", BrrWgts),
        ("BRR", BrrWgts),
        ("  sdr  ", SdrWgts),
        ("Bootstrap", BootstrapWgts),
        ("Jackknife", JackknifeWgts),
        ("BRR", BrrWgts),
        ("SDR", SdrWgts),
    ],
)
def test_method_names_resolve_straight_to_a_variant(given, expected):
    """A method name resolves straight to its variant, with no enum in
    between."""
    assert resolve_rep_variant(given) is expected


@pytest.mark.parametrize("given", ["taylor", "Taylor", "TAYLOR", "Taylor"])
def test_taylor_is_rejected_the_same_way_however_it_is_spelled(given):
    """A real method that carries no replicate weights, not a typo."""
    with pytest.raises(ValueError, match="not a valid replication method"):
        resolve_rep_variant(given)


def test_a_typo_gets_a_different_message_from_taylor():
    with pytest.raises(ValueError, match="Unknown replication method"):
        resolve_rep_variant("bootstrp")


def test_non_string_method_raises_type_error():
    with pytest.raises(TypeError, match="'method' must be a string"):
        resolve_rep_variant(42)


def test_missing_required_argument_message_is_preserved():
    with pytest.raises(TypeError, match="Missing required argument 'method'"):
        RepWeights(prefix="rep", n_reps=50)


def test_taylor_rejection_keeps_its_valueerror_contract():
    with pytest.raises(ValueError, match="not a valid replication method"):
        RepWeights(method="Taylor", prefix="w", n_reps=10)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_round_trips_through_the_tagged_union():
    rw = BootstrapWgts(prefix="bsw", n_reps=1000, kind="poisson")
    raw = msgspec.json.encode(rw)
    back = msgspec.json.Decoder(RepWgts).decode(raw)
    assert back == rw
    assert back.kind == "poisson"


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
    d = svy.Design(wgt="w", rep_wgts=BootstrapWgts(prefix="b", n_reps=10, kind="poisson"))
    updated = d.update_rep_weights(n_reps=20)
    assert updated.rep_wgts.kind == "poisson"
    assert updated.rep_wgts.n_reps == 20


def test_update_rep_weights_drops_kind_when_the_method_changes():
    """A bootstrap kind means nothing on a jackknife design.

    Both variants carry a ``kind`` now, with different vocabularies, so the
    invariant is that the value does not survive the change of method -- not
    that the field is absent.
    """
    d = svy.Design(wgt="w", rep_wgts=BootstrapWgts(prefix="b", n_reps=10, kind="poisson"))
    updated = d.update_rep_weights(method="jackknife")
    assert isinstance(updated.rep_wgts, JackknifeWgts)
    assert updated.rep_wgts.kind is None


def test_variants_are_publicly_exported():
    """They are on the read path: rep_wgts hands one back."""

    for name in (
        "RepWgts",
        "BootstrapWgts",
        "JackknifeWgts",
        "BrrWgts",
        "SdrWgts",
    ):
        assert hasattr(svy, name), name


# =============================================================================
# Jackknife kind resolved against the design (Sample-level)
# =============================================================================
#
# The struct cannot do this: (n_h-1)/n_h needs per-stratum PSU counts and
# coefficients() has no frame. So it happens once at Sample construction.


def _jk_frame(strata, n_reps=4):
    n = len(strata)
    return pl.DataFrame(
        {
            "stratum": strata,
            "psu": [i // 2 + 1 for i in range(n)],
            "w": [5.0] * n,
            "y": [float(i) for i in range(n)],
            **{f"jw{r}": [5.0] * n for r in range(1, n_reps + 1)},
        }
    )


def _jk_sample(strata, *, n_reps=4, **rw_kwargs):
    df = _jk_frame(strata, n_reps)
    rw = JackknifeWgts(prefix="jw", n_reps=n_reps, **rw_kwargs)
    return svy.Sample(df, svy.Design(wgt="w", stratum="stratum", psu="psu", rep_wgts=rw))


def test_declared_jkn_derives_its_coefficients_from_a_balanced_design():
    s = _jk_sample([1, 1, 1, 1, 2, 2, 2, 2], n_reps=4, kind="jkn")
    assert s._design.rep_wgts.rep_coefs == (0.5,) * 4  # (n_h-1)/n_h, n_h=2
    assert s._design.rep_wgts.scale is None  # derived, not asserted


def test_unbalanced_jkn_is_not_derived_and_still_refuses():
    """The replicate->stratum mapping is the producer's, not svy's to infer."""
    s = _jk_sample([1, 1, 1, 1, 2, 2], n_reps=3, kind="jkn")
    assert s._design.rep_wgts.rep_coefs is None
    with pytest.raises(MethodError):
        s._design.rep_wgts.coefficients()


def test_user_scale_is_never_overwritten_by_derivation():
    s = _jk_sample([1, 1, 1, 1, 2, 2, 2, 2], n_reps=4, kind="jkn", scale=0.9)
    assert s._design.rep_wgts.rep_coefs is None
    assert s._design.rep_wgts.coefficients() == [0.9] * 4


def test_unspecified_kind_on_a_stratified_design_warns_but_does_not_guess():
    s = _jk_sample([1, 1, 1, 1, 2, 2, 2, 2], n_reps=4)
    assert s._design.rep_wgts.kind is None  # absence of a claim is not a claim
    assert s._design.rep_wgts.coefficients() == [0.75] * 4  # JK1 global, unchanged
    codes = {w.code for w in s.warnings.list()}
    assert WarnCode.JACKKNIFE_KIND_UNSPECIFIED in codes


def test_unspecified_kind_on_an_unstratified_design_is_silent():
    s = _jk_sample([1, 1, 1, 1], n_reps=4)
    codes = {w.code for w in s.warnings.list()}
    assert WarnCode.JACKKNIFE_KIND_UNSPECIFIED not in codes


def test_a_kind_that_disagrees_with_the_design_warns():
    """jk2 implies one replicate per stratum; here there are 2 strata, not 4."""
    s = _jk_sample([1, 1, 1, 1, 2, 2, 2, 2], n_reps=4, kind="jk2")
    codes = {w.code for w in s.warnings.list()}
    assert WarnCode.JACKKNIFE_KIND_UNSPECIFIED in codes


# =============================================================================
# The factory as a parse door
# =============================================================================


def test_factory_forwards_unknown_parameters_to_the_variant():
    """It resolves the name and forwards; msgspec owns the field checking."""
    rw = RepWeights(method="brr", prefix="b", n_reps=32, fay_coef=0.5, padding=3)
    assert isinstance(rw, BrrWgts)
    assert (rw.fay_coef, rw.padding) == (0.5, 3)


def test_a_foreign_parameter_is_refused_even_at_its_neutral_value():
    """fay_coef=0.0 on a bootstrap is still a Fay parameter on a design that has
    none. Callers that know the variant should not be sending it at all."""
    with pytest.raises(MethodError):
        RepWeights(method="bootstrap", prefix="b", n_reps=10, fay_coef=0.0)


def test_a_foreign_parameter_carrying_a_value_still_names_its_owner():
    with pytest.raises(MethodError) as exc:
        RepWeights(method="bootstrap", prefix="b", n_reps=10, fay_coef=0.5)
    assert "BrrWgts" in str(exc.value)


def test_direct_construction_is_guarded_too():
    """The hand-rolled guard this replaced only covered the factory."""
    with pytest.raises(TypeError):
        BootstrapWgts(prefix="b", n_reps=10, fay_coef=0.5)


def test_scale_is_visible_in_the_repr():
    """Rare and silent is the bad combination for a variance coefficient."""
    assert "scale=0.5" in repr(BootstrapWgts(prefix="b", n_reps=4, scale=0.5))
    assert "(derived)" in repr(JackknifeWgts(prefix="b", n_reps=4, rep_coefs=(0.5,) * 4))
    assert "scale" not in repr(BootstrapWgts(prefix="b", n_reps=4))
