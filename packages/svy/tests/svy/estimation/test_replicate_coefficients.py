# tests/svy/estimation/test_replicate_coefficients.py
"""Per-replicate variance coefficients now come from the RepWeights variant.

They used to be re-derived inside the kernel from a method label plus a
fay_coef that crossed the FFI boundary for every method. These tests pin the
behaviour that move has to preserve, including the estimation-time Fay
override, which nothing covered before.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.repwgts import BootstrapWgts, BrrWgts, JackknifeWgts, SdrWgts


@pytest.fixture
def brr_sample():
    rng = np.random.default_rng(3)
    data = pl.DataFrame(
        {
            "stratum": [1, 1, 2, 2, 3, 3, 4, 4],
            "psu": [1, 2, 3, 4, 5, 6, 7, 8],
            "wgt": [10.0] * 8,
            "y": rng.normal(100, 10, 8),
        }
    )
    s = svy.Sample(data=data, design=svy.Design(wgt="wgt", stratum="stratum", psu="psu"))
    return s.weighting.create_brr_wgts(fay_coef=0.5)


def _se(sample, **kwargs) -> float:
    return sample.estimation.mean("y", method="replication", **kwargs).to_dicts()[0]["se"]


def test_design_fay_coefficient_is_used_by_default(brr_sample):
    assert brr_sample._design.rep_wgts.fay_coef == 0.5
    assert _se(brr_sample) > 0


def test_estimation_time_fay_overrides_the_design(brr_sample):
    """BRR scale is 1/(B(1-f)^2), so the SE scales as 1/(1-f) exactly."""
    at_design = _se(brr_sample)  # f = 0.5
    at_override = _se(brr_sample, fay_coef=0.25)
    assert at_design / at_override == pytest.approx((1 - 0.25) / (1 - 0.5), rel=1e-12)


def test_fay_coef_zero_means_no_override_not_fay_zero(brr_sample):
    """0.0 is the sentinel for 'unset', so it must not reset the design's 0.5."""
    assert _se(brr_sample, fay_coef=0.0) == pytest.approx(_se(brr_sample), rel=1e-15)


def test_fay_override_is_ignored_for_methods_without_one(brr_sample):
    """A bootstrap has no Fay coefficient; passing one must not change anything."""
    data = brr_sample._data
    boot = svy.Sample(
        data=data,
        design=svy.Design(wgt="wgt", stratum="stratum", psu="psu"),
    ).weighting.create_bs_wgts(n_reps=16, rep_prefix="bs", rstate=5)
    assert _se(boot, fay_coef=0.4) == pytest.approx(_se(boot), rel=1e-15)


@pytest.mark.parametrize(
    "variant, expected",
    [
        (BootstrapWgts(prefix="w", n_reps=64), [1.0 / 64] * 64),
        (SdrWgts(prefix="w", n_reps=80), [4.0 / 80] * 80),
        (JackknifeWgts(prefix="w", n_reps=20), [19.0 / 20] * 20),
        (BrrWgts(prefix="w", n_reps=32), [1.0 / 32] * 32),
        (BrrWgts(prefix="w", n_reps=32, fay_coef=0.3), [1.0 / (32 * 0.7**2)] * 32),
    ],
)
def test_variant_coefficients_match_the_kernel_formulas(variant, expected):
    """Mirrors replicate_coefficients() in svy-rs/src/estimation/replication.rs."""
    assert variant.coefficients() == pytest.approx(expected, rel=1e-15)


def test_user_scale_takes_precedence_over_the_method_default():
    rs = tuple(np.linspace(0.5, 1.5, 10))
    assert BootstrapWgts(prefix="w", n_reps=10, scale=rs).coefficients() == pytest.approx(
        list(rs)
    )


def test_coefficients_follow_the_resolved_replicate_count(brr_sample):
    """The column count wins over a stale n_reps recorded on the design."""
    rw = brr_sample._design.rep_wgts
    assert len(rw.coefficients()) == rw.n_reps


# ==========================================================================
# JKn dead ends announce themselves at construction
# ==========================================================================
#
# The MethodError from coefficients() stays lazy on purpose -- such a Sample is
# still usable for Taylor, wrangling and where=. What was missing was the early
# signal, so the failure surfaced at a call site far from its cause.


def _jkn_frame(counts, rng):
    """One frame per PSUs-per-stratum layout, with R = total PSUs replicates."""
    rows = []
    for h, n_h in enumerate(counts, start=1):
        for p in range(1, n_h + 1):
            rows.extend([(h, f"{h}_{p}")] * 25)
    df = pl.DataFrame(rows, schema=["stratum", "psu"], orient="row")
    n, n_reps = len(df), sum(counts)
    df = df.with_columns(
        wgt=pl.Series(rng.uniform(50, 500, n)),
        y=pl.Series(rng.normal(100, 15, n)),
    )
    for r in range(1, n_reps + 1):
        df = df.with_columns(pl.Series(f"rw{r}", rng.uniform(50, 500, n)))
    return df, n_reps


@pytest.mark.parametrize(
    "design_kwargs, expect_psu_in_hint",
    [
        ({"wgt": "wgt"}, True),
        ({"stratum": "stratum", "wgt": "wgt"}, True),
        # Design carries both, and it still does not count: the units have to be
        # declared on the replicate weights themselves.
        ({"stratum": "stratum", "psu": "psu", "wgt": "wgt"}, True),
    ],
)
def test_jkn_without_declared_units_warns_at_construction(design_kwargs, expect_psu_in_hint):
    df, n_reps = _jkn_frame([2, 2, 2, 2], np.random.default_rng(21))
    s = svy.Sample(
        data=df,
        design=svy.Design(
            rep_wgts=JackknifeWgts(prefix="rw", n_reps=n_reps, kind="jkn"), **design_kwargs
        ),
    )
    warns = s.warnings.list(code="JACKKNIFE_COEFS_UNAVAILABLE")
    assert warns, "a declared JKn that cannot be derived must warn at construction"
    assert ("psu" in warns[0].hint) is expect_psu_in_hint


def test_jkn_unbalanced_strata_warns_and_does_not_offer_psu():
    """Adding psu cannot help here -- it is already there and still not enough."""
    df, n_reps = _jkn_frame([3, 2, 2, 2], np.random.default_rng(22))
    s = svy.Sample(
        data=df,
        design=svy.Design(
            stratum="stratum",
            psu="psu",
            wgt="wgt",
            rep_wgts=JackknifeWgts(
                prefix="rw", n_reps=n_reps, kind="jkn", stratum="stratum", psu="psu"
            ),
        ),
    )
    warns = s.warnings.list(code="JACKKNIFE_COEFS_UNAVAILABLE")
    assert warns
    assert "unbalanced" in warns[0].detail
    assert "name the units" not in warns[0].hint  # already named, and still not enough


def test_derivable_jkn_does_not_warn():
    df, n_reps = _jkn_frame([2, 2, 2, 2], np.random.default_rng(23))
    s = svy.Sample(
        data=df,
        design=svy.Design(
            stratum="stratum",
            psu="psu",
            wgt="wgt",
            rep_wgts=JackknifeWgts(
                prefix="rw", n_reps=n_reps, kind="jkn", stratum="stratum", psu="psu"
            ),
        ),
    )
    assert not s.warnings.list(code="JACKKNIFE_COEFS_UNAVAILABLE")
    assert s._design.rep_wgts.coefficients() == pytest.approx([0.5] * n_reps)


def test_jkn_error_names_both_fixes():
    """`scale` used to be the only remedy named, sending anyone whose file does
    carry psu off to hand-compute what svy would have derived."""
    df, n_reps = _jkn_frame([2, 2, 2, 2], np.random.default_rng(24))
    s = svy.Sample(
        data=df,
        design=svy.Design(
            wgt="wgt", rep_wgts=JackknifeWgts(prefix="rw", n_reps=n_reps, kind="jkn")
        ),
    )
    with pytest.raises(svy.MethodError) as exc:
        s.estimation.mean("y", method="replication")
    text = str(exc.value)
    assert "psu" in text and "scale=" in text
    # The template used to append a second period to a reason that already had
    # one. Checked on the detail line only -- "JackknifeWgts(...)" in the hint
    # legitimately contains a run of dots.
    detail = next(ln for ln in text.splitlines() if "cannot be used here" in ln)
    assert not detail.rstrip().endswith("..")


def test_df_keeps_a_fractional_value():
    """df feeds a t-quantile and Satterthwaite-style df is fractional, so the
    field stores what it is given rather than rounding it to an int."""
    rw = JackknifeWgts(prefix="rw", n_reps=8, kind="jk1", df=6.5)
    assert rw.df == 6.5
    assert isinstance(rw.df, float)


# ==========================================================================
# Recorded units are column references, and are tracked like any other
# ==========================================================================


def _units_sample(rng, *, rw_stratum="vstrat", rw_psu="psu"):
    rows = [(h, f"{h}_{p}", f"v{h}") for h in range(1, 5) for p in (1, 2) for _ in range(25)]
    df = pl.DataFrame(rows, schema=["stratum", "psu", "vstrat"], orient="row")
    n = len(df)
    df = df.with_columns(
        wgt=pl.Series(rng.uniform(50, 500, n)), y=pl.Series(rng.normal(100, 15, n))
    )
    for r in range(1, 9):
        df = df.with_columns(pl.Series(f"rw{r}", rng.uniform(50, 500, n)))
    return svy.Sample(
        data=df,
        design=svy.Design(
            stratum="stratum",
            psu="psu",
            wgt="wgt",
            # kind left unset on purpose: these are about tracking column
            # references, and the units say which scheme it is anyway.
            rep_wgts=JackknifeWgts(prefix="rw", n_reps=8, stratum=rw_stratum, psu=rw_psu),
        ),
    )


def test_renaming_a_unit_column_follows_through_to_the_rep_weights():
    """A rename that touches no replicate column still has to reach these:
    the prefix branch returns early when nothing matched the replicates."""
    s = _units_sample(np.random.default_rng(31))
    r = s.wrangling.rename_columns({"vstrat": "VARSTRAT"})
    assert r._design.rep_wgts.stratum == "VARSTRAT"
    assert r._design.rep_wgts.psu == "psu"


def test_renaming_both_design_and_rep_units_keeps_them_independent():
    s = _units_sample(np.random.default_rng(32), rw_stratum="stratum")
    r = s.wrangling.rename_columns({"stratum": "S2", "psu": "P2"})
    assert (r._design.stratum, r._design.psu) == ("S2", "P2")
    assert (r._design.rep_wgts.stratum, r._design.rep_wgts.psu) == ("S2", "P2")


def test_a_unit_column_is_protected_from_a_casual_drop():
    s = _units_sample(np.random.default_rng(33))
    with pytest.raises(svy.MethodError):
        s.wrangling.remove_columns(["vstrat"])


def test_force_dropping_a_unit_column_clears_the_reference():
    """Better a recorded None than a reference to a column that is gone."""
    s = _units_sample(np.random.default_rng(34))
    d = s.wrangling.remove_columns(["vstrat"], force=True)
    assert d._design.rep_wgts.stratum is None
    assert d._design.rep_wgts.psu == "psu"


def test_declared_units_must_resolve_at_construction():
    rng = np.random.default_rng(35)
    df, n_reps = _jkn_frame([2, 2, 2, 2], rng)
    with pytest.raises(ValueError, match="not found in data"):
        svy.Sample(
            data=df,
            design=svy.Design(
                stratum="stratum",
                psu="psu",
                wgt="wgt",
                rep_wgts=JackknifeWgts(
                    prefix="rw", n_reps=n_reps, kind="jkn", stratum="NOPE", psu="psu"
                ),
            ),
        )


def test_jkn_with_a_psu_but_no_stratum_refuses_rather_than_reproducing_jk1():
    """(n_h-1)/n_h is per-stratum. With no stratum named, counting every PSU as
    one stratum yields (R-1)/R -- the JK1 global -- handed back under a JKn
    label. On 4 strata x 2 PSUs that is 0.875 where 0.5 is correct."""
    df, n_reps = _jkn_frame([2, 2, 2, 2], np.random.default_rng(41))
    s = svy.Sample(
        data=df,
        design=svy.Design(
            stratum="stratum",
            psu="psu",
            wgt="wgt",
            rep_wgts=JackknifeWgts(prefix="rw", n_reps=n_reps, kind="jkn", psu="psu"),
        ),
    )
    assert s._design.rep_wgts.rep_coefs is None
    warns = s.warnings.list(code="JACKKNIFE_COEFS_UNAVAILABLE")
    assert warns and "no stratum" in warns[0].detail
    with pytest.raises(svy.MethodError):
        s._design.rep_wgts.coefficients()


def test_naming_the_stratum_derives_the_right_coefficient():
    """The same weights, with the unit named, get (n_h-1)/n_h = 0.5."""
    df, n_reps = _jkn_frame([2, 2, 2, 2], np.random.default_rng(41))
    s = svy.Sample(
        data=df,
        design=svy.Design(
            stratum="stratum",
            psu="psu",
            wgt="wgt",
            rep_wgts=JackknifeWgts(
                prefix="rw", n_reps=n_reps, kind="jkn", stratum="stratum", psu="psu"
            ),
        ),
    )
    assert s._design.rep_wgts.coefficients() == pytest.approx([0.5] * n_reps)


# ==========================================================================
# Units take the same shapes Design's do
# ==========================================================================


def _multi_unit_frame(rng):
    rows = [
        (r, u, f"{r}{u}_{p}")
        for r in ("N", "S")
        for u in ("urban", "rural")
        for p in (1, 2)
        for _ in range(25)
    ]
    df = pl.DataFrame(rows, schema=["region", "urban", "psu"], orient="row")
    n = len(df)
    df = df.with_columns(
        wgt=pl.Series(rng.uniform(50, 500, n)), y=pl.Series(rng.normal(100, 15, n))
    )
    for r in range(1, 9):
        df = df.with_columns(pl.Series(f"jk{r}", rng.uniform(50, 500, n)))
    return df


def _multi_unit_sample(rng, stratum=("region", "urban")):
    df = _multi_unit_frame(rng)
    return svy.Sample(
        data=df,
        design=svy.Design(
            stratum=("region", "urban"),
            psu="psu",
            wgt="wgt",
            rep_wgts=JackknifeWgts(prefix="jk", n_reps=8, kind="jkn", stratum=stratum, psu="psu"),
        ),
    )


def test_a_tuple_unit_derives_the_jkn_coefficient():
    """4 strata (region x urban) x 2 PSUs -> (n_h-1)/n_h = 0.5. Grouping happens
    on the source columns directly, not through Design's internal concat."""
    s = _multi_unit_sample(np.random.default_rng(51))
    assert s._design.rep_wgts.stratum == ("region", "urban")
    assert s._design.rep_wgts.coefficients() == pytest.approx([0.5] * 8)
    assert not s.warnings.list(code="JACKKNIFE_COEFS_UNAVAILABLE")


def test_a_one_element_sequence_stays_a_tuple():
    """("region",) and "region" resolve to the same column; collapsing one into
    the other would make a round trip lossy."""
    rw = JackknifeWgts(prefix="jk", n_reps=8, stratum=["region"], psu="psu")
    assert rw.stratum == ("region",)


def test_renaming_remaps_every_member_of_a_tuple_unit():
    s = _multi_unit_sample(np.random.default_rng(52))
    r = s.wrangling.rename_columns({"urban": "URB"})
    assert r._design.rep_wgts.stratum == ("region", "URB")
    assert r._design.stratum == ("region", "URB")


def test_force_dropping_one_member_coarsens_rather_than_clears():
    """A multi-column unit that loses a member is a coarser unit, not a missing
    one; only an empty remainder clears the field."""
    s = _multi_unit_sample(np.random.default_rng(53))
    d = s.wrangling.remove_columns(["urban"], force=True)
    assert d._design.rep_wgts.stratum == ("region",)


def test_every_member_of_a_tuple_unit_must_resolve():
    rng = np.random.default_rng(54)
    with pytest.raises(ValueError, match="not found in data"):
        svy.Sample(
            data=_multi_unit_frame(rng),
            design=svy.Design(
                psu="psu",
                wgt="wgt",
                rep_wgts=JackknifeWgts(
                    prefix="jk", n_reps=8, kind="jkn", stratum=("region", "NOPE"), psu="psu"
                ),
            ),
        )


@pytest.mark.parametrize("bad", [5, [], ("region", 5), ("region", "")])
def test_malformed_units_are_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        JackknifeWgts(prefix="jk", n_reps=8, stratum=bad)
