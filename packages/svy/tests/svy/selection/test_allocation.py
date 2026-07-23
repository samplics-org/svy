"""
Allocation helper tests (round 8, SZ11).

Covers population capping with redistribution, missing-SD validation,
and min_n over-allocation handling for proportional / neyman.
"""

import pytest

from svy.selection.allocation import allocate


# =============================================================================
# Neyman — group_sds validation
# =============================================================================


def test_neyman_missing_sds_raises():
    """Missing SD keys silently defaulted to 0.0 (min_n allocation) before."""
    with pytest.raises(ValueError, match="missing entries"):
        allocate(
            {"a": 100, "b": 100},
            method="neyman",
            n_total=50,
            group_sds={"a": 10.0},
        )


def test_neyman_missing_sds_for_empty_group_ok():
    """Empty groups need no SD entry."""
    out = allocate(
        {"a": 100, "b": 0},
        method="neyman",
        n_total=50,
        group_sds={"a": 10.0},
    )
    assert out == {"a": 50, "b": 0}


# =============================================================================
# Population caps (cap_at_population=True is the default)
# =============================================================================


def test_neyman_caps_at_frame_size_and_redistributes():
    """a: N*SD = 10*100 = 1000, b: 1000*1 = 1000 -> raw 50/50, but a has
    only 10 units; surplus flows to b."""
    out = allocate(
        {"a": 10, "b": 1000},
        method="neyman",
        n_total=100,
        group_sds={"a": 100.0, "b": 1.0},
    )
    assert out == {"a": 10, "b": 90}
    assert sum(out.values()) == 100


def test_neyman_uncapped_when_disabled():
    out = allocate(
        {"a": 10, "b": 1000},
        method="neyman",
        n_total=100,
        group_sds={"a": 100.0, "b": 1.0},
        cap_at_population=False,
    )
    assert out["a"] == 50
    assert out["b"] == 50


def test_proportional_min_n_over_allocation_rescales_and_respects_caps():
    """min_n=5 floors (a:5, b:49) exceed n_total=50 -> warn, drop min_n,
    rescale; nothing exceeds its frame count."""
    with pytest.warns(UserWarning, match="min_n"):
        out = allocate(
            {"a": 2, "b": 100},
            method="proportional",
            n_total=50,
            min_n=5,
        )
    assert sum(out.values()) == 50
    assert out["a"] <= 2
    assert out["b"] <= 100


def test_neyman_n_total_exceeding_frame_warns_and_caps():
    with pytest.warns(UserWarning, match="exceeds the total frame"):
        out = allocate(
            {"a": 10, "b": 20},
            method="neyman",
            n_total=100,
            group_sds={"a": 1.0, "b": 1.0},
        )
    assert out == {"a": 10, "b": 20}


# =============================================================================
# min_n over-allocation
# =============================================================================


def test_proportional_min_n_overallocation_warns():
    """Floors (5 groups x min_n=1) exceed n_total=3 -> warn + rescale."""
    sizes = {g: 100 for g in "abcde"}
    with pytest.warns(UserWarning, match="min_n"):
        out = allocate(sizes, method="proportional", n_total=3)
    assert sum(out.values()) == 3


def test_neyman_min_n_overallocation_warns():
    sizes = {g: 100 for g in "abcde"}
    sds = {g: 1.0 for g in "abcde"}
    with pytest.warns(UserWarning, match="min_n"):
        out = allocate(sizes, method="neyman", n_total=3, group_sds=sds)
    assert sum(out.values()) == 3


# =============================================================================
# Exact-sum property preserved
# =============================================================================


@pytest.mark.parametrize("n_total", [7, 50, 111])
def test_proportional_exact_sum(n_total):
    out = allocate({"a": 300, "b": 500, "c": 200}, method="proportional", n_total=n_total)
    assert sum(out.values()) == n_total


@pytest.mark.parametrize("n_total", [7, 50, 111])
def test_neyman_exact_sum(n_total):
    out = allocate(
        {"a": 300, "b": 500, "c": 200},
        method="neyman",
        n_total=n_total,
        group_sds={"a": 5.0, "b": 2.0, "c": 8.0},
    )
    assert sum(out.values()) == n_total
