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


# =============================================================================
# power= and method="size"
# =============================================================================


def test_proportional_power_one_is_unchanged():
    sizes = {"a": 120, "b": 300, "c": 580}
    assert allocate(sizes, n_total=100) == allocate(sizes, n_total=100, power=1.0)
    assert allocate(sizes, n_total=100) == {"a": 12, "b": 30, "c": 58}


def test_proportional_square_root():
    out = allocate({"a": 100, "b": 400, "c": 900}, n_total=60, power=0.5)
    assert out == {"a": 10, "b": 20, "c": 30}


def test_proportional_power_zero_is_equal_split():
    out = allocate({"a": 100, "b": 400, "c": 900, "d": 0}, n_total=60, power=0.0)
    assert out == {"a": 20, "b": 20, "c": 20, "d": 0}


@pytest.mark.parametrize("n_total", [7, 50, 333])
def test_power_and_size_exact_sum(n_total):
    sizes = {"a": 1000, "b": 2000, "c": 3000, "d": 4000}
    mos = {"a": 17.0, "b": 230.5, "c": 99.0, "d": 1000.0}
    assert sum(allocate(sizes, n_total=n_total, power=0.5).values()) == n_total
    out = allocate(sizes, method="size", n_total=n_total, group_mos=mos, power=0.7)
    assert sum(out.values()) == n_total


def test_size_proportional_to_mos_totals():
    out = allocate(
        {"a": 50, "b": 50}, method="size", n_total=40, group_mos={"a": 1000.0, "b": 3000.0}
    )
    assert out == {"a": 10, "b": 30}


def test_size_with_power():
    out = allocate(
        {"a": 50, "b": 50, "c": 50},
        method="size",
        n_total=60,
        group_mos={"a": 100.0, "b": 400.0, "c": 900.0},
        power=0.5,
    )
    assert out == {"a": 10, "b": 20, "c": 30}


def test_size_caps_at_frame_size_and_redistributes():
    out = allocate(
        {"a": 5, "b": 100}, method="size", n_total=40, group_mos={"a": 500.0, "b": 500.0}
    )
    assert out == {"a": 5, "b": 35}


def test_size_zero_mos_group_gets_none():
    """No unit of a group without size can be drawn by PPS, so min_n does not apply."""
    out = allocate(
        {"a": 10, "b": 10, "c": 10},
        method="size",
        n_total=10,
        group_mos={"a": 0.0, "b": 10.0, "c": 40.0},
        min_n=2,
    )
    assert out == {"a": 0, "b": 2, "c": 8}


def test_size_caps_at_units_in_groups_with_size():
    from svy.errors import SvyUserWarning

    with pytest.warns(SvyUserWarning, match="positive size total"):
        out = allocate(
            {"a": 50, "b": 5}, method="size", n_total=20, group_mos={"a": 0.0, "b": 1.0}
        )
    assert out == {"a": 0, "b": 5}


def test_size_missing_mos_raises_but_empty_group_ok():
    with pytest.raises(ValueError, match="missing entries"):
        allocate({"a": 10, "b": 10}, method="size", n_total=5, group_mos={"a": 1.0})
    out = allocate({"a": 10, "b": 0}, method="size", n_total=5, group_mos={"a": 1.0})
    assert out == {"a": 5, "b": 0}


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_size_rejects_invalid_mos(bad):
    with pytest.raises(ValueError, match="finite and >= 0"):
        allocate({"a": 10, "b": 10}, method="size", n_total=5, group_mos={"a": 1.0, "b": bad})


def test_size_all_zero_mos_raises():
    with pytest.raises(ValueError, match="every group's size total is zero"):
        allocate({"a": 10, "b": 10}, method="size", n_total=5, group_mos={"a": 0.0, "b": 0.0})


def test_size_requires_group_mos_and_n_total():
    with pytest.raises(ValueError, match="requires group_mos="):
        allocate({"a": 10}, method="size", n_total=5)
    with pytest.raises(ValueError, match="requires n_total="):
        allocate({"a": 10}, method="size", group_mos={"a": 1.0})


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_invalid_power_raises(bad):
    with pytest.raises(ValueError, match="power must be"):
        allocate({"a": 10, "b": 20}, n_total=5, power=bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "neyman", "n_total": 5, "group_sds": {"a": 1.0}},
        {"method": "equal", "n_per_group": 2},
        {"method": "rate", "rate": 0.1},
    ],
)
def test_power_only_for_proportional_and_size(kwargs):
    with pytest.raises(ValueError, match="does not take power="):
        allocate({"a": 10}, power=0.5, **kwargs)


def test_group_mos_only_for_size():
    with pytest.raises(ValueError, match="does not take group_mos="):
        allocate({"a": 10}, n_total=5, group_mos={"a": 1.0})


def test_unknown_method_lists_size():
    with pytest.raises(ValueError, match="'size'"):
        allocate({"a": 10}, method="pps", n_total=5)  # type: ignore[arg-type]


# =============================================================================
# Sample facade: group_totals() and allocate(method="size")
# =============================================================================


def _frame():
    import polars as pl

    return pl.DataFrame(
        {
            "reg": ["N", "N", "S", "S", "S", "W"],
            "urb": [1, 2, 1, 1, 2, 1],
            "hh": [10.0, 30.0, 5.0, None, -2.0, 50.0],
            "id": list(range(6)),
        }
    )


def test_group_totals_by_stratum_and_by():
    import svy

    s = svy.Sample(_frame(), svy.Design(stratum="reg", mos="hh"))
    assert s.sampling.group_totals() == {"N": 40.0, "S": 5.0, "W": 50.0}
    totals = s.sampling.group_totals(by="urb")
    assert totals.keys() == s.sampling.group_sizes(by="urb").keys()
    assert totals == {
        "N__by__1": 10.0,
        "N__by__2": 30.0,
        "S__by__1": 5.0,
        "S__by__2": 0.0,
        "W__by__1": 50.0,
    }


def test_group_totals_explicit_mos_without_design_mos():
    import svy

    s = svy.Sample(_frame(), svy.Design())
    assert s.sampling.group_totals("hh", by="reg") == {"N": 40.0, "S": 5.0, "W": 50.0}
    assert s.sampling.group_totals("hh") == {"__all__": 95.0}


def test_group_totals_errors():
    import svy

    from svy.errors import MethodError

    s = svy.Sample(_frame(), svy.Design(stratum="reg"))
    with pytest.raises(MethodError, match="measure of size") as e:
        s.sampling.group_totals()
    assert e.value.code == "MOS_MISSING"
    with pytest.raises(MethodError) as e:
        s.sampling.group_totals("nope")
    assert e.value.code == "MOS_MISSING"
    with pytest.raises(MethodError) as e:
        s.sampling.group_totals("reg")
    assert e.value.code == "MOS_NOT_NUMERIC"


def test_size_allocation_feeds_pps_selection():
    import polars as pl

    import svy

    n = 300
    df = pl.DataFrame(
        {
            "reg": ["N"] * 100 + ["S"] * 200,
            "hh": [30.0] * 100 + [5.0] * 200,
            "id": list(range(n)),
        }
    )
    s = svy.Sample(df, svy.Design(stratum="reg", mos="hh"))
    sizes = s.sampling.group_sizes()
    mos = s.sampling.group_totals()
    n_map = s.sampling.allocate(sizes, method="size", n_total=20, group_mos=mos)
    assert n_map == {"N": 15, "S": 5}
    drawn = s.sampling.pps_sys(n_map, rstate=1)
    assert drawn.data.group_by("reg").len().sort("reg")["len"].to_list() == [15, 5]
