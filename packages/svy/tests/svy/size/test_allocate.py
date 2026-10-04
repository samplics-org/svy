"""SampleSize.allocate: splitting an overall n across strata.

Moved from tests/svy/selection/test_allocation.py (round 8, SZ11; power from
#245) when allocation left sample.sampling: same inputs and expected values.
"""

import pytest

import svy

from svy.errors import MethodError


def alloc(n, pop_size, **kw):
    return svy.SampleSize().allocate(n, pop_size=pop_size, **kw).n


# =============================================================================
# Neyman — sigma validation
# =============================================================================


def test_neyman_missing_sigma_raises():
    with pytest.raises(MethodError, match="sigma"):
        alloc(50, {"a": 100, "b": 100}, method="neyman", sigma={"a": 10.0})


def test_neyman_missing_sigma_for_empty_stratum_ok():
    out = alloc(50, {"a": 100, "b": 0}, method="neyman", sigma={"a": 10.0})
    assert out == {"a": 50, "b": 0}


def test_neyman_requires_sigma():
    with pytest.raises(MethodError, match="standard deviation"):
        alloc(50, {"a": 100, "b": 100}, method="neyman")


def test_neyman_scalar_sigma_is_proportional():
    assert alloc(30, {"a": 100, "b": 200}, method="neyman", sigma=2.0) == {"a": 10, "b": 20}


# =============================================================================
# Population caps (cap_at_population=True is the default)
# =============================================================================


def test_neyman_caps_at_frame_size_and_redistributes():
    """a: N*SD = 10*100 = 1000, b: 1000*1 = 1000 -> raw 50/50, but a has
    only 10 units; surplus flows to b."""
    out = alloc(100, {"a": 10, "b": 1000}, method="neyman", sigma={"a": 100.0, "b": 1.0})
    assert out == {"a": 10, "b": 90}
    assert sum(out.values()) == 100


def test_neyman_uncapped_when_disabled():
    out = alloc(
        100,
        {"a": 10, "b": 1000},
        method="neyman",
        sigma={"a": 100.0, "b": 1.0},
        cap_at_population=False,
    )
    assert out["a"] == 50
    assert out["b"] == 50


def test_proportional_min_n_over_allocation_rescales_and_respects_caps():
    """min_n=5 floors (a:5, b:49) exceed n=50 -> warn, drop min_n,
    rescale; nothing exceeds its frame count."""
    with pytest.warns(UserWarning, match="min_n"):
        out = alloc(50, {"a": 2, "b": 100}, min_n=5)
    assert sum(out.values()) == 50
    assert out["a"] <= 2
    assert out["b"] <= 100


def test_neyman_n_exceeding_frame_warns_and_caps():
    with pytest.warns(UserWarning, match="exceeds the total frame"):
        out = alloc(100, {"a": 10, "b": 20}, method="neyman", sigma={"a": 1.0, "b": 1.0})
    assert out == {"a": 10, "b": 20}


# =============================================================================
# min_n over-allocation
# =============================================================================


def test_proportional_min_n_overallocation_warns():
    """Floors (5 strata x min_n=1) exceed n=3 -> warn + rescale."""
    sizes = {g: 100 for g in "abcde"}
    with pytest.warns(UserWarning, match="min_n"):
        out = alloc(3, sizes)
    assert sum(out.values()) == 3


def test_neyman_min_n_overallocation_warns():
    sizes = {g: 100 for g in "abcde"}
    sds = {g: 1.0 for g in "abcde"}
    with pytest.warns(UserWarning, match="min_n"):
        out = alloc(3, sizes, method="neyman", sigma=sds)
    assert sum(out.values()) == 3


# =============================================================================
# Exact-sum property preserved
# =============================================================================


@pytest.mark.parametrize("n", [7, 50, 111])
def test_proportional_exact_sum(n):
    assert sum(alloc(n, {"a": 300, "b": 500, "c": 200}).values()) == n


@pytest.mark.parametrize("n", [7, 50, 111])
def test_neyman_exact_sum(n):
    out = alloc(
        n, {"a": 300, "b": 500, "c": 200}, method="neyman", sigma={"a": 5.0, "b": 2.0, "c": 8.0}
    )
    assert sum(out.values()) == n


@pytest.mark.parametrize("n", [7, 50, 333])
def test_power_exact_sum(n):
    sizes = {"a": 1000, "b": 2000, "c": 3000, "d": 4000}
    assert sum(alloc(n, sizes, power=0.5).values()) == n


# =============================================================================
# power=
# =============================================================================


def test_proportional_power_one_is_unchanged():
    sizes = {"a": 120, "b": 300, "c": 580}
    assert alloc(100, sizes) == alloc(100, sizes, power=1.0) == {"a": 12, "b": 30, "c": 58}


def test_proportional_square_root():
    assert alloc(60, {"a": 100, "b": 400, "c": 900}, power=0.5) == {"a": 10, "b": 20, "c": 30}


def test_proportional_power_zero_is_equal_split():
    out = alloc(60, {"a": 100, "b": 400, "c": 900, "d": 0}, power=0.0)
    assert out == {"a": 20, "b": 20, "c": 20, "d": 0}


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_invalid_power_raises(bad):
    with pytest.raises(MethodError, match="power must be"):
        alloc(5, {"a": 10, "b": 20}, power=bad)


@pytest.mark.parametrize("kwargs", [{"method": "neyman", "sigma": 1.0}, {"method": "equal"}])
def test_power_only_for_proportional(kwargs):
    with pytest.raises(MethodError, match="power="):
        alloc(5, {"a": 10}, power=0.5, **kwargs)


# =============================================================================
# Size totals as pop_size (the PPS case)
# =============================================================================


def test_size_totals_drive_the_split():
    households = {"a": 1000.0, "b": 3000.0}
    assert alloc(40, households) == {"a": 10, "b": 30}
    assert alloc(60, {"a": 100.0, "b": 400.0, "c": 900.0}, power=0.5) == {
        "a": 10,
        "b": 20,
        "c": 30,
    }


# =============================================================================
# equal
# =============================================================================


def test_equal_splits_n():
    assert alloc(30, {"a": 100, "b": 200}, method="equal") == {"a": 15, "b": 15}
    assert sum(alloc(31, {"a": 100, "b": 200}, method="equal").values()) == 31


def test_equal_caps_and_redistributes():
    assert alloc(20, {"a": 3, "b": 50}, method="equal") == {"a": 3, "b": 17}


# =============================================================================
# Inputs, keys and the result
# =============================================================================


@pytest.mark.parametrize(
    ("n", "pop_size", "kwargs"),
    [
        (0, {"a": 1}, {}),
        (2.5, {"a": 1}, {}),
        (True, {"a": 1}, {}),
        (5, {}, {}),
        (5, 100, {}),
        (5, {"a": -1}, {}),
        (5, {"a": float("nan")}, {}),
        (5, {"a": "x"}, {}),
        (5, {"a": 10}, {"method": "rate"}),
        (5, {"a": 10}, {"sigma": 1.0}),
        (5, {"a": 10, "b": 10}, {"method": "neyman", "sigma": {"a": 1.0, "b": 1.0, "c": 1.0}}),
    ],
)
def test_bad_input_raises(n, pop_size, kwargs):
    with pytest.raises(MethodError):
        svy.SampleSize().allocate(n, pop_size=pop_size, **kwargs)


def test_tuple_strata_and_result():
    pop = {("N", "u"): 100, ("N", "r"): 200, ("S", "u"): 50}
    ss = svy.SampleSize().allocate(35, pop_size=pop)
    assert ss.n == {("N", "u"): 10, ("N", "r"): 20, ("S", "u"): 5}
    assert [a.stratum for a in ss.allocation] == list(pop)
    df = ss.to_polars()
    assert df.columns == ["stratum", "pop_size", "n"]
    assert df["stratum"].to_list() == ["N, u", "N, r", "S, u"]
    assert repr(ss) == "SampleSize(allocation, strata=3, n=35)"
    assert "pop_size" in str(ss)


def test_neyman_result_shows_sigma():
    ss = svy.SampleSize().allocate(30, pop_size={"a": 100, "b": 200}, method="neyman", sigma=2.0)
    assert ss.to_polars().columns == ["stratum", "pop_size", "sigma", "n"]


def test_another_goal_replaces_the_allocation():
    ss = svy.SampleSize().allocate(30, pop_size={"a": 100, "b": 200})
    ss.estimate_prop(p=0.5, moe=0.05)
    assert ss.allocation is None
    assert isinstance(ss.n, float)


def test_allocation_feeds_selection():
    import polars as pl

    df = pl.DataFrame(
        {"reg": ["N"] * 100 + ["S"] * 200, "urb": ["u", "r"] * 150, "id": list(range(300))}
    )
    frame = svy.Sample(df, svy.Design(stratum=["reg", "urb"]))
    n = (
        svy.SampleSize()
        .allocate(30, pop_size={("N", "u"): 50, ("N", "r"): 50, ("S", "u"): 100, ("S", "r"): 100})
        .n
    )
    drawn = frame.sampling.srs(n, rstate=1)
    counts = drawn.data.group_by(["reg", "urb"]).len().sort(["reg", "urb"])["len"].to_list()
    assert counts == [5, 5, 10, 10]
