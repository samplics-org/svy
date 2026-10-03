"""sample.sampling.allocate counts the frame itself; custom counts only for a custom allocation."""

import polars as pl
import pytest

import svy

from svy.errors import MethodError


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "reg": ["N", "N", "S", "S", "S", "W"],
            "urb": [1, 2, 1, 1, 2, 1],
            "hh": [10.0, 30.0, 5.0, None, -2.0, 50.0],
            "id": list(range(6)),
        }
    )


def _big() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "reg": ["N"] * 100 + ["S"] * 200,
            "hh": [30.0] * 100 + [5.0] * 200,
            "id": list(range(300)),
        }
    )


def test_counts_the_frame_by_stratum():
    s = svy.Sample(_big(), svy.Design(stratum="reg"))
    assert s.sampling.allocate(n_total=30) == {"N": 10, "S": 20}


def test_counts_the_frame_by_stratum_and_by():
    s = svy.Sample(_frame(), svy.Design(stratum="reg"))
    out = s.sampling.allocate(method="equal", n_per_group=5, by="urb")
    assert out == {"N__by__1": 1, "N__by__2": 1, "S__by__1": 2, "S__by__2": 1, "W__by__1": 1}


def test_by_without_strata():
    s = svy.Sample(_big())
    assert s.sampling.allocate(n_total=30, by="reg") == {"N": 10, "S": 20}


def test_size_sums_the_design_mos():
    s = svy.Sample(_big(), svy.Design(stratum="reg", mos="hh"))
    # N: 100 * 30 = 3000, S: 200 * 5 = 1000
    assert s.sampling.allocate(method="size", n_total=20) == {"N": 15, "S": 5}


def test_size_with_an_explicit_mos_and_power():
    s = svy.Sample(_big().with_columns(m=pl.lit(1.0)), svy.Design(stratum="reg", mos="hh"))
    # m sums to the counts, 100 and 200; square root gives 10 : 14.14
    out = s.sampling.allocate(method="size", n_total=24, mos="m", power=0.5)
    assert out == {"N": 10, "S": 14}


def test_size_ignores_missing_and_non_positive_mos():
    s = svy.Sample(_frame(), svy.Design(stratum="reg", mos="hh"))
    # N: 40, S: 5 (null and -2 add nothing), W: 50
    out = s.sampling.allocate(method="size", n_total=19, min_n=0, cap_at_population=False)
    assert out == {"N": 8, "S": 1, "W": 10}


def test_size_allocation_feeds_pps_selection():
    s = svy.Sample(_big(), svy.Design(stratum="reg", mos="hh"))
    n_map = s.sampling.allocate(method="size", n_total=20)
    drawn = s.sampling.pps_sys(n_map, rstate=1)
    assert drawn.data.group_by("reg").len().sort("reg")["len"].to_list() == [15, 5]


def test_custom_counts():
    s = svy.Sample(_big(), svy.Design(stratum="reg"))
    out = s.sampling.allocate({"N": 1000, "S": 3000}, n_total=40)
    assert out == {"N": 10, "S": 30}
    out = s.sampling.allocate(
        {"N": 50, "S": 50}, method="size", n_total=40, group_mos={"N": 1.0, "S": 3.0}
    )
    assert out == {"N": 10, "S": 30}


def test_custom_counts_and_by_refused():
    s = svy.Sample(_big())
    with pytest.raises(MethodError) as e:
        s.sampling.allocate({"N": 10}, n_total=5, by="reg")
    assert e.value.code == "ALLOCATE_SIZES_AND_BY"


def test_custom_counts_with_size_need_custom_totals():
    s = svy.Sample(_big(), svy.Design(stratum="reg", mos="hh"))
    with pytest.raises(MethodError) as e:
        s.sampling.allocate({"N": 10, "S": 10}, method="size", n_total=5)
    assert e.value.code == "ALLOCATE_MOS_MISSING"


def test_mos_only_with_size():
    s = svy.Sample(_big(), svy.Design(stratum="reg"))
    with pytest.raises(MethodError) as e:
        s.sampling.allocate(n_total=5, mos="hh")
    assert e.value.code == "ALLOCATE_MOS_UNUSED"


def test_size_mos_errors():
    s = svy.Sample(_frame(), svy.Design(stratum="reg"))
    with pytest.raises(MethodError, match="measure of size") as e:
        s.sampling.allocate(method="size", n_total=5)
    assert e.value.code == "MOS_MISSING"
    with pytest.raises(MethodError) as e:
        s.sampling.allocate(method="size", n_total=5, mos="nope")
    assert e.value.code == "MOS_MISSING"
    with pytest.raises(MethodError) as e:
        s.sampling.allocate(method="size", n_total=5, mos="reg")
    assert e.value.code == "MOS_NOT_NUMERIC"


def test_group_sizes_is_deprecated_and_still_works():
    s = svy.Sample(_big(), svy.Design(stratum="reg"))
    with pytest.warns(DeprecationWarning, match="allocate"):
        sizes = s.sampling.group_sizes()
    assert sizes == {"N": 100, "S": 200}


def test_group_totals_is_gone():
    s = svy.Sample(_big(), svy.Design(stratum="reg", mos="hh"))
    assert not hasattr(s.sampling, "group_totals")
