from __future__ import annotations

import msgspec
import polars as pl
import pytest

from svy import Design, Sample
from svy.checks import MarginCheck, check_margins
from svy.errors import MethodError, WeightingError


def _sample() -> Sample:
    df = pl.DataFrame(
        {
            "zone": [1, 1, 2, 2, 3, 3, 1, 2],
            "sex": ["M", "F"] * 4,
            "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    return Sample(df, Design(wgt="w"))


def _controls(sex_total: float) -> dict:
    return {
        "zone": {1: 400_000.0, 2: 350_000.0, 3: 250_000.0},
        "sex": {"M": sex_total / 2, "F": sex_total / 2},
    }


def test_agreeing_margins():
    r = check_margins(_controls(1_000_000.0))
    assert r == MarginCheck(
        totals={"zone": 1_000_000.0, "sex": 1_000_000.0}, agree=True, max_rel_diff=0.0, rtol=1e-6
    )


def test_disagreeing_margins():
    r = check_margins({"zone": {1: 3, 2: 3}, "sex": {"M": 3, "F": 4}})
    assert r.totals == {"zone": 6.0, "sex": 7.0}
    assert not r.agree
    assert r.max_rel_diff == pytest.approx(1 / 7)


def test_single_margin_agrees():
    r = check_margins({"zone": {1: 3.0, 2: 4.0}})
    assert (r.totals, r.agree, r.max_rel_diff) == ({"zone": 7.0}, True, 0.0)


# 999_999 is exactly the tolerance away from 1_000_000; 999_998.9 is past it.
@pytest.mark.parametrize(("sex_total", "agree"), [(999_999.0, True), (999_998.9, False)])
def test_tolerance_boundary_matches_rake(sex_total, agree):
    controls = _controls(sex_total)
    r = check_margins(controls)
    assert r.agree is agree
    assert r.max_rel_diff == pytest.approx((1_000_000 - sex_total) / 1_000_000)
    if agree:
        _sample().weighting.rake(controls=controls)
    else:
        with pytest.raises(WeightingError) as exc:
            _sample().weighting.rake(controls=controls)
        assert exc.value.code == "MARGINS_DISAGREE"


def test_rtol():
    controls = {"a": {1: 100.0}, "b": {1: 99.0}}
    assert not check_margins(controls).agree
    assert check_margins(controls, rtol=0.01).agree
    assert not check_margins(controls, rtol=0.0099).agree
    assert check_margins(controls, rtol=0.01).rtol == 0.01


def test_bad_input():
    with pytest.raises(WeightingError) as exc:
        check_margins([("zone", {1: 3})])
    assert exc.value.code == "CONTROLS_TYPE_INVALID"
    with pytest.raises(WeightingError) as exc:
        check_margins({"zone": {}})
    assert exc.value.code == "CONTROLS_TYPE_INVALID"
    with pytest.raises(WeightingError) as exc:
        check_margins({"zone": {1: float("nan"), 2: "x"}})
    assert exc.value.code == "CONTROLS_VALUE_INVALID"
    for rtol in (-1e-6, "0.1", True, float("nan")):
        with pytest.raises(MethodError):
            check_margins({"zone": {1: 3}}, rtol=rtol)


def test_report_serializes_and_prints():
    r = check_margins({"zone": {1: 3, 2: 3}, "sex": {"M": 3, "F": 4}})
    assert msgspec.json.decode(msgspec.json.encode(r), type=MarginCheck) == r
    text = repr(r)
    assert text.splitlines()[0] == "Margin check"
    assert "Total zone" in text and "Agree" in text and ": no" in text
