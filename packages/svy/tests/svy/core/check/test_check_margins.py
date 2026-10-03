"""rake's margin agreement rule, shared with the internal margin check."""

from __future__ import annotations

import polars as pl
import pytest

from svy import Design, Sample
from svy.core._check import MARGINS_RTOL, margin_totals_check
from svy.errors import WeightingError


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


# 999_999 is exactly the tolerance away from 1_000_000; 999_998.9 is past it.
@pytest.mark.parametrize(("sex_total", "agree"), [(999_999.0, True), (999_998.9, False)])
def test_tolerance_boundary_matches_rake(sex_total, agree):
    r = margin_totals_check({"zone": 1_000_000.0, "sex": sex_total})
    assert r.agree is agree
    assert r.max_rel_diff == pytest.approx((1_000_000 - sex_total) / 1_000_000)
    assert r.rtol == MARGINS_RTOL
    if agree:
        _sample().weighting.rake(controls=_controls(sex_total))
    else:
        with pytest.raises(WeightingError) as exc:
            _sample().weighting.rake(controls=_controls(sex_total))
        assert exc.value.code == "MARGINS_DISAGREE"


def test_single_margin_agrees():
    r = margin_totals_check({"zone": 7.0})
    assert (r.agree, r.max_rel_diff) == (True, 0.0)
