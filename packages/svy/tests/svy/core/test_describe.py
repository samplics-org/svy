# tests/svy/core/test_describe.py
import math

import polars as pl
import pytest

from svy import Design, Sample
from svy.core.enumerations import MeasurementType


def _make_df():
    return pl.DataFrame(
        {
            "wgt": [1.2, 0.8, 1.0, 2.3, 0.5, 1.4],
            "age": [23, 35, 41, 35, None, 28],
            "kids": [0, 1, 2, 0, 3, 1],
            "sex": ["F", "M", "F", "F", "M", "M"],
            # non-missing = ["a","bbb","aaaa","cc","a"] -> n_unique=4, len_min=1, len_max=4
            "status": pl.Series(["a", "bbb", "a", None, "cc", "aaaa"], dtype=pl.Utf8),
            "income": [52000, 45000, 61500, 52000, 48500, None],
            "is_urban": [True, False, True, True, True, False],
            "ts": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 12),
                end=pl.datetime(2024, 1, 6, 12),
                interval="1d",
                eager=True,
            ).alias("ts"),
        }
    )


@pytest.fixture
def sample():
    df = _make_df()
    s = Sample(data=df, design=Design(wgt="wgt"))

    # Optional: mark intended types (metadata already infers; this locks our intent)
    s.set_type("income", MeasurementType.CONTINUOUS)
    s.set_type("age", MeasurementType.DISCRETE)
    s.set_type("kids", MeasurementType.DISCRETE)
    s.set_type("sex", MeasurementType.NOMINAL)
    s.set_type("is_urban", MeasurementType.BOOLEAN)
    s.set_type("status", MeasurementType.STRING)

    return s


def test_describe_unweighted_basic(sample: Sample):
    res = sample.describe(
        columns=["income", "age", "sex", "is_urban", "status", "ts"],
        weighted=False,
        top_k=5,
        drop_nulls=True,
    )

    # basic container checks
    assert hasattr(res, "items")
    assert res.weighted is False
    assert res.drop_nulls is True
    assert isinstance(res.percentiles, tuple)

    # pull out helpers
    by_name = {it.name: it for it in res.items}

    # income (continuous) — unweighted mean over non-missing values
    # non-missing: [52000, 45000, 61500, 52000, 48500] -> mean = 51800
    inc = by_name["income"]
    assert math.isclose(inc.mean, 51800.0, rel_tol=1e-12, abs_tol=1e-8)
    assert inc.n_missing == 1
    assert inc.min == 45000
    assert inc.max == 61500
    assert inc.p50 == 52000

    # age (discrete)
    age = by_name["age"]
    # non-missing: [23,35,41,35,28] -> mean = 32.4
    assert math.isclose(age.mean, 32.4, rel_tol=1e-12, abs_tol=1e-8)
    assert age.n_missing == 1
    assert age.min == 23
    assert age.max == 41
    assert age.p50 == 35

    # categorical — mode + level count check
    sex = by_name["sex"]
    assert sex.n_levels == 2
    # F=3, M=3 → either is acceptable as mode; impl may pick first seen
    assert sex.mode in ("F", "M")

    # boolean — false/true counts and props (unweighted)
    urb = by_name["is_urban"]
    # False=2/6 True=4/6
    assert math.isclose(urb.false.count, 2.0, rel_tol=1e-12)
    assert math.isclose(urb.true.count, 4.0, rel_tol=1e-12)
    assert math.isclose(urb.false.prop, 2 / 6, rel_tol=1e-12)
    assert math.isclose(urb.true.prop, 4 / 6, rel_tol=1e-12)

    # string — only shortest/longest lengths now
    st = by_name["status"]
    # non-missing = ["a", "bbb", "aaaa", "cc", "a"] -> shortest=1, longest=4
    assert st.len_min == 1
    assert st.len_max == 4

    # datetime — min/max
    dti = by_name["ts"]
    assert dti.min is not None and dti.max is not None
    assert str(dti.min).startswith("2024-01-01")
    assert str(dti.max).startswith("2024-01-06")


def test_describe_weighted_mean_and_boolean(sample: Sample):
    res = sample.describe(columns=["income", "is_urban"], weighted=True, top_k=5, drop_nulls=True)

    assert res.weighted is True
    assert res.weight_col == "wgt"

    by_name = {it.name: it for it in res.items}

    # Weighted mean for income (exclude missing)
    # values & weights for non-missing rows:
    # (52000,1.2), (45000,0.8), (61500,1.0), (52000,2.3), (48500,0.5)
    num = 52000 * 1.2 + 45000 * 0.8 + 61500 * 1.0 + 52000 * 2.3 + 48500 * 0.5
    den = 1.2 + 0.8 + 1.0 + 2.3 + 0.5
    w_mean = num / den

    inc = by_name["income"]
    assert math.isclose(inc.mean, w_mean, rel_tol=1e-12, abs_tol=1e-8)

    # Weighted boolean proportions:
    # False weights: rows 1(0.8), 5(1.4) => 2.2
    # True  weights: rows 0(1.2),2(1.0),3(2.3),4(0.5) => 5.0
    tot = 2.2 + 5.0
    urb = by_name["is_urban"]
    assert math.isclose(urb.false.count, 2.2, rel_tol=1e-12)
    assert math.isclose(urb.true.count, 5.0, rel_tol=1e-12)
    assert math.isclose(urb.false.prop, 2.2 / tot, rel_tol=1e-12)
    assert math.isclose(urb.true.prop, 5.0 / tot, rel_tol=1e-12)


def test_describe_drop_nulls_false_includes_missing(sample: Sample):
    # When drop_nulls=False:
    res = sample.describe(columns=["income", "age", "status"], weighted=False, drop_nulls=False)

    by_name = {it.name: it for it in res.items}
    assert by_name["income"].n_missing == 1
    assert by_name["age"].n_missing == 1
    assert by_name["status"].n_missing == 1


def test_describe_string_lengths_and_unique(sample: Sample):
    # Focus on strings; top_k isn't displayed in printing now, but n_unique should be filled
    res = sample.describe(columns=["status"], weighted=False, top_k=3)
    it = res.items[0]
    assert it.name == "status"
    # unique over non-missing: {"a","bbb","aaaa","cc"} → 4
    assert it.n_unique == 4
    assert it.len_min == 1
    assert it.len_max == 4


def test_describe_str_and_repr_dont_crash(sample: Sample):
    res = sample.describe(columns=["income", "sex", "is_urban"], weighted=False, top_k=5)
    s = str(res)
    r = repr(res)
    # sanity checks
    assert "Describe" in s
    assert "Numeric" in s
    assert "Categorical" in s
    assert "Boolean" in s
    assert "DescribeResult(" in r
