# tests/svy/core/test_on_parameters.py
"""Every ``on_*`` parameter means the same.

"error" raises and leaves the sample as it was; "warn" records the finding and
raises it once; "ignore" records it at INFO without raising.
"""

from __future__ import annotations

import inspect
import json
import re
import warnings

import polars as pl
import pytest

import svy

from svy import SvyUserWarning, col
from svy.core.functions import combine_samples
from svy.core.warnings import Severity
from svy.errors import MethodError
from svy.wrangling.base import Wrangling


def _svy(rec) -> list:
    return [r for r in rec if issubclass(r.category, SvyUserWarning)]


def _persons() -> svy.Sample:
    df = pl.DataFrame(
        {
            "hh": [3, 1, 1, 2, 2, 2, 4],
            "stratum": ["a", "a", "a", "b", "b", "b", "b"],
            "psu": [1, 1, 5, 2, 2, 6, 3],
            "wgt": [2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 4.0],
        }
    )
    return svy.Sample(df, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))


def _households() -> svy.Sample:
    df = pl.DataFrame({"hh_id": [1, 2, 3], "rooms": [2, 5, 3]})
    return svy.Sample(df, svy.Design())


def _cycle(scale: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strat": [1, 1, 2, 2, 1, 2],
            "psu": [1, 2, 1, 2, 3, 3],
            "w": [x * scale for x in [10.0, 12.0, 8.0, 9.0, 11.0, 7.0]],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _mixed() -> list[svy.Sample]:
    s1 = svy.Sample(_cycle(), svy.Design(stratum="strat", psu="psu", wgt="w"))
    s2 = svy.Sample(_cycle(2.0).drop("psu"), svy.Design(stratum="strat", wgt="w"))
    return [s1, s2]


# name -> (build, call(sample, mode, inplace), finding code, error code)
CASES = {
    "join.on_unmatched": (
        _persons,
        lambda s, mode, inplace: s.wrangling.join(
            _households(), on={"hh": "hh_id"}, on_unmatched=mode, inplace=inplace
        ),
        "JOIN_UNMATCHED",
        "JOIN_UNMATCHED",
    ),
    "filter_records.on_singletons": (
        _persons,
        lambda s, mode, inplace: s.wrangling.filter_records(
            col("psu") != 5, check_singletons=True, on_singletons=mode, inplace=inplace
        ),
        "SINGLETONS_DETECTED",
        "SINGLETONS_AFTER_FILTER",
    ),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("inplace", [False, True])
def test_ignore_records_info_without_raising(name, inplace):
    build, call, code, _ = CASES[name]
    s = build()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = call(s, "ignore", inplace)
    assert not [r for r in _svy(rec) if f"[{code}]" in str(r.message)]
    found = out.warnings.list(code=code)
    assert len(found) == 1 and found[0].level == Severity.INFO
    json.dumps(found[0].to_dict())
    assert (out is s) is inplace


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("inplace", [False, True])
def test_warn_records_and_raises_once(name, inplace):
    build, call, code, _ = CASES[name]
    s = build()
    with pytest.warns(SvyUserWarning, match=rf"\[{code}\]") as rec:
        out = call(s, "warn", inplace)
    assert len([r for r in _svy(rec) if f"[{code}]" in str(r.message)]) == 1
    found = out.warnings.list(code=code)
    assert len(found) == 1 and found[0].level == Severity.WARNING


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("inplace", [False, True])
def test_error_raises_and_leaves_sample(name, inplace):
    build, call, code, err_code = CASES[name]
    s = build()
    data, design = s.data, s.design
    with pytest.raises(MethodError) as ei:
        call(s, "error", inplace)
    assert ei.value.code == err_code
    assert s.data.equals(data) and s.design == design
    assert not s.warnings.list(code=code)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("value", ["raise", "Warn", None, False])
def test_bad_value_refused(name, value):
    build, call, _, _ = CASES[name]
    with pytest.raises(MethodError) as ei:
        call(build(), value, False)
    assert ei.value.code == "INVALID_CHOICE"
    assert ei.value.expected == ["error", "warn", "ignore"]


def test_join_all_matched_records_nothing_in_any_mode():
    s = _persons().wrangling.filter_records(col("hh") != 4)
    for mode in ("error", "warn", "ignore"):
        out = s.wrangling.join(_households(), on={"hh": "hh_id"}, on_unmatched=mode)
        assert not out.warnings.list(code="JOIN_UNMATCHED")


def test_join_default_still_warns():
    with pytest.warns(SvyUserWarning, match=r"\[JOIN_UNMATCHED\]"):
        out = _persons().wrangling.join(_households(), on={"hh": "hh_id"})
    assert out.warnings.list(code="JOIN_UNMATCHED")[0].level == Severity.WARNING


def test_filter_records_default_checks_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _persons().wrangling.filter_records(col("psu") != 5)
    assert not out.warnings.list(code="SINGLETONS_DETECTED")


def test_filter_records_no_singletons_records_nothing():
    for mode in ("error", "warn", "ignore"):
        out = _persons().wrangling.filter_records(
            col("hh") > 0, check_singletons=True, on_singletons=mode
        )
        assert not out.warnings.list(code="SINGLETONS_DETECTED")


def test_filter_records_inplace_warn_filters_the_sample():
    s = _persons()
    with pytest.warns(SvyUserWarning):
        out = s.wrangling.filter_records(
            col("psu") != 5, check_singletons=True, on_singletons="warn", inplace=True
        )
    assert out is s and s.data.height == 6


# ---------------------------------------------------------------------------
# combine_samples(on_mixed_design=)
# ---------------------------------------------------------------------------


def test_combine_mixed_ignore_records_info():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c = combine_samples(_mixed(), on_mixed_design="ignore")
    assert not [r for r in _svy(rec) if "COMBINE_MIXED_DESIGN" in str(r.message)]
    found = c.warnings.list(code="COMBINE_MIXED_DESIGN")
    assert len(found) == 1 and found[0].level == Severity.INFO


# The mixed waves lack a design column, so it is null-filled: a side effect here.
@pytest.mark.filterwarnings(r"ignore:\[COMBINE_COLUMNS_NULL_FILLED\]:svy.SvyUserWarning")
def test_combine_mixed_warn_records_and_raises_once():
    with pytest.warns(SvyUserWarning, match=r"\[COMBINE_MIXED_DESIGN\]") as rec:
        c = combine_samples(_mixed(), on_mixed_design="warn")
    assert len([r for r in _svy(rec) if "COMBINE_MIXED_DESIGN" in str(r.message)]) == 1
    found = c.warnings.list(code="COMBINE_MIXED_DESIGN")
    assert len(found) == 1 and found[0].level == Severity.WARNING


# The mixed waves lack a design column, so it is null-filled: a side effect here.
@pytest.mark.filterwarnings(r"ignore:\[COMBINE_COLUMNS_NULL_FILLED\]:svy.SvyUserWarning")
def test_combine_mixed_ignore_and_warn_agree():
    with pytest.warns(SvyUserWarning):
        warned = combine_samples(_mixed(), on_mixed_design="warn")
    ignored = combine_samples(_mixed(), on_mixed_design="ignore")
    assert warned.data.equals(ignored.data) and warned.design == ignored.design


def test_combine_mixed_error_leaves_inputs():
    samples = _mixed()
    data = [s.data for s in samples]
    with pytest.raises(MethodError, match="on_mixed_design"):
        combine_samples(samples)
    assert all(s.data.equals(d) for s, d in zip(samples, data))


@pytest.mark.parametrize("value", ["raise", None, True])
def test_combine_mixed_bad_value(value):
    with pytest.raises(MethodError) as ei:
        combine_samples(_mixed(), on_mixed_design=value)
    assert ei.value.code == "INVALID_CHOICE"


# ---------------------------------------------------------------------------
# One documented meaning
# ---------------------------------------------------------------------------

ON_TRIAD = (
    '"error" raises and leaves the sample as it was; "warn" records the finding in '
    '``sample.warnings`` and raises it once as a ``SvyUserWarning``; "ignore" records it '
    "at INFO level without raising."
)


@pytest.mark.parametrize(
    "fn", [Wrangling.join, Wrangling.filter_records, combine_samples], ids=lambda f: f.__name__
)
def test_on_triad_documented(fn):
    assert any(p.startswith("on_") for p in inspect.signature(fn).parameters)
    assert ON_TRIAD in re.sub(r"\s+", " ", fn.__doc__)
