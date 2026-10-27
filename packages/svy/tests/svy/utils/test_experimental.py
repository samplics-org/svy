# tests/svy/utils/test_experimental.py
"""@experimental warns once per function, notes every result, and marks the docstring."""

from __future__ import annotations

import json
import warnings

import polars as pl
import pytest

import svy

from svy.errors import Code, ExperimentalWarning, SvyWarning
from svy.serialize import from_json, to_json
from svy.utils import experimental


NOTE = "note: Experimental method [EXPERIMENTAL]"


@pytest.fixture
def sample() -> svy.Sample:
    return svy.Sample(
        pl.DataFrame({"y": [1.0, 2.0, 3.0], "w": [1.0, 1.0, 1.0]}), svy.Design(wgt="w")
    )


def _calls(fn, *args, n: int = 1) -> tuple[list, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = [fn(*args) for _ in range(n)]
    return out, [r for r in rec if isinstance(r.message, ExperimentalWarning)]


def test_first_call_warns_and_later_calls_do_not(sample):
    @experimental
    def mean_y(s):
        return s.estimation.mean("y")

    _, rec = _calls(mean_y, sample, n=3)
    (only,) = rec
    w = only.message
    assert type(w) is ExperimentalWarning and w.code == Code.EXPERIMENTAL
    assert "mean_y() is experimental" in w.detail
    assert str(w).startswith("[EXPERIMENTAL] Experimental method: ")
    assert only.filename == __file__


def test_each_function_warns_once_on_its_own(sample):
    @experimental
    def first(s):
        return s.estimation.mean("y")

    @experimental
    def second(s):
        return s.estimation.total("y")

    _, a = _calls(first, sample, n=2)
    _, b = _calls(second, sample, n=2)
    assert len(a) == len(b) == 1


def test_every_result_carries_the_note(sample):
    @experimental
    def mean_y(s):
        return s.estimation.mean("y")

    results, _ = _calls(mean_y, sample, n=2)
    for r in results:
        assert [f.code for f in r.findings] == [Code.EXPERIMENTAL]
        assert str(r).count(NOTE) == 1
        assert NOTE in r.__plain_str__()


def test_the_note_is_saved_with_the_result(sample):
    @experimental
    def mean_y(s):
        return s.estimation.mean("y")

    (r,), _ = _calls(mean_y, sample)
    payload = json.loads(to_json(r))
    assert [f["code"] for f in payload["findings"]] == ["EXPERIMENTAL"]
    assert [f.code for f in from_json(to_json(r)).findings] == ["EXPERIMENTAL"]


def test_the_note_joins_the_call_s_own_warnings():
    df = pl.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "w": [1.0] * 4, "r1": [1.0] * 4, "r2": [1.0] * 4}
    )
    s = svy.Sample(df, svy.Design(wgt="w", rep_wgts=svy.BrrWgts(prefix="r", n_reps=2)))

    @experimental
    def taylor_mean(s):
        return s.estimation.mean("y", method="taylor")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SvyWarning)
        r = taylor_mean(s)
    assert [f.code for f in r.findings] == [Code.TAYLOR_WITHOUT_DESIGN, Code.EXPERIMENTAL]


def test_not_recorded_on_the_sample(sample):
    @experimental
    def mean_y(s):
        return s.estimation.mean("y")

    _calls(mean_y, sample, n=2)
    assert sample.warnings.list(code=Code.EXPERIMENTAL) == []


def test_a_value_that_is_not_a_result_passes_through():
    @experimental
    def answer():
        return 42

    (out,), rec = _calls(answer)
    assert out == 42 and len(rec) == 1


def test_silencing_the_warning_keeps_the_note(sample):
    @experimental
    def mean_y(s):
        return s.estimation.mean("y")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        r = mean_y(sample)
    assert not [x for x in rec if isinstance(x.message, ExperimentalWarning)]
    assert [f.code for f in r.findings] == [Code.EXPERIMENTAL]


def test_a_failing_call_still_warns_first():
    @experimental
    def broken():
        raise RuntimeError("boom")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="boom"):
            broken()
    assert len([r for r in rec if isinstance(r.message, ExperimentalWarning)]) == 1


@pytest.mark.parametrize(
    "doc, expected",
    [
        (
            """Mean of y.

            More text.""",
            "Mean of y.\n\n**Experimental.** It may change in any release, without a "
            "deprecation period.\n\nMore text.",
        ),
        (
            "One line.",
            "One line.\n\n**Experimental.** It may change in any release, without a "
            "deprecation period.",
        ),
        (None, "**Experimental.** It may change in any release, without a deprecation period."),
    ],
    ids=["paragraphs", "one line", "none"],
)
def test_the_docstring_says_so(doc, expected):
    def fn():
        pass

    fn.__doc__ = doc
    assert experimental(fn).__doc__ == expected


def test_the_wrapper_keeps_the_function_s_identity():
    def mean_y():
        """Doc."""

    wrapped = experimental(mean_y)
    assert wrapped.__name__ == "mean_y" and wrapped.__wrapped__ is mean_y
    assert wrapped.__svy_experimental__ is True
    assert not hasattr(mean_y, "__svy_experimental__")


def test_a_method_is_named_by_its_class(sample):
    class Tools:
        @experimental
        def mean_y(self, s):
            return s.estimation.mean("y")

    (_,), (w,) = _calls(Tools().mean_y, sample)
    assert "Tools.mean_y() is experimental" in w.message.detail


def test_the_warning_is_an_svy_warning_in_svy_errors():
    assert issubclass(ExperimentalWarning, SvyWarning) and issubclass(
        ExperimentalWarning, UserWarning
    )
    assert ExperimentalWarning.__module__ == "svy.errors"
    assert svy.errors.ExperimentalWarning is ExperimentalWarning
