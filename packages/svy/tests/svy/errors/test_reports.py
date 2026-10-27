# tests/svy/errors/test_reports.py
"""Errors and warnings share one base; a warning is the record and the Python warning at once."""

from __future__ import annotations

import copy
import importlib
import pickle
import re
import warnings

import polars as pl
import pytest

import svy

from svy.errors import ModelError, Severity, SvyError, SvyWarning, WarningStore
from svy.errors._findings import SvyWarningsError, emit_finding, warn_no_sample


_SHARED = {
    "type",
    "level",
    "code",
    "title",
    "detail",
    "where",
    "param",
    "expected",
    "got",
    "hint",
    "docs_url",
    "extra",
}


def _sample() -> svy.Sample:
    return svy.Sample(pl.DataFrame({"y": [1.0, 2.0], "w": [1.0, 1.0]}), svy.Design(wgt="w"))


def _warning(**kw) -> SvyWarning:
    fields = {"code": "SOME_CODE", "title": "A title", "detail": "What happened."}
    return SvyWarning(**{**fields, **kw})


# ---- public surface ---------------------------------------------------------


def test_public_surface():
    assert set(svy.errors.__all__) == {
        "DatasetError",
        "DimensionError",
        "IoError",
        "LabelError",
        "MethodError",
        "ModelError",
        "SerializationError",
        "Severity",
        "SingletonError",
        "SvyError",
        "SvyWarning",
        "UnknownNameError",
        "UsageError",
        "WarnCode",
        "WarningStore",
        "WeightingError",
    }
    assert svy.SvyWarning is svy.errors.SvyWarning
    assert svy.SvyError is svy.errors.SvyError


@pytest.mark.parametrize(
    "owner, replacement", [(svy, "svy.SvyWarning"), (svy.errors, "svy.errors.SvyWarning")]
)
def test_old_warning_name_says_where_it_went(owner, replacement):
    with pytest.raises(AttributeError) as err:
        owner.SvyUserWarning
    assert err.value.code == "UNKNOWN_NAME"
    assert replacement in err.value.detail


def test_old_module_says_where_it_went():
    with pytest.raises(ModuleNotFoundError, match=r"svy\.core\.warnings moved to svy\.errors"):
        importlib.import_module("svy.core.warnings")


@pytest.mark.parametrize("old", ["base_errors", "model_errors", "weighting_errors"])
def test_error_modules_are_private(old):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"svy.errors.{old}")


def test_severity_has_no_error_level():
    assert [s.name for s in Severity] == ["INFO", "WARNING"]


# ---- the warning is the Python warning ----------------------------------------


def test_caught_warning_is_the_recorded_one():
    s = _sample()
    with pytest.warns(SvyWarning) as rec:
        recorded = s.warn(
            code="SOME_CODE", title="t", detail="d", where="w", hint="h", var="y", rows=[0]
        )
    (caught,) = rec
    assert caught.message is recorded
    assert s.warnings.list() == [recorded]
    assert (caught.message.code, caught.message.hint, caught.message.var) == (
        "SOME_CODE",
        "h",
        "y",
    )
    assert caught.message.rows == (0,)


def test_info_is_recorded_not_raised():
    s = _sample()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w = s.warn(code="AUDIT", title="t", detail="d", where="w", level=Severity.INFO)
    assert s.warnings.list() == [w]


def test_str_is_the_line_python_prints_and_filters_on():
    assert str(_warning()) == "[SOME_CODE] A title: What happened."
    assert str(_warning(detail="")) == "[SOME_CODE] A title"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=re.escape("[SOME_CODE]"))
        warnings.warn(_warning())
        warnings.warn(_warning(code="OTHER"))
    assert [w.message.code for w in rec] == ["OTHER"]


def test_category_filter_silences_every_svy_warning():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=SvyWarning)
        warnings.warn(_warning())
        warnings.warn("not svy's", UserWarning)
    assert [str(w.message) for w in rec] == ["not svy's"]


def test_error_filter_raises_the_instance():
    w = _warning()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SvyWarning) as err:
            warnings.warn(w)
    assert err.value is w


def test_emit_without_a_sample_raises_the_record():
    with pytest.warns(SvyWarning) as rec:
        emit_finding(code="LOOSE", title="t", detail="d", where="w", hint="h")
    assert (rec[0].message.code, rec[0].message.hint) == ("LOOSE", "h")


def test_uncoded_message_keeps_its_text():
    with pytest.warns(SvyWarning, match="Accessor 'x' is already defined") as rec:
        warn_no_sample("Accessor 'x' is already defined on Sample. Overwriting.")
    assert rec[0].message.code == "SVY_WARNING"


# ---- one shape, one set of renderers --------------------------------------------


def test_to_dict_has_one_shape():
    e = SvyError(title="T", detail="D", code="E").to_dict()
    w = _warning(var="y", rows=(1, 2)).to_dict()
    assert set(e) == _SHARED
    assert set(w) == _SHARED | {"var", "rows", "ts"}
    assert (e["level"], w["level"], _warning(level=Severity.INFO).to_dict()["level"]) == (
        "ERROR",
        "WARNING",
        "INFO",
    )
    assert (e["type"], w["type"]) == ("SvyError", "SvyWarning")
    assert w["rows"] == [1, 2]


@pytest.mark.parametrize(
    "level, mark, kind", [(Severity.WARNING, "⚠️", "warning"), (Severity.INFO, "ℹ️", "info")]
)
def test_warning_renders_with_its_level(level, mark, kind):
    w = _warning(level=level, param="n")
    assert w.text().strip().startswith(f"{mark} A title [SOME_CODE]")
    assert "- param: n" in w.text()
    assert w.markdown().startswith(f"**{mark} A title**")
    assert w._KIND == kind


def test_error_text_is_unchanged():
    e = SvyError(title="Bad", detail="d", code="BAD", param="n")
    assert str(e) == e.text() == "\n  ❌ Bad [BAD]\n  d\n  - param: n\n"


# ---- copies --------------------------------------------------------------------


@pytest.mark.parametrize(
    "report",
    [
        SvyError(title="T", detail="D", code="E", got={"a": 1}, extra={"k": [1]}),
        ModelError.not_fitted(where="w"),
        SvyWarning(code="W", title="T", detail="D", var="y", rows=(1, 2), state=3),
    ],
    ids=["error", "subclass", "warning"],
)
@pytest.mark.parametrize("how", ["deepcopy", "pickle"])
def test_copies_keep_class_and_fields(report, how):
    copied = copy.deepcopy(report) if how == "deepcopy" else pickle.loads(pickle.dumps(report))
    assert type(copied) is type(report)
    assert copied is not report
    assert vars(copied) == vars(report)


def test_derived_sample_copies_the_warnings():
    s = _sample()
    with pytest.warns(SvyWarning):
        s.warn(code="FIRST", title="t", detail="d", where="w")
    derived = s.wrangling.mutate({"z": pl.col("y") * 2})
    with pytest.warns(SvyWarning):
        derived.warn(code="SECOND", title="t", detail="d", where="w")
    assert [w.code for w in s.warnings] == ["FIRST"]
    assert [w.code for w in derived.warnings] == ["FIRST", "SECOND"]
    assert derived.warnings.list()[0] is not s.warnings.list()[0]


# ---- the store -----------------------------------------------------------------


def test_store_counts_and_escalates_warnings():
    store = WarningStore()
    store.add(_warning(code="A"))
    store.add(_warning(code="B", level=Severity.INFO))
    assert store.counts() == {"INFO": 1, "WARNING": 1, "TOTAL": 2}
    assert not hasattr(store, "raise_if_errors")
    with pytest.raises(SvyWarningsError) as err:
        store.escalate()
    assert [w["code"] for w in err.value.extra["warnings"]] == ["A"]


def test_store_exports_the_one_shape():
    store = WarningStore()
    store.add(_warning(code="A", var="y"))
    (row,) = store.to_dicts()
    assert row == store.list()[0].to_dict()
    assert store.to_polars().select("code", "level", "var").row(0) == ("A", "WARNING", "y")
