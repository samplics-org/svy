# tests/svy/errors/test_areas.py
"""Errors and warnings name the part of svy that raised them; each error is also a builtin."""

from __future__ import annotations

import ast
import copy
import inspect
import pathlib
import pickle
import traceback
import warnings

import numpy as np
import polars as pl
import pytest

import svy

from svy import col
from svy.errors import (
    DataError,
    DatasetError,
    DataWarning,
    DesignError,
    DesignWarning,
    EstimationError,
    EstimationWarning,
    IoError,
    LabelError,
    LabelWarning,
    ModelError,
    ModelWarning,
    SelectionWarning,
    SerializationError,
    SingletonWarning,
    SvyError,
    SvyWarning,
    UnknownNameError,
    UsageError,
    WeightingError,
    WeightingWarning,
)
from svy.errors._common import _Constructors
from svy.errors._label import LabelAPIRemoved, LabelsReadOnly
from svy.errors._singleton import SingletonAPIRemoved
from svy.selection import Stage


SRC = pathlib.Path(svy.__file__).parent

AREA_CODES = {
    "Usage": "USAGE_ERROR",
    "Design": "DESIGN_ERROR",
    "Singleton": "SINGLETON_ERROR",
    "Estimation": "ESTIMATION_ERROR",
    "Model": "MODEL_ERROR",
    "Weighting": "WEIGHTING_ERROR",
    "Selection": "SELECTION_ERROR",
    "Label": "LABEL_ERROR",
    "Data": "DATA_ERROR",
    "Io": "IO_ERROR",
    "Dataset": "DATASET_ERROR",
    "Serialization": "SERIALIZATION_ERROR",
}
WARNING_AREAS = [
    "Usage",
    "Design",
    "Singleton",
    "Estimation",
    "Model",
    "Weighting",
    "Selection",
    "Label",
    "Data",
    "Io",
    "Dataset",
]
KINDS = [
    TypeError,
    ValueError,
    KeyError,
    AttributeError,
    NotImplementedError,
    RuntimeError,
    ImportError,
    OSError,
    FileNotFoundError,
    PermissionError,
]


def _area(name: str) -> type[SvyError]:
    return getattr(svy.errors, f"{name}Error")


def _make(cls: type[SvyError]) -> SvyError:
    return cls(title="T", detail="D", code="SOME_CODE")


# ---- every area, every kind ---------------------------------------------------


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.__name__)
@pytest.mark.parametrize("area", list(AREA_CODES))
def test_kind_is_its_area_and_its_builtin(area, kind):
    cls = _area(area)
    pair = getattr(cls, kind.__name__)
    assert issubclass(pair, cls) and issubclass(pair, kind) and issubclass(pair, SvyError)
    assert f"{pair.__module__}.{pair.__qualname__}" == f"svy.errors.{area}Error.{kind.__name__}"
    err = _make(pair)
    for catch in (cls, kind, SvyError):
        with pytest.raises(catch) as caught:
            raise err
        assert caught.value is err


@pytest.mark.parametrize("area", list(AREA_CODES))
def test_an_area_does_not_catch_another(area):
    other = "Data" if area != "Data" else "Model"
    with pytest.raises(_area(area)):
        try:
            raise _make(_area(area).ValueError)
        except _area(other):
            pytest.fail(f"{other}Error caught a {area}Error")


@pytest.mark.parametrize("area, code", list(AREA_CODES.items()))
def test_default_code_names_the_area(area, code):
    assert _make(_area(area)).code == "SOME_CODE"
    assert _area(area)(title="T", detail="D").code == code
    assert _area(area).TypeError(title="T", detail="D").code == code


@pytest.mark.parametrize("how", ["deepcopy", "pickle"])
@pytest.mark.parametrize("area", list(AREA_CODES))
def test_kind_survives_copies(area, how):
    err = _area(area).KeyError(title="T", detail="D", code="C", got={"a": [1]})
    back = copy.deepcopy(err) if how == "deepcopy" else pickle.loads(pickle.dumps(err))
    assert type(back) is type(err) and vars(back) == vars(err)


def test_traceback_names_the_public_class():
    err = ModelError.not_fitted(where="w")
    assert traceback.format_exception_only(err)[0].startswith("svy.errors.ModelError.ValueError:")
    assert repr(err).startswith("ModelError.ValueError(")
    assert err.to_dict()["type"] == "ModelError.ValueError"


@pytest.mark.parametrize(
    "cls, area, kind",
    [
        (UnknownNameError, UsageError, AttributeError),
        (LabelAPIRemoved, LabelError, AttributeError),
        (LabelsReadOnly, LabelError, TypeError),
        (SingletonAPIRemoved, svy.errors.SingletonError, AttributeError),
    ],
)
def test_named_kinds_are_their_area_and_builtin(cls, area, kind):
    assert issubclass(cls, area) and issubclass(cls, kind)
    assert not any(k.__name__ in vars(cls) for k in KINDS)


@pytest.mark.parametrize("area", list(AREA_CODES))
def test_kinds_have_no_kinds_of_their_own(area):
    pair = _area(area).ValueError
    assert not any(k.__name__ in vars(pair) for k in KINDS)


@pytest.mark.parametrize("area", WARNING_AREAS)
def test_warning_classes_are_svy_warnings_not_errors(area):
    cls = getattr(svy.errors, f"{area}Warning")
    assert issubclass(cls, SvyWarning) and issubclass(cls, UserWarning)
    assert not issubclass(cls, SvyError)
    assert cls.__module__ == "svy.errors"


@pytest.mark.parametrize("old", ["MethodError", "DimensionError"])
@pytest.mark.parametrize("owner", [svy, svy.errors])
def test_dissolved_classes_say_what_replaced_them(owner, old):
    with pytest.raises(UnknownNameError) as err:
        getattr(owner, old)
    assert "EstimationError" in err.value.detail and "DataError" in err.value.detail


# ---- shared constructors ---------------------------------------------------------

SHARED = {
    "invalid_choice": (
        dict(where="w", param="p", got="x", allowed=["a"]),
        ValueError,
        "INVALID_CHOICE",
    ),
    "not_applicable": (
        dict(where="w", method="m", reason="r"),
        ValueError,
        "METHOD_NOT_APPLICABLE",
    ),
    "invalid_range": (
        dict(where="w", param="p", got=2, min_=0, max_=1),
        ValueError,
        "INVALID_RANGE",
    ),
    "invalid_type": (dict(where="w", param="p", got=1, expected="str"), TypeError, "INVALID_TYPE"),
    "invalid_value": (dict(where="w", param="p", got="", detail="d"), ValueError, "INVALID_VALUE"),
    "missing_columns": (dict(where="w", param="p", missing=["z"]), KeyError, "MISSING_COLUMNS"),
    "empty_estimates": (dict(where="w"), ValueError, "EMPTY_ESTIMATES"),
    "group_levels_mismatch": (
        dict(where="w", var="v", expected_levels=[1, 2], got_levels=[3]),
        ValueError,
        "GROUP_LEVELS_MISMATCH",
    ),
    "weight_overwrite": (
        dict(where="w", columns={"w": "the weight"}),
        ValueError,
        "WEIGHT_OVERWRITE",
    ),
    "reserved_columns": (dict(where="w", columns=["__svy_x"]), ValueError, "RESERVED_COLUMN"),
    "data_rows_changed": (dict(where="w", n_old=3, n_new=4), ValueError, "DATA_ROWS_CHANGED"),
    "no_rep_wgts": (dict(where="w"), ValueError, "REP_WGTS_MISSING"),
    "invalid_rep_prefix": (dict(where="w", got=""), ValueError, "REP_WGTS_PREFIX_INVALID"),
    "unknown_rep_prefix": (
        dict(where="w", got="q", known={"r": ("w", 4)}),
        KeyError,
        "REP_WGTS_UNKNOWN_PREFIX",
    ),
    "rep_rename_collision": (
        dict(where="w", prefix="r", existing=["r1"]),
        ValueError,
        "REP_WGTS_RENAME_COLLISION",
    ),
    "param_renamed": (dict(where="w", method="m", old="a", new="b"), TypeError, "PARAM_RENAMED"),
}


def test_every_shared_constructor_is_listed():
    shared = {n for n, v in vars(_Constructors).items() if isinstance(v, classmethod)}
    assert shared == set(SHARED)


@pytest.mark.parametrize("area", [EstimationError, DataError], ids=lambda c: c.__name__)
@pytest.mark.parametrize("name", list(SHARED))
def test_shared_constructor_raises_its_kind_in_the_callers_area(name, area):
    kwargs, kind, code = SHARED[name]
    err = getattr(area, name)(**kwargs)
    assert type(err) is getattr(area, kind.__name__)
    assert err.code == code and err.where == "w"


# ---- area constructors -----------------------------------------------------------

OWN = [
    (lambda: IoError.not_found(where="w", path="x.csv"), IoError, FileNotFoundError),
    (lambda: IoError.permission_denied(where="w", path="x.csv"), IoError, PermissionError),
    (lambda: IoError.read_failed(where="w", path="x.csv", reason="r"), IoError, OSError),
    (lambda: DatasetError.remote_unavailable(where="w"), DatasetError, ImportError),
    (lambda: DatasetError.not_found(where="w", slug="s"), DatasetError, KeyError),
    (
        lambda: DatasetError.download_failed(where="w", slug="s", url="https://x", reason="r"),
        DatasetError,
        OSError,
    ),
    (
        lambda: DatasetError.bundled_unavailable(where="w", slug="s", reason="r"),
        DatasetError,
        FileNotFoundError,
    ),
    (lambda: ModelError.not_fitted(where="w"), ModelError, ValueError),
    (
        lambda: WeightingError.not_converged(
            where="w", method="rake", what="margins", max_iter=5, hint="h"
        ),
        WeightingError,
        RuntimeError,
    ),
    (
        lambda: WeightingError.bounds_invalid(where="w", got=1, reason="r", as_type=True),
        WeightingError,
        TypeError,
    ),
    (
        lambda: WeightingError.bounds_invalid(where="w", got=1, reason="r"),
        WeightingError,
        ValueError,
    ),
    (
        lambda: WeightingError.not_supported_yet(where="w", param="p", got=1, hint="h"),
        WeightingError,
        NotImplementedError,
    ),
    (
        lambda: WeightingError.missing_columns(
            where="w", param="p", missing=["z"], available=["a"]
        ),
        WeightingError,
        KeyError,
    ),
    (
        lambda: WeightingError.param_renamed(where="w", method="rake", old="a", new="b"),
        WeightingError,
        TypeError,
    ),
    (lambda: LabelError.unknown_scheme(where="w", param="p", got="s"), LabelError, KeyError),
    (
        lambda: SerializationError.unsupported_type(got_type="list", registered=["Estimate"]),
        SerializationError,
        TypeError,
    ),
    (lambda: UsageError.not_polars(where="w", got=1), UsageError, TypeError),
    (
        lambda: UsageError.unknown_arguments(where="w", unknown=["x"], known=["y"]),
        UsageError,
        TypeError,
    ),
]


@pytest.mark.parametrize("build, area, kind", OWN)
def test_area_constructor_raises_its_kind(build, area, kind):
    err = build()
    assert type(err) is getattr(area, kind.__name__)


# ---- real calls ------------------------------------------------------------------


def _sample(**kw) -> svy.Sample:
    rng = np.random.default_rng(20261027)
    n = 60
    df = pl.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": np.tile(["a", "b", "c"], n // 3),
            "w": np.ones(n),
            "psu": np.repeat(np.arange(12), 5),
            "st": np.repeat([1, 2], n // 2),
        }
    ).with_columns(**kw)
    return svy.Sample(df, svy.Design(stratum="st", psu="psu", wgt="w"))


CALLS = {
    "wrangling, dtype missing": (
        lambda s: s.wrangling.cast("y"),
        DataError,
        TypeError,
        "CAST_DTYPE_REQUIRED",
    ),
    "wrangling, reserved name": (
        lambda s: s.wrangling.with_row_index("__svy_row_index__"),
        DataError,
        ValueError,
        "RESERVED_COLUMN",
    ),
    "Sample, reserved name": (
        lambda s: svy.Sample(s.data.with_columns(__svy_row_index__=pl.lit(1)), svy.Design()),
        DesignError,
        ValueError,
        "RESERVED_COLUMN",
    ),
    "estimation, deff=True": (
        lambda s: s.estimation.mean("y", deff=True),
        EstimationError,
        TypeError,
        "DEFF_BOOL_REJECTED",
    ),
    "any call, alpha out of range": (
        lambda s: s.estimation.mean("y", alpha=1.5),
        UsageError,
        ValueError,
        "INVALID_RANGE",
    ),
    "any call, on_* option": (
        lambda s: s.weighting.trim(upper=5.0, on_nonconvergence="raise"),
        UsageError,
        ValueError,
        "INVALID_CHOICE",
    ),
    "weighting, control keys": (
        lambda s: s.weighting.poststratify({"zz": 1.0}, cells="g"),
        WeightingError,
        ValueError,
        "CONTROLS_KEYS_MISMATCH",
    ),
    "glm, reference level": (
        lambda s: s.glm.fit("y", x=[svy.Cat("g", ref="zz")]),
        ModelError,
        ValueError,
        "CAT_REF_NOT_FOUND",
    ),
    "sampling, stages": (
        lambda s: s.sampling.multistage("x"),
        svy.errors.SelectionError,
        ValueError,
        "INVALID_STAGE",
    ),
    "io, missing file": (
        lambda s: svy.io.read_csv("/nonexistent/x.csv"),
        IoError,
        FileNotFoundError,
        "FILE_NOT_FOUND",
    ),
    "Sample, not polars": (
        lambda s: svy.Sample({"a": [1]}, svy.Design()),
        UsageError,
        TypeError,
        "DATA_NOT_POLARS",
    ),
    "labels, read-only view": (
        lambda s: s.meta.value_labels.__setitem__("g", {}),
        LabelError,
        TypeError,
        "LABEL_READ_ONLY",
    ),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_real_call_raises_its_area_and_kind(name):
    call, area, kind, code = CALLS[name]
    with pytest.raises(area) as caught:
        call(_sample())
    assert isinstance(caught.value, kind)
    assert caught.value.code == code


# ---- warnings by area ------------------------------------------------------------


def _persons() -> svy.Sample:
    df = pl.DataFrame(
        {
            "stratum": ["a", "a", "a", "b", "b", "b", "b"],
            "psu": [1, 1, 5, 2, 2, 6, 3],
            "wgt": [2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 4.0],
        }
    )
    return svy.Sample(df, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))


def _separated(s: svy.Sample) -> svy.Sample:
    yb = pl.when(pl.col("g") == "c").then(0).otherwise((pl.col("y") > 0).cast(pl.Int64))
    out = _sample(yb=yb)
    out.glm.fit("yb", x=[svy.Cat("g")], family="binomial")
    return out


def _null_cell(s: svy.Sample) -> svy.Sample:
    df = s.data.with_columns(
        c=pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("g"))
    )
    return svy.Sample(df, svy.Design(wgt="w")).weighting.poststratify(
        {"a": 20.0, "b": 20.0, "c": 20.0}, cells="c"
    )


def _stage_takes_all(s: svy.Sample) -> svy.Sample:
    frame = pl.DataFrame(
        {
            "region": ["a"] * 6 + ["b"] * 4,
            "district": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2],
            "ea": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    return svy.Sample(frame, svy.Design(stratum="region")).sampling.multistage(
        [Stage("district", n=1), Stage("ea", n=5)], rstate=1
    )


WARNS = {
    "GLM_SEPARATION": (_separated, ModelWarning),
    "CELLS_NULL_UNADJUSTED": (_null_cell, WeightingWarning),
    "JOIN_UNMATCHED": (
        lambda s: s.wrangling.join(
            pl.DataFrame({"g": ["a", "b"], "z": [1, 2]}), on="g", on_unmatched="warn"
        ),
        DataWarning,
    ),
    "SINGLETONS_DETECTED": (
        lambda s: _persons().wrangling.filter_records(
            col("psu") != 5, check_singletons=True, on_singletons="warn"
        ),
        SingletonWarning,
    ),
    "DESIGN_FIELDS_REMOVED": (
        lambda s: s.wrangling.remove_columns("psu", force=True),
        DesignWarning,
    ),
    "LABEL_KEY_NOT_IN_DATA": (
        lambda s: svy.Sample(pl.DataFrame({"a": [1, 2]})).wrangling.apply_labels(
            categories={"a": {1: "One", 2: "Two", 99: "Unknown"}}
        ),
        LabelWarning,
    ),
    "STAGE_TAKES_ALL": (_stage_takes_all, SelectionWarning),
}


# Recorded at INFO: kept on the sample, not raised.
_INFO = {"LABEL_KEY_NOT_IN_DATA", "STAGE_TAKES_ALL"}
_RAISED = [c for c in WARNS if c not in _INFO]


def _run(call, *ignore: type[Warning]):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for category in ignore:
            warnings.filterwarnings("ignore", category=category)
        out = call(_sample())
    return out, rec


@pytest.mark.parametrize("code", list(WARNS))
def test_warning_is_recorded_as_its_area(code):
    call, area = WARNS[code]
    out, rec = _run(call)
    (recorded,) = out.warnings.list(code=code)
    assert type(recorded) is area
    raised = [w for w in rec if getattr(w.message, "code", None) == code]
    assert len(raised) == (0 if code in _INFO else 1)


@pytest.mark.parametrize("code", _RAISED)
def test_raised_warning_is_the_recorded_one(code):
    call, area = WARNS[code]
    out, rec = _run(call)
    (raised,) = [w for w in rec if getattr(w.message, "code", None) == code]
    assert raised.category is area
    assert out.warnings.list(code=code) == [raised.message]


@pytest.mark.parametrize("code", _RAISED)
def test_filtering_one_area_leaves_the_others(code):
    call, area = WARNS[code]
    other = ModelWarning if area is not ModelWarning else DataWarning
    _, rec = _run(call, other)
    assert [getattr(w.message, "code", None) for w in rec].count(code) == 1
    _, rec = _run(call, area)
    assert code not in [getattr(w.message, "code", None) for w in rec]


def test_result_findings_carry_their_area():
    s = _sample(b=pl.lit(1))
    (finding,) = s.estimation.prop("b").findings
    assert type(finding) is EstimationWarning and finding.code == "PROP_CI_BOUNDARY"


# ---- the rules hold across the source --------------------------------------------

_AREA_ERRORS = {f"{a}Error" for a in AREA_CODES}


def _calls():
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                yield path, node


def test_no_area_error_is_built_without_a_kind():
    bare = [
        f"{p.relative_to(SRC)}:{n.lineno}"
        for p, n in _calls()
        if isinstance(n.func, ast.Name) and n.func.id in _AREA_ERRORS and p.parts[-2] != "errors"
    ]
    assert bare == []


def test_constructors_return_a_kind():
    plain = []
    for path in (SRC / "errors").glob("_*.py"):
        tree = ast.parse(path.read_text())
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            named_kind = any(
                isinstance(s, ast.AnnAssign) and getattr(s.target, "id", "") == "_IS_KIND"
                for s in cls.body
            )
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                for ret in [n for n in ast.walk(fn) if isinstance(n, ast.Return)]:
                    call = ret.value
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "cls"
                        and not named_kind
                    ):
                        plain.append(f"{path.name}:{cls.name}.{fn.name}")
    assert plain == []


def test_warnings_are_raised_as_records():
    keyword_form = [
        f"{p.relative_to(SRC)}:{n.lineno}"
        for p, n in _calls()
        if (getattr(n.func, "attr", None) or getattr(n.func, "id", None))
        in ("warn", "emit_finding")
        and not (
            isinstance(n.func, ast.Attribute) and getattr(n.func.value, "id", "") == "warnings"
        )
        and n.keywords
        and not (
            isinstance(n.func, ast.Attribute) and getattr(n.func.value, "id", "") == "_pywarnings"
        )
    ]
    assert keyword_form == []


def test_sample_warn_takes_a_record():
    params = list(inspect.signature(svy.Sample.warn).parameters)
    assert params == ["self", "finding"]
