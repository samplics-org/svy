# tests/svy/errors/test_codes.py
"""Every code is in svy.errors.Code, belongs to one area and has a reference entry."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import warnings

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import (
    Code,
    ModelError,
    ModelWarning,
    SelectionWarning,
    SvyError,
    UnknownNameError,
    UsageWarning,
    WeightingError,
    WeightingWarning,
)
from svy.errors._codes import CODES_URL, code_url
from svy.extensions import _sample_accessors, register_sample_accessor


SRC = Path(svy.__file__).parent
SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "render_codes.py"
UPPER_SNAKE = re.compile(r"[A-Z][A-Z0-9]*(_[A-Z0-9]+)*")

# Section of _codes.py -> the area its classes name.
SECTION_AREA = {"API use": "Usage", "Files": "Io"}
# Classes named for a kind rather than built as Area.Kind.
NAMED_KINDS = {
    "UnknownNameError": "Usage",
    "LabelAPIRemoved": "Label",
    "LabelsReadOnly": "Label",
    "SingletonAPIRemoved": "Singleton",
}


@pytest.fixture(scope="module")
def render_codes():
    if not SCRIPT.exists():
        pytest.skip("scripts/ not shipped in sdist")
    spec = importlib.util.spec_from_file_location("render_codes", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_codes"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def entries(render_codes):
    return render_codes.collect()


def _trees():
    for path in SRC.rglob("*.py"):
        yield path, ast.parse(path.read_text())


def _area(raised_as: str) -> str:
    cls = raised_as.split(".")[0]
    if cls in NAMED_KINDS:
        return NAMED_KINDS[cls]
    return re.sub(r"(Error|Warning)$", "", cls)


# ---- the catalogue -------------------------------------------------------------


def test_codes_are_their_names_in_upper_snake():
    assert len(Code) > 0
    for code in Code:
        assert code.value == code.name and UPPER_SNAKE.fullmatch(code.name), code


def test_no_code_is_a_bare_string():
    bare = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            values = []
            if isinstance(node, ast.keyword) and node.arg == "code":
                values = [node.value]
            elif isinstance(node, ast.Dict):
                values = [
                    v
                    for k, v in zip(node.keys, node.values)
                    if isinstance(k, ast.Constant) and k.value == "code"
                ]
            for v in values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    if UPPER_SNAKE.fullmatch(v.value):
                        bare.append(f"{path.relative_to(SRC)}:{v.lineno} {v.value}")
    assert bare == []


def test_every_code_is_raised_or_is_a_default():
    referenced = set()
    for path, tree in _trees():
        if path.name == "_codes.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and getattr(node.value, "id", "") == "Code":
                referenced.add(node.attr)
    assert referenced == {c.name for c in Code}


def test_every_code_has_a_class_and_a_message(entries):
    missing = [e.code for e in entries.values() if not e.raised_as or not e.title]
    assert missing == []


def test_a_code_belongs_to_one_area(entries):
    wrong = {}
    for e in entries.values():
        if e.area in ("Shared", "Defaults"):
            continue
        area = SECTION_AREA.get(e.area, e.area)
        areas = {_area(r) for r in e.raised_as}
        if areas != {area}:
            wrong[e.code] = (area, sorted(e.raised_as))
    assert wrong == {}


def test_shared_codes_are_raised_by_any_area(entries):
    for e in entries.values():
        if e.area == "Shared":
            assert any(r.startswith("<Area>Error.") for r in e.raised_as), e.code


@pytest.mark.parametrize("old", ["MAX_ITER_REACHED", "SINGLETON_PSU"])
def test_split_or_unused_codes_are_gone(old):
    assert old not in Code.__members__


def test_old_enum_says_where_it_went():
    with pytest.raises(UnknownNameError) as err:
        svy.errors.WarnCode
    assert "svy.errors.Code" in err.value.detail


# ---- the reference page --------------------------------------------------------


def test_reference_page_lists_every_code_once(render_codes, entries):
    page = render_codes.render(entries)
    anchors = re.findall(r'<span id="([a-z0-9-]+)">', page)
    assert len(anchors) == len(set(anchors)) == len(Code)
    for code in Code:
        assert code_url(code) == f"{CODES_URL}#{code.name.lower().replace('_', '-')}"
        assert code_url(code).rsplit("#", 1)[1] in anchors


def test_reference_page_shows_class_and_message(render_codes, entries):
    page = render_codes.render(entries)
    row = next(line for line in page.splitlines() if 'id="glm-separation"' in line)
    assert "`ModelWarning`" in row and "Outcome separated" in row
    row = next(line for line in page.splitlines() if 'id="rank-deficient"' in line)
    assert "`ModelError.ValueError`" in row and "GLM cannot be fitted" in row


def test_render_check_mode(render_codes, tmp_path, monkeypatch):
    out = tmp_path / "codes.qmd"
    out.write_text(render_codes.render(render_codes.collect()))
    monkeypatch.setattr(sys, "argv", ["render_codes", str(out), "--check"])
    assert render_codes.main() == 0
    out.write_text("stale")
    assert render_codes.main() == 1


# ---- reports carry their code and its link -------------------------------------


def test_report_links_its_code():
    err = ModelError.not_fitted(where="w")
    assert err.docs_url == f"{CODES_URL}#model-not-fitted"
    assert f"Docs: {CODES_URL}#model-not-fitted" in err.text()
    w = ModelWarning(code=Code.GLM_SEPARATION, title="t", detail="d")
    assert w.docs_url == f"{CODES_URL}#glm-separation"
    assert str(w) == "[GLM_SEPARATION] t: d"


def test_own_docs_url_and_unlisted_codes_keep_theirs():
    assert SvyError(title="t", detail="d", code=Code.INVALID_N, docs_url="x").docs_url == "x"
    assert SvyError(title="t", detail="d", code="MY_OWN").docs_url is None
    assert code_url("MY_OWN") is None


def test_code_is_kept_as_plain_text():
    err = SvyError(title="t", detail="d")
    assert type(err.code) is str and err.code == Code.SVY_ERROR
    assert "code='SVY_ERROR'" in repr(err)
    assert type(ModelWarning(code=Code.GLM_SEPARATION, title="t", detail="d").code) is str


# ---- one event, one code -------------------------------------------------------


def _raking_sample() -> svy.Sample:
    df = pl.DataFrame(
        {
            "age": ["18-34", "35-54", "55+", "18-34", "35-54", "55+", "18-34", "35-54"],
            "region": ["North"] * 4 + ["South"] * 4,
            "w": [10.0] * 8,
        }
    )
    return svy.Sample(df, svy.Design(wgt="w"))


def _rake(s: svy.Sample, on: str) -> svy.Sample:
    controls = {
        "age": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    return s.weighting.rake(controls=controls, max_iter=3, tol=1e-20, on_nonconvergence=on)


def test_weighting_nonconvergence_has_one_code_as_error_and_warning():
    with pytest.raises(WeightingError) as err:
        _rake(_raking_sample(), "error")
    assert isinstance(err.value, RuntimeError)
    assert err.value.code == Code.WEIGHTING_NOT_CONVERGED
    with pytest.warns(WeightingWarning) as rec:
        out = _rake(_raking_sample(), "warn")
    (w,) = [r.message for r in rec if r.message.code == Code.WEIGHTING_NOT_CONVERGED]
    assert out.warnings.list(code=Code.WEIGHTING_NOT_CONVERGED) == [w]


def test_glm_nonconvergence_has_its_own_code():
    rng = np.random.default_rng(20261027)
    n = 200
    x = rng.normal(size=n)
    df = pl.DataFrame(
        {"x": x, "w": np.ones(n), "y": (x + rng.normal(size=n) > 0).astype(np.int64)}
    )
    s = svy.Sample(df, svy.Design(wgt="w"))
    with pytest.warns(ModelWarning, match=r"^\[GLM_NOT_CONVERGED\]"):
        s.glm.fit("y", x=["x"], family="binomial", max_iter=1)
    (w,) = s.warnings.list(code=Code.GLM_NOT_CONVERGED)
    assert type(w) is ModelWarning


# ---- warnings that had no code -------------------------------------------------


@pytest.fixture
def accessor_name():
    name = "_svy_test_accessor"
    yield name
    if name in vars(svy.Sample):
        delattr(svy.Sample, name)
    _sample_accessors.pop(name, None)


def test_overwritten_accessor_is_a_coded_usage_warning(accessor_name):
    register_sample_accessor(accessor_name)(type("First", (), {"__init__": lambda s, x: None}))
    with pytest.warns(UsageWarning) as rec:
        register_sample_accessor(accessor_name)(
            type("Second", (), {"__init__": lambda s, x: None})
        )
    (w,) = [r.message for r in rec]
    assert w.code == Code.ACCESSOR_OVERWRITTEN
    assert str(w).startswith("[ACCESSOR_OVERWRITTEN] Accessor overwritten: Accessor '_svy_test")


def test_allocation_beyond_the_frame_is_a_coded_selection_warning():
    with pytest.warns(SelectionWarning) as rec:
        svy.SampleSize().allocate(500, pop_size={"a": 50, "b": 80})
    codes = [r.message.code for r in rec]
    assert Code.ALLOCATION_EXCEEDS_FRAME in codes


def test_ignored_export_option_is_a_coded_io_warning(tmp_path):
    s = svy.Sample(pl.DataFrame({"y": [1.0, 2.0], "w": [1.0, 1.0]}), svy.Design(wgt="w"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        svy.io.write_sas(s, tmp_path / "t.xpt", format="sas7bdat")
    (w,) = [r.message for r in rec if getattr(r.message, "code", None) == Code.IO_OPTION_IGNORED]
    assert type(w) is svy.errors.IoWarning
    assert "ReadStat writes SAS Transport (XPT) only" in w.detail
