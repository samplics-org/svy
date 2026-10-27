# tests/svy/datasets/test_input_errors.py
"""Bad input to svy.datasets raises an svy error that is also the builtin it replaces."""

from __future__ import annotations

import numpy as np
import pytest

import svy

from svy.datasets import _bundled
from svy.errors import Code, DatasetError, SvyError, UsageError


SLUG = "ea_frame_wb_2023"


@pytest.fixture(autouse=True)
def _offline_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SVYLAB_CACHE_DIR", str(tmp_path))


@pytest.fixture(scope="module")
def rows() -> int:
    assert _bundled.has(SLUG)
    return svy.datasets.load(SLUG, source="bundled").height


def _load(**kw):
    return svy.datasets.load(SLUG, source="bundled", **kw)


def _raises(area, kind, code, call, *, param=None, says=None):
    with pytest.raises(area) as err:
        call()
    e = err.value
    assert type(e) is getattr(area, kind.__name__)
    assert isinstance(e, kind) and isinstance(e, SvyError)
    assert e.code == code
    if param is not None:
        assert e.param == param
    if says is not None:
        assert says in e.detail
    return e


@pytest.mark.parametrize(
    "kw, param, kind, code",
    [
        ({"n": 5, "rate": 0.5}, "n", ValueError, Code.INVALID_VALUE),
        ({"n": {"a": 1}}, "n", ValueError, Code.INVALID_VALUE),
        ({"rate": {"a": 0.5}}, "rate", ValueError, Code.INVALID_VALUE),
        ({"order_type": "up"}, "order_type", ValueError, Code.INVALID_CHOICE),
        ({"n": -1}, "n", ValueError, Code.INVALID_RANGE),
        ({"n": "5"}, "n", TypeError, Code.INVALID_TYPE),
        ({"n": 1.5}, "n", TypeError, Code.INVALID_TYPE),
        ({"n": True}, "n", TypeError, Code.INVALID_TYPE),
        ({"rate": 0}, "rate", ValueError, Code.INVALID_RANGE),
        ({"rate": 2}, "rate", ValueError, Code.INVALID_RANGE),
        ({"rate": "a"}, "rate", TypeError, Code.INVALID_TYPE),
        ({"rate": {"x": 2.0}, "by": "urbrur"}, "rate", ValueError, Code.INVALID_RANGE),
        ({"source": "web"}, "source", ValueError, Code.INVALID_CHOICE),
    ],
    ids=[
        "n and rate",
        "n map without by",
        "rate map without by",
        "order_type",
        "n negative",
        "n str",
        "n float",
        "n bool",
        "rate 0",
        "rate 2",
        "rate str",
        "group rate 2",
        "source",
    ],
)
def test_load_option(kw, param, kind, code):
    args = {"source": "bundled", **kw}
    _raises(DatasetError, kind, code, lambda: svy.datasets.load(SLUG, **args), param=param)


def test_rate_messages_kept():
    _raises(DatasetError, ValueError, Code.INVALID_RANGE, lambda: _load(rate=2), says="(0, 1]")
    _raises(
        DatasetError,
        ValueError,
        Code.INVALID_RANGE,
        lambda: _load(rate={"x": 2.0}, by="urbrur"),
        says="per-group rate",
    )


def test_whole_number_rate_and_n(rows):
    assert _load(rate=1).height == rows
    assert _load(n=0).height == 0
    assert _load(n=np.int64(3)).height == 3


@pytest.mark.parametrize(
    "kw, param",
    [
        ({"select": ["geo1", "nope"]}, "select"),
        ({"where": svy.col("nope") > 1}, "where"),
        ({"order_by": "nope"}, "order_by"),
        ({"n": 1, "by": "nope"}, "by"),
    ],
    ids=["select", "where", "order_by", "by"],
)
def test_missing_column(kw, param):
    e = _raises(DatasetError, KeyError, Code.MISSING_COLUMNS, lambda: _load(**kw), param=param)
    assert e.got == ["nope"]


def test_rstate_of_wrong_type():
    _raises(
        UsageError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: _load(n=2, order_type="random", rstate="x"),
        param="rstate",
    )


def test_numpy_integer_seed_is_reproducible():
    a = _load(n=5, order_type="random", rstate=np.int64(7))
    b = _load(n=5, order_type="random", rstate=7)
    assert a.equals(b)


@pytest.mark.parametrize(
    "call",
    [
        lambda: svy.datasets.catalog(source="web"),
        lambda: svy.datasets.describe(SLUG, source="web"),
    ],
    ids=["catalog", "describe"],
)
def test_unknown_source_elsewhere(call):
    _raises(DatasetError, ValueError, Code.INVALID_CHOICE, call, param="source")
