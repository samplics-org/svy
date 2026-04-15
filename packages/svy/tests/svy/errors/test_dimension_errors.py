import io

import pytest

from svy.errors.base_errors import SvyError
from svy.errors.dimension_errors import DimensionError


def test_dimension_error_defaults_code():
    err = DimensionError(title="Bad shape", detail="Mismatch.")
    assert isinstance(err, SvyError)
    assert err.code in {"DIMENSION_ERROR", "SVY_ERROR"}  # tolerate future default changes
    s = str(err)
    assert "Bad shape" in s


def test_invalid_n_ctor():
    err = DimensionError.invalid_n(where="sample.show_data", got=-3)
    assert err.code == "INVALID_N"
    assert err.param == "n"
    assert err.expected == "n >= 0 or None"
    assert err.got == -3

    # string contains clear guidance
    s = str(err)
    assert "Invalid row count" in s and "non-negative" in s

    # to_dict and markdown
    d = err.to_dict()
    assert d["error"]["param"] == "n"
    md = err.markdown()
    assert "`[INVALID_N]`" in md


def test_missing_columns_ctor():
    err = DimensionError.missing_columns(
        where="sample.show_data",
        param="columns",
        missing=["foo", "bar"],
        available=["a", "b", "c"],
    )
    assert err.code == "MISSING_COLUMNS"
    assert err.param == "columns"
    assert err.got == ["foo", "bar"]
    assert isinstance(err.extra, dict) and "available_preview" in err.extra

    s = err.text()
    assert "Column(s) not found" in s
    assert "foo" in s and "bar" in s

    html = err.html()
    assert "Param:" in html and "columns" in html


def test_sample_too_large_ctor():
    err = DimensionError.sample_too_large(where="sample.show_data", n=100, available_rows=42)
    assert err.code == "SAMPLE_TOO_LARGE"
    assert err.param == "n"
    assert err.expected == "n ≤ 42"
    assert err.got == 100
    s = err.summary()
    assert "Sampling failed" in s and "42" in s


@pytest.mark.optional
def test_rich_console_if_available():
    try:
        from rich.console import Console
    except Exception:
        pytest.skip("rich not installed")
    err = DimensionError.invalid_n(where="sample.show_data", got=-1)
    buf = io.StringIO()
    Console(file=buf, width=80, force_terminal=True, color_system=None).print(err)
    out = buf.getvalue()
    assert "Invalid row count" in out or "❌ Invalid row count" in out
    assert "INVALID_N" in out
