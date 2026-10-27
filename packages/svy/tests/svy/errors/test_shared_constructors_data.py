import io

import pytest

from svy.errors import DesignError
from svy.errors._base import SvyError


def test_area_error_defaults_code():
    err = DesignError(title="Bad shape", detail="Mismatch.")
    assert isinstance(err, SvyError)
    assert err.code == "DESIGN_ERROR"
    s = str(err)
    assert "Bad shape" in s


def test_missing_columns_ctor():
    err = DesignError.missing_columns(
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


@pytest.mark.optional
def test_rich_console_if_available():
    try:
        from rich.console import Console
    except Exception:
        pytest.skip("rich not installed")
    err = DesignError.missing_columns(where="sample.show_data", param="columns", missing=["z"])
    buf = io.StringIO()
    Console(file=buf, width=80, force_terminal=True, color_system=None).print(err)
    out = buf.getvalue()
    assert "Column(s) not found" in out
    assert "MISSING_COLUMNS" in out
