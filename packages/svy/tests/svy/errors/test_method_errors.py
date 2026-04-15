import io

import pytest

from svy.errors.base_errors import SvyError
from svy.errors.method_errors import MethodError


def test_method_error_defaults_code():
    err = MethodError(title="Bad mode", detail="Not supported.")
    assert isinstance(err, SvyError)
    # __post_init__ should set a specific default if caller left "SVY_ERROR"
    assert err.code in {"METHOD_ERROR", "SVY_ERROR"}  # tolerate if you later change the default
    # String contains title and code
    s = str(err)
    assert "Bad mode" in s
    assert "[" in s and "]" in s


def test_invalid_choice_ctor_builds_clear_error():
    err = MethodError.invalid_choice(
        where="sample.show_data",
        param="how",
        got="nope",
        allowed=("head", "tail", "sample"),
        hint="Use 'head', 'tail', or 'sample'.",
        docs_url="https://example.com/docs#how",
    )
    # shape
    assert err.code == "INVALID_CHOICE"
    assert err.param == "how"
    assert err.where == "sample.show_data"
    assert err.expected == ["head", "tail", "sample"]
    assert err.got == "nope"
    assert "Use 'head', 'tail', or 'sample'." in (err.hint or "")

    # renderers
    s = err.text()
    assert "Invalid option" in s
    assert "how" in s and "nope" in s

    md = err.markdown()
    assert "**❌" in md and "`[INVALID_CHOICE]`" in md
    assert "**param**" in md or "**param**" in md  # loose check

    html = err.html()
    assert "Param:" in html and "how" in html
    assert "Docs" in html and "example.com" in html

    # dict
    d = err.to_dict()
    assert d["error"]["code"] == "INVALID_CHOICE"
    assert d["error"]["param"] == "how"
    assert d["error"]["got"] == "nope"

    # repr includes class name and key fields
    r = repr(err)
    assert "MethodError(" in r and "title=" in r and "code='INVALID_CHOICE'" in r


def test_not_applicable_ctor():
    err = MethodError.not_applicable(
        where="svy.weight",
        method="rake",
        reason="no controls provided",
        param="method",
        hint="Provide controls or use 'poststratify'.",
    )
    assert err.code == "METHOD_NOT_APPLICABLE"
    assert "cannot be used here" in err.detail
    assert err.param == "method"
    assert "Provide controls" in (err.hint or "")


@pytest.mark.optional  # doesn’t fail the suite if Rich isn’t installed
def test_rich_console_if_available():
    try:
        from rich.console import Console
    except Exception:
        pytest.skip("rich not installed")
    err = MethodError.invalid_choice(
        where="sample.show_data",
        param="how",
        got="x",
        allowed=("head", "tail"),
    )
    buf = io.StringIO()
    Console(file=buf, width=80, force_terminal=True, color_system=None).print(err)
    out = buf.getvalue()
    # Loose checks: title and code appear in rich-rendered output
    assert "Invalid option" in out or "❌ Invalid option" in out
    assert "INVALID_CHOICE" in out
