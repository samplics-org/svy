# tests/test_base_errors.py

from svy.errors.base_errors import SvyError


def _make_error(**kw) -> SvyError:
    defaults = dict(
        title="Invalid option <x>",
        detail="Parameter 'alpha' is out of range & must be 0 < alpha < 1.",
        code="SVY_ERROR",
        where="module.fn",
        param="alpha",
        expected="(0, 1)",
        got="2.5",
        hint="Use a value between 0 and 1.",
        docs_url="https://example.com/docs#alpha",
        extra={"context": "unit-test"},
    )
    defaults.update(kw)
    return SvyError(**defaults)


# ---------------------------
# text() and __str__
# ---------------------------


def test_text_default_includes_fields_and_padding():
    err = _make_error()
    s = err.text()  # default: indent=2, surround=True
    # surrounding blank lines
    assert s.startswith("\n") and s.endswith("\n")
    # padding of 2 spaces
    assert "\n  ❌ Invalid option <x> [SVY_ERROR]" in s
    assert "Parameter 'alpha' is out of range" in s
    assert "- where: module.fn" in s
    assert "- param: alpha" in s
    assert "- expected: (0, 1)" in s
    assert "- got: 2.5" in s
    assert "Hint: Use a value between 0 and 1." in s
    assert "Docs: https://example.com/docs#alpha" in s


def test_text_flags_indent_and_surround():
    err = _make_error()
    s = err.text(indent=0, surround=False)
    assert not s.startswith("\n") and not s.endswith("\n")
    assert s.startswith("❌ Invalid option <x> [SVY_ERROR]")  # no padding


def test_str_uses_text_defaults():
    err = _make_error()
    assert str(err) == err.text(indent=2, surround=True)


# ---------------------------
# __repr__ and summary()
# ---------------------------


def test_repr_contains_core_fields():
    err = _make_error()
    r = repr(err)
    assert r.startswith("SvyError(")
    assert "title='Invalid option <x>'" in r
    assert "code='SVY_ERROR'" in r
    assert "where='module.fn'" in r
    assert "param='alpha'" in r
    # extra small payload included
    assert "extra={'context': 'unit-test'}" in r


def test_summary_compact_contains_expected_and_got():
    err = _make_error()
    s = err.summary()
    assert "SVY_ERROR: Invalid option <x> — Parameter 'alpha'" in s
    assert "[at module.fn]" in s
    assert "[param alpha]" in s
    assert "[expected (0, 1)]" in s
    assert "[got 2.5]" in s
    assert "→ Use a value between 0 and 1." in s


def test_truncation_applies_in_summary_and_text_for_long_got():
    long = "x" * 200
    err = _make_error(got=long)
    s = err.summary()
    # should be truncated and end with ellipsis
    assert "x…" in s and len(s) < len(long) + 200  # sanity check
    t = err.text()
    assert "x…" in t


# ---------------------------
# ansi(), markdown(), html()
# ---------------------------


def test_ansi_contains_control_codes_and_core_fields():
    err = _make_error()
    a = err.ansi()
    # ANSI bold or color escape present
    assert "\033[" in a
    assert "❌ Invalid option <x>" in a
    assert "[SVY_ERROR]" in a
    assert "param" in a
    assert "expected" in a
    assert "got" in a


def test_markdown_structure_and_content():
    err = _make_error()
    md = err.markdown()
    # header
    assert md.splitlines()[0].startswith("**❌ Invalid option <x>** `[SVY_ERROR]`")
    # bullets exist
    assert "- **where**: `module.fn`" in md
    assert "- **param**: `alpha`" in md
    assert "- **expected**: `(0, 1)`" in md
    assert "- **got**: `2.5`" in md
    # hint and docs
    assert "> 💡 **Hint:** Use a value between 0 and 1." in md
    assert "[Docs](https://example.com/docs#alpha)" in md


def test_html_is_escaped_and_contains_core_fields():
    # detail & title with special chars to verify escaping
    err = _make_error(
        title='Bad <tag> & "quote"',
        detail='Need <min> & <max> & "q"',
        expected="<min,max>",
        got="<weird>",
    )
    h = err.html()
    # minimal structure sanity
    assert h.startswith('<div role="alert"')
    # escaped content in title & detail
    assert "Bad &lt;tag&gt; &amp; &quot;quote&quot;" in h
    assert "Need &lt;min&gt; &amp; &lt;max&gt; &amp; &quot;q&quot;" in h
    # labels & escaped values for expected/got should be present
    assert '<span class="text-slate-500">Expected:</span>' in h
    assert "&lt;min,max&gt;" in h
    assert '<span class="text-slate-500">Got:</span>' in h
    assert "&lt;weird&gt;" in h


# ---------------------------
# to_dict()
# ---------------------------


def test_to_dict_payload_and_truncation():
    long_got = "y" * 300
    err = _make_error(got=long_got, extra={"a": 1})
    d = err.to_dict()
    assert set(d.keys()) == {"error"}
    e = d["error"]
    # required fields present
    for k in [
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
    ]:
        assert k in e
    # truncation applied to 'got'
    assert e["got"].endswith("…")
    assert e["extra"] == {"a": 1}


def test_to_dict_handles_none_fields_cleanly():
    err = SvyError(title="X", detail="Y")
    d = err.to_dict()
    e = d["error"]
    assert e["code"] == "SVY_ERROR"
    assert e["where"] is None
    assert e["param"] is None
    assert e["expected"] is None
    assert e["got"] is None


def test_to_dict_truncates_non_none_got():
    err = SvyError(title="X", detail="Y", got="z" * 300)
    got_out = err.to_dict()["error"]["got"]
    assert got_out.endswith("…")
    assert len(got_out) < 300
