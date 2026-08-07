from svy.errors.base_errors import SvyError
from svy.errors.serialization_errors import SerializationError


def test_serialization_error_defaults_code():
    err = SerializationError(title="Bad payload", detail="Could not serialize.")
    assert isinstance(err, SvyError)
    assert err.code in {"SERIALIZATION_ERROR", "SVY_ERROR"}
    s = str(err)
    assert "Bad payload" in s
    assert "[" in s and "]" in s


def test_unsupported_type_ctor_builds_clear_error():
    err = SerializationError.unsupported_type(
        got_type="list",
        registered=("Estimate", "Table", "GLMFit"),
    )
    # shape
    assert err.code == "UNSUPPORTED_RESULT_TYPE"
    assert err.where == "serialize"
    assert err.got == "list"
    assert err.expected == ["Estimate", "Table", "GLMFit"]
    assert "result object" in (err.hint or "")

    # renderers
    s = err.text()
    assert "No serializer registered" in s
    assert "list" in s

    md = err.markdown()
    assert "**❌" in md and "`[UNSUPPORTED_RESULT_TYPE]`" in md

    # dict
    d = err.to_dict()
    assert d["error"]["code"] == "UNSUPPORTED_RESULT_TYPE"
    assert d["error"]["got"] == "list"

    # repr includes class name and key fields
    r = repr(err)
    assert "SerializationError(" in r and "code='UNSUPPORTED_RESULT_TYPE'" in r


def test_missing_kind_ctor():
    err = SerializationError.missing_kind()
    assert err.code == "PAYLOAD_MISSING_KIND"
    assert err.where == "from_json"
    assert err.param == "kind"
    assert "to_json" in (err.hint or "")


def test_unknown_kind_ctor():
    err = SerializationError.unknown_kind(
        kind="flux_capacitor",
        known=("estimate", "table"),
    )
    assert err.code == "PAYLOAD_UNKNOWN_KIND"
    assert err.where == "from_json"
    assert err.param == "kind"
    assert err.got == "flux_capacitor"
    assert err.expected == ["estimate", "table"]
    assert "newer svy" in (err.hint or "")
