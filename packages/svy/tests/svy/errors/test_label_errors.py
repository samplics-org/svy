from svy.errors.base_errors import SvyError
from svy.errors.label_errors import LabelError


def test_label_error_is_subclass_and_default_code_namespace():
    # When no explicit code is provided, __post_init__ should flip SVY_ERROR -> LABEL_ERROR
    e = LabelError(title="t", detail="d")
    assert isinstance(e, SvyError)
    assert e.code == "LABEL_ERROR"
    # sanity on base renderers still working
    assert "LABEL_ERROR" in e.summary()


def test_unknown_scheme_constructor():
    e = LabelError.unknown_scheme(where="labels.get", param="scheme_id", got="abc")
    assert e.code == "LABEL_UNKNOWN_SCHEME"
    assert e.title == "Scheme not found"
    assert e.where == "labels.get"
    assert e.param == "scheme_id"
    assert e.expected == "existing scheme id or concept"
    assert e.got == "abc"
    assert "catalog.list()" in (e.hint or "")


def test_scheme_exists_constructor():
    e = LabelError.scheme_exists(where="labels.register", scheme_id="g:en")
    assert e.code == "LABEL_SCHEME_EXISTS"
    assert e.title == "Scheme already exists"
    assert e.param == "id"
    assert e.expected == "unique id"
    assert e.got == "g:en"
    assert "overwrite=True" in (e.hint or "")


def test_invalid_missing_codes_constructor():
    e = LabelError.invalid_missing_codes(
        where="labels.validate", param="missing", not_in_mapping=[9, 99]
    )
    assert e.code == "LABEL_INVALID_MISSING_CODES"
    assert e.param == "missing"
    assert e.expected == "missing ⊆ mapping.keys()"
    assert e.got == [9, 99]
    assert "mapping" in (e.detail or "").lower()


def test_inconsistent_missing_kinds_constructor():
    e = LabelError.inconsistent_missing_kinds(where="labels.validate", offending_keys=[-1, -2])
    assert e.code == "LABEL_INCONSISTENT_MISSING_KINDS"
    assert e.param == "missing_kinds"
    assert e.expected == "keys(missing_kinds) ⊆ missing"
    assert e.got == [-1, -2]
    # hint present
    assert "Add these codes" in (e.hint or "")


def test_nan_key_forbidden_constructor():
    e = LabelError.nan_key_forbidden(where="labels.validate")
    assert e.code == "LABEL_NAN_KEY"
    assert e.param == "mapping"
    assert e.expected == "non-NaN keys"
    assert e.got == "NaN"
    assert "NaN" in e.title or "NaN" in e.detail


def test_invalid_locale_constructor():
    e = LabelError.invalid_locale(where="labels.pick", got="xx_YY")
    assert e.code == "LABEL_INVALID_LOCALE"
    assert e.param == "locale"
    assert "language tag" in (e.expected or "")
    assert e.got == "xx_YY"
    assert "BCP-47" in (e.hint or "")


def test_ambiguous_pick_constructor():
    e = LabelError.ambiguous_pick(
        where="labels.pick", concept="gender", candidates=["gender:en", "gender:fr"]
    )
    assert e.code == "LABEL_AMBIGUOUS_PICK"
    assert e.param == "concept"
    assert e.expected == "unique best match"
    assert e.got == ["gender:en", "gender:fr"]
    assert "Ambiguous" in e.title or "Multiple equally" in e.detail


def test_serialization_error_constructor():
    e = LabelError.serialization_error(
        where="labels.to_bytes", reason="sets not JSON-serializable", extra={"n": 2}
    )
    assert e.code == "LABEL_SERIALIZATION_ERROR"
    assert e.title.startswith("Label catalog serialization failed")
    assert e.where == "labels.to_bytes"
    assert "JSON" in e.detail
    assert e.extra == {"n": 2}
    # rendering still works
    d = e.to_dict()
    assert d["error"]["code"] == "LABEL_SERIALIZATION_ERROR"


def test_invalid_scheme_id_constructor():
    e = LabelError.invalid_scheme_id(where="labels.make_scheme", got="Bad ID")
    assert e.code == "LABEL_INVALID_ID"
    assert e.param == "id"
    assert "concept:locale" in (e.expected or "")
    assert e.got == "Bad ID"
    # summary should include code and title
    s = e.summary()
    assert "LABEL_INVALID_ID" in s and "Invalid scheme id" in s
