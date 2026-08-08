from svy.core.warnings import Severity, SvyWarning, SvyWarningsError, WarnCode
from svy.errors.base_errors import SvyError


def test_svy_warnings_error_defaults_code():
    err = SvyWarningsError(title="Warnings escalated", detail="Two warnings were escalated.")
    assert isinstance(err, SvyError)
    assert err.code in {"SVY_WARNINGS_ERROR", "SVY_ERROR"}
    s = str(err)
    assert "Warnings escalated" in s
    assert "[" in s and "]" in s


def test_explicit_code_is_preserved():
    err = SvyWarningsError(
        title="Warnings escalated",
        detail="Escalated with a custom code.",
        code="CUSTOM_ESCALATION",
    )
    assert err.code == "CUSTOM_ESCALATION"


def test_from_warnings_ctor():
    warnings = [
        SvyWarning(
            code=WarnCode.SINGLETON_PSU,
            title="Singleton PSU",
            detail="Stratum 'A' has a single PSU.",
            where="design",
            level=Severity.ERROR,
        ),
        SvyWarning(
            code=WarnCode.ZERO_WEIGHT,
            title="Zero weight",
            detail="3 rows have zero weight.",
            level=Severity.WARNING,
        ),
    ]
    err = SvyWarningsError.from_warnings(warnings, where="estimation")

    # shape
    assert err.code == "SVY_WARNINGS_ERROR"
    assert err.where == "estimation"
    assert err.title == "2 warning(s) escalated"
    assert isinstance(err.extra, dict)
    assert len(err.extra["warnings"]) == 2

    # detail lists each warning with level and code
    d = err.detail
    assert d.startswith("Escalated warnings:")
    assert "ERROR SINGLETON_PSU at design: Singleton PSU" in d
    assert "WARNING ZERO_WEIGHT: Zero weight" in d


def test_from_warnings_empty():
    err = SvyWarningsError.from_warnings([])
    assert err.code == "SVY_WARNINGS_ERROR"
    assert err.title == "No warnings to escalate"
