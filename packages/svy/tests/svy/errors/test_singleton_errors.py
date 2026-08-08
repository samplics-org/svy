from typing import Any, Mapping

import msgspec

from svy.errors.base_errors import SvyError
from svy.errors.singleton_errors import SingletonError


class _Info(msgspec.Struct):
    """Minimal stand-in matching the _SingletonInfoLike protocol."""

    stratum_key: str
    stratum_values: Mapping[str, Any] | None
    psu_key: str
    n_observations: int


def test_singleton_error_defaults_code():
    err = SingletonError(title="Singleton PSU", detail="Stratum has a single PSU.")
    assert isinstance(err, SvyError)
    assert err.code in {"SINGLETON_ERROR", "SVY_ERROR"}
    s = str(err)
    assert "Singleton PSU" in s
    assert "[" in s and "]" in s


def test_explicit_code_is_preserved():
    err = SingletonError(
        title="No valid merge targets",
        detail="No non-singleton strata available.",
        code="NO_MERGE_TARGETS",
    )
    assert err.code == "NO_MERGE_TARGETS"


def test_from_singletons_ctor():
    singles = [
        _Info(
            stratum_key="region=West",
            stratum_values={"region": "West"},
            psu_key="psu_7",
            n_observations=1,
        ),
        _Info(
            stratum_key="region=East",
            stratum_values=None,
            psu_key="psu_2",
            n_observations=3,
        ),
    ]
    err = SingletonError.from_singletons(singles, where="estimation")

    # shape
    assert err.code == "SINGLETON_ERROR"
    assert err.where == "estimation"
    assert err.title == "2 singleton PSU(s) detected"
    assert isinstance(err.extra, dict)
    assert err.extra["singletons"] == [msgspec.to_builtins(s) for s in singles]

    # detail lists each singleton and the remediation menu
    d = err.detail
    assert "region=West (PSU=psu_7, n=1)" in d
    assert "region=East (PSU=psu_2, n=3)" in d
    assert "sample.singleton.summary()" in d
    assert "sample.singleton.collapse()" in d

    # renderers
    s = err.text()
    assert "2 singleton PSU(s) detected" in s
    md = err.markdown()
    assert "`[SINGLETON_ERROR]`" in md


def test_from_singletons_truncates_after_five():
    singles = [
        _Info(
            stratum_key=f"stratum_{i}",
            stratum_values=None,
            psu_key=f"psu_{i}",
            n_observations=1,
        )
        for i in range(7)
    ]
    err = SingletonError.from_singletons(singles)

    assert err.where == "singleton_handling"
    assert err.title == "7 singleton PSU(s) detected"
    assert "... and 2 more" in err.detail
    # payload keeps everything even when the display truncates
    assert len(err.extra["singletons"]) == 7
