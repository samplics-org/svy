# tests/datasets/test_errors.py
"""Tests for ``svy.errors.dataset_errors.DatasetError``."""

from __future__ import annotations

import pytest

from svy.errors import SvyError
from svy.errors.dataset_errors import DatasetError


class TestDefaultCode:
    def test_bare_construction_has_dataset_error_code(self):
        e = DatasetError(title="x", detail="y")
        assert e.code == "DATASET_ERROR"

    def test_explicit_code_preserved(self):
        e = DatasetError(title="x", detail="y", code="CUSTOM")
        assert e.code == "CUSTOM"

    def test_inherits_svyerror(self):
        e = DatasetError(title="x", detail="y")
        assert isinstance(e, SvyError)
        assert isinstance(e, Exception)


class TestCatalogUnreachable:
    def test_code_and_fields(self):
        e = DatasetError.catalog_unreachable(where="w", url="http://u", reason="timeout")
        assert e.code == "CATALOG_UNREACHABLE"
        assert e.title == "Catalog unreachable"
        assert "http://u" in e.detail
        assert "timeout" in e.detail
        assert e.where == "w"
        assert e.extra == {"url": "http://u"}

    def test_custom_hint_is_used(self):
        e = DatasetError.catalog_unreachable(
            where="w", url="http://u", reason="timeout", hint="custom hint"
        )
        assert e.hint == "custom hint"


class TestCatalogBadStatus:
    def test_code_and_fields(self):
        e = DatasetError.catalog_bad_status(where="w", url="http://u", status=503)
        assert e.code == "CATALOG_BAD_STATUS"
        assert "503" in e.detail
        assert e.extra == {"url": "http://u", "status": 503}

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_any_status_encoded(self, status):
        e = DatasetError.catalog_bad_status(where="w", url="http://u", status=status)
        assert str(status) in e.detail


class TestNotFound:
    def test_code_and_fields(self):
        e = DatasetError.not_found(where="w", slug="missing")
        assert e.code == "DATASET_NOT_FOUND"
        assert "missing" in e.detail
        assert e.got == "missing"
        assert e.param == "name"

    def test_hint_mentions_catalog_function(self):
        e = DatasetError.not_found(where="w", slug="x")
        assert "catalog" in (e.hint or "")


class TestShaMismatch:
    def test_code_and_fields(self):
        e = DatasetError.sha_mismatch(where="w", slug="bad", expected="aa", got="bb")
        assert e.code == "DATASET_SHA_MISMATCH"
        assert e.expected == "aa"
        assert e.got == "bb"
        assert e.extra == {"slug": "bad"}


class TestCatching:
    """
    Library convention: one class, differentiate via `.code`.  These tests
    document the intended catch patterns so future refactors don't silently
    break users' try/except.
    """

    def test_catch_all_as_dataset_error(self):
        errors = [
            DatasetError.catalog_unreachable(where="x", url="http://u", reason="r"),
            DatasetError.not_found(where="x", slug="s"),
            DatasetError.sha_mismatch(where="x", slug="s", expected="a", got="b"),
        ]
        for e in errors:
            with pytest.raises(DatasetError):
                raise e

    def test_branch_on_code(self):
        e = DatasetError.not_found(where="x", slug="s")
        try:
            raise e
        except DatasetError as err:
            if err.code == "DATASET_NOT_FOUND":
                caught = "not_found"
            else:
                caught = "other"
        assert caught == "not_found"
