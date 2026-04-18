# tests/datasets/test_types.py
"""Tests for ``svy.datasets.types.Dataset``.

These tests are about the typed struct itself — decoding, frozenness,
equality, summary formatting.  They bypass the backend adapter entirely
by constructing ``Dataset`` values directly and encoding/decoding to JSON.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from svy.datasets.types import Dataset


# --------------------------------------------------------------------------- #
# Canonical Dataset payload (typed-shape, not backend-shape)
# --------------------------------------------------------------------------- #


def _typed_payload(**overrides: Any) -> bytes:
    """Build a typed-shape Dataset JSON payload suitable for msgspec.decode."""
    base: dict[str, Any] = {
        "slug": "toy",
        "title": "Toy",
        "description": "Fixture dataset.",
        "version": "1.0.0",
        "download_url": "http://x/y.parquet",
        "sha256": "",
        "size_bytes": 0,
        "n_rows": 0,
        "n_cols": 0,
        "license": "CC-BY-4.0",
        "citation": "n/a",
        "source": "fixture",
        "design": None,
        "variables": {},
        "tags": [],
    }
    base.update(overrides)
    return msgspec.json.encode(base)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


class TestDatasetDecoding:
    def test_minimal_payload_decodes(self):
        """Only required fields present; optionals take defaults."""
        raw = msgspec.json.encode(
            {
                "slug": "toy",
                "title": "Toy",
                "description": "d",
                "version": "1.0",
                "download_url": "http://x/y.parquet",
                "sha256": "",
                "size_bytes": 0,
                "n_rows": 0,
                "n_cols": 0,
                "license": "MIT",
                "citation": "n/a",
                "source": "n/a",
            }
        )
        ds = msgspec.json.decode(raw, type=Dataset)
        assert ds.slug == "toy"
        assert ds.design is None
        assert ds.variables == {}
        assert ds.tags == ()

    def test_full_payload_decodes(self):
        raw = _typed_payload(
            slug="phia_like",
            design={"stratum": ["urban", "region"], "psu": "ea", "wgt": "wt"},
            variables={"age": {"label": "Age"}, "sex": {"label": "Sex"}},
            tags=["health", "household"],
        )
        ds = msgspec.json.decode(raw, type=Dataset)
        assert ds.slug == "phia_like"
        assert ds.design == {"stratum": ["urban", "region"], "psu": "ea", "wgt": "wt"}
        assert "age" in ds.variables
        assert ds.tags == ("health", "household")

    def test_round_trip_is_lossless(self):
        raw = _typed_payload(
            design={"stratum": "region", "psu": "ea", "wgt": "wt"},
            tags=["a", "b", "c"],
        )
        ds1 = msgspec.json.decode(raw, type=Dataset)
        ds2 = msgspec.json.decode(msgspec.json.encode(ds1), type=Dataset)
        assert ds1 == ds2

    def test_missing_required_field_raises(self):
        """``sha256`` is required even if empty; drop it entirely."""
        payload = {
            "slug": "toy",
            "title": "Toy",
            "description": "d",
            "version": "1.0",
            "download_url": "http://x/y.parquet",
            # sha256 missing
            "size_bytes": 0,
            "n_rows": 0,
            "n_cols": 0,
            "license": "MIT",
            "citation": "n/a",
            "source": "n/a",
        }
        raw = msgspec.json.encode(payload)
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(raw, type=Dataset)

    def test_wrong_type_raises(self):
        raw = _typed_payload(n_rows="not-an-int")  # type: ignore[arg-type]
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(raw, type=Dataset)


# --------------------------------------------------------------------------- #
# Frozen / immutability
# --------------------------------------------------------------------------- #


class TestDatasetImmutable:
    def test_cannot_mutate_slug(self):
        ds = msgspec.json.decode(_typed_payload(), type=Dataset)
        with pytest.raises(AttributeError):
            ds.slug = "other"

    def test_equal_datasets_compare_equal(self):
        raw = _typed_payload(slug="a")
        ds1 = msgspec.json.decode(raw, type=Dataset)
        ds2 = msgspec.json.decode(raw, type=Dataset)
        assert ds1 == ds2
        # Note: not hashable because `variables` / `design` are dicts.
        # That's fine — we rarely need to use Datasets as dict keys.


# --------------------------------------------------------------------------- #
# Design spread-usage: the intended integration pattern
# --------------------------------------------------------------------------- #


class TestDesignSpread:
    """
    The raison d'être of keeping ``design`` as a plain dict: users should be
    able to spread it directly into ``svy.Design(**ds.design)``.
    """

    def test_design_is_dict_like(self):
        ds = msgspec.json.decode(
            _typed_payload(design={"stratum": "region", "psu": "ea", "wgt": "wt"}),
            type=Dataset,
        )
        assert ds.design is not None
        kwargs = {**ds.design}
        assert kwargs == {"stratum": "region", "psu": "ea", "wgt": "wt"}

    def test_design_none_when_absent(self):
        ds = msgspec.json.decode(_typed_payload(design=None), type=Dataset)
        assert ds.design is None


# --------------------------------------------------------------------------- #
# Summary formatting
# --------------------------------------------------------------------------- #


class TestSummary:
    def test_summary_contains_key_fields(self):
        ds = msgspec.json.decode(
            _typed_payload(slug="phia_like", n_rows=12500, size_bytes=5 * 1024 * 1024),
            type=Dataset,
        )
        s = ds.summary()
        assert "phia_like" in s
        assert "12,500" in s
        assert "5.0 MB" in s

    def test_summary_single_line(self):
        ds = msgspec.json.decode(_typed_payload(), type=Dataset)
        assert "\n" not in ds.summary()
