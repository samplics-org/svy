# tests/svy/selection/test_normalize_n.py
"""
Unit tests for _normalize_n_for_groups and _coerce_key.

These are pure-function tests -- no Sample, no Polars, no engine.
Every behaviour of the n-normalization logic is covered here so that
integration tests (test_srs.py, test_pps.py) only need to verify the
selection call itself, not the n-resolution logic.

Coverage map
------------
_coerce_key
    string key passes through unchanged
    int key coerced to str
    tuple key joined with __by__
    nested tuple (unusual but safe)

Scalar n
    ungrouped (G=[])          -> plain int
    stratum only              -> broadcast to all strata
    stratum + by              -> broadcast to all cells
    by only (no stratum)      -> broadcast to all by-levels

Dict n -- ungrouped (Problem 1 -- must error)
    any dict with G=[]        -> MethodError

Dict n -- exact G match
    all cells covered         -> pass-through
    all cells, some zero      -> pass-through (explicit zero allowed)

Dict n -- exact B match (by-level broadcast)
    all by-levels, with stratum -> correct cross-product
    all by-levels, no stratum   -> direct pass-through
    INCOMPLETE by-levels        -> MethodError (Problem 2)

Dict n -- exact S match (stratum broadcast)
    all strata, with by       -> correct cross-product
    all strata, no by         -> direct pass-through
    INCOMPLETE strata         -> MethodError (Problem 2)

Dict n -- sublevel component match
    keys match one component per cell, all cells covered -> correct map
    one cell has zero matching keys (incomplete)         -> MethodError
    one cell has two matching keys (ambiguous)           -> MethodError
    unrecognized keys (match nothing)                    -> MethodError

Dict n -- tuple keys
    S-level tuple keys, all strata                       -> correct map
    B-level tuple keys (single-element tuples)           -> correct map
    G-level tuple keys for multi-col stratum             -> correct map
    incomplete tuple S-keys                              -> MethodError

Dict n -- mixed errors
    partial G (some cells missing)                       -> MethodError
    partial S (some strata missing)                      -> MethodError
    partial B (some by-levels missing)                   -> MethodError
    extra unknown keys                                   -> MethodError
"""

from __future__ import annotations

import pytest

from svy.errors import MethodError
from svy.selection._group_keys import _coerce_key, _normalize_n_for_groups


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Ungrouped
UG = dict(G=[], B=[], S=[])

# Stratum only: region = North / South
SO = dict(G=["North", "South"], B=[], S=["North", "South"])

# By only: area = urban / rural (no design stratum)
BO = dict(G=["urban", "rural"], B=["urban", "rural"], S=[])

# Stratum + by: region x area
SB = dict(
    G=["North__by__urban", "North__by__rural", "South__by__urban", "South__by__rural"],
    B=["urban", "rural"],
    S=["North", "South"],
)

# Multi-column stratum (province x zone) + by area
# S values are already __by__-joined composite stratum labels
MC = dict(
    G=[
        "Dakar__by__Zone1__by__urban",
        "Dakar__by__Zone1__by__rural",
        "Dakar__by__Zone2__by__urban",
        "Dakar__by__Zone2__by__rural",
        "Thies__by__Zone1__by__urban",
        "Thies__by__Zone1__by__rural",
    ],
    B=["urban", "rural"],
    S=["Dakar__by__Zone1", "Dakar__by__Zone2", "Thies__by__Zone1"],
)


def nn(n, *, G, B, S):
    """Shorthand."""
    return _normalize_n_for_groups(n, G=G, B=B, S=S)


# ---------------------------------------------------------------------------
# _coerce_key
# ---------------------------------------------------------------------------


class TestCoerceKey:
    def test_string_passthrough(self):
        assert _coerce_key("North") == "North"

    def test_int_to_str(self):
        assert _coerce_key(1) == "1"

    def test_tuple_two_parts(self):
        assert _coerce_key(("Dakar", "Zone1")) == "Dakar__by__Zone1"

    def test_tuple_three_parts(self):
        assert _coerce_key(("A", "B", "C")) == "A__by__B__by__C"

    def test_tuple_single_part(self):
        assert _coerce_key(("urban",)) == "urban"

    def test_tuple_with_int_parts(self):
        assert _coerce_key((1, 2)) == "1__by__2"


# ---------------------------------------------------------------------------
# Scalar n
# ---------------------------------------------------------------------------


class TestScalarN:
    def test_ungrouped_returns_int(self):
        assert nn(10, **UG) == 10

    def test_ungrouped_float_coerced(self):
        assert nn(5.0, **UG) == 5

    def test_stratum_only_broadcasts(self):
        result = nn(10, **SO)
        assert result == {"North": 10, "South": 10}

    def test_by_only_broadcasts(self):
        result = nn(5, **BO)
        assert result == {"urban": 5, "rural": 5}

    def test_stratum_and_by_broadcasts_to_all_cells(self):
        result = nn(3, **SB)
        assert result == {
            "North__by__urban": 3,
            "North__by__rural": 3,
            "South__by__urban": 3,
            "South__by__rural": 3,
        }

    def test_zero_scalar_broadcasts(self):
        result = nn(0, **SO)
        assert result == {"North": 0, "South": 0}


# ---------------------------------------------------------------------------
# Dict n -- ungrouped (Problem 1)
# ---------------------------------------------------------------------------


class TestDictUngrouped:
    def test_dict_with_no_grouping_raises(self):
        with pytest.raises(MethodError):
            nn({"A": 3, "B": 5}, **UG)

    def test_dict_single_key_no_grouping_raises(self):
        with pytest.raises(MethodError):
            nn({"North": 5}, **UG)

    def test_dict_matching_nothing_no_grouping_raises(self):
        with pytest.raises(MethodError):
            nn({"ZZZ": 10}, **UG)

    def test_error_message_mentions_scalar(self):
        with pytest.raises(MethodError) as exc:
            nn({"A": 5}, **UG)
        assert "scalar" in str(exc.value).lower() or "grouping" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Dict n -- exact G match
# ---------------------------------------------------------------------------


class TestExactGMatch:
    def test_all_cells_passthrough(self):
        n = {
            "North__by__urban": 5,
            "North__by__rural": 3,
            "South__by__urban": 4,
            "South__by__rural": 2,
        }
        assert nn(n, **SB) == n

    def test_all_cells_with_explicit_zero(self):
        n = {
            "North__by__urban": 5,
            "North__by__rural": 0,
            "South__by__urban": 4,
            "South__by__rural": 0,
        }
        result = nn(n, **SB)
        assert result["North__by__rural"] == 0
        assert result["South__by__rural"] == 0
        assert result["North__by__urban"] == 5

    def test_stratum_only_exact(self):
        n = {"North": 5, "South": 3}
        assert nn(n, **SO) == n

    def test_by_only_exact(self):
        n = {"urban": 10, "rural": 7}
        assert nn(n, **BO) == n


# ---------------------------------------------------------------------------
# Dict n -- exact B match (by-level broadcast)
# ---------------------------------------------------------------------------


class TestExactBMatch:
    def test_complete_b_keys_with_stratum_broadcasts(self):
        result = nn({"urban": 10, "rural": 15}, **SB)
        assert result == {
            "North__by__urban": 10,
            "North__by__rural": 15,
            "South__by__urban": 10,
            "South__by__rural": 15,
        }

    def test_complete_b_keys_no_stratum_passthrough(self):
        result = nn({"urban": 10, "rural": 15}, **BO)
        assert result == {"urban": 10, "rural": 15}

    def test_complete_b_with_explicit_zero(self):
        result = nn({"urban": 10, "rural": 0}, **SB)
        assert result["North__by__rural"] == 0
        assert result["South__by__rural"] == 0
        assert result["North__by__urban"] == 10

    def test_incomplete_b_keys_raises(self):
        """Passing only 'urban' missing 'rural' -> MethodError (Problem 2)."""
        with pytest.raises(MethodError):
            nn({"urban": 10}, **SB)

    def test_incomplete_b_single_key_raises(self):
        with pytest.raises(MethodError):
            nn({"rural": 5}, **SB)

    def test_incomplete_b_error_mentions_missing(self):
        with pytest.raises(MethodError) as exc:
            nn({"urban": 10}, **SB)
        msg = str(exc.value)
        assert "rural" in msg or "missing" in msg.lower() or "incomplete" in msg.lower()


# ---------------------------------------------------------------------------
# Dict n -- exact S match (stratum broadcast)
# ---------------------------------------------------------------------------


class TestExactSMatch:
    def test_complete_s_keys_with_by_broadcasts(self):
        result = nn({"North": 3, "South": 5}, **SB)
        assert result == {
            "North__by__urban": 3,
            "North__by__rural": 3,
            "South__by__urban": 5,
            "South__by__rural": 5,
        }

    def test_complete_s_keys_no_by_passthrough(self):
        result = nn({"North": 3, "South": 5}, **SO)
        assert result == {"North": 3, "South": 5}

    def test_complete_s_with_explicit_zero(self):
        result = nn({"North": 5, "South": 0}, **SB)
        assert result["South__by__urban"] == 0
        assert result["South__by__rural"] == 0
        assert result["North__by__urban"] == 5

    def test_incomplete_s_keys_raises(self):
        """Passing only 'North' missing 'South' -> MethodError (Problem 2)."""
        with pytest.raises(MethodError):
            nn({"North": 5}, **SB)

    def test_incomplete_s_single_key_raises(self):
        with pytest.raises(MethodError):
            nn({"South": 3}, **SO)

    def test_incomplete_s_error_mentions_missing(self):
        with pytest.raises(MethodError) as exc:
            nn({"North": 5}, **SB)
        msg = str(exc.value)
        assert "South" in msg or "missing" in msg.lower() or "incomplete" in msg.lower()


# ---------------------------------------------------------------------------
# Dict n -- sublevel component match
# ---------------------------------------------------------------------------


class TestSublevelMatch:
    def test_all_cells_matched_by_component(self):
        """Each cell has exactly one matching key component."""
        result = nn({"urban": 10, "rural": 15}, **SB)
        # This is also the B-match path; verify sublevel gives same result
        assert result["North__by__urban"] == 10
        assert result["North__by__rural"] == 15

    def test_incomplete_coverage_raises(self):
        """One G-cell has no matching key -> should raise."""
        with pytest.raises(MethodError):
            nn({"urban": 10}, **SB)

    def test_ambiguous_keys_raises(self):
        """Keys match multiple components of the same cell -> MethodError."""
        # "North__by__urban" contains both "North" and "urban"
        # -> ambiguous which value to use
        with pytest.raises(MethodError):
            nn({"North": 3, "urban": 10}, **SB)

    def test_unrecognized_keys_raises(self):
        with pytest.raises(MethodError):
            nn({"ZZZ": 5, "YYY": 3}, **SB)

    def test_partially_recognized_raises(self):
        """One key is valid, other is not -> MethodError."""
        with pytest.raises(MethodError):
            nn({"North": 3, "ZZZ": 5}, **SB)


# ---------------------------------------------------------------------------
# Tuple key support
# ---------------------------------------------------------------------------


class TestTupleKeys:
    def test_s_level_tuple_keys_complete(self):
        """Tuple keys for multi-column stratum, all strata covered."""
        result = nn(
            {("Dakar", "Zone1"): 5, ("Dakar", "Zone2"): 3, ("Thies", "Zone1"): 4},
            **MC,
        )
        assert result["Dakar__by__Zone1__by__urban"] == 5
        assert result["Dakar__by__Zone1__by__rural"] == 5
        assert result["Dakar__by__Zone2__by__urban"] == 3
        assert result["Thies__by__Zone1__by__urban"] == 4

    def test_s_level_tuple_keys_explicit_zero(self):
        result = nn(
            {("Dakar", "Zone1"): 5, ("Dakar", "Zone2"): 0, ("Thies", "Zone1"): 4},
            **MC,
        )
        assert result["Dakar__by__Zone2__by__urban"] == 0
        assert result["Dakar__by__Zone2__by__rural"] == 0

    def test_incomplete_tuple_s_keys_raises(self):
        """Missing one tuple stratum key -> MethodError."""
        with pytest.raises(MethodError):
            nn(
                {("Dakar", "Zone1"): 5, ("Dakar", "Zone2"): 3},  # Thies missing
                **MC,
            )

    def test_b_level_single_element_tuples(self):
        """Single-element tuple keys for by-levels."""
        result = nn({("urban",): 10, ("rural",): 15}, **SB)
        assert result["North__by__urban"] == 10
        assert result["South__by__rural"] == 15

    def test_mixed_tuple_and_string_keys(self):
        """Mixing tuple and string keys for the same level is valid."""
        result = nn({("Dakar", "Zone1"): 5, "Dakar__by__Zone2": 3, ("Thies", "Zone1"): 4}, **MC)
        assert result["Dakar__by__Zone1__by__urban"] == 5
        assert result["Dakar__by__Zone2__by__rural"] == 3

    def test_tuple_keys_unrecognized_raises(self):
        with pytest.raises(MethodError):
            nn({("Unknown", "Zone1"): 5}, **MC)


# ---------------------------------------------------------------------------
# Partial / incomplete coverage (Problem 2 consolidated)
# ---------------------------------------------------------------------------


class TestPartialCoverage:
    def test_partial_G_raises(self):
        """Subset of full composite keys -> MethodError."""
        partial = {k: 5 for k in list(SB["G"])[:2]}
        with pytest.raises(MethodError):
            nn(partial, **SB)

    def test_partial_S_raises(self):
        with pytest.raises(MethodError):
            nn({"North": 5}, **SO)

    def test_partial_B_raises(self):
        with pytest.raises(MethodError):
            nn({"urban": 10}, **BO)

    def test_partial_S_in_stratum_by_design_raises(self):
        with pytest.raises(MethodError):
            nn({"North": 5}, **SB)

    def test_partial_B_in_stratum_by_design_raises(self):
        with pytest.raises(MethodError):
            nn({"urban": 10}, **SB)

    def test_explicit_zero_is_not_partial(self):
        """Providing all keys with some set to zero is valid."""
        result = nn({"North": 5, "South": 0}, **SO)
        assert result == {"North": 5, "South": 0}

    def test_all_zeros_is_valid(self):
        """All-zero dict is valid -- draws nothing from any group."""
        result = nn({"North": 0, "South": 0}, **SO)
        assert result == {"North": 0, "South": 0}

    def test_extra_key_raises(self):
        """An extra unrecognized key -> MethodError."""
        with pytest.raises(MethodError):
            nn({"North": 5, "South": 3, "East": 2}, **SO)

    def test_error_is_method_error(self):
        """Errors must be MethodError, not bare ValueError/KeyError."""
        with pytest.raises(MethodError):
            nn({"North": 5}, **SB)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_stratum_complete(self):
        G = ["North"]
        result = nn({"North": 5}, G=G, B=[], S=["North"])
        assert result == {"North": 5}

    def test_single_by_level_complete(self):
        G = ["urban"]
        result = nn({"urban": 5}, G=G, B=["urban"], S=[])
        assert result == {"urban": 5}

    def test_single_cell_stratum_by(self):
        G = ["North__by__urban"]
        result = nn({"North__by__urban": 5}, G=G, B=["urban"], S=["North"])
        assert result == {"North__by__urban": 5}

    def test_many_strata_all_required(self):
        provinces = ["Dakar", "Thies", "Ziguinchor", "Kaolack", "Diourbel"]
        G = provinces[:]
        S = provinces[:]
        # Missing one province raises
        partial = {p: 5 for p in provinces[:-1]}
        with pytest.raises(MethodError):
            nn(partial, G=G, B=[], S=S)
        # All present is fine
        complete = {p: 5 for p in provinces}
        result = nn(complete, G=G, B=[], S=S)
        assert len(result) == len(provinces)

    def test_scalar_zero_is_valid(self):
        result = nn(0, **SB)
        assert all(v == 0 for v in result.values())
