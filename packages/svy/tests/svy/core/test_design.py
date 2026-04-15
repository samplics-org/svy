# tests/svy/core/test_design.py
from __future__ import annotations

import pytest

from svy.core.design import Design, PopSize


# ---------- field groups ----------
# Fields that must be str | None
STR_ONLY_FIELDS = ("row_index", "wgt", "prob", "hit", "mos")

# Fields that accept str | Sequence[str] | None (normalized to tuple[str, ...] | None)
MULTI_COL_FIELDS = ("stratum", "psu", "ssu")

# All stringly fields (for a few generic tests)
STR_FIELDS = STR_ONLY_FIELDS + MULTI_COL_FIELDS


# ---------- construction & validation ----------


def test_init_defaults_and_repr_str():
    d = Design()
    # defaults
    assert d.row_index is None
    assert d.stratum is None
    assert d.wgt is None
    assert d.prob is None
    assert d.hit is None
    assert d.mos is None
    assert d.psu is None
    assert d.ssu is None
    assert d.pop_size is None
    assert d.wr is False

    # pretty string (now a Rich-styled panel) should include the class name and not crash
    s = str(d)  # pretty/console view
    assert "Design" in s

    # repr is concise and machine-friendly (starts with Design(...)
    r = repr(d)
    assert r.startswith("Design(")
    assert "rep_wgts=None" in r and "wr=True" not in r


@pytest.mark.parametrize("name", STR_FIELDS)
def test_init_rejects_empty_string(name: str):
    kwargs = {name: ""}
    with pytest.raises(ValueError, match=f"{name!r} must not be an empty string"):
        Design(**kwargs)


@pytest.mark.parametrize("name", STR_ONLY_FIELDS)
def test_init_rejects_non_string_type_str_only(name: str):
    kwargs = {name: 123}
    with pytest.raises(TypeError, match=f"{name!r} must be str \\| None"):
        Design(**kwargs)


@pytest.mark.parametrize("name", MULTI_COL_FIELDS)
def test_init_rejects_non_string_type_multi(name: str):
    kwargs = {name: 123}
    with pytest.raises(TypeError, match=f"'{name}' must be str \\| Sequence\\[str\\] \\| None"):
        Design(**kwargs)


@pytest.mark.parametrize("name", MULTI_COL_FIELDS)
def test_init_rejects_empty_sequence_multi(name: str):
    kwargs = {name: []}
    with pytest.raises(ValueError, match=f"'{name}' sequence must not be empty"):
        Design(**kwargs)


@pytest.mark.parametrize("name", MULTI_COL_FIELDS)
def test_init_rejects_non_string_items_multi(name: str):
    kwargs = {name: ["ok", 5]}  # type: ignore[list-item]
    with pytest.raises(TypeError, match=f"'{name}' items must be str; got int"):
        Design(**kwargs)


@pytest.mark.parametrize("name", MULTI_COL_FIELDS)
def test_init_rejects_empty_string_items_multi(name: str):
    kwargs = {name: ["ok", ""]}  # empty item not allowed
    with pytest.raises(ValueError, match=f"'{name}' items must not contain empty strings"):
        Design(**kwargs)


def test_init_rejects_non_bool_wr():
    with pytest.raises(TypeError, match="'wr' must be bool"):
        Design(wr="yes")  # type: ignore[arg-type]


def test_init_accepts_valid_values():
    d = Design(
        row_index="row_id",
        stratum="strat",
        wgt="w",
        prob="p",
        hit="h",
        mos="m",
        psu="psu_id",
        ssu="ssu_id",
        wr=True,
    )
    assert d.row_index == "row_id"
    assert d.stratum == "strat"
    assert d.psu == "psu_id"
    assert d.ssu == "ssu_id"
    assert d.wr is True


def test_init_accepts_list_and_tuple_multi():
    d1 = Design(stratum=["region", "sex"], psu=["region", "cluster"], ssu=("cluster", "id"))
    assert d1.stratum == ("region", "sex")
    assert d1.psu == ("region", "cluster")
    assert d1.ssu == ("cluster", "id")


# ---------- immutability (frozen after __init__) ----------


def test_frozen_setattr_raises():
    d = Design(wgt="w")
    with pytest.raises(AttributeError, match="Design is frozen"):
        d.wgt = "w2"  # type: ignore[misc]


def test_frozen_delattr_raises():
    d = Design(wgt="w")
    with pytest.raises(AttributeError, match="Design is frozen"):
        del d.wgt  # type: ignore[misc]


# ---------- update ----------


def test_update_returns_new_and_original_unchanged():
    d = Design(wgt="w", wr=True)
    d2 = d.update(wgt="w2", wr=False)

    # new object with changes
    assert isinstance(d2, Design)
    assert d2.wgt == "w2"
    assert d2.wr is False

    # original unchanged
    assert d.wgt == "w"
    assert d.wr is True

    # other fields still None
    for name in STR_FIELDS:
        if name != "wgt":
            assert getattr(d2, name) is None


def test_update_can_set_to_none():
    d = Design(wgt="w", psu="psu1")
    d2 = d.update(wgt=None)
    assert d2.wgt is None
    assert d2.psu == "psu1"  # untouched


def test_update_validation_errors():
    d = Design(wgt="w")
    with pytest.raises(TypeError, match="must be str \\| None"):
        d.update(wgt=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be an empty string"):
        d.update(wgt="")  # empty string invalid


# ---------- equality & hashing ----------


def test_eq_and_hash():
    a = Design(wgt="w", psu="p", wr=False)
    b = Design(wgt="w", psu="p", wr=False)
    c = Design(wgt="w", psu="p", wr=True)

    assert a == b
    assert hash(a) == hash(b)

    assert a != c
    # Hash collisions are theoretically possible but unlikely; this asserts typical behavior.
    assert hash(a) != hash(c)


# ---------- alias / basic properties ----------


def test_probs_alias_property():
    d = Design(prob="pcol")
    assert d.prob == "pcol"
    # ensure alias reflects updates
    d2 = d.update(prob="qcol")
    assert d2.prob == "qcol"


# ---------- specified_fields ----------


def test_specified_fields_default_ignore_wr():
    # wr is bool (ignored), set some string fields
    d = Design(wgt="w", psu="psu_id", ssu="ssu_id")
    names = d.specified_fields()
    # expect order to follow field declaration in _FIELDS
    assert names == ["w", "psu_id", "ssu_id"]


def test_specified_fields_with_ignore_cols():
    d = Design(row_index="r", stratum="s", wgt="w", psu="p")
    names = d.specified_fields(ignore_cols=["row_index", "psu", "not_a_field"])
    # "not_a_field" is ignored; result excludes row_index and psu
    assert names == ["s", "w"]


def test_specified_fields_flattens_multi_specs():
    d = Design(stratum=["region", "sex"], psu=("region", "cluster"), ssu=["cluster", "id"])
    names = d.specified_fields()
    for col in ("region", "sex", "cluster", "id"):
        assert names.count(col) == 1


def test_specified_fields_when_none_or_missing():
    d = Design()  # all None
    assert d.specified_fields() == []

    d2 = Design(hit="h")
    assert d2.specified_fields() == ["h"]


# ---------- PopSize & pop_size field ----------


class TestPopSize:
    """Tests for the PopSize NamedTuple and its integration with Design."""

    # -- PopSize construction --

    def test_popsize_namedtuple_fields(self):
        ps = PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
        assert ps.psu == "TOTAL_HOSP"
        assert ps.ssu == "TOTAL_DISCHARGES"

    def test_popsize_unpacking(self):
        ps = PopSize(psu="N_PSU", ssu="N_UNIT")
        n_psu, n_unit = ps
        assert n_psu == "N_PSU"
        assert n_unit == "N_UNIT"

    def test_popsize_equality_and_hash(self):
        a = PopSize(psu="A", ssu="B")
        b = PopSize(psu="A", ssu="B")
        c = PopSize(psu="A", ssu="C")
        assert a == b
        assert hash(a) == hash(b)
        assert a != c

    def test_popsize_repr(self):
        ps = PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
        r = repr(ps)
        assert "TOTAL_HOSP" in r
        assert "TOTAL_DISCHARGES" in r

    # -- Design with pop_size as str (single-stage, backwards compat) --

    def test_design_pop_size_str(self):
        d = Design(pop_size="N_h")
        assert d.pop_size == "N_h"
        assert isinstance(d.pop_size, str)

    def test_design_pop_size_str_in_specified_fields(self):
        d = Design(pop_size="N_h", wgt="w")
        names = d.specified_fields()
        assert "N_h" in names
        assert "w" in names

    # -- Design with pop_size as PopSize (multistage) --

    def test_design_pop_size_popsize(self):
        ps = PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
        d = Design(pop_size=ps)
        assert d.pop_size == ps
        assert isinstance(d.pop_size, PopSize)
        assert d.pop_size.psu == "TOTAL_HOSP"
        assert d.pop_size.ssu == "TOTAL_DISCHARGES"

    def test_design_pop_size_popsize_in_specified_fields(self):
        ps = PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
        d = Design(pop_size=ps, wgt="w")
        names = d.specified_fields()
        assert "TOTAL_HOSP" in names
        assert "TOTAL_DISCHARGES" in names
        assert "w" in names

    def test_design_pop_size_popsize_deduplicates(self):
        """If psu and ssu reference the same column, it should appear once."""
        ps = PopSize(psu="N", ssu="N")
        d = Design(pop_size=ps)
        names = d.specified_fields()
        assert names.count("N") == 1

    # -- Design with pop_size as None --

    def test_design_pop_size_none(self):
        d = Design(pop_size=None)
        assert d.pop_size is None

    # -- Validation --

    def test_design_pop_size_rejects_empty_string(self):
        with pytest.raises(ValueError, match="'pop_size' must not be an empty string"):
            Design(pop_size="")

    def test_design_pop_size_rejects_invalid_type(self):
        with pytest.raises(TypeError, match="'pop_size' must be str \\| PopSize \\| None"):
            Design(pop_size=123)  # type: ignore[arg-type]

    def test_design_pop_size_rejects_plain_tuple(self):
        """A plain tuple of strings is not a PopSize — should be rejected."""
        with pytest.raises(TypeError, match="'pop_size' must be str \\| PopSize \\| None"):
            Design(pop_size=("N_PSU", "N_UNIT"))  # type: ignore[arg-type]

    def test_popsize_rejects_empty_psu(self):
        with pytest.raises(ValueError, match="PopSize.psu must be a non-empty string"):
            Design(pop_size=PopSize(psu="", ssu="N"))

    def test_popsize_rejects_empty_ssu(self):
        with pytest.raises(ValueError, match="PopSize.ssu must be a non-empty string"):
            Design(pop_size=PopSize(psu="N", ssu=""))

    # -- Update & fill_missing --

    def test_update_pop_size_str_to_popsize(self):
        d = Design(pop_size="N_h")
        ps = PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
        d2 = d.update(pop_size=ps)
        assert d2.pop_size == ps
        # original unchanged
        assert d.pop_size == "N_h"

    def test_update_pop_size_popsize_to_none(self):
        ps = PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
        d = Design(pop_size=ps)
        d2 = d.update(pop_size=None)
        assert d2.pop_size is None

    def test_update_pop_size_popsize_to_str(self):
        ps = PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
        d = Design(pop_size=ps)
        d2 = d.update(pop_size="N_h")
        assert d2.pop_size == "N_h"

    def test_fill_missing_pop_size_none_to_popsize(self):
        d = Design(pop_size=None)
        ps = PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
        d2 = d.fill_missing(pop_size=ps)
        assert d2.pop_size == ps

    def test_fill_missing_pop_size_preserves_existing(self):
        ps = PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
        d = Design(pop_size=ps)
        d2 = d.fill_missing(pop_size="N_h")
        # fill_missing should NOT overwrite existing
        assert d2.pop_size == ps

    # -- Display --

    def test_display_pop_size_str(self):
        d = Design(pop_size="N_h")
        plain = d.__plain_str__()
        assert "N_h" in plain

    def test_display_pop_size_popsize(self):
        ps = PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
        d = Design(pop_size=ps)
        plain = d.__plain_str__()
        assert "TOTAL_HOSP" in plain
        assert "TOTAL_DISCHARGES" in plain

    def test_display_pop_size_none(self):
        d = Design(pop_size=None)
        plain = d.__plain_str__()
        assert "None" in plain

    # -- Equality with PopSize --

    def test_eq_with_pop_size(self):
        ps = PopSize(psu="A", ssu="B")
        a = Design(pop_size=ps, wgt="w")
        b = Design(pop_size=PopSize(psu="A", ssu="B"), wgt="w")
        c = Design(pop_size="A", wgt="w")
        assert a == b
        assert hash(a) == hash(b)
        assert a != c

    # -- NIS-style usage example --

    def test_nis_style_design(self):
        """Integration test: NIS-style two-stage design with FPC."""
        d = Design(
            stratum="NIS_STRATUM",
            psu="HOESSION",
            wgt="DISCWT",
            pop_size=PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES"),
        )
        assert d.stratum == "NIS_STRATUM"
        assert d.psu == "HOESSION"
        assert d.wgt == "DISCWT"
        assert isinstance(d.pop_size, PopSize)
        assert d.pop_size.psu == "TOTAL_HOSP"
        assert d.pop_size.ssu == "TOTAL_DISCHARGES"

        names = d.specified_fields()
        for col in ("NIS_STRATUM", "HOESSION", "DISCWT", "TOTAL_HOSP", "TOTAL_DISCHARGES"):
            assert col in names
