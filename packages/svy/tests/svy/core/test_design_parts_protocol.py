# tests/svy/core/test_design_parts_protocol.py
"""The design-parts protocol: a new part needs one class and one register call.

A test-only part ("note": a text and the column it annotates) is registered
here, and every place that handles parts picks it up with no edit: Design
construction, update rules, equality, repr, columns, the Sample column check,
wrangling protection, forced removal, renames, derived state, the saved form
and the code form.
"""

from __future__ import annotations

import json
import warnings

import msgspec
import polars as pl
import pytest

import svy

from svy.core import design_parts
from svy.core.design import Design
from svy.core.design_parts import DesignPart
from svy.serialize import from_json, serialize, to_design, to_json


pytestmark = pytest.mark.filterwarnings("error")


class Note(msgspec.Struct, frozen=True):
    text: str
    column: str

    def _to_code(self) -> str:
        return f"svy.core.design_parts._REGISTRY['note'].make({self.text!r}, {self.column!r})"


class NoteData(msgspec.Struct, frozen=True):
    text: str
    column: str


class NotePart(DesignPart):
    name = "note"
    data_type = NoteData

    def make(self, text: str, column: str) -> Note:
        return Note(text, column)

    def check(self, value, fields):
        if value is not None and not isinstance(value, Note):
            raise TypeError("'note' must be Note | None")
        return value

    def after_update(self, old, value, passed, fields):
        if passed:
            return value
        # Goes with the weight, for the test's sake.
        return old.note if fields["wgt"] == old.wgt else None

    def columns(self, value, design, data_columns):
        return (value.column,)

    def renamed(self, value, renames, data_columns):
        return msgspec.structs.replace(value, column=renames.get(value.column, value.column))

    def removed(self, value, old, fields, present):
        return (value, []) if value.column in present else (None, ["note"])

    def derive(self, sample):
        note = sample._design.note
        sample.__dict__["note_seen"] = (
            None if note is None else (note.text, sample._data[note.column].n_unique())
        )
        sample.__dict__["note_calls"] = sample.__dict__.get("note_calls", 0) + 1

    def to_data(self, value):
        return NoteData(value.text, value.column)

    def from_data(self, data):
        return Note(data.text, data.column)


@pytest.fixture(autouse=True)
def note_part():
    part = design_parts.register(NotePart())
    try:
        yield part
    finally:
        design_parts.unregister("note")


NOTE = Note("region codes from 2020", "reg")
DATA = pl.DataFrame(
    {
        "st": ["a", "a", "b", "b"],
        "psu": ["1", "2", "3", "4"],
        "reg": ["N", "S", "N", "N"],
        "w": [1.0, 2.0, 1.5, 1.0],
        "y": [1.0, 2.0, 3.0, 4.0],
    }
)


def design(**kw) -> Design:
    return Design(stratum="st", psu="psu", wgt="w", note=NOTE, **kw)


def test_the_design_carries_it():
    d = design()
    assert d.note == NOTE
    assert Design(stratum="st").note is None
    with pytest.raises(TypeError, match="Note"):
        Design(note="text")


def test_update_rule():
    d = design()
    assert d.update(psu="psu2").note == NOTE
    assert d.update(wgt="w2").note is None
    assert d.update(note=None).note is None
    assert d.update(wgt="w2", note=NOTE).note == NOTE


def test_equality_hash_repr():
    assert design() == design()
    assert hash(design()) == hash(design())
    assert design() != Design(stratum="st", psu="psu", wgt="w")
    assert "note=Note(text='region codes from 2020', column='reg')" in repr(design())


def test_columns():
    assert design().columns() == ["st", "w", "psu", "reg"]


def test_saved_form_goes_in_parts():
    d = design()
    raw = json.loads(to_json(d))
    assert raw["parts"] == {"note": {"text": "region codes from 2020", "column": "reg"}}
    assert to_design(from_json(to_json(d))) == d
    assert to_design(serialize(d)) == d


def test_code_form():
    d = design()
    assert "note=svy.core.design_parts._REGISTRY['note'].make(" in d._to_code()
    assert eval(d._to_code(), {"svy": svy}) == d


def test_the_sample_checks_its_columns():
    with pytest.raises(ValueError, match=r"not found in data: \['reg'\]"):
        svy.Sample(DATA.drop("reg"), design())


def test_derive_runs_on_every_change():
    s = svy.Sample(DATA, design())
    assert s.__dict__["note_seen"] == ("region codes from 2020", 2)
    f = s.wrangling.filter_records(svy.col("reg") == "N")
    f.design
    assert f.__dict__["note_seen"] == ("region codes from 2020", 1)
    calls = f.__dict__["note_calls"]
    f.design
    f.design_history
    assert f.__dict__["note_calls"] == calls
    s.update_design(note=None)
    assert s.__dict__["note_seen"] is None


def test_wrangling_protects_and_force_cleans():
    s = svy.Sample(DATA, design())
    with pytest.raises(svy.MethodError, match="design-referenced"):
        s.wrangling.remove_columns("reg")
    with pytest.warns(
        UserWarning, match=r"Removed from the design with the dropped columns: note\."
    ):
        r = s.wrangling.remove_columns("reg", force=True)
    assert r.design.note is None
    assert s.design.note == NOTE


def test_rename_is_carried():
    s = svy.Sample(DATA, design())
    r = s.wrangling.rename_columns({"reg": "region"})
    assert r.design.note == Note("region codes from 2020", "region")
    assert r.design_history[0].note == Note("region codes from 2020", "region")


def test_history_and_update_design():
    s = svy.Sample(DATA, Design(stratum="st", psu="psu", wgt="w"))
    s.update_design(note=NOTE)
    assert [d.note for d in s.design_history] == [None, NOTE]
    s.set_design(s.design_history[0])
    assert s.design.note is None


def test_estimates_are_untouched():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = svy.Sample(DATA, design()).estimation.mean("y").estimates[0]
        b = svy.Sample(DATA, Design(stratum="st", psu="psu", wgt="w")).estimation.mean("y")
    assert (a.est, a.se) == (b.estimates[0].est, b.estimates[0].se)


def test_unregistered_parts_are_refused():
    design_parts.unregister("note")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Design(note=NOTE)
    design_parts.register(NotePart())


def test_names_are_checked():
    class Bad(DesignPart):
        name = "stratum"

    with pytest.raises(ValueError, match="taken"):
        design_parts.register(Bad())
    Bad.name = "columns"
    with pytest.raises(ValueError, match="shadows"):
        design_parts.register(Bad())
    with pytest.raises(ValueError, match="taken"):
        design_parts.register(NotePart())
