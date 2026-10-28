# tests/svy/datasets/test_input_errors.py
"""Bad input to svy.datasets raises an svy error that is also the builtin it replaces."""

from __future__ import annotations

import pytest

import svy

from svy.errors import Code, DatasetError, SvyError


SLUG = "ea_frame_wb_2023"


@pytest.mark.parametrize(
    "call",
    [
        lambda: svy.datasets.load(SLUG, source="web"),
        lambda: svy.datasets.info(source="web"),
        lambda: svy.datasets.info(SLUG, source="web"),
    ],
    ids=["load", "info", "info(name)"],
)
def test_unknown_source(call):
    with pytest.raises(DatasetError) as err:
        call()
    e = err.value
    assert type(e) is DatasetError.ValueError
    assert isinstance(e, ValueError) and isinstance(e, SvyError)
    assert e.code == Code.INVALID_CHOICE
    assert e.param == "source"
    assert e.got == "web"
