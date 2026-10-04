"""Enums apps offer as choices carry a label and a description per member."""

from enum import StrEnum

import pytest

import svy

from svy.core.enumerations import (
    DistFamily,
    LinkFunction,
    QuantileMethod,
    RankScoreMethod,
    SingletonMethod,
)


DESCRIBED = [SingletonMethod, DistFamily, LinkFunction, QuantileMethod, RankScoreMethod]


@pytest.mark.parametrize("enum", DESCRIBED, ids=lambda e: e.__name__)
def test_every_member_has_a_label_and_a_description(enum):
    for member in enum:
        assert member.label.strip() and member.description.strip()
        assert member.description.endswith(".")


@pytest.mark.parametrize("enum", DESCRIBED, ids=lambda e: e.__name__)
def test_describe_lists_every_member_in_order(enum):
    rows = enum.describe()
    assert [r[0] for r in rows] == [m.value for m in enum]
    assert all(len(r) == 3 for r in rows)


@pytest.mark.parametrize("enum", DESCRIBED, ids=lambda e: e.__name__)
def test_still_plain_str_enums(enum):
    member = next(iter(enum))
    assert isinstance(member, StrEnum) and isinstance(member, str)
    assert member == member.value
    assert enum(member.value) is member


def test_labels_are_distinct_within_an_enum():
    for enum in DESCRIBED:
        labels = [m.label for m in enum]
        assert len(labels) == len(set(labels)), enum.__name__


def test_public_singleton_method_is_the_described_one():
    assert svy.SingletonMethod is SingletonMethod
    assert svy.SingletonMethod.SKIP.label == "Skip"


def test_a_mismatched_description_table_is_refused():
    from svy.core.enumerations import _describe

    with pytest.raises(RuntimeError, match="do not match"):
        _describe(QuantileMethod, {"Lower": ("Lower", "x.")})
