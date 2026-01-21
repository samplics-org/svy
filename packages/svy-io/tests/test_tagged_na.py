# tests/test_tagged_na.py
from svy_io.tagged_na import (
    TaggedNA,
    format_tagged_na,
    is_tagged_na,
    na_tag,
    print_tagged_na,
    tagged_na,
)


def test_tagged_na_is_missing_not_nan():
    x = tagged_na("a")
    # “Missing” in our world = TaggedNA instance (distinct from None/NaN semantics)
    assert isinstance(x, TaggedNA)
    # We don’t overload IEEE NaN; check via our helpers instead:
    assert is_tagged_na(x) is True


def test_na_tag_vector_and_scalars():
    xs = tagged_na(["a", "z"])
    assert na_tag(xs) == ["a", "z"]
    assert na_tag(tagged_na("m")) == "m"
    assert na_tag(None) is None  # system NA -> None tag
    assert na_tag(1) is None  # non-missing -> None tag


def test_is_tagged_na_variants():
    assert is_tagged_na(None) is False
    assert is_tagged_na(1) is False
    tz = tagged_na(["a", "z"])
    assert is_tagged_na(tz) == [True, True]
    assert is_tagged_na(tz, "a") == [True, False]
    assert is_tagged_na(tz, "z") == [False, True]


def test_format_tagged_na_matches_examples():
    x = [1, tagged_na("a"), None]
    assert format_tagged_na(x) == ["    1", "NA(a)", "   NA"]


def test_print_tagged_na_snapshot_stability():
    vec = list(range(1, 6)) + tagged_na(list("abc")) + [None]
    out = print_tagged_na(vec)
    # Basic shape/contains checks; adjust if you add a snapshot tester later
    lines = out.splitlines()
    assert lines[0].endswith("1")
    assert "NA(a)" in out and "NA(b)" in out and "NA(c)" in out
    assert lines[-1].strip() == "NA"
