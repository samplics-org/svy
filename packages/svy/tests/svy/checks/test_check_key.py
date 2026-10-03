from __future__ import annotations

from typing import Any

import msgspec
import polars as pl
import pytest

from svy.checks import KeyCheck, check_key
from svy.core.panel import duplicate_case_ids
from svy.errors import DimensionError, MethodError


def test_unique_key():
    r = check_key(pl.DataFrame({"id": [3, 1, 2]}), "id")
    assert r == KeyCheck(
        columns=["id"], n=3, n_null=0, n_duplicated=0, n_rows_duplicated=0, examples=[]
    )


def test_duplicates_on_one_column():
    r = check_key(pl.DataFrame({"id": [3, 1, 1, 2, 3, 3]}), "id")
    assert (r.n, r.n_null, r.n_duplicated, r.n_rows_duplicated) == (6, 0, 2, 5)
    assert r.examples == [1, 3]


def test_null_keys_are_counted_apart():
    r = check_key(pl.DataFrame({"id": [1, None, None, 2, 2]}), ["id"])
    assert (r.n_null, r.n_duplicated, r.n_rows_duplicated, r.examples) == (2, 1, 2, [2])


def test_duplicates_on_several_columns():
    df = pl.DataFrame(
        {
            "hh": [2, 1, 1, 2, 2, 1, 3],
            "line": [1, 1, 2, 1, 1, 2, None],
        }
    )
    r = check_key(df, ["hh", "line"])
    assert r.columns == ["hh", "line"]
    assert (r.n, r.n_null, r.n_duplicated, r.n_rows_duplicated) == (7, 1, 2, 5)
    assert r.examples == [(1, 2), (2, 1)]
    assert check_key(df, "hh").examples == [1, 2]


def test_limit_caps_examples_not_counts():
    df = pl.DataFrame({"id": [5, 5, 4, 4, 3, 3]})
    r = check_key(df, "id", limit=2)
    assert (r.n_duplicated, r.examples) == (3, [3, 4])
    assert check_key(df, "id", limit=0).examples == []


def test_key_column_named_like_the_count():
    r = check_key(pl.DataFrame({"len": [1, 1]}), "len")
    assert (r.n_duplicated, r.n_rows_duplicated, r.examples) == (1, 2, [1])


def test_lazy_frame_and_string_keys():
    r = check_key(pl.LazyFrame({"k": ["b", "a", "b"]}), "k")
    assert r.examples == ["b"]


def test_bad_input():
    df = pl.DataFrame({"id": [1]})
    with pytest.raises(DimensionError) as exc:
        check_key(df, ["id", "line"])
    assert exc.value.code == "MISSING_COLUMNS" and exc.value.got == ["line"]
    for cols in ([], [1], None):
        with pytest.raises(MethodError):
            check_key(df, cols)
    for limit in (-1, 1.5, True):
        with pytest.raises(MethodError):
            check_key(df, "id", limit=limit)


def test_report_serializes_and_prints():
    r = check_key(pl.DataFrame({"a": [1, 1], "b": ["x", "x"]}), ["a", "b"])
    back = msgspec.json.decode(msgspec.json.encode(r), type=KeyCheck)
    assert back.n_duplicated == 1 and back.examples == [[1, "x"]]
    text = repr(r)
    assert text.splitlines()[0] == "Key check: a, b"
    assert "(1, 'x')" in text


# duplicate_case_ids is expressed on check_key; this is its implementation
# before that, kept to pin the behavior.
def _old_duplicate_case_ids(data, case_id, wave, *, limit=10) -> list[Any]:
    cols = [case_id] if isinstance(case_id, str) else list(case_id)
    keys = cols if wave is None else [*cols, wave]
    dup = data.group_by(keys).len().filter(pl.col("len") > 1)
    out = dup.select(cols).unique().sort(cols).head(limit)
    if len(cols) == 1:
        return out.get_column(cols[0]).to_list()
    return list(out.iter_rows())


_PANEL = pl.DataFrame(
    {
        "hh": [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
        "ln": [1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1],
        "wave": [1, 1, 1, 2, 1, 2, 1, 2, 2, None, None, None, None],
    }
)


@pytest.mark.parametrize("case_id", ["hh", ["hh", "ln"]])
@pytest.mark.parametrize("wave", [None, "wave"])
@pytest.mark.parametrize("limit", [10, 1, 0])
def test_duplicate_case_ids_unchanged(case_id, wave, limit):
    got = duplicate_case_ids(_PANEL, case_id, wave, limit=limit)
    assert got == _old_duplicate_case_ids(_PANEL, case_id, wave, limit=limit)


def test_duplicate_case_ids_values():
    assert duplicate_case_ids(_PANEL, "hh", "wave") == [1, 3, 4, 5]
    assert duplicate_case_ids(_PANEL, ["hh", "ln"], "wave") == [(1, 1), (3, 1), (5, 1)]
    assert duplicate_case_ids(_PANEL, ["hh", "ln"], None) == [
        (1, 1),
        (1, 2),
        (2, 1),
        (3, 1),
        (5, 1),
    ]
