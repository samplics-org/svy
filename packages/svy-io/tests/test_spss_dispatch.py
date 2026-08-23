"""`read_spss` routing and the user-missing accessor.

Both are public and neither was referenced by any test. `read_spss` is not a
reader -- it is a dispatcher that picks one by file extension, so the routing
is the whole function, and nothing checked it. `get_user_missing_for_column`
is the documented way to reach the user-missing rules a reader attached.
"""

import shutil

import polars as pl
import pytest

from svy_io import get_user_missing_for_column, read_sav, read_spss


SAV = "spss/labelled-num-na.sav"


@pytest.fixture
def sav(test_data_dir):
    return test_data_dir / SAV


# ---- read_spss dispatch ---------------------------------------------------


def test_sav_routes_to_the_sav_reader(sav):
    """Dispatching must produce exactly what calling the reader directly does."""
    via_dispatch, meta_dispatch = read_spss(sav)
    direct, meta_direct = read_sav(sav)

    assert via_dispatch.equals(direct)
    assert meta_dispatch["vars"] == meta_direct["vars"]


def test_dispatch_forwards_its_arguments(sav):
    """A dispatcher that drops kwargs silently reads the wrong thing."""
    limited, _ = read_spss(sav, n_max=1)
    full, _ = read_spss(sav)

    assert limited.height == 1
    assert full.height >= limited.height


def test_unknown_extension_is_refused(tmp_path, sav):
    """Guessing a format from content would be worse than saying no."""
    masked = tmp_path / "data.wat"
    shutil.copy(sav, masked)

    with pytest.raises(ValueError, match="Unknown SPSS file extension"):
        read_spss(masked)


def test_extension_match_ignores_case(tmp_path, sav):
    upper = tmp_path / "data.SAV"
    shutil.copy(sav, upper)

    df, _ = read_spss(upper)

    assert df.height > 0


# ---- get_user_missing_for_column -----------------------------------------


def test_user_missing_is_returned_for_a_column_that_has_rules(sav):
    _df, meta = read_sav(sav, user_na=True)
    named = next(v["name"] for v in meta["vars"])

    got = get_user_missing_for_column(meta, named)

    # Either the column carries rules or it does not, but the accessor must
    # agree with what the reader actually attached to that variable.
    expected = next(v for v in meta["vars"] if v["name"] == named).get("user_missing")
    assert got == expected


def test_unknown_column_returns_none_rather_than_raising(sav):
    _df, meta = read_sav(sav)

    assert get_user_missing_for_column(meta, "no_such_column") is None


def test_empty_metadata_returns_none():
    assert get_user_missing_for_column({}, "x") is None
    assert get_user_missing_for_column({"vars": []}, "x") is None


def test_column_without_rules_returns_none():
    meta = {"vars": [{"name": "x", "label": None, "label_set": None, "fmt": None}]}

    assert get_user_missing_for_column(meta, "x") is None


@pytest.fixture
def test_data_dir():
    from pathlib import Path

    d = Path(__file__).parent / "data"
    if not d.exists():
        pytest.skip("Test data directory not found")
    return d


def test_polars_frame_is_returned(sav):
    """Guard the contract the dispatcher advertises: (DataFrame, dict)."""
    df, meta = read_spss(sav)

    assert isinstance(df, pl.DataFrame)
    assert isinstance(meta, dict)
