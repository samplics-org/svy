"""Labels survive a write/read round trip through the statistical formats.

Neither writer could run before: `_write_spss` called `svy_io.write_spss` and
`_write_sas` called `svy_io.write_sas`, and neither name has ever existed. Both
calls carried `# type: ignore[attr-defined]`, which was the type checker saying
so. There was no test because there was nothing that worked to test.
"""

import pathlib
import tempfile
import warnings

import polars as pl
import pytest

from svy.engine.io.core import import_labels_from_svyio_meta
from svy.engine.io.sas import _read_sas, _write_sas
from svy.engine.io.spss import _read_spss, _write_spss
from svy.engine.io.stata import _read_stata, _write_stata
from svy.metadata import MetadataStore


@pytest.fixture
def store():
    s = MetadataStore()
    s.set_label("q", "Question")
    s.set_value_labels("q", {1: "Yes", 99: "Refused"})
    s.set_label("age", "Age")
    return s


@pytest.fixture
def df():
    return pl.DataFrame({"q": [1.0, 99.0, 1.0], "age": [20.0, 30.0, 40.0]})


@pytest.fixture
def tmp():
    return pathlib.Path(tempfile.mkdtemp())


def test_spss_carries_variable_and_value_labels(store, df, tmp):
    path = tmp / "t.sav"
    _write_spss(df, store, path)
    back, meta, _ = _read_spss(path)

    assert back["q"].to_list() == [1.0, 99.0, 1.0]

    recovered = MetadataStore()
    import_labels_from_svyio_meta(recovered, meta)
    assert recovered.get("q").label == "Question"
    assert recovered.get("age").label == "Age"
    # 99 is a value like any other, and it travels with its label.
    assert set(recovered.get("q").labels.values()) == {"Yes", "Refused"}


def test_a_label_read_back_from_spss_still_applies(store, df, tmp):
    """The reason `display` tolerates a code-type mismatch.

    SPSS stores value-label keys as strings, so a `.sav` round trip yields
    {"1": "Yes"} against a Float64 column. A literal lookup missed, and
    `display` returned the bare number — the one job the labels exist for.
    `display_series` never showed it because it stringifies both sides.
    """
    path = tmp / "t.sav"
    _write_spss(df, store, path)
    _, meta, _ = _read_spss(path)

    recovered = MetadataStore()
    import_labels_from_svyio_meta(recovered, meta)
    resolved = recovered.resolve_labels("q")

    assert resolved.display(1.0) == "Yes"
    assert resolved.display(99.0) == "Refused"
    assert resolved.display_series(pl.Series("q", [1.0, 99.0])).to_list() == ["Yes", "Refused"]


def test_stata_carries_labels_too(store, df, tmp):
    path = tmp / "t.dta"
    _write_stata(df, store, path)
    back, meta, _ = _read_stata(path)

    assert back["q"].to_list() == [1.0, 99.0, 1.0]
    recovered = MetadataStore()
    import_labels_from_svyio_meta(recovered, meta)
    assert recovered.get("q").label == "Question"


def test_sas_writes_xpt_and_says_the_labels_are_dropped(store, df, tmp):
    """XPT is the only SAS format ReadStat writes, and it carries no labels.

    Degrading is right; degrading quietly is not — someone exporting a labelled
    dataset to SAS should be told the labels did not come with it.
    """
    path = tmp / "t.xpt"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write_sas(df, store, path)

    assert any("carries no variable or value labels" in str(w.message) for w in caught)

    back, _, _ = _read_sas(path)
    assert back["q"].to_list() == [1.0, 99.0, 1.0]


def test_the_sas_format_argument_is_reported_as_ignored(store, df, tmp):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write_sas(df, store, tmp / "t.xpt", format="sas7bdat")

    assert any("ReadStat writes SAS Transport (XPT) only" in str(w.message) for w in caught)
