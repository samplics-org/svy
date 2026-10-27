import copy
import pickle

import polars as pl
import pytest

import svy

from svy.errors._api import UnknownNameError, UsageError


@pytest.fixture
def sample():
    df = pl.DataFrame(
        {
            "s": ["a", "a", "b", "b"],
            "c": ["c1", "c2", "c3", "c4"],
            "w": [1.0, 2.0, 1.5, 1.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "g": ["u", "v", "u", "v"],
        }
    )
    return svy.Sample(df, svy.Design(stratum="s", psu="c", wgt="w"))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Survey", "svy.Sample(data, svy.Design("),
        ("SurveyDesign", "svy.Sample(data, svy.Design("),
        ("Desgin", "Did you mean Design?"),
    ],
)
def test_module_unknown_name_guides(name, expected):
    with pytest.raises(UnknownNameError, match="UNKNOWN_NAME") as info:
        getattr(svy, name)
    assert expected in info.value.detail


def test_module_lazy_submodule_still_loads():
    assert svy.datasets.load is not None


def test_unknown_names_keep_hasattr_false(sample):
    assert not hasattr(svy, "Survey")
    assert not hasattr(sample, "nope")
    assert not hasattr(sample.estimation, "nope")
    assert not hasattr(sample.estimation.mean("y"), "nope")
    assert getattr(sample.glm, "nope", None) is None


@pytest.mark.parametrize(
    "namespace, name, expected",
    [
        ("estimation", "proportion", "Use prop."),
        ("estimation", "tabulate", "It is sample.categorical.tabulate."),
        ("estimation", "fit", "It is sample.glm.fit."),
        ("categorical", "mean", "It is sample.estimation.mean."),
        ("categorical", "crosstab", "Use tabulate(rowvar, colvar)."),
        ("weighting", "calibration", "Use calibrate."),
        ("wrangling", "filter", "where="),
        ("sampling", "pps", "pps_sys"),
        ("glm", "summary", "model.to_polars()"),
        ("estimation", "meen", "Did you mean mean"),
    ],
)
def test_namespace_unknown_name_guides(sample, namespace, name, expected):
    with pytest.raises(UnknownNameError) as info:
        getattr(getattr(sample, namespace), name)
    assert expected in info.value.detail
    assert info.value.hint.startswith("Available: ")


@pytest.mark.parametrize(
    "name, expected", [("mean", "sample.estimation.mean"), ("regression", "sample.glm")]
)
def test_sample_unknown_name_guides(sample, name, expected):
    with pytest.raises(UnknownNameError) as info:
        getattr(sample, name)
    assert expected in info.value.detail
    assert "sample.categorical" in info.value.hint


def test_sample_removed_api_errors_unchanged(sample):
    from svy.errors._label import LabelAPIRemoved

    with pytest.raises(LabelAPIRemoved):
        sample.labels  # noqa: B018


def test_private_names_raise_plain_attribute_error(sample):
    with pytest.raises(AttributeError) as info:
        sample.estimation._nope  # noqa: B018
    assert not isinstance(info.value, UnknownNameError)


@pytest.mark.parametrize(
    "make",
    [
        lambda s: s.estimation.mean("y"),
        lambda s: s.estimation.mean(["y", "w"]),
        lambda s: s.estimation.mean("y", by="g").contrast(svy.estd("u") - svy.estd("v")),
        lambda s: s.categorical.tabulate("g"),
        lambda s: s.categorical.ttest("y", mean_h0=1),
        lambda s: s.categorical.ttest("y", group="g"),
    ],
    ids=["estimate", "estimate_list", "contrast", "table", "ttest_one", "ttest_two"],
)
def test_results_point_to_to_polars(sample, make):
    result = make(sample)
    with pytest.raises(UnknownNameError) as info:
        result.est  # noqa: B018
    assert ".to_polars()" in info.value.detail


def test_results_still_copy_and_pickle(sample):
    est = sample.estimation.mean("y")
    assert copy.deepcopy(est).to_polars().equals(est.to_polars())
    tt = sample.categorical.ttest("y", group="g")
    assert pickle.loads(pickle.dumps(tt)).to_polars().equals(tt.to_polars())


def test_design_given_data_guides():
    with pytest.raises(UsageError, match="DESIGN_GIVEN_DATA"):
        svy.Design(pl.DataFrame({"w": [1.0]}))


def test_design_given_data_is_type_error():
    with pytest.raises(TypeError):
        svy.Design(pl.DataFrame({"w": [1.0]}))


def test_design_unknown_arguments_suggest_names():
    with pytest.raises(UsageError) as info:
        svy.Design(data=None, strata="s", weights="w", cluster="c", stratumm="s")
    detail = info.value.detail
    assert "unexpected keyword argument(s): data, strata, weights, cluster, stratumm" in detail
    assert "data: the data goes to svy.Sample" in detail
    assert "strata: use stratum=" in detail
    assert "weights: use wgt=" in detail
    assert "cluster: use psu=" in detail
    assert "stratumm: did you mean stratum=?" in detail


def test_design_update_unknown_argument_guides():
    with pytest.raises(UsageError, match="weights: use wgt="):
        svy.Design(wgt="w").update(weights="v")


def test_design_unknown_attribute_guides():
    with pytest.raises(UnknownNameError, match="Did you mean stratum"):
        svy.Design(stratum="s").stratm  # noqa: B018


def test_sample_given_pandas_guides():
    pd = pytest.importorskip("pandas")
    with pytest.raises(UsageError, match="DATA_NOT_POLARS") as info:
        svy.Sample(pd.DataFrame({"w": [1.0]}), svy.Design(wgt="w"))
    assert "polars.from_pandas" in info.value.hint


def test_sample_given_other_data_guides():
    with pytest.raises(UsageError) as info:
        svy.Sample({"w": [1.0]}, svy.Design(wgt="w"))
    assert "svy.read_csv" in info.value.hint


def test_module_docstring_points_to_sample():
    assert "svy.Sample(data, svy.Design(" in svy.__doc__
