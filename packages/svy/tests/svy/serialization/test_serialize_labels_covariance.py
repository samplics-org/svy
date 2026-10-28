# tests/svy/serialization/test_serialize_labels_covariance.py
"""A saved estimate keeps its covariance; a saved table or t-test by domain keeps its labels."""

import numpy as np
import polars as pl
import pytest

import svy

from svy.serialize import from_dict, from_json, to_dict, to_json


SLUG = "hld_sample_wb_2023"


@pytest.fixture(scope="module")
def sample():
    return svy.datasets.load(SLUG, source="bundled")


@pytest.fixture(scope="module")
def registry():
    """The bundled registry's labels: the independent source for the saved ones."""
    return svy.datasets.info(SLUG, source="bundled").variables


def _expected(registry, var, codes):
    entry = registry[var]
    return {
        "var": var,
        "var_label": entry["label"],
        "values": [{"code": c, "label": entry["values"][str(c)]} for c in codes],
    }


# ---- covariance ----------------------------------------------------------------------


def test_covariance_round_trips_and_its_diagonal_is_each_rows_variance(sample):
    est = sample.estimation.mean("pc_exp", by="geo1")
    d = to_dict(est)
    cov = np.array(d["covariance"])
    assert cov.shape == (len(est.estimates), len(est.estimates))
    np.testing.assert_array_equal(cov, est.covariance)
    np.testing.assert_allclose(np.diag(cov), [p.se**2 for p in est.estimates], rtol=1e-12)
    assert d["covariance_filled"] is True
    back = from_json(to_json(est))
    np.testing.assert_array_equal(np.array(back.covariance), est.covariance)
    assert back.covariance_filled is True


def test_saved_covariance_gives_the_variance_of_a_difference(sample):
    est = sample.estimation.mean("pc_exp", by="urbrur")
    rows = [p.by_level[0] for p in est.estimates]
    i, j = rows.index(1), rows.index(2)
    c = np.array(from_json(to_json(est)).covariance)
    diff = est.contrast(svy.estd(1) - svy.estd(2)).estimates[0]
    assert c[i, i] + c[j, j] - 2 * c[i, j] == pytest.approx(diff.se**2, rel=1e-12)


def test_unfilled_covariance_is_flagged(sample):
    # Medians by domain are estimated one domain at a time: the off-diagonals are zeros.
    d = to_dict(sample.estimation.median("pc_exp", by="urbrur"))
    assert d["covariance_filled"] is False
    assert d["covariance"][0][1] == 0.0


def test_nan_in_the_covariance_survives_json():
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "g": ["a", "a", "a", "b", "b", "b"],
            "w": [1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    est = svy.Sample(df, svy.Design(wgt="w")).estimation.mean("y", by="g")
    assert np.isnan(est.covariance).any()
    back = np.array(from_json(to_json(est)).covariance)
    np.testing.assert_array_equal(np.isnan(back), np.isnan(est.covariance))
    np.testing.assert_array_equal(back[~np.isnan(back)], est.covariance[~np.isnan(est.covariance)])


def test_older_estimate_payload_decodes_without_covariance(sample):
    d = to_dict(sample.estimation.mean("pc_exp"))
    for key in ("covariance", "covariance_filled"):
        d.pop(key)
    d["schema_version"] = "svy-result/0.12"
    back = from_dict(d)
    assert back.covariance is None and back.covariance_filled is None


# ---- labels on tables ----------------------------------------------------------------


def test_oneway_table_saves_its_labels(sample, registry):
    d = to_dict(sample.categorical.tabulate("urbrur"))
    assert d["labels"] == [_expected(registry, "urbrur", [1, 2])]


def test_twoway_table_saves_row_and_column_labels(sample, registry):
    d = to_dict(sample.categorical.tabulate("urbrur", "electricity"))
    assert d["labels"] == [
        _expected(registry, "urbrur", [1, 2]),
        _expected(registry, "electricity", [0, 1]),
    ]


def test_table_without_labels_saves_none():
    df = pl.DataFrame({"g": ["a", "b", "a", "b"], "w": [1.0, 2.0, 1.0, 1.0]})
    t = svy.Sample(df, svy.Design(wgt="w")).categorical.tabulate("g")
    assert to_dict(t)["labels"] is None


# ---- labels on t-tests by domain -----------------------------------------------------


def test_ttest_by_saves_the_labels_of_the_levels_present(sample, registry):
    tt = sample.categorical.ttest("pc_exp", mean_h0=3000, by="geo1", where=svy.col("geo1") != 1)
    present = sorted({r.diff[0].by_level for r in tt})
    assert 1 not in present
    d = to_dict(tt)
    assert d["labels"] == [_expected(registry, "geo1", present)]


def test_two_group_ttest_by_saves_by_labels(sample, registry):
    tt = sample.categorical.ttest("pc_exp", group="urbrur", by="electricity")
    d = to_dict(tt)
    assert d["kind"] == "ttest_two_groups_by"
    assert d["labels"] == [_expected(registry, "electricity", [0, 1])]


def test_ttest_by_several_variables_saves_each(sample, registry):
    d = to_dict(sample.categorical.ttest("pc_exp", mean_h0=3000, by=["urbrur", "electricity"]))
    assert d["labels"] == [
        _expected(registry, "urbrur", [1, 2]),
        _expected(registry, "electricity", [0, 1]),
    ]


@pytest.mark.parametrize("kind", ["table", "ttest_one_group_by"])
def test_older_payload_decodes_without_labels(sample, kind):
    result = (
        sample.categorical.tabulate("urbrur")
        if kind == "table"
        else sample.categorical.ttest("pc_exp", mean_h0=3000, by="urbrur")
    )
    d = to_dict(result)
    d.pop("labels")
    d["schema_version"] = "svy-result/0.12"
    assert from_dict(d).labels is None


def test_saved_labels_round_trip_through_json(sample):
    t = sample.categorical.tabulate("urbrur")
    back = from_json(to_json(t))
    assert [(lv.code, lv.label) for lv in back.labels[0].values] == [(1, "Rural"), (2, "Urban")]
