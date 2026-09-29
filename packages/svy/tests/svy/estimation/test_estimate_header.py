# tests/svy/estimation/test_estimate_header.py
"""
The printed header names the estimated variable.

Printing a mean, total, ratio, median or quantile used to show only the
parameter and method, so a loop over several results printed a run of
unlabeled tables. A ``y:`` line (``y / x:`` for a ratio) now sits above
``where:`` in both printers, with the variable label in place of the name
when labels are on. Proportions keep heading their level column with the
variable instead.
"""

import re

from pathlib import Path

import polars as pl
import pytest

from svy.core.sample import Design, Sample
from svy.estimation import Estimate, EstimateList


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"


def plain_head(result) -> list[str]:
    """The plain-text lines above the table."""
    lines = result.__plain_str__().splitlines()
    return lines[: lines.index("")]


def rich_head(result) -> list[str]:
    """The rich panel's body lines above the table, borders and styling removed."""
    out = []
    for line in str(result).splitlines()[1:]:
        text = re.sub(r"\x1b\[[0-9;]*m", "", line).strip("│ ")
        if not text:
            break
        out.append(text)
    return out


def heads(result) -> list[str]:
    plain = [line.strip() for line in plain_head(result)[1:]]
    assert rich_head(result) == plain, "rich and plain headers differ"
    return plain


@pytest.fixture
def sample():
    df = pl.read_csv(DATA_DIR / "apistrat.csv").with_columns(
        (pl.col("api00") > 600).alias("hi_api")
    )
    return Sample(df, Design(wgt="pw", stratum="stype"))


@pytest.fixture
def labelled(sample):
    sample.meta.set_label("api00", "API 2000")
    sample.meta.set_label("api99", "API 1999")
    sample.meta.set_label("enroll", "Enrollment")
    sample.meta.set_label("hi_api", "High API")
    return sample


WHERE = pl.col("enroll") > 300


class TestSingleVariable:
    @pytest.mark.parametrize("by", [None, "stype"])
    @pytest.mark.parametrize(
        "call",
        [
            lambda e, **kw: e.mean("api00", **kw),
            lambda e, **kw: e.total("api00", **kw),
            lambda e, **kw: e.median("api00", **kw),
            lambda e, **kw: e.quantile("api00", p=0.25, **kw),
        ],
        ids=["mean", "total", "median", "quantile"],
    )
    def test_y_line(self, sample, call, by):
        assert heads(call(sample.estimation, by=by)) == ["y: api00"]

    @pytest.mark.parametrize("by", [None, "stype"])
    def test_ratio_names_both(self, sample, by):
        res = sample.estimation.ratio("api00", "api99", by=by)
        assert heads(res) == ["y / x: api00 / api99"]

    def test_y_line_precedes_where(self, sample):
        res = sample.estimation.mean("api00", by="stype", where=WHERE)
        assert heads(res) == ["y: api00", f"where: {res.where_clause}"]

    def test_title_is_unchanged(self, sample):
        assert plain_head(sample.estimation.mean("api00"))[0] == "Estimate: MEAN (TAYLOR)"

    def test_replication(self, sample):
        rep = sample.weighting.create_jk_wgts(psu="snum", rep_prefix="jk")
        res = rep.estimation.mean("api00", method="replication")
        assert heads(res) == ["y: api00"]


class TestTableAlreadyNamesTheVariable:
    @pytest.mark.parametrize("by", [None, "stype"])
    def test_prop_has_no_y_line(self, sample, by):
        res = sample.estimation.prop("hi_api", by=by, where=WHERE)
        assert heads(res) == [f"where: {res.where_clause}"]
        assert "hi_api" in res.to_polars_printable().columns

    def test_factor_mean_has_no_y_line(self, sample):
        assert heads(sample.estimation.mean("hi_api", as_factor=True)) == []


class TestLabels:
    def test_label_replaces_name(self, labelled):
        assert heads(labelled.estimation.mean("api00")) == ["y: API 2000"]

    def test_ratio_labels(self, labelled):
        res = labelled.estimation.ratio("api00", "api99")
        assert heads(res) == ["y / x: API 2000 / API 1999"]

    def test_use_labels_off_shows_name(self, labelled):
        res = labelled.estimation.mean("api00").style(use_labels=False)
        assert heads(res) == ["y: api00"]

    def test_unlabelled_variable_keeps_its_name(self, labelled):
        assert heads(labelled.estimation.mean("api99", by="stype")) == ["y: API 1999"]
        assert heads(labelled.estimation.total("meals")) == ["y: meals"]


class TestEstimateList:
    def test_shared_variable_is_a_header_line_not_a_title_suffix(self, sample):
        res = sample.estimation.quantile("api00", p=(0.25, 0.75))
        assert isinstance(res, EstimateList)
        assert plain_head(res)[0] == "Estimate: QUANTILE (TAYLOR, q_method=higher)"
        assert heads(res) == ["y: api00"]

    def test_varying_variables_stay_a_column(self, sample):
        res = sample.estimation.mean(["api00", "api99"])
        assert heads(res) == []
        assert res._combined()["y"].to_list() == ["api00", "api99"]

    def test_y_column_follows_labels(self, labelled):
        res = labelled.estimation.mean(["api00", "api99"])
        assert res._combined()["y"].to_list() == ["API 2000", "API 1999"]
        assert res._combined(use_labels=False)["y"].to_list() == ["api00", "api99"]

    def test_plain_shows_where(self, sample):
        res = sample.estimation.mean(["api00", "api99"], where=WHERE)
        assert heads(res) == [f"where: {res[0].where_clause}"]

    def test_shared_ratio(self, sample):
        res = EstimateList(
            [sample.estimation.ratio("api00", "api99"), sample.estimation.ratio("api00", "api99")]
        )
        assert heads(res) == ["y / x: api00 / api99"]

    def test_ratio_varying_denominator_gets_an_x_column(self, sample):
        """These rows printed with nothing telling them apart."""
        res = sample.estimation.ratio("api00", ["api99", "enroll"])
        assert heads(res) == ["y: api00"]
        df = res._combined()
        assert df.columns[0] == "x" and "y" not in df.columns
        assert df["x"].to_list() == ["api99", "enroll"]

    def test_ratio_varying_numerator_shares_x(self, labelled):
        res = labelled.estimation.ratio(["api00", "api99"], "enroll")
        assert heads(res) == ["x: Enrollment"]
        assert res._combined()["y"].to_list() == ["API 2000", "API 1999"]

    def test_ratio_both_varying(self, sample):
        res = sample.estimation.ratio(["api00", "api99"], ["enroll", "meals"])
        assert heads(res) == []
        df = res._combined()
        assert df.columns[:2] == ["y", "x"]
        assert df.select("y", "x").rows() == [("api00", "enroll"), ("api99", "meals")]

    def test_prop_list_unchanged(self, sample):
        res = sample.estimation.prop(["hi_api"], drop_nulls=True)
        assert heads(res) == []

    def test_mixed_params_on_one_variable(self, sample):
        res = EstimateList([sample.estimation.mean("api00"), sample.estimation.median("api00")])
        assert heads(res) == ["y: api00"]


def test_empty_estimate_still_prints(sample):
    assert "<no estimates>" in Estimate(sample.estimation.mean("api00").param).__plain_str__()
