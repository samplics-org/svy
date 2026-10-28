"""Every result's ``to_polars()``: identity, levels, numbers, then the settings behind them."""

import numpy as np
import polars as pl
import pytest

import svy

from svy import estd
from svy.errors import EstimationError
from svy.serialize import from_json, to_json, to_polars


N = 400
NUM = ["est", "se", "lci", "uci"]


def _data() -> pl.DataFrame:
    rng = np.random.default_rng(20261030)
    return pl.DataFrame(
        {
            "st": np.repeat([1, 2, 3, 4], N // 4),
            "psu": np.arange(N) // 10,
            "w": rng.uniform(1, 3, N),
            "y": rng.normal(10, 2, N),
            "z": rng.normal(5, 1, N),
            "b": rng.integers(0, 2, N),
            "g": rng.choice([1, 2], N),
            "k": rng.choice([1, 2, 3], N),
        }
    )


@pytest.fixture(scope="module")
def sample() -> svy.Sample:
    s = svy.Sample(_data(), svy.Design(stratum="st", psu="psu", wgt="w"))
    s.set_value_labels("g", {1: "Men", 2: "Women"})
    return s.weighting.create_jk_wgts()


CASES = {
    "mean": (
        lambda s, **kw: s.estimation.mean("y", by="g", **kw),
        {},
        ["y_var", "g", "g_label", *NUM, "cv", "df", "n", "param", "variance", "alpha"],
    ),
    "ratio": (
        lambda s, **kw: s.estimation.ratio("y", x="z", **kw),
        {},
        ["y_var", "x_var", *NUM, "cv", "df", "n", "param", "variance", "alpha"],
    ),
    "quantile": (
        lambda s, **kw: s.estimation.quantile("y", p=0.5, by="g", **kw),
        {},
        ["y_var", "prob", "g", "g_label", *NUM, "cv", "df", "n", "param", "variance", "alpha"],
    ),
    "contrast": (
        lambda s, **kw: s.estimation.mean("y", by="g", **kw).contrast(estd(2) - estd(1)),
        {},
        ["contrast", "y_var", *NUM, "cv", "t", "df", "p_value", "param", "variance", "alpha"],
    ),
    "table": (
        lambda s, **kw: s.categorical.tabulate("k", "g", **kw),
        {},
        ["k", "g", "g_label", *NUM, "cv", "n", "share_of", "units", "variance", "alpha"],
    ),
    "ttest_two": (
        lambda s, **kw: s.categorical.ttest("y", group="g", **kw),
        {},
        ["y_var", "group_var", *NUM, "t", "df", "p_value"]
        + ["alternative", "paired", "variance", "alpha"],
    ),
    "ttest_two_by": (
        lambda s, **kw: s.categorical.ttest("y", group="b", by="g", **kw),
        {},
        ["y_var", "group_var", "g", "g_label", *NUM, "t", "df", "p_value"]
        + ["alternative", "paired", "variance", "alpha"],
    ),
    "ttest_one": (
        lambda s, **kw: s.categorical.ttest("y", mean_h0=10, **kw),
        {},
        ["y_var", *NUM, "t", "df", "p_value", "alternative", "mean_h0", "variance", "alpha"],
    ),
    "ttest_two_est": (
        lambda s, **kw: s.categorical.ttest("y", group="g", **kw),
        {"component": "estimates"},
        ["y_var", "g", "g_label", *NUM, "cv", "n", "variance", "alpha"],
    ),
    "ttest_one_est": (
        lambda s, **kw: s.categorical.ttest("y", mean_h0=10, **kw),
        {"component": "estimates"},
        ["y_var", *NUM, "cv", "n", "variance", "alpha"],
    ),
    "rank_two": (
        lambda s, **kw: s.categorical.ranktest("y", group="g", **kw),
        {},
        ["y_var", "group_var", *NUM, "t", "df", "p_value"]
        + ["method", "alternative", "variance", "alpha"],
    ),
    "rank_k": (
        lambda s, **kw: s.categorical.ranktest("y", group="k", score="kruskal-wallis", **kw),
        {},
        ["y_var", "group_var", "f", "df_num", "df_den", "p_value", "method", "variance", "alpha"],
    ),
    "rank_k_by": (
        lambda s, **kw: s.categorical.ranktest(
            "y", group="k", by="g", score="kruskal-wallis", **kw
        ),
        {},
        ["y_var", "group_var", "g", "g_label", "f", "df_num", "df_den", "p_value"]
        + ["method", "variance", "alpha"],
    ),
    "glm": (
        lambda s, **kw: s.glm.fit(y="y", x=["z"], **kw),
        {},
        ["term", "y_var", *NUM, "t", "df", "p_value", "family", "link", "variance", "alpha"],
    ),
    "glm_ratio": (
        lambda s, **kw: s.glm.fit(y="b", x=["z"], family="binomial", **kw),
        {"exponentiate": True},
        ["term", "y_var", "odds_ratio", "se", "lci", "uci", "t", "df", "p_value"]
        + ["family", "link", "variance", "alpha"],
    ),
    "margins": (
        lambda s, **kw: s.glm.fit(y="y", x=["z"], **kw).margins(
            variables=["z"], alpha=kw["alpha"]
        ),
        {},
        ["term", *NUM, "df", "margin_type", "variance", "alpha"],
    ),
    "margins_at": (
        lambda s, **kw: s.glm.fit(y="y", x=["z"], **kw).margins(
            at={"z": [4.0, 6.0]}, alpha=kw["alpha"]
        ),
        {},
        ["term", "z", *NUM, "df", "margin_type", "variance", "alpha"],
    ),
    "predict": (
        lambda s, **kw: s.glm.fit(y="y", x=["z"], **kw).predict(
            s.data.head(5), alpha=kw["alpha"], y_col="y"
        ),
        {},
        [*NUM, "residuals", "df", "variance", "alpha"],
    ),
}
VARIANCE = {"taylor": "Taylor", "replication": "Jackknife"}


def _frame(sample, case, method="taylor", alpha=0.1):
    build, opts, _ = CASES[case]
    result = build(sample, method=method, alpha=alpha)
    return result, result.to_polars(**opts)


@pytest.mark.parametrize("case", list(CASES))
def test_columns_follow_the_contract(sample, case):
    _, df = _frame(sample, case)
    assert df.columns == CASES[case][2]


@pytest.mark.parametrize("method", list(VARIANCE))
@pytest.mark.parametrize("case", list(CASES))
def test_settings_carry_the_call(sample, case, method):
    _, df = _frame(sample, case, method=method, alpha=0.1)
    assert df["variance"].unique().to_list() == [VARIANCE[method]]
    assert df["alpha"].unique().to_list() == [0.1]


@pytest.mark.parametrize("case", [c for c in CASES if "df" in CASES[c][2]])
def test_df_is_float(sample, case):
    _, df = _frame(sample, case)
    assert df.schema["df"] == pl.Float64


@pytest.mark.parametrize(
    "case,col,value",
    [
        ("mean", "param", "Mean"),
        ("ratio", "x_var", "z"),
        ("contrast", "y_var", "y"),
        ("table", "units", "proportion"),
        ("table", "share_of", "total"),
        ("ttest_two", "group_var", "g"),
        ("ttest_two", "paired", False),
        ("ttest_one", "mean_h0", 10.0),
        ("rank_two", "method", "Wilcoxon"),
        ("rank_k", "method", "Kruskal-Wallis"),
        ("glm", "family", "Gaussian"),
        ("margins", "margin_type", "ame"),
    ],
)
def test_setting_values(sample, case, col, value):
    _, df = _frame(sample, case)
    assert df[col].unique().to_list() == [value]


def test_ttest_difference_is_the_estimate(sample):
    r, df = _frame(sample, "ttest_two")
    assert df["est"].to_list() == [r.diff[0].diff]


def test_rank_test_has_no_estimates_component(sample):
    r, _ = _frame(sample, "rank_two")
    with pytest.raises(TypeError):
        r.to_polars("estimates")


@pytest.mark.parametrize("name", ["alpha", "variance", "y_var"])
def test_variable_named_like_a_result_column_raises(name):
    df = _data().with_columns(pl.col("g").alias(name))
    s = svy.Sample(df, svy.Design(wgt="w"))
    with pytest.raises(EstimationError) as exc:
        s.estimation.mean("y", by=name).to_polars()
    assert exc.value.code == "RESULT_COLUMN_CLASH"
    assert f"'{name}'" in str(exc.value)


# Margins and rank tests are not serialized.
SAVED = [c for c in CASES if not c.startswith(("margins", "rank"))]


@pytest.mark.parametrize("method", list(VARIANCE))
@pytest.mark.parametrize("case", SAVED)
def test_saved_result_gives_the_live_frame(sample, case, method):
    build, opts, _ = CASES[case]
    result = build(sample, method=method, alpha=0.1)
    live = result.fitted if hasattr(result, "fitted") else result
    assert to_polars(from_json(to_json(live)), **opts).equals(live.to_polars(**opts))
