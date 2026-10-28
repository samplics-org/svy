"""Tables, t-tests and rank tests export levels in their type, with a ``<var>_label`` column."""

import itertools

import numpy as np
import polars as pl
import pytest

import svy

from svy.categorical.table import Table
from svy.errors import EstimationError


N = 600
METHODS = ["taylor", "replication"]

LEVELS = {
    "vb": [False, True],
    "vi": [1, 2],
    "vf": [1.0, 2.0],
    "vs": ["a", "b"],
    "ve": ["lo", "hi"],
}
LABELS = {
    "vb": {False: "No", True: "Yes"},
    "vi": {1: "Low", 2: "High"},
    "vf": {1: "Urban", 2: "Rural"},  # integer codes on a float column, as SPSS files give
    "vs": {"a": "Alpha", "b": "Beta"},
    "ve": {"lo": "Lower", "hi": "Higher"},
}
DTYPES = {"vb": pl.Boolean, "vi": pl.Int64, "vf": pl.Float64, "vs": pl.String, "ve": pl.String}
KINDS = list(LEVELS)
R_LEVELS = [1, 2, 3]
R_LABELS = {1: "R1", 2: "R2", 3: "R3"}


def _data() -> pl.DataFrame:
    rng = np.random.default_rng(20261029)
    return pl.DataFrame(
        {
            "st": np.repeat([1, 2, 3, 4], N // 4),
            "psu": np.arange(N) // 10,
            "w": rng.uniform(1, 3, N),
            "y": rng.normal(size=N),
            "g": rng.choice([1, 2], N),
            "k": rng.choice([1, 2, 3], N),
            "r": rng.choice(R_LEVELS, N),
            "u": rng.choice([1, 2], N),
            "vb": rng.choice(LEVELS["vb"], N),
            "vi": rng.choice(LEVELS["vi"], N),
            "vf": rng.choice(LEVELS["vf"], N),
            "vs": rng.choice(LEVELS["vs"], N),
            "ve": rng.choice(LEVELS["ve"], N),
        }
    ).with_columns(pl.col("ve").cast(pl.Enum(LEVELS["ve"])))


def _labelled(df: pl.DataFrame) -> svy.Sample:
    s = svy.Sample(df, svy.Design(stratum="st", psu="psu", wgt="w"))
    for var, labels in LABELS.items():
        s.set_value_labels(var, labels)
    s.set_value_labels("r", R_LABELS)
    return s.weighting.create_jk_wgts()


@pytest.fixture(scope="module")
def sample() -> svy.Sample:
    return _labelled(_data())


def _labels_of(var, values):
    return [LABELS[var][v] for v in values]


# ── Tables ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("var", KINDS)
def test_one_way_table_levels_and_labels(sample, method, var):
    df = sample.categorical.tabulate(var, method=method).to_polars()
    assert df.columns[:2] == [var, f"{var}_label"]
    assert df.schema[var] == DTYPES[var]
    assert df[var].to_list() == LEVELS[var]
    assert df[f"{var}_label"].to_list() == _labels_of(var, LEVELS[var])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("var", KINDS)
def test_two_way_table_levels_and_labels(sample, method, var):
    df = sample.categorical.tabulate("r", var, method=method).to_polars()
    assert df.columns[:4] == ["r", "r_label", var, f"{var}_label"]
    assert (df.schema["r"], df.schema[var]) == (pl.Int64, DTYPES[var])
    cells = list(itertools.product(R_LEVELS, LEVELS[var]))
    assert df.select("r", var).rows() == cells
    assert df["r_label"].to_list() == [R_LABELS[a] for a, _ in cells]
    assert df[f"{var}_label"].to_list() == _labels_of(var, [b for _, b in cells])


@pytest.mark.parametrize("var", KINDS)
def test_table_levels_have_the_dtype_an_estimate_gives(sample, var):
    one = sample.categorical.tabulate(var).to_polars()
    two = sample.categorical.tabulate("r", var).to_polars()
    est = sample.estimation.prop(var, by="r").to_polars()
    assert one.schema[var] == two.schema[var] == est.schema[var]
    assert two.schema["r"] == est.schema["r"]


def _labels_off(t: Table) -> Table:
    t.use_labels = False
    return t


@pytest.mark.parametrize(
    "switch_off",
    [
        lambda s: s.categorical.tabulate("r", "vi", use_labels=False).to_polars(),
        lambda s: s.categorical.tabulate("r", "vi").to_polars(use_labels=False),
        lambda s: _labels_off(s.categorical.tabulate("r", "vi")).to_polars(),
    ],
    ids=["call", "to_polars", "instance"],
)
def test_labels_off_keeps_typed_codes_only(sample, switch_off):
    df = switch_off(sample)
    assert not [c for c in df.columns if c.endswith("_label")]
    assert (df.schema["r"], df.schema["vi"]) == (pl.Int64, pl.Int64)


def test_class_default_switches_labels_off(sample, monkeypatch):
    monkeypatch.setattr(Table, "USE_LABELS", False)
    assert "vi_label" not in sample.categorical.tabulate("vi").to_polars().columns


def test_to_polars_labels_override_the_instance(sample):
    t = sample.categorical.tabulate("vi", use_labels=False)
    assert t.to_polars(use_labels=True)["vi_label"].to_list() == ["Low", "High"]


def test_unlabelled_variable_has_no_label_column(sample):
    df = sample.categorical.tabulate("u", "vi").to_polars()
    assert "u_label" not in df.columns
    assert df.columns[:3] == ["u", "vi", "vi_label"]


def test_partly_labelled_variable_falls_back_to_the_code():
    s = svy.Sample(_data(), svy.Design(wgt="w"))
    s.set_value_labels("vi", {1: "Low"})
    assert s.categorical.tabulate("vi").to_polars()["vi_label"].to_list() == ["Low", "2"]


def test_single_level_table():
    df = _data().with_columns(one=pl.lit(1))
    s = svy.Sample(df, svy.Design(wgt="w"))
    s.set_value_labels("one", {1: "Only"})
    out = s.categorical.tabulate("one").to_polars()
    assert out.select("one", "one_label").rows() == [(1, "Only")]


def test_label_column_clash_raises():
    df = _data().with_columns(vi_label=pl.col("u"))
    s = svy.Sample(df, svy.Design(wgt="w"))
    s.set_value_labels("vi", LABELS["vi"])
    with pytest.raises(EstimationError) as exc:
        s.categorical.tabulate("vi", "vi_label").to_polars()
    assert exc.value.code == "LABEL_COLUMN_CLASH"
    assert "vi_label" in str(exc.value)


def test_untidy_table_keeps_the_raw_cells(sample):
    df = sample.categorical.tabulate("r", "vi").to_polars(tidy=False)
    assert not [c for c in df.columns if c.endswith("_label")]
    assert df["rowvar"].to_list()[:2] == ["1", "1"]


# ── T-tests and rank tests ───────────────────────────────────────────────


TESTS = {
    "ttest_two": lambda s, **kw: s.categorical.ttest("y", group="g", **kw),
    "ttest_one": lambda s, **kw: s.categorical.ttest("y", mean_h0=0, **kw),
    "rank_two": lambda s, **kw: s.categorical.ranktest("y", group="g", **kw),
    "rank_k": lambda s, **kw: s.categorical.ranktest("y", group="k", score="kruskal-wallis", **kw),
}
COMPONENTS = {
    "ttest_two": ["test", "estimates"],
    "ttest_one": ["test", "estimates"],
    "rank_two": ["test"],
    "rank_k": ["test"],
}
TEST_CASES = [(t, c) for t in TESTS for c in COMPONENTS[t]]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("var", KINDS)
@pytest.mark.parametrize("test,component", TEST_CASES)
def test_by_levels_and_labels(sample, method, var, test, component):
    df = TESTS[test](sample, by=var, method=method).to_polars(component)
    i = df.columns.index(var)
    assert df.columns[i + 1] == f"{var}_label"
    assert df.schema[var] == DTYPES[var]
    assert set(df[var].to_list()) == set(LEVELS[var])
    assert df[f"{var}_label"].to_list() == _labels_of(var, df[var].to_list())


@pytest.mark.parametrize("test,component", TEST_CASES)
def test_several_by_variables_each_get_labels(sample, test, component):
    df = TESTS[test](sample, by=["vi", "vs"]).to_polars(component)
    assert df["vi_label"].to_list() == _labels_of("vi", df["vi"].to_list())
    assert df["vs_label"].to_list() == _labels_of("vs", df["vs"].to_list())


@pytest.mark.parametrize("test,component", TEST_CASES)
@pytest.mark.parametrize("where", ["call", "to_polars"])
def test_test_labels_off(sample, test, component, where):
    kw = {"use_labels": False}
    r = TESTS[test](sample, by="vi", **(kw if where == "call" else {}))
    df = r.to_polars(component, **(kw if where == "to_polars" else {}))
    assert not [c for c in df.columns if c.endswith("_label")]
    assert df.schema["vi"] == pl.Int64


def test_group_label_follows_to_polars_switch():
    s = svy.Sample(_data(), svy.Design(wgt="w"))
    s.set_value_labels("g", {1: "Men", 2: "Women"})
    t = s.categorical.ttest("y", group="g")
    assert t.to_polars("estimates")["g_label"].to_list() == ["Men", "Women"]
    assert "g_label" not in t.to_polars("estimates", use_labels=False).columns
    assert "group_level_label" not in t.to_polars("estimates", tidy=False, use_labels=False)


@pytest.mark.parametrize("test,component", TEST_CASES)
def test_untidy_rows_carry_by_label(sample, test, component):
    df = TESTS[test](sample, by="vi").to_polars(component, tidy=False)
    assert df["by_label"].to_list() == _labels_of("vi", df["by_level"].to_list())


@pytest.mark.parametrize("test,component", TEST_CASES)
def test_untidy_rows_carry_a_label_struct_with_several_by(sample, test, component):
    df = TESTS[test](sample, by=["vi", "vs"]).to_polars(component, tidy=False)
    for level, label in zip(df["by_level"].to_list(), df["by_label"].to_list()):
        assert label == {"vi": LABELS["vi"][level["vi"]], "vs": LABELS["vs"][level["vs"]]}


@pytest.mark.parametrize("test,component", TEST_CASES)
def test_unlabelled_by_has_no_label_column(sample, test, component):
    df = TESTS[test](sample, by="u").to_polars(component)
    assert "u_label" not in df.columns
    assert "by_label" not in TESTS[test](sample, by="u").to_polars(component, tidy=False).columns


def test_saved_t_test_keeps_labels_of_a_float_coded_by(sample):
    from svy.serialize import serialize

    data = serialize(sample.categorical.ttest("y", group="g", by="vf"))
    (vf,) = [v for v in data.labels if v.var == "vf"]
    assert {lv.code: lv.label for lv in vf.values} == LABELS["vf"]
