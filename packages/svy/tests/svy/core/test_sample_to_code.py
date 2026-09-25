# tests/svy/core/test_sample_to_code.py
"""
sample.to_code(): source that rebuilds the sample's design.

Every case runs the generated script and checks both the design and the
estimates (Taylor, and replication where the design has replicates) against
the live sample, with warnings as errors.
"""

from __future__ import annotations

import datetime as dt
import warnings

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import MethodError


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"
STYPE = {"E": 4421.0, "H": 755.0, "M": 1018.0}
SCHWIDE = {"No": 1000.0, "Yes": 5194.0}


def _run(script: str, data: pl.DataFrame | None = None) -> svy.Sample:
    ns: dict = {} if data is None else {"data": data}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exec(compile(script, "<to_code>", "exec"), ns)
    return ns["sample"]


def _estimates(s: svy.Sample, y: str, by: str | None, cat: str | None, method=None) -> list:
    e = s.estimation
    out = []
    results = [e.mean(y, method=method), e.total(y, method=method)]
    if by is not None:
        results.append(e.mean(y, by=by, method=method))
    if cat is not None:
        results.append(e.prop(cat, method=method))
    for r in results:
        rows = [(p.by_level, p.y_level, p.est, p.se) for p in r.estimates]
        out += sorted(rows, key=lambda row: (str(row[0]), str(row[1])))
    return out


def _assert_rebuilds(s: svy.Sample, y="y", by=None, cat=None) -> svy.Sample:
    back = _run(s.to_code(), s.data)
    assert back.design == s.design
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _estimates(back, y, by, cat) == _estimates(s, y, by, cat)
        if s.design.rep_wgts is not None:
            assert _estimates(back, y, by, cat, "replication") == _estimates(
                s, y, by, cat, "replication"
            )
    return back


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api() -> pl.DataFrame:
    df = pl.read_csv(DATA_DIR / "apiclus1.csv", null_values=["NA"])
    return df.with_columns(
        pl.lit(1.0).alias("one"),
        pl.when(pl.col("api00") > 650).then(pl.lit("rr")).otherwise(pl.lit("nr")).alias("resp"),
    )


@pytest.fixture(scope="module")
def base(api) -> svy.Sample:
    return svy.Sample(api, svy.Design(wgt="pw", psu="dnum", pop_size="fpc"))


@pytest.fixture(scope="module")
def strat() -> pl.DataFrame:
    """Four strata of three PSUs, plus one singleton stratum (4)."""
    rng = np.random.default_rng(3)
    rows = []
    for h in range(5):
        for p in range(1 if h == 4 else 3):
            for k in range(4):
                rows.append((h, h % 2 == 0, f"{h}-{p}", f"{h}-{p}-{k % 2}", 1 + rng.random()))
    df = pl.DataFrame(rows, schema=["h", "even", "psu", "ssu", "w"], orient="row")
    return df.with_columns(
        pl.Series("y", rng.normal(10, 2, df.height)),
        pl.Series("g", rng.choice(["a", "b"], df.height)),
        pl.Series("day", [dt.date(2020, 1, 1 + h) for h in df["h"]]),
    )


# ---------------------------------------------------------------------------
# Shape of the output
# ---------------------------------------------------------------------------


def test_short_design_is_one_line(base):
    code = base.to_code()
    assert code == (
        "import svy\n\nsample = svy.Sample(data, svy.Design(wgt='pw', psu='dnum', pop_size='fpc'))\n"
    )


def test_long_design_puts_one_argument_per_line(base):
    code = base.weighting.create_jk_wgts().to_code()
    lines = code.splitlines()
    assert lines[:3] == ["import svy", "", "sample = svy.Sample("]
    assert "    svy.Design(" in lines
    assert any(line.startswith("        rep_wgts=svy.JackknifeWgts(") for line in lines)
    assert lines[-2:] == ["    ),", ")"]


def test_repeated_coefficients_are_compact(base):
    code = base.weighting.create_jk_wgts().to_code()
    assert "rep_coefs=(0.9333333333333333,) * 15" in code


def test_output_is_deterministic_and_compiles(base):
    s = base.weighting.rake(controls={"stype": STYPE, "sch.wide": SCHWIDE}, tol=1e-12)
    assert s.to_code() == s.to_code()
    compile(s.to_code(data="x.parquet"), "<to_code>", "exec")


def test_no_metadata_in_the_script(api):
    s = svy.Sample(api, svy.Design(wgt="pw", psu="dnum"))
    s.meta.set_label("stype", "School type")
    assert "School type" not in s.to_code()


# ---------------------------------------------------------------------------
# data=
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [str, Path])
def test_parquet_path_is_read_back(base, tmp_path, kind):
    s = base.weighting.poststratify(STYPE, cells="stype")
    path = tmp_path / "survey.parquet"
    svy.write_parquet(s, path)
    script = s.to_code(data=kind(path))
    assert f"data = svy.read_parquet({str(path)!r})" in script
    back = _run(script)
    assert back.design == s.design
    assert _estimates(back, "api00", "sch.wide", "stype") == _estimates(
        s, "api00", "sch.wide", "stype"
    )


def test_uppercase_extension_and_awkward_path(base, tmp_path):
    folder = tmp_path / "o'brien \\ data"
    folder.mkdir()
    path = folder / "SURVEY.PARQUET"
    svy.write_parquet(base, path)
    back = _run(base.to_code(data=path))
    assert back.design == base.design


def test_without_data_the_script_expects_a_frame(base):
    code = base.to_code()
    assert "read_parquet" not in code
    with pytest.raises(NameError, match="data"):
        exec(code, {})


@pytest.mark.parametrize("path", ["survey.csv", "survey.dta", "survey", "survey.parquet.gz"])
def test_other_formats_are_refused_with_a_hint(base, path):
    with pytest.raises(MethodError) as exc:
        base.to_code(data=path)
    assert exc.value.code == "TO_CODE_DATA_NOT_PARQUET"
    assert "svy.write_parquet" in (exc.value.hint or "")


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------


def test_plain_and_empty_designs(strat):
    _assert_rebuilds(svy.Sample(strat, svy.Design(wgt="w")))
    _assert_rebuilds(svy.Sample(strat.drop("w")), y="y")


def test_every_design_field(strat):
    df = strat.with_columns(
        pl.int_range(pl.len()).alias("id"),
        pl.lit(1).alias("wave"),
        (1 / pl.col("w")).alias("p"),
        pl.lit(1).alias("hit"),
        pl.col("w").alias("size"),
        pl.lit(50.0).alias("N1"),
        pl.lit(200.0).alias("N2"),
    )
    d = svy.Design(
        case_id="id",
        wave="wave",
        stratum=("h", "even"),
        wgt="w",
        prob="p",
        hit="hit",
        mos="size",
        psu=("psu", "h"),
        ssu="ssu",
        pop_size=svy.PopSize(psu="N1", ssu="N2"),
    )
    s = svy.Sample(df.filter(pl.col("h") != 4), d)
    back = _assert_rebuilds(s, by="g")
    assert isinstance(back.design.pop_size, svy.PopSize)
    assert back.design.stratum == ("h", "even")


def test_with_replacement(strat):
    _assert_rebuilds(
        svy.Sample(
            strat.filter(pl.col("h") != 4), svy.Design(stratum="h", psu="psu", wgt="w", wr=True)
        )
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda s: s.weighting.create_jk_wgts(),
        lambda s: s.weighting.create_bs_wgts(n_reps=20, rstate=7),
        lambda s: s.weighting.create_sdr_wgts(),
    ],
    ids=["jackknife", "bootstrap", "sdr"],
)
def test_replicates_svy_created(base, make):
    _assert_rebuilds(make(base), y="api00", by="sch.wide", cat="stype")


def test_received_replicates_with_every_setting(strat):
    df = strat.filter(pl.col("h") != 4)
    reps = {f"r{i:02d}": df["w"] * (0.5 + (i % 3) / 2) for i in range(1, 5)}
    df = df.with_columns(**reps)
    rw = svy.BrrWgts(
        prefix="r", n_reps=4, fay_coef=0.3, scale=(0.2, 0.3, 0.2, 0.3), padding=2, df=3.5
    )
    _assert_rebuilds(
        svy.Sample(df, svy.Design(stratum="h", psu="psu", wgt="w", rep_wgts=rw)), by="g"
    )


def test_renamed_replicates(base):
    s = base.weighting.create_jk_wgts()
    s = s.wrangling.rename_rep_wgts({s.design.rep_wgts.prefix: "rep_"})
    assert "prefix='rep_'" in s.to_code()
    _assert_rebuilds(s, y="api00")


# ---------------------------------------------------------------------------
# Weight-adjustment records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adjust",
    [
        lambda s: s.weighting.poststratify(STYPE, cells="stype"),
        lambda s: s.weighting.rake(controls={"stype": STYPE, "sch.wide": SCHWIDE}, tol=1e-12),
        lambda s: s.weighting.calibrate(controls={"one": 6194.0, "api99": 3914069.0}),
        lambda s: s.weighting.standardize("stype", shares=STYPE, by="sch.wide"),
        lambda s: s.weighting.normalize(controls=100.0),
        lambda s: s.weighting.trim(upper=40.0),
        lambda s: s.weighting.adjust(resp_status="resp", cells="stype", respondents_only=False),
        lambda s: s.weighting.rake(
            controls={"stype": STYPE, "sch.wide": SCHWIDE}, tol=1e-12
        ).weighting.trim(upper=40.0),
        lambda s: s.weighting.create_jk_wgts().weighting.poststratify(STYPE, cells="stype"),
    ],
    ids=[
        "poststratify",
        "rake",
        "calibrate",
        "standardize",
        "normalize",
        "trim",
        "nonresponse",
        "rake_trim",
        "jk_poststratify",
    ],
)
def test_every_record_kind(base, adjust):
    s = adjust(base)
    assert s.design.wgt_adjustment is not None
    assert "svy.WgtAdjustment(" in s.to_code()
    _assert_rebuilds(s, y="api00", by="sch.wide", cat="stype")


def test_switching_back_to_an_earlier_weight(base):
    rk = base.weighting.rake(controls={"stype": STYPE, "sch.wide": SCHWIDE}, tol=1e-12)
    back = rk.weighting.trim(upper=40.0).update_design(wgt=rk.design.wgt)
    _assert_rebuilds(back, y="api00", by="sch.wide")


# ---------------------------------------------------------------------------
# Singleton handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        lambda s: s.singleton.collapse(using={4: 3}),
        lambda s: s.singleton.collapse(),
        lambda s: s.singleton.pool(),
        lambda s: s.singleton.certainty(),
        lambda s: s.singleton.skip(),
        lambda s: s.singleton.scale(),
        lambda s: s.singleton.center(),
    ],
    ids=["collapse_map", "collapse_default", "pool", "certainty", "skip", "scale", "center"],
)
def test_every_singleton_rule(strat, rule):
    s = rule(svy.Sample(strat, svy.Design(stratum="h", psu="psu", ssu="ssu", wgt="w")))
    assert s.design.singleton is not None
    assert "singleton=svy.SingletonSpec." in s.to_code()
    _assert_rebuilds(s, by="g")


def test_singleton_rule_on_tuple_and_date_strata(strat):
    s = svy.Sample(strat, svy.Design(stratum=("day", "even"), psu="psu", wgt="w"))
    s = s.singleton.collapse(using={(dt.date(2020, 1, 5), True): (dt.date(2020, 1, 4), False)})
    _assert_rebuilds(s, by="g")


def test_a_cleared_rule_is_not_in_the_script(strat):
    s = svy.Sample(strat, svy.Design(stratum="h", psu="psu", wgt="w")).singleton.collapse(
        using={4: 3}
    )
    with pytest.warns(UserWarning, match="cleared"):
        s = s.wrangling.filter_records(pl.col("h") != 4)
        s.estimation.mean("y")
    assert "singleton" not in s.to_code()
    _assert_rebuilds(s)


def test_singleton_with_replicates_and_record(strat):
    s = svy.Sample(strat, svy.Design(stratum="h", psu="psu", wgt="w")).singleton.collapse(
        using={4: 3}
    )
    s = s.weighting.create_bs_wgts(n_reps=10, rstate=2)
    s = s.weighting.poststratify({"a": 60.0, "b": 50.0}, cells="g")
    code = s.to_code()
    assert "rep_wgts=" in code and "wgt_adjustment=" in code and "singleton=" in code
    _assert_rebuilds(s, by="g")
