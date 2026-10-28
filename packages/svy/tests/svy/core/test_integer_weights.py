# tests/svy/core/test_integer_weights.py
"""Integer weight columns, as read from a CSV, give the results of the same weights as floats."""

import numpy as np
import polars as pl
import pytest

import svy


N_REPS = 4


@pytest.fixture
def frame():
    rng = np.random.default_rng(20261028)
    n = 40
    df = pl.DataFrame(
        {
            "y": rng.normal(10, 2, n),
            "x": rng.normal(5, 1, n),
            "g": rng.choice(["a", "b", "c"], n),
            "h": rng.choice(["u", "v"], n),
            "w": rng.integers(1, 6, n),
            "psu": np.arange(n),
        }
    )
    reps = {f"r{i}": pl.col("w") * rng.integers(0, 3, n) for i in range(1, N_REPS + 1)}
    return df.with_columns(**reps)


def _samples(df):
    """The same design with integer weights and with the weights cast to Float64."""
    rep = svy.RepWeights(method="bootstrap", prefix="r", n_reps=N_REPS)
    wcols = ["w"] + [f"r{i}" for i in range(1, N_REPS + 1)]
    as_float = df.with_columns(pl.col(wcols).cast(pl.Float64))
    return [
        svy.Sample(d, svy.Design(stratum="h", psu="psu", wgt="w", rep_wgts=rep))
        for d in (df, as_float)
    ]


@pytest.mark.parametrize("method", ["taylor", "replication"])
@pytest.mark.parametrize("cols", [("g",), ("g", "h")], ids=["oneway", "twoway"])
def test_tabulate_counts(frame, method, cols):
    ints, floats = _samples(frame)
    tab_i = ints.categorical.tabulate(*cols, units="count", method=method)
    tab_f = floats.categorical.tabulate(*cols, units="count", method=method)

    totals = frame.group_by(list(cols)).agg(pl.col("w").sum())
    expected = {tuple(r[:-1]): float(r[-1]) for r in totals.rows()}
    got = {(c.rowvar,) if len(cols) == 1 else (c.rowvar, c.colvar): c.est for c in tab_i.estimates}
    assert got == pytest.approx(expected, abs=1e-9)
    assert [c.se for c in tab_i.estimates] == pytest.approx(
        [c.se for c in tab_f.estimates], abs=1e-12
    )


@pytest.mark.parametrize(
    "controls",
    [{"x": 1500.0}, {svy.Cat("g"): {"a": 100.0, "b": 120.0, "c": 80.0}}],
    ids=["continuous", "categorical"],
)
def test_calibrate(frame, controls):
    ints, floats = _samples(frame)
    out_i = ints.weighting.calibrate(controls=controls, wgt_name="cw").data
    out_f = floats.weighting.calibrate(controls=controls, wgt_name="cw").data

    if "x" in controls:
        assert (out_i["cw"] * out_i["x"]).sum() == pytest.approx(1500.0, rel=1e-10)
    else:
        sums = dict(out_i.group_by("g").agg(pl.col("cw").sum()).rows())
        assert sums == pytest.approx({"a": 100.0, "b": 120.0, "c": 80.0}, rel=1e-10)
    cols = ["cw"] + [c for c in out_i.columns if c.startswith("cw") and c != "cw"]
    np.testing.assert_allclose(
        out_i.select(cols).to_numpy(), out_f.select(cols).to_numpy(), rtol=1e-12
    )


def test_rake_with_integer_replicate_weights(frame):
    controls = {"g": {"a": 100.0, "b": 120.0, "c": 80.0}, "h": {"u": 150.0, "v": 150.0}}
    ints, floats = _samples(frame)
    out_i = ints.weighting.rake(controls=controls, wgt_name="rw").data
    out_f = floats.weighting.rake(controls=controls, wgt_name="rw").data

    assert dict(out_i.group_by("h").agg(pl.col("rw").sum()).rows()) == pytest.approx(
        {"u": 150.0, "v": 150.0}, rel=1e-8
    )
    cols = [c for c in out_i.columns if c.startswith("rw")]
    assert len(cols) == N_REPS + 1
    np.testing.assert_allclose(
        out_i.select(cols).to_numpy(), out_f.select(cols).to_numpy(), rtol=1e-12
    )
