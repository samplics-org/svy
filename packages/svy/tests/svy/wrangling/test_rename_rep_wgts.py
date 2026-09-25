# tests/svy/wrangling/test_rename_rep_wgts.py
"""wrangling.rename_rep_wgts, and replicate columns identified exactly.

Replicate columns are the spec's own list (prefix + 1..n_reps with its padding,
resolved against the data), never whatever matches a pattern: look-alikes such
as ``w2023``, ``W1`` or ``w01`` next to unpadded ``w1`` are ordinary columns.
"""

import warnings

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import Design, Sample
from svy.core.repwgts import BrrWgts
from svy.errors import MethodError


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"
STYPE_POP = {"E": 4421.0, "H": 755.0, "M": 1018.0}
N_REPS = 20


@contextmanager
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _se(s, y="y"):
    return s.estimation.mean(y).to_polars()["se"][0]


def _est(s, y="y"):
    return s.estimation.mean(y).to_polars()["est"][0]


def _frame(padded: bool, look_alikes: list[str]) -> pl.DataFrame:
    rng = np.random.default_rng(11)
    n = 40
    w = rng.uniform(1, 3, n)
    cols: dict[str, object] = {
        "id": list(range(n)),
        "g": ["a", "b"] * (n // 2),
        "w": w,
        "y": rng.normal(10, 2, n),
    }
    for i in range(1, N_REPS + 1):
        cols[f"w{i:02d}" if padded else f"w{i}"] = w * rng.uniform(0.5, 1.5, n)
    for name in look_alikes:
        cols[name] = rng.normal(0, 1, n)
    return pl.DataFrame(cols)


UNPADDED_LOOK_ALIKES = ["w_base", "weight", "w2023", "w0", "w21", "W1", "w01"]
PADDED_LOOK_ALIKES = ["w_base", "weight", "w2023", "w0", "w21", "W01", "w1"]


def _received(padded: bool, look_alikes: list[str] | None = None):
    la = (PADDED_LOOK_ALIKES if padded else UNPADDED_LOOK_ALIKES) if look_alikes is None else []
    df = _frame(padded, la if look_alikes is None else look_alikes)
    return Sample(df, Design(wgt="w", rep_wgts=BrrWgts(prefix="w", n_reps=N_REPS)))


def _rep_names(prefix: str, padded: bool) -> list[str]:
    return [f"{prefix}{i:02d}" if padded else f"{prefix}{i}" for i in range(1, N_REPS + 1)]


# ---------------------------------------------------------------------------
# Exact identification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
def test_look_alikes_do_not_change_the_replicate_columns(padded):
    s = _received(padded)
    assert s.design.columns(data_columns=s.data.columns) == ["w", *_rep_names("w", padded)]


@pytest.mark.parametrize("inplace", [False, True], ids=["fork", "inplace"])
@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
def test_rename_rep_wgts_touches_only_the_replicates(padded, inplace):
    s = _received(padded)
    look = PADDED_LOOK_ALIKES if padded else UNPADDED_LOOK_ALIKES
    before = s.data.select(look)
    se, est = _se(s), _est(s)
    with no_warnings():
        out = s.wrangling.rename_rep_wgts({"w": "final_w"}, inplace=inplace)
        assert_allclose(_se(out), se)
        assert_allclose(_est(out), est)
    assert (out is s) is inplace
    assert out.data.select(look).equals(before)
    new = _rep_names("final_w", padded)
    assert all(c in out.data.columns for c in new)
    assert not any(c in out.data.columns for c in _rep_names("w", padded))
    rw = out.design.rep_wgts
    assert (rw.prefix, rw.wgt, out.design.wgt) == ("final_w", "w", "w")
    assert out.design.columns(data_columns=out.data.columns) == ["w", *new]


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
def test_renaming_a_look_alike_leaves_the_replicate_spec_alone(padded):
    s = _received(padded)
    se = _se(s)
    look = PADDED_LOOK_ALIKES if padded else UNPADDED_LOOK_ALIKES
    for name in look:
        with no_warnings():
            out = s.wrangling.rename_columns({name: f"{name}_x"})
            assert_allclose(_se(out), se)
        assert out.design.rep_wgts == s.design.rep_wgts
        assert out.data.get_column(f"{name}_x").equals(s.data.get_column(name), check_names=False)


def test_renaming_the_weight_does_not_move_its_replicates():
    s = _received(False)
    with no_warnings():
        out = s.wrangling.rename_columns({"w": "base"})
    assert out.design.wgt == "base"
    assert out.design.rep_wgts.prefix == "w"
    assert out.design.rep_wgts.wgt == "base"
    with no_warnings():
        out = out.wrangling.rename_rep_wgts({"w": "base"})
    assert out.design.rep_wgts.prefix == "base"
    assert all(f"base{i}" in out.data.columns for i in range(1, N_REPS + 1))
    assert_allclose(_se(out), _se(s))


def test_a_look_alike_target_is_a_collision_and_nothing_changes():
    s = _received(False, look_alikes=["final_w1"])
    for inplace in (False, True):
        data, design = s.data, s.design
        with pytest.raises(MethodError) as ei:
            s.wrangling.rename_rep_wgts({"w": "final_w"}, inplace=inplace)
        assert ei.value.code == "REP_WGTS_RENAME_COLLISION"
        assert "final_w1" in ei.value.detail
        assert s.data.equals(data) and s.design == design


# ---------------------------------------------------------------------------
# Every replicate method, created by svy
# ---------------------------------------------------------------------------


@pytest.fixture
def strat():
    rng = np.random.default_rng(2)
    n = 24
    return pl.DataFrame(
        {
            "id": list(range(n)),
            "s": [1] * 8 + [2] * 8 + [3] * 8,
            "p": [i // 2 for i in range(n)],
            "w": rng.uniform(1, 3, n),
            "y": rng.normal(10, 2, n),
        }
    )


@pytest.mark.parametrize("inplace", [False, True], ids=["fork", "inplace"])
@pytest.mark.parametrize(
    "create",
    [
        lambda s: s.weighting.create_bs_wgts(n_reps=6, rstate=np.random.default_rng(1)),
        lambda s: s.weighting.create_jk_wgts(),
        lambda s: s.weighting.create_brr_wgts(),
        lambda s: s.weighting.create_sdr_wgts(n_reps=4, order_col="id"),
    ],
    ids=["bootstrap", "jackknife", "brr", "sdr"],
)
def test_each_method(strat, create, inplace):
    s = create(Sample(strat, Design(wgt="w", stratum="s", psu="p")))
    rw = s.design.rep_wgts
    old = rw.columns_from_data(s.data.columns)
    se, est = _se(s), _est(s)
    with no_warnings():
        out = s.wrangling.rename_rep_wgts({rw.prefix: "rw_"}, inplace=inplace)
        assert_allclose(_se(out), se)
        assert_allclose(_est(out), est)
    new_rw = out.design.rep_wgts
    assert new_rw.prefix == "rw_"
    assert new_rw.n_reps == rw.n_reps and new_rw.coefficients() == rw.coefficients()
    assert new_rw.columns_from_data(out.data.columns) == [f"rw_{c[1:]}" for c in old]


# ---------------------------------------------------------------------------
# Several replicate sets: the design's and earlier ones
# ---------------------------------------------------------------------------


@pytest.fixture
def two_sets():
    df = pl.read_csv(DATA_DIR / "apiclus1.csv", null_values=["NA"])

    def build():
        s = Sample(df, Design(wgt="pw", psu="dnum", pop_size="fpc"))
        s = s.weighting.create_bs_wgts(n_reps=10, rstate=np.random.default_rng(4))
        return s.weighting.poststratify(STYPE_POP, cells="stype", wgt_name="ps")

    return build


def _api_se(s):
    return s.estimation.mean("api00").to_polars()["se"][0]


def test_two_sets_renamed_together(two_sets):
    s = two_sets()
    se_ps = _api_se(s)
    s.update_design(wgt="pw")
    se_pw = _api_se(s)
    s.update_design(wgt="ps")
    with no_warnings():
        s = s.wrangling.rename_rep_wgts({"pw": "base_r", "ps": "ps_r"})
        assert s.design.rep_wgts.prefix == "ps_r"
        assert_allclose(_api_se(s), se_ps)
        s.update_design(wgt="pw")
        assert s.design.rep_wgts.prefix == "base_r"
        assert_allclose(_api_se(s), se_pw)
        s.update_design(wgt="ps")
        assert s.design.rep_wgts.prefix == "ps_r"
    assert not any(c.startswith("pw") and c[2:].isdigit() for c in s.data.columns)


def test_renaming_only_a_history_set(two_sets):
    s = two_sets()
    current = s.design
    s2 = s.wrangling.rename_rep_wgts({"pw": "base_r"})
    assert s2.design == current
    assert s2.design_history[-2].rep_wgts.prefix == "base_r"
    s.update_design(wgt="pw")
    se_pw = _api_se(s)
    with no_warnings():
        s2.update_design(wgt="pw")
        assert s2.design.rep_wgts.prefix == "base_r"
        assert_allclose(_api_se(s2), se_pw)


def test_renamed_replicates_stay_protected(two_sets):
    s = two_sets().wrangling.rename_rep_wgts({"pw": "base_r", "ps": "ps_r"})
    for name, owner in (("ps_r3", "'ps'"), ("base_r3", "'pw'")):
        with pytest.raises(MethodError) as ei:
            s.wrangling.mutate({name: 0.0})
        assert ei.value.code == "WEIGHT_OVERWRITE"
        assert f"replicate weight of {owner}" in ei.value.detail


# ---------------------------------------------------------------------------
# Errors and no-ops
# ---------------------------------------------------------------------------


def _untouched(s, call):
    data, design, history = s.data, s.design, s.design_history
    for inplace in (False, True):
        with pytest.raises(MethodError) as ei:
            call(inplace)
        assert s.data.equals(data)
        assert s.design == design and s.design_history == history
    return ei.value


def test_no_replicates(strat):
    s = Sample(strat, Design(wgt="w"))
    err = _untouched(s, lambda ip: s.wrangling.rename_rep_wgts({"w": "x"}, inplace=ip))
    assert err.code == "REP_WGTS_MISSING"


def test_unknown_prefix_lists_the_known_sets(two_sets):
    s = two_sets()
    err = _untouched(s, lambda ip: s.wrangling.rename_rep_wgts({"rk": "x"}, inplace=ip))
    assert err.code == "REP_WGTS_UNKNOWN_PREFIX"
    assert "'ps' (10 replicates of 'ps')" in err.detail
    assert "'pw' (10 replicates of 'pw')" in err.detail


@pytest.mark.parametrize("bad", ["", "  ", None])
def test_empty_new_prefix(two_sets, bad):
    s = two_sets()
    err = _untouched(s, lambda ip: s.wrangling.rename_rep_wgts({"ps": bad}, inplace=ip))
    assert err.code == "REP_WGTS_PREFIX_INVALID"


def test_a_bare_string_is_refused(two_sets):
    s = two_sets()
    err = _untouched(s, lambda ip: s.wrangling.rename_rep_wgts("x", inplace=ip))
    assert err.code == "REP_WGTS_PREFIX_INVALID"


def test_two_sets_onto_one_prefix(two_sets):
    s = two_sets()
    err = _untouched(s, lambda ip: s.wrangling.rename_rep_wgts({"ps": "r", "pw": "r"}, inplace=ip))
    assert err.code == "REP_WGTS_RENAME_COLLISION"
    assert "r1" in err.detail


def test_existing_column_collision(two_sets):
    s = two_sets().wrangling.mutate({"new3": 1.0})
    err = _untouched(s, lambda ip: s.wrangling.rename_rep_wgts({"ps": "new"}, inplace=ip))
    assert err.code == "REP_WGTS_RENAME_COLLISION"
    assert "new3" in err.detail


def test_a_prefix_mapped_to_itself_is_a_no_op(two_sets):
    s = two_sets()
    with no_warnings():
        assert s.wrangling.rename_rep_wgts({"ps": "ps"}) is s
        out = s.wrangling.rename_rep_wgts({"ps": "ps", "pw": "base_r"})
    assert out.design.rep_wgts.prefix == "ps"
    assert out.design_history[-2].rep_wgts.prefix == "base_r"


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
def test_estimation_reads_only_the_replicates_next_to_look_alikes(padded):
    with_look = _received(padded)
    without = _received(padded, look_alikes=[])
    with no_warnings():
        assert_allclose(_se(with_look), _se(without))
        assert_allclose(_est(with_look), _est(without))
