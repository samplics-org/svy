"""Kernel behaviour on degenerate input: no rows, one row, all-zero weights, singletons, ties."""

import math

import numpy as np
import polars as pl
import pytest
import svy_rs as ps


def _frame(y, w, *, strata=None, psu=None, by=None):
    cols = {"y": pl.Series(y, dtype=pl.Float64), "w": pl.Series(w, dtype=pl.Float64)}
    if strata is not None:
        cols["st"] = pl.Series(strata, dtype=pl.String)
    if psu is not None:
        cols["psu"] = pl.Series(psu, dtype=pl.String)
    if by is not None:
        cols["g"] = pl.Series(by, dtype=pl.String)
    return pl.DataFrame(cols)


def _with_ratio_denominator(df):
    return df.with_columns(x=pl.col("y") + 1.0)


def _row(result):
    return (result[0] if isinstance(result, tuple) else result).row(0, named=True)


# =============================================================================
# ESTIMATION
# =============================================================================

EMPTY = _frame([], [])
ALL_ZERO = _frame([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])

ESTIMATORS = {
    "mean": lambda d: ps.taylor_mean(d, "y", "w"),
    "ratio": lambda d: ps.taylor_ratio(_with_ratio_denominator(d), "y", "x", "w"),
    "median": lambda d: ps.taylor_median(d, "y", "w"),
    "replicate_mean": lambda d: ps.replicate_mean(
        d.with_columns(r1=pl.col("w"), r2=pl.col("w")), "y", "w", ["r1", "r2"], [0.5, 0.5]
    ),
}


@pytest.mark.parametrize("data", [EMPTY, ALL_ZERO], ids=["empty", "all_zero_weights"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_estimate_without_weight_is_nan_with_n_zero(estimator, data):
    row = _row(ESTIMATORS[estimator](data))
    assert math.isnan(row["est"])
    assert math.isnan(row["se"])
    assert row["n"] == 0


@pytest.mark.parametrize("data", [EMPTY, ALL_ZERO], ids=["empty", "all_zero_weights"])
def test_total_without_weight_is_zero(data):
    row = _row(ps.taylor_total(data, "y", "w"))
    assert (row["est"], row["se"], row["n"]) == (0.0, 0.0, 0)


def test_prop_of_empty_input_has_no_levels():
    assert ps.taylor_prop(EMPTY, "y", "w")[0].height == 0


def test_prop_with_all_zero_weights_keeps_every_level_as_nan():
    result = ps.taylor_prop(ALL_ZERO, "y", "w")[0]
    assert sorted(result["level"]) == ["1.0", "2.0", "3.0"]
    assert result["est"].is_nan().all()
    assert result["se"].is_nan().all()
    assert (result["n"] == 0).all()


def test_domain_with_all_zero_weights_keeps_a_nan_row():
    data = _frame([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 0.0, 0.0], by=["a", "a", "b", "b"])
    result = ps.taylor_mean(data, "y", "w", by_col="g")[0].sort("g")
    assert result["est"][0] == 1.5
    assert math.isnan(result["est"][1])
    assert result["n"].to_list() == [2, 0]


# One row is one PSU in the only stratum, which contributes no variance, as R's
# survey.lonely.psu = "certainty" would (R refuses to build a one-PSU design).
SINGLE = _frame([5.0], [2.0])


@pytest.mark.parametrize(
    "estimator,expected",
    [
        ("mean", 5.0),
        ("ratio", 5.0 / 6.0),
        ("median", 5.0),
        ("replicate_mean", 5.0),
    ],
)
def test_single_row_estimate_has_zero_se(estimator, expected):
    row = _row(ESTIMATORS[estimator](SINGLE))
    assert row["est"] == pytest.approx(expected, rel=1e-15)
    assert row["se"] == 0.0
    assert row["n"] == 1


def test_single_row_total_and_prop():
    total = _row(ps.taylor_total(SINGLE, "y", "w"))
    assert (total["est"], total["se"], total["df"]) == (10.0, 0.0, 0)
    prop = _row(ps.taylor_prop(SINGLE, "y", "w"))
    assert (prop["level"], prop["est"], prop["se"]) == ("5.0", 1.0, 0.0)


# Stratum c has one PSU. R survey 4.5, svymean(~y) with survey.lonely.psu set to
# "certainty", "adjust" and "average".
SINGLETON = _frame(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.0] * 5,
    strata=["a", "a", "b", "b", "c"],
    psu=["1", "2", "3", "4", "5"],
)


@pytest.mark.parametrize(
    "rule,r_se",
    [
        (None, 0.28284271247461906),
        ("center", 0.48989794855663571),
        ("scale", 0.34641016151377552),
    ],
    ids=["none_is_certainty", "center_is_adjust", "scale_is_average"],
)
def test_singleton_stratum_rules_match_r(rule, r_se):
    row = _row(
        ps.taylor_mean(SINGLETON, "y", "w", strata_col="st", psu_col="psu", singleton_method=rule)
    )
    assert row["est"] == 3.0
    assert row["se"] == pytest.approx(r_se, rel=1e-12)


TIES = _frame([1.0, 2.0, 2.0, 2.0, 3.0], [1.0] * 5)


def test_quantiles_of_tied_values():
    # cdf = [0.2, 0.4, 0.6, 0.8, 1.0]; every q-hat of 2 has F = 0.8, so the SE of
    # F(q-hat) is sqrt(0.8 * 0.2 / 4) = 0.2, as it is at q-hat = 1 with F = 0.2.
    result = ps.taylor_quantile(TIES, "y", "w", [0.2, 0.4, 0.5, 0.8], quantile_method="higher")
    assert result["est"].to_list() == [1.0, 2.0, 2.0, 2.0]
    np.testing.assert_allclose(result["se"].to_numpy(), 0.2, rtol=1e-12)


def test_linear_quantile_interpolates_into_a_tie():
    # p = 0.3 lies halfway between cdf 0.2 (y = 1) and cdf 0.4 (y = 2).
    result = ps.taylor_quantile(TIES, "y", "w", [0.3], quantile_method="linear")
    assert result["est"][0] == pytest.approx(1.5, rel=1e-15)


# =============================================================================
# WEIGHTING
# =============================================================================

ZERO_WGT = np.zeros((3, 1))
CLASSES = np.array([0, 0, 1])
CONTROLS = np.array([10.0, 5.0])


@pytest.mark.parametrize(
    "adjust",
    [
        lambda: ps.poststratify(ZERO_WGT, CLASSES, CONTROLS),
        lambda: ps.normalize(ZERO_WGT, CLASSES, CONTROLS),
        lambda: ps.normalize(ZERO_WGT),
    ],
    ids=["poststratify", "normalize_by_class", "normalize"],
)
def test_scaling_all_zero_weights_to_a_positive_target_raises(adjust):
    with pytest.raises(ps.KernelValueError, match="current sum is 0") as info:
        adjust()
    assert info.value.code == "INVALID_VALUE"


def test_raking_all_zero_weights_raises():
    with pytest.raises(ps.KernelValueError, match="All weights are zero in group 0") as info:
        ps.rake(ZERO_WGT, [CLASSES], [CONTROLS])
    assert info.value.code == "ZERO_WEIGHT"


@pytest.mark.parametrize(
    "adjust",
    [
        lambda w: ps.poststratify(w, np.array([0]), np.array([10.0])),
        lambda w: ps.rake(w, [np.array([0])], [np.array([10.0])]),
        lambda w: ps.calibrate(w, np.ones((1, 1)), np.array([10.0])),
    ],
    ids=["poststratify", "rake", "calibrate"],
)
def test_single_row_takes_the_whole_control(adjust):
    np.testing.assert_allclose(adjust(np.array([[2.0]])), [[10.0]], rtol=1e-12)


def test_normalize_single_row_to_sample_size():
    np.testing.assert_array_equal(ps.normalize(np.array([[2.0]])), [[1.0]])


@pytest.mark.parametrize(
    "adjust",
    [
        lambda w: ps.poststratify(w, np.zeros(0, dtype=np.int64), np.zeros(0)),
        lambda w: ps.normalize(w),
        lambda w: ps.adjust_nr(w, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)),
    ],
    ids=["poststratify", "normalize", "adjust_nr"],
)
def test_no_rows_give_no_weights(adjust):
    assert np.asarray(adjust(np.zeros((0, 1)))).shape == (0, 1)


def test_trimming_no_weights_gives_no_weights():
    assert ps.trim_weights(np.zeros(0), upper=1.0)[0].shape == (0,)


def test_nonresponse_single_nonrespondent_has_no_one_to_carry_its_weight():
    with pytest.raises(ps.KernelValueError) as info:
        ps.adjust_nr(np.array([[2.0]]), np.array([0]), np.array([1]))
    assert info.value.code == "NO_RESPONDENTS"


def test_calibration_keeps_a_zero_weight_at_zero():
    # Linear calibration scales the positive weights: 2 * (1 + lambda) = 10.
    out = ps.calibrate(np.array([[1.0], [0.0], [1.0]]), np.ones((3, 1)), np.array([10.0]))
    np.testing.assert_allclose(out, [[5.0], [0.0], [5.0]], rtol=1e-12)


def test_trimming_ties_at_the_cap():
    # Both 5s cap at 3 and their excess 4 goes to the two 1s in proportion: 1 + 2 each.
    out = ps.trim_weights(np.array([1.0, 5.0, 5.0, 1.0]), upper=3.0)[0]
    np.testing.assert_allclose(out, [3.0, 3.0, 3.0, 3.0], rtol=1e-12)


# =============================================================================
# SAMPLING
# =============================================================================


@pytest.mark.parametrize("n", [0, 1])
def test_srs_from_an_empty_frame_raises(n):
    with pytest.raises(ps.KernelValueError, match="empty frame") as info:
        ps.select_srs_rs(np.zeros(0, dtype=np.int64), n_scalar=n, seed=1)
    assert info.value.code == "INVALID_VALUE"


@pytest.mark.parametrize("n,expected", [(1, [7]), (0, [])])
def test_srs_from_a_single_unit_frame(n, expected):
    units, *_ = ps.select_srs_rs(np.array([7]), n_scalar=n, seed=1)
    assert list(units) == expected


def test_srs_takes_a_singleton_stratum_with_certainty():
    units, _, probs = ps.select_srs_rs(
        np.arange(4), n_map={0: 1, 1: 1}, stratum=np.array([0, 0, 0, 1]), seed=1
    )
    prob_of = dict(zip(units, probs, strict=True))
    assert prob_of.pop(3) == 1.0
    assert list(prob_of.values()) == pytest.approx([1.0 / 3.0], rel=1e-15)


def test_pps_with_all_zero_measures_of_size_raises():
    with pytest.raises(ps.KernelValueError, match=r"sum\(MOS\) must be > 0") as info:
        ps.select_pps_rs(np.arange(4), np.zeros(4), n_scalar=1, seed=1)
    assert info.value.code == "INVALID_VALUE"


def test_pps_ties_give_equal_probabilities():
    units, hits, probs, certainty = ps.select_pps_rs(
        np.arange(4), np.full(4, 2.0), n_scalar=2, seed=1
    )
    assert len(set(units)) == 2
    assert list(hits) == [1, 1]
    np.testing.assert_allclose(probs, [0.5, 0.5], rtol=1e-15)
    assert not any(certainty)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_pps_with_replacement_can_draw_more_than_the_frame(seed):
    # Three draws at p = 1/2 each: inclusion probability 1 - (1/2)^3 = 0.875.
    units, hits, probs, _ = ps.select_pps_rs(
        np.array([1, 2]), np.ones(2), n_scalar=3, method="wr", seed=seed
    )
    assert sum(hits) == 3
    np.testing.assert_allclose(probs, 0.875, rtol=1e-15)


# =============================================================================
# REPLICATE WEIGHTS
# =============================================================================

W4 = np.ones(4)
PSU4 = np.array([1, 2, 3, 4])
STRATA_WITH_SINGLETON = np.array([1, 1, 1, 2])


def test_jackknife_singleton_stratum_contributes_no_variance():
    # R survey 4.5 as.svrepdesign(type = "JKn") drops the singleton's replicate
    # and keeps the first three columns; ours stays with rscale 0, same variance.
    rep, df, rscales = ps.create_jk_wgts(W4, PSU4, STRATA_WITH_SINGLETON)
    np.testing.assert_array_equal(
        rep,
        [
            [0.0, 1.5, 1.5, 1.0],
            [1.5, 0.0, 1.5, 1.0],
            [1.5, 1.5, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
    )
    assert df == 2.0
    np.testing.assert_allclose(rscales, [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0], rtol=1e-15)


def test_bootstrap_keeps_a_singleton_stratum_at_its_weight():
    rep, _ = ps.create_bootstrap_wgts(W4, PSU4, 5, STRATA_WITH_SINGLETON, seed=1)
    np.testing.assert_array_equal(rep[3], np.ones(5))


def test_brr_rejects_a_singleton_stratum():
    with pytest.raises(ps.KernelValueError, match="stratum 2 has 1") as info:
        ps.create_brr_wgts(np.ones(3), np.array([1, 1, 2]), np.array([1, 2, 3]))
    assert info.value.code == "INVALID_VALUE"


def test_sdr_leaves_a_single_unit_unchanged():
    rep, _ = ps.create_sdr_wgts(np.array([2.0]), 4)
    np.testing.assert_array_equal(rep, [[2.0, 2.0, 2.0, 2.0]])


def test_jackknife_needs_two_psus():
    with pytest.raises(ps.KernelValueError, match="required 2, got 0") as info:
        ps.create_jk_wgts(np.zeros(0), np.zeros(0, dtype=np.int64))
    assert info.value.code == "INVALID_VALUE"


@pytest.mark.parametrize(
    "create,n_reps",
    [
        (lambda: ps.create_bootstrap_wgts(np.zeros(0), np.zeros(0, dtype=np.int64), 3, seed=1), 3),
        (lambda: ps.create_sdr_wgts(np.zeros(0), 4), 4),
    ],
    ids=["bootstrap", "sdr"],
)
def test_no_rows_give_an_empty_replicate_matrix(create, n_reps):
    rep, _ = create()
    assert rep.shape == (0, n_reps)
