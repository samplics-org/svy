"""
Comprehensive tests for SampleSize.

Coverage:
  - estimate_prop: Wald and Fleiss, unstratified and stratified
  - estimate_mean: Wald, unstratified and stratified
  - compare_props: Wald, unstratified and stratified
  - compare_means: placeholder smoke test
  - Mixed scalar/dict broadcasting
  - Identity checks (deff=1, resp_rate=1, large pop)
  - Alpha and power sensitivity
  - alloc_ratio != 1 in compare_props
  - two_sides=True vs False
  - Boundary proportions (p=0.1, 0.5, 0.9)
  - Return type and chaining
  - Key mismatch validation
  - Invalid method raises
"""

import pytest

from svy import SampleSize
from svy.size.types import Size


# =============================================================================
# Smoke / return-type tests
# =============================================================================


def test_returns_sample_size_instance():
    ss = SampleSize().estimate_prop(p=0.5, moe=0.05)
    assert isinstance(ss, SampleSize)


def test_size_is_size_instance():
    ss = SampleSize().estimate_prop(p=0.5, moe=0.05)
    assert isinstance(ss.size, Size)


def test_stratified_size_is_list_of_size():
    ss = SampleSize().estimate_prop(
        p={"r1": 0.5, "r2": 0.3},
        moe=0.05,
    )
    assert isinstance(ss.size, list)
    assert all(isinstance(s, Size) for s in ss.size)
    assert len(ss.size) == 2


def test_chaining_returns_same_instance():
    ss = SampleSize()
    result = ss.estimate_prop(p=0.5, moe=0.05)
    assert result is ss


def test_compare_means_placeholder():
    """compare_means is a placeholder — should return without error."""
    ss = SampleSize().compare_means(mu1=10.0, mu2=12.0)
    assert isinstance(ss, SampleSize)


# =============================================================================
# estimate_prop — Wald, unstratified
# =============================================================================


@pytest.mark.parametrize(
    "args,n0,n_fpc,n_deff,n",
    [
        (
            dict(p=0.8, moe=0.10, pop_size=None, deff=1.0, resp_rate=1.0),
            62,
            62,
            62,
            62,
        ),
        (
            dict(p=0.8, moe=0.10, pop_size=1000, deff=1.0, resp_rate=1.0),
            62,
            59,
            59,
            59,
        ),
        (
            dict(p=0.8, moe=0.10, pop_size=1000, deff=1.5, resp_rate=1.0),
            62,
            59,
            89,
            89,
        ),
        (
            dict(p=0.8, moe=0.10, pop_size=1000, deff=1.5, resp_rate=0.85),
            62,
            59,
            89,
            105,
        ),
    ],
    ids=[
        "inf_pop_no_deff_no_nr",
        "finite_pop_no_deff_no_nr",
        "finite_pop_deff_no_nr",
        "finite_pop_deff_nr_85",
    ],
)
def test_estimate_prop_wald(args, n0, n_fpc, n_deff, n):
    ss = SampleSize().estimate_prop(**args)
    assert ss.size.n0 == n0
    assert ss.size.n1_fpc == n_fpc
    assert ss.size.n2_deff == n_deff
    assert ss.size.n == n


# =============================================================================
# estimate_prop — Fleiss, unstratified
# =============================================================================


@pytest.mark.parametrize(
    "args,n0,n_fpc,n_deff,n",
    [
        (
            dict(
                p=0.8,
                moe=0.10,
                pop_size=None,
                method="fleiss",
                deff=1.0,
                resp_rate=1.0,
            ),
            88,
            88,
            88,
            88,
        ),
        (
            dict(
                p=0.8,
                moe=0.10,
                pop_size=1000,
                method="fleiss",
                deff=1.0,
                resp_rate=1.0,
            ),
            88,
            81,
            81,
            81,
        ),
        (
            dict(
                p=0.8,
                moe=0.10,
                pop_size=1000,
                method="fleiss",
                deff=1.5,
                resp_rate=1.0,
            ),
            88,
            81,
            122,
            122,
        ),
        (
            dict(
                p=0.8,
                moe=0.10,
                pop_size=1000,
                method="fleiss",
                deff=1.5,
                resp_rate=0.85,
            ),
            88,
            81,
            122,
            144,
        ),
    ],
    ids=[
        "inf_pop_no_deff_no_nr",
        "finite_pop_no_deff_no_nr",
        "finite_pop_deff_no_nr",
        "finite_pop_deff_nr_85",
    ],
)
def test_estimate_prop_fleiss(args, n0, n_fpc, n_deff, n):
    ss = SampleSize().estimate_prop(**args)
    assert ss.size.n0 == n0
    assert ss.size.n1_fpc == n_fpc
    assert ss.size.n2_deff == n_deff
    assert ss.size.n == n


# =============================================================================
# estimate_prop — stratified (all dicts)
# =============================================================================


def test_estimate_prop_stratified_all_dicts():
    ss = SampleSize().estimate_prop(
        p={"r1": 0.3, "r2": 0.5, "r3": 0.7},
        moe={"r1": 0.05, "r2": 0.08, "r3": 0.06},
        pop_size={"r1": 5000, "r2": 8000, "r3": 3000},
        deff=1.5,
        resp_rate=0.85,
    )
    sizes = {s.stratum: s for s in ss.size}
    assert sizes["r1"].n0 == 323
    assert sizes["r1"].n1_fpc == 304
    assert sizes["r1"].n2_deff == 456
    assert sizes["r1"].n == 537

    assert sizes["r2"].n0 == 151
    assert sizes["r2"].n1_fpc == 149
    assert sizes["r2"].n2_deff == 224
    assert sizes["r2"].n == 264

    assert sizes["r3"].n0 == 225
    assert sizes["r3"].n1_fpc == 210
    assert sizes["r3"].n2_deff == 315
    assert sizes["r3"].n == 371


def test_estimate_prop_stratified_strata_count():
    ss = SampleSize().estimate_prop(
        p={"r1": 0.3, "r2": 0.5, "r3": 0.7},
        moe=0.05,
    )
    assert len(ss.size) == 3


# =============================================================================
# estimate_prop — mixed scalar/dict broadcasting
# =============================================================================


def test_estimate_prop_mixed_pop_size_dict_others_scalar():
    """pop_size as dict, p/moe/deff/resp_rate as scalars."""
    ss = SampleSize().estimate_prop(
        p=0.5,
        moe=0.05,
        pop_size={"r1": 500, "r2": 2000, "r3": 10000},
        deff=1.2,
        resp_rate=0.90,
    )
    sizes = {s.stratum: s for s in ss.size}
    assert sizes["r1"].n0 == 385
    assert sizes["r1"].n1_fpc == 218
    assert sizes["r1"].n2_deff == 262
    assert sizes["r1"].n == 292

    assert sizes["r2"].n0 == 385
    assert sizes["r2"].n1_fpc == 323
    assert sizes["r2"].n2_deff == 388
    assert sizes["r2"].n == 432

    assert sizes["r3"].n0 == 385
    assert sizes["r3"].n1_fpc == 371
    assert sizes["r3"].n2_deff == 446
    assert sizes["r3"].n == 496


def test_estimate_prop_mixed_resp_rate_dict_others_scalar():
    """resp_rate as dict, others scalar — stratification inferred from resp_rate."""
    ss = SampleSize().estimate_prop(
        p=0.5,
        moe=0.05,
        deff=1.0,
        resp_rate={"r1": 1.0, "r2": 0.90},
    )
    sizes = {s.stratum: s for s in ss.size}
    assert sizes["r1"].n == sizes["r1"].n2_deff  # resp_rate=1 -> n == n2_deff
    assert sizes["r2"].n > sizes["r2"].n2_deff  # resp_rate<1 -> n > n2_deff


# =============================================================================
# estimate_prop — identity checks
# =============================================================================


def test_estimate_prop_deff_one_identity():
    """deff=1.0 -> n2_deff == n1_fpc."""
    ss = SampleSize().estimate_prop(p=0.5, moe=0.05, pop_size=1000, deff=1.0, resp_rate=1.0)
    assert ss.size.n2_deff == ss.size.n1_fpc


def test_estimate_prop_resp_rate_one_identity():
    """resp_rate=1.0 -> n == n2_deff."""
    ss = SampleSize().estimate_prop(p=0.5, moe=0.05, pop_size=1000, deff=1.5, resp_rate=1.0)
    assert ss.size.n == ss.size.n2_deff


def test_estimate_prop_large_pop_no_fpc():
    """Very large pop_size -> n1_fpc ~= n0."""
    ss = SampleSize().estimate_prop(p=0.5, moe=0.05, pop_size=10_000_000)
    assert ss.size.n1_fpc == ss.size.n0


def test_estimate_prop_no_pop_size_no_fpc():
    """pop_size=None -> n1_fpc == n0."""
    ss = SampleSize().estimate_prop(p=0.5, moe=0.05)
    assert ss.size.n1_fpc == ss.size.n0


# =============================================================================
# estimate_prop — alpha sensitivity
# =============================================================================


@pytest.mark.parametrize(
    "alpha,expected_n0",
    [
        (0.01, 664),
        (0.05, 385),
        (0.10, 271),
    ],
)
def test_estimate_prop_alpha_sensitivity(alpha, expected_n0):
    ss = SampleSize().estimate_prop(p=0.5, moe=0.05, alpha=alpha)
    assert ss.size.n0 == expected_n0


# =============================================================================
# estimate_prop — boundary proportions
# =============================================================================


@pytest.mark.parametrize(
    "p,expected_n0",
    [
        (0.1, 139),
        (0.5, 385),
        (0.9, 139),
    ],
)
def test_estimate_prop_boundary_proportions(p, expected_n0):
    """p=0.1 and p=0.9 are symmetric and produce the same n0."""
    ss = SampleSize().estimate_prop(p=p, moe=0.05)
    assert ss.size.n0 == expected_n0


def test_estimate_prop_symmetry():
    """p and 1-p should produce identical n0."""
    ss1 = SampleSize().estimate_prop(p=0.2, moe=0.05)
    ss2 = SampleSize().estimate_prop(p=0.8, moe=0.05)
    assert ss1.size.n0 == ss2.size.n0


# =============================================================================
# estimate_prop — monotonicity
# =============================================================================


def test_estimate_prop_smaller_moe_larger_n():
    ss1 = SampleSize().estimate_prop(p=0.5, moe=0.10)
    ss2 = SampleSize().estimate_prop(p=0.5, moe=0.05)
    assert ss2.size.n0 > ss1.size.n0


def test_estimate_prop_larger_deff_larger_n():
    ss1 = SampleSize().estimate_prop(p=0.5, moe=0.05, deff=1.0)
    ss2 = SampleSize().estimate_prop(p=0.5, moe=0.05, deff=2.0)
    assert ss2.size.n > ss1.size.n


def test_estimate_prop_lower_resp_rate_larger_n():
    ss1 = SampleSize().estimate_prop(p=0.5, moe=0.05, resp_rate=1.0)
    ss2 = SampleSize().estimate_prop(p=0.5, moe=0.05, resp_rate=0.70)
    assert ss2.size.n > ss1.size.n


def test_estimate_prop_finite_pop_smaller_n():
    ss_inf = SampleSize().estimate_prop(p=0.5, moe=0.05)
    ss_fin = SampleSize().estimate_prop(p=0.5, moe=0.05, pop_size=500)
    assert ss_fin.size.n1_fpc <= ss_inf.size.n0


# =============================================================================
# estimate_mean — Wald, unstratified
# =============================================================================


@pytest.mark.parametrize(
    "args,n0,n_fpc,n_deff,n",
    [
        (
            dict(sigma=30, moe=5, pop_size=None, deff=1.0, resp_rate=1.0),
            139,
            139,
            139,
            139,
        ),
        (
            dict(sigma=30, moe=5, pop_size=1000, deff=1.0, resp_rate=1.0),
            139,
            123,
            123,
            123,
        ),
        (
            dict(sigma=30, moe=5, pop_size=1000, deff=1.5, resp_rate=1.0),
            139,
            123,
            185,
            185,
        ),
        (
            dict(sigma=30, moe=5, pop_size=1000, deff=1.5, resp_rate=0.85),
            139,
            123,
            185,
            218,
        ),
    ],
    ids=[
        "inf_pop_no_deff_no_nr",
        "finite_pop_no_deff_no_nr",
        "finite_pop_deff_no_nr",
        "finite_pop_deff_nr_85",
    ],
)
def test_estimate_mean_wald(args, n0, n_fpc, n_deff, n):
    ss = SampleSize().estimate_mean(**args)
    assert ss.size.n0 == n0
    assert ss.size.n1_fpc == n_fpc
    assert ss.size.n2_deff == n_deff
    assert ss.size.n == n


# =============================================================================
# estimate_mean — stratified
# =============================================================================


def test_estimate_mean_stratified_all_dicts():
    ss = SampleSize().estimate_mean(
        sigma={"r1": 7000, "r2": 11000, "r3": 5000},
        moe={"r1": 1000, "r2": 1300, "r3": 700},
        deff=1.2,
        resp_rate={"r1": 0.95, "r2": 0.90, "r3": 0.85},
    )
    sizes = {s.stratum: s for s in ss.size}

    assert sizes["r1"].n0 == 189
    assert sizes["r1"].n1_fpc == 189
    assert sizes["r1"].n2_deff == 227
    assert sizes["r1"].n == 239

    assert sizes["r2"].n0 == 276
    assert sizes["r2"].n1_fpc == 276
    assert sizes["r2"].n2_deff == 332
    assert sizes["r2"].n == 369

    assert sizes["r3"].n0 == 196
    assert sizes["r3"].n1_fpc == 196
    assert sizes["r3"].n2_deff == 236
    assert sizes["r3"].n == 278


def test_estimate_mean_stratified_strata_count():
    ss = SampleSize().estimate_mean(
        sigma={"r1": 7000, "r2": 11000, "r3": 5000},
        moe=1000,
    )
    assert len(ss.size) == 3


# =============================================================================
# estimate_mean — identity and monotonicity
# =============================================================================


def test_estimate_mean_deff_one_identity():
    ss = SampleSize().estimate_mean(sigma=30, moe=5, pop_size=1000, deff=1.0, resp_rate=1.0)
    assert ss.size.n2_deff == ss.size.n1_fpc


def test_estimate_mean_resp_rate_one_identity():
    ss = SampleSize().estimate_mean(sigma=30, moe=5, deff=1.5, resp_rate=1.0)
    assert ss.size.n == ss.size.n2_deff


def test_estimate_mean_larger_sigma_larger_n():
    ss1 = SampleSize().estimate_mean(sigma=10, moe=5)
    ss2 = SampleSize().estimate_mean(sigma=30, moe=5)
    assert ss2.size.n0 > ss1.size.n0


def test_estimate_mean_smaller_moe_larger_n():
    ss1 = SampleSize().estimate_mean(sigma=30, moe=10)
    ss2 = SampleSize().estimate_mean(sigma=30, moe=5)
    assert ss2.size.n0 > ss1.size.n0


# =============================================================================
# compare_props — Wald, unstratified
# =============================================================================


@pytest.mark.parametrize(
    "args,n0,n_fpc,n_deff,n",
    [
        (
            dict(
                p1=0.65,
                p2=0.85,
                pop_size=None,
                delta=-0.10,
                two_sides=False,
                alpha=0.05,
                power=0.80,
                deff=1.0,
                resp_rate=1.0,
            ),
            (25, 25),
            (25, 25),
            (25, 25),
            (25, 25),
        ),
        (
            dict(
                p1=0.65,
                p2=0.85,
                pop_size=1000,
                delta=-0.10,
                two_sides=False,
                alpha=0.05,
                power=0.80,
                deff=1.0,
                resp_rate=1.0,
            ),
            (25, 25),
            (25, 25),
            (25, 25),
            (25, 25),
        ),
        (
            dict(
                p1=0.65,
                p2=0.85,
                pop_size=1000,
                delta=-0.10,
                two_sides=False,
                alpha=0.05,
                power=0.80,
                deff=1.5,
                resp_rate=1.0,
            ),
            (25, 25),
            (25, 25),
            (38, 38),
            (38, 38),
        ),
        (
            dict(
                p1=0.65,
                p2=0.85,
                pop_size=1000,
                delta=-0.10,
                two_sides=False,
                alpha=0.05,
                power=0.80,
                deff=1.5,
                resp_rate=0.85,
            ),
            (25, 25),
            (25, 25),
            (38, 38),
            (45, 45),
        ),
    ],
    ids=[
        "inf_pop_no_deff_no_nr",
        "finite_pop_no_deff_no_nr",
        "finite_pop_deff_no_nr",
        "finite_pop_deff_nr_85",
    ],
)
def test_compare_props_wald(args, n0, n_fpc, n_deff, n):
    ss = SampleSize().compare_props(**args)
    assert ss.size.n0 == n0
    assert ss.size.n1_fpc == n_fpc
    assert ss.size.n2_deff == n_deff
    assert ss.size.n == n


# =============================================================================
# compare_props — two_sides
# =============================================================================


def test_compare_props_two_sides_true_larger_than_false():
    """Two-sided test requires larger sample than one-sided."""
    ss_two = SampleSize().compare_props(p1=0.3, p2=0.5, two_sides=True)
    ss_one = SampleSize().compare_props(p1=0.3, p2=0.5, two_sides=False)
    assert ss_two.size.n0[0] > ss_one.size.n0[0]


def test_compare_props_two_sides_true_values():
    ss = SampleSize().compare_props(p1=0.3, p2=0.5, two_sides=True)
    assert ss.size.n0 == (91, 91)


def test_compare_props_two_sides_false_values():
    ss = SampleSize().compare_props(p1=0.3, p2=0.5, two_sides=False)
    assert ss.size.n0 == (72, 72)


# =============================================================================
# compare_props — power sensitivity
# =============================================================================


def test_compare_props_higher_power_larger_n():
    ss1 = SampleSize().compare_props(p1=0.3, p2=0.5, power=0.80)
    ss2 = SampleSize().compare_props(p1=0.3, p2=0.5, power=0.90)
    assert ss2.size.n0[0] > ss1.size.n0[0]


def test_compare_props_power_90_values():
    ss = SampleSize().compare_props(p1=0.3, p2=0.5, power=0.90)
    assert ss.size.n0 == (121, 121)


# =============================================================================
# compare_props — alloc_ratio
# =============================================================================


def test_compare_props_alloc_ratio_unequal_groups():
    """alloc_ratio=2 -> n2 = 2*n1."""
    ss = SampleSize().compare_props(p1=0.3, p2=0.5, alloc_ratio=2.0)
    n1, n2 = ss.size.n0
    assert n2 == 2 * n1


def test_compare_props_alloc_ratio_2_values():
    ss = SampleSize().compare_props(p1=0.3, p2=0.5, alloc_ratio=2.0)
    assert ss.size.n0 == (66, 132)


def test_compare_props_alloc_ratio_one_equal_groups():
    """alloc_ratio=1 -> n1 == n2."""
    ss = SampleSize().compare_props(p1=0.3, p2=0.5, alloc_ratio=1.0)
    n1, n2 = ss.size.n0
    assert n1 == n2


# =============================================================================
# compare_props — identity checks
# =============================================================================


def test_compare_props_deff_one_identity():
    ss = SampleSize().compare_props(p1=0.3, p2=0.5, pop_size=1000, deff=1.0, resp_rate=1.0)
    assert ss.size.n2_deff == ss.size.n1_fpc


def test_compare_props_resp_rate_one_identity():
    ss = SampleSize().compare_props(p1=0.3, p2=0.5, deff=1.5, resp_rate=1.0)
    assert ss.size.n == ss.size.n2_deff


# =============================================================================
# compare_props — stratified
# =============================================================================


def test_compare_props_stratified_strata_count():
    ss = SampleSize().compare_props(
        p1={"r1": 0.3, "r2": 0.4},
        p2={"r1": 0.5, "r2": 0.6},
    )
    assert len(ss.size) == 2


def test_compare_props_stratified_sizes_are_tuples():
    ss = SampleSize().compare_props(
        p1={"r1": 0.3, "r2": 0.4},
        p2={"r1": 0.5, "r2": 0.6},
    )
    for s in ss.size:
        assert isinstance(s.n0, tuple)
        assert len(s.n0) == 2


# =============================================================================
# Validation errors
# =============================================================================


def test_mismatched_dict_keys_raises():
    """Different keys across dict params should raise."""
    with pytest.raises((ValueError, KeyError)):
        SampleSize().estimate_prop(
            p={"r1": 0.5, "r2": 0.3},
            moe={"r1": 0.05, "r3": 0.05},  # r3 not in p
        )


def test_invalid_method_raises():
    """Passing an invalid method string should raise."""
    with pytest.raises((ValueError, AttributeError, TypeError)):
        SampleSize().estimate_prop(p=0.5, moe=0.05, method="invalid")


def test_fleiss_mean_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        SampleSize().estimate_mean(sigma=30, moe=5, method="fleiss")


# =============================================================================
# param property
# =============================================================================


def test_param_prop_after_estimate_prop():
    from svy.core.enumerations import PopParam

    ss = SampleSize().estimate_prop(p=0.5, moe=0.05)
    assert ss.param == PopParam.PROP


def test_param_prop_after_estimate_mean():
    from svy.core.enumerations import PopParam

    ss = SampleSize().estimate_mean(sigma=30, moe=5)
    assert ss.param == PopParam.MEAN


def test_param_none_before_any_call():
    ss = SampleSize()
    assert ss.param is None


# =============================================================================
# to_polars export
# =============================================================================


def test_to_polars_unstratified_columns():
    ss = SampleSize().estimate_prop(p=0.5, moe=0.05)
    df = ss.to_polars()
    assert set(df.columns) == {"n0", "n1_fpc", "n2_deff", "n"}


def test_to_polars_stratified_has_stratum_column():
    ss = SampleSize().estimate_prop(
        p={"r1": 0.5, "r2": 0.3},
        moe=0.05,
    )
    df = ss.to_polars()
    assert "stratum" in df.columns
    assert len(df) == 2


def test_to_polars_comparison_has_group_column():
    ss = SampleSize().compare_props(p1=0.3, p2=0.5)
    df = ss.to_polars()
    assert "group" in df.columns
    assert len(df) == 2  # g1 and g2


# =============================================================================
# group_labels
# =============================================================================


def test_compare_props_default_group_labels():
    """Default labels should be group1/group2, not g1/g2."""
    ss = SampleSize().compare_props(p1=0.3, p2=0.5)
    df = ss.to_polars()
    assert list(df["group"]) == ["group1", "group2"]


def test_compare_props_custom_group_labels():
    ss = SampleSize().compare_props(p1=0.4, p2=0.5, group_labels=["urban", "rural"])
    df = ss.to_polars()
    assert set(df["group"]) == {"urban", "rural"}


def test_compare_props_custom_labels_in_ascii():
    ss = SampleSize().compare_props(p1=0.4, p2=0.5, group_labels=["urban", "rural"])
    table = ss._format_table_ascii()
    assert "urban" in table
    assert "rural" in table


def test_compare_means_default_group_labels():
    ss = SampleSize().compare_means(mu1=10.0, mu2=12.0)
    # compare_means is a placeholder so _size is None — just check no error
    assert ss._group_labels is None


def test_compare_means_custom_group_labels_stored():
    ss = SampleSize().compare_means(mu1=10.0, mu2=12.0, group_labels=["control", "treatment"])
    assert ss._group_labels == ["control", "treatment"]


def test_to_polars_group_labels_override():
    """to_polars() group_labels param should override stored labels."""
    ss = SampleSize().compare_props(p1=0.3, p2=0.5, group_labels=["urban", "rural"])
    df = ss.to_polars(group_labels=["a", "b"])
    assert list(df["group"]) == ["a", "b"]


def test_group_labels_wrong_length_raises():
    ss = SampleSize().compare_props(p1=0.3, p2=0.5)
    with pytest.raises(ValueError):
        ss.to_polars(group_labels=["only_one"])
