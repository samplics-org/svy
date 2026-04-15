import pytest

from svy import SampleSize


def test_samp_size_number_wald():
    samp_size = SampleSize().estimate_mean(sigma=7000, moe=1000, deff=1.2, resp_rate=0.90)


def test_samp_size_dict_wald():
    resp_rate = {"region1": 0.95, "region2": 0.90, "region3": 0.85}
    sigma = {"region1": 7000, "region2": 11000, "region3": 5000}
    moe = {"region1": 1000, "region2": 1300, "region3": 700}
    samp_size = SampleSize().estimate_mean(sigma=sigma, moe=moe, deff=1.2, resp_rate=resp_rate)


# ============================================================
# Sample size to estimate proportion
# ============================================================


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
def test_samp_size_prop_wald(args, n0, n_fpc, n_deff, n):
    ss = SampleSize().estimate_prop(**args)

    assert ss.size.n0 == n0
    assert ss.size.n1_fpc == n_fpc
    assert ss.size.n2_deff == n_deff
    assert ss.size.n == n


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
def test_samp_size_prop_fleiss(args, n0, n_fpc, n_deff, n):
    ss = SampleSize().estimate_prop(**args)

    assert ss.size.n0 == n0
    assert ss.size.n1_fpc == n_fpc
    assert ss.size.n2_deff == n_deff
    assert ss.size.n == n


# ============================================================
# Sample size to estimate mean
# ============================================================


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
def test_samp_size_mean_wald(args, n0, n_fpc, n_deff, n):
    ss = SampleSize().estimate_mean(**args)

    assert ss.size.n0 == n0
    assert ss.size.n1_fpc == n_fpc
    assert ss.size.n2_deff == n_deff
    assert ss.size.n == n


# ============================================================
# Sample size to compare proportions
# ============================================================


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
def test_samp_size_compare_props(args, n0, n_fpc, n_deff, n):
    ss = SampleSize().compare_props(**args)

    assert ss.size.n0 == n0
    assert ss.size.n1_fpc == n_fpc
    assert ss.size.n2_deff == n_deff
    assert ss.size.n == n
