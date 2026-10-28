"""An unknown method name raises instead of falling back to a default."""

import polars as pl
import pytest

import svy_rs as ps


@pytest.fixture
def df():
    # Stratum "c" has one PSU, so the singleton rule changes the variance.
    return pl.DataFrame(
        {
            "y": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.0],
            "w": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.5],
            "st": ["a", "a", "b", "b", "b", "b", "c"],
            "psu": ["1", "2", "3", "4", "5", "6", "7"],
            "g": ["u", "v", "u", "v", "u", "v", "u"],
        }
    )


def _mean(df, rule):
    return ps.taylor_mean(
        df, "y", "w", strata_col="st", psu_col="psu", singleton_method=rule
    )[0]


UNKNOWN = [
    (
        lambda d: _mean(d, "centre"),
        (
            "unknown singleton method 'centre'; expected 'none', 'center' or 'scale', "
            "optionally followed by ':domains'"
        ),
    ),
    (
        lambda d: _mean(d, "center:domain"),
        "unknown singleton method 'center:domain'",
    ),
    (
        lambda d: ps.fit_glm_rs(
            y_name="y", x_names=["y"], weight_name="w", data=d, singleton_method="skip"
        ),
        "unknown singleton method 'skip'",
    ),
    (
        lambda d: ps.taylor_quantile(d, "y", "w", [0.5], quantile_method="cubic"),
        (
            "unknown quantile method 'cubic'; expected 'lower', 'higher', 'middle', 'nearest' "
            "or 'linear'"
        ),
    ),
    (
        lambda d: ps.replicate_quantile(
            d, "y", "w", ["w"], [1.0], [0.5], quantile_method="cubic"
        ),
        "unknown quantile method 'cubic'",
    ),
    (
        lambda d: ps.ranktest_rs(d, "y", "g", "w", score_method="savage"),
        (
            "unknown rank score method 'savage'; expected 'wilcoxon', 'vanderwaerden', 'median' "
            "or 'kruskalwallis'"
        ),
    ),
]


@pytest.mark.parametrize(("call", "message"), UNKNOWN)
def test_unknown_name_is_a_coded_value_error(df, call, message):
    with pytest.raises(ps.KernelValueError) as info:
        call(df)
    assert info.value.code == "INVALID_CHOICE"
    assert message in str(info.value)


@pytest.mark.parametrize(
    ("alias", "name"),
    [("adjust", "center"), ("AVERAGE", "scale"), ("None", None), ("none", None)],
)
def test_singleton_aliases_match_their_rule(df, alias, name):
    assert _mean(df, alias)["se"][0] == _mean(df, name)["se"][0]


@pytest.mark.parametrize("rule", ["none:domains", "center:DOMAINS", "scale:domains"])
def test_singleton_domains_suffix_is_accepted(df, rule):
    se = _mean(df, rule)["se"][0]
    assert se > 0
