# tests/svy/regression/test_glm_negative_binomial.py
"""
Negative binomial, against R.

There is no negative binomial in `survey` itself, so the reference depends on
whether the dispersion is estimated or known — and so does what svy computes.

* **theta estimated.** theta is a parameter, so the variance is the joint
  (coefficients, theta) sandwich: `survey::svymle` over the negative binomial
  likelihood, which is the method Lumley gives in *Complex Surveys*
  (Appendix E, p254ff) and what `sjstats::svyglm.nb` implements. svy reports a
  design-based standard error for theta alongside.

* **theta supplied.** theta is known, not estimated, so there is no row for it
  and the variance conditions on it — an ordinary
  `svyglm(family = MASS::negative.binomial(theta))`, and `stats.theta_se` is
  None.

The two are not interchangeable. On the model below the joint standard errors
run from 25% under to 13% over the conditional ones.

All references come from survey 4.5 / MASS 7.3-65 at `epsilon = 1e-12`, via
scripts/gen_r_reference.R.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.sample import Design, RepWeights, Sample
from svy.errors.model_errors import ModelError


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"

X_COLS = ["ell", "meals", "mobility"]
Y = "enroll"
TOL_TIGHT = 1e-12

# Both sides converge to the joint optimum to ~1e-8; `svymle` polishes with
# optim, svy with coordinate ascent, and the log-likelihoods at the two points
# differ by 1.2e-14 relative. Same bar as the rest of the R-parity suite.
RTOL = 1e-6
ATOL = 1e-6


@pytest.fixture
def api():
    return pl.read_csv(DATA_DIR / "apistrat.csv")


DESIGNS = {
    "weights_only": Design(wgt="pw"),
    "psu_only": Design(wgt="pw", psu="dnum"),
    "stratified": Design(wgt="pw", stratum="stype"),
    "psu_stratified": Design(wgt="pw", stratum="stype", psu="dnum"),
}

# ---------------------------------------------------------------------------
# theta estimated: survey::svymle over (theta, beta)
# ---------------------------------------------------------------------------

JOINT_BETA = np.array(
    [
        6.4104037649499466,
        0.0020814251367638681,
        -0.0019666487480131138,
        0.0014604152986345562,
    ]
)
JOINT_THETA = 2.7135545330416027

# design -> (theta_se, se, df)
JOINT = {
    "weights_only": (
        0.2399070291580481,
        [
            0.093016749035377044,
            0.0030115602630177256,
            0.0024071298974609212,
            0.0029125103645788273,
        ],
        196,
    ),
    "psu_only": (
        0.26724295802662734,
        [
            0.092906788671465609,
            0.0026490554271487382,
            0.0022991311056829813,
            0.0029953033257325095,
        ],
        131,
    ),
    "stratified": (
        0.21524208287373212,
        [
            0.079782145946012087,
            0.003013177317689057,
            0.0023962885351251914,
            0.0029221725287920742,
        ],
        194,
    ),
    "psu_stratified": (
        0.24371486363874204,
        [
            0.090228493595899392,
            0.0032049080839025388,
            0.0025383731623479827,
            0.0029094560930737019,
        ],
        156,
    ),
}


class TestJointAgainstSvymle:
    @pytest.mark.parametrize("design", sorted(JOINT))
    def test_matches_r(self, api, design):
        res = Sample(api, DESIGNS[design]).glm.fit(
            y=Y, x=X_COLS, family="negative_binomial", tol=TOL_TIGHT, max_iter=100
        )
        theta_se_r, se_r, df_r = JOINT[design]

        np.testing.assert_allclose([c.est for c in res.coefs], JOINT_BETA, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose([c.se for c in res.coefs], se_r, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(res.stats.theta, JOINT_THETA, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(res.stats.theta_se, theta_se_r, rtol=RTOL, atol=ATOL)
        assert res.coefs[0].wald.df == df_r

    def test_theta_matches_mass_glm_nb(self, api):
        """`MASS::glm.nb` on the same rows, prior weights pw/mean(pw)."""
        res = Sample(api, DESIGNS["weights_only"]).glm.fit(
            y=Y, x=X_COLS, family="negative_binomial", tol=TOL_TIGHT, max_iter=100
        )

        # glm.nb stops at theta.ml's default eps, which leaves theta good to
        # ~1e-8; svymle polishes it to the value asserted above.
        np.testing.assert_allclose(res.stats.theta, 2.71355451164479, rtol=1e-7)

    def test_the_dispersion_is_reported(self, api):
        res = Sample(api, DESIGNS["weights_only"]).glm.fit(
            y=Y, x=X_COLS, family="negative_binomial", tol=TOL_TIGHT
        )

        assert res.stats.theta is not None
        assert res.stats.theta_se is not None
        assert res.stats.theta_se > 0


# ---------------------------------------------------------------------------
# theta known: svyglm(family = MASS::negative.binomial(2.5))
# ---------------------------------------------------------------------------

FIXED_THETA = 2.5
FIXED_BETA = np.array(
    [
        6.4104046222622895,
        0.0020815048969284886,
        -0.0019666695801718597,
        0.0014603179082650361,
    ]
)
FIXED_SE = {
    "weights_only": [
        0.10030019390752351,
        0.002654244057929734,
        0.0023823643448716222,
        0.0039050028379986491,
    ],
    "psu_only": [
        0.098661407131972009,
        0.002306713511613111,
        0.0023429762220112016,
        0.0040170333938916126,
    ],
    "stratified": [
        0.08742494248860623,
        0.0026557857696118499,
        0.0023702494903620362,
        0.0039176480495443531,
    ],
    "psu_stratified": [
        0.098348957568028375,
        0.0028293803444440096,
        0.0025004248132958415,
        0.003898654058209639,
    ],
}


class TestFixedThetaAgainstSvyglm:
    @pytest.mark.parametrize("design", sorted(FIXED_SE))
    def test_matches_r(self, api, design):
        res = Sample(api, DESIGNS[design]).glm.fit(
            y=Y,
            x=X_COLS,
            family="negative_binomial",
            theta=FIXED_THETA,
            tol=TOL_TIGHT,
            max_iter=100,
        )

        np.testing.assert_allclose([c.est for c in res.coefs], FIXED_BETA, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            [c.se for c in res.coefs], FIXED_SE[design], rtol=1e-8, atol=1e-8
        )

    def test_a_known_theta_has_no_standard_error(self, api):
        """It is not estimated, so there is nothing to put an interval on."""
        res = Sample(api, DESIGNS["weights_only"]).glm.fit(
            y=Y, x=X_COLS, family="negative_binomial", theta=FIXED_THETA, tol=TOL_TIGHT
        )

        assert res.stats.theta == FIXED_THETA
        assert res.stats.theta_se is None

    def test_the_two_routes_disagree_materially(self, api):
        """
        The whole reason svy carries the joint sandwich: conditioning on
        theta-hat is not a cheaper way to get the same answer.
        """
        sample = Sample(api, DESIGNS["weights_only"])
        joint = sample.glm.fit(
            y=Y, x=X_COLS, family="negative_binomial", tol=TOL_TIGHT, max_iter=100
        )
        conditional = sample.glm.fit(
            y=Y,
            x=X_COLS,
            family="negative_binomial",
            theta=joint.stats.theta,
            tol=TOL_TIGHT,
        )

        ratio = np.array([c.se for c in joint.coefs]) / np.array([c.se for c in conditional.coefs])
        np.testing.assert_allclose(
            [c.est for c in joint.coefs], [c.est for c in conditional.coefs], rtol=1e-9
        )
        assert ratio.min() < 0.8, ratio
        assert ratio.max() > 1.1, ratio


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestOverdispersion:
    def test_wider_intervals_than_poisson(self, api):
        """Poisson pins the dispersion at 1; these counts are nowhere near it."""
        sample = Sample(api, DESIGNS["weights_only"])
        nb = sample.glm.fit(y=Y, x=X_COLS, family="negative_binomial", tol=TOL_TIGHT)
        pois = sample.glm.fit(y=Y, x=X_COLS, family="poisson", tol=TOL_TIGHT)

        assert pois.stats.scale > 100
        assert nb.stats.scale < 5
        assert nb.stats.theta < 10

    def test_the_dispersion_is_near_one_on_its_own_scale(self, api):
        """Pearson/(n-k) under V(mu) = mu + mu^2/theta, so ~1 when NB fits."""
        res = Sample(api, DESIGNS["weights_only"]).glm.fit(
            y=Y, x=X_COLS, family="negative_binomial", tol=TOL_TIGHT
        )

        assert 0.5 < res.stats.scale < 2.0


class TestApi:
    def test_family_aliases(self, api):
        sample = Sample(api, DESIGNS["weights_only"])
        thetas = [
            sample.glm.fit(y=Y, x=X_COLS, family=name, tol=TOL_TIGHT).stats.theta
            for name in ("negative_binomial", "negativebinomial", "nb")
        ]

        assert thetas[0] == thetas[1] == thetas[2]

    @pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
    def test_admitted_links(self, api, link):
        res = Sample(api, DESIGNS["weights_only"]).glm.fit(
            y=Y, x=X_COLS, family="negative_binomial", link=link, tol=1e-10
        )

        assert res.fitted.link == link

    def test_refuses_a_link_r_does_not(self, api):
        with pytest.raises(ValueError, match="does not admit"):
            Sample(api, DESIGNS["weights_only"]).glm.fit(
                y=Y, x=X_COLS, family="negative_binomial", link="logit"
            )

    def test_exponentiated_coefficients_are_rate_ratios(self, api):
        res = Sample(api, DESIGNS["weights_only"]).glm.fit(
            y=Y, x=X_COLS, family="negative_binomial", tol=TOL_TIGHT
        )

        assert "rate_ratio" in res.fitted.to_polars(exponentiate=True).columns


class TestRefusals:
    def test_negative_counts(self, api):
        df = api.with_columns((pl.col("enroll") - 500).alias("bad"))

        with pytest.raises(ModelError, match="Negative values"):
            Sample(df, DESIGNS["weights_only"]).glm.fit(
                y="bad", x=X_COLS, family="negative_binomial"
            )

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_theta(self, api, bad):
        with pytest.raises(ModelError, match="finite and positive"):
            Sample(api, DESIGNS["weights_only"]).glm.fit(
                y=Y, x=X_COLS, family="negative_binomial", theta=bad
            )

    def test_replicate_weights_need_a_known_theta(self, api):
        """
        The spread of replicate refits only means something if each replicate
        re-estimates theta, and none of them does.
        """
        df = api.with_columns(
            [(pl.col("pw") * (1.0 + 0.01 * i)).alias(f"rep_{i}") for i in range(1, 21)]
        )
        design = Design(
            wgt="pw",
            stratum="stype",
            psu="dnum",
            rep_wgts=RepWeights(method="Bootstrap", prefix="rep_", n_reps=20),
        )

        with pytest.raises(ModelError, match="known dispersion"):
            Sample(df, design).glm.fit(y=Y, x=X_COLS, family="negative_binomial")

    def test_replicate_weights_are_fine_with_a_known_theta(self, api):
        df = api.with_columns(
            [(pl.col("pw") * (1.0 + 0.01 * i)).alias(f"rep_{i}") for i in range(1, 21)]
        )
        design = Design(
            wgt="pw",
            stratum="stype",
            psu="dnum",
            rep_wgts=RepWeights(method="Bootstrap", prefix="rep_", n_reps=20),
        )

        res = Sample(df, design).glm.fit(
            y=Y, x=X_COLS, family="negative_binomial", theta=2.5, tol=1e-10
        )

        assert all(c.se > 0 for c in res.coefs)


class TestDomain:
    def test_where_restricts_the_fit(self, api):
        """A domain fit equals the same model on the filtered frame."""
        domain = Sample(api, DESIGNS["weights_only"]).glm.fit(
            y=Y,
            x=X_COLS,
            family="negative_binomial",
            where=svy.col("stype") == "E",
            tol=TOL_TIGHT,
        )
        filtered = Sample(api.filter(pl.col("stype") == "E"), DESIGNS["weights_only"]).glm.fit(
            y=Y, x=X_COLS, family="negative_binomial", tol=TOL_TIGHT
        )

        np.testing.assert_allclose(
            [c.est for c in domain.coefs], [c.est for c in filtered.coefs], rtol=1e-7
        )
        np.testing.assert_allclose(domain.stats.theta, filtered.stats.theta, rtol=1e-7)
