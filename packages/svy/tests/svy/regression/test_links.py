# tests/svy/regression/test_links.py
"""
The link derivatives, against central differences of the inverse link.

`link_mu_eta` drives every delta-method standard error and `link_mu_eta2` the
average marginal effects, so a wrong arm is a wrong SE rather than a crash.
Each link is differentiated numerically over a grid inside its own domain.
"""

import numpy as np
import pytest

from svy.regression.links import (
    DEFAULT_LINKS,
    FAMILY_LINKS,
    link_inverse,
    link_mu_eta,
    link_mu_eta2,
)


# (link, eta grid). The grids stay inside each link's domain and away from the
# clamps: `log` saturates at |eta| = 30, `inverse` is undefined at 0, and
# `inverse_squared` needs eta > 0.
GRIDS: dict[str, np.ndarray] = {
    "identity": np.linspace(-3.0, 3.0, 13),
    "logit": np.linspace(-4.0, 4.0, 17),
    "probit": np.linspace(-3.0, 3.0, 13),
    "cauchit": np.linspace(-5.0, 5.0, 21),
    "cloglog": np.linspace(-3.0, 1.5, 19),
    "log": np.linspace(-3.0, 3.0, 13),
    "sqrt": np.linspace(0.2, 5.0, 13),
    "inverse": np.concatenate([np.linspace(-3.0, -0.4, 9), np.linspace(0.4, 3.0, 9)]),
    "inverse_squared": np.linspace(0.2, 3.0, 12),
}

ALL_LINKS = sorted(GRIDS)


def _central(f, eta: np.ndarray, h: float) -> np.ndarray:
    return (f(eta + h) - f(eta - h)) / (2.0 * h)


def test_every_link_in_the_family_table_is_covered():
    """A link admitted by some family but missing here would go untested."""
    admitted = set().union(*FAMILY_LINKS.values())

    assert admitted == set(ALL_LINKS)
    assert set(DEFAULT_LINKS.values()) <= admitted


@pytest.mark.parametrize("link", ALL_LINKS)
def test_mu_eta_matches_a_central_difference(link):
    eta = GRIDS[link]
    h = 1e-6

    analytic = link_mu_eta(link, eta)
    numeric = _central(lambda e: link_inverse(link, e), eta, h)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("link", ALL_LINKS)
def test_mu_eta2_matches_a_central_difference_of_mu_eta(link):
    eta = GRIDS[link]
    h = 1e-5

    analytic = link_mu_eta2(link, eta)
    numeric = _central(lambda e: link_mu_eta(link, e), eta, h)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("link", ALL_LINKS)
def test_the_derivative_arms_return_the_right_shape(link):
    """`sqrt` returns a constant; a scalar would break the delta method."""
    eta = GRIDS[link]

    assert np.shape(link_mu_eta(link, eta)) == eta.shape
    assert np.shape(link_mu_eta2(link, eta)) == eta.shape


class TestCauchitTails:
    """`arctan(e)/pi + 0.5` cancels in the tails; R's pcauchy does not."""

    def test_lower_tail_keeps_its_digits(self):
        # 1 / (pi * |eta|) to leading order, exact to ~1e-16 at eta = -1e6.
        eta = np.array([-1e3, -1e4, -1e5, -1e6])
        mu = link_inverse("cauchit", eta)

        np.testing.assert_allclose(mu, 1.0 / (np.pi * np.abs(eta)), rtol=1e-6)

    def test_upper_tail_is_the_mirror_image(self):
        eta = np.array([1e3, 1e4, 1e5, 1e6])

        np.testing.assert_allclose(
            link_inverse("cauchit", eta), 1.0 - link_inverse("cauchit", -eta), rtol=0, atol=1e-16
        )

    def test_the_naive_form_would_have_lost_them(self):
        """Guards the reason the tail branch exists."""
        eta = np.array([-1e6])
        naive = np.arctan(eta) / np.pi + 0.5

        assert abs(naive[0] / link_inverse("cauchit", eta)[0] - 1.0) > 1e-11


class TestUnknownLink:
    @pytest.mark.parametrize(
        "fn", [link_inverse, link_mu_eta, link_mu_eta2], ids=["inverse", "mu_eta", "mu_eta2"]
    )
    def test_raises(self, fn):
        with pytest.raises(ValueError, match="Unknown link"):
            fn("tanh", np.zeros(3))
