# src/svy/regression/links.py
"""
Base definitions for regression module.
"""

from __future__ import annotations

import numpy as np


# eta bound R's probit linkinv applies: -qnorm(.Machine$double.eps).
_PROBIT_ETA_MAX = 8.1258906647019042
_EPS = np.finfo(float).eps


# =============================================================================
# Link Function Math
# =============================================================================


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal density, phi(x)."""
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2.0 * np.pi)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, Phi(x). scipy is imported lazily: this module is
    loaded by ``import svy`` and scipy is not, which keeps startup cheap."""
    from scipy.special import ndtr

    return ndtr(x)


def link_inverse(link: str, eta: np.ndarray) -> np.ndarray:
    """Apply inverse link: mu = g^{-1}(eta)."""
    name = link.lower()

    if name == "identity":
        return eta

    elif name == "logit":
        return np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))

    elif name == "probit":
        return _norm_cdf(np.clip(eta, -_PROBIT_ETA_MAX, _PROBIT_ETA_MAX))

    elif name == "cloglog":
        return np.clip(-np.expm1(-np.exp(np.minimum(eta, 700.0))), _EPS, 1.0 - _EPS)

    elif name == "log":
        return np.exp(np.clip(eta, -30, 30))

    elif name == "inverse":
        return 1.0 / np.where(np.abs(eta) > 1e-10, eta, 1e-10)

    elif name == "inverse_squared":
        return 1.0 / np.sqrt(np.maximum(eta, 1e-10))

    else:
        raise ValueError(f"Unknown link: {name}")


def link_mu_eta(link: str, eta: np.ndarray) -> np.ndarray:
    """Compute d(mu)/d(eta) for delta method."""
    name = link.lower()

    if name == "identity":
        return np.ones_like(eta)

    elif name == "logit":
        mu = link_inverse(link, eta)
        return mu * (1.0 - mu)

    elif name == "probit":
        return np.maximum(_norm_pdf(eta), _EPS)

    elif name == "cloglog":
        e = np.exp(np.minimum(eta, 700.0))
        return np.maximum(e * np.exp(-e), _EPS)

    elif name == "log":
        return link_inverse(link, eta)

    elif name == "inverse":
        mu = link_inverse(link, eta)
        return -(mu * mu)

    elif name == "inverse_squared":
        mu = link_inverse(link, eta)
        return -0.5 * (mu**3)

    else:
        raise ValueError(f"Unknown link: {name}")


def link_mu_eta2(link: str, eta: np.ndarray) -> np.ndarray:
    """Compute d^2(mu)/d(eta)^2 for the AME delta method."""
    name = link.lower()

    if name == "identity":
        return np.zeros_like(eta)

    elif name == "logit":
        mu = link_inverse(link, eta)
        return mu * (1.0 - mu) * (1.0 - 2.0 * mu)

    elif name == "probit":
        return -eta * _norm_pdf(eta)

    elif name == "cloglog":
        e = np.exp(np.minimum(eta, 700.0))
        return e * np.exp(-e) * (1.0 - e)

    elif name == "log":
        return link_inverse(link, eta)

    elif name == "inverse":
        mu = link_inverse(link, eta)
        return 2.0 * (mu**3)

    elif name == "inverse_squared":
        mu = link_inverse(link, eta)
        return 0.75 * (mu**5)

    else:
        raise ValueError(f"Unknown link: {name}")


# =============================================================================
# Families and their links
# =============================================================================

DEFAULT_LINKS: dict[str, str] = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "inverse",
    "inversegaussian": "inverse_squared",
}


# Display labels, matching DistFamily's values. str.capitalize() would render
# the one compound name as "Inversegaussian".
FAMILY_LABELS: dict[str, str] = {
    "gaussian": "Gaussian",
    "binomial": "Binomial",
    "poisson": "Poisson",
    "gamma": "Gamma",
    "inversegaussian": "InverseGaussian",
}


# Which links each family admits: the `okLinks` set of R's family constructors,
# intersected with the links implemented here (R additionally offers "cauchit"
# for binomial and "sqrt" for poisson). A pairing outside the table is not a
# model, and silence about it is worse than an error: binomial + inverse squared
# converges on a meaningless fit rather than failing, while gaussian + logit only
# surfaces as "did not produce finite coefficients" from the kernel.
#
# Stricter than R in practice, which enforces okLinks only when the link is
# passed as a symbol and lets any string through — including the ones above.
FAMILY_LINKS: dict[str, frozenset[str]] = {
    "gaussian": frozenset({"identity", "log", "inverse"}),
    "binomial": frozenset({"logit", "probit", "cloglog", "log", "identity"}),
    "poisson": frozenset({"log", "identity"}),
    "gamma": frozenset({"inverse", "identity", "log"}),
    "inversegaussian": frozenset({"inverse_squared", "inverse", "identity", "log"}),
}


def resolve_link(family: str, link: str | None) -> str:
    """
    Resolve the link for a family, defaulting to the family's canonical link.

    Raises if the family does not admit the link. The link name itself is
    validated upstream by ``_normalize_link``; this check is about the pairing.
    """
    fam = family.lower()
    if link is None:
        return DEFAULT_LINKS.get(fam, "identity")

    name = link.lower()
    allowed = FAMILY_LINKS.get(fam)
    if allowed is not None and name not in allowed:
        raise ValueError(
            f"family {fam!r} does not admit link {name!r}. "
            f"Use one of: {', '.join(sorted(allowed))}."
        )
    return name
