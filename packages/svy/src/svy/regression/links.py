# src/svy/regression/base.py
"""
Base definitions for regression module.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Link Function Math
# =============================================================================


def link_inverse(link: str, eta: np.ndarray) -> np.ndarray:
    """Apply inverse link: mu = g^{-1}(eta)."""
    name = link.lower()

    if name == "identity":
        return eta

    elif name == "logit":
        return np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))

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


# =============================================================================
# Default Links
# =============================================================================

DEFAULT_LINKS: dict[str, str] = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "inverse",
    "inversegaussian": "inverse_squared",
}


def resolve_link(family: str, link: str | None) -> str:
    """Resolve link function, using canonical default if not specified."""
    if link is not None:
        return link.lower()
    return DEFAULT_LINKS.get(family.lower(), "identity")
