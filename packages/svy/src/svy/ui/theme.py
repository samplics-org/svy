# svy/ui/theme.py
from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Dict


log = logging.getLogger(__name__)

# ---- Brand palette (from svyLab brand sheet) ----
PALETTE: Dict[str, Dict[str, str]] = {
    "brand": {
        "50": "#F0FDFA",
        "100": "#CCFBF1",
        "500": "#14B8A6",
        "600": "#0D9488",
        "700": "#0F766E",
        "800": "#115E59",
    },
    "neutral": {
        "50": "#F8FAFC",
        "100": "#F1F5F9",
        "200": "#E2E8F0",
        "300": "#CBD5E1",
        "600": "#475569",
        "900": "#0F172A",
    },
    "support": {
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
    },
}


@dataclass(frozen=True)
class Components:
    panel_border: str = "cyan"
    header_style: str = "bold"
    title_style: str = "bold"

    # Specialized components
    sample_border: str = "cyan"
    estimate_border: str = "cyan"
    ttest_border: str = "cyan"
    size_border: str = "cyan"

    # Error styles
    error_border: str = "red"
    error_header_style: str = "bold red"
    error_title_style: str = "bold red"


@dataclass(frozen=True)
class Theme:
    components: Components = Components()


THEME = Theme()


def set_theme(theme: Theme) -> None:
    """Replace the active theme."""
    global THEME
    THEME = theme
