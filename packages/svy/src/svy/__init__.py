# svy/__init__.py
"""Design-based analysis of complex survey data.

Bind data to its design, then call a namespace of the sample::

    sample = svy.Sample(data, svy.Design(stratum="stratum", psu="cluster", wgt="weight"))
    sample.estimation.mean("income", by="sex").to_polars()

Namespaces: ``wrangling``, ``estimation``, ``categorical``, ``glm``, ``weighting``, ``sampling``.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from svy import (  # noqa: F401  (subpackages, reached as svy.categorical, ...)
    categorical,
    enums,
    errors,
    estimation,
    extensions,
    io,
    regression,
    selection,
    serialize,
    size,
    typing,
    utils,
    weighting,
)
from svy.core import (
    RE,
    BootstrapWgts,
    BrrWgts,
    Cap,
    Cat,
    Cross,
    Design,
    JackknifeWgts,
    PopSize,
    RepWeights,
    Sample,
    SdrWgts,
    Singleton,
    all_horizontal,
    any_horizontal,
    coalesce,
    col,
    cols,
    combine_samples,
    concat_str,
    date,
    lit,
    max_horizontal,
    min_horizontal,
    sum_horizontal,
    when,
)
from svy.errors import SvyError, SvyWarning
from svy.estimation import estd
from svy.size import Power, SampleSize, TargetMean, TargetProp, TargetTwoMeans, TargetTwoProps
from svy.weighting import Threshold, TrimConfig


# ---------------------------------------------------------------------------
# Configuration & Metadata
# ---------------------------------------------------------------------------

# Ensure no “No handlers could be found” warnings in user apps
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.36.0"

__all__ = [
    # --- Modules ---
    "datasets",
    "io",
    "serialize",
    # --- Core ---
    "Sample",
    "Design",
    "PopSize",
    "Singleton",
    "combine_samples",
    # --- Replicate weights ---
    "RepWeights",
    "BootstrapWgts",
    "BrrWgts",
    "JackknifeWgts",
    "SdrWgts",
    # --- Expressions ---
    "col",
    "cols",
    "lit",
    "when",
    "coalesce",
    "concat_str",
    "date",
    "all_horizontal",
    "any_horizontal",
    "sum_horizontal",
    "min_horizontal",
    "max_horizontal",
    # --- Model terms ---
    "Cat",
    "Cross",
    "Cap",
    "RE",
    # --- Inputs ---
    "estd",
    "SampleSize",
    "Power",
    "TargetMean",
    "TargetProp",
    "TargetTwoMeans",
    "TargetTwoProps",
    "Threshold",
    "TrimConfig",
    # --- Errors ---
    "SvyError",
    "SvyWarning",
]

# Names that left the top level in 0.36.0: the error says where each one went.
_MOVED: dict[str, str] = {
    **dict.fromkeys(io.__all__, "svy.io"),
    **dict.fromkeys(
        ["Table", "TableStats", "CellEst", "TTestOneGroup", "TTestTwoGroups", "Categorical"],
        "svy.categorical",
    ),
    **dict.fromkeys(
        [
            "Estimate",
            "EstimateList",
            "ParamEst",
            "Contrast",
            "ContrastEst",
            "ContrastExpr",
            "Estimation",
        ],
        "svy.estimation",
    ),
    **dict.fromkeys(["GLM", "GLMFit", "GLMPred"], "svy.regression"),
    "Selection": "svy.selection",
    "TrimResult": "svy.weighting",
    **dict.fromkeys(enums.__all__, "svy.enums"),
    **dict.fromkeys(typing.__all__, "svy.typing"),
    **dict.fromkeys(
        ["DatasetError", "LabelError", "ModelError", "SerializationError"], "svy.errors"
    ),
    **dict.fromkeys(
        ["enable_logging", "enable_debug", "temporary_log_level", "seed_from_random_state"],
        "svy.utils",
    ),
    "register_sample_accessor": "svy.extensions",
    **dict.fromkeys(
        ["ChiSquare", "FDist", "DescribeResult", "SingletonInfo", "StratumInfo", "WgtAdjustment"],
        "svy.core",
    ),
}


# ---------------------------------------------------------------------------
# Rich integration (optional pretty printing + tracebacks)
# ---------------------------------------------------------------------------


def _maybe_install_rich() -> None:
    """
    Install Rich pretty printing + tracebacks if available.
    """
    import os
    import sys

    # Don't double-install
    if getattr(sys, "_svy_rich_installed", False):
        return

    # Decide enablement
    want = os.environ.get("SVY_RICH", "auto").lower()
    is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    under_pytest = "PYTEST_CURRENT_TEST" in os.environ

    if want in {"0", "false", "no", "off"}:
        return
    if want in {"1", "true", "yes", "on"}:
        enable = True
        explicit = True
    else:  # "auto"
        # default: enable on TTY; disable under pytest unless forced
        enable = is_tty and not under_pytest
        explicit = False

    if not enable:
        return

    try:
        from rich.console import Console
        from rich.pretty import install as pretty_install
        from rich.traceback import install as tb_install

        from svy.core.constants import SVY_DEFAULT_PRINT_WIDTH
    except ImportError:
        # Rich not installed; silently skip
        return

    try:
        env_width = os.environ.get("SVY_PRINT_WIDTH")

        if env_width:
            try:
                width = int(env_width)
            except ValueError:
                width = SVY_DEFAULT_PRINT_WIDTH
        else:
            width = SVY_DEFAULT_PRINT_WIDTH

        # This Console carries SVY_PRINT_WIDTH / SVY_DEFAULT_PRINT_WIDTH into
        # Rich's pretty-print and traceback formatters.  The install() calls
        # below retain the reference internally — do not drop this variable
        # even though it looks unused after the last install() call.
        svy_console = Console(width=width)

        # Pretty-printing only touches the interactive displayhook.
        pretty_install(console=svy_console)
        # Replacing sys.excepthook affects the whole host application (any
        # program that merely imports svy), so Rich tracebacks are installed
        # only on explicit opt-in (SVY_RICH=1) — never in "auto" mode.
        if explicit:
            tb_install(
                console=svy_console,
                show_locals=False,
                extra_lines=1,
            )
        sys._svy_rich_installed = True  # type: ignore[attr-defined]
    except Exception:
        # Never make import of svy fail just because Rich init blew up
        pass


_maybe_install_rich()


if TYPE_CHECKING:
    from svy import datasets  # noqa: F401


# Subpackages exposed as attributes but imported only when first touched.
# ``svy.datasets`` adds ~30 ms of catalog/bundled-data machinery on top of the
# polars and numpy it shares with the rest of svy, and most callers never load a
# dataset. Deferring it does not change the public API: ``svy.datasets.load(...)``,
# ``from svy import datasets`` and ``import svy.datasets`` all still work.
_LAZY_SUBMODULES = frozenset({"datasets"})


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        from importlib import import_module

        module = import_module(f"svy.{name}")
        globals()[name] = module  # later lookups skip __getattr__
        return module
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from svy.errors._api import UnknownNameError

    if name in _MOVED:
        raise UnknownNameError.moved(label="svy", name=name, home=_MOVED[name], since="0.36.0")
    raise UnknownNameError.unknown(owner="svy", label="svy", name=name, available=__all__)
