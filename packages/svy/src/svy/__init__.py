# svy/__init__.py
from __future__ import annotations

import logging

from svy.categorical import (
    Categorical,
    CellEst,
    Table,
    TableStats,
    TTestOneGroup,
    TTestTwoGroups,
)
from svy.core import (
    DF,
    DT,
    RE,
    Cap,
    CaseStyle,
    Cat,
    Category,
    ChiSquare,
    Cross,
    DescribeResult,
    Design,
    EstimationMethod,
    FDist,
    Feature,
    FitMethod,
    LetterCase,
    LinkFunction,
    MeasurementType,
    ModelType,
    Number,
    OnePropSizeMethod,
    PopParam,
    PPSMethod,
    QuantileMethod,
    RankScoreMethod,
    RepWeights,
    Sample,
    Singleton,
    SingletonHandling,
    SingletonInfo,
    SingletonResult,
    SingletonSummary,
    StratumInfo,
    TableType,
    TableUnits,
    all_horizontal,
    any_horizontal,
    coalesce,
    col,
    cols,
    concat_str,
    lit,
    max_horizontal,
    min_horizontal,
    sum_horizontal,
    when,
)
from svy.datasets import load_dataset
from svy.errors import (
    CertaintyError,
    DimensionError,
    LabelError,
    MethodError,
    ModelError,
    ProbError,
    SinglePSUError,
    SvyError,
)
from svy.estimation import Estimate, Estimation, ParamEst
from svy.extensions import register_sample_accessor
from svy.io import (
    create_from_csv,
    create_from_dta,
    create_from_parquet,
    create_from_sas,
    create_from_sav,
    create_from_spss,
    create_from_stata,
    read_csv,
    read_dta,
    read_parquet,
    read_sas,
    read_sav,
    read_spss,
    read_stata,
    scan_csv,
    scan_parquet,
    write_csv,
    write_dta,
    write_parquet,
    write_sas,
    write_sav,
    write_spss,
    write_stata,
)
from svy.regression import GLM as GLM
from svy.regression import GLMFit as GLMFit
from svy.regression import GLMPred as GLMPred
from svy.selection import Selection
from svy.size import (
    SampleSize,
    Target,
    TargetMean,
    TargetProp,
    TargetTwoMeans,
    TargetTwoProps,
)
from svy.utils import (
    RandomState,
    enable_debug,
    enable_logging,
    seed_from_random_state,
    temporary_log_level,
)
from svy.weighting import Threshold, TrimConfig, TrimResult


# ---------------------------------------------------------------------------
# Configuration & Metadata
# ---------------------------------------------------------------------------

# Ensure no “No handlers could be found” warnings in user apps
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.16.1"

__all__ = [
    # --- Core Classes ---
    "Design",
    "Sample",
    # --- Datasets ---
    "load_dataset",
    # --- Categorical Analysis ---
    "Table",
    "TableStats",
    "CellEst",
    "TTestOneGroup",
    "TTestTwoGroups",
    "Categorical",
    # --- Estimation & Stats ---
    "Estimation",
    "Estimate",
    "ParamEst",
    "ChiSquare",
    "FDist",
    "DescribeResult",
    # --- Sample Size ---
    "SampleSize",
    "Target",
    "TargetMean",
    "TargetProp",
    "TargetTwoMeans",
    "TargetTwoProps",
    # --- Sample Selection ---
    "Selection",
    # --- IO & Datasets ---
    "create_from_sas",
    "read_sas",
    "write_sas",
    "create_from_spss",
    "read_spss",
    "write_spss",
    "create_from_sav",
    "read_sav",
    "write_sav",
    "create_from_stata",
    "read_stata",
    "write_stata",
    "create_from_dta",
    "read_dta",
    "write_dta",
    "create_from_csv",
    "read_csv",
    "write_csv",
    "scan_csv",
    "create_from_parquet",
    "read_parquet",
    "scan_parquet",
    "write_parquet",
    # --- Types & Enumerations ---
    "Category",
    "DF",
    "DT",
    "Number",
    "CaseStyle",
    "EstimationMethod",
    "FitMethod",
    "LetterCase",
    "LinkFunction",
    "MeasurementType",
    "ModelType",
    "PopParam",
    "OnePropSizeMethod",
    "PPSMethod",
    "QuantileMethod",
    "RankScoreMethod",
    "RepWeights",
    "SingletonHandling",
    "Singleton",
    "SingletonInfo",
    "SingletonResult",
    "SingletonSummary",
    "StratumInfo",
    "TableType",
    "TableUnits",
    "Threshold",
    "TrimConfig",
    "TrimResult",
    # --- Errors ---
    "CertaintyError",
    "DimensionError",
    "LabelError",
    "MethodError",
    "ModelError",
    "ProbError",
    "SvyError",
    "SinglePSUError",
    # --- Expressions ---
    "col",
    "cols",
    "lit",
    "when",
    "coalesce",
    "concat_str",
    "all_horizontal",
    "any_horizontal",
    "sum_horizontal",
    "min_horizontal",
    "max_horizontal",
    # --- Terms ---
    "Cap",
    "Cat",
    "Cross",
    "Feature",
    "RE",
    # --- Utilities ---
    "enable_logging",
    "enable_debug",
    "temporary_log_level",
    "RandomState",
    "seed_from_random_state",
    # --- Extensions ---
    "register_sample_accessor",
]


# src/svy/__init__.py


def _maybe_install_rich() -> None:
    """
    Install Rich pretty printing + tracebacks if available.
    """
    import os
    import sys

    # Don’t double-install
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
    else:  # "auto"
        # default: enable on TTY; disable under pytest unless forced
        enable = is_tty and not under_pytest

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

        # Create the console with the unified width
        svy_console = Console(width=width)

        # Install using the configured console
        pretty_install(console=svy_console)
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
