# src/svy/core/constants.py
from typing import Final


# Default width for Rich console printing and __str__ representations.
SVY_DEFAULT_PRINT_WIDTH: Final[int] = 120

SVY_PREFIX = "svy_"
SVY_PRIV_PREFIX = "__svy__"

SVY_HIT = f"{SVY_PREFIX}number_of_hits"
SVY_PROB = f"{SVY_PREFIX}prob_selection"
SVY_WEIGHT = f"{SVY_PREFIX}sample_weight"
SVY_NUMBER_OF_HITS = "svy_number_of_hits"
SVY_PROB_SELECTION = "svy_prob_selection"
SVY_CERTAINTY = "svy_certainty"

SVY_PROB_STAGE1: str = "svy_prob_selection_stage1"
SVY_PROB_STAGE2: str = "svy_prob_selection_stage2"
SVY_WGT_STAGE1: str = "svy_sample_weight_stage1"
SVY_WGT_STAGE2: str = "svy_sample_weight_stage2"
SVY_CERT_STAGE1: str = "svy_certainty_stage1"
SVY_HITS_STAGE1: str = "svy_number_of_hits_stage1"

# Columns the selection methods write under their default names.
SELECTION_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        SVY_PROB,
        SVY_WEIGHT,
        SVY_HIT,
        SVY_CERTAINTY,
        SVY_PROB_STAGE1,
        SVY_PROB_STAGE2,
        SVY_WGT_STAGE1,
        SVY_WGT_STAGE2,
        SVY_CERT_STAGE1,
        SVY_HITS_STAGE1,
    }
)

_INTERNAL_PREFIX: Final[str] = "__svy__"
_BY_SEP = "\x00\x1f\x00"  # null + unit separator + null

# `svy_*` columns are made for the user (the selection outputs); `__svy_*` are svy's.
SVY_OWN_PREFIX: Final[str] = "__svy_"
SVY_ROW_INDEX: Final[str] = "__svy_row_index__"


def key_col(group: str) -> str:
    """The column holding a design group's columns (stratum, psu, ssu, by) as one key."""
    return f"{SVY_OWN_PREFIX}{group}_key__"


SVY_VAR_STRATUM: Final[str] = "__svy_var_stratum__"
SVY_VAR_PSU: Final[str] = "__svy_var_psu__"
SVY_VAR_EXCLUDE: Final[str] = "__svy_var_exclude__"
SVY_VAR_IS_SINGLETON: Final[str] = "__svy_var_is_singleton__"

# Derived from the data and design and rebuilt on every change, so kept in
# ``sample._data`` only: never shown by ``sample.data``, never saved, and
# refused as the name of a user column.
BOOKKEEPING_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        SVY_ROW_INDEX,
        key_col("stratum"),
        key_col("psu"),
        key_col("ssu"),
        SVY_VAR_STRATUM,
        SVY_VAR_PSU,
        SVY_VAR_EXCLUDE,
        SVY_VAR_IS_SINGLETON,
    }
)


def rep_col(i: int) -> str:
    return f"{SVY_PREFIX}rep_wgt_{i:03d}"


def tmp_col(tag: str) -> str:
    return f"{SVY_PRIV_PREFIX}tmp_{tag}"


def ensure_new_col(cols: list[str], name: str) -> str:
    if name not in cols:
        return name
    i = 1
    while f"{name}_{i}" in cols:
        i += 1
    return f"{name}_{i}"
