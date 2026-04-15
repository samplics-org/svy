# src/svy/core/constants.py
from typing import Final


# Default width for Rich console printing and __str__ representations.
SVY_DEFAULT_PRINT_WIDTH: Final[int] = 120

SVY_PREFIX = "svy_"
SVY_PRIV_PREFIX = "__svy__"

SVY_ROW_INDEX = f"{SVY_PREFIX}row_index"
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

_INTERNAL_PREFIX: Final[str] = "__svy__"
_INTERNAL_CONCAT_SUFFIX: Final[str] = "_svy_internal_cols_concatenated"
_BY_SEP = "\x00\x1f\x00"  # null + unit separator + null


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
