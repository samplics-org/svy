"""`ordered=True` produces an ordered factor, rather than being ignored.

The parameter is public on `read_dta`, `read_sas`, `as_factor`,
`as_factor_expr` and `apply_value_labels`, and it did nothing. The lazy path
passed `ordering="physical"` to `pl.Categorical`, which polars deprecated in
1.32.0 and now ignores -- both packages already require a newer polars than
that, so it had no effect for anyone. The eager path never read the argument
at all; it was marked "reserved".

A Categorical sorts its categories alphabetically, so an education scale came
back Higher < None < Primary < Secondary. An Enum keeps the order it is given.
"""

import polars as pl
import pytest

from svy_io.factor import as_factor, ordered_categories
from svy_io.sas import as_factor_expr


EDUC = {0: "None", 1: "Primary", 2: "Secondary", 3: "Higher"}
EDUC_STR = {str(k): v for k, v in EDUC.items()}
IN_ORDER = ["None", "Primary", "Secondary", "Higher"]


# ---- category ordering ----------------------------------------------------


def test_categories_follow_the_codes_not_the_alphabet():
    assert ordered_categories(EDUC, levels="labels") == IN_ORDER


def test_codes_sort_numerically_not_as_text():
    """
    Readers return value labels keyed by the code's string form, so iteration
    order puts "10" between "1" and "2". Taking that as given would sequence an
    11-category scale wrongly.
    """
    as_returned = {"1": "a", "10": "j", "2": "b", "3": "c", "97": "refused"}
    assert list(as_returned) == ["1", "10", "2", "3", "97"]  # the trap

    assert ordered_categories(as_returned, levels="values") == ["1", "2", "3", "10", "97"]
    assert ordered_categories(as_returned, levels="labels") == ["a", "b", "c", "j", "refused"]


def test_non_numeric_codes_keep_string_order():
    """String codes have no numeric order; their own is the only one there is."""
    assert ordered_categories({"M": "Male", "F": "Female"}, levels="values") == ["F", "M"]


def test_negative_and_float_codes_sort_by_value():
    mapping = {"-9": "missing", "2": "two", "1.5": "one and a half", "0": "zero"}
    assert ordered_categories(mapping, levels="labels") == [
        "missing",
        "zero",
        "one and a half",
        "two",
    ]


# ---- eager path (svy_io.factor.as_factor) ---------------------------------


def test_eager_ordered_sorts_by_code():
    got = as_factor(pl.Series("v106", [3, 0, 2, 1]), labels=EDUC, levels="labels", ordered=True)

    assert isinstance(got.dtype, pl.Enum)
    assert got.sort().to_list() == IN_ORDER


def test_eager_unordered_is_still_a_categorical():
    """The default must not change: only ordered=True opts into Enum."""
    got = as_factor(pl.Series("v106", [3, 0, 2, 1]), labels=EDUC, levels="labels", ordered=False)

    assert got.dtype == pl.Categorical
    assert got.sort().to_list() == sorted(IN_ORDER)  # alphabetical


def test_eager_ordered_values_mode_orders_the_codes():
    got = as_factor(
        pl.Series("x", [10, 2, 1]), labels={1: "a", 2: "b", 10: "j"}, levels="values", ordered=True
    )

    assert got.sort().to_list() == ["1", "2", "10"]


# ---- lazy path (svy_io.sas.as_factor_expr) --------------------------------


def test_lazy_ordered_sorts_by_code():
    df = pl.DataFrame({"v106": ["3", "0", "2", "1"]})
    got = df.select(as_factor_expr("v106", value_labels=EDUC_STR, levels="labels", ordered=True))[
        "v106"
    ]

    assert isinstance(got.dtype, pl.Enum)
    assert got.sort().to_list() == IN_ORDER


def test_lazy_unordered_is_unchanged():
    df = pl.DataFrame({"v106": ["3", "0", "2", "1"]})
    got = df.select(as_factor_expr("v106", value_labels=EDUC_STR, levels="labels", ordered=False))[
        "v106"
    ]

    assert got.dtype == pl.Categorical
    assert got.sort().to_list() == sorted(IN_ORDER)


# ---- what ordered=True cannot do -----------------------------------------


@pytest.mark.parametrize("levels", ["default", "both"])
def test_ordered_rejects_modes_with_no_closed_category_set(levels):
    """
    'default' and 'both' fall back to the raw value for anything unlabelled, so
    their categories depend on data the label set cannot describe and have no
    defined order. Refusing beats silently dropping the unlabelled values.
    """
    with pytest.raises(ValueError, match="levels='labels' or 'values'"):
        as_factor(pl.Series("x", [1]), labels=EDUC, levels=levels, ordered=True)

    with pytest.raises(ValueError, match="levels='labels' or 'values'"):
        as_factor_expr("x", value_labels=EDUC_STR, levels=levels, ordered=True)


def test_ordered_without_labels_is_refused():
    """The code order is what defines the order; with no labels there is none."""
    with pytest.raises(ValueError, match="needs value labels"):
        as_factor(pl.Series("x", [1, 2]), labels=None, levels="labels", ordered=True)

    with pytest.raises(ValueError, match="needs value labels"):
        as_factor_expr("x", value_labels=None, levels="labels", ordered=True)
