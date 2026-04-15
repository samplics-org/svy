import polars as pl
import pytest

from svy.core.enumerations import EstimationMethod
from svy.core.design import make_rep_weights
from svy.core.sample import SVY_ROW_INDEX, Design, RepWeights, Sample


# ---------------------------
# Fixtures
# ---------------------------


@pytest.fixture()
def df_base() -> pl.DataFrame:
    # numeric + string mix; includes 3 replicate weights
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "wgt": [1.0, 2.0, 1.5],  # float -> numeric
            "prob": [0.10, 0.20, 0.30],  # float -> numeric
            "hit": [1, 0, 1],  # int -> integer dtype
            "mos": [10.0, 20.0, 30.0],  # float -> numeric
            "strataA": ["a", "b", "a"],
            "strataB": ["x", "y", "x"],
            "psu1": ["p1", "p2", "p1"],
            "psu2": ["q1", "q2", "q1"],
            "rep1": [1.1, 1.2, 1.3],
            "rep2": [0.9, 1.0, 1.1],
            "rep3": [1.0, 0.8, 0.9],
            "text": ["t1", "t2", "t3"],  # non-numeric column for negative tests
        }
    )


@pytest.fixture()
def design_ok() -> Design:
    return Design(
        row_index="id",
        stratum=("strataA", "strataB"),
        wgt="wgt",
        prob="prob",
        hit="hit",
        mos="mos",
        psu=("psu1", "psu2"),
        ssu=None,
        pop_size=None,
        wr=False,
        rep_wgts=make_rep_weights("brr", prefix="rep", n_reps=3),
    )


# ---------------------------
# Happy path
# ---------------------------


def test_valid_design_passes(df_base, design_ok):
    s = Sample(df_base, design_ok)
    # Ensure the method was correctly attached
    assert s.design.rep_wgts.method == EstimationMethod.BRR
    assert s.design.rep_wgts.columns == ["rep1", "rep2", "rep3"]


def test_valid_design_update_method(df_base, design_ok):
    # Test updating just the method (should inherit prefix/n_reps from existing)
    new_design = design_ok.update_rep_weights(method="jackknife")
    s = Sample(df_base, new_design)
    assert s.design.rep_wgts.method == EstimationMethod.JACKKNIFE
    assert s.design.rep_wgts.n_reps == 3


# ---------------------------
# Presence checks
# ---------------------------


def test_missing_referenced_column_in_design(df_base, design_ok):
    bad = design_ok.update(wgt="does_not_exist")
    with pytest.raises(ValueError, match="not found"):
        Sample(df_base, bad)


def test_row_index_missing_in_data(df_base, design_ok):
    bad = design_ok.update(row_index="rid_not_in_df")
    with pytest.raises(ValueError, match="row_index .* not found"):
        Sample(df_base, bad)


# ---------------------------
# RepWeights: Strict Class Validation
# ---------------------------


def test_rep_weights_init_requires_method():
    # Attempting to init RepWeights without a method should fail (TypeError from msgspec)
    with pytest.raises(TypeError, match="Missing required argument 'method'"):
        RepWeights(prefix="rep", n_reps=50)


def test_rep_weights_invalid_method():
    # Taylor is not a replicate method
    with pytest.raises(ValueError, match="is not a valid replication method"):
        RepWeights(method=EstimationMethod.TAYLOR, prefix="rep", n_reps=10)


def test_rep_weights_n_reps_too_small():
    # n_reps must be >= 2
    with pytest.raises(ValueError, match="n_reps must be >= 2"):
        make_rep_weights("brr", prefix="rep", n_reps=1)


def test_rep_weights_invalid_fay_coef():
    with pytest.raises(ValueError, match="fay_coef cannot be negative"):
        make_rep_weights("brr", prefix="rep", n_reps=10, fay_coef=-0.5)


def test_rep_weights_invalid_df():
    with pytest.raises(ValueError, match="df must be > 0"):
        make_rep_weights("brr", prefix="rep", n_reps=10, df=0)


def test_rep_weights_accepts_string_method():
    rw = RepWeights(method="bootstrap", prefix="rep", n_reps=10)
    assert rw.method == EstimationMethod.BOOTSTRAP


# ---------------------------
# Design.update_rep_weights API Logic
# ---------------------------


def test_update_rep_weights_fresh_init_missing_args():
    # Starting from a blank design (no rep_wgts)
    d = Design()

    # Must provide method, prefix, and n_reps
    with pytest.raises(ValueError, match="'method' is mandatory"):
        d.update_rep_weights(prefix="rep", n_reps=10)

    with pytest.raises(ValueError, match="'n_reps' is mandatory"):
        d.update_rep_weights(method="brr", prefix="rep")


def test_update_rep_weights_clearing():
    # Setting method=None should remove the weights
    d = Design(rep_wgts=make_rep_weights("brr", prefix="r", n_reps=10))
    d_cleared = d.update_rep_weights(method=None)
    assert d_cleared.rep_wgts is None


# ---------------------------
# Sample Validation: Replicate Columns
# ---------------------------


def test_rep_weights_missing_column(df_base, design_ok):
    # design_ok has n_reps=3 -> expects rep1, rep2, rep3.
    # We update it to n_reps=4, so it looks for 'rep4' which is missing.
    bad = design_ok.update_rep_weights(n_reps=4)

    with pytest.raises(ValueError, match="columns not found in data"):
        Sample(df_base, bad)


def test_rep_weights_not_numeric(df_base, design_ok):
    # We rename 'text' column to 'rep2' to force a type error
    df_bad = df_base.with_columns(pl.col("text").alias("rep2"))

    with pytest.raises((TypeError, ValueError), match="Replicate weight columns must be numeric"):
        Sample(df_bad, design_ok)


# ---------------------------
# Weight / prob / mos / hit dtypes
# ---------------------------


def test_weight_column_not_numeric(df_base, design_ok):
    bad = design_ok.update(wgt="strataA")  # string col
    with pytest.raises(TypeError, match="Weight column .* must be numeric"):
        Sample(df_base, bad)


def test_prob_not_numeric(df_base, design_ok):
    bad = design_ok.update(prob="strataA")
    with pytest.raises(TypeError, match="'prob' column .* must be numeric"):
        Sample(df_base, bad)


def test_mos_not_numeric(df_base, design_ok):
    bad = design_ok.update(mos="strataA")
    with pytest.raises(TypeError, match="'mos' column .* must be numeric"):
        Sample(df_base, bad)


def test_hit_not_integer_dtype(df_base, design_ok):
    bad = design_ok.update(hit="wgt")  # float column
    with pytest.raises(TypeError, match="'hit' column .* must be an integer"):
        Sample(df_base, bad)


# ---------------------------
# Row index handling in Sample
# ---------------------------


def test_sample_adds_row_index_when_missing(df_base):
    # Remove the id column; rely on Sample to add its own SVY_ROW_INDEX
    df = df_base.drop("id")
    s = Sample(df, Design(row_index=None))
    assert SVY_ROW_INDEX in s._data.columns
