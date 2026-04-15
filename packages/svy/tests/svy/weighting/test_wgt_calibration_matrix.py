# tests/svy/weighting/test_wgt_calibration_matrix.py
import numpy as np
import polars as pl
import pytest

from svy import Design, Sample


CALIB_WGT = "calib_wgt"

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _dupe_n(df: pl.DataFrame, n: int) -> pl.DataFrame:
    return pl.concat([df] * n, how="vertical")


def _unique_sorted(df: pl.DataFrame, subset: list[str], sort_by: list[str]) -> pl.DataFrame:
    return df.unique(subset=subset, keep="first").sort(sort_by)


def _as_np(series: pl.Series) -> np.ndarray:
    return series.to_numpy()


def _domain_120():
    return (["D1", "D2"] * 60)[:120]


@pytest.fixture
def sample_cat_df() -> pl.DataFrame:
    base = pl.DataFrame(
        {
            "A": ["A1", "A1", "A1", "A2", "A2", "A2", "A3", "A3", "A3", "A4", "A4", "A4"],
            "B": ["B1", "B2", "B3", "B1", "B2", "B3", "B1", "B2", "B3", "B1", "B2", "B3"],
            "wgt": pl.Series([2, 4, 4, 5, 14, 31, 10, 5, 5, 3, 10, 7], dtype=pl.Float64),
            "bench": pl.Series(
                [8, 4, 5.5, 6, 15, 34, 17, 6, 20, 5.5, 16.5, 12.5], dtype=pl.Float64
            ),
        }
    )
    df = _dupe_n(base, 10).sort(["A", "B"])
    df = df.with_columns(pl.Series("Domain", _domain_120()).cast(pl.Utf8))
    region = df["A"].map_elements(
        lambda a: "R1" if a in ("A1", "A2") else "R2", return_dtype=pl.Utf8
    )
    sex = pl.Series("Sex", ["M", "F"] * (df.height // 2) + (["M"] if df.height % 2 else []))
    return df.with_columns(region.alias("Region"), sex).select(
        ["Domain", "Region", "Sex", "A", "B", "wgt", "bench"]
    )


@pytest.fixture
def sample_num_df() -> pl.DataFrame:
    base = pl.DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "B": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "wgt": pl.Series([2, 4, 4, 5, 14, 31, 10, 5, 5, 3, 10, 7], dtype=pl.Float64),
            "bench": pl.Series(
                [8, 4, 5.5, 6, 15, 34, 17, 6, 20, 5.5, 16.5, 12.5], dtype=pl.Float64
            ),
        }
    )
    df = _dupe_n(base, 10).sort(["A", "B"])
    df = df.with_columns(pl.Series("Domain", _domain_120()).cast(pl.Utf8))
    region = df["A"].map_elements(lambda a: "R1" if a in (1, 2) else "R2", return_dtype=pl.Utf8)
    sex = pl.Series("Sex", ["M", "F"] * (df.height // 2) + (["M"] if df.height % 2 else []))
    return df.with_columns(region.alias("Region"), sex).select(
        ["Domain", "Region", "Sex", "A", "B", "wgt", "bench"]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_global_categorical_with_labels(sample_cat_df):
    s = Sample(data=sample_cat_df, design=Design(wgt="wgt"))

    X_df = s.data.select(
        pl.concat_str([pl.col("A"), pl.lit("_&_"), pl.col("B")]).alias("__cat__")
    ).to_dummies()
    X_df = X_df.rename({c: c.replace("__cat___", "") for c in X_df.columns})
    cols = sorted(X_df.columns, key=str)
    X = X_df.select(cols).to_numpy()
    labels = [tuple(c.split("_&_")) for c in cols]

    ctrl = {
        ("A1", "B1"): 80,
        ("A1", "B2"): 40,
        ("A1", "B3"): 55,
        ("A2", "B1"): 60,
        ("A2", "B2"): 150,
        ("A2", "B3"): 340,
        ("A3", "B1"): 170,
        ("A3", "B2"): 60,
        ("A3", "B3"): 200,
        ("A4", "B1"): 55,
        ("A4", "B2"): 165,
        ("A4", "B3"): 125,
    }

    w = s.weighting.calibrate_matrix(aux_vars=X, control=ctrl, labels=labels, weights_only=True)
    df = s.data.with_columns(pl.Series("_calib_wgt", w)).with_columns(
        (pl.col("_calib_wgt") / pl.col("wgt")).alias("_adj")
    )

    out = _unique_sorted(df, ["A", "B"], ["A", "B"])
    expected = np.array(
        [
            4.000000,
            1.000000,
            1.375000,
            1.200000,
            1.071429,
            1.096774,
            1.700000,
            1.200000,
            4.000000,
            1.833333,
            1.650000,
            1.785714,
        ]
    )
    assert np.isclose(_as_np(out["_adj"]), expected, rtol=1e-6, atol=1e-8).all()


def test_global_numeric_vector_control(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    X = s.data.select(["A", "B"]).to_numpy()
    ctrl = np.array([3945.0, 3355.0], dtype=float)

    w = s.weighting.calibrate_matrix(aux_vars=X, control=ctrl, weights_only=True)
    df = s.data.with_columns(pl.Series("_w", w))

    assert np.isclose(float((df["A"] * df["_w"]).sum()), ctrl[0], rtol=1e-9, atol=1e-8)
    assert np.isclose(float((df["B"] * df["_w"]).sum()), ctrl[1], rtol=1e-9, atol=1e-8)


def test_by_single_column_categorical(sample_cat_df):
    s = Sample(data=sample_cat_df, design=Design(wgt="wgt"))

    X_df = s.data.select(
        pl.concat_str([pl.col("A"), pl.lit("_&_"), pl.col("B")]).alias("__cat__")
    ).to_dummies()
    X_df = X_df.rename({c: c.replace("__cat___", "") for c in X_df.columns})
    cols = sorted(X_df.columns, key=str)
    X = X_df.select(cols).to_numpy()
    labels = [tuple(c.split("_&_")) for c in cols]

    targets = {
        ("A1", "B1"): 40,
        ("A1", "B2"): 20,
        ("A1", "B3"): 27.5,
        ("A2", "B1"): 30,
        ("A2", "B2"): 75,
        ("A2", "B3"): 170,
        ("A3", "B1"): 85,
        ("A3", "B2"): 30,
        ("A3", "B3"): 100,
        ("A4", "B1"): 27.5,
        ("A4", "B2"): 82.5,
        ("A4", "B3"): 62.5,
    }
    ctrld = {"D1": targets, "D2": targets}

    w = s.weighting.calibrate_matrix(
        aux_vars=X, control=ctrld, by="Domain", labels=labels, weights_only=True
    )
    df = s.data.with_columns(pl.Series("_w", w))

    for dom in ("D1", "D2"):
        sl = df.filter(pl.col("Domain") == dom)
        for a in ("A1", "A2", "A3", "A4"):
            for b in ("B1", "B2", "B3"):
                tot = float(sl.filter((pl.col("A") == a) & (pl.col("B") == b))["_w"].sum())
                assert np.isfinite(tot)
                assert np.isclose(tot, ctrld[dom][(a, b)], rtol=1e-8, atol=1e-8)


def test_by_multiple_columns_numeric(sample_num_df):
    """Tuple of strings for multi-column 'by'."""
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    X = s.data.select(["A", "B"]).to_numpy()
    labels = ["A", "B"]

    ctrl = {
        ("R1", "M"): {"A": 1972.5, "B": 1677.5},
        ("R1", "F"): {"A": 1972.5, "B": 1677.5},
        ("R2", "M"): {"A": 1972.5, "B": 1677.5},
        ("R2", "F"): {"A": 1972.5, "B": 1677.5},
    }

    w = s.weighting.calibrate_matrix(
        aux_vars=X, control=ctrl, by=("Region", "Sex"), labels=labels, weights_only=True
    )
    df = s.data.with_columns(pl.Series("_w", w))

    for (r, sex), targets in ctrl.items():
        sl = df.filter((pl.col("Region") == r) & (pl.col("Sex") == sex))
        assert np.isclose(float((sl["A"] * sl["_w"]).sum()), targets["A"], rtol=1e-8, atol=1e-8)
        assert np.isclose(float((sl["B"] * sl["_w"]).sum()), targets["B"], rtol=1e-8, atol=1e-8)


def test_scalar_control_rejected_for_multi_column_X(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    X = s.data.select(["A", "B"]).to_numpy()
    with pytest.raises(Exception):
        s.weighting.calibrate_matrix(aux_vars=X, control=100.0, weights_only=True)


def test_wrong_control_length_raises(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    X = s.data.select(["A", "B"]).to_numpy()
    with pytest.raises(Exception):
        s.weighting.calibrate_matrix(aux_vars=X, control=[1.0], weights_only=True)


def test_control_dict_keys_must_match_labels(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    X = s.data.select(["A", "B"]).to_numpy()
    with pytest.raises(Exception):
        s.weighting.calibrate_matrix(
            aux_vars=X, control={"A": 1.0, "C": 2.0}, labels=["A", "B"], weights_only=True
        )


def test_by_requires_mapping_control(sample_cat_df):
    s = Sample(data=sample_cat_df, design=Design(wgt="wgt"))
    X = s.data.select(pl.when(pl.col("A") == "A1").then(1.0).otherwise(0.0).alias("A1")).to_numpy()
    with pytest.raises(Exception):
        s.weighting.calibrate_matrix(aux_vars=X, control=[1.0], by="Domain", weights_only=True)


def test_shape_mismatch_X_rows_vs_data_rows(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    X = s.data.select(["A", "B"]).to_numpy()
    with pytest.raises(Exception):
        s.weighting.calibrate_matrix(aux_vars=X[:-1, :], control=[1.0, 2.0], weights_only=True)


def test_weights_only_vs_attach_column(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    X = s.data.select(["A", "B"]).to_numpy()
    ctrl = {"A": 3945.0, "B": 3355.0}

    # weights_only=True returns array without attaching
    w = s.weighting.calibrate_matrix(
        aux_vars=X, control=ctrl, labels=["A", "B"], weights_only=True
    )
    assert isinstance(w, np.ndarray)
    assert "_cal" not in s.data.columns

    # explicit wgt_name attaches column
    s2 = s.weighting.calibrate_matrix(aux_vars=X, control=ctrl, labels=["A", "B"], wgt_name="_cal")
    assert "_cal" in s2.data.columns

    # attempting to attach same name again raises
    with pytest.raises(Exception):
        s2.weighting.calibrate_matrix(aux_vars=X, control=ctrl, labels=["A", "B"], wgt_name="_cal")


def test_calibrate_matrix_auto_col_name_uses_calib_wgt(sample_num_df):
    """Without wgt_name, column name must be calib_wgt."""
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    X = s.data.select(["A", "B"]).to_numpy()
    ctrl = {"A": 3945.0, "B": 3355.0}
    s2 = s.weighting.calibrate_matrix(aux_vars=X, control=ctrl, labels=["A", "B"])
    assert CALIB_WGT in s2.data.columns
    assert "svy_calib_wgt" not in s2.data.columns


# ---------------------------------------------------------------------------
# Numeric categorical levels
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cat_num_levels_df() -> pl.DataFrame:
    base = pl.DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "B": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "wgt": pl.Series([2, 4, 4, 5, 14, 31, 10, 5, 5, 3, 10, 7], dtype=pl.Float64),
            "bench": pl.Series(
                [8, 4, 5.5, 6, 15, 34, 17, 6, 20, 5.5, 16.5, 12.5], dtype=pl.Float64
            ),
        }
    )
    df = _dupe_n(base, 10).sort(["A", "B"])
    region_int = df["A"].map_elements(lambda a: 1 if a in (1, 2) else 2, return_dtype=pl.Int64)
    sex_int = pl.Series("SexInt", [0, 1] * (df.height // 2) + ([0] if df.height % 2 else []))
    return df.with_columns(region_int.alias("RegionInt"), sex_int).select(
        ["RegionInt", "SexInt", "A", "B", "wgt", "bench"]
    )


def test_global_categorical_numeric_level_labels(sample_cat_num_levels_df):
    s = Sample(data=sample_cat_num_levels_df, design=Design(wgt="wgt"))

    X_df = s.data.select(
        pl.concat_str([pl.col("A").cast(pl.Utf8), pl.lit("_&_"), pl.col("B").cast(pl.Utf8)]).alias(
            "__cat__"
        )
    ).to_dummies()
    X_df = X_df.rename({c: c.replace("__cat___", "") for c in X_df.columns})
    cols = sorted(X_df.columns, key=str)
    X = X_df.select(cols).to_numpy()
    labels = [tuple(map(int, c.split("_&_"))) for c in cols]

    ctrl = {
        (1, 1): 80,
        (1, 2): 40,
        (1, 3): 55,
        (2, 1): 60,
        (2, 2): 150,
        (2, 3): 340,
        (3, 1): 170,
        (3, 2): 60,
        (3, 3): 200,
        (4, 1): 55,
        (4, 2): 165,
        (4, 3): 125,
    }

    w = s.weighting.calibrate_matrix(aux_vars=X, control=ctrl, labels=labels, weights_only=True)
    df = s.data.with_columns(pl.Series("_w", w)).with_columns(
        (pl.col("_w") / pl.col("wgt")).alias("_adj")
    )

    out = _unique_sorted(df, ["A", "B"], ["A", "B"])
    expected = np.array(
        [
            4.000000,
            1.000000,
            1.375000,
            1.200000,
            1.071429,
            1.096774,
            1.700000,
            1.200000,
            4.000000,
            1.833333,
            1.650000,
            1.785714,
        ]
    )
    assert np.isclose(_as_np(out["_adj"]), expected, rtol=1e-6, atol=1e-8).all()


def test_by_numeric_multi_columns_categorical(sample_cat_num_levels_df):
    s = Sample(data=sample_cat_num_levels_df, design=Design(wgt="wgt"))

    X_df = s.data.select(
        pl.concat_str([pl.col("A").cast(pl.Utf8), pl.lit("_&_"), pl.col("B").cast(pl.Utf8)]).alias(
            "__cat__"
        )
    ).to_dummies()
    X_df = X_df.rename({c: c.replace("__cat___", "") for c in X_df.columns})
    cols = sorted(X_df.columns, key=str)
    X = X_df.select(cols).to_numpy()
    labels = [tuple(map(int, c.split("_&_"))) for c in cols]

    base_targets = {
        (1, 1): 40.0,
        (1, 2): 20.0,
        (1, 3): 27.5,
        (2, 1): 30.0,
        (2, 2): 75.0,
        (2, 3): 170.0,
        (3, 1): 85.0,
        (3, 2): 30.0,
        (3, 3): 100.0,
        (4, 1): 27.5,
        (4, 2): 82.5,
        (4, 3): 62.5,
    }

    ctrld = {}
    for dom in ((1, 0), (1, 1), (2, 0), (2, 1)):
        region, _sex = dom
        allowed_A = {1, 2} if region == 1 else {3, 4}
        ctrld[dom] = {
            lab: base_targets.get(lab, 0.0) if lab[0] in allowed_A else 0.0 for lab in labels
        }

    w = s.weighting.calibrate_matrix(
        aux_vars=X,
        control=ctrld,
        by=("RegionInt", "SexInt"),
        labels=labels,
        weights_only=True,
    )
    df = s.data.with_columns(pl.Series("_w", w))

    for dom, inner in ctrld.items():
        rint, sint = dom
        sl = df.filter((pl.col("RegionInt") == rint) & (pl.col("SexInt") == sint))
        for (a, b), target in inner.items():
            tot = float(sl.filter((pl.col("A") == a) & (pl.col("B") == b))["_w"].sum())
            assert np.isfinite(tot)
            assert np.isclose(tot, target, rtol=1e-8, atol=1e-8)
