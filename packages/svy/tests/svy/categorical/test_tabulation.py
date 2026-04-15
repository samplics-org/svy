# tests/svy/core/test_tabulation.py
import builtins
import math

import polars as pl
import pytest

from svy import CellEst, Design, Sample, Table, TableType, TableUnits


@pytest.fixture
def mock_design():
    """Provides a valid instance of the Design class for testing."""
    return Design(wgt="weight", psu="psu", stratum="str")


@pytest.fixture
def df_basic():
    # 6 rows, no nulls
    return pl.DataFrame(
        {
            "row": ["r1", "r1", "r2", "r2", "r2", "r1"],
            "col": ["c1", "c2", "c1", "c1", "c2", "c2"],
            "weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "str": ["A", "A", "A", "B", "B", "B"],
            "psu": [10, 10, 20, 30, 30, 40],
            # no ssu
        }
    )


@pytest.fixture
def df_with_nulls():
    # Has nulls in row/col to test drop_missing vs assert_no_missing paths
    return pl.DataFrame(
        {
            "row": ["r1", None, "r2", "r2", "r2", "r1"],
            "col": ["c1", "c2", None, "c1", "c2", "c2"],
            "weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "str": ["A", "A", "A", "B", "B", "B"],
            "psu": [10, 10, 20, 30, 30, 40],
        }
    )


# def test_table_show_uses_rich(capsys, df_basic, mock_design):
#     tbl = Sample(data=df_basic, design=mock_design).categorical.tabulate(
#         rowvar="row", colvar="col", drop_nulls=True
#     )
#     out = capsys.readouterr().out
#     assert "Estimate" in out


def test_plain_fallback_when_rich_missing(monkeypatch, capsys):
    # Make importing 'rich' fail inside your rendering path
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("rich"):
            raise ImportError("simulate missing rich")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    tbl = Table(
        type=TableType.TWO_WAY,
        rowvar="r",
        colvar="c",
        estimates=[CellEst("A", "X", 0.1, 0.01, 0.08, 0.05, 0.15)],
    )

    # If you have a .show() helper, use it; otherwise print(str(tbl))
    if hasattr(tbl, "show"):
        tbl.show(use_rich=True)
        out = capsys.readouterr().out
    else:
        out = str(tbl)

    assert "Table(" in out or "Row" in out  # some plain-text evidence


def test_crosstab_prop(df_basic, mock_design):
    # proportions are the default; the call should succeed as-is
    sample = Sample(data=df_basic, design=mock_design)
    _ = sample.categorical.tabulate(rowvar="row", colvar="col", drop_nulls=True)

    # Explicit PROPORTION still works
    _ = sample.categorical.tabulate(
        rowvar="row", colvar="col", drop_nulls=True, units=TableUnits.PROPORTION
    )


## Count
@pytest.fixture
def df_basic_count():
    # 12 rows, no nulls (heavier weights to exercise scaling)
    return pl.DataFrame(
        {
            "row": ["r1", "r1", "r2", "r2", "r2", "r1", "r1", "r1", "r2", "r2", "r2", "r1"],
            "col": ["c1", "c2", "c1", "c1", "c2", "c2", "c1", "c2", "c1", "c1", "c2", "c2"],
            "weight": [11.0, 23.0, 23.0, 14.0, 15.0, 65.0, 17.0, 22.0, 38.0, 40.0, 57.0, 65.0],
            "str": ["A", "A", "A", "B", "B", "B", "A", "A", "A", "B", "B", "B"],
            "psu": [10, 10, 20, 30, 30, 40, 10, 10, 20, 30, 30, 40],
            # no ssu
        }
    )


def test_crosstab_count(df_basic_count, mock_design):
    sample = Sample(data=df_basic_count, design=mock_design)
    result = sample.categorical.tabulate(rowvar="row", colvar="col", drop_nulls=True)
    result


def test_tabulate_count(df_basic_count, mock_design):
    sample = Sample(data=df_basic_count, design=mock_design)
    result = sample.categorical.tabulate(rowvar="row", drop_nulls=True)
    result


# ------ New unit scaling tests -------------------------------------------------


def _sum_estimates(tbl: Table) -> float:
    return float(sum(float(c.est) for c in (tbl.estimates or [])))


def test_units_percent_one_way_sums_to_100(df_basic, mock_design):
    sample = Sample(data=df_basic, design=mock_design)
    tbl = sample.categorical.tabulate(rowvar="row", units=TableUnits.PERCENT, drop_nulls=True)
    assert math.isclose(_sum_estimates(tbl), 100.0, rel_tol=1e-12, abs_tol=1e-9)


def test_units_count_one_way_defaults_to_sum_weight(df_basic, mock_design):
    sample = Sample(data=df_basic, design=mock_design)
    sumw = float(df_basic["weight"].sum())
    tbl = sample.categorical.tabulate(rowvar="row", units=TableUnits.COUNT, drop_nulls=True)
    assert math.isclose(_sum_estimates(tbl), sumw, rel_tol=1e-12, abs_tol=1e-9)


def test_units_count_one_way_custom_total(df_basic, mock_design):
    sample = Sample(data=df_basic, design=mock_design)
    target = 1_000.0
    tbl = sample.categorical.tabulate(
        rowvar="row", units=TableUnits.COUNT, count_total=target, drop_nulls=True
    )
    assert math.isclose(_sum_estimates(tbl), target, rel_tol=1e-12, abs_tol=1e-9)


@pytest.fixture
def df_multikey():
    # Construct a small dataset with two-key strata and two-key PSUs.
    # This used to trigger IndexError when S/P were 2-D (boolean mask mismatch).
    return pl.DataFrame(
        {
            "row": ["r1", "r1", "r2", "r2", "r2", "r1", "r1", "r2"],
            "col": ["c1", "c2", "c1", "c1", "c2", "c2", "c1", "c2"],
            "weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            # two-key stratum
            "s1": ["A", "A", "A", "B", "B", "B", "A", "B"],
            "s2": ["x", "x", "y", "y", "y", "x", "y", "x"],
            # two-key psu
            "p1": [10, 10, 20, 30, 30, 40, 20, 30],
            "p2": [1, 1, 2, 3, 3, 4, 2, 3],
        }
    )


def test_tabulate_two_way_with_multikey_stratum_psu(df_multikey):
    """
    Verifies that two-way tabulation works when both stratum and psu are
    specified as *multiple columns* (2-D labels).
    """
    design = Design(wgt="weight", stratum=("s1", "s2"), psu=("p1", "p2"))
    sample = Sample(data=df_multikey, design=design)

    tbl = sample.categorical.tabulate(rowvar="row", colvar="col", drop_nulls=True)
    assert tbl.estimates is not None

    # Basic sanity checks: no exception, right table type, non-empty estimates
    assert tbl.type == TableType.TWO_WAY
    assert len(tbl.estimates) > 0


def test_tabulate_one_way_with_multikey_psu_single_stratum(df_multikey):
    """
    Verifies single-stratum path (S=None) with a *multi-key* PSU (2-D).
    """
    # Single stratum path (no stratum specified), but multi-key psu
    design = Design(wgt="weight", psu=("p1", "p2"))
    sample = Sample(data=df_multikey, design=design)

    tbl = sample.categorical.tabulate(rowvar="row", drop_nulls=True)
    assert tbl.estimates is not None

    assert tbl.type == TableType.ONE_WAY
    assert len(tbl.estimates) > 0


def test_tabulate_two_way_with_multikey_stratum_singlekey_psu(df_multikey):
    """
    Mixed case: multi-key stratum with single-key PSU. Ensures mask alignment
    is correct when only one of (S, P) is 2-D.
    """
    design = Design(wgt="weight", stratum=("s1", "s2"), psu="p1")
    sample = Sample(data=df_multikey, design=design)

    tbl = sample.categorical.tabulate(rowvar="row", colvar="col", drop_nulls=True)
    assert tbl.estimates is not None

    assert tbl.type == TableType.TWO_WAY
    assert len(tbl.estimates) > 0


def _sum_estimates(tbl: Table) -> float:
    return float(sum(float(c.est) for c in (tbl.estimates or [])))


def test_units_arbitrary_total_scales_proportions(df_basic, mock_design):
    sample = Sample(data=df_basic, design=mock_design)
    tbl = sample.categorical.tabulate(
        rowvar="row", units=TableUnits.PROPORTION, count_total=100, drop_nulls=True
    )
    # proportions (0–1) scaled to 100
    assert math.isclose(_sum_estimates(tbl), 100.0, rel_tol=1e-12, abs_tol=1e-9)


def test_units_count_with_custom_total_overrides_sumw(df_basic, mock_design):
    sample = Sample(data=df_basic, design=mock_design)
    custom = 1_000.0
    tbl = sample.categorical.tabulate(
        rowvar="row", units=TableUnits.COUNT, count_total=custom, drop_nulls=True
    )
    assert math.isclose(_sum_estimates(tbl), custom, rel_tol=1e-12, abs_tol=1e-9)


# ----------------------- NEW: numeric row/col tests ---------------------------


@pytest.mark.parametrize("as_float", [False, True])
def test_one_way_numeric_levels_int_and_float(mock_design, as_float: bool):
    # levels 0,1,1,2,2,2,3  -> counts 1,2,3,1
    vals = [0, 1, 1, 2, 2, 2, 3]
    df = pl.DataFrame(
        {
            "bedrooms": pl.Series(vals, dtype=pl.Float64 if as_float else pl.Int64),
            "weight": [1.0] * len(vals),
            "str": ["S"] * len(vals),
            "psu": list(range(len(vals))),
        }
    )
    sample = Sample(data=df, design=mock_design)

    tbl = sample.categorical.tabulate(
        rowvar="bedrooms",
        units=TableUnits.COUNT,
        drop_nulls=True,
    )
    assert tbl.estimates is not None

    # Sum of counts equals sum of weights
    assert math.isclose(_sum_estimates(tbl), float(df["weight"].sum()), rel_tol=1e-12)

    # Verify level set (string-compare for robustness)
    got_levels = {str(e.rowvar) for e in tbl.estimates}
    assert got_levels == {"0", "1", "2", "3"}


@pytest.mark.parametrize(
    "row_is_float,col_is_float",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_two_way_numeric_levels_int_and_float(mock_design, row_is_float, col_is_float):
    # x: 0,0,1,1,1,2,3
    # y: 1,2,1,1,2,2,1
    x_vals = [0, 0, 1, 1, 1, 2, 3]
    y_vals = [1, 2, 1, 1, 2, 2, 1]

    df = pl.DataFrame(
        {
            "x": pl.Series(x_vals, dtype=pl.Float64 if row_is_float else pl.Int64),
            "y": pl.Series(y_vals, dtype=pl.Float64 if col_is_float else pl.Int64),
            "weight": [1.0] * len(x_vals),
            "str": ["S"] * len(x_vals),
            "psu": list(range(len(x_vals))),
        }
    )
    sample = Sample(data=df, design=mock_design)

    tbl = sample.categorical.tabulate(
        rowvar="x",
        colvar="y",
        units=TableUnits.COUNT,
        drop_nulls=True,
    )
    assert tbl.estimates is not None

    # Sum of all cell counts equals total weight
    assert math.isclose(_sum_estimates(tbl), float(df["weight"].sum()), rel_tol=1e-12)

    # Build {(row, col)->est} and compare to expected raw counts
    got = {}
    for e in tbl.estimates:
        got[(str(e.rowvar), str(e.colvar))] = float(e.est)

    expected = {
        ("0", "1"): 1.0,
        ("0", "2"): 1.0,
        ("1", "1"): 2.0,
        ("1", "2"): 1.0,
        ("2", "2"): 1.0,
        ("3", "1"): 1.0,
    }
    for k, v in expected.items():
        assert math.isclose(got.get(k, 0.0), v, rel_tol=1e-12)

    # And table type should be TWO_WAY
    assert tbl.type == TableType.TWO_WAY


def test_one_way_rows_align_natural_numeric_order():
    ests = [
        CellEst("1", "", 1.0, 0.1, 0.0, 0, 0),
        CellEst("2", "", 2.0, 0.1, 0.0, 0, 0),
        CellEst("10", "", 3.0, 0.1, 0.0, 0, 0),
    ]
    t = Table.one_way(rowvar="hhsize", estimates=ests, rowvals=["1", "2", "10"])

    from svy.categorical.table import _rows_for_display

    rows = list(_rows_for_display(t, default_dec=4))  # [Row, Estimate, Std Err, CV, Lower, Upper]
    # Assert natural numeric order and value alignment (no Rich parsing)
    assert [r[0] for r in rows] == ["1", "2", "10"]
    assert [float(r[1]) for r in rows] == [1.0, 2.0, 3.0]


def test_two_way_print_alignment():
    ests = [
        CellEst("1", "1", 1, 0, 0, 0, 0),
        CellEst("1", "2", 2, 0, 0, 0, 0),
        CellEst("2", "1", 3, 0, 0, 0, 0),
        CellEst("2", "2", 4, 0, 0, 0, 0),
        CellEst("10", "1", 5, 0, 0, 0, 0),
        CellEst("10", "2", 6, 0, 0, 0, 0),
    ]
    t = Table.two_way(
        rowvar="r", colvar="c", estimates=ests, rowvals=["1", "2", "10"], colvals=["1", "2"]
    )
    wide = t.crosstab("est")
    assert wide.drop("r").to_numpy().tolist() == [[1, 2], [3, 4], [5, 6]]
