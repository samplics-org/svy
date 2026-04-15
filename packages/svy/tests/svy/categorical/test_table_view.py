import polars as pl
import pytest

from svy.categorical.table import CellEst, Table


# -----------------------------------------------------------------------------
# Rich / printing
# -----------------------------------------------------------------------------


def test_table_show_uses_rich_is_captured_by_capsys(capsys):
    # Two-way table ensures "Col" header appears
    tbl = Table.two_way(
        rowvar="row",
        colvar="col",
        estimates=[CellEst("A", "X", 1.0, 0.1, 0.0, 0.9, 1.1)],
    )
    tbl.show(use_rich=True)
    out = capsys.readouterr().out
    assert "Estimate" in out and "Row" in out and "Col" in out


def test_str_one_way_hides_col_header():
    tbl = Table.one_way(
        rowvar="row",
        estimates=[
            CellEst("A", "", 0.4, 0.05, 0.0, 0.3, 0.5),
            CellEst("B", "", 0.6, 0.06, 0.0, 0.5, 0.7),
        ],
    )
    s = str(tbl)
    assert "Estimate" in s
    assert "Col" not in s  # important: no column header for one-way


# -----------------------------------------------------------------------------
# Crosstab: one-way
# -----------------------------------------------------------------------------


def test_crosstab_one_way_single_stat_numeric():
    tbl = Table.one_way(
        rowvar="row",
        estimates=[
            CellEst("A", "", 0.4, 0.05, 0.0, 0.3, 0.5),
            CellEst("B", "", 0.6, 0.06, 0.0, 0.5, 0.7),
        ],
    )
    wide = tbl.crosstab("est")
    assert wide.columns == ["row", "est"]
    assert wide.height == 2
    # values are numeric (not strings)
    assert wide.schema["est"].is_numeric()


def test_crosstab_one_way_multi_stats_formatted():
    tbl = Table.one_way(
        rowvar="row",
        estimates=[
            CellEst("A", "", 0.4, 0.05, 0.0, 0.3, 0.5),
            CellEst("B", "", 0.6, 0.06, 0.0, 0.5, 0.7),
        ],
    )
    wide = tbl.crosstab(("est", "se"))  # default auto -> "est ± se"
    assert wide.columns == ["row", "value"]
    # spot-check formatted cell
    a = wide.filter(pl.col("row") == "A").select("value").to_series().item()
    assert "±" in a


# -----------------------------------------------------------------------------
# Crosstab: two-way
# -----------------------------------------------------------------------------


def _tbl_two_way():
    return Table.two_way(
        rowvar="row",
        colvar="col",
        estimates=[
            CellEst("A", "X", 0.1, 0.01, 0.0, 0.08, 0.12),
            CellEst("A", "Y", 0.2, 0.02, 0.0, 0.17, 0.23),
            CellEst("B", "X", 0.3, 0.03, 0.0, 0.25, 0.35),
            # (B, Y) intentionally missing to test fill_missing/order
        ],
    )


def test_crosstab_two_way_single_stat_and_fill_missing():
    tbl = _tbl_two_way().set_levels(rowvals=["B", "A"], colvals=["X", "Y", "Z"])
    wide = tbl.crosstab("est", fill_missing=0)
    assert wide.columns == ["row", "X", "Y", "Z"]  # preserves set_levels order
    # check filled/mapped values
    ax = wide.filter(pl.col("row") == "A").select("X").to_series().item()
    ay = wide.filter(pl.col("row") == "A").select("Y").to_series().item()
    bz = wide.filter(pl.col("row") == "B").select("Z").to_series().item()
    assert ax == pytest.approx(0.1)
    assert ay == pytest.approx(0.2)
    assert bz == 0  # missing col level got filled


def test_crosstab_two_way_multi_stats_auto_and_ci():
    tbl = _tbl_two_way()
    auto = tbl.crosstab(("est", "se"))  # default -> "est ± se"
    ci = tbl.crosstab(("est", "lci", "uci"), cellfmt="ci")
    cell_auto = auto.filter(pl.col("row") == "A").select("X").to_series().item()
    cell_ci = ci.filter(pl.col("row") == "A").select("X").to_series().item()
    assert "±" in cell_auto
    assert "[" in cell_ci and "]" in cell_ci and "," in cell_ci


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


def test_crosstab_empty_estimates_returns_empty_frame():
    tbl = Table.one_way(rowvar="row", estimates=[])
    out = tbl.crosstab("est")
    assert out.height == 0
    assert out.columns == ["row"]
