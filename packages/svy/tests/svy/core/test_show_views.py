# tests/svy/core/test_show_views.py
import polars as pl
import pytest

from svy.core.design import Design
from svy.core.expr import col, lit, when
from svy.core.sample import Sample
from svy.errors.dimension_errors import DimensionError
from svy.errors.method_errors import MethodError


# ---------- fixtures ----------


@pytest.fixture
def sample_small() -> Sample:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "age": [16, 22, 35, 70, 40, 28],
            "height": [160, 180, 170, 165, 175, 172],
            "weight": [55.0, 80.0, 72.0, 60.0, 85.0, 70.0],
            "geo1": ["NORTH", "SOUTH", "NORTH", "EAST", "SOUTH", "NORTH"],
            "year": [2024, 2024, 2023, 2024, 2024, 2023],
            "w": [1.2, 0.9, 1.1, 1.3, 1.0, 0.95],
        }
    )
    design = Design(wgt="w")
    return Sample(data=df, design=design)


# ============================================================
# show_data() — basic slicing
# ============================================================


class TestShowDataBasic:
    def test_default_returns_head(self, sample_small):
        out = sample_small.show_data(n=3)
        assert isinstance(out, pl.DataFrame)
        assert out.height == 3
        assert out.columns[:3] == ["svy_row_index", "id", "age"]

    def test_n_none_returns_all_rows(self, sample_small):
        out = sample_small.show_data(n=None)
        assert out.height == sample_small.data.height

    def test_select_columns(self, sample_small):
        out = sample_small.show_data(columns=["id", "age"], n=3)
        assert out.columns == ["id", "age"]
        assert out.height == 3

    def test_random_deterministic(self, sample_small):
        out1 = sample_small.show_data(order_type="random", n=2, rstate=123)
        out2 = sample_small.show_data(order_type="random", n=2, rstate=123)
        assert out1.equals(out2)


# ============================================================
# show_data() — ordering
# ============================================================


class TestShowDataOrderBy:
    def test_order_by_ascending(self, sample_small):
        out = sample_small.show_data(
            columns=["id", "age"],
            order_by="age",
            order_type="ascending",
            n=6,
        )
        ages = out["age"].to_list()
        assert ages == sorted(ages)

    def test_order_by_descending(self, sample_small):
        out = sample_small.show_data(
            columns=["id", "age"],
            order_by="age",
            order_type="descending",
            n=6,
        )
        ages = out["age"].to_list()
        assert ages == sorted(ages, reverse=True)

    def test_order_by_default_is_ascending(self, sample_small):
        out = sample_small.show_data(columns=["id", "age"], order_by="age", n=6)
        ages = out["age"].to_list()
        assert ages == sorted(ages)

    def test_order_type_random_shuffles(self, sample_small):
        natural = sample_small.show_data(columns=["id"], n=6)["id"].to_list()
        results = [
            sample_small.show_data(columns=["id"], order_type="random", n=6, rstate=seed)[
                "id"
            ].to_list()
            for seed in range(20)
        ]
        assert any(r != natural for r in results)

    def test_order_type_random_deterministic(self, sample_small):
        out1 = sample_small.show_data(columns=["id"], order_type="random", n=6, rstate=42)
        out2 = sample_small.show_data(columns=["id"], order_type="random", n=6, rstate=42)
        assert out1.equals(out2)

    def test_order_type_random_returns_all_rows(self, sample_small):
        out = sample_small.show_data(order_type="random", n=None, rstate=7)
        assert out.height == sample_small.data.height
        assert set(out["id"].to_list()) == set(sample_small.data["id"].to_list())

    def test_order_type_random_then_head(self, sample_small):
        out = sample_small.show_data(order_type="random", n=3, rstate=1)
        assert out.height == 3

    def test_order_by_multi_column(self, sample_small):
        out = sample_small.show_data(
            columns=["id", "geo1", "age"],
            order_by=["geo1", "age"],
            order_type="ascending",
            n=6,
        )
        assert out.height == 6
        geos = out["geo1"].to_list()
        assert geos == sorted(geos)

    def test_order_by_none_order_type_ascending_is_natural_order(self, sample_small):
        out = sample_small.show_data(columns=["id"], order_by=None, order_type="ascending", n=6)
        assert out["id"].to_list() == [1, 2, 3, 4, 5, 6]

    def test_nulls_last(self, sample_small):
        out = sample_small.show_data(
            columns=["id", "age"],
            order_by="age",
            order_type="ascending",
            nulls_last=True,
            n=6,
        )
        assert out.height == 6

    def test_order_by_with_random_shuffles_within_groups(self, sample_small):
        """
        order_by + order_type='random': sort by geo1, shuffle within each
        geo1 group. The geo1 column must remain sorted; id order within
        each group must vary across runs.
        """
        results = [
            sample_small.show_data(
                columns=["id", "geo1"],
                order_by="geo1",
                order_type="random",
                n=6,
                rstate=seed,
            )
            for seed in range(30)
        ]
        for out in results:
            geos = out["geo1"].to_list()
            assert geos == sorted(geos), f"geo1 not sorted: {geos}"

        north_id_orders = [
            out.filter(pl.col("geo1") == "NORTH")["id"].to_list() for out in results
        ]
        assert len(set(map(tuple, north_id_orders))) > 1, (
            "Expected within-group id order to vary across seeds"
        )

    def test_order_by_with_random_keeps_all_rows(self, sample_small):
        out = sample_small.show_data(order_by="geo1", order_type="random", n=None, rstate=5)
        assert out.height == sample_small.data.height
        assert set(out["id"].to_list()) == set(sample_small.data["id"].to_list())


# ============================================================
# show_data() — where (filtering)
# ============================================================


class TestShowDataWhere:
    def test_where_dict_eq_and_in(self, sample_small):
        out = sample_small.show_data(
            where={"geo1": ("NORTH", "SOUTH"), "year": 2024},
            columns=["id", "geo1", "year"],
            order_by="id",
        )
        assert set(out.columns) == {"id", "geo1", "year"}
        assert all(g in ("NORTH", "SOUTH") for g in out["geo1"].to_list())
        assert all(y == 2024 for y in out["year"].to_list())

    def test_where_exprs_and(self, sample_small):
        conds = [col("age") >= 30, col("geo1") == lit("NORTH")]
        out = sample_small.show_data(
            where=conds,
            columns=["id", "age", "geo1"],
            order_by="age",
        )
        assert out.height >= 1
        assert all(a >= 30 for a in out["age"].to_list())
        assert all(g == "NORTH" for g in out["geo1"].to_list())

    def test_where_single_expr(self, sample_small):
        out = sample_small.show_data(
            where=when(col("age") >= 18).then(True).otherwise(False),
            columns=["id", "age"],
            order_by="id",
        )
        assert out.height == (sample_small.data["age"] >= 18).sum()

    def test_where_single_svy_expr(self, sample_small):
        out = sample_small.show_data(where=col("age") >= 18, columns=["id", "age"])
        assert out.height == (sample_small.data["age"] >= 18).sum()

    def test_where_none_no_filter(self, sample_small):
        out = sample_small.show_data(where=None, columns=None, n=None)
        assert out.shape == sample_small.data.shape

    def test_where_empty_dict_no_filter(self, sample_small):
        out = sample_small.show_data(where={}, columns=["id"], order_by="id")
        assert out["id"].to_list() == [1, 2, 3, 4, 5]

    def test_where_mix_svy_pl_expr(self, sample_small):
        conds = [col("age") > 20, pl.col("geo1") == "NORTH"]
        out = sample_small.show_data(where=conds, columns=["id", "age", "geo1"])
        assert all(a > 20 for a in out["age"].to_list())
        assert all(g == "NORTH" for g in out["geo1"].to_list())

    def test_where_with_order_by_descending(self, sample_small):
        out = sample_small.show_data(
            where={"year": 2024},
            columns=["id", "year", "age"],
            order_by="age",
            order_type="descending",
            n=None,
        )
        ages = out["age"].to_list()
        assert ages == sorted(ages, reverse=True)

    def test_where_with_order_type_random(self, sample_small):
        results = [
            sample_small.show_data(
                where=None,
                columns=["id"],
                order_type="random",
                n=4,
            )
            for _ in range(20)
        ]
        id_orders = [out["id"].to_list() for out in results]
        assert any(a != b for a, b in zip(id_orders, id_orders[1:]))

    def test_where_with_order_type_random_correct_count(self, sample_small):
        out = sample_small.show_data(where=None, columns=["id"], order_type="random", n=4)
        assert out.height == 4


# ============================================================
# show_data() — offset (paging)
# ============================================================


class TestShowDataOffset:
    def test_paging(self, sample_small):
        out1 = sample_small.show_data(columns=["id"], order_by="id", n=2, offset=0)
        out2 = sample_small.show_data(columns=["id"], order_by="id", n=2, offset=2)
        assert out1["id"].to_list() == [1, 2]
        assert out2["id"].to_list() == [3, 4]

    def test_n_none_with_offset(self, sample_small):
        out = sample_small.show_data(columns=["id"], order_by="id", n=None, offset=3)
        assert out["id"].to_list() == [4, 5, 6]

    def test_offset_with_where(self, sample_small):
        out = sample_small.show_data(
            where={"geo1": "NORTH"},
            columns=["id"],
            order_by="id",
            n=1,
            offset=1,
        )
        # NORTH ids are [1, 3, 6]; offset=1 skips first, n=1 takes one
        assert out.height == 1
        assert out["id"].to_list() == [3]


# ============================================================
# show_data() — columns="*"
# ============================================================


class TestShowDataColumnsStar:
    def test_columns_star(self, sample_small):
        out = sample_small.show_data(columns="*", n=None)
        assert out.columns == sample_small.data.columns


# ============================================================
# show_data() — errors
# ============================================================


class TestShowDataErrors:
    def test_invalid_n(self, sample_small):
        with pytest.raises(DimensionError) as ei:
            sample_small.show_data(n=-5)
        assert ei.value.code == "INVALID_N"

    def test_invalid_offset(self, sample_small):
        with pytest.raises(DimensionError) as ei:
            sample_small.show_data(offset=-5)
        assert ei.value.code == "INVALID_OFFSET"

    def test_missing_column(self, sample_small):
        with pytest.raises(DimensionError) as ei:
            sample_small.show_data(columns=["id", "does_not_exist"], n=2)
        assert ei.value.code == "MISSING_COLUMNS"

    def test_sort_missing_column(self, sample_small):
        with pytest.raises(DimensionError) as ei:
            sample_small.show_data(columns=["id"], order_by="age_not_here", n=2)
        assert ei.value.code == "MISSING_SORT_COLUMNS"

    def test_invalid_order_type(self, sample_small):
        with pytest.raises(MethodError) as ei:
            sample_small.show_data(order_type="sideways", n=2)
        assert ei.value.code == "INVALID_ORDER_TYPE"

    def test_missing_filter_column(self, sample_small):
        with pytest.raises(DimensionError) as ei:
            sample_small.show_data(where={"not_a_col": 1})
        assert ei.value.code == "MISSING_FILTER_COLUMN"

    def test_unsupported_where_type(self, sample_small):
        with pytest.raises(MethodError) as ei:
            sample_small.show_data(where=object())
        assert ei.value.code == "UNSUPPORTED_WHERE"

    def test_where_empty_in_values(self, sample_small):
        with pytest.raises(MethodError) as ei:
            sample_small.show_data(where={"geo1": []})
        assert ei.value.code == "EMPTY_IN_VALUES"
