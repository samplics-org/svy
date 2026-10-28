# tests/svy/datasets/test_load.py
"""``svy.datasets.load`` resolves a remote slug to a ``Sample`` with its design and labels."""

from __future__ import annotations

import polars as pl
import pytest

import svy

from svy.datasets import load
from svy.errors._dataset import DatasetError


URL = "https://svylab.test/data/toy.parquet"


@pytest.fixture
def wired(routes, make_parquet, make_backend_entry):
    """Register the catalog entry and the parquet; return a function to set the entry's fields."""
    data, sha = make_parquet(n_rows=200)  # 200 rows, 4 regions (50 each)
    routes.add_bytes("/data/toy.parquet", data)

    def _wire(**fields):
        entry = make_backend_entry(slug="toy", download_url=URL, sha256=sha, **fields)
        routes.add_json("/api/data/examples/registry", [entry])

    return _wire


class TestSample:
    def test_returns_sample_holding_the_file(self, wired):
        wired()
        sample = load("toy")
        assert isinstance(sample, svy.Sample)
        assert sample.data.height == 200
        assert sample.data.columns == ["id", "value", "region", "age"]

    def test_design_comes_from_the_entry(self, wired):
        wired(design={"stratum": "region", "psu": "id", "wgt": "value"})
        design = load("toy").design
        assert (design.stratum, design.psu, design.wgt) == ("region", "id", "value")

    def test_no_design_gives_an_empty_design(self, wired):
        wired()
        design = load("toy").design
        assert (design.stratum, design.psu, design.wgt) == (None, None, None)

    def test_labels_come_from_the_entry(self, wired):
        wired(
            variables={
                "age": {"label": "Age in years"},
                "region": {"label": "Region", "values": {"north": "North", "south": "South"}},
            }
        )
        meta = load("toy").meta
        assert dict(meta.var_labels) == {"age": "Age in years", "region": "Region"}
        assert dict(meta.value_labels) == {"region": {"north": "North", "south": "South"}}

    @pytest.mark.parametrize(
        "dtype, code", [(pl.Int64, 18), (pl.Int8, 18), (pl.Float64, 18.0)], ids=str
    )
    def test_json_codes_take_the_column_dtype(
        self, routes, make_backend_entry, tmp_path, dtype, code
    ):
        path = tmp_path / "coded.parquet"
        pl.DataFrame({"age": pl.Series([18, 19, 18], dtype=dtype)}).write_parquet(path)
        routes.add_bytes("/data/coded.parquet", path.read_bytes())
        entry = make_backend_entry(
            slug="coded",
            download_url="https://svylab.test/data/coded.parquet",
            variables={"age": {"label": "Age", "values": {"18": "Eighteen", "19": "Nineteen"}}},
        )
        routes.add_json("/api/data/examples/registry", [entry])
        labels = load("coded").meta.value_labels["age"]
        assert labels == {code: "Eighteen", code + 1: "Nineteen"}
        assert all(type(k) is type(code) for k in labels)

    def test_labels_for_columns_not_in_the_file_are_ignored(self, wired):
        wired(variables={"region": {"label": "Region"}, "gone": {"label": "Not in the file"}})
        assert dict(load("toy").meta.var_labels) == {"region": "Region"}

    def test_cached_download_records_its_entry(self, wired):
        from svy.datasets import _cache

        wired(variables={"age": {"label": "Age in years"}})
        load("toy", source="remote")
        path, ds = _cache.latest_cached("toy")
        assert ds.slug == "toy"
        assert dict(ds.variables) == {"age": {"label": "Age in years"}}


class TestErrors:
    def test_unknown_dataset_raises_not_found(self, routes):
        routes.add_json("/api/data/examples/registry", [])
        with pytest.raises(DatasetError) as exc_info:
            load("unknown")
        assert exc_info.value.code == "DATASET_NOT_FOUND"

    def test_bad_sha_bubbles_up(self, routes, make_parquet, make_dataset_dict):
        data, _ = make_parquet(n_rows=10)
        routes.add_json(
            "/api/data/examples/registry",
            [
                make_dataset_dict(
                    slug="bad",
                    download_url="https://svylab.test/data/bad.parquet",
                    sha256="a" * 64,  # wrong; forces verification to fail
                )
            ],
        )
        routes.add_bytes("/data/bad.parquet", data)
        with pytest.raises(DatasetError) as exc_info:
            load("bad")
        assert exc_info.value.code == "DATASET_SHA_MISMATCH"

    @pytest.mark.parametrize("kw", ["n", "where", "select", "lazy"])
    def test_removed_options_are_not_accepted(self, kw):
        with pytest.raises(TypeError, match=kw):
            load("toy", **{kw: None})
