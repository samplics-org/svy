"""
Performance benchmarks for svy-io parsing functions.

Run with: uv run pytest tests/benchmark_test.py --benchmark-only
Compare: uv run pytest tests/benchmark_test.py --benchmark-compare
"""

from pathlib import Path

import svy_io


def get_test_file(filename):
    """Get path to test data file."""
    return str(Path(__file__).parent / "data" / filename)


# ============================================================================
# STATA BENCHMARKS
# ============================================================================


def test_bench_stata_types_13(benchmark):
    """Benchmark: Parse Stata 13 file with various types."""
    file_path = get_test_file("stata/WLD_2023_SYNTH-SVY-HLD-EN_v01_M.dta")
    # file_path = get_test_file("stata/WLD_2023_SYNTH-CEN-IND-EN_v01_M.dta")
    result = benchmark(svy_io.read_stata, file_path)
    df, meta = result
    assert df is not None
    assert meta is not None


def test_bench_stata_types_14(benchmark):
    """Benchmark: Parse Stata 14 file."""
    file_path = get_test_file("stata/WLD_2023_SYNTH-SVY-HLD-EN_v01_M.dta")
    result = benchmark(svy_io.read_stata, file_path)
    df, meta = result
    assert df is not None


def test_bench_stata_types_15(benchmark):
    """Benchmark: Parse Stata 15 file."""
    file_path = get_test_file("stata/WLD_2023_SYNTH-SVY-HLD-EN_v01_M.dta")
    result = benchmark(svy_io.read_stata, file_path)
    df, meta = result
    assert df is not None


def test_bench_stata_with_column_skip(benchmark):
    """Benchmark: Parse Stata file with column skipping."""
    file_path = get_test_file("stata/WLD_2023_SYNTH-SVY-HLD-EN_v01_M.dta")
    cols_skip = ["vfloat", "vdouble"]
    result = benchmark(svy_io.read_stata, file_path, cols_skip=cols_skip)
    df, meta = result
    assert "vfloat" not in df.columns


def test_bench_stata_with_row_limit(benchmark):
    """Benchmark: Parse Stata file with row limit."""
    file_path = get_test_file("stata/WLD_2023_SYNTH-SVY-HLD-EN_v01_M.dta")
    result = benchmark(svy_io.read_stata, file_path, n_max=100)
    df, meta = result
    assert len(df) <= 100


# ============================================================================
# SPSS BENCHMARKS
# ============================================================================


# def test_bench_spss_labelled_num(benchmark):
#     """Benchmark: Parse SPSS file with numeric labels."""
#     file_path = get_test_file("labelled-num.sav")
#     result = benchmark(svy_io.read_spss, file_path)
#     df, meta = result
#     assert df is not None


# def test_bench_spss_labelled_chr(benchmark):
#     """Benchmark: Parse SPSS file with character labels."""
#     file_path = get_test_file("labelled-chr.sav")
#     result = benchmark(svy_io.read_spss, file_path)
#     df, meta = result
#     assert df is not None


# ============================================================================
# SAS BENCHMARKS
# ============================================================================


# def test_bench_sas_hadley(benchmark):
#     """Benchmark: Parse SAS7BDAT file."""
#     file_path = get_test_file("hadley.sas7bdat")
#     result = benchmark(svy_io.read_sas, file_path)
#     df, meta = result
#     assert df is not None


# def test_bench_sas_with_catalog(benchmark):
#     """Benchmark: Parse SAS file with catalog."""
#     file_path = get_test_file("hadley.sas7bdat")
#     catalog_path = get_test_file("formats.sas7bcat")
#     result = benchmark(svy_io.read_sas, file_path, catalog=catalog_path)
#     df, meta = result
#     assert df is not None


# ============================================================================
# XPT BENCHMARKS
# ============================================================================


# def test_bench_xpt_read(benchmark):
#     """Benchmark: Parse SAS Transport (XPT) file."""
#     file_path = get_test_file("sample.xpt")
#     result = benchmark(svy_io.read_xpt, file_path)
#     df, meta = result
#     assert df is not None


# ============================================================================
# COMPARATIVE BENCHMARKS
# ============================================================================


# @pytest.mark.parametrize(
#     "format_type,filename,reader",
#     [
#         ("stata", "types-13.dta", svy_io.read_stata),
#         ("spss", "labelled-num.sav", svy_io.read_spss),
#         ("sas", "hadley.sas7bdat", svy_io.read_sas),
#     ],
# )
# def test_bench_format_comparison(benchmark, format_type, filename, reader):
#     """Benchmark: Compare parsing speeds across formats."""
#     file_path = get_test_file(filename)
#     result = benchmark(reader, file_path)
#     df, meta = result
#     assert df is not None
#     # Tag for grouping
#     benchmark.extra_info["format"] = format_type


# ============================================================================
# THROUGHPUT BENCHMARKS
# ============================================================================


# def test_bench_stata_throughput_small(benchmark):
#     """Benchmark: Small file throughput."""
#     file_path = get_test_file("types-13.dta")

#     def parse_multiple():
#         for _ in range(10):
#             svy_io.read_stata(file_path)

#     benchmark(parse_multiple)


# def test_bench_metadata_extraction(benchmark):
#     """Benchmark: Metadata extraction performance."""
#     file_path = get_test_file("types-13.dta")

#     def extract_meta():
#         _, meta = svy_io.read_stata(file_path)
#         return meta

#     result = benchmark(extract_meta)
#     assert "vars" in result


# ============================================================================
# WRITE BENCHMARKS
# ============================================================================


# def test_bench_stata_write(benchmark, tmp_path):
#     """Benchmark: Write Stata file."""
#     import polars as pl

#     # Create sample data
#     df = pl.DataFrame(
#         {
#             "x": range(1000),
#             "y": [f"value_{i}" for i in range(1000)],
#         }
#     )

#     output_path = tmp_path / "output.dta"

#     benchmark(svy_io.write_stata, df, str(output_path))
#     assert output_path.exists()


# def test_bench_spss_write(benchmark, tmp_path):
#     """Benchmark: Write SPSS file."""
#     import polars as pl

#     df = pl.DataFrame(
#         {
#             "x": range(1000),
#             "y": [f"value_{i}" for i in range(1000)],
#         }
#     )

#     output_path = tmp_path / "output.sav"

#     benchmark(svy_io.write_spss, df, str(output_path))
#     assert output_path.exists()
