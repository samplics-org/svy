// benches/parse_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pyo3::prelude::*;

// Import the library functions
use svyreadstat_rs::{df_parse_dta_file, df_parse_sav_file};

fn bench_parse_dta(c: &mut Criterion) {
    // Initialize Python once for all benchmarks
    pyo3::prepare_freethreaded_python();

    let test_file = "path/to/test.dta"; // Update with actual test file path

    c.bench_function("parse_dta_small", |b| {
        b.iter(|| {
            Python::with_gil(|py| {
                let result = df_parse_dta_file(
                    py,
                    black_box(test_file),
                    None, // cols_skip
                    None, // n_max
                    0,    // rows_skip
                );
                result.unwrap()
            })
        })
    });
}

fn bench_parse_dta_with_skip(c: &mut Criterion) {
    pyo3::prepare_freethreaded_python();

    let test_file = "path/to/test.dta";
    let cols_to_skip = vec!["col1".to_string(), "col2".to_string()];

    c.bench_function("parse_dta_with_skip", |b| {
        b.iter(|| {
            Python::with_gil(|py| {
                let result = df_parse_dta_file(
                    py,
                    black_box(test_file),
                    Some(cols_to_skip.clone()),
                    None,
                    0,
                );
                result.unwrap()
            })
        })
    });
}

fn bench_parse_sav(c: &mut Criterion) {
    pyo3::prepare_freethreaded_python();

    let test_file = "path/to/test.sav";

    c.bench_function("parse_sav", |b| {
        b.iter(|| {
            Python::with_gil(|py| {
                let result = df_parse_sav_file(
                    py,
                    black_box(test_file),
                    None,  // encoding
                    false, // user_na
                    None,  // cols_skip
                    None,  // n_max
                    0,     // rows_skip
                );
                result.unwrap()
            })
        })
    });
}

criterion_group!(
    benches,
    bench_parse_dta,
    bench_parse_dta_with_skip,
    bench_parse_sav
);
criterion_main!(benches);
