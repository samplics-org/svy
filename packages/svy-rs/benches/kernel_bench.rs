// benches/kernel_bench.rs
//
// Criterion micro-benchmarks for the Taylor estimation kernel, measured on
// native polars `ChunkedArray`s with NO Python/PyO3 boundary. This isolates the
// exact functions the PERF_PLAN Phase B/C optimizations touch, so a change to
// (say) the design-indexing hash or the point-estimate summation loop can be
// proven in nanoseconds-per-element instead of being lost in end-to-end noise.
//
// Run (host build, extension-module off so it links as a normal binary):
//   PYO3_PYTHON=../../.venv/bin/python cargo bench --no-default-features
// or `make bench-svy-rs` from the repo root.
//
// Design shape mirrors the WB workload used in the Python harness and the
// PERF_PLAN baseline table: ~20 strata, ~50 PSUs per stratum. Inputs are
// single-chunk and null-free (the common post-`prepare_data` case).

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use polars::prelude::*;

use _internal::estimation::taylor::{
    index_categorical, index_categorical_pair, point_estimate_mean, scores_mean, srs_variance_mean,
    taylor_variance,
};

const N_STRATA: usize = 20;
const PSU_PER_STRATUM: usize = 50;
const SIZES: [usize; 2] = [100_000, 1_000_000];

// ── Deterministic input builders (no RNG → stable across runs) ──────────────

fn make_y(n: usize) -> Float64Chunked {
    let v: Vec<f64> = (0..n).map(|i| 100.0 + (i % 1000) as f64 * 0.1).collect();
    Float64Chunked::from_slice("y".into(), &v)
}

fn make_w(n: usize) -> Float64Chunked {
    let v: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 * 0.25).collect();
    Float64Chunked::from_slice("w".into(), &v)
}

fn make_strata(n: usize, long: bool) -> StringChunked {
    let owned: Vec<String> = (0..n)
        .map(|i| {
            let s = i % N_STRATA;
            if long {
                format!("stratum_region_{s}")
            } else {
                s.to_string()
            }
        })
        .collect();
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    StringChunked::from_slice("s".into(), &refs)
}

fn make_psu(n: usize, long: bool) -> StringChunked {
    let owned: Vec<String> = (0..n)
        .map(|i| {
            let p = i % PSU_PER_STRATUM;
            if long {
                format!("enumeration_area_{p}")
            } else {
                p.to_string()
            }
        })
        .collect();
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    StringChunked::from_slice("p".into(), &refs)
}

// ── Design indexing (Phase B FxHashMap / Phase C integer codes target) ──────

fn bench_index_categorical(c: &mut Criterion) {
    let mut g = c.benchmark_group("index_categorical");
    g.sample_size(20);
    for &n in &SIZES {
        for &long in &[false, true] {
            let strata = make_strata(n, long);
            let id = format!("{n}/{}", if long { "long" } else { "short" });
            g.throughput(Throughput::Elements(n as u64));
            g.bench_function(BenchmarkId::from_parameter(id), |b| {
                b.iter(|| black_box(index_categorical(black_box(&strata))));
            });
        }
    }
    g.finish();
}

fn bench_index_categorical_pair(c: &mut Criterion) {
    let mut g = c.benchmark_group("index_categorical_pair");
    g.sample_size(20);
    for &n in &SIZES {
        for &long in &[false, true] {
            let strata = make_strata(n, long);
            let psu = make_psu(n, long);
            let id = format!("{n}/{}", if long { "long" } else { "short" });
            g.throughput(Throughput::Elements(n as u64));
            g.bench_function(BenchmarkId::from_parameter(id), |b| {
                b.iter(|| black_box(index_categorical_pair(black_box(&strata), black_box(&psu))));
            });
        }
    }
    g.finish();
}

// ── Point estimates & scores (Phase B fused no-null loop target) ────────────

fn bench_point_estimates(c: &mut Criterion) {
    let mut g = c.benchmark_group("point_estimate_mean");
    g.sample_size(20);
    for &n in &SIZES {
        let y = make_y(n);
        let w = make_w(n);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::new("point", n), |b| {
            b.iter(|| black_box(point_estimate_mean(black_box(&y), black_box(&w)).unwrap()));
        });
        g.bench_function(BenchmarkId::new("scores", n), |b| {
            b.iter(|| black_box(scores_mean(black_box(&y), black_box(&w)).unwrap()));
        });
        g.bench_function(BenchmarkId::new("srs_variance", n), |b| {
            b.iter(|| black_box(srs_variance_mean(black_box(&y), black_box(&w)).unwrap()));
        });
    }
    g.finish();
}

// ── Full variance pass (indexing + stratum accumulation end-to-kernel) ──────

fn bench_taylor_variance(c: &mut Criterion) {
    let mut g = c.benchmark_group("taylor_variance_mean_strat_cluster");
    g.sample_size(20);
    for &n in &SIZES {
        let y = make_y(n);
        let w = make_w(n);
        let scores = scores_mean(&y, &w).unwrap();
        for &long in &[false, true] {
            let strata = make_strata(n, long);
            let psu = make_psu(n, long);
            let id = format!("{n}/{}", if long { "long" } else { "short" });
            g.throughput(Throughput::Elements(n as u64));
            g.bench_function(BenchmarkId::from_parameter(id), |b| {
                b.iter(|| {
                    black_box(
                        taylor_variance(
                            black_box(&scores),
                            Some(&strata),
                            Some(&psu),
                            None,
                            None,
                            None,
                            None,
                        )
                        .unwrap(),
                    )
                });
            });
        }
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_index_categorical,
    bench_index_categorical_pair,
    bench_point_estimates,
    bench_taylor_variance,
);
criterion_main!(benches);
