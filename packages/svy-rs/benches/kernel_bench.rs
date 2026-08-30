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

use _internal::estimation::association::{
    AssocKind, PairProducts, bivar_moments, multi_moments, point_estimate_corr,
    replicate_association,
};
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
            // taylor_variance takes design columns; build them outside the
            // timed closure so the conversion is not measured.
            let strata = Column::from(make_strata(n, long).into_series());
            let psu = Column::from(make_psu(n, long).into_series());
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

// ── Association: all-pairs sweep vs repeated pairwise kernels ───────────────
//
// Pairwise costs k(k+1)/2 two-pass kernels, each re-reading its two columns;
// the matrix form reads each column twice regardless of k. This measures how
// much of that O(k^2) → O(k) traffic reduction survives to the clock.

fn make_col(n: usize, seed: usize) -> Float64Chunked {
    let v: Vec<f64> = (0..n)
        .map(|i| 100.0 + ((i * (7 + seed) + seed * 13) % 997) as f64 * 0.37)
        .collect();
    Float64Chunked::from_slice("c".into(), &v)
}

fn bench_all_pairs(c: &mut Criterion) {
    let mut g = c.benchmark_group("association_all_pairs");
    g.sample_size(20);
    for &n in &SIZES {
        for &k in &[5usize, 10] {
            let owned: Vec<Float64Chunked> = (0..k).map(|s| make_col(n, s)).collect();
            let cols: Vec<&Float64Chunked> = owned.iter().collect();
            let w = make_w(n);
            g.throughput(Throughput::Elements((n * k) as u64));

            g.bench_with_input(BenchmarkId::new("matrix", format!("{n}/k{k}")), &k, |b, _| {
                b.iter(|| black_box(multi_moments(black_box(&cols), black_box(&w), None).unwrap()))
            });

            g.bench_with_input(BenchmarkId::new("pairwise", format!("{n}/k{k}")), &k, |b, _| {
                b.iter(|| {
                    let mut acc = 0.0f64;
                    for j in 0..k {
                        for l in j..k {
                            let m = bivar_moments(cols[j], cols[l], &w, None).unwrap();
                            acc += m.m_yx;
                        }
                    }
                    black_box(acc)
                })
            });
        }
    }
    g.finish();
}

// ── Association: replicate recomputation, precomputed vs from scratch ──────
//
// y*x, y^2 and x^2 do not depend on the weights. Forming them once turns each
// replicate into six weighted dot products; recomputing them per replicate
// repeats R*n multiplications.

fn bench_replicate_association(c: &mut Criterion) {
    let mut g = c.benchmark_group("association_replicates");
    g.sample_size(10);
    const N_REPS: usize = 200;

    for &n in &[100_000usize] {
        let y = make_col(n, 1);
        let x = make_col(n, 2);
        let w = make_w(n);

        // JK1-shaped replicate weights over PSU_PER_STRATUM clusters.
        let reps: Vec<Vec<f64>> = (0..N_REPS)
            .map(|r| {
                (0..n)
                    .map(|i| {
                        let psu = i % PSU_PER_STRATUM;
                        if psu == r % PSU_PER_STRATUM { 0.0 } else { 0.5 + (i % 7) as f64 * 0.25 }
                    })
                    .collect()
            })
            .collect();
        let cols: Vec<&[f64]> = reps.iter().map(|v| v.as_slice()).collect();
        g.throughput(Throughput::Elements((n * N_REPS) as u64));

        g.bench_function(BenchmarkId::new("precomputed", n), |b| {
            let p = PairProducts::new(&y, &x, &w, None).unwrap();
            b.iter(|| {
                black_box(replicate_association(
                    black_box(&p),
                    black_box(&cols),
                    AssocKind::Corr,
                ))
            })
        });

        g.bench_function(BenchmarkId::new("recompute", n), |b| {
            b.iter(|| {
                let out: Vec<f64> = cols
                    .iter()
                    .map(|col| {
                        let rw = Float64Chunked::from_slice("rw".into(), col);
                        point_estimate_corr(&y, &x, &rw).unwrap()
                    })
                    .collect();
                black_box(out)
            })
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_index_categorical,
    bench_index_categorical_pair,
    bench_point_estimates,
    bench_taylor_variance,
    bench_all_pairs,
    bench_replicate_association,
);
criterion_main!(benches);
