# Changelog

All notable changes to **svy_rs**, the internal Rust extension powering `svy`'s estimation and replicate-weight kernels, are recorded here. It is not a supported public API — depend on [`svy`](../svy/CHANGELOG.md), not on this directly; entries are technical and describe what changed for `svy`'s use of the extension. Follows [Semantic Versioning](https://semver.org/) and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

<!-- ### Added, ### Changed, ### Fixed, ### Deprecated, ### Removed, ### Security -->

## [0.12.0] — 2026-07-24

### Fixed

- **A by-group's degrees of freedom are counted on that group's own PSUs and strata.** The five grouped Taylor kernels (mean, total, ratio, prop, median) computed one frame-level df and broadcast it to every row — `vec![df_val; n_groups]` — so the per-group `domain_mask` that drives the estimate and the scores never reached the df. A `where=` domain was unaffected, because a filter arrives as zero weights and `degrees_of_freedom` already masks on those; the two paths therefore disagreed for the same subpopulation. On a 4×4×5 design (full df 12), `by="dom"` reported 12 for both groups where R's `degf(subset(...))` gives 7 and 3, and `by="stratum"` inside a domain reported 7 for all four cells against R's 1, 3, 3, 0. The error ran one way — a cell inherited the df of a larger population — so **every grouped confidence interval was too narrow**, by 22% on the smallest domain of a realistic fixture. `degrees_of_freedom` gains a `degrees_of_freedom_in_domain` variant taking an optional membership mask; the existing signature delegates to it with `None`, so there is still one implementation of the rule.
- **Zero-weight rows no longer inflate `n` in the domain SRS variance.** `srs_variance_mean_domain`, `srs_variance_total_domain` and `srs_variance_ratio_domain` counted every in-domain row when building `n`, including rows carrying zero weight. Under `drop_nulls` a missing-`y` row is kept and zero-weighted rather than dropped, so those rows understated the SRS variance and inflated `deff` by exactly the row-count ratio (293/290, 187/185, 291/286 on the synthetic education fixture) while groups with no dropped rows were exact. The ungrouped `srs_variance_mean` already excluded them and its comment asserted the `_domain` variants did the same.

## [0.11.0] — 2026-07-23

### Changed

- **Bounded RNG draws are exactly uniform.** `next_index` used `next_u64() % n`, which selects values below `2^64 mod n` slightly more often. Replaced with Lemire's nearly-divisionless rejection sampling. Affects SRS draws, Fisher–Yates shuffles, bootstrap PSU resampling, and BRR random ordering — **the sampled units for a given seed change with this release.**

### Fixed

- **GLM sandwich scores use the direct estimating function.** PSU score contributions were `w · working_resid` with an epsilon-guarded working-residual denominator that biased the score wherever `dmu/deta` was small (~1e-7 relative error in Taylor SEs). Scores are now `w · (y − mu) · (dmu/deta) / V(mu)`, so GLM SEs agree with a first-principles sandwich and R `svyglm` to ~1e-10.
- **Successive-difference replicate (SDR) weights follow the Fay–Train construction.** Factors were composed multiplicatively per adjacent pair (a spurious ±½ cross-term) with Hadamard rows assigned from the all-ones row (zero between-replicate variance). Units now receive the single additive factor `1 + 2^{−3/2}(h[row(k),r] − h[row(k+1),r])` with rows cycled over `1..R-1`; mean replicate weights preserve the full-sample weights exactly and the SDR variance of a total reproduces the successive-difference identity to 1e-10.
- **GLM design gaps.** Family-specific unit deviance (McCullagh–Nelder) replaces the weighted-SSE-for-every-family deviance; per-stratum FPC `(1 − f_h)` factors multiply the sandwich meat; the model-based `(X'WX)⁻¹` covariance is returned alongside the sandwich for Rao–Scott/dAIC design effects; degenerate fits return a typed error instead of panicking through the SVD unwrap, and `fit_glm_by` drops failed levels instead of failing the whole call; `n_obs` counts positive-weight rows.
- **True Sampford rejective PPS sampling.** `pps_rs` drew the remaining `n − 1` units without replacement (making the acceptance check vacuous) and silently fell back to equal-probability sampling after 1000 attempts, so the claimed `π_i = n·p_i` never held. It now implements Sampford's procedure — first draw with `p_i`, the rest with replacement with `λ_i ∝ p_i/(1 − n·p_i)`, rejecting the whole sample on any duplicate; `n·p < 1` violations and degenerate acceptance are errors, not silent design changes.
- **With-replacement and Murphy boundary semantics.** The take-all shortcut no longer applies to with-replacement sampling (`wr` performs exactly `n` draws, repeats allowed, instead of returning each unit once with `π = 1`); `murphy` errors for any `n ≠ 2` instead of silently answering 2. All kernel-side PPS validation errors now translate to typed svy errors.
- **Stratified jackknife variance, df, and centering.** Replicate coefficients were a global `(R − 1)/R` for every jackknife; stratified JKn now uses per-replicate `(n_h − 1)/n_h` and paired JK2 uses `1.0` (the replicate estimation API accepts optional per-replicate `rscales`, and `create_jkn`/`jk2_weights` return them). `variance_from_replicates` uses the direct `svrVar` formula, honoring `variance_center` (R's `mse=TRUE`) instead of a pseudo-value path that ignored it and degenerated for coefficients ≥ 1. `create_jkn_weights` df is `#PSUs − #strata` (was total PSUs), matching R's `degf`.
- **Balanced BRR.** Strata were assigned to Hadamard rows starting at row 0 — the all-ones row — so one PSU per stratum carried zero weight in every replicate and vanished from variance estimation. Strata now map to rows `1..order-1` of a Hadamard matrix of order `> n_strata`; requesting `n_reps` beyond the Hadamard order errors instead of duplicating replicate columns, and `brr_hadamard_size()` is exposed for pre-validation.
- **Raking bounds are enforced on both exit paths.** `max_iter` exhaustion previously returned weights that violated the documented bounds.
- **Sorted control order in `normalize_by_group`,** matching `poststratify`'s (and the documented) convention instead of first-appearance order.
- **`deff` excludes zero-weight rows from the SRS baseline.** The plain `srs_variance_mean/total/ratio` kernels counted out-of-domain zero-weight rows in `n`, inflating the baseline sample size and understating `deff` for domain estimation; they now filter `w > 0` (full-sample results are bit-identical).

### Removed

- Dead `estimation/api.rs` (never declared in `mod.rs`; a stale copy of the regression GLM wrapper) and the non-compiling `parse_benchmarks` bench.

### Build

- Bump `rand` to 0.9.5.
- Batch Rust security updates: `rustls-webpki` → 0.103.13, `quinn-proto` → 0.11.16, `bytes` → 1.12.1, `rand` (0.8.x) → 0.8.7.

## [0.10.0] — 2026-07-12

### Added

- **Batched multi-variable estimation kernels.** `mean`, `total`, `ratio`, `prop`, and `median` accept multiple variables and share a single design build, running them in parallel for ungrouped Taylor estimation. Backs `svy`'s list-input estimation API.

### Fixed

- **Deterministic Taylor variance.** Per-stratum PSU contributions were summed in the iteration order of a standard `HashSet`, making the last digits of a standard error vary run-to-run. PSUs are now summed in a canonical (sorted) order, so results are bit-reproducible.

### Performance

- **Phase C–E estimation and replicate-weight optimizations:** dtype-polymorphic design indexing with cached integer codes; the design is indexed once per by-group/level loop; replicate mean/total/ratio/proportion kernels accumulate without materializing the replicate matrix; cache-blocked parallel replicate-weight matrix build; fused mean/total/ratio domain kernels; by-group Taylor and by-domain GLM parallelized with the GIL released; Taylor kernel fast paths (~2× at 1M rows).

### Build

- Bump `ethnum` 1.5.2 → 1.5.3 for newer Rust toolchains.

## [0.9.0] — 2026-05-19

Baseline for this changelog. For earlier history, see the [Git tags](https://github.com/samplics-org/svy/tags).

[Unreleased]: https://github.com/samplics-org/svy/compare/svy-rs-v0.12.0...HEAD
[0.12.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.12.0
[0.11.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.11.0
[0.10.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.10.0
[0.9.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.9.0
