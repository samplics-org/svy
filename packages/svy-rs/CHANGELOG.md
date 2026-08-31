# Changelog

All notable changes to **svy_rs**, the internal Rust extension powering `svy`'s estimation and replicate-weight kernels, are recorded here. It is not a supported public API — depend on [`svy`](../svy/CHANGELOG.md), not on this directly; entries are technical and describe what changed for `svy`'s use of the extension. Follows [Semantic Versioning](https://semver.org/) and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

<!-- ### Added, ### Changed, ### Fixed, ### Deprecated, ### Removed, ### Security -->

### Added

- `offset_name` on `fit_glm_rs`: a known term on the link scale, entered as `z <- (eta - offset) + (y - mu)/mu.eta` in the IRLS working response and added back wherever eta is rebuilt (R's `glm.fit`). Absent, it materialises as zeros, so the existing path is bit-identical. Carries a dedicated one-parameter IRLS for the null deviance, since the intercept-only MLE is no longer the weighted mean of y once mu varies by row.

- `Link::Probit` and `Link::Cloglog` in the GLM kernel, with the clamps R's `make.link` applies (eta bounded at `+/-qnorm(eps)`, mu at `[eps, 1-eps]`). Backed by a normal CDF written as `erfc(-x/sqrt2)/2` — confluent series below 1, modified-Lentz continued fraction above — which agrees with R's `pnorm` to ~1e-14 relative and keeps full precision in the tails, where a coarser erf shows up directly in the fitted coefficients.

## [0.15.0] — 2026-08-26

### Added

- **`create_poisson_bootstrap_wgts`** ([#131](https://github.com/samplics-org/svy/pull/131)). The Beaumont–Patak generalized bootstrap kernel, backing `svy`'s `create_bs_wgts(kind="poisson")`. Takes a weight vector, `n_reps` and an optional `seed`, and returns the replicate matrix together with its per-replicate coefficient. It needs no stratum or PSU: the draws are independent per unit, which is the point — it exists for public-use files where those identifiers are suppressed.

  The kernel builds one flat column-major buffer and calibrates through a `k x B` scratch table, so peak memory is the single output allocation rather than the three matrices the obvious NumPy or polars route holds. 113,603 x 1,000 runs in 0.066 s at 1.06 GiB peak, against 1.70 s and 2.88 GiB for the naive path.

  Weights below 1 raise rather than producing `NaN` from `sqrt((w-1)/w)`.

- **Unit tests for the coefficient rules** in `estimation/replication.rs`: that the variance is independent of the method label, that non-uniform coefficients are honoured, that bootstrap gives `1/B` and jackknife `(B-1)/B`, that BRR collapses to `1/B` at `fay_coef = 0`, and that the Fay coefficient affects nothing but BRR.

### Changed

- **BREAKING: the replication estimation entry points take `rep_coefs` instead of `method`, `fay_coef` and `rscales`** ([#131](https://github.com/samplics-org/svy/pull/131)). Every one of `replicate_mean`, `replicate_total`, `replicate_assoc`, `replicate_ratio`, `replicate_prop`, `replicate_quantile` and `replicate_median` changed signature:

  ```
  before  (..., rep_weight_cols, method, fay_coef=0.0, rscales=None, center="rep_mean", ...)
  after   (..., rep_weight_cols, rep_coefs, center="rep_mean", ...)
  ```

  Three parameters collapse into one required vector. The kernel no longer derives coefficients from a method label — it consumes the coefficients it is given, so a scheme whose scale differs needs no change here. Deriving them is now the caller's job, which is where the design information lives. `RepMethod` and `replicate_coefficients` remain in the crate for the weighting side.

  This is why `svy` must pin `svy-rs>=0.15.0`: a 0.14.0 extension does not have the new signatures, and the mismatch surfaces as a bare failure rather than an import error.

### Removed

- **`parse_rep_method`**, the method-label parser behind the old signatures, and the `domain` parameter on the Rust entry point together with the now-unused `_domain_codes` helper.

## [0.14.0] — 2026-08-10

### Added

- **Association kernels: design-based covariance and Pearson correlation** ([#124](https://github.com/samplics-org/svy/pull/124)). Both are smooth functions of the same six weighted totals, so they share one moment routine and differ only in the final combination and the linearization handed to the variance machinery. Exposed as `taylor_assoc` and `replicate_assoc`, which take column pairs rather than a single response.

  The linearization is the load-bearing part. For the covariance the terms from estimating the means cancel identically, leaving `a*b - m_yx` — which is why R's `svymean(a*b*n/(n-1))` shortcut is exactly the delta method rather than an approximation. For the correlation they do not cancel, because the standard deviations are themselves estimated, so the score carries `-(rho/2)(yt² + xt²)`. Feeding a covariance-style score to a correlation would silently understate its variance.

  Moments accumulate in two passes rather than from raw sums of squares: the one-pass form cancels catastrophically when the mean is large relative to the spread, and the second pass is free on the Taylor path since scores cannot be formed until rho and the means are known.

  Two optimizations, both benchmarked. Replication precomputes the weight-independent products once, turning each replicate into six dot products — 12.8x faster at n=100k with 200 replicates. An all-pairs sweep builds the whole moment matrix in one pass, cutting column traffic from O(k²) to O(k): level at k=5, ~1.5–1.8x at k=10.

- **The simple-random-sample reference for a design effect is now explicit.** `SrsRef` selects what the design variance is compared against — `WithReplacement` (`S²/n`) or `WithoutReplacement { pop_total }` (`S²/n · (1 - n/N)`) — and is threaded through every `srs_variance_*` kernel. The nine estimator entry points that report a design effect gain `deff_ref` and `deff_pop_total`; both default to the previous behaviour, so no reported number changes. `taylor_quantile`/`taylor_median` are untouched, having no design effect to report.

  The finite-population correction is the only scale-dependent term in the whole calculation — the design variance and the population variance are both invariant to the weight scale. So once weights stop being reciprocals of selection probabilities, only that factor moves: halving them shifted a design effect by 0.94%, exactly the FPC ratio. `WithReplacement` omits it and is therefore scale-free.

  A degenerate correction is reported as NaN rather than as an error. The kernels compute the reference on every call, including calls that never asked for a design effect, so failing there would break estimation for anyone whose weights merely sum to about `n` — unit weights being the obvious case. `svy` diagnoses it where the request is known.

  The guard uses a tolerance rather than a sign test: weights normalized to sum to `n` land within ~1e-12 of it on either side, and a hair above zero yields an FPC near 1e-16 and a design effect of ~1e16. A genuine near-census design bottoms out around 1e-6, well clear of the threshold.

  Validated against R survey 4.5 across five design shapes — stratified SRS, one-stage cluster, stratified plus cluster, each with and without a design fpc, plus domain variants — crossed with both references: 16 golden values, all reproduced. The design's own fpc and the reference's are independent, and a test asserts that directly rather than by coincidence of values.

## [0.13.0] — 2026-08-05

### Fixed

- **The Woodruff score is centered on its own weighted mean.** The linearization used `(w_i / sum_w) * (I(y_i > q) - (1 - p))`, centering the residual on the *nominal* target rather than on the mean it actually realizes. `u_bar` is zero only when `F(q)` lands exactly on `p`, which discreteness prevents — on a 5000-record fixture it sits near -4e-5, and omitting it moved the probability-scale SE by ~2e-4 relative. R reaches the centered form by computing the variance as `svymean(U, design)`. The `Higher`/`Lower` rules were unaffected in their output, because the CDF inversion snaps to an order statistic and absorbs the wobble, so medians and default-`q_method` quantiles do not move; `Linear`, `Middle` and `Nearest` interpolate continuously and did drift — up to 4e-4 relative on the confidence limits. All rules now agree with R to floating-point noise.

### Changed

- **polars 0.52 → 0.55.1, and the three crates version-locked to it** ([#109](https://github.com/samplics-org/svy/issues/109)). `pyo3-polars` pins polars, and `pyo3` pins `pyo3-polars`, and `numpy` pins `pyo3`, so the polars bump is really a four-crate move: `pyo3-polars` 0.25 → 0.28, `pyo3` 0.26 → 0.29.1, `numpy` 0.26 → 0.29. `criterion` 0.5 → 0.8.2 rides along as a dev-dependency.

  Three API changes carried the migration. `IntoIterator for &ChunkedArray<T>` is gone, so 50 call sites move from `.into_iter()` to `.iter()` — both yield `Option<T>`, so the iteration is unchanged. `DataFrame::as_single_chunk_par` is renamed `rechunk_mut_par` (same parallel rechunk, same doc contract). `DataFrame::with_column` takes a `Column` rather than a `Series`.

  **Results are numerically inert.** Estimates, standard errors and confidence limits are bit-for-bit identical to the polars 0.52 build, and the quantile path still agrees bit-for-bit with R's `oldsvyquantile` across p = 0.01 to 0.99. 80 Rust and 2844 Python tests pass, including the suite's R and Stata reference comparisons.

- **`ndarray` stays at 0.16 and is no longer safe to bump on its own.** `numpy` requires `>= 0.15, <=0.17`, which cargo reads as ≤ 0.17.0, so bumping to 0.17 resolves 0.17.x for this crate while numpy keeps its own 0.16 — two `ArrayBase` types in one graph, and every ndarray-typed call stops typechecking. It moves only when numpy's range moves.

### Security

- **`pyo3` 0.26 → 0.29.1** closes the bump deferred on 2026-07-22, when it was blocked on `pyo3-polars` not yet supporting a current pyo3. `svy-io` moved to 0.29 at the time ([#66](https://github.com/samplics-org/svy/pull/66)); both native crates are now on the same major.

### Added

- **`rust-version = "1.91"`.** polars calls `i64::strict_abs` (`strict_overflow_ops`, stable since 1.91) and declares no `rust-version` itself, so on an older toolchain the build failed with a raw `E0658` inside `polars-core` instead of an MSRV error. CI is unaffected — it floats on stable.

- **The Woodruff kernels take a probability instead of assuming 0.5.** `scores_median` hardcoded `let _p = 0.5` and its indicator `I(y > q) - 0.5`; the generalized `scores_quantile` uses `I(y > q) - (1 - p)`, which reduces to the old expression at `p = 0.5`. `weighted_quantile_domain`, `scores_quantile_domain` and the replication kernels (`weighted_quantiles_vec`, `matrix_quantile_estimates`, `matrix_quantile_by_domain`) are generalized the same way. The median entry points remain, delegating with `probs = [0.5]` and dropping the probability column, so their result schema is unchanged.

- **`quantiles_woodruff` evaluates many probabilities in one pass.** Quantiles are sort-bound, so the sort behind the weighted CDF and the design indexing are done once and only the score vector and its variance are per-probability. New `taylor_quantile`, `taylor_quantile_multi` and `replicate_quantile` entry points return one row per probability in a `prob` column.

- **`weighted_quantile_at`** exposes the interpolation rule to Python over a pre-sorted variable and its CDF, so `svy` can invert the CDF for confidence limits using the same rule as the point estimate rather than reimplementing it. One implementation, no drift.

## [0.12.1] — 2026-08-04

### Changed

- **A Taylor estimate indexes its design once instead of twice.** `degrees_of_freedom` re-derived the stratum codes and nested (stratum, PSU) codes that `build_taylor_design` had just built, from the same columns via the same calls — two full densification passes over the design columns per estimate. At 1M rows over 2000 PSUs that was 11.3 ms on top of the design build's 11.0 ms, i.e. **63% of a 35 ms kernel spent indexing the same two columns twice.** A new `degrees_of_freedom_from_design` takes the df off the design's own code vectors; because those codes come from identical calls on identical inputs, the df is bit-identical by construction rather than merely equivalent. Applied to the ungrouped and batched mean, total, ratio and proportion kernels. The median kernels still double-index — their design is built inside the Woodruff variance and is not available to reuse — but they are sort-bound, so the share is smaller.

- **The ungrouped kernels overlap their two independent halves.** Indexing the design reads none of the response columns, and the point estimate, scores and SRS variance read none of the design columns; likewise the variance and the df both read the finished design but neither feeds the other. Both pairs now run under `rayon::join`, and the ungrouped entry points release the GIL (`Python::detach`) so the second half can actually be picked up — previously only the by-group and batched paths did, and `taylor_mean`'s ungrouped branch held the interpreter lock through the whole call. Each half is internally unchanged and still accumulates in row order.

- **`scores_mean_arr` skips a score-vector round-trip.** `taylor_variance` immediately converted the `Float64Chunked` from `scores_mean` back into a `Vec<f64>`, so the chunked array was two 8 MB copies at 1M rows that existed only to be undone. Callers that need the array take it directly; null positions still map to `0.0` exactly as before.

**Results are numerically inert.** Estimates, standard errors, variances and degrees of freedom are bit-for-bit identical to 0.12.0 across stratified/clustered, stratified-only, PSU-only and unstratified designs, with and without `where=` zero weights, and identical at 1, 2 and 10 rayon threads — the thread-count invariance the parallelism policy in `estimation/mod.rs` requires.

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

[Unreleased]: https://github.com/samplics-org/svy/compare/svy-rs-v0.15.0...HEAD
[0.15.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.15.0
[0.14.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.14.0
[0.13.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.13.0
[0.12.1]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.12.1
[0.12.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.12.0
[0.11.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.11.0
[0.10.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.10.0
[0.9.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.9.0
