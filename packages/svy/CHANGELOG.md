# Changelog

All notable changes to **svy**, the Python package for design-based analysis of complex survey data — means, totals, ratios, proportions, regression, weighting, and sample selection — are recorded here. Releases follow [Semantic Versioning](https://semver.org/); the layout follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Companion packages track their own changes: [`svy-io`](../svy-io/CHANGELOG.md) (SAS/SPSS/Stata I/O) and [`svy-rs`](../svy-rs/CHANGELOG.md) (internal Rust extension).

## [Unreleased]

<!-- ### Added, ### Changed, ### Fixed, ### Deprecated, ### Removed, ### Security -->

## [0.21.0] — 2026-07-24

Requires [`svy-rs`](../svy-rs/CHANGELOG.md) 0.12.0, which carries the two variance-estimation fixes below; [`svy-io`](../svy-io/CHANGELOG.md) 0.2.0 is unchanged. Grouped confidence intervals and domain design effects change in this release — point estimates and standard errors do not.

### Fixed

- **`by=` groups now use their own degrees of freedom.** A by-group is a domain, so its df must be counted on the PSUs and strata that group covers. It was instead given the df of the surrounding analysis — the full design with no filter, or the `where=` mask with one — so the same subpopulation got a different interval depending on whether it was reached through `by=` or `where=`. Confidence intervals for grouped means, totals, ratios, proportions and medians were consistently **too narrow**; the effect is negligible for groups spanning most of the sample and large for small ones (22% on a 10-record domain with 6 df rather than 56). `est`, `se`, `cv` are unaffected. Verified against R `survey` 4.5 `degf(subset(design, ...))`.
- **Design effects no longer count zero-weight rows.** Under `drop_nulls`, rows with a missing response are kept and zero-weighted rather than dropped; they were still counted in the domain SRS variance's `n`, inflating `deff` for any group containing them (~1–2% on the synthetic fixtures). Only `deff` is affected.

### Removed

- **`Estimate.degrees_freedom`.** Degrees of freedom are a per-row property — a domain or by-group is counted on its own active PSUs and strata, so grouped results legitimately carry a different df per cell. The scalar could not represent that: it was `min()` across rows, so a grouped estimate reported its *smallest* group, and for a by-group inside a domain that meant a headline df of 0. Use `ParamEst.df` (also a `df` column in `to_polars()`) for the per-row value, and `n_psus - n_strata` for the full-design df, which stays at design level under a domain filter.
- **`EstimateData.degrees_freedom`** leaves the serialized payload; `SCHEMA_VERSION` moves to `svy-result/0.2`. Strictly a field removal warrants a major bump under the policy in `serialize/DESIGN.md`; 0.2 was chosen deliberately because no known consumer binds to the field, and the reasoning is recorded there.

### Added

- **`ParamEst.df`** — the design df backing each row's t-quantile, carried through `to_polars()` and the serialized payload. It is deliberately not shown in the printed table: it is constant for most results, so a column would repeat one number down the page and widen every table.

## [0.20.1] — 2026-07-23

Patch release on top of 0.20.0; [`svy-rs`](../svy-rs/CHANGELOG.md) (0.11.0) and [`svy-io`](../svy-io/CHANGELOG.md) (0.2.0) are unchanged.

### Fixed

- **`tabulate` percent and `count_total` cells used an un-centered variance.** A cell percentage is a ratio of two estimated totals, so its variance needs the centered (Hájek) linearization. Because the internal totals flag was inferred from `sum(weights) != 1`, scaling weights to sum to 100 (`units="percent"`) or to a caller-supplied `count_total` routed the standard error through the un-centered total path, dropping the numerator/denominator covariance term. Cell SEs were inflated by a `p`-dependent amount (up to ~12% on high-proportion cells) and the confidence interval fell back to Wald, which could dip below zero. `units="proportion"`, `units="percent"`, and `count_total=N` are now the same estimator scaled by a constant and agree exactly; they match `estimation.prop` and R `survey`'s `svymean(~interaction(...))`. Bare `units="count"` is unchanged and still matches R's `svytotal`, and the Rao-Scott chi-square/F test was never affected.

## [0.20.0] — 2026-07-23

Builds on [`svy-rs`](../svy-rs/CHANGELOG.md) 0.11.0 and [`svy-io`](../svy-io/CHANGELOG.md) 0.2.0. This release lands the round 7–8 review: correctness fixes across estimation, regression, weighting, size/power, categorical, and the dataset downloader, several of which shift standard errors closer to R `survey` 4.5.

### Added

- **`RepWeights.rscales` — exact stratified-JKn variance.** `RepWeights` gains an optional `rscales` tuple (per-replicate variance coefficients, R's `scale`×`rscales` combined); `create_jk_wgts` fills it from the design's strata and estimation threads it to the Rust kernels. svy-generated JKn weights now reproduce R's `as.svrepdesign(type="JKn")` mean/total SEs and `mse=TRUE` centering to 13+ digits (df = degf). Absent `rscales`, each method keeps its global default, so user-supplied replicate weights behave exactly as before unless the file's documented `rscales` are provided.

### Fixed

- **`drop_nulls` zeroes weights instead of dropping rows** (R `na.rm=TRUE` / `subset()` semantics). `prepare_data` physically removed any row with a missing analysis value before the domain machinery ran; under standard skip patterns (`y` null outside the domain) this deleted whole PSUs and strata, understating domain SEs — 15% on the reference dataset — and corrupting df. Missing analysis values now keep their rows with main and replicate weights zeroed. Verified against R `survey` 4.5 to 13+ digits; **the R-calibrated ttest and ratio fixtures were regenerated with these semantics** (the old expectations matched R only on physically-filtered complete-case data).
- **Float-typed stratum/PSU columns are accepted.** Numeric design codes from CSVs (e.g. MEPS `VARSTR`/`VARPSU`) frequently arrive as `Float64`; the factorized-design cache cast them straight to `Categorical`, which polars forbids for floats, crashing estimation with "conversion from f64 to cat failed". Non-string, non-integer dtypes now route through `Utf8` first (float- and int-coded designs produce identical results).
- **SSU-level FPC is grouped by `(stratum, PSU)`, not PSU alone.** PSU labels are commonly reused across strata, so `build_fpc_ssu_column` merged distinct PSUs — valid designs raised `FPC_NOT_CONSTANT`, and matching `M_hi` values pooled SSU counts across strata, understating the two-stage SSU FPC.
- **`method=None` auto-detects** as documented — replication when the only variance information is replicate weights (no strata/PSU), Taylor otherwise. Previously `None` always meant Taylor, silently giving replication-only designs an SRS-like variance.
- **Core polish and API consistency (review round 7).** Replicate-weight prefix matching is strict `^prefix\d+$` (a loose `startswith` absorbed columns like `repwt_flag`; a count/`n_reps` mismatch is a typed `DimensionError`); `set_data`/`update_data`/`set_design`/`update_design` rebuild internal concat columns and re-run singleton detection + design validation instead of leaving stale state; `describe()` reports weighted std/var/quantiles (aweight convention) and computes categorical proportions over all levels; `SingletonHandling` enum values are accepted by `singleton.handle()`; `PopSize(psu=..., ssu=None)` is accepted for PSU-only FPC; `polars_mask()` is null-safe; the design-fields cache is bounded (512 entries); importing `svy` no longer replaces the host's `sys.excepthook` (Rich tracebacks install only on `SVY_RICH=1`). Deleted the unused content-based `Sample.__hash__` and the dead `_calculate_fpc`.
- **GLM design gaps (round 8).** Family-specific unit deviance (matching R `family$dev.resids`) and null deviance at the intercept-only fit; deviance/AIC follow R `survey` exactly (Lumley–Scott dAIC; `bic` is `None`); replicate-weight designs get true replicate variance instead of silently falling back to Taylor SEs; `design.pop_size` feeds per-stratum FPC into the sandwich; `Cat(ref=...)` with an absent reference level raises a typed error listing observed levels; Cat levels, the response, and the invalid-weight filter are evaluated on in-domain rows under `where=`, eliminating phantom all-zero dummies; covariate/`where`-column nulls keep-and-zero-weight (preserving PSUs in stratum centering). Validated against R `survey` 4.5 to ~1e-6 or better.
- **GLM margins rewritten on the fitted frame with delta-method SEs (round 8).** `margins` recomputed from raw sample data with ad-hoc SE formulas; it now averages over exactly the fitted rows (post null-drop, post weight filter, with the domain column), rebuilds interaction columns from counterfactual data, differentiates the full linear predictor for AME, and uses full delta-method SEs `g'V(β)g` over the design-based covariance (Stata `vce(delta)` convention). Validated against R `survey` + `marginaleffects`: points to ~1e-8, SEs to ~1e-4.
- **Weighting adjustment/calibration/trimming marshalling (round 8).** `adjust` raises a typed error on unmatched response statuses (was silently encoding them as respondents and inflating weights) and derives `respondents_only` from the encoded codes (case-insensitive); `adjust(trimming=..., update_design_wgts=False)` trims the freshly created adjusted weight instead of the caller's original; `calibrate(bounded=True)` raises `NotImplementedError` instead of being silently ignored; calibration targets are assembled as ordered per-term lists (fixing a "Design matrix label alignment mismatch" on shared numeric codes); the trim-calibrate cycle runs on arrays before writing (a strict non-convergence failure leaves data/design/replicates untouched) and honors `TrimConfig.by`/`min_cell_size`; `build_aux_matrix` raises on nulls in a continuous auxiliary instead of filling `0.0`.
- **Weighting typed errors and sorted control order** for the svy-rs 0.11.0 changes: `create_brr_wgts` pre-validates `n_reps` against the Hadamard order (`MethodError.invalid_range`); raking-bounds violations surface as `MethodError` at all four kernel call sites; `normalize()` orders control values by sorted group id, matching the kernel and `poststratify`.
- **Wrangling edge cases (round 8).** `categorize()` closes the outer bin edge (R `cut(include.lowest=TRUE)`) so boundary values no longer vanish from tabulations; `remove_columns(force=True)` cleans `design.pop_size`; a partial replicate-weight `rename_columns` raises instead of corrupting the `RepWeights` prefix; `mutate()` specs see same-call redefinitions (dependents no longer read stale values); `clean_names()` preserves internal concat columns; `filter_records()` counts and reports Kleene-null-dropped rows; `fill_null(strategy="mean")` casts integer columns to `Float64` for an exact mean; `cast(strict=True)` raises on lossy float-to-integer casts.
- **Size and power formulas (round 8).** `compare_means` is implemented (was a no-op stub returning `None`); non-inferiority sizing keeps `epsilon` signed (the old `|eps|` collapse under-sized NI designs ~5×); the one-mean two-sided clamp that produced astronomically wrong `n` is removed; one-sided power follows `sign(delta)`; pooled two-proportion variance and the optimal allocation ratio are un-inverted; the adjustment pipeline is reordered to `n0 → DEFF → FPC → nonresponse` so the FPC caps the deff-inflated size toward `pop_size`; parameter validation (p/moe/sigma/power/deff/resp_rate) raises typed `MethodError` instead of silently clipping.
- **`tabulate` count CIs use the design-df t** instead of the normal critical value; with few PSUs (df = 6) count CIs were ~20% too narrow, now matching Stata `svy: tabulate` and svy's own `estimation.total`.
- **`ranktest` with a custom `score_fn` honors `by=`** (each by-level is its own domain, returning one result per level) and **group labels reflect the levels actually tested** under `where=`/`by=` (estimates were always correct; only the reported labels were wrong).

### Security

- **Dataset downloader hardened against a hostile catalog.** Slugs from registry JSON flowed unvalidated into cache paths, glob patterns, and tempfile prefixes (a slug like `../../foo` wrote outside `~/.svy/datasets`); slugs are now allowlisted at the registry boundary and defensively in `path_for`/`clear`. Downloads without a catalog hash pin the first-seen sha256 (trust-on-first-use) and enforce it thereafter. Plain-http URLs and https→http redirect downgrades are rejected (localhost exempt for development). New error codes `DATASET_INVALID_SLUG` and `DATASET_INSECURE_URL`.

## [0.19.1] — 2026-07-21

### Added

- **Bundled offline example datasets.** `svy.datasets.load` / `catalog` / `describe` now take a `source=` argument — `"bundled"`, `"remote"`, or `"auto"` (default: remote if reachable, else bundled). A small, self-consistent synthetic survey — a sampling frame, its household census, and a two-stage sample drawn from that census (design weights sum to the census) — ships inside the wheel, so the docs and your own experiments run fully offline and reproducibly. `SVYLAB_OFFLINE=1` forces the bundled path.
- **`DatasetCatalog` type and richer `Dataset` metadata.** `catalog()` returns a `DatasetCatalog` that prints as a compact table and drills into any entry's full metadata with `.get(slug)` (also `.slugs`, `.to_polars()`). `Dataset` prints as a branded panel and gained a `notes` field documenting how a bundled subset was derived from its remote counterpart.

### Changed

- **All dataset failures route through the `DatasetError` taxonomy** with actionable messages and codes: `DATASET_NOT_BUNDLED` (lists the available bundled slugs), `DATASET_DOWNLOAD_FAILED`, and `BUNDLED_UNAVAILABLE`, alongside the existing not-found, catalog, and integrity errors.

### Fixed

- **`SvyError` panels render again.** The Rich panel path imported its renderers from a module that had since been renamed, so every error silently fell back to plain text; it now renders the branded panel. The panel also stays aligned in HTML/notebook output — the status marker is a width-1 glyph instead of a two-cell emoji — and the title, body, and metadata are spaced for readability.

## [0.19.0] — 2026-07-12

### Added

- **Batched multi-variable estimation.** `estimation.mean`, `total`, `ratio`, `prop`, and `median` now accept a list of columns and return a `list[Estimate]` (one per variable; `ratio` pairs numerator/denominator element-wise and broadcasts a scalar side). A single string still returns a single `Estimate`. For ungrouped Taylor estimation the list form shares one design build across variables and runs them in parallel — 4–13× faster than a manual loop at 1M rows depending on the estimator. `by=`, replication, `drop_nulls`, and the singleton scale double-pass transparently fall back to independent per-variable calls (identical results).
- **A variable may now appear in both `by=` and `where=`.** `where` is domain estimation (out-of-domain weights zeroed) and `by` groups on the original values — the two are orthogonal, so the previous guard forbidding overlap is removed. When a `where` predicate excludes an entire `by` level (e.g. a "don't know" code), that level is correctly absent from the results — matching R's `filter(...) %>% group_by(...)` — while every row still contributes to the shared design and degrees of freedom, so surviving groups' estimates, standard errors, and df are byte-identical. Covers Taylor and replication, all estimands, and multi-`by`.
- **Serialization for result objects.** New `svy.serialize` module provides stable, versioned serialization of every result type (estimates, t-tests, chi-square, tables, GLM fits/predictions, describe): `serialize(result)` returns a kind-tagged struct, `to_json` / `to_dict` export, and `from_json` round-trips. Payloads carry a `SCHEMA_VERSION` for forward compatibility.
- **Single-stage designs and explicit population sizes.** The design's `ssu` (second-stage unit) is now optional, so single-stage designs no longer need a placeholder. A `PopSize` type is exported for specifying finite-population sizes (FPC).

### Changed

- **Estimation now fails fast on unhandled singleton PSUs** instead of silently under-reporting the variance. Taylor estimation (`mean`, `total`, `prop`, `ratio`, `median`) raises `SingletonError` when a design has single-PSU strata and no handling strategy was chosen — matching R's `options(survey.lonely.psu = "fail")`. Pick a strategy explicitly with `sample.singleton.skip()` / `.certainty()` / `.center()` / `.scale()` / `.collapse()` / `.pool()`. Previously such strata were dropped from the variance with no error or warning.

### Fixed

- **Taylor standard errors are now bit-reproducible.** The stratified variance summed each stratum's PSUs in the iteration order of a randomized hash set, so a repeated estimate on identical data could differ in its last digits run-to-run (far below reporting precision, but not reproducible). PSUs are now summed in a canonical order, so `mean`/`total`/`ratio`/`prop`/`median` return identical standard errors across runs.
- **Stale design cache could return silently wrong results.** Estimation design caches were keyed on the identity of the data frame without holding a reference to it; after an in-place mutation freed and reallocated the frame, identity reuse could make a stale entry look current and serve design arrays for the old data. Caches are now keyed on a monotonic per-`Sample` data version bumped on every rebind, so every mutation, weighting, selection, and fork path invalidates correctly.
- **Replication-design crashes and related correctness fixes.** Clone, column keep/remove/rename, and singleton handling now work on replication designs (previously hit stale replicate-weight API usage and could crash). `Expr` now raises `TypeError` on boolean use (`and`/`or`/`not`/chained comparisons) so a malformed `where=` predicate fails loudly instead of silently filtering wrong, and derived samples deep-copy metadata/warnings/design so they no longer share mutable state with the original.

## [0.18.2] — 2026-05-20

First release tracked in this changelog. For the history prior to 0.18.2, see the [Git tags](https://github.com/samplics-org/svy/tags) and [GitHub Releases](https://github.com/samplics-org/svy/releases).

[Unreleased]: https://github.com/samplics-org/svy/compare/svy-v0.21.0...HEAD
[0.21.0]: https://github.com/samplics-org/svy/releases/tag/svy-v0.21.0
[0.20.1]: https://github.com/samplics-org/svy/releases/tag/svy-v0.20.1
[0.20.0]: https://github.com/samplics-org/svy/releases/tag/svy-v0.20.0
[0.19.1]: https://github.com/samplics-org/svy/releases/tag/svy-v0.19.1
[0.19.0]: https://github.com/samplics-org/svy/releases/tag/svy-v0.19.0
[0.18.2]: https://github.com/samplics-org/svy/releases/tag/svy-v0.18.2
