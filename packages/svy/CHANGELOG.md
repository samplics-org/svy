# Changelog

All notable changes to **svy** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
`svy` adheres to [Semantic Versioning](https://semver.org/). Entries are written
for readers — _what changed and why it matters_ — not raw commit logs. Add each
change under `[Unreleased]` in the same PR that makes it; on release, rename
`[Unreleased]` to the new version + date and open a fresh `[Unreleased]`.

## [Unreleased]

### Added

- **Batched multi-variable estimation.** `estimation.mean`, `total`, `ratio`,
  `prop`, and `median` now accept a list of columns and return a
  `list[Estimate]` (one per variable; `ratio` pairs numerator/denominator
  element-wise and broadcasts a scalar side). A single string still returns a
  single `Estimate`. For ungrouped Taylor estimation the list form shares one
  design build across variables and runs them in parallel — 4–13× faster than a
  manual loop at 1M rows depending on the estimator. `by=`, replication,
  `drop_nulls`, and the singleton scale double-pass transparently fall back to
  independent per-variable calls (identical results).

- **A variable may now appear in both `by=` and `where=`.** `where` is domain
  estimation (out-of-domain weights zeroed) and `by` groups on the original
  values — the two are orthogonal, so the previous guard forbidding overlap is
  removed. When a `where` predicate excludes an entire `by` level (e.g. a
  "don't know" code), that level is correctly absent from the results — matching
  R's `filter(...) %>% group_by(...)` — while every row still contributes to the
  shared design and degrees of freedom, so surviving groups' estimates, standard
  errors, and df are byte-identical. Covers Taylor and replication, all
  estimands, and multi-`by`.

- **Serialization for result objects.** New `svy.serialize` module provides
  stable, versioned serialization of every result type (estimates, t-tests,
  chi-square, tables, GLM fits/predictions, describe): `serialize(result)`
  returns a kind-tagged struct, `to_json` / `to_dict` export, and `from_json`
  round-trips. Payloads carry a `SCHEMA_VERSION` for forward compatibility.

- **Single-stage designs and explicit population sizes.** The design's `ssu`
  (second-stage unit) is now optional, so single-stage designs no longer need a
  placeholder. A `PopSize` type is exported for specifying finite-population
  sizes (FPC).

### Changed

- **Estimation now fails fast on unhandled singleton PSUs** instead of silently
  under-reporting the variance. Taylor estimation (`mean`, `total`, `prop`,
  `ratio`, `median`) raises `SingletonError` when a design has single-PSU strata
  and no handling strategy was chosen — matching R's
  `options(survey.lonely.psu = "fail")`. Pick a strategy explicitly with
  `sample.singleton.skip()` / `.certainty()` / `.center()` / `.scale()` /
  `.collapse()` / `.pool()`. Previously such strata were dropped from the
  variance with no error or warning.

### Fixed

- **Taylor standard errors are now bit-reproducible.** The stratified variance
  summed each stratum's PSUs in the iteration order of a randomized hash set, so
  a repeated estimate on identical data could differ in its last digits
  run-to-run (far below reporting precision, but not reproducible). PSUs are now
  summed in a canonical order, so `mean`/`total`/`ratio`/`prop`/`median` return
  identical standard errors across runs.

- **Stale design cache could return silently wrong results.** Estimation design
  caches were keyed on the identity of the data frame without holding a
  reference to it; after an in-place mutation freed and reallocated the frame,
  identity reuse could make a stale entry look current and serve design arrays
  for the old data. Caches are now keyed on a monotonic per-`Sample` data version
  bumped on every rebind, so every mutation, weighting, selection, and fork path
  invalidates correctly.

- **Replication-design crashes and related correctness fixes.** Clone, column
  keep/remove/rename, and singleton handling now work on replication designs
  (previously hit stale replicate-weight API usage and could crash). `Expr` now
  raises `TypeError` on boolean use (`and`/`or`/`not`/chained comparisons) so a
  malformed `where=` predicate fails loudly instead of silently filtering wrong,
  and derived samples deep-copy metadata/warnings/design so they no longer share
  mutable state with the original.

<!-- Also available when needed: ### Deprecated, ### Removed, ### Security -->

## [0.18.2] — 2026-05-20

First release tracked in this changelog. For the history prior to 0.18.2, see the
[Git tags](https://github.com/samplics-org/svy/tags) and
[GitHub Releases](https://github.com/samplics-org/svy/releases).

<!-- Optional: backfill a few notable 0.18.2 highlights here. -->

[Unreleased]: https://github.com/samplics-org/svy/compare/svy-v0.18.2...HEAD
[0.18.2]: https://github.com/samplics-org/svy/releases/tag/svy-v0.18.2
