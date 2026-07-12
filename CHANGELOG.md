# Changelog

All notable changes to **svy** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
`svy` adheres to [Semantic Versioning](https://semver.org/). Entries are written
for readers — *what changed and why it matters* — not raw commit logs. Add each
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

<!-- Also available when needed: ### Deprecated, ### Removed, ### Security -->

## [0.18.2] — 2026-05-20

First release tracked in this changelog. For the history prior to 0.18.2, see the
[Git tags](https://github.com/samplics-org/svy/tags) and
[GitHub Releases](https://github.com/samplics-org/svy/releases).

<!-- Optional: backfill a few notable 0.18.2 highlights here. -->

[Unreleased]: https://github.com/samplics-org/svy/compare/svy-v0.18.2...HEAD
[0.18.2]: https://github.com/samplics-org/svy/releases/tag/svy-v0.18.2
