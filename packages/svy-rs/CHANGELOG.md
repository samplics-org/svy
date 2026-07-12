# Changelog

All notable changes to **svy_rs** are documented in this file.

`svy_rs` is the internal Rust extension backing the `svy` package and is not a
supported public API — depend on `svy`, not on this directly. Entries here are
technical and describe what changed for `svy`'s consumers of the extension. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

<!-- ### Added, ### Changed, ### Fixed, ### Deprecated, ### Removed, ### Security -->

## [0.10.0] — 2026-07-12

### Added

- **Batched multi-variable estimation kernels.** `mean`, `total`, `ratio`,
  `prop`, and `median` accept multiple variables and share a single design build,
  running them in parallel for ungrouped Taylor estimation. Backs `svy`'s
  list-input estimation API.

### Fixed

- **Deterministic Taylor variance.** Per-stratum PSU contributions were summed in
  the iteration order of a standard `HashSet`, making the last digits of a
  standard error vary run-to-run. PSUs are now summed in a canonical (sorted)
  order, so results are bit-reproducible.

### Performance

- **Phase C–E estimation and replicate-weight optimizations:** dtype-polymorphic
  design indexing with cached integer codes; the design is indexed once per
  by-group/level loop; replicate mean/total/ratio/proportion kernels accumulate
  without materializing the replicate matrix; cache-blocked parallel
  replicate-weight matrix build; fused mean/total/ratio domain kernels; by-group
  Taylor and by-domain GLM parallelized with the GIL released; Taylor kernel fast
  paths (~2× at 1M rows).

### Build

- Bump `ethnum` 1.5.2 → 1.5.3 for newer Rust toolchains.

## [0.9.0] — 2026-05-19

Baseline for this changelog. For earlier history, see the
[Git tags](https://github.com/samplics-org/svy/tags).

[Unreleased]: https://github.com/samplics-org/svy/compare/svy-rs-v0.10.0...HEAD
[0.10.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.10.0
[0.9.0]: https://github.com/samplics-org/svy/releases/tag/svy-rs-v0.9.0
