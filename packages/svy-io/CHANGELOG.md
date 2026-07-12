# Changelog

All notable changes to **svy-io** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
`svy-io` adheres to [Semantic Versioning](https://semver.org/). Entries are
written for readers — _what changed and why it matters_ — not raw commit logs.
Add each change under `[Unreleased]` in the same PR that makes it; on release,
rename `[Unreleased]` to the new version + date and open a fresh `[Unreleased]`.

## [Unreleased]

<!-- ### Added, ### Changed, ### Fixed, ### Deprecated, ### Removed, ### Security -->

## [0.1.1] — 2026-07-12

### Fixed

- **SAS datetime values are now decoded correctly.** Datetime formats were
  matched by a `"date"` prefix test, so `DATETIME` columns were sent through the
  days-since-1960 date path and lost their time-of-day. Datetime formats are now
  checked first and decoded against a true `Datetime` epoch, preserving the time
  component. Also fixes an `AttributeError` on variables with a null format on
  the default `read_sav` path.
- **Numeric ID columns stay numeric.** The magnitude/name-based temporal
  inference heuristics are now gated behind `infer_temporal_formats` (opt-in), so
  numeric identifier columns are no longer coerced to dates/times.
- **`as_factor(levels="both")` on numeric coded columns** no longer mis-handles
  literal separators; values are stringified before the `Categorical` cast.
- **`write_sav` no longer mutates the caller's `value_labels`.**
- **All file-like reader inputs** work again (a missing `tempfile` import
  crashed them).
- **FFI hardening.** Builder pre-allocation is clamped to 65,536 rows so a
  crafted header can no longer drive a multi-GB eager allocation and abort the
  process; ReadStat callbacks are wrapped in panic guards that raise Python
  exceptions instead of aborting; the XPT writer now emits all record batches
  (rows after the first batch were silently dropped) and handles
  StringView/dictionary columns.

### Packaging

- **Source builds and Intel macOS now work.** The sdist previously shipped
  without ReadStat's C sources (an `include` glob matched zero files), so every
  source build failed — and Intel Macs always hit that path because no
  `x86_64-apple-darwin` wheel was published. The sdist now bundles all ReadStat
  sources, and prebuilt Intel macOS wheels are published.

## [0.1.0] — 2026-05

First release tracked in this changelog. For earlier history, see the
[Git tags](https://github.com/samplics-org/svy/tags).

[Unreleased]: https://github.com/samplics-org/svy/compare/svy-io-v0.1.1...HEAD
[0.1.1]: https://github.com/samplics-org/svy/releases/tag/svy-io-v0.1.1
[0.1.0]: https://github.com/samplics-org/svy/releases/tag/svy-io-v0.1.0
