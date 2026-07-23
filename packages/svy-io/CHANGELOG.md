# Changelog

All notable changes to **svy-io**, high-speed reading and writing of survey files (SAS, SPSS, Stata) as Polars frames via ReadStat, are recorded here. Releases follow [Semantic Versioning](https://semver.org/); the layout follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Part of the [svy](../svy/CHANGELOG.md) project.

## [Unreleased]

<!-- ### Added, ### Changed, ### Fixed, ### Deprecated, ### Removed, ### Security -->

## [0.2.0] — 2026-07-23

### Changed

- **BREAKING: unified `user_missing` metadata schema.** Three producers emitted three shapes — `{var, discrete, ranges}` from the native layer, `{col, values, range}` from `read_sav(user_na=True)`, while `zap_missing` looked for `na_values`/`na_range` — so zapping the missing metadata returned by a real `read_sav` silently did nothing (the zap tests passed only on hand-crafted dicts). All readers now emit one haven-compatible schema, `{col, na_values, na_range}` per column, via `normalize_user_missing()` (tolerating the legacy shapes). Code that reads `user_missing` metadata must use the new keys.

### Fixed

- **Native-layer hardening.** Encoding parameters (and the SAS `catalog_encoding`) now flow into `readstat_set_file_character_encoding` (iconv) instead of being accepted and silently ignored, so legacy code-page files no longer arrive riddled with U+FFFD; invalid encoding names error, and metadata gains `had_invalid_utf8` so silent lossy decoding is detectable. `n_rows` is counted once per row independent of kept columns (it reported 0 when the first column was skipped); the native `n_max=0` off-by-one is fixed and `set_row_limit` is applied as a defense-in-depth guard for untrusted files. `write_xpt` and `write_sav` now derive and validate string widths from the data (width was hardcoded to 200 / silently capped, leaving truncated files) before writing any bytes. Hand-declared ReadStat externs were replaced with the bindgen bindings so signature drift is caught at build time.
- **Data-handling edge cases.** `n_max=0` now opens and validates the file and returns the full schema with zero rows (haven behavior) instead of a schemaless empty frame; `write_sav` encodes categorical columns against their own observed categories with per-column codes under a global `pl.StringCache` (was value labels for every cached string in cache order); `as_factor_expr(levels="labels")` maps unlabelled values to null; `LabelledSPSS.to_int()` keeps missing values `None` instead of `0`; `_stata_file_format` rejects nonexistent format codes; and `_adjust_temporals` narrows a bare `except` to polars/value errors with a warning.
- **Column-name collisions.** `_normalize_names` disambiguates two source names that normalize to the same column with numeric suffixes (was a duplicate-rename error) and renames `user_missing` columns alongside.

### Security

- **Private per-call temp dir for zip extraction.** `read_sas` extracted archive members to predictable paths in the shared system temp dir and never cleaned them up — cross-run collisions and symlink-planting exposure on multi-user machines, plus an unbounded temp leak from the `delete=False` spool of file-like inputs. All temp artifacts are now scoped to an `ExitStack` closed right after the native parse; spooled inputs are unlinked.

### Build

- Bump `pyo3` to 0.29 and `bytes` to 1.12.1 in the native extension.

## [0.1.1] — 2026-07-12

### Fixed

- **SAS datetime values are now decoded correctly.** Datetime formats were matched by a `"date"` prefix test, so `DATETIME` columns were sent through the days-since-1960 date path and lost their time-of-day. Datetime formats are now checked first and decoded against a true `Datetime` epoch, preserving the time component. Also fixes an `AttributeError` on variables with a null format on the default `read_sav` path.
- **Numeric ID columns stay numeric.** The magnitude/name-based temporal inference heuristics are now gated behind `infer_temporal_formats` (opt-in), so numeric identifier columns are no longer coerced to dates/times.
- **`as_factor(levels="both")` on numeric coded columns** no longer mis-handles literal separators; values are stringified before the `Categorical` cast.
- **`write_sav` no longer mutates the caller's `value_labels`.**
- **All file-like reader inputs** work again (a missing `tempfile` import crashed them).
- **FFI hardening.** Builder pre-allocation is clamped to 65,536 rows so a crafted header can no longer drive a multi-GB eager allocation and abort the process; ReadStat callbacks are wrapped in panic guards that raise Python exceptions instead of aborting; the XPT writer now emits all record batches (rows after the first batch were silently dropped) and handles StringView/dictionary columns.

### Packaging

- **Source builds and Intel macOS now work.** The sdist previously shipped without ReadStat's C sources (an `include` glob matched zero files), so every source build failed — and Intel Macs always hit that path because no `x86_64-apple-darwin` wheel was published. The sdist now bundles all ReadStat sources, and prebuilt Intel macOS wheels are published.

## [0.1.0] — 2026-05

First release tracked in this changelog. For earlier history, see the [Git tags](https://github.com/samplics-org/svy/tags).

[Unreleased]: https://github.com/samplics-org/svy/compare/svy-io-v0.2.0...HEAD
[0.2.0]: https://github.com/samplics-org/svy/releases/tag/svy-io-v0.2.0
[0.1.1]: https://github.com/samplics-org/svy/releases/tag/svy-io-v0.1.1
[0.1.0]: https://github.com/samplics-org/svy/releases/tag/svy-io-v0.1.0
