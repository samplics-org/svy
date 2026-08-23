# Changelog

All notable changes to **svy-io**, high-speed reading and writing of survey files (SAS, SPSS, Stata) as Polars frames via ReadStat, are recorded here. Releases follow [Semantic Versioning](https://semver.org/); the layout follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Part of the [svy](../svy/CHANGELOG.md) project.

## [Unreleased]

<!-- ### Added, ### Changed, ### Fixed, ### Deprecated, ### Removed, ### Security -->

### Added

- **The public surface is now pinned by a test.** Every name in `__all__` must resolve, be callable, appear once, and match what `from svy_io import *` actually yields; each documented alias must be the *same object* as its target; and every module must be imported by something. This guards a failure that has already happened: `svy` called `svy_io.write_spss` and `svy_io.write_sas`, neither of which has ever existed, and both calls shipped with a `# type: ignore[attr-defined]` silencing the type checker. Nothing failed until someone ran the writer. Verified with probes — adding a phantom name to `__all__` fails three of these tests, and adding a module nothing imports fails another.
- **`read_spss` dispatch and `get_user_missing_for_column` are tested.** Both were public and referenced by no test. `read_spss` is not a reader but a dispatcher that picks one by file extension, so the routing is the whole function: `.sav` now provably produces exactly what `read_sav` does, arguments are forwarded rather than dropped, the match is case-insensitive, and an unrecognized extension raises instead of guessing from content.

### Removed

- **BREAKING: `VarMeta`, `ValueLabels`, `MissingRule` and `SvyMetadata` are no longer exported.** These dataclasses were public and nothing in the package ever constructed one. They also disagreed with what the readers return — `SvyMetadata.value_labels` was declared `Dict[str, ValueLabels]` where a reader hands back a `list` of plain dicts, and `VarMeta` lacked fields the native layer emits. Anyone importing them to type code against `read_sav` was being misled by them. Nothing in `svy` or in this package referenced them.
- **`utils.py`.** Six functions, 36 statements, 0% coverage — because `__init__` did not import it and nothing else in the package or the tests did either. Not undertested: unreachable.

### Added

- **Readers surface the declared measurement level ([#130](https://github.com/samplics-org/svy/issues/130)).** Each entry in `meta["vars"]` now carries `measure` — `"nominal"`, `"ordinal"`, or `"scale"` — read from ReadStat's `readstat_variable_get_measure`. A format that carries no such attribute (Stata) or a variable whose writer never set one reports `None` rather than `"scale"`, so a caller can tell "declared continuous" from "never declared". Worth knowing before relying on it: SPSS defaults numeric variables to `"scale"` whether or not anyone meant it, so only `"nominal"` and `"ordinal"` are positive declarations.

### Fixed

- **`ordered=True` did nothing.** The parameter is public on `read_dta`, `read_sas`, `as_factor`, `as_factor_expr` and `apply_value_labels`, and it had no effect on either path. The lazy one passed `ordering="physical"` to `pl.Categorical`, which polars deprecated in 1.32.0 and now ignores — and this package already requires polars ≥ 1.34, so it was inert for every supported version. The eager one never read the argument; it was marked "reserved". A `Categorical` sorts its categories alphabetically, so an education scale came back `Higher < None < Primary < Secondary`. `ordered=True` now builds a `pl.Enum`, which keeps the order it is given, and the scale sorts `None < Primary < Secondary < Higher`. Categories are ordered by the **numeric** value of the code: readers return value labels keyed by the code's string form, so the mapping iterates `"1", "10", "2"` and using that order directly would sequence an 11-category scale wrongly. Non-numeric codes keep their string order.
- **`ordered=True` now refuses what it cannot order.** `levels="default"` and `levels="both"` fall back to the raw value for anything unlabelled, so their categories depend on data the label set cannot describe; combining them with `ordered=True` raises rather than silently dropping the unlabelled values. `ordered=True` without value labels raises for the same reason — the code order is what defines the order.
- **Deprecation warnings fail the test run.** The suite carried a polars `DeprecationWarning` in its summary for releases while the deprecated argument silently did nothing. Note for anyone tempted to narrow the filter back to `error::DeprecationWarning:polars`: that form never matches. The module field is compared against the frame the warning is attributed to, and polars sets `stacklevel` to point at the calling code, so the qualified filter silently does nothing.

### Fixed

- **`write_dta` silently dropped value labels ([#129](https://github.com/samplics-org/svy/issues/129)).** The native writer accepted `value_labels_json` and never read it, so a `.dta` was written with its variable labels intact and its value labels gone — no error, and the loss only visible on read-back. The Stata writer now emits a label set per labelled column and points the variable at it, verified against `pandas.io.stata` for formats 113 through 119. The set is named after the column (Stata's own `label values v106 v106` convention) rather than the SAV writer's `{col}_labels`, because dta 113–117 allow only 33 bytes for the name and ReadStat truncates a longer one without complaint.
- **`write_dta` value-label validation.** Codes are now rejected when they fall outside `[-2147483647, 2147483620]` (Stata stores a code as int32 and reserves the top of that range for `.` through `.z`, so a wider code was truncated silently), when they name a column absent from the frame, and when they target a string column (Stata has no string value labels, unlike SPSS). `bool` and whole-`float` codes are canonicalized to plain integers, so `{True: "yes"}` writes code `1` instead of failing at the JSON boundary.

### Removed

- **The benchmark suite.** Three of its five tests were the same benchmark: `test_bench_stata_types_13`, `_14` and `_15` all read one file with no arguments, and reported 29.5 / 29.2 / 29.4 ms — one number, printed three times, under names promising three Stata formats. Half the file was commented out, so `make bench-spss` and `make bench-sas` selected zero tests and exited green. Nothing compared any number to a baseline. It cost ~6s on every test run and an 876 KB data file to say nothing actionable. `svy`'s harness (`bench_kernel.py`, `check_regression.py`, tracked `baselines/`) is the shape to copy if svy-io wants perf tracking later.

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
