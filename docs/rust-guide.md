# Rust guide for svy-rs and svy-io

These are the rules `rustfmt` and `clippy` can't check, plus the decisions we've made about the
Python boundary. For anything this guide doesn't cover, follow the
[Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).

Both crates are Python extensions. Their public API is the Python module, not the Rust
crate. Most rules below follow from that.

## Tooling

- `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` pass. The
  edition is 2024: running `rustfmt` on a single file needs `--edition 2024`.
- `#[allow(clippy::...)]` goes on the item, not the module. `too_many_arguments` is fine on
  `#[pyfunction]` wrappers, since their arguments mirror the Python call.

## Layout

- PyO3 appears only in the `api.rs` / `*_api.rs` wrappers and in `lib.rs`. Kernel modules take
  slices, `ndarray` views or polars types and return `Result`. That keeps them testable with
  `cargo test` and benchmarkable without Python.
- A wrapper converts its inputs, calls one kernel function and maps its error. Statistics
  live in the kernel modules, never in a wrapper.
- Every `#[pyfunction]` is registered in `lib.rs`. Items are `pub(crate)` unless a bench needs
  them.
- Name a module after its main item, and split a file once it needs a table of contents.

## Performance

Speed is a requirement. Every kernel is written for millions of rows and hundreds of
replicates. Choose the algorithm and data layout for that scale from the start. A
micro-optimisation still needs a benchmark showing it pays.

**Algorithm**
- Make one pass over the rows. Never re-filter the data for each domain, replicate or
  stratum. Accumulate every group in the same pass.
- Do expensive work once and reuse it: sort `y` once for every replicate quantile, and
  factorize design columns once per data version.
- Batch across the FFI boundary: one call per estimate, not one per domain or variable (see
  the `*_multi` kernels).

**Data layout**
- Design columns become dense `u32` codes (`design_col_codes`), and accumulators are `Vec`s
  indexed by code. Hash maps are used to build the codes and never appear in a loop over
  rows. When you do hash, use `FxHashMap`, not `std::collections::HashMap` (SipHash).
- Read columns as slices: `into_contiguous`, then `cont_slice()`. Fall back to per-element
  access only for null or chunked columns. Never call `.get(i)` on a `ChunkedArray` inside a
  loop.
- Write memory in order. When a strided write can't be avoided, block it to fit in cache (see
  the replicate-weight transpose). Don't build an n×R matrix when a streaming pass works.
- Zip slices that have already been length-checked, so the compiler drops the bounds checks.
  Don't reach for `get_unchecked`.

**Allocation**
- Don't allocate inside a loop over rows or replicates. Use `with_capacity`, and reuse
  scratch buffers across iterations.
- Borrow inputs (`&[T]`, `Cow`) and never copy them. Numpy arrays are read in place through
  `Array1`.

**Parallelism**
- Use `rayon` across independent outputs: replicates, domains, columns or row blocks. Each
  output's own sum stays sequential, so results don't change with the thread count.
- Parallelise only above a size where the benchmark shows it pays.
- Release the GIL (`py.detach`) around every kernel call that may run long.
- Use `faer` for linear algebra beyond tiny k. Build `X'WX` in blocks, never row by row.

**Build**
- A change to `[profile.release]` needs a benchmark showing it helps. Don't assume it does:
  `lto = true` with `codegen-units = 1` made svy-rs slower overall, and `corr`/`cov` 50% slower.
- Don't set `target-cpu=native`, because wheels must run on any CPU of their target.
- Don't set `panic = "abort"`. `guard_cb` and PyO3's `PanicException` rely on unwinding.

**Measure**
- Use criterion for kernels: `cargo bench --no-default-features` (`benches/kernel_bench.rs`).
- Use `make bench-check` end to end, on a `maturin develop --release` build. Debug timings
  are meaningless.
- A perf PR states before and after times on the same machine, comparing against the
  previous commit with `git archive`.
- A fast path gives bit-identical results to the path it replaces, and a test pins that.
- The fast version stays readable, with a one-line comment saying why it's shaped that way.

## Errors

- In svy-rs, kernel functions return `Result<T, AreaError>`, where `AreaError` is a
  `thiserror` enum for the area (`SamplingError`, `WeightingError`, ...). Don't use
  `Result<_, String>`. svy-io uses `anyhow`, with context naming the file, the variable and the
  format.
- The wrapper maps those errors: invalid input becomes `PyValueError`, and a numerical
  failure such as no convergence or a singular matrix becomes `PyRuntimeError`. The kernel
  doesn't know about `svy.errors`. Python checks user input first and raises the svy error
  itself.
- If Python has to react to a particular kernel failure, return a status or a separate
  exception type. Python must never match on message text.
- A result that is statistically undefined is `f64::NAN` with the row kept, not an error.
  Examples: a zero-weight domain, a singleton stratum, a separated coefficient. Errors are
  for input the caller should not have sent.

## Panics

- No input that Python can send may panic. A panic reaches Python as `PanicException`, and
  inside a ReadStat callback it would abort the process.
- Don't call `unwrap()` outside tests. Use `let Some(x) = .. else`, `match` or `?`. Use
  `expect("why this holds")` only when the invariant is established a few lines above.
- Index slices only after their lengths have been checked against each other. Check once at
  the boundary and return `DimensionMismatch`.

## Results are reproducible

- All randomness goes through `crate::rng::Rng`, seeded from the caller. Never seed from the
  clock or the OS.
- Output order never depends on hash-map iteration or thread scheduling. Sort group levels
  (`sorted_groups`), and reduce `rayon` results in a fixed order.
- The same seed and inputs give the same output on every platform and for any thread count.

## Frames and dtypes

- Every incoming `PyDataFrame` goes through `into_contiguous`. Chunked frames (filtered or
  concatenated) are 30x slower otherwise.
- The polars feature flags decide which dtypes the kernel can import. A missing flag panics
  inside pyo3-polars. Enabling a new dtype ships with a case in `tests/test_frame_dtypes.py`.

## unsafe (svy-io only)

- svy-rs has no `unsafe` code and keeps it that way.
- In svy-io, `unsafe` sits next to the ReadStat FFI and nowhere else. Each `unsafe` block
  has a `// SAFETY:` line, and each `unsafe fn` has a `/// # Safety` section with the
  caller's obligations.
- Every `extern "C"` callback body runs inside `guard_cb`, so no panic crosses into C.
- Never bind `.as_ptr()` of a temporary `CString` to a name, because the string is dropped at
  the end of that statement. Inline it in the call that uses it, or bind the `CString` itself
  if C keeps the pointer after the call returns.

## Tests

- Rust tests sit in `#[cfg(test)] mod tests` next to the code. Behaviour seen from Python is
  tested in `tests/*.py`.
- Expected values come from R, Stata or a hand calculation. They never come from the
  kernel's own output.
- Compare floats with `approx`. The tolerance comes from the math: about `1e-12` for closed
  forms, and the solver tolerance for iterative fits. Never loosen it to pass.
- Cover empty input, a single row, all-zero weights, a singleton stratum and ties. For every
  error path, assert the variant and its fields.

## Comments

- Use `///` for items and `//!` for modules: one line saying what the item does, not how.
- Comment only what a reader would otherwise get wrong: an invariant, a pitfall, or a match
  with R or Stata that looks wrong but isn't. Don't write history; that goes in the commit
  message.
