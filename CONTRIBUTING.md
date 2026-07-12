# Contributing to svy

Thanks for your interest in improving **svy**! This guide keeps it short and practical.

> tl;dr — Fork → feature branch → small PR with tests/docs → CI green → review → merge.

---

## Ground rules

- Be kind and constructive. We follow the spirit of the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct.
- No breaking changes without discussion (open an issue first).
- Keep PRs small and focused (under ~400 lines of diff when possible; split otherwise).

---

## Monorepo layout

This is a **uv workspace** — one repository, three packages, one shared virtual environment at the root.

| Package | What it is |
|---|---|
| [`packages/svy`](packages/svy) | End-to-end surveys: design, sample, analyze, report (pure Python) |
| [`packages/svy-io`](packages/svy-io) | SAS/SPSS/Stata I/O with Polars, powered by ReadStat + PyO3 (Rust native extension) |
| [`packages/svy-rs`](packages/svy-rs) | Internal Rust extension for `svy` — do not depend on it directly |

---

## Dev setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages Python and the workspace; Python version is pinned in `.python-version`)
- Rust toolchain (stable, via `rustup`) — only needed for `svy-io` / `svy-rs` native builds
- C build tools for `svy-io`: Xcode CLT (macOS), MSVC Build Tools (Windows), `build-essential` (Linux)

### Setup

```bash
git clone https://github.com/samplics-org/svy.git
cd svy
make deps          # uv sync — installs all workspace deps into ./.venv
```

**One venv, at the root.** The workspace resolves every package into the root `./.venv`.
Do **not** create per-package virtual environments or run `uv venv` / `uv lock` inside
`packages/*` — `uv run` from any package directory automatically uses the root environment.

### Native builds

```bash
make build-svy-io           # build the svy-io native extension
# or, inside packages/svy-io:
make dev                    # debug build (fast iteration)
make build                  # release build
```

> Tip: install `sccache` and set `RUSTC_WRAPPER=sccache` to speed up Rust rebuilds.

---

## Test, lint, format

Run from the repo root:

```bash
make test-all      # tests for all packages
make lint          # ruff check across all packages
make fmt           # ruff format --check
make ci            # full local pipeline: lint + fmt + test

# Or directly:
uv run pytest                       # whole workspace
uv run pytest packages/svy-io/tests # one package

# Rust (if you touch native code)
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Lint/format/type-check settings (ruff, mypy, pytest) are configured once in the root
`pyproject.toml` and apply to all packages. Keep PRs green in CI by running these locally.

---

## Making changes

1. **Branch**: `git checkout -b feat/spss-user-missing-range`
2. **Commit style**: [Conventional Commits](https://www.conventionalcommits.org/), scoped by package where useful:
   - `feat(svy): add poststratification to ReplicateDesign`
   - `fix(svy-io): guard long string error path in Stata reader`
   - `docs: clarify read_sav user_na behavior`
3. **Tests**: add or adjust tests next to the code you change (`packages/<pkg>/tests/`).
   Prefer tiny generated fixtures over large binary files; add round-trip tests when
   changing I/O behavior; cover edge cases (encodings, missing values, temporals).
4. **Docs**: update README examples and docstrings if public behavior changes.
5. **PR**: explain the "why", link issues, note trade-offs, keep CI green.

### svy-io API guidance

- **Naming**: `read_*` / `write_*` by format. Readers return `(polars.DataFrame, metadata_dict)`; do **not** return Pandas.
- **New options are keyword-only**; reuse common names across formats:
  `cols_skip`, `n_max`, `rows_skip`, `coerce_temporals`, `zap_empty_str`, `factorize`, `levels`, `ordered`.
- Keep metadata **JSON-serializable** and documented in docstrings.
- **Errors**: raise clear `ValueError` / `RuntimeError` with actionable, format-specific context.

### Native (Rust/PyO3) contributions

- Keep the FFI boundary small and ergonomic in Python.
- Prefer safe Rust; clearly justify any `unsafe` and guard it with tests.
- Map ReadStat errors to Python exceptions with context (file, variable, format).
- `rustfmt` and `clippy -D warnings` must pass.

---

## Vendored ReadStat (git subtree)

`packages/svy-io/ReadStat` is upstream [WizardMac/ReadStat](https://github.com/WizardMac/ReadStat)
vendored as a **git subtree**, currently pinned to **v1.1.9**. It is *not* a git submodule:
maturin bundles `ReadStat/src/**` into every wheel and sdist, so the sources must live in
the repository itself — and a plain `git clone` is always complete (no `--recursive`).

**Rules:**

- **Never hand-edit, reformat, or format-on-save files under `ReadStat/`** (add the
  directory to your editor's format exclusions). Local edits turn every future update
  into a conflict-ridden merge. Patch upstream instead, or work around it in
  `packages/svy-io/native/readstat-sys`.
- **Never run ReadStat's own autotools build in-tree** (`autogen.sh`, `configure`, `make`).
  The Rust build script compiles the C sources directly with the `cc` crate; autotools
  artifacts just pollute the tree.

**Updating to a new upstream release** (from `packages/svy-io`, with a clean working tree):

```bash
make update-readstat REF=v1.1.10   # any upstream tag or commit SHA
make dev && make test              # rebuild and verify before pushing
```

This wraps `git subtree pull --squash` and produces two commits: a squashed snapshot of
the upstream tree and a merge recording the exact ref. Review the diff for API changes
that affect `native/readstat-sys/wrapper.h` before merging.

---

## Versioning & releases

- **SemVer** per package; breaking changes require a major bump.
- Add entries to the affected package's `CHANGELOG.md` (Keep a Changelog style):
  `packages/svy/CHANGELOG.md`, `packages/svy-io/CHANGELOG.md`, or
  `packages/svy-rs/CHANGELOG.md`.
- Packages are published separately; tagging a release triggers CI wheel builds
  (manylinux/macOS/Windows for native packages).

---

## Reporting issues & security

- Bugs/requests: open a GitHub issue with a minimal repro (small sample file if possible).
- Security concerns: do **not** open a public issue — use GitHub Security Advisories or
  contact the maintainers privately.

---

## License & ownership

By contributing, you agree your contributions are licensed under the project's license
(see `LICENSE` files at the root and in each package).

Thank you! 🙌
