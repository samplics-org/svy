# ====== config ======
DIST_DIR   := dist
PYTHON     := uv run python

# Paths
PKG_SVY    := packages/svy
PKG_SVY_IO := packages/svy-io
PKG_SVY_RS := packages/svy-rs

# ====== helpers ======
.PHONY: help
help:
	@echo "Global Targets:"
	@echo "  deps                - install all workspace deps (uv sync)"
	@echo "  clean               - remove build artifacts"
	@echo "  lint                - ruff check across all packages"
	@echo "  lint-fix            - auto-fix lint issues"
	@echo "  fmt                 - check formatting (ruff format --check)"
	@echo "  fmt-fix             - auto-format all code"
	@echo "  test-all            - run tests for all packages"
	@echo "  ci                  - full CI pipeline (lint + fmt + test)"
	@echo ""
	@echo "svy Targets:"
	@echo "  build-svy           - build sdist + wheel for svy"
	@echo "  test-svy            - run tests for svy"
	@echo "  lint-svy            - lint svy"
	@echo "  bench-check         - check for perf regressions vs the baseline (needs release build)"
	@echo "  bench-record        - re-record the perf baseline (commit the result)"
	@echo ""
	@echo "Release Targets (PKG=svy|svy-io|svy-rs):"
	@echo "  release-notes       - preview the notes CI will publish for the current version"
	@echo "  release-check       - pre-flight a release BEFORE tagging (safe, read-only)"
	@echo "  release-tag         - run release-check, then create and push the tag"
	@echo ""
	@echo "svy-io Targets (local dev only — published separately):"
	@echo "  build-svy-io        - build svy-io native extension locally"
	@echo "  test-svy-io         - run tests for svy-io"
	@echo "  lint-svy-io         - lint svy-io"
	@echo ""
	@echo "svy-rs Targets (local dev only — published separately):"
	@echo "  develop-svy-rs      - install svy-rs in editable mode (maturin develop)"
	@echo "  test-svy-rs         - run tests for svy-rs"
	@echo "  lint-svy-rs         - lint svy-rs python bindings"

# ====== Dependencies ======
.PHONY: deps
deps:
	uv sync --all-packages
	@echo "Deps installed."

# ====== Linting ======
.PHONY: lint lint-fix fmt fmt-fix lint-svy lint-svy-io lint-svy-rs

lint: lint-svy lint-svy-io lint-svy-rs
	@echo "All lint checks passed."

lint-svy:
	@echo "▶ Linting svy..."
	uv run ruff check $(PKG_SVY)/src $(PKG_SVY)/tests

lint-svy-io:
	@echo "▶ Linting svy-io..."
	uv run ruff check $(PKG_SVY_IO)/python $(PKG_SVY_IO)/tests

lint-svy-rs:
	@echo "▶ Linting svy-rs..."
	uv run ruff check $(PKG_SVY_RS)/tests

lint-fix:
	uv run ruff check --fix $(PKG_SVY)/src $(PKG_SVY)/tests
	uv run ruff check --fix $(PKG_SVY_IO)/python $(PKG_SVY_IO)/tests
	uv run ruff check --fix $(PKG_SVY_RS)/tests

fmt:
	@echo "▶ Checking formatting..."
	uv run ruff format --check $(PKG_SVY)/src $(PKG_SVY)/tests
	uv run ruff format --check $(PKG_SVY_IO)/python $(PKG_SVY_IO)/tests
	uv run ruff format --check $(PKG_SVY_RS)/tests

fmt-fix:
	uv run ruff format $(PKG_SVY)/src $(PKG_SVY)/tests
	uv run ruff format $(PKG_SVY_IO)/python $(PKG_SVY_IO)/tests
	uv run ruff format $(PKG_SVY_RS)/tests

# ====== Testing ======
.PHONY: test-all test-svy test-svy-io test-svy-rs

test-all: test-svy test-svy-io test-svy-rs
	@echo "All tests passed."

test-svy:
	@echo "▶ Testing svy..."
	cd $(PKG_SVY) && uv run pytest

test-svy-io:
	@echo "▶ Testing svy-io..."
	cd $(PKG_SVY_IO) && uv run pytest

test-svy-rs:
	@echo "▶ Testing svy-rs..."
	cd $(PKG_SVY_RS) && uv run pytest

# ====== Benchmarks & native Rust tests ======
# The svy-rs crate builds the Python extension by default (pyo3/extension-module).
# Host-native `cargo test`/`cargo bench` must drop that feature (--no-default-
# features) so the binary links normally, and point pyo3 at the venv interpreter.
.PHONY: test-svy-rs-cargo bench-svy-rs bench-svy bench
PYO3_PYTHON := $(CURDIR)/.venv/bin/python

test-svy-rs-cargo:
	@echo "▶ Rust unit tests (svy-rs, host build)..."
	cd $(PKG_SVY_RS) && PYO3_PYTHON=$(PYO3_PYTHON) cargo test --lib --no-default-features

bench-svy-rs:
	@echo "▶ Criterion kernel benches (svy-rs)..."
	cd $(PKG_SVY_RS) && PYO3_PYTHON=$(PYO3_PYTHON) cargo bench --no-default-features

bench-svy:
	@echo "▶ Python end-to-end + direct-kernel benches..."
	cd $(PKG_SVY) && uv run python benchmarks/bench_kernel.py

bench: bench-svy-rs bench-svy
	@echo "All benchmarks complete."

# ------ Performance regression gate (LOCAL only, not CI) ------
# Absolute timings only mean something on the machine that recorded them, so
# this is deliberately not a CI check: hosted runners are too noisy to gate on.
# Run before a release, or after touching a hot path. Requires a RELEASE build
# (`maturin develop --release`) -- a debug build is detected and reported as
# such rather than as a wall of regressions.
.PHONY: bench-check bench-record

bench-check:
	@echo "▶ Checking for performance regressions vs the recorded baseline..."
	cd $(PKG_SVY) && uv run python benchmarks/check_regression.py

bench-record:
	@echo "▶ Recording a new performance baseline (deliberate: commit the result)..."
	cd $(PKG_SVY) && uv run python benchmarks/check_regression.py --record

# ====== svy (pure Python — the one we publish from this repo) ======
.PHONY: build-svy
build-svy:
	@echo "▶ Building svy..."
	cd $(PKG_SVY) && uv build

# ====== svy-io (local dev convenience) ======

# Build optimized release wheel
.PHONY: release-svy-io
release-svy-io:
	@echo "▶ Building RELEASE wheel for svy-rs..."
	cd $(PKG_SVY_IO) && uv run maturin develop --uv --release

.PHONY: build-svy-io
build-svy-io:
	@echo "▶ Building svy-io locally..."
	cd $(PKG_SVY_IO) && uv build

# ====== svy-rs (local dev convenience) ======

# Build the wheel into the local dist/ folder
.PHONY: build-svy-rs
build-svy-rs:
	@echo "▶ Building svy-rs (Maturin)..."
	cd $(PKG_SVY_RS) && uv run maturin build

# Build optimized release wheel
.PHONY: release-svy-rs
release-svy-rs:
	@echo "▶ Building RELEASE wheel for svy-rs..."
	cd $(PKG_SVY_RS) && uv run maturin develop --uv --release

.PHONY: develop-svy-rs
develop-svy-rs:
	@echo "▶ Installing svy-rs in dev mode (maturin develop)..."
	cd $(PKG_SVY_RS) && uv run maturin develop

# ====== CI aggregate ======
.PHONY: ci
ci: lint fmt test-all
	@echo "CI passed."

# ====== Clean ======
.PHONY: clean
clean:
	rm -rf "$(DIST_DIR)" build *.egg-info .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -prune -exec rm -rf {} +
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} +
	rm -rf $(PKG_SVY)/dist
	rm -rf $(PKG_SVY_IO)/dist
	rm -rf $(PKG_SVY_RS)/target $(PKG_SVY_RS)/dist


# ====== Release ======
# Pushing a <pkg>-v* tag publishes to PyPI, and PyPI never lets a version number
# be reused. CI checks the tag against the committed version, but it can only do
# so once the tag exists — by which point the mistake is already made. These run
# the same check locally, before the tag is created, which is the last point
# where a wrong version costs nothing.
.PHONY: release-notes release-check release-tag
PKG     ?= svy
VERSION  = $(shell sed -n 's/^version = "\(.*\)"/\1/p' packages/$(PKG)/pyproject.toml | head -1)
TAG      = $(PKG)-v$(VERSION)

release-notes:
	@echo "▶ Notes CI would publish for $(PKG) $(VERSION):"
	@$(PYTHON) .github/scripts/release_notes.py --package $(PKG) --tag $(TAG) --out /dev/stdout

release-check:
	@echo "▶ Pre-flight for $(TAG)"
	@test -z "$$(git status --porcelain)" || { echo "  ✗ working tree is dirty"; exit 1; }
	@echo "  ✓ working tree clean"
	@test "$$(git rev-parse --abbrev-ref HEAD)" = "main" || { echo "  ✗ not on main"; exit 1; }
	@echo "  ✓ on main"
	@git fetch -q origin main && test -z "$$(git rev-list HEAD..origin/main)" || \
		{ echo "  ✗ behind origin/main — pull first"; exit 1; }
	@echo "  ✓ up to date with origin/main"
	@git rev-parse -q --verify "refs/tags/$(TAG)" >/dev/null && \
		{ echo "  ✗ tag $(TAG) already exists locally"; exit 1; } || true
	@test -z "$$(git ls-remote --tags origin '$(TAG)')" || \
		{ echo "  ✗ tag $(TAG) already exists on origin"; exit 1; }
	@echo "  ✓ tag $(TAG) is free"
	@$(PYTHON) .github/scripts/release_notes.py --package $(PKG) --tag $(TAG) --out /dev/null
	@echo "  ✓ version and CHANGELOG agree"
	@echo "▶ Ready. 'make release-tag PKG=$(PKG)' will publish $(PKG) $(VERSION) to PyPI."

release-tag: release-check
	@echo "▶ Tagging $(TAG) — this publishes to PyPI and cannot be undone."
	git tag -a $(TAG) -m "$(PKG) $(VERSION)"
	git push origin $(TAG)
	@echo "▶ Pushed. Watch: gh run list --workflow=$(PKG)-wheels.yml --limit 1"
