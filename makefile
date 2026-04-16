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

# ====== svy (pure Python — the one we publish from this repo) ======
.PHONY: build-svy
build-svy:
	@echo "▶ Building svy..."
	cd $(PKG_SVY) && uv build

# ====== svy-io (local dev convenience) ======
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
