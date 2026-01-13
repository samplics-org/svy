# ====== config ======
DIST_DIR := dist
PYTHON   := uv run python

# Paths
PKG_POLARS_SVY := packages/polars-svy

# ====== helpers ======
.PHONY: help
help:
	@echo "Global Targets:"
	@echo "  deps                - install dev dependencies (uv, maturin)"
	@echo "  clean               - remove build artifacts (global)"
	@echo "  test-all            - run tests for all packages"
	@echo ""
	@echo "Polars-Svy Targets:"
	@echo "  build-polars-svy    - build wheel for polars-svy (maturin)"
	@echo "  develop-polars-svy  - install polars-svy in editable mode (recompiles on import)"
	@echo "  test-polars-svy     - run tests specifically for polars-svy"
	@echo "  release-polars-svy  - build optimized release wheel"

.PHONY: clean
clean:
	rm -rf "$(DIST_DIR)" build *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "target" -type d -prune -exec rm -rf {} +
	# Clean inside packages
	rm -rf $(PKG_POLARS_SVY)/target $(PKG_POLARS_SVY)/dist

.PHONY: deps
deps:
	# Install global tools needed for rust builds
	uv tool install maturin
	uv sync
	@echo "Deps installed."

# ====== Testing ======

.PHONY: test-all
test-all: test-polars-svy
	@echo "All tests passed."

.PHONY: test-polars-svy
test-polars-svy:
	@echo "▶ Testing polars-svy..."
	# We run pytest inside the package directory so it picks up the local pyproject config
	cd $(PKG_POLARS_SVY) && uv run pytest

# ====== Polars-Svy (Rust/Maturin) ======

# Build the wheel into the local dist/ folder
.PHONY: build-polars-svy
build-polars-svy:
	@echo "▶ Building polars-svy (Maturin)..."
	cd $(PKG_POLARS_SVY) && uv run maturin build

# Build optimized release wheel
.PHONY: release-polars-svy
release-polars-svy:
	@echo "▶ Building RELEASE wheel for polars-svy..."
	cd $(PKG_POLARS_SVY) && uv run maturin develop --uv --release

# Install in editable mode (changes to Rust code take effect on next import)
.PHONY: develop-polars-svy
develop-polars-svy:
	@echo "▶ Installing polars-svy in development mode..."
	cd $(PKG_POLARS_SVY) && uv run maturin develop

# ====== Publishing ======

.PHONY: check
check:
	uv run twine check "$(PKG_POLARS_SVY)/target/wheels/"*

.PHONY: upload-polars-svy
upload-polars-svy:
	# Uploads whatever is in the target/wheels directory of the package
	uv run twine upload "$(PKG_POLARS_SVY)/target/wheels/"*
