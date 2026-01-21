# ====== config ======
DIST_DIR := dist
PYTHON   := uv run python

# Paths
PKG_SVY_RS := packages/svy-rs

# ====== helpers ======
.PHONY: help
help:
	@echo "Global Targets:"
	@echo "  deps                - install dev dependencies (uv, maturin)"
	@echo "  clean               - remove build artifacts (global)"
	@echo "  test-all            - run tests for all packages"
	@echo ""
	@echo "svy-rs Targets:"
	@echo "  build-svy-rs    - build wheel for svy-rs (maturin)"
	@echo "  develop-svy-rs  - install svy-rs in editable mode (recompiles on import)"
	@echo "  test-svy-rs     - run tests specifically for svy-rs"
	@echo "  release-svy-rs  - build optimized release wheel"

.PHONY: clean
clean:
	rm -rf "$(DIST_DIR)" build *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "target" -type d -prune -exec rm -rf {} +
	# Clean inside packages
	rm -rf $(PKG_SVY_RS)/target $(PKG_SVY_RS)/dist

.PHONY: deps
deps:
	# Install global tools needed for rust builds
	uv tool install maturin
	uv sync
	@echo "Deps installed."

# ====== Testing ======

.PHONY: test-all
test-all: test-svy-rs
	@echo "All tests passed."

.PHONY: test-svy-rs
test-svy-rs:
	@echo "▶ Testing svy-rs..."
	# We run pytest inside the package directory so it picks up the local pyproject config
	cd $(PKG_SVY_RS) && uv run pytest

# ====== svy-rs (Rust/Maturin) ======

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

# Install in editable mode (changes to Rust code take effect on next import)
.PHONY: develop-svy-rs
develop-svy-rs:
	@echo "▶ Installing svy-rs in development mode..."
	cd $(PKG_SVY_RS) && uv run maturin develop

# ====== Publishing ======

.PHONY: check
check:
	uv run twine check "$(PKG_SVY_RS)/target/wheels/"*

.PHONY: upload-svy-rs
upload-svy-rs:
	# Uploads whatever is in the target/wheels directory of the package
	uv run twine upload "$(PKG_SVY_RS)/target/wheels/"*
