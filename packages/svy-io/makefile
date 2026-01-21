# Makefile for svy-io (svyreadstat)

.PHONY: help build test bench clean install dev format lint check

# Default target
.DEFAULT_GOAL := help

# Project paths
NATIVE_DIR := native/svyreadstat_rs
PYTHON_DIR := python

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)svy-io Development Makefile$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Build

build: ## Build the Rust native module
	@echo "$(BLUE)Building native module...$(NC)"
	uv run maturin build --release
	@echo "$(GREEN)✓ Build complete$(NC)"

dev: ## Build in development mode
	@echo "$(BLUE)Building in dev mode...$(NC)"
	uv run maturin build
	@echo "$(GREEN)✓ Dev build complete$(NC)"

install: build ## Build and install Python package
	@echo "$(BLUE)Installing Python package...$(NC)"
	uv pip install -e .
	@echo "$(GREEN)✓ Install complete$(NC)"

##@ Testing

test: ## Run all tests (Rust + Python)
	@echo "$(BLUE)Running Rust tests...$(NC)"
	cd $(NATIVE_DIR) && cargo test
	@echo "$(BLUE)Running Python tests...$(NC)"
	uv run pytest -v tests/
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-rust: ## Run only Rust tests
	@echo "$(BLUE)Running Rust tests...$(NC)"
	cd $(NATIVE_DIR) && cargo test
	@echo "$(GREEN)✓ Rust tests passed$(NC)"

test-python: ## Run only Python tests
	@echo "$(BLUE)Running Python tests...$(NC)"
	uv run pytest -v tests/
	@echo "$(GREEN)✓ Python tests passed$(NC)"

test-quick: ## Run Python tests with fast fail
	@echo "$(BLUE)Running quick tests...$(NC)"
	uv run pytest -x -v tests/
	@echo "$(GREEN)✓ Quick tests done$(NC)"

test-verbose: ## Run Python tests with verbose output
	@echo "$(BLUE)Running verbose tests...$(NC)"
	uv run pytest -x -v -s tests/
	@echo "$(GREEN)✓ Verbose tests done$(NC)"

##@ Benchmarking

bench: ## Run all Python benchmarks
	@echo "$(BLUE)Running Python benchmarks...$(NC)"
	uv run pytest tests/benchmark_test.py --benchmark-only -v
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

bench-save: ## Run benchmarks and save as baseline
	@echo "$(BLUE)Running benchmarks and saving baseline...$(NC)"
	uv run pytest tests/benchmark_test.py --benchmark-only --benchmark-save=main
	@echo "$(GREEN)✓ Baseline 'main' saved$(NC)"

bench-compare: ## Run benchmarks and compare to baseline
	@echo "$(BLUE)Running benchmarks and comparing to baseline 'main'...$(NC)"
	uv run pytest tests/benchmark_test.py --benchmark-only --benchmark-compare=main
	@echo "$(GREEN)✓ Comparison complete$(NC)"

bench-stata: ## Run only Stata parsing benchmarks
	@echo "$(BLUE)Running Stata benchmarks...$(NC)"
	uv run pytest tests/benchmark_test.py::test_bench_stata -k stata --benchmark-only
	@echo "$(GREEN)✓ Stata benchmarks complete$(NC)"

bench-spss: ## Run only SPSS parsing benchmarks
	@echo "$(BLUE)Running SPSS benchmarks...$(NC)"
	uv run pytest tests/benchmark_test.py::test_bench_spss -k spss --benchmark-only
	@echo "$(GREEN)✓ SPSS benchmarks complete$(NC)"

bench-sas: ## Run only SAS parsing benchmarks
	@echo "$(BLUE)Running SAS benchmarks...$(NC)"
	uv run pytest tests/benchmark_test.py::test_bench_sas -k sas --benchmark-only
	@echo "$(GREEN)✓ SAS benchmarks complete$(NC)"

bench-report: ## Show benchmark statistics
	@echo "$(BLUE)Benchmark statistics:$(NC)"
	uv run pytest tests/benchmark_test.py --benchmark-only --benchmark-columns=mean,stddev,median,iqr,outliers
	@echo "$(GREEN)✓ Statistics complete$(NC)"

bench-hist: ## Generate benchmark histogram
	@echo "$(BLUE)Generating benchmark histogram...$(NC)"
	uv run pytest tests/benchmark_test.py --benchmark-only --benchmark-histogram=.benchmarks/histogram
	@echo "$(GREEN)✓ Histogram saved to .benchmarks/histogram.svg$(NC)"
	@echo "$(YELLOW)Open with: open .benchmarks/histogram.svg$(NC)"

##@ Code Quality

format: ## Format Rust and Python code
	@echo "$(BLUE)Formatting Rust code...$(NC)"
	cd $(NATIVE_DIR) && cargo fmt
	@echo "$(BLUE)Formatting Python code...$(NC)"
	uv run ruff format python/ tests/
	@echo "$(GREEN)✓ Formatting complete$(NC)"

format-check: ## Check code formatting without changes
	@echo "$(BLUE)Checking Rust formatting...$(NC)"
	cd $(NATIVE_DIR) && cargo fmt -- --check
	@echo "$(BLUE)Checking Python formatting...$(NC)"
	uv run ruff format --check python/ tests/
	@echo "$(GREEN)✓ Format check complete$(NC)"

lint: ## Run linters (clippy for Rust, ruff for Python)
	@echo "$(BLUE)Running Rust linter (clippy)...$(NC)"
	cd $(NATIVE_DIR) && cargo clippy -- -D warnings
	@echo "$(BLUE)Running Python linter (ruff)...$(NC)"
	uv run ruff check python/ tests/
	@echo "$(GREEN)✓ Linting complete$(NC)"

lint-fix: ## Fix linting issues automatically
	@echo "$(BLUE)Fixing Rust linting issues...$(NC)"
	cd $(NATIVE_DIR) && cargo clippy --fix --allow-dirty --allow-staged
	@echo "$(BLUE)Fixing Python linting issues...$(NC)"
	uv run ruff check --fix python/ tests/
	@echo "$(GREEN)✓ Auto-fixes applied$(NC)"

check: ## Run all checks (format, lint, test)
	@echo "$(BLUE)Running all checks...$(NC)"
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) test
	@echo "$(GREEN)✓ All checks passed$(NC)"

##@ Performance

profile: ## Run benchmarks with profiling
	@echo "$(BLUE)Running benchmarks with profiling...$(NC)"
	cd $(NATIVE_DIR) && cargo bench --profile-time 10
	@echo "$(GREEN)✓ Profiling complete$(NC)"

flamegraph: ## Generate flamegraph (requires cargo-flamegraph)
	@echo "$(BLUE)Generating flamegraph...$(NC)"
	@command -v cargo-flamegraph >/dev/null 2>&1 || { \
		echo "$(YELLOW)Installing cargo-flamegraph...$(NC)"; \
		cargo install flamegraph; \
	}
	cd $(NATIVE_DIR) && cargo flamegraph --bench parse_benchmarks
	@echo "$(GREEN)✓ Flamegraph generated: $(NATIVE_DIR)/flamegraph.svg$(NC)"

##@ Maintenance

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	cd $(NATIVE_DIR) && cargo clean
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	@echo "$(GREEN)✓ Clean complete$(NC)"

clean-bench: ## Clean benchmark results
	@echo "$(BLUE)Cleaning benchmark results...$(NC)"
	rm -rf $(NATIVE_DIR)/target/criterion
	@echo "$(GREEN)✓ Benchmark results cleaned$(NC)"

update-deps: ## Update Rust dependencies
	@echo "$(BLUE)Updating Rust dependencies...$(NC)"
	cd $(NATIVE_DIR) && cargo update
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

##@ Documentation

docs: ## Build Rust documentation
	@echo "$(BLUE)Building Rust docs...$(NC)"
	cd $(NATIVE_DIR) && cargo doc --no-deps --open
	@echo "$(GREEN)✓ Documentation built$(NC)"

docs-python: ## Build Python documentation
	@echo "$(BLUE)Building Python docs...$(NC)"
	uv run pdoc python/svy_io -o docs/
	@echo "$(GREEN)✓ Python documentation built at docs/$(NC)"

##@ CI/CD

ci: ## Run CI checks locally
	@echo "$(BLUE)Running CI checks...$(NC)"
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) build
	@$(MAKE) test
	@$(MAKE) bench-save
	@echo "$(GREEN)✓ All CI checks passed$(NC)"

##@ Development Workflows

quick: format test-quick ## Quick development cycle (format + fast tests)
	@echo "$(GREEN)✓ Quick check complete$(NC)"

full: clean format lint test bench ## Full check (clean + format + lint + test + bench)
	@echo "$(GREEN)✓ Full check complete$(NC)"

watch-test: ## Watch for changes and run tests
	@echo "$(BLUE)Watching for changes...$(NC)"
	uv run pytest-watch tests/

watch-rust: ## Watch Rust code and rebuild
	@echo "$(BLUE)Watching Rust code...$(NC)"
	@command -v cargo-watch >/dev/null 2>&1 || { \
		echo "$(YELLOW)Installing cargo-watch...$(NC)"; \
		cargo install cargo-watch; \
	}
	cd $(NATIVE_DIR) && cargo watch -x build

##@ Release

release: ## Build release version
	@echo "$(BLUE)Building release...$(NC)"
	cd $(NATIVE_DIR) && cargo build --release
	uv pip install -e .
	@echo "$(GREEN)✓ Release build complete$(NC)"

version: ## Show version information
	@echo "$(BLUE)Version Information:$(NC)"
	@cd $(NATIVE_DIR) && cargo --version
	@uv --version
	@python --version
	@echo ""
	@echo "$(BLUE)Package Version:$(NC)"
	@grep '^version' pyproject.toml | head -1
