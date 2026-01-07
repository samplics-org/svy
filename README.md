# svy

Python ecosystem for complex survey design, analysis, and reporting.

**Website:** [svylab.com](https://svylab.com) | **Docs:** [svylab.com/docs](https://svylab.com/docs/) | **LinkedIn:** [@svylab](https://www.linkedin.com/company/svylab)

> **Status:** Active development. Core packages are stable and used in production. Some features are in beta. Contributions and feedback welcome.

---

## 📦 Packages

| Package                                  | Description                                | Status    | Install               |
| ---------------------------------------- | ------------------------------------------ | --------- | --------------------- |
| **[svy](./packages/svy/)**               | Survey design, weighting, and estimation   | ✅ Stable | `pip install svy`     |
| **[svy-sae](./packages/svy-sae/)**       | Small Area Estimation (Fay-Herriot, EBLUP) | ✅ Stable | `pip install svy-sae` |
| **[svy-io](./packages/svy-io/)**         | Read/write SPSS, Stata, SAS files          | 🚧 Beta   | `pip install svy-io`  |
| **[svy-agents](./packages/svy-agents/)** | AI-assisted survey analysis                | 🚧 Beta   | _In development_      |

---

## 📚 Documentation

All documentation is published at [svylab.com/docs](https://svylab.com/docs/):

- **[svy](https://svylab.com/docs/svy/)** — Survey design, sampling, weighting, estimation
- **[svy-sae](https://svylab.com/docs/svy-sae/)** — Small area estimation methods
- **[svy-io](https://svylab.com/docs/svy-io/)** — Data import/export formats

Documentation sources (Quarto): [`docs/`](./docs/)

---

## 🌐 Web Application

**[svyLab](https://svylab.com)** — Interactive survey analysis platform

Source code: [`app/svylab/`](./app/svylab/)

---

## 🚀 Quick Start

### Install Packages

```bash
# Install core survey package
pip install svy

# Install with extras
pip install svy[sae]     # Include small area estimation
pip install svy[io]      # Include data I/O
pip install svy[all]     # Everything

# Or install packages individually
pip install svy svy-sae svy-io
```

### Basic Example

```python
import svy
import pandas as pd

# Load survey data
data = pd.read_csv("survey.csv")

# Define complex survey design
design = svy.SurveyDesign(
    data=data,
    strata="stratum",
    cluster="psu",
    weights="weight"
)

# Calculate population mean with correct standard error
mean = design.mean("income")
print(f"Mean: {mean.estimate:,.0f} (SE: {mean.se:,.0f})")
```

---

## 🛠 Development

### Prerequisites

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Quarto](https://quarto.org/) (for documentation)
- [Docker](https://www.docker.com/) (for app development)

### Setup

```bash
# Clone repository
git clone https://github.com/samplics-org/svy.git
cd svy

# Install all packages in development mode
make install-all

# Run all tests
make test-all

# Build all documentation
make docs-all
```

### Working on a Package

```bash
cd packages/svy
uv sync              # Install dependencies
pytest               # Run tests
ruff check .         # Lint code
```

### Working on Documentation

```bash
cd docs/svy
quarto render        # Build docs
make upload          # Deploy to GCS (requires access)
```

### Working on the App

```bash
cd app/svylab
make up              # Start app + database
make migrate         # Run migrations
make devdb-all       # Seed database
```

---

## 📋 Available Commands

Run `make help` to see all targets. Key commands:

```bash
# Installation
make install-all          # Install all packages
make install-svy          # Install svy only
make install-sae          # Install svy-sae only

# Testing
make test-all             # Run all tests
make test-svy             # Test svy package
make test-sae             # Test svy-sae package

# Documentation
make docs-all             # Build all docs
make upload-docs-all      # Upload all docs to GCS

# Deployment
make deploy-app           # Deploy svylab application
make deploy-all           # Deploy everything

# Development
make dev-app              # Start app in dev mode
make clean                # Clean build artifacts
```

---

## 🤝 Contributing

We welcome contributions from researchers, statisticians, data scientists, and developers!

### How to Contribute

1. **Report Issues** — Found a bug or have a feature request? [Open an issue](https://github.com/samplics-org/svy/issues)
2. **Improve Documentation** — Fix typos, add examples, clarify explanations
3. **Add Examples** — Share real-world use cases
4. **Submit Code** — Fix bugs or implement features

### Contribution Guidelines

- **Python ≥ 3.10** required
- **Write tests** for new functionality
- **Add docstrings** following NumPy style
- **Run tests** before submitting: `pytest`
- **Lint code**: `ruff check .`
- **Format code**: `ruff format .`

### Issue Labels

When filing issues, indicate which component:

```
[svy] Issue with core package
[svy-sae] Small area estimation problem
[svy-io] Data I/O issue
[docs] Documentation improvement
[app] svyLab web application
```

---

## 🧠 Design Philosophy

The **svy** ecosystem is built on these principles:

1. **Design-based inference** — Respect survey sampling theory
2. **Production-ready** — Not just academic code, but tools for real workflows
3. **Explicit over implicit** — No hidden magic, clear parameters
4. **Interoperable** — Works with pandas, NumPy, scikit-learn, etc.
5. **Scalable** — From notebooks to production pipelines
6. **Well-documented** — Every method explained with examples

---

## 📖 Repository Structure

```
svy/
├── packages/              # Python packages
│   ├── svy/              # Core survey package
│   ├── svy-sae/          # Small area estimation
│   ├── svy-io/           # Data I/O
│   └── svy-agents/       # AI agents
│
├── docs/                 # Documentation sources (Quarto)
│   ├── svy/
│   ├── svy-sae/
│   └── svy-io/
│
├── app/                  # Web application
│   └── svylab/          # FastAPI + React app
│
├── scripts/              # Utility scripts
├── .github/              # CI/CD workflows
├── Makefile              # Build automation
└── README.md             # This file
```

---

## 🧪 Testing

We maintain high test coverage across all packages:

```bash
# Run all tests
make test-all

# Run tests for specific package
cd packages/svy && pytest
cd packages/svy-sae && pytest
cd packages/svy-io && pytest

# Run with coverage
pytest --cov=svy --cov-report=html
```

Current test status:

- **svy**: 95%+ coverage
- **svy-sae**: 90%+ coverage
- **svy-io**: 97%+ coverage (32/33 tests passing)

---

## 📦 Package Dependencies

### svy (Core)

```
pandas, numpy, scipy, statsmodels
```

### svy-sae (Small Area Estimation)

```
svy, scipy, numpy
```

### svy-io (Data I/O)

```
svy, pyreadstat (Rust-based), pandas
```

### svy-agents (AI Agents)

```
svy, anthropic, openai, langchain
```

---

## 🚀 Deployment

The svyLab application is deployed to Google Cloud Run:

```bash
# Deploy application
make deploy-app

# Deploy documentation
make upload-docs-all

# Deploy everything
make deploy-all
```

**Live site:** [https://svylab.com](https://svylab.com)

---

## 📊 SEO & Discoverability

This repository is optimized for search engines:

- **Sitemap:** [svylab.com/sitemap.xml](https://svylab.com/sitemap.xml)
- **Robots.txt:** [svylab.com/robots.txt](https://svylab.com/robots.txt)
- **Schema.org markup** on all documentation pages
- **Strategic priorities** in sitemap for better indexing

---

## 🔗 Related Projects

- **[samplics](https://github.com/samplics-org/samplics)** — Original Python package for survey sampling
- **[survey (R)](https://r-survey.r-forge.r-project.org/)** — R package for complex surveys
- **[Stata survey commands](https://www.stata.com/features/survey-data/)** — Survey analysis in Stata

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.

---

## 📞 Contact

- **Email:** [info@svylab.com](mailto:info@svylab.com)
- **Issues:** [GitHub Issues](https://github.com/samplics-org/svy/issues)
- **Discussions:** [GitHub Discussions](https://github.com/samplics-org/svy/discussions)
- **LinkedIn:** [@svylab](https://www.linkedin.com/company/svylab)

---

## 🎓 Citation

If you use **svy** in your research, please cite:

```bibtex
@software{svy2025,
  title = {svy: Python Ecosystem for Complex Survey Analysis},
  author = {Diallo, Mamadou S.},
  year = {2025},
  url = {https://github.com/samplics-org/svy},
  version = {1.0.0}
}
```

---

## 🙏 Acknowledgments

The **svy** ecosystem builds on decades of survey methodology research and open-source software development. We thank the survey statistics community and all contributors.

---

**Built with ❤️ by the svyLab team**
