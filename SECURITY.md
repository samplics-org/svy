# Security

This page covers the three packages built from this repository: `svy`, `svy-rs` and `svy-io`.

## Reporting a vulnerability

Report it privately through GitHub: the repository's **Security** tab, then **Report a vulnerability** ([direct link](https://github.com/samplics-org/svy/security/advisories/new)). Please do not open a public issue. Expect an acknowledgement within five working days; a fix is released as a patch and credited in the changelog unless you ask otherwise.

Fixes go into the latest release of each package.

## Data handling

svy runs entirely on your machine. Your survey data is never sent anywhere.

- **No telemetry.** svy collects no usage data, crash reports or analytics.
- **No network access in a base install.** `pip install svy` installs no HTTP client, and the analysis, weighting and estimation code opens no network connections. The readers open what you give them: `read_csv` and `read_parquet` hand the path to polars, which also reads a URL or cloud path if you pass one.
- **One optional exception: example datasets.** The `remote` extra (`pip install "svy[remote]"`) lets `svy.datasets` download the full example datasets from the svyLab catalog at `svylab.com`. It connects only when `svy.datasets` is called, and requests only the public catalog and dataset files. Under the default `source="auto"`, a dataset already in the local cache (`~/.svy/datasets`, or `SVYLAB_CACHE_DIR`) is read without connecting, and `SVYLAB_OFFLINE=1` stops it connecting at all; only an explicit `source="remote"` or `force_download=True` always connects.
- **No files written** except the files you ask svy to write, and that dataset cache.

## Supply chain

- **Runtime dependencies.** `svy`: numpy, scipy, polars (with pyarrow), msgspec, `svy-rs` and `svy-io`. `svy-rs` and `svy-io` depend on polars only. Optional: rich (`report`), httpx (`remote`).
- **Compiled code.** `svy-rs` is Rust; `svy-io` is Rust around the [ReadStat](https://github.com/WizardMac/ReadStat) C library, whose source is vendored in this repository. Both are compiled into their wheels. Every Rust dependency is pinned in the `Cargo.lock` files and the Python dependencies in `uv.lock`, all in this repository.
- **Builds.** Wheels are built from tagged commits by the GitHub Actions workflows in [`.github/workflows`](.github/workflows), with every action pinned to a commit hash.
- **Publishing.** Releases are uploaded to PyPI through trusted publishing (OIDC), with no stored API tokens. `svy` and `svy-rs` releases carry PyPI [attestations](https://docs.pypi.org/attestations/) linking each file to the workflow run that built it.
- **Updates.** Dependabot watches the Python, Rust and GitHub Actions dependencies and opens pull requests for security updates as they are published.
- **Integrity of example datasets.** Downloads must use https, including after redirects, and are checked against a SHA-256 hash before use.

## Platforms

Prebuilt wheels, so no compiler or Rust toolchain is needed:

| Platform | Architectures      | Minimum                                             |
| -------- | ------------------ | --------------------------------------------------- |
| Linux    | x86_64, aarch64    | glibc 2.28 (RHEL/Rocky/Alma 8, Debian 10, Ubuntu 20.04) |
| macOS    | arm64, x86_64      | macOS 11 (arm64), 10.12 (x86_64)                    |
| Windows  | x86_64             |                                                     |

Python 3.11 or later; one wheel per platform covers every supported Python version. On other platforms (for example older Linux such as RHEL 7, or musl-based Alpine), pip builds from source, which needs Rust 1.91 or later and a C compiler. The test suite runs on Linux for Python 3.11 to 3.14.
