# ---------------------------------------------------------------------------
# --no-rich flag: simulate rich not being installed
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--no-rich",
        action="store_true",
        default=False,
        help="Run tests as if rich is not installed (patches rich_available).",
    )


def pytest_configure(config):
    if config.getoption("--no-rich", default=False):
        from unittest.mock import patch

        # Patch for the duration of the entire session
        _patcher = patch("svy.ui.printing.rich_available", return_value=False)
        _patcher.start()
        config._no_rich_patcher = _patcher


def pytest_unconfigure(config):
    patcher = getattr(config, "_no_rich_patcher", None)
    if patcher is not None:
        patcher.stop()
