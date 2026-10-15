from importlib.metadata import version

import svy_rs


def test_version_matches_package_metadata():
    assert svy_rs.__version__ == version("svy-rs")
