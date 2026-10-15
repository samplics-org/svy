from importlib.metadata import version

import svy


def test_version_matches_package_metadata():
    assert svy.__version__ == version("svy")
