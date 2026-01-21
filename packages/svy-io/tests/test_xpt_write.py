# tests/test_xpt_write.py


import polars as pl
import pytest

from svy_io import write_xpt


def test_write_xpt_basic(tmp_path):
    """Test basic XPT writing"""

    df = pl.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "score": [95.5, 87.3, 91.2]}
    )

    path = tmp_path / "test.xpt"
    write_xpt(df, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_write_xpt_with_label(tmp_path):
    """Test XPT with dataset label"""
    df = pl.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "labeled.xpt"

    write_xpt(df, path, label="Test Dataset")
    assert path.exists()


def test_write_xpt_version_validation(tmp_path):
    """Test that invalid version raises error"""
    df = pl.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "bad.xpt"

    with pytest.raises(ValueError, match="version must be 5 or 8"):
        write_xpt(df, path, version=7)


def test_write_xpt_name_too_long_v5(tmp_path):
    """Test that long names raise error for version 5"""
    df = pl.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "data.xpt"

    with pytest.raises(ValueError, match="name must be <= 8 characters"):
        write_xpt(df, path, version=5, name="VERYLONGNAME")


def test_write_xpt_label_too_long(tmp_path):
    """Test that long labels raise error"""
    df = pl.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "data.xpt"

    long_label = "x" * 41
    with pytest.raises(ValueError, match="label must be <= 40 characters"):
        write_xpt(df, path, label=long_label)
