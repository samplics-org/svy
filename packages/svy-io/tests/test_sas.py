# tests/test_sas.py
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from svy_io import read_sas, read_sas_arrow, read_xpt, write_xpt
from svy_io.tagged_na import na_tag


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/sas"


def tpath(rel: str) -> str:
    """Return absolute path inside tests/sas/."""
    return str((DATA / rel).resolve())


# ─────────────────────────── read_sas ───────────────────────────


def test_variable_label_stored_as_attributes():
    """Variable labels should be in metadata"""
    df, meta = read_sas(tpath("hadley.sas7bdat"))
    col_meta = {v["name"]: v for v in meta["vars"]}

    # gender has no variable label
    assert col_meta["gender"]["label"] is None
    # q1 has a label
    assert col_meta["q1"]["label"] == "The instructor was well prepared"


def test_value_labels_parsed_from_bcat_file():
    """Value labels from catalog file should be parsed correctly"""
    df, meta = read_sas(
        tpath("hadley.sas7bdat"),
        catalog_path=tpath("formats.sas7bcat"),
    )

    lbl_sets = {vl["set_name"]: vl["mapping"] for vl in meta["value_labels"]}
    col_meta = {v["name"]: v for v in meta["vars"]}

    # Check gender format
    gender_set = col_meta["gender"]["label_set"]
    assert gender_set is not None
    gender_labels = lbl_sets[gender_set]
    assert gender_labels.get("f") == "Female"
    assert gender_labels.get("m") == "Male"

    # Check workshop format
    workshop_set = col_meta["workshop"]["label_set"]
    assert workshop_set is not None
    workshop_labels = lbl_sets[workshop_set]
    # Keys might be strings or integers depending on implementation
    keys = set(workshop_labels.keys())
    # Should map 1->R, 2->SAS
    assert workshop_labels.get("1", workshop_labels.get(1)) == "R"
    assert workshop_labels.get("2", workshop_labels.get(2)) == "SAS"


def test_value_labels_read_in_as_same_type_as_vector():
    """Label codes should match the type of the vector they label"""
    df, meta = read_sas(tpath("hadley.sas7bdat"), catalog_path=tpath("formats.sas7bcat"))

    lbl_sets = {vl["set_name"]: vl["mapping"] for vl in meta["value_labels"]}
    col_meta = {v["name"]: v for v in meta["vars"]}

    def codes_match_dtype(col: str) -> bool:
        """Check if label codes match column dtype"""
        dt = df.schema[col]
        label_set = col_meta[col]["label_set"]
        if not label_set:
            return True

        mapping = lbl_sets[label_set]

        # String columns should have string keys
        if dt == pl.String:
            return all(isinstance(k, str) for k in mapping.keys())

        # Numeric columns can have numeric keys or numeric-parseable strings
        def is_numeric_like(k):
            if isinstance(k, (int, float)):
                return True
            if isinstance(k, str):
                try:
                    float(k)
                    return True
                except ValueError:
                    return False
            return False

        return all(is_numeric_like(k) for k in mapping.keys())

    assert codes_match_dtype("gender")
    assert codes_match_dtype("workshop")


def test_date_times_are_converted_into_corresponding_r_types():
    """Date/time values should convert correctly from SAS epoch"""
    df, _ = read_sas(tpath("datetime.sas7bdat"))

    # SAS epoch is 1960-01-01 00:00:00 UTC
    SAS_EPOCH = datetime(1960, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # VAR1: SAS datetime (seconds since 1960-01-01)
    # Expected: 2015-02-02 14:42:12 UTC
    var1 = df["VAR1"][0]
    if isinstance(df.schema["VAR1"], pl.Datetime):
        var1_str = df["VAR1"].dt.strftime("%Y-%m-%d %H:%M:%S").to_list()[0]
        assert var1_str == "2015-02-02 14:42:12"
    else:
        # Raw numeric -> convert manually
        secs = float(var1)
        ts = SAS_EPOCH + timedelta(seconds=secs)
        assert ts.strftime("%Y-%m-%d %H:%M:%S") == "2015-02-02 14:42:12"

    # VAR2, VAR3, VAR4: SAS date (days since 1960-01-01)
    # Expected: 2015-02-02
    for col in ("VAR2", "VAR3", "VAR4"):
        val = df[col][0]
        if df.schema[col] == pl.Date:
            date_str = df[col].dt.strftime("%Y-%m-%d").to_list()[0]
            assert date_str == "2015-02-02"
        else:
            # Raw numeric days -> convert manually
            days = float(val)
            d = (SAS_EPOCH + timedelta(days=days)).date()
            assert d.strftime("%Y-%m-%d") == "2015-02-02"

    # VAR5: SAS time (seconds since midnight)
    # Expected: 14:42:12 (52932 seconds)
    var5 = df["VAR5"][0]
    if isinstance(df.schema["VAR5"], pl.Time):
        time_str = pl.Series([var5]).dt.strftime("%H:%M:%S").to_list()[0]
        assert time_str == "14:42:12"
    elif isinstance(df.schema["VAR5"], pl.Duration):
        secs = int(pl.Series([var5]).dt.total_seconds().to_list()[0])
        assert secs == 52932
    else:
        # Raw numeric seconds
        secs = int(float(var5))
        assert secs == 52932


@pytest.mark.xfail(reason="SAS tagged missings (.a/.h/.z) not surfaced by reader yet")
def test_tagged_missings_are_read_correctly():
    """Tagged missing values should be preserved with their tags"""
    df, meta = read_sas(tpath("tagged-na.sas7bdat"), catalog_path=tpath("tagged-na.sas7bcat"))

    x = df["x"].to_list()
    tags = [na_tag(v) for v in x]

    # First 5 values are not tagged (regular missing or non-missing)
    assert tags[:5] == [None, None, None, None, None]
    # Last 3 values have tags a, h, z
    assert tags[-3:] == ["a", "h", "z"]

    # Value labels should also include tagged missing codes
    lbl_sets = {vl["set_name"]: vl["mapping"] for vl in meta["value_labels"]}
    col_meta = {v["name"]: v for v in meta["vars"]}
    label_set = col_meta["x"].get("label_set")

    if label_set and label_set in lbl_sets:
        mapping = lbl_sets[label_set]
        # Should have at least 2 tagged NA labels (a and z in the example)
        tagged_count = sum(1 for k in mapping.keys() if "a" in str(k) or "z" in str(k))
        assert tagged_count >= 2


def test_connections_are_read():
    """File-like objects should be readable"""
    with open(tpath("hadley.sas7bdat"), "rb") as fh:
        df_conn, _ = read_sas(fh)
        df_path, _ = read_sas(tpath("hadley.sas7bdat"))
        assert df_conn.equals(df_path)


@pytest.mark.xfail(reason="zip input not supported yet")
def test_zip_files_are_read():
    """ZIP files should be transparently decompressed"""
    df_zip, _ = read_sas(tpath("hadley.zip"))
    df_plain, _ = read_sas(tpath("hadley.sas7bdat"))
    assert df_zip.equals(df_plain)


# ─────────────────────────── Row skipping ───────────────────────────


def test_using_skip_returns_correct_number_of_rows():
    """Row skipping should return correct number of rows"""

    def rows_after_skipping(n: int) -> int:
        df, _ = read_sas(tpath("hadley.sas7bdat"), rows_skip=n)
        return df.height

    n = rows_after_skipping(0)

    assert rows_after_skipping(1) == n - 1
    assert rows_after_skipping(n - 1) == 1
    assert rows_after_skipping(n + 0) == 0
    assert rows_after_skipping(n + 1) == 0


# ─────────────────────────── Row limiting ───────────────────────────


def test_can_limit_the_number_of_rows_to_read():
    """n_max parameter should limit rows read"""

    def rows_with_limit(n_max) -> int:
        df, _ = read_sas(tpath("hadley.sas7bdat"), n_max=n_max)
        return df.height

    n = rows_with_limit(None)  # None = unlimited

    assert rows_with_limit(0) == 0
    assert rows_with_limit(1) == 1
    assert rows_with_limit(n) == n
    assert rows_with_limit(n + 1) == n

    # Python API uses None for unlimited (not NA or -1 like R)
    assert rows_with_limit(None) == n


def test_throws_informative_error_on_bad_row_limit():
    """Invalid n_max values should raise TypeError"""
    with pytest.raises(TypeError):
        read_sas(tpath("hadley.sas7bdat"), n_max="foo")

    with pytest.raises(TypeError):
        read_sas(tpath("hadley.sas7bdat"), n_max=[1, 5])


# ─────────────────────────── Column selection ───────────────────────────


def test_can_skip_columns_with_cols_skip():
    """cols_skip parameter should exclude specified columns"""
    df_all, _ = read_sas(tpath("hadley.sas7bdat"))
    all_cols = df_all.columns

    # Skip first column
    to_skip = [all_cols[0]]
    df_skipped, _ = read_sas(tpath("hadley.sas7bdat"), cols_skip=to_skip)

    # Skipped column should not be present
    for col in to_skip:
        assert col not in df_skipped.columns

    # All other columns should be present
    assert df_skipped.columns == [c for c in all_cols if c not in to_skip]


def test_can_skip_columns_when_catalog_present():
    """Column skipping should work with catalog files"""
    df_full, _ = read_sas(tpath("hadley.sas7bdat"), catalog_path=tpath("formats.sas7bcat"))

    # Skip all but workshop
    keep = ["workshop"]
    skip = [c for c in df_full.columns if c not in keep]

    df_filtered, _ = read_sas(
        tpath("hadley.sas7bdat"), catalog_path=tpath("formats.sas7bcat"), cols_skip=skip
    )

    assert df_filtered.columns == keep


def test_throws_error_on_empty_column_selection():
    """Skipping all columns should raise an error"""
    df_full, _ = read_sas(tpath("hadley.sas7bdat"))

    # Skip all columns - should raise RuntimeError
    with pytest.raises(
        RuntimeError, match="must either specify a row count or at least one column"
    ):
        read_sas(tpath("hadley.sas7bdat"), cols_skip=df_full.columns)


@pytest.mark.skip(reason="tidyselect-style column selection not in Python API")
def test_can_select_columns_with_tidyselect_semantics():
    """Python API uses cols_skip (exclusion) not tidyselect (inclusion)"""
    # R's col_select with tidyselect is not directly translated to Python
    # Python API provides cols_skip for exclusion
    pass


# ─────────────────────────── Arrow metadata ───────────────────────────


def test_variable_label_in_arrow_metadata():
    """Variable labels should be in Arrow field metadata"""
    tbl, _ = read_sas_arrow(tpath("hadley.sas7bdat"))
    schema = tbl.schema

    # Check q1 label
    q1_idx = schema.get_field_index("q1")
    assert q1_idx != -1
    q1_field = schema[q1_idx]

    md = q1_field.metadata or {}
    assert md.get(b"label") == b"The instructor was well prepared"


def test_value_label_set_in_arrow_metadata():
    """Label set names should be in Arrow field metadata"""
    tbl, _ = read_sas_arrow(tpath("hadley.sas7bdat"))

    gender_field = tbl.schema.field(tbl.schema.get_field_index("gender"))
    md = gender_field.metadata or {}
    label_set = md.get(b"label_set")

    # Should have a label set (format name might vary: $GENDER, gender, etc.)
    assert label_set in {b"$GENDER", b"gender", b"$gender"}


# ─────────────────────────── XPT format (not implemented) ───────────────────────────


# @pytest.mark.skip(reason="read_xpt not implemented yet")
def test_xpt_can_read_date_times(tmp_path):
    """XPT: Date/time roundtrip"""
    path = tmp_path / "roundtrip.xpt"
    df = pl.DataFrame(
        {
            "date": [date.today()],
            "datetime": [datetime.now()],
        }
    )
    write_xpt(df, path)
    df2, _meta = read_xpt(path)  # <-- unpack

    assert df2.schema["date"] == pl.Date
    # datetime dtype can include a time unit; check via isinstance
    assert isinstance(df2.schema["datetime"], pl.Datetime)


@pytest.mark.skip(reason="write_xpt not implemented yet")
def test_xpt_can_roundtrip_basic_types():
    """XPT: Basic type roundtrip"""
    # Would test writing and reading back:
    # - floats, integers, booleans, strings
    pass


@pytest.mark.skip(reason="write_xpt not implemented yet")
def test_xpt_can_roundtrip_missing_values():
    """XPT: Missing value roundtrip"""
    # Would test:
    # None -> NA_integer_
    # float('nan') -> NA_real_
    # None for string -> ""
    pass


@pytest.mark.skip(reason="write_xpt not implemented yet")
def test_xpt_can_roundtrip_date_times():
    """XPT: Date/datetime with timezone handling"""
    pass


@pytest.mark.skip(reason="write_xpt not implemented yet")
def test_xpt_invalid_files_generate_errors():
    """XPT: Invalid paths should raise errors"""
    pass


@pytest.mark.skip(reason="write_xpt not implemented yet")
def test_xpt_can_roundtrip_file_labels():
    """XPT: File-level labels"""
    pass


@pytest.mark.skip(reason="write_xpt not implemented yet")
def test_xpt_can_roundtrip_format_attribute():
    """XPT: SAS format attributes"""
    pass


@pytest.mark.skip(reason="write_xpt not implemented yet")
def test_xpt_user_width_warns_when_data_wider():
    """XPT: Width attribute validation"""
    pass


# ─────────────────────────── Additional Python-specific tests ───────────────────────────


def test_catalog_path_optional():
    """Reading without catalog should work"""
    df_no_cat, meta_no_cat = read_sas(tpath("hadley.sas7bdat"))
    assert df_no_cat.height > 0
    # Should have no value labels without catalog
    assert len(meta_no_cat["value_labels"]) == 0


def test_metadata_structure():
    """Metadata should have expected structure"""
    df, meta = read_sas(tpath("hadley.sas7bdat"))

    # Check required keys
    assert "file_label" in meta
    assert "vars" in meta
    assert "value_labels" in meta
    assert "n_rows" in meta

    # Check vars structure
    assert len(meta["vars"]) > 0
    for var in meta["vars"]:
        assert "name" in var
        assert "label" in var
        assert "kind" in var


def test_empty_file_handling():
    """Empty or minimal files should be handled gracefully"""
    # This would test edge cases like:
    # - File with 0 rows
    # - File with only headers
    # - Corrupted files
    # Add tests when you have such test files
    pass
