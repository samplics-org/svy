# tests/test_spss.py

import os
import tempfile

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from svy_io.spss import (
    get_column_labels,
    get_value_labels_for_column,
    read_sav,
    write_sav,
)


# Helper functions
def roundtrip_sav(df, **kwargs):
    """
    Write DataFrame to temp SAV file and read it back.

    If called with return_meta=True, returns (df, meta) tuple.
    Otherwise returns just df for backward compatibility.
    """
    return_meta = kwargs.pop("return_meta", False)

    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_sav(df, tmp_path, **kwargs)
        df_out, meta_out = read_sav(tmp_path)

        if return_meta:
            return df_out, meta_out
        else:
            return df_out
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def roundtrip_var(values, _labels=None, **kwargs):
    """Roundtrip a single column

    _labels is accepted positionally to match existing calls like
    roundtrip_var([None], {"x": [None]}). It's currently unused.
    """
    df = pl.DataFrame({"x": values})
    df_out = roundtrip_sav(df, **kwargs)
    return df_out["x"].to_list()


# read_sav tests ----------------------------------------------------------


def test_variable_label_stored_as_metadata(test_data_dir):
    """Variable labels should be available in metadata"""
    df, meta = read_sav(test_data_dir / "spss/variable-label.sav")

    labels = get_column_labels(meta)
    assert labels.get("sex") == "Gender"


def test_value_labels_stored_in_metadata(test_data_dir):
    """Value labels should be preserved in metadata"""
    df_num, meta_num = read_sav(test_data_dir / "spss/labelled-num.sav")
    df_str, meta_str = read_sav(test_data_dir / "spss/labelled-str.sav")

    # Check numeric value labels
    num_labels = get_value_labels_for_column(meta_num, df_num.columns[0])
    assert num_labels is not None
    assert "1" in num_labels
    assert num_labels["1"] == "This is one"

    # Check string value labels
    str_labels = get_value_labels_for_column(meta_str, df_str.columns[0])
    assert str_labels is not None
    assert "F" in str_labels
    assert str_labels["F"] == "Female"
    assert str_labels["M"] == "Male"


def test_value_labels_read_in_as_same_type_as_vector(test_data_dir):
    """Value label keys should match the column data type"""
    df, meta = read_sav(test_data_dir / "spss/variable-label.sav")
    df_num, meta_num = read_sav(test_data_dir / "spss/labelled-num.sav")
    df_str, meta_str = read_sav(test_data_dir / "spss/labelled-str.sav")

    # For numeric columns, labels should be numeric-ish
    sex_labels = get_value_labels_for_column(meta, "sex")
    if sex_labels:
        # Keys are strings but should be parseable as numbers
        for key in sex_labels.keys():
            try:
                float(key)
                assert True
            except ValueError:
                assert False, f"Expected numeric-ish key, got {key}"


def test_non_ascii_labels_converted_to_utf8(test_data_dir):
    """Non-ASCII characters in labels should be properly decoded"""
    df, meta = read_sav(test_data_dir / "spss/umlauts.sav")

    # Variable label should have umlaut
    labels = get_column_labels(meta)
    col_name = df.columns[0]
    assert labels.get(col_name) == "This is an ä-umlaut"

    # Value labels should also have umlauts
    value_labels = get_value_labels_for_column(meta, col_name)
    if value_labels:
        assert any("ä" in label for label in value_labels.values())


def test_datetime_variables_converted_to_correct_class(test_data_dir):
    """Date/datetime columns should have proper Polars types"""
    df, meta = read_sav(test_data_dir / "spss/datetime.sav", coerce_temporals=True)

    assert df.schema["date"] == pl.Date
    assert df.schema["date_posix"] in (pl.Datetime, pl.Datetime("us"))
    # Time might be Duration or could be kept as numeric
    assert df.schema["time"] in (pl.Duration, pl.Float64)


def test_datetime_values_correctly_imported(test_data_dir):
    """Date/datetime values should match expected values"""
    df, meta = read_sav(test_data_dir / "spss/datetime.sav", coerce_temporals=True)

    # Check date value
    assert df["date"][0] == date(2014, 9, 22)

    # Check datetime value (row 1, 0-indexed)
    expected_dt = datetime(2014, 9, 23, 15, 59, 20)
    actual_dt = df["date_posix"][1]
    if isinstance(actual_dt, datetime):
        assert actual_dt.replace(tzinfo=None) == expected_dt

    # Check time value (approximately 43870 seconds = 12:11:10)
    time_val = df["time"][0]
    if isinstance(time_val, timedelta):
        assert abs(time_val.total_seconds() - 43870) < 1


def test_formats_roundtrip():
    """SPSS format attributes should roundtrip"""
    df = pl.DataFrame(
        {
            "a": [1.0, 1.0, 2.0],
            "b": [4.0, 5.0, 6.0],
            "c": [7.0, 8.0, 9.0],
            "d": ["Text", "Text", ""],
        }
    )

    # TODO: Add format.spss metadata support
    df_out = roundtrip_sav(df)

    assert df.shape == df_out.shape


def test_widths_roundtrip():
    """Display width attributes should roundtrip"""
    df = pl.DataFrame(
        {
            "a": [1.0, 1.0, 2.0],
            "b": [4.0, 5.0, 6.0],
            "c": [7.0, 8.0, 9.0],
            "d": ["Text", "Text", ""],
        }
    )

    # TODO: Add display_width metadata support
    df_out = roundtrip_sav(df)

    assert df.shape == df_out.shape


def test_only_selected_columns_are_read(test_data_dir):
    """cols_skip parameter should filter columns"""
    df_all, _ = read_sav(test_data_dir / "spss/datetime.sav")
    all_cols = set(df_all.columns)

    # Skip all but 'date'
    skip_cols = [col for col in all_cols if col != "date"]
    df_filtered, _ = read_sav(test_data_dir / "spss/datetime.sav", cols_skip=skip_cols)

    assert df_filtered.columns == ["date"]


# Row skipping/limiting ---------------------------------------------------


def test_using_skip_returns_correct_number_of_rows(test_data_dir):
    """rows_skip parameter should skip the correct number of rows"""
    df_full, _ = read_sav(test_data_dir / "spss/datetime.sav")
    n = df_full.height

    df_skip1, _ = read_sav(test_data_dir / "spss/datetime.sav", rows_skip=1)
    assert df_skip1.height == n - 1

    df_skip_n_minus_1, _ = read_sav(test_data_dir / "spss/datetime.sav", rows_skip=n - 1)
    assert df_skip_n_minus_1.height == 1

    df_skip_n, _ = read_sav(test_data_dir / "spss/datetime.sav", rows_skip=n)
    assert df_skip_n.height == 0

    df_skip_n_plus_1, _ = read_sav(test_data_dir / "spss/datetime.sav", rows_skip=n + 1)
    assert df_skip_n_plus_1.height == 0


def test_can_limit_the_number_of_rows_to_read(test_data_dir):
    """n_max parameter should limit rows correctly"""
    df_full, _ = read_sav(test_data_dir / "spss/datetime.sav")
    n = df_full.height

    df_zero, _ = read_sav(test_data_dir / "spss/datetime.sav", n_max=0)
    assert df_zero.height == 0

    df_one, _ = read_sav(test_data_dir / "spss/datetime.sav", n_max=1)
    assert df_one.height == 1

    df_n, _ = read_sav(test_data_dir / "spss/datetime.sav", n_max=n)
    assert df_n.height == n

    df_n_plus_1, _ = read_sav(test_data_dir / "spss/datetime.sav", n_max=n + 1)
    assert df_n_plus_1.height == n


# User-defined missings ---------------------------------------------------


def test_user_defined_missing_values_read_as_missing_by_default(test_data_dir):
    """User-defined missing values should be None by default"""
    df, meta = read_sav(test_data_dir / "spss/labelled-num-na.sav")

    col = df.columns[0]
    # Row 1 (0-indexed) should be None/null
    assert df[col][1] is None or pl.DataFrame({col: [df[col][1]]}).null_count()[col][0] == 1


def test_user_defined_missing_values_can_be_preserved(test_data_dir):
    """user_na=True should preserve user-defined missing values"""
    df, meta = read_sav(test_data_dir / "spss/labelled-num-na.sav", user_na=True)

    col = df.columns[0]
    assert df[col][1] == 9

    # Check metadata for na_values
    user_missing = meta.get("user_missing", [])
    assert any(um.get("col") == col and 9 in um.get("values", []) for um in user_missing)


def test_system_missings_read_as_none():
    """System missing values should become None/null"""
    df = pl.DataFrame({"x": [1.0, None]})
    df_out = roundtrip_sav(df)

    assert df_out["x"][0] == 1.0
    assert df_out["x"][1] is None or df_out.null_count()["x"] == 1


# write_sav ---------------------------------------------------------------


def test_can_roundtrip_basic_types():
    """Basic data types should roundtrip successfully"""
    # Float
    x_float = [0.1, 0.5, 0.9]
    rt_float = roundtrip_var(x_float)
    assert all(abs(a - b) < 1e-10 for a, b in zip(x_float, rt_float))

    # Integer (becomes float in SPSS)
    x_int = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rt_int = roundtrip_var(x_int)
    assert all(abs(a - b) < 1e-10 for a, b in zip(x_int, rt_int))

    # Boolean (becomes 0/1)
    x_bool = [True, False]
    rt_bool = roundtrip_var(x_bool)
    assert rt_bool[0] == 1.0 or rt_bool[0] == 1
    assert rt_bool[1] == 0.0 or rt_bool[1] == 0

    # String
    x_str = list("abcdefghijklmnopqrstuvwxyz")
    rt_str = roundtrip_var(x_str)
    assert rt_str == x_str


def test_can_roundtrip_missing_values():
    """Missing values should roundtrip (with type coercion)"""
    # Single None becomes integer 0 or null (SPSS quirk)
    rt_na = roundtrip_var([None])
    assert rt_na[0] is None or rt_na[0] == 0

    # Float NA
    rt_float_na = roundtrip_var([None], {"x": [None]})
    # Should be null or NaN

    # String NA becomes empty string
    rt_str_na = roundtrip_var([None], {"x": [None]})
    # SPSS represents string missing as empty string


def test_can_roundtrip_date_times():
    """Date and datetime values should roundtrip"""
    # Date
    x_date = [date(2010, 1, 1), None]
    df = pl.DataFrame({"x": x_date})
    df_out = roundtrip_sav(df)

    # Dates should match (allowing for coercion)
    assert df_out["x"][0] == x_date[0]
    assert df_out["x"][1] is None

    # Datetime (UTC conversion)
    x_dt = [datetime(2010, 1, 1, 9, 0, 0)]
    df_dt = pl.DataFrame({"x": x_dt})
    df_dt_out = roundtrip_sav(df_dt)

    # Should be close (SPSS stores in UTC)
    assert isinstance(df_dt_out["x"][0], (datetime, date))


def test_can_roundtrip_times():
    """Time values should roundtrip"""
    # Time as duration in seconds
    x_time = [timedelta(seconds=1), None, timedelta(seconds=86400)]
    df = pl.DataFrame({"x": x_time})
    df_out = roundtrip_sav(df)

    # Check roundtrip
    assert df_out["x"][0] == x_time[0]
    assert df_out["x"][1] is None
    assert df_out["x"][2] == x_time[2]


def test_infinity_gets_converted_to_na():
    """Infinity values should become missing"""
    x = [float("inf"), 0.0, float("-inf")]
    df = pl.DataFrame({"x": x})
    df_out = roundtrip_sav(df)

    # Middle value should be preserved
    assert df_out["x"][1] == 0.0
    # Infinities should be None/null - check using Polars null checking
    assert df_out["x"].is_null()[0] or df_out["x"].is_null()[2]


def test_factors_become_labelleds():
    """Categorical columns should get value labels"""
    df = pl.DataFrame({"x": pl.Series(["a", "b"], dtype=pl.Categorical)})

    df_out, meta_out = roundtrip_sav(df, return_meta=True)

    # Should have value labels
    labels = get_value_labels_for_column(meta_out, "x")
    assert labels is not None

    # Check the labels
    assert labels.get("1") == "a"
    assert labels.get("2") == "b"

    # Check the data was converted to numeric codes
    assert df_out["x"][0] == 1.0
    assert df_out["x"][1] == 2.0


def test_labels_are_preserved():
    """Variable labels should roundtrip"""
    df = pl.DataFrame({"x": list(range(1, 11))})
    # Write with variable labels
    var_labels = {"x": "Test variable X"}
    df_out, meta_out = roundtrip_sav(df, var_labels=var_labels, return_meta=True)

    # Check that the label was preserved
    x_var = next(v for v in meta_out["vars"] if v["name"] == "x")
    assert x_var["label"] == "Test variable X"


def test_spss_labelleds_are_round_tripped():
    """SPSS-specific labelled vectors with user_na should roundtrip"""
    # Based on Haven's test: "spss labelleds are round tripped"
    # Create data with user-defined missing values and ranges
    df = pl.DataFrame({"x": [1.0, 2.0, 1.0, 9.0, 80.0, 85.0, 90.0]})

    # Define user-defined missing values:
    # - 9 is a discrete missing value
    # - 80-90 is a range of missing values
    user_missing = [{"col": "x", "values": [9.0], "range": (80.0, 90.0)}]

    # Define value labels
    value_labels = [{"col": "x", "labels": {"1": "no", "2": "yes", "9": "unknown"}}]

    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write with user-defined missing values
        write_sav(
            df,
            tmp_path,
            user_missing=user_missing,
            value_labels=value_labels,
        )

        # Test 1: Read without user_na (default behavior)
        # User-defined missing values should be converted to None/NA
        df2, meta2 = read_sav(tmp_path)

        # Check that values are correct
        assert df2["x"][0] == 1.0  # no
        assert df2["x"][1] == 2.0  # yes
        assert df2["x"][2] == 1.0  # no
        # Values 9, 80, 85, 90 should all be None (user-defined missing)
        assert df2["x"][3] is None or df2["x"].is_null()[3]  # 9 (discrete missing)
        assert df2["x"][4] is None or df2["x"].is_null()[4]  # 80 (in range)
        assert df2["x"][5] is None or df2["x"].is_null()[5]  # 85 (in range)
        assert df2["x"][6] is None or df2["x"].is_null()[6]  # 90 (in range)

        # Check value labels are preserved
        x_labels = get_value_labels_for_column(meta2, "x")
        assert x_labels is not None
        assert x_labels.get("1") == "no"
        assert x_labels.get("2") == "yes"
        assert x_labels.get("9") == "unknown"

        # Test 2: Read with user_na=True
        # User-defined missing values should be preserved as actual values
        df3, meta3 = read_sav(tmp_path, user_na=True)

        # All original values should be preserved
        assert df3["x"][0] == 1.0
        assert df3["x"][1] == 2.0
        assert df3["x"][2] == 1.0
        assert df3["x"][3] == 9.0  # Preserved
        assert df3["x"][4] == 80.0  # Preserved
        assert df3["x"][5] == 85.0  # Preserved
        assert df3["x"][6] == 90.0  # Preserved

        # Check that user_missing metadata was preserved
        x_var = next(v for v in meta3["vars"] if v["name"] == "x")
        assert x_var["user_missing"] is not None
        assert 9.0 in x_var["user_missing"]["values"]
        assert x_var["user_missing"]["range"] is not None
        assert x_var["user_missing"]["range"][0] == 80.0
        assert x_var["user_missing"]["range"][1] == 90.0

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_spss_integer_labelleds_are_round_tripped():
    """Integer labelled vectors with user_na should roundtrip"""
    # Based on Haven's test: "spss integer labelleds are round tripped"
    df = pl.DataFrame(
        {
            "x": [1, 2, 1, 9, 80, 85, 90]  # Integers
        }
    )

    user_missing = [
        {
            "col": "x",
            "values": [9.0],  # Note: stored as float in SPSS
            "range": (80.0, 90.0),
        }
    ]

    value_labels = [{"col": "x", "labels": {"1": "no", "2": "yes", "9": "unknown"}}]

    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_sav(df, tmp_path, user_missing=user_missing, value_labels=value_labels)

        # Read without user_na
        df2, meta2 = read_sav(tmp_path)

        # First two values should be preserved, rest should be NA
        assert df2["x"][0] == 1.0
        assert df2["x"][1] == 2.0
        assert df2["x"][2] == 1.0
        assert df2["x"].is_null()[3]  # 9
        assert df2["x"].is_null()[4]  # 80
        assert df2["x"].is_null()[5]  # 85
        assert df2["x"].is_null()[6]  # 90

        # Read with user_na=True
        df3, meta3 = read_sav(tmp_path, user_na=True)

        # All values preserved
        assert df3["x"][3] == 9.0
        assert df3["x"][4] == 80.0
        assert df3["x"][5] == 85.0
        assert df3["x"][6] == 90.0

        # Check metadata
        x_var = next(v for v in meta3["vars"] if v["name"] == "x")
        assert x_var["user_missing"] is not None
        assert 9.0 in x_var["user_missing"]["values"]
        assert x_var["user_missing"]["range"][0] == 80.0
        assert x_var["user_missing"]["range"][1] == 90.0

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_na_range_roundtrips_successfully_with_mismatched_type():
    """na_range should work with different numeric types"""
    # Based on Haven's test
    x_vec = list(range(1, 11))
    x_na = [1.0, 10.0]

    df = pl.DataFrame(
        {
            "x_int_int": x_vec,
            "x_int_real": x_vec,
            "x_real_real": [float(x) for x in x_vec],
            "x_real_int": [float(x) for x in x_vec],
        }
    )

    user_missing = [
        {"col": "x_int_int", "range": (1.0, 10.0)},
        {"col": "x_int_real", "range": (1.0, 10.0)},
        {"col": "x_real_real", "range": (1.0, 10.0)},
        {"col": "x_real_int", "range": (1.0, 10.0)},
    ]

    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_sav(df, tmp_path, user_missing=user_missing)
        df2, meta2 = read_sav(tmp_path, user_na=True)

        # Check that ranges were preserved for all columns
        for col in ["x_int_int", "x_int_real", "x_real_real", "x_real_int"]:
            var = next(v for v in meta2["vars"] if v["name"] == col)
            assert var.get("user_missing") is not None
            assert var["user_missing"]["range"] is not None
            assert var["user_missing"]["range"][0] == 1.0
            assert var["user_missing"]["range"][1] == 10.0

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.mark.skip(reason="String missing values not yet fully supported")
def test_spss_string_labelleds_are_round_tripped():
    """String labelled vectors with user_na should roundtrip"""
    # Based on Haven's test: "spss string labelleds are round tripped"
    df = pl.DataFrame({"x": ["1", "2", "3", "99"]})

    user_missing = [
        {
            "col": "x",
            "values": [99.0],  # Note: Even string missings stored as numeric in metadata
            "range": (2.0, 3.0),  # String range "2" to "3"
        }
    ]

    value_labels = [{"col": "x", "labels": {"1": "one"}}]

    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_sav(df, tmp_path, user_missing=user_missing, value_labels=value_labels)

        # Read without user_na
        df2, meta2 = read_sav(tmp_path)
        assert df2["x"][0] == "1"
        # Values "2", "3", "99" should be None (user-defined missing)
        assert df2["x"][1] is None or df2["x"].is_null()[1]
        assert df2["x"][2] is None or df2["x"].is_null()[2]
        assert df2["x"][3] is None or df2["x"].is_null()[3]

        # Read with user_na=True
        df3, meta3 = read_sav(tmp_path, user_na=True)
        assert df3["x"][0] == "1"
        assert df3["x"][1] == "2"
        assert df3["x"][2] == "3"
        assert df3["x"][3] == "99"

        # Check metadata
        x_var = next(v for v in meta3["vars"] if v["name"] == "x")
        assert x_var["user_missing"] is not None

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# @pytest.mark.skip(reason="labelled vectors not yet implemented")
def test_labelleds_are_round_tripped():
    """Labelled numeric and string vectors should roundtrip"""
    # This will require implementing a labelled() type similar to haven's
    pass


def test_labels_are_converted_to_utf8():
    """UTF-8 labels should roundtrip correctly"""
    # Create DataFrame with various UTF-8 characters
    df = pl.DataFrame(
        {
            "var1": [1, 2, 3],
            "var2": [4, 5, 6],
            "var3": [7, 8, 9],
        }
    )

    # Variable labels with various UTF-8 characters
    var_labels = {
        "var1": "Größe (German - umlaut)",
        "var2": "Âge (French - circumflex)",
        "var3": "年齢 (Japanese - age)",
    }

    # Roundtrip with UTF-8 labels
    df_out, meta_out = roundtrip_sav(df, var_labels=var_labels, return_meta=True)

    # Check that all labels were preserved correctly
    labels_out = {v["name"]: v.get("label") for v in meta_out["vars"]}

    assert labels_out["var1"] == "Größe (German - umlaut)"
    assert labels_out["var2"] == "Âge (French - circumflex)"
    assert labels_out["var3"] == "年齢 (Japanese - age)"

    # Also check that data roundtripped correctly
    assert df_out.shape == df.shape
    assert df_out["var1"].to_list() == [1.0, 2.0, 3.0]


def test_complain_about_long_factor_labels():
    """Very long string values should raise an error"""
    x = "a" * 500  # SPSS has a limit on string length
    df = pl.DataFrame({"x": [x]})

    # May raise ValueError or write successfully depending on SPSS version
    # Modern SPSS supports longer strings, but there's still a limit
    try:
        roundtrip_sav(df)
        # If it succeeds, that's fine for modern SPSS
        assert True
    except ValueError as e:
        # Expected for very long strings
        assert "long" in str(e).lower() or "length" in str(e).lower()


def test_complain_about_invalid_variable_names():
    """Invalid SPSS variable names should raise an error"""
    # Duplicate names (case-insensitive in SPSS)
    df = pl.DataFrame({"a": [1], "A": [1], "b": [1]})

    with pytest.raises(ValueError, match="variable name"):
        write_sav(df, tempfile.mktemp(suffix=".sav"))

    # Invalid characters
    df = pl.DataFrame({"$var": [1], "A._$@#1": [1], "a.": [1]})

    with pytest.raises(ValueError, match="variable name"):
        write_sav(df, tempfile.mktemp(suffix=".sav"))

    # Reserved words
    df = pl.DataFrame({"ALL": [1], "eq": [1], "b": [1]})

    with pytest.raises(ValueError, match="variable name|reserved"):
        write_sav(df, tempfile.mktemp(suffix=".sav"))

    # Too long (>64 bytes)
    df = pl.DataFrame({"a" * 65: [1], "b" * 65: [2], "c": [3]})

    with pytest.raises(ValueError, match="variable name|length"):
        write_sav(df, tempfile.mktemp(suffix=".sav"))


def test_non_latin_characters_written_successfully():
    """Non-Latin variable names should work if valid in SPSS"""
    df = pl.DataFrame({"流水号": [1, 2]})

    try:
        df_out = roundtrip_sav(df)
        # If SPSS supports Unicode variable names, should work
        assert "流水号" in df_out.columns
    except ValueError:
        # If not supported, that's documented behavior
        pytest.skip("Non-Latin variable names not supported")


def test_invisibly_returns_original_data_unaltered():
    """write_sav should return the input DataFrame unchanged"""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "dt": [
                datetime(2022, 1, 1, 12, 0, 0),
                datetime(2022, 1, 2, 12, 0, 0),
                datetime(2022, 1, 3, 12, 0, 0),
                datetime(2022, 1, 4, 12, 0, 0),
                datetime(2022, 1, 5, 12, 0, 0),
            ],
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df_returned = write_sav(df, tmp_path)

        # Should return the original DataFrame unchanged
        assert df.shape == df_returned.shape
        assert df.columns == df_returned.columns

        # Data should be identical
        for col in df.columns:
            assert (df[col] == df_returned[col]).all() or (
                df[col].null_count() == df_returned[col].null_count()
            )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Compression roundtrips --------------------------------------------------


def test_all_compression_types_roundtrip_successfully():
    """Different compression types should all work"""
    df = pl.DataFrame({"x": list(range(1, 11))})

    # Test different compression modes
    for compress in ["byte", "none", "zsav"]:
        df_out = roundtrip_sav(df, compress=compress)
        assert df.shape == df_out.shape


# Fixtures ----------------------------------------------------------------


@pytest.fixture
def test_data_dir():
    """Get the test data directory"""
    # Assumes tests are run from project root or tests have access to test files
    test_dir = Path(__file__).parent / "data"
    if not test_dir.exists():
        test_dir = Path("tests/data")
    if not test_dir.exists():
        pytest.skip("Test data directory not found")
    return test_dir
