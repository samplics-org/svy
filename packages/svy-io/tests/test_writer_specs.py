import pytest

from svy_io.metadata import (
    canonical_code,
    normalize_user_missing_spec,
    normalize_value_labels,
)


@pytest.mark.parametrize(
    "code, expected",
    [("1", 1), (1.0, 1), (True, 1), ("-15", -15), (1.5, 1.5), ("1.5", 1.5), ("a", "a")],
)
def test_canonical_code(code, expected):
    got = canonical_code(code)
    assert got == expected and type(got) is type(expected)


def test_value_labels_accept_both_shapes():
    as_dict = normalize_value_labels({"x": {"1": "yes"}})
    as_list = normalize_value_labels([{"col": "x", "labels": {1.0: "yes"}}])
    assert as_dict == as_list == {"x": {1: "yes"}}


def test_value_labels_keep_text_codes_on_string_columns():
    import polars as pl

    got = normalize_value_labels({"s": {"1": "one"}}, {"s": pl.String})
    assert got == {"s": {"1": "one"}}


@pytest.mark.parametrize("bad", [[{"labels": {1: "a"}}], [{"col": "x"}], "x", {"x": ["a"]}])
def test_value_labels_reject_malformed_specs(bad):
    with pytest.raises((TypeError, ValueError)):
        normalize_value_labels(bad)


def test_user_missing_accepts_reader_and_writer_keys():
    reader = normalize_user_missing_spec([{"col": "x", "na_values": [9], "na_range": None}])
    writer = normalize_user_missing_spec([{"col": "x", "values": [9.0]}])
    keyed = normalize_user_missing_spec({"x": {"na_values": ["9"]}})
    assert reader == writer == keyed == [{"col": "x", "na_values": [9], "na_range": None}]


@pytest.mark.parametrize(
    "bad", [[{"na_values": [9]}], [{"col": "x"}], [{"col": "x", "na_range": (1,)}]]
)
def test_user_missing_rejects_unusable_entries(bad):
    with pytest.raises(ValueError):
        normalize_user_missing_spec(bad)


def test_write_sav_keeps_specs_given_in_reader_shape(tmp_path):
    import polars as pl

    from svy_io.spss import read_sav, write_sav

    path = tmp_path / "x.sav"
    write_sav(
        pl.DataFrame({"x": [1.0, 9.0]}),
        path,
        value_labels={"x": {1: "yes", 9: "don't know"}},
        user_missing=[{"col": "x", "na_values": [9], "na_range": None}],
    )
    df, meta = read_sav(path)
    assert df["x"].to_list() == [1.0, None]
    assert meta["value_labels"][0]["mapping"] == {"1": "yes", "9": "don't know"}


def test_write_sav_rejects_specs_for_unknown_columns(tmp_path):
    import polars as pl

    from svy_io.spss import write_sav

    with pytest.raises(ValueError, match="not in the DataFrame"):
        write_sav(pl.DataFrame({"x": [1.0]}), tmp_path / "x.sav", value_labels={"y": {1: "a"}})


def test_write_dta_accepts_text_codes_read_from_a_file(tmp_path):
    import polars as pl

    from svy_io.stata import read_dta, write_dta

    path = tmp_path / "x.dta"
    write_dta(
        pl.DataFrame({"x": [1, -1]}), path, value_labels={"x": {"1": "yes", "-1": "inapplicable"}}
    )
    _, meta = read_dta(path)
    assert {"1": "yes", "-1": "inapplicable"} in [vl["mapping"] for vl in meta["value_labels"]]


def _df():
    import polars as pl

    return pl.DataFrame({"x": [1.0, 2.0]})


@pytest.mark.parametrize(
    ("write", "kwargs", "what", "names"),
    [
        ("write_sav", {"var_labels": {"x": "a\x00b"}}, "variable labels", "'x'"),
        ("write_sav", {"value_labels": {"x": {1: "a\x00b"}}}, "value labels", "'x'"),
        ("write_dta", {"file_label": "a\x00b"}, "file_label", "'file_label'"),
        ("write_dta", {"var_labels": {"x": "a\x00b"}}, "variable labels", "'x'"),
        ("write_dta", {"value_labels": {"x": {1: "a\x00b"}}}, "value labels", "'x'"),
        ("write_xpt", {"label": "a\x00b"}, "name and label", "'label'"),
        ("write_xpt", {"name": "a\x00b"}, "name and label", "'name'"),
        ("write_xpt", {"var_labels": {"x": "a\x00b"}}, "variable labels", "'x'"),
    ],
)
def test_writers_reject_nul_in_labels(tmp_path, write, kwargs, what, names):
    import svy_io

    with pytest.raises(ValueError, match=rf"^{what} cannot contain NUL \(\\x00\): \[{names}\]$"):
        getattr(svy_io, write)(_df(), tmp_path / "out.bin", **kwargs)


def _ipc(df):
    import io

    bio = io.BytesIO()
    df.write_ipc(bio)
    return bio.getvalue()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"user_missing": [{"values": [9.0]}]}, r"bad user_missing entry: .*'col'"),
        (
            {"user_missing": [{"col": "x", "values": ["nine"]}]},
            "column 'x' is numeric; its user-missing codes must be numbers",
        ),
        ({"value_labels": [{"col": "x"}]}, r"bad value_labels entry: .*'labels'"),
    ],
)
def test_native_sav_writer_rejects_malformed_specs(tmp_path, kwargs, message):
    from svy_io import svyreadstat_rs as native

    with pytest.raises(ValueError, match=message):
        native.df_write_sav_file(_ipc(_df()), str(tmp_path / "n.sav"), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"var_labels": {"x": "a\x00b"}}, "variable label of column 'x' contains an embedded NUL"),
        (
            {"value_labels": [{"col": "x", "labels": {"one": "1"}}]},
            "value label code \"one\" on column 'x' is not numeric",
        ),
        (
            {"value_labels": [{"col": "x", "labels": {"1": "a\x00b"}}]},
            r"value label \"a\\0b\" on column 'x' contains an embedded NUL",
        ),
    ],
)
def test_native_sav_writer_rejects_bad_labels(tmp_path, kwargs, message):
    from svy_io import svyreadstat_rs as native

    with pytest.raises(ValueError, match=message):
        native.df_write_sav_file(_ipc(_df()), str(tmp_path / "n.sav"), **kwargs)


def test_native_dta_writer_rejects_nul_in_variable_label(tmp_path):
    from svy_io import svyreadstat_rs as native

    with pytest.raises(ValueError, match="variable label of column 'x' contains an embedded NUL"):
        native.df_write_dta_file(
            _ipc(_df()), str(tmp_path / "n.dta"), 118, None, '{"x": "a\\u0000b"}', None, 2045, None
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"version": 6}, "XPT version must be 5 or 8, got 6"),
        ({"label": "a\x00b"}, r"file label \"a\\0b\" contains an embedded NUL"),
        ({"name": "a\x00b"}, r"member name \"a\\0b\" contains an embedded NUL"),
    ],
)
def test_native_xpt_writer_rejects_bad_header(tmp_path, kwargs, message):
    from svy_io import svyreadstat_rs as native

    args = {"version": 8, "name": "DATA", "label": None, **kwargs}
    with pytest.raises(ValueError, match=message):
        native.df_write_xpt_file(_ipc(_df()), str(tmp_path / "n.xpt"), **args)


def _nul_df(dtype):
    import polars as pl

    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "s": pl.Series(["ok", "a\x00b", None]).cast(dtype),
            "t": ["\x00", "u", "v"],
        }
    )


@pytest.mark.parametrize("write", ["write_sav", "write_xpt"])
@pytest.mark.parametrize("dtype", ["String", "Categorical", "Enum"])
def test_writers_reject_nul_in_string_values(tmp_path, write, dtype):
    import polars as pl
    import svy_io

    pl_dtype = pl.Enum(["ok", "a\x00b"]) if dtype == "Enum" else getattr(pl, dtype)
    path = tmp_path / "out.bin"
    with pytest.raises(
        ValueError, match=r"^string values cannot contain NUL \(\\x00\): \['s', 't'\]$"
    ):
        getattr(svy_io, write)(_nul_df(pl_dtype), path)
    assert not path.exists()


@pytest.mark.parametrize(
    ("write", "args"),
    [
        ("df_write_sav_file", ()),
        ("df_write_xpt_file", (8, "DATA", None)),
        ("df_write_dta_file", (118, None, None, None, 2045, None)),
    ],
)
def test_native_writers_reject_nul_in_string_values(tmp_path, write, args):
    import polars as pl

    from svy_io import svyreadstat_rs as native

    path = tmp_path / "n.bin"
    with pytest.raises(
        ValueError, match="string value in column 's' at row 1 contains an embedded NUL"
    ):
        getattr(native, write)(_ipc(_nul_df(pl.String).drop("t")), str(path), *args)
    assert not path.exists()
