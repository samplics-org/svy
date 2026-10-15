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
