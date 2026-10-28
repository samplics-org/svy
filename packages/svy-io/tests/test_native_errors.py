"""Native failures raise a coded exception naming the format and the file."""

import errno
import io

from pathlib import Path

import polars as pl
import pytest

from svy_io import (
    read_dta,
    read_por,
    read_sas,
    read_sas_arrow,
    read_sav,
    read_xpt,
    write_dta,
    write_sav,
    write_xpt,
)
from svy_io import svyreadstat_rs as native


SAS_DATA = Path(__file__).parent / "data" / "sas"

READERS = [
    (read_sav, "SAV", "sav"),
    (read_por, "POR", "por"),
    (read_dta, "DTA", "dta"),
    (read_sas, "SAS", "sas7bdat"),
    (read_xpt, "XPT", "xpt"),
]
WRITERS = [(write_sav, "SAV", "sav"), (write_dta, "DTA", "dta"), (write_xpt, "XPT", "xpt")]


def _ipc(df: pl.DataFrame) -> bytes:
    bio = io.BytesIO()
    df.write_ipc(bio)
    return bio.getvalue()


def test_exception_classes_subclass_the_builtin_ones():
    assert issubclass(native.ReadStatValueError, ValueError)
    assert issubclass(native.ReadStatRuntimeError, RuntimeError)


# ---- readers ----


@pytest.mark.parametrize(("reader", "fmt", "ext"), READERS)
def test_missing_file_raises_file_not_found(tmp_path, reader, fmt, ext):
    path = str(tmp_path / f"missing.{ext}")
    with pytest.raises(FileNotFoundError) as exc:
        reader(path)
    e = exc.value
    assert (e.errno, e.filename, e.code) == (errno.ENOENT, path, "IO_READ_FAILED")
    assert e.strerror.startswith(f"Failed to read {fmt} file: ")


@pytest.mark.parametrize("n_max", [None, 0])
@pytest.mark.parametrize(("reader", "fmt", "ext"), READERS)
def test_unparseable_file_raises_even_when_no_rows_are_asked_for(
    tmp_path, reader, fmt, ext, n_max
):
    path = tmp_path / f"bogus.{ext}"
    path.write_bytes(b"not a data file" * 20)
    with pytest.raises(native.ReadStatRuntimeError) as exc:
        reader(str(path), n_max=n_max)
    assert exc.value.code == "READSTAT_PARSE_FAILED"
    assert str(exc.value).startswith(f'Failed to parse {fmt} file "{path}": ')


def _written(tmp_path: Path, writer, ext: str) -> Path:
    path = tmp_path / f"ok.{ext}"
    writer(pl.DataFrame({"x": [1.0, 2.0], "s": ["a", "b"]}), path)
    return path


@pytest.mark.parametrize(
    ("reader", "writer", "fmt", "ext"),
    [
        (read_sav, write_sav, "SAV", "sav"),
        (read_dta, write_dta, "DTA", "dta"),
        (read_xpt, write_xpt, "XPT", "xpt"),
        (read_sas, None, "SAS", "sas7bdat"),
    ],
)
def test_unknown_encoding_is_a_value_error(tmp_path, reader, writer, fmt, ext):
    path = SAS_DATA / "hadley.sas7bdat" if writer is None else _written(tmp_path, writer, ext)
    with pytest.raises(native.ReadStatValueError) as exc:
        reader(str(path), encoding="no-such-codec")
    assert exc.value.code == "UNSUPPORTED_ENCODING"
    assert str(exc.value) == (
        f'Failed to read {fmt} file "{path}": unsupported encoding "no-such-codec": '
        "File has an unsupported character set (rc=7)"
    )


def test_undecodable_string_keeps_its_class_and_code_with_the_hint(tmp_path):
    path = tmp_path / "bad.dta"
    # Mid-string: ReadStat drops an invalid sequence that ends a field instead of failing.
    write_dta(pl.DataFrame({"s": ["pre cafe! post", "plain"]}), path)
    raw = path.read_bytes()
    assert raw.count(b"cafe!") == 1
    path.write_bytes(raw.replace(b"cafe!", b"caf\xe9!"))

    with pytest.raises(native.ReadStatRuntimeError) as exc:
        read_dta(path, encoding="utf-8")
    assert exc.value.code == "READSTAT_BAD_STRING"
    msg = str(exc.value)
    assert msg.startswith(
        f'Failed to parse DTA file "{path}": Unable to convert string to the requested '
        "encoding (invalid byte sequence) (rc=17)"
    )
    assert ". Hint: " in msg and "utf8-lossy" in msg


def _dta_with_duplicate_name(tmp_path: Path) -> Path:
    path = tmp_path / "dup.dta"
    write_dta(pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]}), path)
    raw = bytearray(path.read_bytes())
    # Format 118 stores each name in a 129-byte field; rename the second to "a".
    start = raw.index(b"<varnames>") + len(b"<varnames>")
    assert raw[start + 129 : start + 131] == b"b\x00"
    raw[start + 129] = ord("a")
    path.write_bytes(bytes(raw))
    return path


def test_duplicate_variable_names_raise_naming_the_variable(tmp_path):
    path = _dta_with_duplicate_name(tmp_path)
    with pytest.raises(native.ReadStatRuntimeError) as exc:
        read_dta(path)
    assert exc.value.code == "READSTAT_PARSE_FAILED"
    assert str(exc.value) == (
        f'Failed to parse DTA file "{path}": the file has more than one variable named "a"'
    )


def test_skipping_a_duplicated_name_skips_every_variable_with_it(tmp_path):
    df, meta = read_dta(_dta_with_duplicate_name(tmp_path), cols_skip=["a"])
    assert df.to_dict(as_series=False) == {"c": [5.0, 6.0]}
    assert [v["name"] for v in meta["vars"]] == ["c"]


@pytest.mark.parametrize(
    ("reader", "data"),
    [
        (read_sas, SAS_DATA / "hadley.sas7bdat"),
        (read_xpt, SAS_DATA / "hadley.xpt"),
        (read_sas_arrow, SAS_DATA / "hadley.sas7bdat"),
    ],
)
@pytest.mark.parametrize(
    ("catalog", "reason"),
    [
        ("missing.sas7bcat", None),
        ("bogus.sas7bcat", "Invalid file, or file has unsupported features (rc=5)"),
    ],
)
def test_unreadable_catalog_warns_and_reads_the_data_without_labels(
    tmp_path, reader, data, catalog, reason
):
    cat = tmp_path / catalog
    if reason is not None:
        cat.write_bytes(b"not a catalog" * 20)
    with pytest.warns(UserWarning) as record:
        got, meta = reader(str(data), catalog_path=str(cat))

    assert len(record) == 1
    msg = str(record[0].message)
    prefix = f'SAS catalog "{cat}" gave no value labels: '
    assert msg == prefix + reason if reason else msg.startswith(prefix)
    assert record[0].filename == __file__
    assert meta["value_labels"] == [] and "catalog_error" not in meta
    assert got.equals(reader(str(data))[0])


# ---- writers ----


@pytest.mark.parametrize(("writer", "fmt", "ext"), WRITERS)
def test_unwritable_path_raises_file_not_found(tmp_path, writer, fmt, ext):
    path = tmp_path / "no_such_dir" / f"x.{ext}"
    with pytest.raises(FileNotFoundError) as exc:
        writer(pl.DataFrame({"x": [1.0]}), path)
    e = exc.value
    assert (e.errno, e.filename, e.code) == (errno.ENOENT, str(path), "IO_WRITE_FAILED")
    assert e.strerror.startswith(f"Failed to write {fmt} file: ")


@pytest.mark.parametrize(
    ("write", "fmt", "ext", "kwargs", "detail"),
    [
        (
            native.df_write_sav_file,
            "SAV",
            "sav",
            {"value_labels": [{"col": "x", "labels": {"one": "1"}}]},
            "value label code \"one\" on column 'x' is not numeric",
        ),
        (
            native.df_write_dta_file,
            "DTA",
            "dta",
            {"version": 118, "user_missing_json": '[{"col": "x", "rows": [5], "tags": ["a"]}]'},
            "column 'x': tagged row 5 is past the last row (1)",
        ),
        (
            native.df_write_xpt_file,
            "XPT",
            "xpt",
            {"version": 6},
            "XPT version must be 5 or 8, got 6",
        ),
    ],
)
def test_native_writer_input_errors_are_coded_value_errors(
    tmp_path, write, fmt, ext, kwargs, detail
):
    path = str(tmp_path / f"n.{ext}")
    with pytest.raises(native.ReadStatValueError) as exc:
        write(_ipc(pl.DataFrame({"x": [1.0]})), path, **kwargs)
    assert exc.value.code == "INVALID_VALUE"
    assert str(exc.value) == f'Failed to write {fmt} file "{path}": {detail}'


@pytest.mark.parametrize(
    ("write", "fmt", "ext", "kwargs"),
    [
        (native.df_write_sav_file, "SAV", "sav", {}),
        (native.df_write_dta_file, "DTA", "dta", {"version": 118}),
        (native.df_write_xpt_file, "XPT", "xpt", {}),
    ],
)
def test_failed_write_removes_the_partial_file(tmp_path, write, fmt, ext, kwargs):
    path = tmp_path / f"n.{ext}"
    path.write_bytes(b"previous contents")
    # Each format rejects a name starting with a digit only once writing has begun.
    with pytest.raises(native.ReadStatRuntimeError) as exc:
        write(_ipc(pl.DataFrame({"1x": [1.0]})), str(path), **kwargs)
    assert exc.value.code == "READSTAT_WRITE_FAILED"
    assert str(exc.value).startswith(f'Failed to write {fmt} file "{path}": ')
    assert not path.exists()
