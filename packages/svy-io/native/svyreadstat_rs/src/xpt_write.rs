use arrow::array::*;
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use arrow::ipc::reader::FileReader;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::ffi::CString;
use std::io::Cursor;
use std::path::Path;

use readstat_sys::{
    readstat_add_variable, readstat_begin_row, readstat_begin_writing_xport, readstat_end_row,
    readstat_end_writing, readstat_insert_double_value, readstat_insert_missing_value,
    readstat_insert_string_value, readstat_set_data_writer, readstat_type_e_READSTAT_TYPE_DOUBLE,
    readstat_type_e_READSTAT_TYPE_STRING, readstat_variable_set_format,
    readstat_variable_set_label, readstat_variable_t, readstat_writer_init,
    readstat_writer_set_file_format_version, readstat_writer_set_file_label,
    readstat_writer_set_table_name,
};

use crate::core::{OutputFile, WriterGuard, data_writer_cb, write_rc};
use crate::errors::{Error, Op, Result};

#[inline]
fn is_text_dt(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

fn get_string_value(arr: &dyn Array, row: usize) -> Option<&str> {
    if arr.is_null(row) {
        return None;
    }
    if let Some(s) = arr.as_any().downcast_ref::<StringArray>() {
        Some(s.value(row))
    } else if let Some(s) = arr.as_any().downcast_ref::<LargeStringArray>() {
        Some(s.value(row))
    } else if let Some(s) = arr.as_any().downcast_ref::<StringViewArray>() {
        Some(s.value(row))
    } else if matches!(arr.data_type(), DataType::Dictionary(_, _)) {
        dict_string_at_any(arr, row)
    } else {
        None
    }
}

fn dict_string_at_any(a: &dyn Array, row: usize) -> Option<&str> {
    macro_rules! try_dict {
        ($T:ty) => {{
            if let Some(d) = a.as_any().downcast_ref::<DictionaryArray<$T>>() {
                if !is_text_dt(d.values().data_type()) || d.is_null(row) {
                    return None;
                }
                // A corrupt dictionary can hold a negative or out-of-range key; `as usize` would wrap.
                let key_usize = usize::try_from(d.keys().value(row)).ok()?;
                let values = d.values();
                if key_usize >= values.len() {
                    return None;
                }
                return get_string_value(values.as_ref(), key_usize);
            }
        }};
    }
    try_dict!(Int8Type);
    try_dict!(Int16Type);
    try_dict!(Int32Type);
    try_dict!(Int64Type);
    try_dict!(UInt8Type);
    try_dict!(UInt16Type);
    try_dict!(UInt32Type);
    try_dict!(UInt64Type);
    None
}

fn as_f64_opt(arr: &dyn Array, row: usize) -> Option<f64> {
    if arr.is_null(row) {
        return None;
    }
    match arr.data_type() {
        DataType::Float64 => Some(arr.as_primitive::<Float64Type>().value(row)),
        DataType::Float32 => Some(arr.as_primitive::<Float32Type>().value(row) as f64),
        DataType::Int64 => Some(arr.as_primitive::<Int64Type>().value(row) as f64),
        DataType::Int32 => Some(arr.as_primitive::<Int32Type>().value(row) as f64),
        DataType::Int16 => Some(arr.as_primitive::<Int16Type>().value(row) as f64),
        DataType::Int8 => Some(arr.as_primitive::<Int8Type>().value(row) as f64),
        DataType::UInt64 => Some(arr.as_primitive::<UInt64Type>().value(row) as f64),
        DataType::UInt32 => Some(arr.as_primitive::<UInt32Type>().value(row) as f64),
        DataType::UInt16 => Some(arr.as_primitive::<UInt16Type>().value(row) as f64),
        DataType::UInt8 => Some(arr.as_primitive::<UInt8Type>().value(row) as f64),
        DataType::Boolean => Some(if arr.as_boolean().value(row) {
            1.0
        } else {
            0.0
        }),
        _ => None,
    }
}

fn write_xpt_impl(
    ipc_bytes: &[u8],
    path: &str,
    version: i32,
    name: Option<&str>,
    label: Option<&str>,
) -> Result<()> {
    let ipc_err =
        |e: arrow::error::ArrowError| Error::Invalid(format!("Arrow IPC decode failed: {e}"));
    let reader = FileReader::try_new(Cursor::new(ipc_bytes), None).map_err(ipc_err)?;
    let schema = reader.schema();
    let mut batches: Vec<_> = reader
        .collect::<std::result::Result<_, _>>()
        .map_err(ipc_err)?;
    // A zero-row frame arrives with no batches; the writer still needs its columns.
    if batches.is_empty() {
        batches.push(RecordBatch::new_empty(schema.clone()));
    }

    // SAFETY: no preconditions; a null result is checked below.
    let writer = unsafe { readstat_writer_init() };
    if writer.is_null() {
        return Err(Error::Internal("readstat_writer_init() failed".into()));
    }
    let _writer_guard = WriterGuard(writer);

    if version != 5 && version != 8 {
        return Err(Error::Invalid(format!(
            "XPT version must be 5 or 8, got {version}"
        )));
    }
    // SAFETY: `writer` is live until `_writer_guard` drops.
    unsafe {
        readstat_writer_set_file_format_version(writer, version as u8);
        readstat_set_data_writer(writer, Some(data_writer_cb));
    }

    if let Some(lbl) = label {
        let lbl_cstr = CString::new(lbl)
            .map_err(|_| Error::Invalid(format!("file label {lbl:?} contains an embedded NUL")))?;
        // SAFETY: `writer` is live; ReadStat copies the label.
        unsafe {
            readstat_writer_set_file_label(writer, lbl_cstr.as_ptr());
        }
    }

    let member_name = name
        .map(|s| s.to_string())
        .or_else(|| {
            Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "DATA".to_string());

    let member_cstr = CString::new(member_name.as_str()).map_err(|_| {
        Error::Invalid(format!(
            "member name {member_name:?} contains an embedded NUL"
        ))
    })?;
    // SAFETY: `writer` is live; ReadStat copies the name.
    unsafe {
        readstat_writer_set_table_name(writer, member_cstr.as_ptr());
    }

    let ncols = schema.fields().len();
    let mut is_str_col: Vec<bool> = vec![false; ncols];
    let mut rvars: Vec<*mut readstat_variable_t> = Vec::with_capacity(ncols);
    let mut _keep_names: Vec<CString> = Vec::with_capacity(ncols);

    for (j, field) in schema.fields().iter().enumerate() {
        is_str_col[j] = is_text_dt(field.data_type())
            || matches!(field.data_type(), DataType::Dictionary(_, v) if is_text_dt(v.as_ref()));
    }

    // XPT caps character fields at 200 bytes; checked before any byte is written.
    const XPT_MAX_STR_BYTES: usize = 200;
    let mut str_widths: Vec<usize> = vec![0; ncols];
    for b in &batches {
        for (j, w) in str_widths.iter_mut().enumerate() {
            if !is_str_col[j] {
                continue;
            }
            let col = b.column(j);
            for i in 0..col.len() {
                if let Some(s) = get_string_value(col.as_ref(), i) {
                    *w = (*w).max(s.len());
                }
            }
        }
    }
    for (j, field) in schema.fields().iter().enumerate() {
        if is_str_col[j] && str_widths[j] > XPT_MAX_STR_BYTES {
            return Err(Error::Invalid(format!(
                "column '{}' contains strings up to {} bytes; the XPT transport \
                 format supports at most {} bytes per character value",
                field.name(),
                str_widths[j],
                XPT_MAX_STR_BYTES
            )));
        }
    }

    for (j, field) in schema.fields().iter().enumerate() {
        let col_name = CString::new(field.name().as_str()).map_err(|_| {
            Error::Invalid(format!(
                "column name {:?} contains an embedded NUL",
                field.name()
            ))
        })?;

        let (var_type, width) = if is_str_col[j] {
            (readstat_type_e_READSTAT_TYPE_STRING, str_widths[j].max(1))
        } else {
            (readstat_type_e_READSTAT_TYPE_DOUBLE, 0)
        };

        // SAFETY: `writer` is live; ReadStat copies the name.
        let var = unsafe { readstat_add_variable(writer, col_name.as_ptr(), var_type, width) };
        if var.is_null() {
            return Err(Error::Write(format!(
                "adding column '{}' failed",
                field.name()
            )));
        }

        let metadata = field.metadata();
        if let Some(label_str) = metadata.get("label") {
            let label_cstr = CString::new(label_str.as_str()).map_err(|_| {
                Error::Invalid(format!(
                    "variable label of column '{}' contains an embedded NUL",
                    field.name()
                ))
            })?;
            // SAFETY: `var` lives as long as `writer`; ReadStat copies the label.
            unsafe {
                readstat_variable_set_label(var, label_cstr.as_ptr());
            }
        }
        if let Some(format_str) = metadata.get("format") {
            let format_cstr = CString::new(format_str.as_str()).map_err(|_| {
                Error::Invalid(format!(
                    "format of column '{}' contains an embedded NUL",
                    field.name()
                ))
            })?;
            // SAFETY: `var` lives as long as `writer`; ReadStat copies the format.
            unsafe {
                readstat_variable_set_format(var, format_cstr.as_ptr());
            }
        }

        _keep_names.push(col_name);
        rvars.push(var);
    }

    // Arrow IPC commonly splits a large frame into several batches.
    let total_rows: i64 = batches.iter().map(|b| b.num_rows() as i64).sum();
    #[allow(clippy::useless_conversion)] // c_long is 32-bit on Windows
    let row_count = total_rows
        .try_into()
        .map_err(|_| Error::Invalid(format!("{total_rows} rows exceed this platform's limit")))?;
    let mut out = OutputFile::create(path)?;
    write_rc(
        // SAFETY: `writer` is live, and `out` outlives every write `data_writer_cb` makes into it.
        unsafe { readstat_begin_writing_xport(writer, out.as_ctx(), row_count) },
        || "beginning the file".into(),
    )?;

    let mut row_offset = 0;
    for batch in &batches {
        for i in 0..batch.num_rows() {
            let row = row_offset + i;
            // SAFETY: `writer` is live.
            write_rc(unsafe { readstat_begin_row(writer) }, || {
                format!("beginning row {row}")
            })?;

            for (j, arr) in batch.columns().iter().enumerate() {
                let rc = if is_str_col[j] {
                    match get_string_value(arr.as_ref(), i).map(CString::new) {
                        // SAFETY: `rvars[j]` is live; ReadStat copies `cs` into the row.
                        Some(Ok(cs)) => unsafe {
                            readstat_insert_string_value(writer, rvars[j], cs.as_ptr())
                        },
                        // A NUL cannot be stored, so the value is written as missing.
                        // SAFETY: `rvars[j]` belongs to the live `writer`.
                        Some(Err(_)) | None => unsafe {
                            readstat_insert_missing_value(writer, rvars[j])
                        },
                    }
                } else {
                    // XPT's IBM floats hold no infinity or NaN (NaN came out as +inf); haven writes
                    // them as missing too.
                    match as_f64_opt(arr.as_ref(), i).filter(|v| v.is_finite()) {
                        // SAFETY: `rvars[j]` belongs to the live `writer`.
                        Some(v) => unsafe { readstat_insert_double_value(writer, rvars[j], v) },
                        // SAFETY: `rvars[j]` belongs to the live `writer`.
                        None => unsafe { readstat_insert_missing_value(writer, rvars[j]) },
                    }
                };
                write_rc(rc, || {
                    format!("writing column '{}' at row {row}", schema.field(j).name())
                })?;
            }

            // SAFETY: `writer` is live.
            write_rc(unsafe { readstat_end_row(writer) }, || {
                format!("ending row {row}")
            })?;
        }
        row_offset += batch.num_rows();
    }

    // SAFETY: `writer` is live, and so is `out`.
    write_rc(unsafe { readstat_end_writing(writer) }, || {
        "finishing the file".into()
    })?;
    out.finish();
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (ipc_bytes, path, version=8, name=None, label=None))]
pub(crate) fn df_write_xpt_file(
    ipc_bytes: Bound<'_, PyBytes>,
    path: &str,
    version: i32,
    name: Option<&str>,
    label: Option<&str>,
) -> PyResult<()> {
    write_xpt_impl(ipc_bytes.as_bytes(), path, version, name, label)
        .map_err(|e| e.into_py(Op::Write, "XPT", path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema};
    use arrow::ipc::writer::FileWriter;
    use arrow::record_batch::RecordBatch;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn ipc(fields: Vec<Field>, columns: Vec<ArrayRef>) -> Vec<u8> {
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let mut buf = Vec::new();
        let mut w = FileWriter::try_new(&mut buf, &schema).unwrap();
        w.write(&batch).unwrap();
        w.finish().unwrap();
        drop(w);
        buf
    }

    fn one_double(name: &str) -> Vec<u8> {
        ipc(
            vec![Field::new(name, DataType::Float64, true)],
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        )
    }

    fn fresh_path(stem: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("svy_io_{stem}.xpt"));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn invalid(
        bytes: &[u8],
        stem: &str,
        version: i32,
        name: Option<&str>,
        label: Option<&str>,
    ) -> String {
        let path = fresh_path(stem);
        let err = write_xpt_impl(bytes, path.to_str().unwrap(), version, name, label).unwrap_err();
        assert!(!path.exists(), "{stem}: a rejected write left a file");
        match err {
            Error::Invalid(m) => m,
            other => panic!("{stem}: expected Error::Invalid, got {other:?}"),
        }
    }

    #[test]
    fn a_valid_frame_writes_a_file() {
        let path = fresh_path("xpt_valid");
        write_xpt_impl(&one_double("x"), path.to_str().unwrap(), 8, None, None).unwrap();
        assert!(path.exists());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn bytes_that_are_not_arrow_ipc_are_rejected() {
        let m = invalid(b"not arrow", "xpt_not_ipc", 8, None, None);
        assert!(m.starts_with("Arrow IPC decode failed"), "{m}");
    }

    #[test]
    fn only_versions_5_and_8_are_written() {
        for version in [4, 6, 9] {
            let m = invalid(&one_double("x"), "xpt_version", version, None, None);
            assert_eq!(m, format!("XPT version must be 5 or 8, got {version}"));
        }
    }

    #[test]
    fn an_embedded_nul_is_rejected_wherever_it_appears() {
        let m = invalid(&one_double("x"), "xpt_nul_label", 8, None, Some("a\0b"));
        assert!(m.starts_with("file label"), "{m}");
        let m = invalid(&one_double("x"), "xpt_nul_member", 8, Some("a\0b"), None);
        assert!(m.starts_with("member name"), "{m}");
        let m = invalid(&one_double("a\0b"), "xpt_nul_column", 8, None, None);
        assert!(m.starts_with("column name"), "{m}");

        let labelled = Field::new("x", DataType::Float64, true)
            .with_metadata(HashMap::from([("label".to_string(), "a\0b".to_string())]));
        let bytes = ipc(
            vec![labelled],
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        );
        let m = invalid(&bytes, "xpt_nul_var_label", 8, None, None);
        assert_eq!(m, "variable label of column 'x' contains an embedded NUL");
    }

    #[test]
    fn a_string_over_200_bytes_is_rejected_before_writing() {
        let bytes = ipc(
            vec![Field::new("s", DataType::Utf8, true)],
            vec![Arc::new(StringArray::from(vec!["a".repeat(201)]))],
        );
        let m = invalid(&bytes, "xpt_long_string", 8, None, None);
        assert!(
            m.contains("column 's' contains strings up to 201 bytes") && m.contains("at most 200"),
            "{m}"
        );
    }

    #[test]
    fn a_missing_directory_is_an_io_error() {
        let path = std::env::temp_dir()
            .join("svy_io_no_such_dir")
            .join("out.xpt");
        let err =
            write_xpt_impl(&one_double("x"), path.to_str().unwrap(), 8, None, None).unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound),
            "{err:?}"
        );
    }
}
