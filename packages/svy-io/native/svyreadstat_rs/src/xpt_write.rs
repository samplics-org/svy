// native/svyreadstat_rs/src/xpt_write.rs
// COMPLETE REWRITE - must use readstat row-by-row API

use anyhow::{anyhow, Result};
use arrow::array::*;
use arrow::datatypes::{
    DataType, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use arrow::ipc::reader::FileReader;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::ffi::CString;
use std::fs::File;
use std::io::{Cursor, Write as IoWrite};
use std::os::raw::c_void;
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

use crate::core::WriterGuard;

unsafe extern "C" fn data_writer_cb(data: *const c_void, len: usize, ctx: *mut c_void) -> isize {
    if data.is_null() || ctx.is_null() {
        return -1;
    }
    let file = &mut *(ctx as *mut File);
    let bytes = std::slice::from_raw_parts(data as *const u8, len);
    match file.write_all(bytes) {
        Ok(_) => len as isize,
        Err(_) => -1,
    }
}

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
                // A corrupt dictionary can hold a negative or out-of-range
                // key; a plain `as usize` cast would wrap and panic below.
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
        DataType::Float64 => Some(
            arr.as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(row),
        ),
        DataType::Float32 => Some(
            arr.as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row) as f64,
        ),
        DataType::Int64 => Some(
            arr.as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row) as f64,
        ),
        DataType::Int32 => Some(
            arr.as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(row) as f64,
        ),
        DataType::Int16 => Some(
            arr.as_any()
                .downcast_ref::<Int16Array>()
                .unwrap()
                .value(row) as f64,
        ),
        DataType::Int8 => Some(arr.as_any().downcast_ref::<Int8Array>().unwrap().value(row) as f64),
        DataType::UInt64 => Some(
            arr.as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(row) as f64,
        ),
        DataType::UInt32 => Some(
            arr.as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .value(row) as f64,
        ),
        DataType::UInt16 => Some(
            arr.as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap()
                .value(row) as f64,
        ),
        DataType::UInt8 => Some(
            arr.as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .value(row) as f64,
        ),
        DataType::Boolean => Some(
            if arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(row)
            {
                1.0
            } else {
                0.0
            },
        ),
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
    let cursor = Cursor::new(ipc_bytes);
    let reader = FileReader::try_new(cursor, None)?;
    let schema = reader.schema();
    let batches: Vec<_> = reader.collect::<std::result::Result<_, _>>()?;

    if batches.is_empty() {
        let _ = File::create(path)?;
        return Ok(());
    }

    let writer = unsafe { readstat_writer_init() };
    if writer.is_null() {
        return Err(anyhow!("readstat_writer_init() failed"));
    }
    // Frees the writer on every exit path (including early `?` returns).
    let _writer_guard = WriterGuard(writer);

    unsafe {
        let xpt_version = if version == 5 { 5 } else { 8 };
        readstat_writer_set_file_format_version(writer, xpt_version);
        readstat_set_data_writer(writer, Some(data_writer_cb));
    }

    if let Some(lbl) = label {
        if let Ok(lbl_cstr) = CString::new(lbl) {
            unsafe {
                readstat_writer_set_file_label(writer, lbl_cstr.as_ptr());
            }
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

    if let Ok(member_cstr) = CString::new(member_name) {
        unsafe {
            readstat_writer_set_table_name(writer, member_cstr.as_ptr());
        }
    }

    // Add variables
    let ncols = schema.fields().len();
    let mut is_str_col: Vec<bool> = vec![false; ncols];
    let mut rvars: Vec<*mut readstat_variable_t> = Vec::with_capacity(ncols);
    let mut _keep_names: Vec<CString> = Vec::with_capacity(ncols);

    for (j, field) in schema.fields().iter().enumerate() {
        is_str_col[j] = is_text_dt(field.data_type())
            || matches!(field.data_type(), DataType::Dictionary(_, v) if is_text_dt(v.as_ref()));
    }

    // Derive string widths from the data and validate BEFORE any bytes are
    // written. The transport format caps character fields at 200 bytes; the
    // old hardcoded width failed at insert time, leaving a truncated file
    // on disk.
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
            return Err(anyhow!(
                "column '{}' contains strings up to {} bytes; the XPT transport \
                 format supports at most {} bytes per character value",
                field.name(),
                str_widths[j],
                XPT_MAX_STR_BYTES
            ));
        }
    }

    for (j, field) in schema.fields().iter().enumerate() {
        let col_name = CString::new(field.name().as_str())?;

        let (var_type, width) = if is_str_col[j] {
            (readstat_type_e_READSTAT_TYPE_STRING, str_widths[j].max(1))
        } else {
            (readstat_type_e_READSTAT_TYPE_DOUBLE, 0)
        };

        let var = unsafe { readstat_add_variable(writer, col_name.as_ptr(), var_type, width) };
        if var.is_null() {
            return Err(anyhow!("Failed to add variable: {}", field.name()));
        }

        let metadata = field.metadata();
        if let Some(label_str) = metadata.get("label") {
            if let Ok(label_cstr) = CString::new(label_str.as_str()) {
                unsafe {
                    readstat_variable_set_label(var, label_cstr.as_ptr());
                }
            }
        }
        if let Some(format_str) = metadata.get("format") {
            if let Ok(format_cstr) = CString::new(format_str.as_str()) {
                unsafe {
                    readstat_variable_set_format(var, format_cstr.as_ptr());
                }
            }
        }

        _keep_names.push(col_name);
        rvars.push(var);
    }

    // Create file and begin writing
    let mut outfile = File::create(Path::new(path))?;

    // All record batches are written; a large frame is commonly split into
    // several batches by Arrow IPC serialization.
    let total_rows: i64 = batches.iter().map(|b| b.num_rows() as i64).sum();
    let row_count = total_rows
        .try_into()
        .map_err(|_| anyhow!("row count {total_rows} exceeds platform limit"))?;

    unsafe {
        let rc = readstat_begin_writing_xport(
            writer,
            &mut outfile as *mut File as *mut c_void,
            row_count,
        );
        if rc != 0 {
            return Err(anyhow!(
                "readstat_begin_writing_xport failed with rc={}",
                rc
            ));
        }
    }

    // NOW write data row by row using readstat API
    for batch in &batches {
        for i in 0..batch.num_rows() {
            unsafe {
                let rc = readstat_begin_row(writer);
                if rc != 0 {
                    return Err(anyhow!("readstat_begin_row failed at row {}", i));
                }
            }

            for (j, arr) in batch.columns().iter().enumerate() {
                unsafe {
                    if is_str_col[j] {
                        if let Some(s) = get_string_value(arr.as_ref(), i) {
                            match CString::new(s) {
                                Ok(cs) => {
                                    let rc =
                                        readstat_insert_string_value(writer, rvars[j], cs.as_ptr());
                                    if rc != 0 {
                                        return Err(anyhow!(
                                            "insert_string_value failed at row {}, col {}",
                                            i,
                                            j
                                        ));
                                    }
                                }
                                Err(_) => {
                                    let rc = readstat_insert_missing_value(writer, rvars[j]);
                                    if rc != 0 {
                                        return Err(anyhow!("insert_missing_value failed"));
                                    }
                                }
                            }
                        } else {
                            let rc = readstat_insert_missing_value(writer, rvars[j]);
                            if rc != 0 {
                                return Err(anyhow!("insert_missing_value failed"));
                            }
                        }
                    } else {
                        if let Some(v) = as_f64_opt(arr.as_ref(), i) {
                            let rc = readstat_insert_double_value(writer, rvars[j], v);
                            if rc != 0 {
                                return Err(anyhow!(
                                    "insert_double_value failed at row {}, col {}",
                                    i,
                                    j
                                ));
                            }
                        } else {
                            let rc = readstat_insert_missing_value(writer, rvars[j]);
                            if rc != 0 {
                                return Err(anyhow!("insert_missing_value failed"));
                            }
                        }
                    }
                }
            }

            unsafe {
                let rc = readstat_end_row(writer);
                if rc != 0 {
                    return Err(anyhow!("readstat_end_row failed at row {}", i));
                }
            }
        }
    }

    // Finalize
    unsafe {
        let rc = readstat_end_writing(writer);
        if rc != 0 {
            return Err(anyhow!("readstat_end_writing failed with rc={}", rc));
        }
    }

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (ipc_bytes, path, version=8, name=None, label=None))]
pub fn df_write_xpt_file(
    ipc_bytes: Bound<'_, PyBytes>,
    path: &str,
    version: i32,
    name: Option<&str>,
    label: Option<&str>,
) -> PyResult<()> {
    let buf = ipc_bytes.as_bytes();
    write_xpt_impl(buf, path, version, name, label)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}
