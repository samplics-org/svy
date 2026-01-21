// native/svyreadstat_rs/src/xpt_write.rs
// COMPLETE REWRITE - must use readstat row-by-row API

use anyhow::{anyhow, Result};
use arrow::array::*;
use arrow::datatypes::DataType;
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
    readstat_variable_set_label, readstat_variable_t, readstat_writer_free, readstat_writer_init,
    readstat_writer_set_file_format_version, readstat_writer_set_file_label,
    readstat_writer_set_table_name,
};

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

fn get_string_value(arr: &dyn Array, row: usize) -> Option<&str> {
    if arr.is_null(row) {
        return None;
    }
    if let Some(s) = arr.as_any().downcast_ref::<StringArray>() {
        Some(s.value(row))
    } else if let Some(s) = arr.as_any().downcast_ref::<LargeStringArray>() {
        Some(s.value(row))
    } else {
        None
    }
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

    let batch = &batches[0];
    let n_rows = batch.num_rows();

    let writer = unsafe { readstat_writer_init() };
    if writer.is_null() {
        return Err(anyhow!("readstat_writer_init() failed"));
    }

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
        let col_name = CString::new(field.name().as_str())?;
        is_str_col[j] = matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8);

        let (var_type, width) = if is_str_col[j] {
            (readstat_type_e_READSTAT_TYPE_STRING, 200)
        } else {
            (readstat_type_e_READSTAT_TYPE_DOUBLE, 0)
        };

        let var = unsafe { readstat_add_variable(writer, col_name.as_ptr(), var_type, width) };
        if var.is_null() {
            unsafe { readstat_writer_free(writer) };
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

    unsafe {
        let rc = readstat_begin_writing_xport(
            writer,
            &mut outfile as *mut File as *mut c_void,
            n_rows.try_into().expect("row count overflow"),
        );
        if rc != 0 {
            readstat_writer_free(writer);
            return Err(anyhow!(
                "readstat_begin_writing_xport failed with rc={}",
                rc
            ));
        }
    }

    // NOW write data row by row using readstat API
    for i in 0..n_rows {
        unsafe {
            let rc = readstat_begin_row(writer);
            if rc != 0 {
                readstat_writer_free(writer);
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
                                    readstat_writer_free(writer);
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
                                    readstat_writer_free(writer);
                                    return Err(anyhow!("insert_missing_value failed"));
                                }
                            }
                        }
                    } else {
                        let rc = readstat_insert_missing_value(writer, rvars[j]);
                        if rc != 0 {
                            readstat_writer_free(writer);
                            return Err(anyhow!("insert_missing_value failed"));
                        }
                    }
                } else {
                    if let Some(v) = as_f64_opt(arr.as_ref(), i) {
                        let rc = readstat_insert_double_value(writer, rvars[j], v);
                        if rc != 0 {
                            readstat_writer_free(writer);
                            return Err(anyhow!(
                                "insert_double_value failed at row {}, col {}",
                                i,
                                j
                            ));
                        }
                    } else {
                        let rc = readstat_insert_missing_value(writer, rvars[j]);
                        if rc != 0 {
                            readstat_writer_free(writer);
                            return Err(anyhow!("insert_missing_value failed"));
                        }
                    }
                }
            }
        }

        unsafe {
            let rc = readstat_end_row(writer);
            if rc != 0 {
                readstat_writer_free(writer);
                return Err(anyhow!("readstat_end_row failed at row {}", i));
            }
        }
    }

    // Finalize
    unsafe {
        let rc = readstat_end_writing(writer);
        if rc != 0 {
            readstat_writer_free(writer);
            return Err(anyhow!("readstat_end_writing failed with rc={}", rc));
        }
        readstat_writer_free(writer);
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
