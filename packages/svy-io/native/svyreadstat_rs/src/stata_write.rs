// native/svyreadstat_rs/src/stata_write.rs
use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use std::collections::HashMap;
use std::ffi::CString;
use std::fs::File;
use std::io::{Cursor, Write as IoWrite};
use std::os::raw::c_void;
use std::path::Path;

use arrow::array::{
    Array, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeStringArray, StringArray, StringViewArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{
    DataType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type,
    UInt8Type,
};
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::record_batch::RecordBatch;

use readstat_sys::{
    readstat_add_variable, readstat_begin_row, readstat_begin_writing_dta, readstat_end_row,
    readstat_end_writing, readstat_insert_double_value, readstat_insert_missing_value,
    readstat_insert_string_value, readstat_set_data_writer,
    readstat_type_e_READSTAT_TYPE_DOUBLE as T_DOUBLE,
    readstat_type_e_READSTAT_TYPE_STRING as T_STRING, readstat_variable_set_label,
    readstat_variable_t, readstat_writer_free, readstat_writer_init,
    readstat_writer_set_file_format_version, readstat_writer_set_file_label,
};

unsafe extern "C" fn data_writer_cb(
    data: *const std::os::raw::c_void,
    len: usize,
    ctx: *mut c_void,
) -> isize {
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

fn ipc_to_batches(buf: &[u8]) -> Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    if buf.starts_with(b"ARROW1") {
        let mut fr = FileReader::try_new(Cursor::new(buf), None)?;
        for b in fr.by_ref() {
            batches.push(b?);
        }
    } else {
        let mut sr = StreamReader::try_new(Cursor::new(buf), None)?;
        while let Some(res) = sr.next() {
            batches.push(res?);
        }
    }
    Ok(batches)
}

#[inline]
fn is_text_dt(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

fn get_string_value(a: &dyn Array, row: usize) -> Option<&str> {
    if a.is_null(row) {
        return None;
    }
    if let Some(s) = a.as_any().downcast_ref::<StringArray>() {
        Some(s.value(row))
    } else if let Some(s) = a.as_any().downcast_ref::<LargeStringArray>() {
        Some(s.value(row))
    } else if let Some(s) = a.as_any().downcast_ref::<StringViewArray>() {
        Some(s.value(row))
    } else if matches!(a.data_type(), DataType::Dictionary(_, _)) {
        dict_string_at_any(a, row)
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
                let key_usize = d.keys().value(row) as usize;
                let values = d.values();
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

fn as_f64_opt(a: &dyn Array, row: usize) -> Option<f64> {
    if a.is_null(row) {
        return None;
    }
    macro_rules! down {
        ($T:ty) => {
            a.as_any().downcast_ref::<$T>().unwrap().value(row)
        };
    }
    use DataType::*;
    match a.data_type() {
        Float64 => Some(down!(Float64Array)),
        Float32 => Some(down!(Float32Array) as f64),
        Int64 => Some(down!(Int64Array) as f64),
        Int32 => Some(down!(Int32Array) as f64),
        Int16 => Some(down!(Int16Array) as f64),
        Int8 => Some(down!(Int8Array) as f64),
        UInt64 => Some(down!(UInt64Array) as f64),
        UInt32 => Some(down!(UInt32Array) as f64),
        UInt16 => Some(down!(UInt16Array) as f64),
        UInt8 => Some(down!(UInt8Array) as f64),
        Boolean => Some(if down!(BooleanArray) { 1.0 } else { 0.0 }),
        _ => None,
    }
}

#[derive(Clone, Copy, Default)]
struct StringColStats {
    max_len: usize,
    has_nul: bool,
}

fn compute_string_metadata(batches: &[RecordBatch]) -> Vec<Option<StringColStats>> {
    if batches.is_empty() {
        return Vec::new();
    }
    let ncols = batches[0].schema().fields().len();
    let mut all_stats: Vec<Option<StringColStats>> = vec![None; ncols];

    for b in batches {
        for (j, f) in b.schema().fields().iter().enumerate() {
            let col = b.column(j);
            let is_str = is_text_dt(f.data_type())
                || matches!(f.data_type(), &DataType::Dictionary(_, ref v) if is_text_dt(v.as_ref()));
            if !is_str {
                continue;
            }
            let stats = all_stats[j].get_or_insert(StringColStats::default());
            for i in 0..col.len() {
                if let Some(s) = get_string_value(col.as_ref(), i) {
                    let blen = s.as_bytes().len();
                    if blen > stats.max_len {
                        stats.max_len = blen;
                    }
                    if !stats.has_nul && s.as_bytes().contains(&0) {
                        stats.has_nul = true;
                    }
                }
            }
        }
    }
    all_stats
}

fn write_stata_minimal(
    batches: &[RecordBatch],
    out_path: &str,
    file_label: Option<&str>,
    version_internal: i32,
    strl_threshold: i32,
    var_labels: Option<&HashMap<String, String>>,
) -> Result<()> {
    if batches.is_empty() {
        let _ = File::create(out_path)?;
        return Ok(());
    }

    let writer = unsafe { readstat_writer_init() };
    if writer.is_null() {
        return Err(anyhow!("readstat_writer_init() failed"));
    }
    unsafe {
        readstat_writer_set_file_format_version(writer, version_internal as u8);
        readstat_set_data_writer(writer, Some(data_writer_cb));
    }

    if let Some(lbl) = file_label {
        let c = CString::new(lbl)?;
        unsafe {
            readstat_writer_set_file_label(writer, c.as_ptr());
        }
    }

    let schema = batches[0].schema();
    let ncols = schema.fields().len();

    let str_stats = compute_string_metadata(batches);
    let mut is_str_col: Vec<bool> = vec![false; ncols];

    for j in 0..ncols {
        let dt = batches[0].column(j).data_type();
        is_str_col[j] = is_text_dt(dt)
            || matches!(dt, DataType::Dictionary(_, ref v) if is_text_dt(v.as_ref()));
    }

    let mut rvars: Vec<*const readstat_variable_t> = Vec::with_capacity(ncols);
    let mut _keep_names: Vec<CString> = Vec::with_capacity(ncols);

    // Define variables
    for (j, field) in schema.fields().iter().enumerate() {
        let mut typ = T_DOUBLE;
        let mut width: usize = 0;

        if is_str_col[j] {
            let stats = str_stats[j].unwrap_or(StringColStats {
                max_len: 1,
                has_nul: false,
            });

            let needs_strl = (stats.max_len as i32) > strl_threshold;

            if needs_strl {
                unsafe { readstat_writer_free(writer) };
                return Err(anyhow!(
                    "Column '{}' contains strings longer than {} bytes (max: {}).\n\
                     \n\
                     strL support is currently unavailable due to a bug in ReadStat library v1.1.9\n\
                     where written strL files cannot be read back (results in parse error rc=5).\n\
                     \n\
                     Workarounds:\n\
                     1. Truncate strings to {} bytes before writing\n\
                     2. Use a different file format (e.g., Parquet, CSV)\n\
                     3. Track github.com/WizardMac/ReadStat for strL fixes in future releases",
                    field.name(),
                    strl_threshold,
                    stats.max_len,
                    strl_threshold
                ));
            }

            typ = T_STRING;
            width = std::cmp::max(1, std::cmp::min(2045, stats.max_len));
        }

        let cname = CString::new(field.name().as_str())?;
        let var = unsafe { readstat_add_variable(writer, cname.as_ptr(), typ, width as _) };
        if var.is_null() {
            unsafe { readstat_writer_free(writer) };
            return Err(anyhow!(
                "readstat_add_variable failed for '{}'",
                field.name()
            ));
        }

        if let Some(map) = var_labels {
            if let Some(lbl) = map.get(field.name()) {
                if !lbl.is_empty() {
                    if let Ok(c) = CString::new(lbl.as_str()) {
                        unsafe {
                            readstat_variable_set_label(var, c.as_ptr());
                        }
                    }
                }
            }
        }

        _keep_names.push(cname);
        rvars.push(var);
    }

    let mut outfile = File::create(Path::new(out_path))?;
    let total_rows: i64 = batches.iter().map(|b| b.num_rows() as i64).sum();
    unsafe {
        let rc = readstat_begin_writing_dta(
            writer,
            &mut outfile as *mut File as *mut c_void,
            total_rows.try_into().expect("row count overflow"),
        );
        if rc != 0 {
            readstat_writer_free(writer);
            return Err(anyhow!("readstat_begin_writing_dta failed with rc={}", rc));
        }
    }

    for b in batches {
        for i in 0..b.num_rows() {
            unsafe {
                let rc = readstat_begin_row(writer);
                if rc != 0 {
                    readstat_writer_free(writer);
                    return Err(anyhow!("readstat_begin_row failed with rc={}", rc));
                }
            };

            for (j, arr) in b.columns().iter().enumerate() {
                if is_str_col[j] {
                    if let Some(s) = get_string_value(arr.as_ref(), i) {
                        unsafe {
                            match CString::new(s) {
                                Ok(cs) => {
                                    let rc =
                                        readstat_insert_string_value(writer, rvars[j], cs.as_ptr());
                                    if rc != 0 {
                                        readstat_writer_free(writer);
                                        return Err(anyhow!(
                                            "insert_string_value failed with rc={}",
                                            rc
                                        ));
                                    }
                                }
                                Err(_) => {
                                    let rc = readstat_insert_missing_value(writer, rvars[j]);
                                    if rc != 0 {
                                        readstat_writer_free(writer);
                                        return Err(anyhow!(
                                            "insert_missing_value (embedded NUL) failed with rc={}",
                                            rc
                                        ));
                                    }
                                }
                            }
                        }
                    } else {
                        unsafe {
                            let rc = readstat_insert_missing_value(writer, rvars[j]);
                            if rc != 0 {
                                readstat_writer_free(writer);
                                return Err(anyhow!(
                                    "insert_missing_value (null) failed with rc={}",
                                    rc
                                ));
                            }
                        }
                    }
                } else {
                    if let Some(v) = as_f64_opt(arr.as_ref(), i) {
                        unsafe {
                            let rc = readstat_insert_double_value(writer, rvars[j], v);
                            if rc != 0 {
                                readstat_writer_free(writer);
                                return Err(anyhow!("insert_double_value failed with rc={}", rc));
                            }
                        }
                    } else {
                        unsafe {
                            let rc = readstat_insert_missing_value(writer, rvars[j]);
                            if rc != 0 {
                                readstat_writer_free(writer);
                                return Err(anyhow!(
                                    "insert_missing_value (double) failed with rc={}",
                                    rc
                                ));
                            }
                        }
                    }
                }
            }

            unsafe {
                let rc = readstat_end_row(writer);
                if rc != 0 {
                    readstat_writer_free(writer);
                    return Err(anyhow!("readstat_end_row failed with rc={}", rc));
                }
            }
        }
    }

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
#[pyo3(signature = (
    ipc_bytes,
    out_path,
    version,
    file_label=None,
    var_labels_json=None,
    _value_labels_json=None,
    strl_threshold=2045,
    _user_missing_json=None
))]
pub fn df_write_dta_file(
    ipc_bytes: Bound<'_, PyBytes>,
    out_path: &str,
    version: i32,
    file_label: Option<&str>,
    var_labels_json: Option<&str>,
    _value_labels_json: Option<&str>,
    strl_threshold: i32,
    _user_missing_json: Option<&str>,
) -> PyResult<()> {
    let buf = ipc_bytes.as_bytes();
    let batches = ipc_to_batches(buf).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Arrow IPC decode failed: {}", e))
    })?;

    let var_labels: Option<HashMap<String, String>> = if let Some(js) = var_labels_json {
        Some(
            serde_json::from_str::<HashMap<String, String>>(js).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "var_labels_json must be a JSON object of {{col: label}} strings: {e}"
                ))
            })?,
        )
    } else {
        None
    };

    write_stata_minimal(
        &batches,
        out_path,
        file_label,
        version,
        strl_threshold,
        var_labels.as_ref(),
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("df_write_dta_file: {}", e)))
}
