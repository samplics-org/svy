// native/svyreadstat_rs/src/spss_write.rs
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
    Array, BooleanArray, Date32Array, Date64Array, DictionaryArray, DurationMicrosecondArray,
    DurationMillisecondArray, DurationNanosecondArray, DurationSecondArray, Float32Array,
    Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, LargeStringArray, StringArray,
    StringViewArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow::datatypes::{
    DataType, Int16Type, Int32Type, Int64Type, Int8Type, TimeUnit, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::record_batch::RecordBatch;

use readstat_sys::{
    readstat_add_label_set, readstat_add_variable, readstat_begin_row, readstat_begin_writing_sav,
    readstat_compress_e_READSTAT_COMPRESS_NONE as COMPRESS_NONE,
    readstat_compress_e_READSTAT_COMPRESS_ROWS as COMPRESS_ROWS, readstat_end_row,
    readstat_end_writing, readstat_insert_double_value, readstat_insert_missing_value,
    readstat_insert_string_value, readstat_label_double_value, readstat_label_set_t,
    readstat_label_string_value, readstat_set_data_writer,
    readstat_type_e_READSTAT_TYPE_DOUBLE as T_DOUBLE,
    readstat_type_e_READSTAT_TYPE_STRING as T_STRING, readstat_variable_add_missing_double_range,
    readstat_variable_add_missing_double_value, readstat_variable_add_missing_string_value,
    readstat_variable_set_format, readstat_variable_set_label, readstat_variable_set_label_set,
    readstat_variable_t, readstat_writer_free, readstat_writer_init,
    readstat_writer_set_compression, readstat_writer_set_file_label,
};

/// ReadStat data sink: write to a std::fs::File
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

// ---- SPSS epoch math ----------------------------------------------------
const SECS_PER_DAY_F64: f64 = 86_400.0;
const MSEC_PER_SEC_F64: f64 = 1_000.0;
const USEC_PER_SEC_F64: f64 = 1_000_000.0;
const NSEC_PER_SEC_F64: f64 = 1_000_000_000.0;

const UNIX_MINUS_SPSS_DAYS: i64 = 141_428;
const UNIX_MINUS_SPSS_SECS_F64: f64 = (UNIX_MINUS_SPSS_DAYS as f64) * SECS_PER_DAY_F64;

fn temporal_as_spss_seconds(a: &dyn Array, row: usize) -> Option<f64> {
    if a.is_null(row) {
        return None;
    }
    match a.data_type() {
        DataType::Date32 => {
            let arr = a.as_any().downcast_ref::<Date32Array>().unwrap();
            let days_since_unix = arr.value(row) as i64;
            let total_secs = ((days_since_unix + UNIX_MINUS_SPSS_DAYS) as f64) * SECS_PER_DAY_F64;
            Some(total_secs)
        }
        DataType::Date64 => {
            let arr = a.as_any().downcast_ref::<Date64Array>().unwrap();
            let ms_since_unix = arr.value(row) as f64;
            Some((ms_since_unix / MSEC_PER_SEC_F64) + UNIX_MINUS_SPSS_SECS_F64)
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            let arr = a.as_any().downcast_ref::<TimestampSecondArray>().unwrap();
            let s = arr.value(row) as f64;
            Some(s + UNIX_MINUS_SPSS_SECS_F64)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let arr = a
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .unwrap();
            let ms = arr.value(row) as f64;
            Some((ms / MSEC_PER_SEC_F64) + UNIX_MINUS_SPSS_SECS_F64)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let arr = a
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let us = arr.value(row) as f64;
            Some((us / USEC_PER_SEC_F64) + UNIX_MINUS_SPSS_SECS_F64)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let arr = a
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap();
            let ns = arr.value(row) as f64;
            Some((ns / NSEC_PER_SEC_F64) + UNIX_MINUS_SPSS_SECS_F64)
        }
        DataType::Duration(TimeUnit::Second) => {
            let arr = a.as_any().downcast_ref::<DurationSecondArray>().unwrap();
            let seconds = arr.value(row) as f64;
            Some(seconds)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let arr = a
                .as_any()
                .downcast_ref::<DurationMillisecondArray>()
                .unwrap();
            let ms = arr.value(row) as f64;
            Some(ms / MSEC_PER_SEC_F64)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let arr = a
                .as_any()
                .downcast_ref::<DurationMicrosecondArray>()
                .unwrap();
            let us = arr.value(row) as f64;
            Some(us / USEC_PER_SEC_F64)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            let arr = a
                .as_any()
                .downcast_ref::<DurationNanosecondArray>()
                .unwrap();
            let ns = arr.value(row) as f64;
            Some(ns / NSEC_PER_SEC_F64)
        }
        _ => None,
    }
}

fn as_f64_opt(a: &dyn Array, row: usize) -> Option<f64> {
    if let Some(sec) = temporal_as_spss_seconds(a, row) {
        return Some(sec);
    }

    if a.is_null(row) {
        return None;
    }
    macro_rules! down {
        ($T:ty) => {
            a.as_any().downcast_ref::<$T>().unwrap().value(row)
        };
    }
    use DataType::*;
    let result = match a.data_type() {
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
    };

    match result {
        Some(val) if val.is_infinite() => None,
        other => other,
    }
}

#[derive(Clone, Copy, Default)]
struct StringColStats {
    max_len: usize,
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
                }
            }
        }
    }
    all_stats
}

#[derive(Debug, Clone)]
struct UserMissingInfo {
    col: String,
    values: Vec<f64>,
    range: Option<(f64, f64)>,
}

#[derive(Debug, Clone)]
struct ValueLabelsInfo {
    col: String,
    labels: HashMap<String, String>,
}

fn write_spss_minimal(
    batches: &[RecordBatch],
    out_path: &str,
    file_label: Option<&str>,
    compress: &str,
    var_labels: Option<&HashMap<String, String>>,
    user_missing: Option<&[UserMissingInfo]>,
    value_labels: Option<&[ValueLabelsInfo]>,
) -> Result<()> {
    if batches.is_empty() {
        let _ = File::create(out_path)?;
        return Ok(());
    }

    let writer = unsafe { readstat_writer_init() };
    if writer.is_null() {
        return Err(anyhow!("readstat_writer_init() failed"));
    }

    let compress_type = match compress {
        "none" => COMPRESS_NONE,
        "byte" => COMPRESS_ROWS,
        _ => COMPRESS_ROWS,
    };
    unsafe {
        readstat_writer_set_compression(writer, compress_type);
        readstat_set_data_writer(writer, Some(data_writer_cb));
    }

    if let Some(lbl) = file_label {
        let c = CString::new(lbl)?;
        unsafe { readstat_writer_set_file_label(writer, c.as_ptr()) };
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
    let mut _keep_label_sets: Vec<(*const readstat_label_set_t, Vec<CString>)> = Vec::new();

    // Define variables
    for (j, field) in schema.fields().iter().enumerate() {
        let dt = batches[0].column(j).data_type();
        let mut typ = T_DOUBLE;
        let mut width: usize = 0;

        if is_str_col[j] {
            let stats = str_stats[j].unwrap_or(StringColStats { max_len: 1 });
            typ = T_STRING;
            width = std::cmp::max(1, std::cmp::min(2000, stats.max_len));
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

        // SPSS display format for temporal columns
        let want_fmt = match dt {
            DataType::Date32 | DataType::Date64 => Some("DATE10"),
            DataType::Timestamp(_, _) => Some("DATETIME20"),
            DataType::Duration(_) => Some("TIME11.2"),
            _ => None,
        };
        if let Some(fmt) = want_fmt {
            if let Ok(cfmt) = CString::new(fmt) {
                unsafe { readstat_variable_set_format(var, cfmt.as_ptr()) };
            }
        }

        // Variable label
        if let Some(map) = var_labels {
            if let Some(lbl) = map.get(field.name()) {
                if !lbl.is_empty() {
                    if let Ok(c) = CString::new(lbl.as_str()) {
                        unsafe { readstat_variable_set_label(var, c.as_ptr()) };
                    }
                }
            }
        }

        // User-defined missing values
        if let Some(user_miss) = user_missing {
            for um in user_miss {
                if um.col == field.name().as_str() {
                    if is_str_col[j] {
                        // String: only discrete user-missing values are supported
                        for val in &um.values {
                            // To string (avoid trailing .0 for whole numbers)
                            let val_string = if val.fract() == 0.0 {
                                format!("{:.0}", val)
                            } else {
                                val.to_string()
                            };
                            if let Ok(c_val) = CString::new(val_string.as_str()) {
                                unsafe {
                                    readstat_variable_add_missing_string_value(var, c_val.as_ptr());
                                }
                            }
                        }
                        // Note: SPSS (ReadStat) does not expose a "string range" API; ignore um.range for strings.
                    } else {
                        // Numeric: discrete + range
                        for &val in &um.values {
                            unsafe {
                                readstat_variable_add_missing_double_value(var, val);
                            }
                        }
                        if let Some((low, high)) = um.range {
                            unsafe {
                                readstat_variable_add_missing_double_range(var, low, high);
                            }
                        }
                    }
                }
            }
        }

        // Value labels
        if let Some(val_labs) = value_labels {
            for vl in val_labs {
                if vl.col == field.name().as_str() && !vl.labels.is_empty() {
                    let label_set_name = format!("{}_labels", field.name());
                    let c_label_set_name = CString::new(label_set_name.as_str())?;

                    let label_set =
                        unsafe { readstat_add_label_set(writer, typ, c_label_set_name.as_ptr()) };

                    if !label_set.is_null() {
                        let mut c_strings = Vec::new();

                        for (value, label) in &vl.labels {
                            if let Ok(c_label) = CString::new(label.as_str()) {
                                if is_str_col[j] {
                                    // String value labels
                                    if let Ok(c_val) = CString::new(value.as_str()) {
                                        unsafe {
                                            readstat_label_string_value(
                                                label_set,
                                                c_val.as_ptr(),
                                                c_label.as_ptr(),
                                            );
                                        }
                                        c_strings.push(c_val);
                                    }
                                } else {
                                    // Numeric value labels
                                    if let Ok(num_val) = value.parse::<f64>() {
                                        unsafe {
                                            readstat_label_double_value(
                                                label_set,
                                                num_val,
                                                c_label.as_ptr(),
                                            );
                                        }
                                    }
                                }
                                c_strings.push(c_label);
                            }
                        }

                        unsafe {
                            readstat_variable_set_label_set(var, label_set);
                        }

                        _keep_label_sets.push((label_set, c_strings));
                    }
                }
            }
        }

        _keep_names.push(cname);
        rvars.push(var);
    }

    // Open output and begin writing
    let mut outfile = File::create(Path::new(out_path))?;
    let total_rows: i64 = batches.iter().map(|b| b.num_rows() as i64).sum();
    unsafe {
        let rc = readstat_begin_writing_sav(
            writer,
            &mut outfile as *mut File as *mut c_void,
            total_rows.try_into().expect("row count overflow"),
        );
        if rc != 0 {
            readstat_writer_free(writer);
            return Err(anyhow!("readstat_begin_writing_sav failed with rc={}", rc));
        }
    }

    // Write rows
    for b in batches {
        for i in 0..b.num_rows() {
            unsafe {
                let rc = readstat_begin_row(writer);
                if rc != 0 {
                    readstat_writer_free(writer);
                    return Err(anyhow!("readstat_begin_row failed with rc={}", rc));
                }
            }

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
                } else if let Some(v) = as_f64_opt(arr.as_ref(), i) {
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
#[pyo3(signature = (ipc_bytes, out_path, file_label=None, compress="byte", var_labels=None, user_missing=None, value_labels=None))]
pub fn df_write_sav_file(
    ipc_bytes: Bound<'_, PyBytes>,
    out_path: &str,
    file_label: Option<&str>,
    compress: &str,
    var_labels: Option<HashMap<String, String>>,
    user_missing: Option<Vec<HashMap<String, PyObject>>>,
    value_labels: Option<Vec<HashMap<String, PyObject>>>,
) -> PyResult<()> {
    let buf = ipc_bytes.as_bytes();
    let batches = ipc_to_batches(buf).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Arrow IPC decode failed: {}", e))
    })?;

    // Convert user_missing from Python-friendly dicts
    let user_missing_converted: Option<Vec<UserMissingInfo>> =
        user_missing.as_ref().map(|um_vec| {
            Python::with_gil(|py| {
                um_vec
                    .iter()
                    .filter_map(|um_dict| {
                        let col = um_dict.get("col")?.extract::<String>(py).ok()?;

                        let values = um_dict
                            .get("values")
                            .and_then(|v| v.extract::<Vec<f64>>(py).ok())
                            .unwrap_or_default();

                        let range = um_dict
                            .get("range")
                            .and_then(|r| r.extract::<(f64, f64)>(py).ok());

                        Some(UserMissingInfo { col, values, range })
                    })
                    .collect()
            })
        });

    // Convert value_labels from Python-friendly dicts
    let value_labels_converted: Option<Vec<ValueLabelsInfo>> =
        value_labels.as_ref().map(|vl_vec| {
            Python::with_gil(|py| {
                vl_vec
                    .iter()
                    .filter_map(|vl_dict| {
                        let col = vl_dict.get("col")?.extract::<String>(py).ok()?;
                        let labels = vl_dict
                            .get("labels")?
                            .extract::<HashMap<String, String>>(py)
                            .ok()?;
                        Some(ValueLabelsInfo { col, labels })
                    })
                    .collect()
            })
        });

    write_spss_minimal(
        &batches,
        out_path,
        file_label,
        compress,
        var_labels.as_ref(),
        user_missing_converted.as_deref(),
        value_labels_converted.as_deref(),
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("df_write_sav_file: {}", e)))
}
