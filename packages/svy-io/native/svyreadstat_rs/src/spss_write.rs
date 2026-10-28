use pyo3::prelude::*;
use pyo3::types::PyBytes;

use std::collections::HashMap;
use std::ffi::CString;
use std::io::Cursor;

use arrow::array::{
    Array, AsArray, DictionaryArray, LargeStringArray, StringArray, StringViewArray,
};
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, DurationMicrosecondType, DurationMillisecondType,
    DurationNanosecondType, DurationSecondType, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::record_batch::RecordBatch;

use readstat_sys::{
    readstat_add_label_set, readstat_add_variable, readstat_begin_row, readstat_begin_writing_sav,
    readstat_compress_e_READSTAT_COMPRESS_BINARY as COMPRESS_BINARY,
    readstat_compress_e_READSTAT_COMPRESS_NONE as COMPRESS_NONE,
    readstat_compress_e_READSTAT_COMPRESS_ROWS as COMPRESS_ROWS, readstat_end_row,
    readstat_end_writing, readstat_insert_double_value, readstat_insert_missing_value,
    readstat_insert_string_value, readstat_label_double_value, readstat_label_set_t,
    readstat_label_string_value, readstat_set_data_writer,
    readstat_type_e_READSTAT_TYPE_DOUBLE as T_DOUBLE,
    readstat_type_e_READSTAT_TYPE_STRING as T_STRING, readstat_variable_add_missing_double_range,
    readstat_variable_add_missing_double_value, readstat_variable_add_missing_string_value,
    readstat_variable_set_format, readstat_variable_set_label, readstat_variable_set_label_set,
    readstat_variable_t, readstat_writer_init, readstat_writer_set_compression,
    readstat_writer_set_file_label,
};

use crate::core::{OutputFile, WriterGuard, data_writer_cb, write_rc};
use crate::errors::{Error, Op, Result};

fn ipc_to_batches(buf: &[u8]) -> arrow::error::Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    let schema = if buf.starts_with(b"ARROW1") {
        let mut fr = FileReader::try_new(Cursor::new(buf), None)?;
        for b in fr.by_ref() {
            batches.push(b?);
        }
        fr.schema()
    } else {
        let mut sr = StreamReader::try_new(Cursor::new(buf), None)?;
        for res in sr.by_ref() {
            batches.push(res?);
        }
        sr.schema()
    };
    // A zero-row frame arrives with no batches; the writers still need its columns.
    if batches.is_empty() {
        batches.push(RecordBatch::new_empty(schema));
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
            let arr = a.as_primitive::<Date32Type>();
            let days_since_unix = arr.value(row) as i64;
            let total_secs = ((days_since_unix + UNIX_MINUS_SPSS_DAYS) as f64) * SECS_PER_DAY_F64;
            Some(total_secs)
        }
        DataType::Date64 => {
            let arr = a.as_primitive::<Date64Type>();
            let ms_since_unix = arr.value(row) as f64;
            Some((ms_since_unix / MSEC_PER_SEC_F64) + UNIX_MINUS_SPSS_SECS_F64)
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            let arr = a.as_primitive::<TimestampSecondType>();
            let s = arr.value(row) as f64;
            Some(s + UNIX_MINUS_SPSS_SECS_F64)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let arr = a.as_primitive::<TimestampMillisecondType>();
            let ms = arr.value(row) as f64;
            Some((ms / MSEC_PER_SEC_F64) + UNIX_MINUS_SPSS_SECS_F64)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let arr = a.as_primitive::<TimestampMicrosecondType>();
            let us = arr.value(row) as f64;
            Some((us / USEC_PER_SEC_F64) + UNIX_MINUS_SPSS_SECS_F64)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let arr = a.as_primitive::<TimestampNanosecondType>();
            let ns = arr.value(row) as f64;
            Some((ns / NSEC_PER_SEC_F64) + UNIX_MINUS_SPSS_SECS_F64)
        }
        DataType::Duration(TimeUnit::Second) => {
            let arr = a.as_primitive::<DurationSecondType>();
            let seconds = arr.value(row) as f64;
            Some(seconds)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let arr = a.as_primitive::<DurationMillisecondType>();
            let ms = arr.value(row) as f64;
            Some(ms / MSEC_PER_SEC_F64)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let arr = a.as_primitive::<DurationMicrosecondType>();
            let us = arr.value(row) as f64;
            Some(us / USEC_PER_SEC_F64)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            let arr = a.as_primitive::<DurationNanosecondType>();
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
            a.as_primitive::<$T>().value(row)
        };
    }
    use DataType::*;
    let result = match a.data_type() {
        Float64 => Some(down!(Float64Type)),
        Float32 => Some(down!(Float32Type) as f64),
        Int64 => Some(down!(Int64Type) as f64),
        Int32 => Some(down!(Int32Type) as f64),
        Int16 => Some(down!(Int16Type) as f64),
        Int8 => Some(down!(Int8Type) as f64),
        UInt64 => Some(down!(UInt64Type) as f64),
        UInt32 => Some(down!(UInt32Type) as f64),
        UInt16 => Some(down!(UInt16Type) as f64),
        UInt8 => Some(down!(UInt8Type) as f64),
        Boolean => Some(if a.as_boolean().value(row) { 1.0 } else { 0.0 }),
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
                || matches!(f.data_type(), DataType::Dictionary(_, v) if is_text_dt(v.as_ref()));
            if !is_str {
                continue;
            }
            let stats = all_stats[j].get_or_insert(StringColStats::default());
            for i in 0..col.len() {
                if let Some(s) = get_string_value(col.as_ref(), i) {
                    let blen = s.len();
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
        return Err(Error::Invalid("no record batches to write".into()));
    }

    // SAFETY: no preconditions; a null result is checked below.
    let writer = unsafe { readstat_writer_init() };
    if writer.is_null() {
        return Err(Error::Internal("readstat_writer_init() failed".into()));
    }
    let _writer_guard = WriterGuard(writer);

    let compress_type = match compress {
        "none" => COMPRESS_NONE,
        "byte" => COMPRESS_ROWS,
        "zsav" => COMPRESS_BINARY,
        other => {
            return Err(Error::Invalid(format!(
                "unknown compress {other:?}; expected 'none', 'byte' or 'zsav'"
            )));
        }
    };
    // SAFETY: `writer` is live until `_writer_guard` drops.
    unsafe {
        readstat_writer_set_compression(writer, compress_type);
        readstat_set_data_writer(writer, Some(data_writer_cb));
    }

    if let Some(lbl) = file_label {
        let c = CString::new(lbl)
            .map_err(|_| Error::Invalid(format!("file label {lbl:?} contains an embedded NUL")))?;
        // SAFETY: `writer` is live; ReadStat copies the label.
        unsafe { readstat_writer_set_file_label(writer, c.as_ptr()) };
    }

    let schema = batches[0].schema();
    let ncols = schema.fields().len();

    let str_stats = compute_string_metadata(batches);
    let is_str_col: Vec<bool> = (0..ncols)
        .map(|j| {
            let dt = batches[0].column(j).data_type();
            is_text_dt(dt) || matches!(dt, DataType::Dictionary(_, v) if is_text_dt(v.as_ref()))
        })
        .collect();

    let mut rvars: Vec<*const readstat_variable_t> = Vec::with_capacity(ncols);
    let mut _keep_names: Vec<CString> = Vec::with_capacity(ncols);
    let mut _keep_label_sets: Vec<(*const readstat_label_set_t, Vec<CString>)> = Vec::new();
    // ReadStat stores string missing values by pointer and reads them in begin_writing.
    let mut _keep_missing: Vec<CString> = Vec::new();

    for (j, field) in schema.fields().iter().enumerate() {
        let dt = batches[0].column(j).data_type();
        let mut typ = T_DOUBLE;
        let mut width: usize = 0;

        if is_str_col[j] {
            let stats = str_stats[j].unwrap_or(StringColStats { max_len: 1 });
            // Checked here; ReadStat would fail later with an opaque rc at row insert.
            if stats.max_len > 2000 {
                return Err(Error::Invalid(format!(
                    "column '{}' contains strings up to {} bytes; the SAV \
                     writer supports at most 2000 bytes per string value",
                    field.name(),
                    stats.max_len
                )));
            }
            typ = T_STRING;
            width = std::cmp::max(1, stats.max_len);
        }

        let cname = CString::new(field.name().as_str()).map_err(|_| {
            Error::Invalid(format!(
                "column name {:?} contains an embedded NUL",
                field.name()
            ))
        })?;
        // SAFETY: `writer` is live; ReadStat copies the name.
        let var = unsafe { readstat_add_variable(writer, cname.as_ptr(), typ, width as _) };
        if var.is_null() {
            return Err(Error::Write(format!(
                "adding column '{}' failed",
                field.name()
            )));
        }

        let want_fmt = match dt {
            DataType::Date32 | DataType::Date64 => Some("DATE10"),
            DataType::Timestamp(_, _) => Some("DATETIME20"),
            DataType::Duration(_) => Some("TIME11.2"),
            _ => None,
        };
        if let Some(fmt) = want_fmt {
            let cfmt = CString::new(fmt).expect("SPSS format names hold no NUL");
            // SAFETY: `var` lives as long as `writer`; ReadStat copies the format.
            unsafe { readstat_variable_set_format(var, cfmt.as_ptr()) };
        }

        if let Some(map) = var_labels
            && let Some(lbl) = map.get(field.name())
            && !lbl.is_empty()
        {
            let c = CString::new(lbl.as_str()).map_err(|_| {
                Error::Invalid(format!(
                    "variable label of column '{}' contains an embedded NUL",
                    field.name()
                ))
            })?;
            // SAFETY: `var` lives as long as `writer`; ReadStat copies the label.
            unsafe { readstat_variable_set_label(var, c.as_ptr()) };
        }

        if let Some(user_miss) = user_missing {
            for um in user_miss {
                if um.col == field.name().as_str() {
                    let mut rcs = Vec::new();
                    if is_str_col[j] {
                        // SAV stores string missing values only for strings of at most 8 bytes.
                        if width > 8 {
                            return Err(Error::Invalid(format!(
                                "column '{}' holds strings up to {} bytes; SPSS user-missing \
                                 values apply only to string columns of at most 8 bytes",
                                field.name(),
                                width
                            )));
                        }
                        for val in &um.values {
                            // `1`, not `1.0`, for whole numbers.
                            let val_string = if val.fract() == 0.0 {
                                format!("{:.0}", val)
                            } else {
                                val.to_string()
                            };
                            let c_val =
                                CString::new(val_string).expect("a formatted number holds no NUL");
                            // SAFETY: `var` is live; `_keep_missing` holds `c_val` until begin_writing reads it.
                            rcs.push(unsafe {
                                readstat_variable_add_missing_string_value(var, c_val.as_ptr())
                            });
                            _keep_missing.push(c_val);
                        }
                        // ReadStat has no string-range API, so `um.range` is ignored.
                    } else {
                        for &val in &um.values {
                            // SAFETY: `var` lives as long as `writer`.
                            rcs.push(unsafe {
                                readstat_variable_add_missing_double_value(var, val)
                            });
                        }
                        if let Some((low, high)) = um.range {
                            // SAFETY: `var` lives as long as `writer`.
                            rcs.push(unsafe {
                                readstat_variable_add_missing_double_range(var, low, high)
                            });
                        }
                    }
                    for rc in rcs {
                        write_rc(rc, || {
                            format!("adding user-missing values to column '{}'", field.name())
                        })?;
                    }
                }
            }
        }

        if let Some(val_labs) = value_labels {
            for vl in val_labs {
                if vl.col == field.name().as_str() && !vl.labels.is_empty() {
                    let label_set_name = format!("{}_labels", field.name());
                    let c_label_set_name = CString::new(label_set_name)
                        .expect("the column name was checked for NUL above");

                    let label_set =
                        // SAFETY: `writer` is live; ReadStat copies the name.
                        unsafe { readstat_add_label_set(writer, typ, c_label_set_name.as_ptr()) };

                    if label_set.is_null() {
                        return Err(Error::Write(format!(
                            "adding the value labels of column '{}' failed",
                            field.name()
                        )));
                    }
                    let mut c_strings = Vec::new();

                    for (value, label) in &vl.labels {
                        let c_label = CString::new(label.as_str()).map_err(|_| {
                            Error::Invalid(format!(
                                "value label {:?} on column '{}' contains an embedded NUL",
                                label,
                                field.name()
                            ))
                        })?;
                        if is_str_col[j] {
                            let c_val = CString::new(value.as_str()).map_err(|_| {
                                Error::Invalid(format!(
                                    "value label code {:?} on column '{}' contains an embedded NUL",
                                    value,
                                    field.name()
                                ))
                            })?;
                            // SAFETY: `label_set` is live; ReadStat copies both strings.
                            unsafe {
                                readstat_label_string_value(
                                    label_set,
                                    c_val.as_ptr(),
                                    c_label.as_ptr(),
                                );
                            }
                            c_strings.push(c_val);
                        } else {
                            let num_val: f64 = value.parse().map_err(|_| {
                                Error::Invalid(format!(
                                    "value label code {:?} on column '{}' is not numeric",
                                    value,
                                    field.name()
                                ))
                            })?;
                            // SAFETY: `label_set` is live; ReadStat copies the label.
                            unsafe {
                                readstat_label_double_value(label_set, num_val, c_label.as_ptr());
                            }
                        }
                        c_strings.push(c_label);
                    }

                    // SAFETY: `var` and `label_set` belong to the live `writer`.
                    unsafe {
                        readstat_variable_set_label_set(var, label_set);
                    }

                    _keep_label_sets.push((label_set, c_strings));
                }
            }
        }

        _keep_names.push(cname);
        rvars.push(var);
    }

    let total_rows: i64 = batches.iter().map(|b| b.num_rows() as i64).sum();
    #[allow(clippy::useless_conversion)] // c_long is 32-bit on Windows
    let row_count = total_rows
        .try_into()
        .map_err(|_| Error::Invalid(format!("{total_rows} rows exceed this platform's limit")))?;
    let mut out = OutputFile::create(out_path)?;
    write_rc(
        // SAFETY: `writer` is live, and `out` outlives every write `data_writer_cb` makes into it.
        unsafe { readstat_begin_writing_sav(writer, out.as_ctx(), row_count) },
        || "beginning the file".into(),
    )?;

    let mut row_offset = 0;
    for b in batches {
        for i in 0..b.num_rows() {
            let row = row_offset + i;
            // SAFETY: `writer` is live.
            write_rc(unsafe { readstat_begin_row(writer) }, || {
                format!("beginning row {row}")
            })?;

            for (j, arr) in b.columns().iter().enumerate() {
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
                    match as_f64_opt(arr.as_ref(), i) {
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
        row_offset += b.num_rows();
    }

    // SAFETY: `writer` is live, and so is `out`.
    write_rc(unsafe { readstat_end_writing(writer) }, || {
        "finishing the file".into()
    })?;
    out.finish();
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
    user_missing: Option<Vec<HashMap<String, Py<PyAny>>>>,
    value_labels: Option<Vec<HashMap<String, Py<PyAny>>>>,
) -> PyResult<()> {
    let to_py = |e: Error| e.into_py(Op::Write, "SAV", out_path);
    let batches = ipc_to_batches(ipc_bytes.as_bytes())
        .map_err(|e| to_py(Error::Invalid(format!("Arrow IPC decode failed: {e}"))))?;

    let bad_entry = |what: &str, e: PyErr| to_py(Error::Invalid(format!("bad {what} entry: {e}")));

    let user_missing_converted: Option<Vec<UserMissingInfo>> = match &user_missing {
        None => None,
        Some(um_vec) => Some(Python::attach(|py| {
            um_vec
                .iter()
                .map(|um_dict| {
                    let col = um_dict
                        .get("col")
                        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("col"))?
                        .extract::<String>(py)?;
                    let values = match um_dict.get("values") {
                        Some(v) if !v.is_none(py) => v.extract::<Vec<f64>>(py)?,
                        _ => Vec::new(),
                    };
                    let range = match um_dict.get("range") {
                        Some(r) if !r.is_none(py) => Some(r.extract::<(f64, f64)>(py)?),
                        _ => None,
                    };
                    Ok(UserMissingInfo { col, values, range })
                })
                .collect::<PyResult<Vec<_>>>()
                .map_err(|e| bad_entry("user_missing", e))
        })?),
    };

    let value_labels_converted: Option<Vec<ValueLabelsInfo>> = match &value_labels {
        None => None,
        Some(vl_vec) => Some(Python::attach(|py| {
            vl_vec
                .iter()
                .map(|vl_dict| {
                    let field = |key: &str| {
                        vl_dict
                            .get(key)
                            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))
                    };
                    let col = field("col")?.extract::<String>(py)?;
                    let labels = field("labels")?.extract::<HashMap<String, String>>(py)?;
                    Ok(ValueLabelsInfo { col, labels })
                })
                .collect::<PyResult<Vec<_>>>()
                .map_err(|e| bad_entry("value_labels", e))
        })?),
    };

    write_spss_minimal(
        &batches,
        out_path,
        file_label,
        compress,
        var_labels.as_ref(),
        user_missing_converted.as_deref(),
        value_labels_converted.as_deref(),
    )
    .map_err(to_py)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float64Array};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn unknown_compress_is_rejected_before_the_file_is_created() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let x: ArrayRef = Arc::new(Float64Array::from(vec![1.0]));
        let batch = RecordBatch::try_new(schema, vec![x]).unwrap();
        let path = std::env::temp_dir().join("svy_io_unknown_compress.sav");
        let _ = std::fs::remove_file(&path);

        let err = write_spss_minimal(
            &[batch],
            path.to_str().unwrap(),
            None,
            "rows",
            None,
            None,
            None,
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("unknown compress \"rows\""),
            "{err}"
        );
        assert!(!path.exists());
    }
}
