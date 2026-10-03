// native/svyreadstat_rs/src/stata_write.rs
use anyhow::{Result, anyhow};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use std::collections::HashMap;
use std::ffi::CString;
use std::fs::File;
use std::io::{Cursor, Write as IoWrite};
use std::os::raw::c_void;
use std::path::Path;

use arrow::array::{
    Array, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{
    DataType, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::record_batch::RecordBatch;

use readstat_sys::{
    readstat_add_label_set, readstat_add_variable, readstat_begin_row, readstat_begin_writing_dta,
    readstat_end_row, readstat_end_writing, readstat_insert_double_value,
    readstat_insert_int8_value, readstat_insert_int16_value, readstat_insert_int32_value,
    readstat_insert_missing_value, readstat_insert_string_value, readstat_label_double_value,
    readstat_label_set_t, readstat_set_data_writer,
    readstat_type_e_READSTAT_TYPE_DOUBLE as T_DOUBLE, readstat_type_e_READSTAT_TYPE_INT8 as T_INT8,
    readstat_type_e_READSTAT_TYPE_INT16 as T_INT16, readstat_type_e_READSTAT_TYPE_INT32 as T_INT32,
    readstat_type_e_READSTAT_TYPE_STRING as T_STRING, readstat_type_t, readstat_variable_set_label,
    readstat_variable_set_label_set, readstat_variable_t, readstat_writer_init,
    readstat_writer_set_file_format_version, readstat_writer_set_file_label,
};

use crate::core::WriterGuard;

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

fn as_i64_opt(a: &dyn Array, row: usize) -> Option<i64> {
    if a.is_null(row) {
        return None;
    }
    macro_rules! down {
        ($T:ty) => {
            a.as_any().downcast_ref::<$T>().unwrap().value(row) as i64
        };
    }
    use DataType::*;
    match a.data_type() {
        Int64 => Some(down!(Int64Array)),
        Int32 => Some(down!(Int32Array)),
        Int16 => Some(down!(Int16Array)),
        Int8 => Some(down!(Int8Array)),
        UInt64 => Some(down!(UInt64Array)),
        UInt32 => Some(down!(UInt32Array)),
        UInt16 => Some(down!(UInt16Array)),
        UInt8 => Some(down!(UInt8Array)),
        _ => None,
    }
}

/// Smallest and largest non-null value of an integer column across batches;
/// `None` when the column is not an integer type. An all-null integer column
/// reports `Some(None)`.
fn int_col_range(batches: &[RecordBatch], j: usize) -> Option<Option<(i128, i128)>> {
    use arrow::compute::{max, min};
    let mut range: Option<(i128, i128)> = None;
    for b in batches {
        let a = b.column(j).as_ref();
        macro_rules! mm {
            ($T:ty) => {{
                let a = a.as_any().downcast_ref::<$T>().unwrap();
                (min(a).map(|v| v as i128), max(a).map(|v| v as i128))
            }};
        }
        use DataType::*;
        let (lo, hi) = match a.data_type() {
            Int64 => mm!(Int64Array),
            Int32 => mm!(Int32Array),
            Int16 => mm!(Int16Array),
            Int8 => mm!(Int8Array),
            UInt64 => mm!(UInt64Array),
            UInt32 => mm!(UInt32Array),
            UInt16 => mm!(UInt16Array),
            UInt8 => mm!(UInt8Array),
            _ => return None,
        };
        if let (Some(lo), Some(hi)) = (lo, hi) {
            range = Some(match range {
                Some((l, h)) => (l.min(lo), h.max(hi)),
                None => (lo, hi),
            });
        }
    }
    Some(range)
}

/// Stata storage for an integer column: the smallest of byte/int/long whose
/// non-missing span holds every value, else double. The upper bounds stop
/// short of the type maximum because Stata reserves the top codes for `.`
/// and `.a`-`.z`; the lower bounds exclude the two's-complement minimum.
fn dta_int_storage(range: Option<(i128, i128)>) -> readstat_type_t {
    const SPANS: [(readstat_type_t, i128, i128); 3] = [
        (T_INT8, -127, 100),
        (T_INT16, -32_767, 32_740),
        (T_INT32, -2_147_483_647, 2_147_483_620),
    ];
    let Some((lo, hi)) = range else {
        return T_INT8;
    };
    SPANS
        .iter()
        .find(|(_, min, max)| *min <= lo && hi <= *max)
        .map_or(T_DOUBLE, |(t, _, _)| *t)
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
    value_labels: Option<&HashMap<String, HashMap<String, String>>>,
) -> Result<()> {
    if batches.is_empty() {
        let _ = File::create(out_path)?;
        return Ok(());
    }

    let writer = unsafe { readstat_writer_init() };
    if writer.is_null() {
        return Err(anyhow!("readstat_writer_init() failed"));
    }
    // Frees the writer on every exit path (including early `?` returns).
    let _writer_guard = WriterGuard(writer);
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
        is_str_col[j] =
            is_text_dt(dt) || matches!(dt, DataType::Dictionary(_, v) if is_text_dt(v.as_ref()));
    }

    let mut col_types: Vec<readstat_type_t> = Vec::with_capacity(ncols);
    let mut rvars: Vec<*const readstat_variable_t> = Vec::with_capacity(ncols);
    let mut _keep_names: Vec<CString> = Vec::with_capacity(ncols);
    let mut _keep_label_sets: Vec<(*const readstat_label_set_t, Vec<CString>)> = Vec::new();

    // Define variables
    for (j, field) in schema.fields().iter().enumerate() {
        let mut typ = match int_col_range(batches, j) {
            Some(range) => dta_int_storage(range),
            None => T_DOUBLE,
        };
        let mut width: usize = 0;

        if is_str_col[j] {
            let stats = str_stats[j].unwrap_or(StringColStats {
                max_len: 1,
                has_nul: false,
            });

            let needs_strl = (stats.max_len as i32) > strl_threshold;

            if needs_strl {
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

        // Value labels. Stata attaches a *named* label set to a variable, so
        // the set is named after the column, matching Stata's own
        // `label values v106 v106` convention. Do not decorate the name (the
        // SAV writer uses "{col}_labels"): dta 113-117 give the name 33 bytes,
        // and a longer one is truncated by ReadStat without an error.
        if let Some(map) = value_labels
            && let Some(labels) = map.get(field.name())
            && !labels.is_empty()
        {
            if is_str_col[j] {
                return Err(anyhow!(
                    "column '{}' holds strings; Stata value labels apply only to \
                     numeric variables",
                    field.name()
                ));
            }

            // Sorted so the emitted table is byte-stable across runs.
            // ReadStat re-sorts by value for dta >= 117 but not below.
            let mut entries: Vec<(f64, &str)> = Vec::with_capacity(labels.len());
            for (value, label) in labels {
                let code: f64 = value.parse().map_err(|_| {
                    anyhow!(
                        "value label code {:?} on column '{}' is not numeric",
                        value,
                        field.name()
                    )
                })?;
                entries.push((code, label.as_str()));
            }
            entries.sort_by(|a, b| a.0.total_cmp(&b.0));

            let c_set_name = CString::new(field.name().as_str())?;
            let label_set =
                unsafe { readstat_add_label_set(writer, T_DOUBLE, c_set_name.as_ptr()) };
            if label_set.is_null() {
                return Err(anyhow!(
                    "readstat_add_label_set failed for '{}'",
                    field.name()
                ));
            }

            let mut c_labels = Vec::with_capacity(entries.len());
            for (code, label) in entries {
                let c_label = CString::new(label).map_err(|_| {
                    anyhow!(
                        "value label {:?} on column '{}' contains an embedded NUL",
                        label,
                        field.name()
                    )
                })?;
                unsafe {
                    readstat_label_double_value(label_set, code, c_label.as_ptr());
                }
                c_labels.push(c_label);
            }

            unsafe {
                readstat_variable_set_label_set(var, label_set);
            }
            c_labels.push(c_set_name);
            _keep_label_sets.push((label_set, c_labels));
        }

        _keep_names.push(cname);
        col_types.push(typ);
        rvars.push(var);
    }

    let mut outfile = File::create(Path::new(out_path))?;
    let total_rows: i64 = batches.iter().map(|b| b.num_rows() as i64).sum();
    let row_count = total_rows
        .try_into()
        .map_err(|_| anyhow!("row count {total_rows} exceeds platform limit"))?;
    unsafe {
        let rc =
            readstat_begin_writing_dta(writer, &mut outfile as *mut File as *mut c_void, row_count);
        if rc != 0 {
            return Err(anyhow!("readstat_begin_writing_dta failed with rc={}", rc));
        }
    }

    for b in batches {
        for i in 0..b.num_rows() {
            unsafe {
                let rc = readstat_begin_row(writer);
                if rc != 0 {
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
                                        return Err(anyhow!(
                                            "insert_string_value failed with rc={}",
                                            rc
                                        ));
                                    }
                                }
                                Err(_) => {
                                    let rc = readstat_insert_missing_value(writer, rvars[j]);
                                    if rc != 0 {
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
                                return Err(anyhow!(
                                    "insert_missing_value (null) failed with rc={}",
                                    rc
                                ));
                            }
                        }
                    }
                } else if col_types[j] != T_DOUBLE {
                    // The storage type was chosen from this column's range, so
                    // the narrowing casts below cannot truncate.
                    let rc = match as_i64_opt(arr.as_ref(), i) {
                        Some(v) => unsafe {
                            match col_types[j] {
                                T_INT8 => readstat_insert_int8_value(writer, rvars[j], v as i8),
                                T_INT16 => readstat_insert_int16_value(writer, rvars[j], v as i16),
                                _ => readstat_insert_int32_value(writer, rvars[j], v as i32),
                            }
                        },
                        None => unsafe { readstat_insert_missing_value(writer, rvars[j]) },
                    };
                    if rc != 0 {
                        return Err(anyhow!(
                            "insert integer value into '{}' failed with rc={}",
                            schema.field(j).name(),
                            rc
                        ));
                    }
                } else {
                    if let Some(v) = as_f64_opt(arr.as_ref(), i) {
                        unsafe {
                            let rc = readstat_insert_double_value(writer, rvars[j], v);
                            if rc != 0 {
                                return Err(anyhow!("insert_double_value failed with rc={}", rc));
                            }
                        }
                    } else {
                        unsafe {
                            let rc = readstat_insert_missing_value(writer, rvars[j]);
                            if rc != 0 {
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
                    return Err(anyhow!("readstat_end_row failed with rc={}", rc));
                }
            }
        }
    }

    unsafe {
        let rc = readstat_end_writing(writer);
        if rc != 0 {
            return Err(anyhow!("readstat_end_writing failed with rc={}", rc));
        }
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
    value_labels_json=None,
    strl_threshold=2045,
    _user_missing_json=None
))]
pub fn df_write_dta_file(
    ipc_bytes: Bound<'_, PyBytes>,
    out_path: &str,
    version: i32,
    file_label: Option<&str>,
    var_labels_json: Option<&str>,
    value_labels_json: Option<&str>,
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

    // JSON object keys are always strings, so integer codes arrive as e.g.
    // {"v106": {"0": "None"}} and are parsed back to f64 by the writer.
    let value_labels: Option<HashMap<String, HashMap<String, String>>> =
        if let Some(js) = value_labels_json {
            Some(serde_json::from_str(js).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "value_labels_json must be a JSON object of {{col: {{code: label}}}}: {e}"
                ))
            })?)
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
        value_labels.as_ref(),
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("df_write_dta_file: {}", e)))
}
