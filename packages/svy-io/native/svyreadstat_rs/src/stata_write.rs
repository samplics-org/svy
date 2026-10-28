use pyo3::prelude::*;
use pyo3::types::PyBytes;

use std::collections::HashMap;
use std::ffi::CString;
use std::io::Cursor;
use std::os::raw::c_char;

use arrow::array::{
    Array, AsArray, DictionaryArray, LargeStringArray, StringArray, StringViewArray,
};
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::record_batch::RecordBatch;

use readstat_sys::{
    readstat_add_label_set, readstat_add_variable, readstat_begin_row, readstat_begin_writing_dta,
    readstat_end_row, readstat_end_writing, readstat_error_t, readstat_insert_double_value,
    readstat_insert_int8_value, readstat_insert_int16_value, readstat_insert_int32_value,
    readstat_insert_missing_value, readstat_insert_string_value,
    readstat_insert_tagged_missing_value, readstat_label_double_value, readstat_label_set_t,
    readstat_set_data_writer, readstat_type_e_READSTAT_TYPE_DOUBLE as T_DOUBLE,
    readstat_type_e_READSTAT_TYPE_INT8 as T_INT8, readstat_type_e_READSTAT_TYPE_INT16 as T_INT16,
    readstat_type_e_READSTAT_TYPE_INT32 as T_INT32,
    readstat_type_e_READSTAT_TYPE_STRING as T_STRING, readstat_type_t,
    readstat_variable_set_format, readstat_variable_set_label, readstat_variable_set_label_set,
    readstat_variable_t, readstat_writer_init, readstat_writer_set_file_format_version,
    readstat_writer_set_file_label, readstat_writer_t,
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

fn as_f64_opt(a: &dyn Array, row: usize) -> Option<f64> {
    if a.is_null(row) {
        return None;
    }
    macro_rules! down {
        ($T:ty) => {
            a.as_primitive::<$T>().value(row)
        };
    }
    use DataType::*;
    match a.data_type() {
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
    }
}

const MS_PER_DAY: i64 = 86_400_000;
const UNIX_MINUS_STATA_DAYS: i64 = 3_653;
const UNIX_MINUS_STATA_MS: i64 = UNIX_MINUS_STATA_DAYS * MS_PER_DAY;

/// Stata display format for a date (`%td`) or datetime (`%tc`) column.
fn stata_temporal_format(dt: &DataType) -> Option<&'static str> {
    match dt {
        DataType::Date32 | DataType::Date64 => Some("%td"),
        DataType::Timestamp(_, _) => Some("%tc"),
        _ => None,
    }
}

/// Days (`%td`) or ms (`%tc`) since 1960-01-01, floored to Stata's millisecond; a zoned time is its UTC instant.
fn temporal_as_stata(a: &dyn Array, row: usize) -> Option<f64> {
    if a.is_null(row) {
        return None;
    }
    macro_rules! down {
        ($T:ty) => {
            a.as_primitive::<$T>().value(row)
        };
    }
    // i128 because shifting a timestamp near i64::MAX to the 1960 epoch overflows i64.
    let unix_ms: i128 = match a.data_type() {
        DataType::Date32 => {
            return Some((down!(Date32Type) as i64 + UNIX_MINUS_STATA_DAYS) as f64);
        }
        DataType::Date64 => {
            let days = down!(Date64Type).div_euclid(MS_PER_DAY);
            return Some((days + UNIX_MINUS_STATA_DAYS) as f64);
        }
        DataType::Timestamp(TimeUnit::Second, _) => down!(TimestampSecondType) as i128 * 1_000,
        DataType::Timestamp(TimeUnit::Millisecond, _) => down!(TimestampMillisecondType).into(),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            down!(TimestampMicrosecondType).div_euclid(1_000).into()
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            down!(TimestampNanosecondType).div_euclid(1_000_000).into()
        }
        _ => return None,
    };
    Some((unix_ms + UNIX_MINUS_STATA_MS as i128) as f64)
}

fn as_i64_opt(a: &dyn Array, row: usize) -> Option<i64> {
    if a.is_null(row) {
        return None;
    }
    macro_rules! down {
        ($T:ty) => {
            a.as_primitive::<$T>().value(row) as i64
        };
    }
    use DataType::*;
    match a.data_type() {
        Int64 => Some(down!(Int64Type)),
        Int32 => Some(down!(Int32Type)),
        Int16 => Some(down!(Int16Type)),
        Int8 => Some(down!(Int8Type)),
        UInt64 => Some(down!(UInt64Type)),
        UInt32 => Some(down!(UInt32Type)),
        UInt16 => Some(down!(UInt16Type)),
        UInt8 => Some(down!(UInt8Type)),
        Boolean => Some(a.as_boolean().value(row) as i64),
        _ => None,
    }
}

/// `None` when the column is not integer or boolean; `Some(None)` when it is all null.
fn int_col_range(batches: &[RecordBatch], j: usize) -> Option<Option<(i128, i128)>> {
    use arrow::compute::{max, min};
    let mut range: Option<(i128, i128)> = None;
    for b in batches {
        let a = b.column(j).as_ref();
        macro_rules! mm {
            ($T:ty) => {{
                let a = a.as_primitive::<$T>();
                (min(a).map(|v| v as i128), max(a).map(|v| v as i128))
            }};
        }
        use DataType::*;
        let (lo, hi) = match a.data_type() {
            Int64 => mm!(Int64Type),
            Int32 => mm!(Int32Type),
            Int16 => mm!(Int16Type),
            Int8 => mm!(Int8Type),
            UInt64 => mm!(UInt64Type),
            UInt32 => mm!(UInt32Type),
            UInt16 => mm!(UInt16Type),
            UInt8 => mm!(UInt8Type),
            Boolean => (Some(0), Some(1)),
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

/// The smallest of byte/int/long whose non-missing span holds every value, else double.
fn dta_int_storage(range: Option<(i128, i128)>) -> readstat_type_t {
    // Upper bounds leave the top codes to `.` and `.a`-`.z`; lower bounds skip the two's-complement minimum.
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
                    if !stats.has_nul && s.as_bytes().contains(&0) {
                        stats.has_nul = true;
                    }
                }
            }
        }
    }
    all_stats
}

/// The cells of one column that hold a tagged missing (`TaggedNA`), from Python's `user_missing_json`.
#[derive(serde::Deserialize)]
struct TaggedMissing {
    col: String,
    rows: Vec<usize>,
    tags: Vec<String>,
}

/// Row -> tag maps, one per column, after checking each spec against the frame.
fn tags_by_column(
    specs: &[TaggedMissing],
    names: &[&str],
    is_str_col: &[bool],
    n_rows: usize,
) -> Result<Vec<HashMap<usize, c_char>>> {
    let mut by_col = vec![HashMap::new(); names.len()];
    for spec in specs {
        let Some(j) = names.iter().position(|n| *n == spec.col) else {
            return Err(Error::Invalid(format!(
                "tagged missings name column '{}', which is not in the data",
                spec.col
            )));
        };
        if is_str_col[j] {
            return Err(Error::Invalid(format!(
                "column '{}' holds strings; tagged missings apply only to numeric variables",
                spec.col
            )));
        }
        if spec.rows.len() != spec.tags.len() {
            return Err(Error::Invalid(format!(
                "column '{}': {} tagged rows but {} tags",
                spec.col,
                spec.rows.len(),
                spec.tags.len()
            )));
        }
        for (&row, tag) in spec.rows.iter().zip(&spec.tags) {
            let &[b @ b'a'..=b'z'] = tag.as_bytes() else {
                return Err(Error::Invalid(format!(
                    "column '{}': tag {:?} is not one of the Stata tags a-z",
                    spec.col, tag
                )));
            };
            if row >= n_rows {
                return Err(Error::Invalid(format!(
                    "column '{}': tagged row {} is past the last row ({})",
                    spec.col, row, n_rows
                )));
            }
            by_col[j].insert(row, b as c_char);
        }
    }
    Ok(by_col)
}

/// A tagged missing (`.a`-`.z`) when the cell has a tag, else the system missing `.`.
///
/// # Safety
///
/// `writer` must be live and `var` one of its variables.
unsafe fn insert_missing(
    writer: *mut readstat_writer_t,
    var: *const readstat_variable_t,
    tag: Option<&c_char>,
) -> readstat_error_t {
    match tag {
        // SAFETY: `writer` and `var` are live (caller's guarantee).
        Some(&t) => unsafe { readstat_insert_tagged_missing_value(writer, var, t) },
        // SAFETY: `writer` and `var` are live (caller's guarantee).
        None => unsafe { readstat_insert_missing_value(writer, var) },
    }
}

/// Compared as usize: a string over 2 GiB would wrap negative as an i32.
fn exceeds_strl_threshold(max_len: usize, strl_threshold: i32) -> bool {
    match usize::try_from(strl_threshold) {
        Ok(threshold) => max_len > threshold,
        Err(_) => true,
    }
}

#[allow(clippy::too_many_arguments)]
fn write_stata_minimal(
    batches: &[RecordBatch],
    out_path: &str,
    file_label: Option<&str>,
    version_internal: i32,
    strl_threshold: i32,
    var_labels: Option<&HashMap<String, String>>,
    value_labels: Option<&HashMap<String, HashMap<String, String>>>,
    tagged_missings: &[TaggedMissing],
) -> Result<()> {
    if batches.is_empty() {
        return Err(Error::Invalid("no record batches to write".into()));
    }
    let version = u8::try_from(version_internal).map_err(|_| {
        Error::Invalid(format!(
            "Stata file format {version_internal} is out of range"
        ))
    })?;

    // SAFETY: no preconditions; a null result is checked below.
    let writer = unsafe { readstat_writer_init() };
    if writer.is_null() {
        return Err(Error::Internal("readstat_writer_init() failed".into()));
    }
    let _writer_guard = WriterGuard(writer);
    // SAFETY: `writer` is live until `_writer_guard` drops.
    unsafe {
        readstat_writer_set_file_format_version(writer, version);
        readstat_set_data_writer(writer, Some(data_writer_cb));
    }

    if let Some(lbl) = file_label {
        let c = CString::new(lbl)
            .map_err(|_| Error::Invalid(format!("file label {lbl:?} contains an embedded NUL")))?;
        // SAFETY: `writer` is live; ReadStat copies the label.
        unsafe {
            readstat_writer_set_file_label(writer, c.as_ptr());
        }
    }

    let schema = batches[0].schema();
    let ncols = schema.fields().len();

    let str_stats = compute_string_metadata(batches);
    let is_temporal_col: Vec<bool> = schema
        .fields()
        .iter()
        .map(|f| stata_temporal_format(f.data_type()).is_some())
        .collect();

    let is_str_col: Vec<bool> = (0..ncols)
        .map(|j| {
            let dt = batches[0].column(j).data_type();
            is_text_dt(dt) || matches!(dt, DataType::Dictionary(_, v) if is_text_dt(v.as_ref()))
        })
        .collect();

    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let n_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let tags = tags_by_column(tagged_missings, &names, &is_str_col, n_rows)?;

    let mut col_types: Vec<readstat_type_t> = Vec::with_capacity(ncols);
    let mut rvars: Vec<*const readstat_variable_t> = Vec::with_capacity(ncols);
    let mut _keep_names: Vec<CString> = Vec::with_capacity(ncols);
    let mut _keep_label_sets: Vec<(*const readstat_label_set_t, Vec<CString>)> = Vec::new();

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

            let needs_strl = exceeds_strl_threshold(stats.max_len, strl_threshold);

            if needs_strl {
                return Err(Error::Invalid(format!(
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
                )));
            }

            typ = T_STRING;
            width = stats.max_len.clamp(1, 2045);
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

        if let Some(fmt) = stata_temporal_format(field.data_type()) {
            let c = CString::new(fmt).expect("Stata format names hold no NUL");
            // SAFETY: `var` lives as long as `writer`; ReadStat copies the format.
            unsafe {
                readstat_variable_set_format(var, c.as_ptr());
            }
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
            unsafe {
                readstat_variable_set_label(var, c.as_ptr());
            }
        }

        // Named after the column, as Stata's `label values v106 v106`; no SAV-style "_labels"
        // suffix, since dta 113-117 cap the name at 33 bytes and ReadStat truncates silently.
        if let Some(map) = value_labels
            && let Some(labels) = map.get(field.name())
            && !labels.is_empty()
        {
            if is_str_col[j] {
                return Err(Error::Invalid(format!(
                    "column '{}' holds strings; Stata value labels apply only to \
                     numeric variables",
                    field.name()
                )));
            }

            // Sorted for byte-stable output: ReadStat sorts by value only for dta >= 117.
            let mut entries: Vec<(f64, &str)> = Vec::with_capacity(labels.len());
            for (value, label) in labels {
                let code: f64 = value.parse().map_err(|_| {
                    Error::Invalid(format!(
                        "value label code {:?} on column '{}' is not numeric",
                        value,
                        field.name()
                    ))
                })?;
                entries.push((code, label.as_str()));
            }
            entries.sort_by(|a, b| a.0.total_cmp(&b.0));

            let c_set_name = cname.clone();
            let label_set =
                // SAFETY: `writer` is live; ReadStat copies the name.
                unsafe { readstat_add_label_set(writer, T_DOUBLE, c_set_name.as_ptr()) };
            if label_set.is_null() {
                return Err(Error::Write(format!(
                    "readstat_add_label_set failed for '{}'",
                    field.name()
                )));
            }

            let mut c_labels = Vec::with_capacity(entries.len());
            for (code, label) in entries {
                let c_label = CString::new(label).map_err(|_| {
                    Error::Invalid(format!(
                        "value label {:?} on column '{}' contains an embedded NUL",
                        label,
                        field.name()
                    ))
                })?;
                // SAFETY: `label_set` is live; ReadStat copies the label.
                unsafe {
                    readstat_label_double_value(label_set, code, c_label.as_ptr());
                }
                c_labels.push(c_label);
            }

            // SAFETY: `var` and `label_set` belong to the live `writer`.
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

    let total_rows: i64 = batches.iter().map(|b| b.num_rows() as i64).sum();
    #[allow(clippy::useless_conversion)] // c_long is 32-bit on Windows
    let row_count = total_rows
        .try_into()
        .map_err(|_| Error::Invalid(format!("{total_rows} rows exceed this platform's limit")))?;
    let mut out = OutputFile::create(out_path)?;
    write_rc(
        // SAFETY: `writer` is live, and `out` outlives every write `data_writer_cb` makes into it.
        unsafe { readstat_begin_writing_dta(writer, out.as_ctx(), row_count) },
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
                } else if col_types[j] != T_DOUBLE {
                    // Storage was chosen from this column's range, so these casts cannot truncate.
                    match as_i64_opt(arr.as_ref(), i) {
                        // SAFETY: `rvars[j]` belongs to the live `writer`.
                        Some(v) => unsafe {
                            match col_types[j] {
                                T_INT8 => readstat_insert_int8_value(writer, rvars[j], v as i8),
                                T_INT16 => readstat_insert_int16_value(writer, rvars[j], v as i16),
                                _ => readstat_insert_int32_value(writer, rvars[j], v as i32),
                            }
                        },
                        // SAFETY: `rvars[j]` belongs to the live `writer`.
                        None => unsafe { insert_missing(writer, rvars[j], tags[j].get(&row)) },
                    }
                } else {
                    let v = if is_temporal_col[j] {
                        temporal_as_stata(arr.as_ref(), i)
                    } else {
                        as_f64_opt(arr.as_ref(), i)
                    };
                    match v {
                        // SAFETY: `rvars[j]` belongs to the live `writer`.
                        Some(v) => unsafe { readstat_insert_double_value(writer, rvars[j], v) },
                        // SAFETY: `rvars[j]` belongs to the live `writer`.
                        None => unsafe { insert_missing(writer, rvars[j], tags[j].get(&row)) },
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
#[pyo3(signature = (
    ipc_bytes,
    out_path,
    version,
    file_label=None,
    var_labels_json=None,
    value_labels_json=None,
    strl_threshold=2045,
    user_missing_json=None
))]
#[allow(clippy::too_many_arguments)]
pub fn df_write_dta_file(
    ipc_bytes: Bound<'_, PyBytes>,
    out_path: &str,
    version: i32,
    file_label: Option<&str>,
    var_labels_json: Option<&str>,
    value_labels_json: Option<&str>,
    strl_threshold: i32,
    user_missing_json: Option<&str>,
) -> PyResult<()> {
    write_dta_impl(
        ipc_bytes.as_bytes(),
        out_path,
        version,
        file_label,
        var_labels_json,
        value_labels_json,
        strl_threshold,
        user_missing_json,
    )
    .map_err(|e| e.into_py(Op::Write, "DTA", out_path))
}

#[allow(clippy::too_many_arguments)]
fn write_dta_impl(
    buf: &[u8],
    out_path: &str,
    version: i32,
    file_label: Option<&str>,
    var_labels_json: Option<&str>,
    value_labels_json: Option<&str>,
    strl_threshold: i32,
    user_missing_json: Option<&str>,
) -> Result<()> {
    let batches =
        ipc_to_batches(buf).map_err(|e| Error::Invalid(format!("Arrow IPC decode failed: {e}")))?;

    let var_labels: Option<HashMap<String, String>> = if let Some(js) = var_labels_json {
        Some(
            serde_json::from_str::<HashMap<String, String>>(js).map_err(|e| {
                Error::Invalid(format!(
                    "var_labels_json must be a JSON object of {{col: label}} strings: {e}"
                ))
            })?,
        )
    } else {
        None
    };

    // JSON keys are strings, so codes arrive as {"v106": {"0": "None"}} and are parsed back to f64.
    let value_labels: Option<HashMap<String, HashMap<String, String>>> =
        if let Some(js) = value_labels_json {
            Some(serde_json::from_str(js).map_err(|e| {
                Error::Invalid(format!(
                    "value_labels_json must be a JSON object of {{col: {{code: label}}}}: {e}"
                ))
            })?)
        } else {
            None
        };

    let tagged_missings: Vec<TaggedMissing> = match user_missing_json {
        Some(js) => serde_json::from_str(js).map_err(|e| {
            Error::Invalid(format!(
                "user_missing_json must be a JSON list of {{col, rows, tags}}: {e}"
            ))
        })?,
        None => Vec::new(),
    };

    write_stata_minimal(
        &batches,
        out_path,
        file_label,
        version,
        strl_threshold,
        var_labels.as_ref(),
        value_labels.as_ref(),
        &tagged_missings,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, Date32Array, Date64Array, Float64Array, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
    };
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    // 2024-01-31 and 2024-01-31 12:30:05.250, as Unix days and milliseconds.
    const UNIX_DAYS: i32 = 19_753;
    const UNIX_MS: i64 = UNIX_DAYS as i64 * MS_PER_DAY + 45_005_250;
    const STATA_DAYS: f64 = 23_406.0;
    const STATA_MS: f64 = 2_022_323_405_250.0;

    #[test]
    fn dates_and_datetimes_convert_to_days_and_ms_since_1960() {
        let date = Date32Array::from(vec![
            Some(UNIX_DAYS),
            Some(-UNIX_MINUS_STATA_DAYS as i32),
            None,
        ]);
        assert_eq!(temporal_as_stata(&date, 0), Some(STATA_DAYS));
        assert_eq!(temporal_as_stata(&date, 1), Some(0.0));
        assert_eq!(temporal_as_stata(&date, 2), None);

        let date64 = Date64Array::from(vec![UNIX_DAYS as i64 * MS_PER_DAY]);
        assert_eq!(temporal_as_stata(&date64, 0), Some(STATA_DAYS));

        let arrays: [ArrayRef; 4] = [
            Arc::new(TimestampSecondArray::from(vec![UNIX_MS / 1_000])),
            Arc::new(TimestampMillisecondArray::from(vec![UNIX_MS])),
            Arc::new(TimestampMicrosecondArray::from(vec![UNIX_MS * 1_000 + 999])),
            Arc::new(
                TimestampNanosecondArray::from(vec![UNIX_MS * 1_000_000 + 999_999])
                    .with_timezone("UTC"),
            ),
        ];
        let got: Vec<f64> = arrays
            .iter()
            .map(|a| temporal_as_stata(a.as_ref(), 0).unwrap())
            .collect();
        assert_eq!(got, [STATA_MS - 250.0, STATA_MS, STATA_MS, STATA_MS]);
    }

    #[test]
    fn sub_millisecond_times_before_1970_floor_rather_than_round_toward_zero() {
        let ts = TimestampMicrosecondArray::from(vec![-1]);
        assert_eq!(
            temporal_as_stata(&ts, 0),
            Some((UNIX_MINUS_STATA_MS - 1) as f64)
        );
    }

    #[test]
    fn written_dates_read_back_with_td_and_tc_formats() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Date32, true),
            Field::new("t", DataType::Timestamp(TimeUnit::Microsecond, None), true),
            Field::new("x", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Date32Array::from(vec![Some(UNIX_DAYS), None])),
                Arc::new(TimestampMicrosecondArray::from(vec![
                    Some(UNIX_MS * 1_000),
                    None,
                ])),
                Arc::new(Float64Array::from(vec![1.5, 2.5])),
            ],
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("svy_dta_dates_{}.dta", std::process::id()));
        let path = path.to_str().unwrap();
        write_stata_minimal(&[batch], path, None, 118, 2045, None, None, &[]).unwrap();
        let (ipc, meta) =
            crate::stata_read::parse_dta_impl(path, 0, None, None, None, false).unwrap();
        let _ = std::fs::remove_file(path);

        let fmts: Vec<Option<&str>> = meta.vars.iter().map(|v| v.fmt.as_deref()).collect();
        assert_eq!(fmts[..2], [Some("%td"), Some("%tc")]);
        assert_ne!(fmts[2], Some("%td"));

        let back = &ipc_to_batches(&ipc).unwrap()[0];
        let col = |j: usize| {
            (0..2)
                .map(|i| as_f64_opt(back.column(j).as_ref(), i))
                .collect::<Vec<_>>()
        };
        assert_eq!(col(0), [Some(STATA_DAYS), None]);
        assert_eq!(col(1), [Some(STATA_MS), None]);
        assert_eq!(col(2), [Some(1.5), Some(2.5)]);
    }

    #[test]
    fn extreme_timestamps_shift_to_1960_without_overflow() {
        let ms = TimestampMillisecondArray::from(vec![i64::MAX]);
        assert_eq!(temporal_as_stata(&ms, 0), Some(9.223372352473975e18));

        let secs = TimestampSecondArray::from(vec![i64::MAX, i64::MIN]);
        assert_eq!(temporal_as_stata(&secs, 0), Some(9.223372037170395e21));
        assert_eq!(temporal_as_stata(&secs, 1), Some(-9.223372036539157e21));
    }

    #[test]
    fn strl_threshold_compares_lengths_past_i32_max() {
        let cases = [
            (2045, 2045, false),
            (2046, 2045, true),
            (0, 0, false),
            (1, -1, true),
            (i32::MAX as usize + 1, 2045, true),
            (usize::MAX, i32::MAX, true),
        ];
        for (max_len, threshold, expected) in cases {
            assert_eq!(
                exceeds_strl_threshold(max_len, threshold),
                expected,
                "{max_len} vs {threshold}"
            );
        }
    }

    #[test]
    fn out_of_range_format_version_is_refused_before_writing() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(vec![1.5]))]).unwrap();
        let path = std::env::temp_dir().join(format!("svy_dta_version_{}.dta", std::process::id()));
        let path = path.to_str().unwrap();
        // 374 wraps to 118 as a u8, a valid format.
        for version in [374, -1] {
            let err = write_stata_minimal(
                std::slice::from_ref(&batch),
                path,
                None,
                version,
                2045,
                None,
                None,
                &[],
            )
            .unwrap_err();
            assert!(matches!(err, Error::Invalid(_)), "{err:?}");
            assert_eq!(
                err.to_string(),
                format!("Stata file format {version} is out of range")
            );
            assert!(!std::path::Path::new(path).exists());
        }
    }
}
