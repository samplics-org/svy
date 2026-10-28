use arrow::array::{ArrayRef, Float64Builder, Int64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use arrow::record_batch::RecordBatch;
use serde::Serialize;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::Arc;

use readstat_sys::{
    readstat_double_value, readstat_get_file_label, readstat_get_row_count, readstat_int32_value,
    readstat_measure_e_READSTAT_MEASURE_NOMINAL as MEASURE_NOMINAL,
    readstat_measure_e_READSTAT_MEASURE_ORDINAL as MEASURE_ORDINAL,
    readstat_measure_e_READSTAT_MEASURE_SCALE as MEASURE_SCALE, readstat_metadata_t,
    readstat_string_value, readstat_type_class_e_READSTAT_TYPE_CLASS_STRING as TCLASS_STRING,
    readstat_type_e_READSTAT_TYPE_FLOAT as T_FLOAT, readstat_type_e_READSTAT_TYPE_INT8 as T_INT8,
    readstat_type_e_READSTAT_TYPE_INT16 as T_INT16, readstat_type_e_READSTAT_TYPE_INT32 as T_INT32,
    readstat_type_e_READSTAT_TYPE_STRING as T_STRING,
    readstat_type_e_READSTAT_TYPE_STRING_REF as T_STRING_REF, readstat_value_is_system_missing,
    readstat_value_is_tagged_missing, readstat_value_t, readstat_value_tag,
    readstat_value_type as rs_value_type, readstat_variable_get_format,
    readstat_variable_get_index, readstat_variable_get_label, readstat_variable_get_measure,
    readstat_variable_get_missing_range_hi, readstat_variable_get_missing_range_lo,
    readstat_variable_get_missing_ranges_count, readstat_variable_get_name,
    readstat_variable_get_type, readstat_variable_get_type_class, readstat_variable_t,
};

use crate::errors::{Error, Op, Result, readstat_message};

pub(crate) const HANDLER_OK: c_int = 0;

/// Sets the iconv input encoding and a row limit backing up n_max; keep the returned CString alive through the parse.
pub(crate) fn configure_parser(
    parser: &ParserGuard,
    encoding: Option<&str>,
    rows_skip: usize,
    n_max: Option<usize>,
) -> Result<Option<std::ffi::CString>> {
    let mut keep_enc = None;
    if let Some(enc) = encoding {
        let c = std::ffi::CString::new(enc)
            .map_err(|_| Error::Invalid(format!("encoding {enc:?} contains an embedded NUL")))?;
        // SAFETY: `parser` is live; ReadStat keeps this pointer, so `c` goes back to the caller.
        let rc =
            unsafe { readstat_sys::readstat_set_file_character_encoding(parser.0, c.as_ptr()) };
        if rc != readstat_sys::readstat_error_e_READSTAT_OK {
            return Err(Error::UnsupportedEncoding {
                encoding: enc.to_string(),
                detail: readstat_message(rc),
            });
        }
        keep_enc = Some(c);
    }
    if let Some(limit) = n_max.and_then(|nm| row_limit(rows_skip, nm)) {
        // SAFETY: `parser` is live.
        unsafe { readstat_sys::readstat_set_row_limit(parser.0, limit) };
    }
    Ok(keep_enc)
}

/// `None` for no limit: zero, or past a C `long` (32 bits on Windows), which no file can hold.
fn row_limit(rows_skip: usize, n_max: usize) -> Option<::std::os::raw::c_long> {
    let limit = rows_skip.saturating_add(n_max);
    if limit == 0 {
        return None;
    }
    limit.try_into().ok()
}
pub(crate) const HANDLER_ABORT: c_int = 1;

const DEFAULT_CAPACITY: usize = 8192;
const STRING_CAPACITY_MULTIPLIER: usize = 16;
/// Caps builder pre-allocation only: the header's row count is untrusted and may claim billions.
const MAX_PREALLOC_ROWS: usize = 65_536;

/// ---------- Metadata we ship back to Python ----------

#[derive(Serialize, Clone)]
pub struct UserMissing {
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub values: Vec<MissingValue>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub range: Option<Vec<f64>>,
}

/// A discrete user-missing code: a number, or a string variable's text.
#[derive(Serialize, Clone)]
#[serde(untagged)]
pub enum MissingValue {
    Num(f64),
    Str(String),
}

#[derive(Serialize, Clone)]
pub struct VarMeta {
    pub name: String,
    pub label: Option<String>,
    pub label_set: Option<String>,
    pub fmt: Option<String>,
    pub kind: String,
    /// `None` when undeclared (always in Stata); SPSS defaults numerics to "scale", so only nominal/ordinal are deliberate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measure: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_missing: Option<UserMissing>,
}

#[derive(Serialize)]
pub struct ValueLabels {
    pub set_name: String,
    pub mapping: BTreeMap<String, String>,
}

#[derive(Serialize)]
pub struct MissingRule {
    pub var: String,
    pub discrete: Vec<String>,
    pub ranges: Vec<(String, String)>,
}

#[derive(Serialize)]
pub struct MetaOut {
    pub file_label: Option<String>,
    pub vars: Vec<VarMeta>,
    pub value_labels: Vec<ValueLabels>,
    pub user_missing: Vec<MissingRule>,
    pub n_rows: usize,
    pub tagged_missings: Vec<TaggedSpec>,
    pub notes: Vec<String>,
    /// Invalid UTF-8 was replaced by U+FFFD: usually a legacy code page needing an explicit `encoding=`.
    pub had_invalid_utf8: bool,
    /// Where the reader knows it (Stata: explicit, detected, or the format's fixed UTF-8).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    /// Why the SAS format catalog gave no value labels; the data still read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog_error: Option<String>,
}

#[derive(serde::Serialize)]
pub struct TaggedSpec {
    pub col: String,
    pub rows: Vec<usize>,
    pub tags: Vec<String>,
}

/// ---------- Parse context ----------
pub(crate) struct ColBuilders {
    pub(crate) kind: ColKind,
    pub(crate) name: String,
    pub(crate) label: Option<String>,
    pub(crate) label_set: Option<String>,
    pub(crate) fmt: Option<String>,
    pub(crate) measure: Option<String>,
    pub(crate) user_missing: Option<UserMissing>,
    pub(crate) sb: Option<StringBuilder>,
    pub(crate) fb: Option<Float64Builder>,
    pub(crate) ib: Option<Int64Builder>,
}

/// Stata byte/int/long all widen to Int64: arithmetic on narrow polars types overflows silently.
pub(crate) enum ColKind {
    Str,
    F64 { float32: bool },
    I64 { storage: &'static str },
}

pub(crate) struct ParseCtx {
    pub(crate) cols: Vec<ColBuilders>,
    /// By ReadStat variable index, not name: duplicate names in a crafted file must not share a column.
    pub(crate) var_cols: Vec<Option<usize>>,
    pub(crate) cols_skip: Option<HashMap<String, ()>>,
    pub(crate) rows_skip: usize,
    pub(crate) n_max: Option<usize>,
    pub(crate) n_rows_seen: usize,
    pub(crate) n_rows_emitted: usize,
    /// Rows count once, on their first delivered value, whichever columns are kept.
    pub(crate) last_counted_row: Option<usize>,
    pub(crate) had_invalid_utf8: bool,
    /// Needs ISO-8859-1 parser input, one char per byte, so `decode` can recover the raw bytes.
    pub(crate) lossy_utf8: bool,
    pub(crate) label_sets: HashMap<String, BTreeMap<String, String>>,
    pub(crate) file_label: Option<String>,
    pub(crate) last_err: Option<String>,
    pub(crate) tagged: HashMap<String, (Vec<usize>, Vec<String>)>,
    pub(crate) notes: Vec<String>,
    pub(crate) detect_tagged: bool,
    pub(crate) row_capacity: Option<usize>,
    /// A panic caught in a callback (unwinding into C aborts the process); check it after `readstat_parse_*`.
    pub(crate) panic_err: Option<String>,
}

impl ParseCtx {
    /// Decode a string handed back by readstat, flagging any lossy replacement.
    pub(crate) fn decode<'a>(&mut self, cs: &'a CStr) -> Cow<'a, str> {
        if self.lossy_utf8 {
            let s = cs.to_string_lossy();
            if s.is_ascii() {
                return s;
            }
            let bytes: Vec<u8> = s.chars().map(|c| c as u32 as u8).collect();
            return match String::from_utf8(bytes) {
                Ok(v) => Cow::Owned(v),
                Err(e) => {
                    self.had_invalid_utf8 = true;
                    Cow::Owned(String::from_utf8_lossy(e.as_bytes()).into_owned())
                }
            };
        }
        match cs.to_str() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => {
                self.had_invalid_utf8 = true;
                cs.to_string_lossy()
            }
        }
    }
}

/// ISO-8859-1 is byte-transparent, so the lossy path gets the raw bytes to decode itself.
pub(crate) fn input_encoding(encoding: Option<&str>, lossy_utf8: bool) -> Option<&str> {
    if lossy_utf8 {
        Some("ISO-8859-1")
    } else {
        encoding
    }
}

pub(crate) fn c_path(path: &str) -> Result<std::ffi::CString> {
    std::ffi::CString::new(path)
        .map_err(|_| Error::Invalid(format!("path {path:?} contains an embedded NUL")))
}

/// A reader's return value: the Arrow IPC bytes and the metadata as JSON.
pub(crate) fn to_python(
    py: pyo3::Python<'_>,
    ipc: &[u8],
    meta: &MetaOut,
    format: &str,
    path: &str,
) -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, String)> {
    let meta_json = serde_json::to_string(meta).map_err(|e| {
        Error::Internal(format!("metadata JSON: {e}")).into_py(Op::Read, format, path)
    })?;
    Ok((
        pyo3::types::PyBytes::new(py, ipc).into_any().unbind(),
        meta_json,
    ))
}

/// `USER_ABORT` is the `n_max` stop; a caught panic aborts too, so it is checked first.
pub(crate) fn check_parse(
    rc: readstat_sys::readstat_error_t,
    ctx: &mut ParseCtx,
    path: &str,
    encoding: Option<&str>,
) -> Result<()> {
    if let Some(msg) = ctx.panic_err.take() {
        return Err(Error::Internal(format!(
            "internal error in readstat callback: {msg}"
        )));
    }
    if rc == readstat_sys::readstat_error_e_READSTAT_OK
        || rc == readstat_sys::readstat_error_e_READSTAT_ERROR_USER_ABORT
    {
        return Ok(());
    }
    let mut msg = readstat_message(rc);
    if let Some(err) = ctx.last_err.take() {
        msg = format!("{msg}: {err}");
    }
    Err(match rc {
        readstat_sys::readstat_error_e_READSTAT_ERROR_OPEN => match std::fs::File::open(path) {
            Err(e) => Error::Io(e),
            Ok(_) => Error::Parse(msg),
        },
        readstat_sys::readstat_error_e_READSTAT_ERROR_CONVERT_BAD_STRING => Error::BadString(msg),
        // An encoding iconv lacks is the caller's input, not the file's fault.
        readstat_sys::readstat_error_e_READSTAT_ERROR_UNSUPPORTED_CHARSET => match encoding {
            Some(enc) => Error::UnsupportedEncoding {
                encoding: enc.to_string(),
                detail: msg,
            },
            None => Error::Parse(msg),
        },
        _ => Error::Parse(msg),
    })
}

#[inline(always)]
fn safe_capacity(maybe: Option<usize>) -> usize {
    match maybe {
        Some(n) if n > 0 => n.min(MAX_PREALLOC_ROWS),
        _ => DEFAULT_CAPACITY,
    }
}

/// ---------- Helpers on builders ----------
// Called from FFI callbacks, so a kind/builder mismatch is skipped rather than panicking.
impl ColBuilders {
    #[inline(always)]
    pub(crate) fn push_missing(&mut self) {
        match self.kind {
            ColKind::Str => {
                if let Some(sb) = self.sb.as_mut() {
                    sb.append_null();
                }
            }
            ColKind::F64 { .. } => {
                if let Some(fb) = self.fb.as_mut() {
                    fb.append_null();
                }
            }
            ColKind::I64 { .. } => {
                if let Some(ib) = self.ib.as_mut() {
                    ib.append_null();
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn push_str(&mut self, s: &str) {
        match self.kind {
            ColKind::Str => {
                if let Some(sb) = self.sb.as_mut() {
                    sb.append_value(s);
                }
            }
            ColKind::F64 { .. } => {
                if let Some(fb) = self.fb.as_mut() {
                    fb.append_null();
                }
            }
            ColKind::I64 { .. } => {
                if let Some(ib) = self.ib.as_mut() {
                    ib.append_null();
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn push_f64(&mut self, v: f64) {
        match self.kind {
            ColKind::F64 { .. } => {
                if let Some(fb) = self.fb.as_mut() {
                    fb.append_value(v);
                }
            }
            ColKind::I64 { .. } => {
                if let Some(ib) = self.ib.as_mut() {
                    ib.append_value(v as i64);
                }
            }
            ColKind::Str => {
                if let Some(sb) = self.sb.as_mut() {
                    sb.append_value(format!("{v}"));
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn push_i64(&mut self, v: i64) {
        match self.kind {
            ColKind::I64 { .. } => {
                if let Some(ib) = self.ib.as_mut() {
                    ib.append_value(v);
                }
            }
            _ => self.push_f64(v as f64),
        }
    }
}

// ---------- Common callbacks ----------

fn describe_panic(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Turns a callback panic into `panic_err` and a parse abort; unwinding into C would abort the process.
/// A crafted file can trigger one: Arrow panics past 2 GiB of strings in one column.
///
/// # Safety
///
/// `ctx` must be a live `ParseCtx` nothing else touches during the call; `readstat_parse_*`
/// hands it to one callback at a time.
#[inline]
unsafe fn guard_cb(ctx: *mut c_void, f: impl FnOnce(&mut ParseCtx) -> c_int) -> c_int {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the caller guarantees `ctx` is a live, unaliased `ParseCtx`.
        let rctx = unsafe { &mut *(ctx as *mut ParseCtx) };
        f(rctx)
    })) {
        Ok(rc) => rc,
        Err(p) => {
            // SAFETY: as above; the panicking closure and its borrow have been unwound.
            let rctx = unsafe { &mut *(ctx as *mut ParseCtx) };
            rctx.panic_err = Some(describe_panic(p.as_ref()));
            HANDLER_ABORT
        }
    }
}

/// # Safety
///
/// Called only by ReadStat: `message` is null or a NUL-terminated string, and `ctx` is
/// as `guard_cb` requires.
pub(crate) unsafe extern "C" fn on_error_cb(message: *const c_char, ctx: *mut c_void) {
    if message.is_null() || ctx.is_null() {
        return;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; `message` is non-null and NUL-terminated.
    unsafe {
        guard_cb(ctx, |rctx| {
            let msg = CStr::from_ptr(message).to_string_lossy().into_owned();
            rctx.last_err = Some(msg);
            HANDLER_OK
        })
    };
}

/// # Safety
///
/// Called only by ReadStat: `metadata` is null or live for the call, and `ctx` is as
/// `guard_cb` requires.
pub(crate) unsafe extern "C" fn on_metadata_cb(
    metadata: *mut readstat_metadata_t,
    ctx: *mut c_void,
) -> c_int {
    if metadata.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; `metadata` is non-null and live.
    unsafe { guard_cb(ctx, |rctx| on_metadata_impl(metadata, rctx)) }
}

/// # Safety
///
/// `metadata` must be non-null and live for the call.
unsafe fn on_metadata_impl(metadata: *mut readstat_metadata_t, rctx: &mut ParseCtx) -> c_int {
    // SAFETY: `metadata` is live (caller's guarantee).
    let label_ptr = unsafe { readstat_get_file_label(metadata) };
    if !label_ptr.is_null() {
        // SAFETY: ReadStat's metadata strings are NUL-terminated and live for the call.
        let label = rctx
            .decode(unsafe { CStr::from_ptr(label_ptr) })
            .into_owned();
        rctx.file_label = if label.trim().is_empty() {
            None
        } else {
            Some(label.trim().to_string())
        };
    }

    // SAFETY: `metadata` is live (caller's guarantee).
    let nrows = unsafe { readstat_get_row_count(metadata) };
    if nrows > 0 {
        rctx.row_capacity = Some(nrows as usize);
    }

    HANDLER_OK
}

/// # Safety
///
/// Called only by ReadStat: `var` is null or live for the call, `label_set_name` is null
/// or NUL-terminated, and `ctx` is as `guard_cb` requires.
pub(crate) unsafe extern "C" fn on_variable_cb(
    index: c_int,
    var: *mut readstat_variable_t,
    label_set_name: *const c_char,
    ctx: *mut c_void,
) -> c_int {
    if var.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; `var` is non-null and live.
    unsafe {
        guard_cb(ctx, |rctx| {
            on_variable_impl(index, var, label_set_name, rctx)
        })
    }
}

/// # Safety
///
/// `var` must be non-null and live for the call; `label_set_name` null or NUL-terminated.
unsafe fn on_variable_impl(
    index: c_int,
    var: *mut readstat_variable_t,
    label_set_name: *const c_char,
    rctx: &mut ParseCtx,
) -> c_int {
    let name = {
        // SAFETY: `var` is live (caller's guarantee).
        let p = unsafe { readstat_variable_get_name(var) };
        if p.is_null() {
            format!("V{index}")
        } else {
            // SAFETY: ReadStat's variable strings are NUL-terminated and live for the call.
            rctx.decode(unsafe { CStr::from_ptr(p) }).trim().to_string()
        }
    };

    // SAFETY: `var` is live (caller's guarantee).
    let Ok(var_index) = usize::try_from(unsafe { readstat_variable_get_index(var) }) else {
        return HANDLER_OK;
    };
    if rctx.var_cols.len() <= var_index {
        rctx.var_cols.resize(var_index + 1, None);
    }

    if let Some(skip) = &rctx.cols_skip
        && skip.contains_key(&name)
    {
        return HANDLER_OK;
    }

    let cap = safe_capacity(rctx.row_capacity);

    let label = {
        // SAFETY: `var` is live (caller's guarantee).
        let p = unsafe { readstat_variable_get_label(var) };
        if p.is_null() {
            None
        } else {
            // SAFETY: ReadStat's variable strings are NUL-terminated and live for the call.
            let s = rctx.decode(unsafe { CStr::from_ptr(p) }).into_owned();
            let st = s.trim();
            if st.is_empty() {
                None
            } else {
                Some(st.to_string())
            }
        }
    };

    let fmt = {
        // SAFETY: `var` is live (caller's guarantee).
        let p = unsafe { readstat_variable_get_format(var) };
        if p.is_null() {
            None
        } else {
            // SAFETY: ReadStat's variable strings are NUL-terminated and live for the call.
            let s = rctx.decode(unsafe { CStr::from_ptr(p) }).into_owned();
            let st = s.trim();
            if st.is_empty() {
                None
            } else {
                Some(st.to_string())
            }
        }
    };

    // Stata always reports UNKNOWN; None keeps "not declared" apart from "scale".
    // SAFETY: `var` is live (caller's guarantee).
    let measure = match unsafe { readstat_variable_get_measure(var) } {
        m if m == MEASURE_NOMINAL => Some("nominal".to_string()),
        m if m == MEASURE_ORDINAL => Some("ordinal".to_string()),
        m if m == MEASURE_SCALE => Some("scale".to_string()),
        _ => None,
    };

    // Only Stata stores integers; SAS and SPSS numerics are always doubles.
    // SAFETY: `var` is live (caller's guarantee).
    let kind = if unsafe { readstat_variable_get_type_class(var) } == TCLASS_STRING {
        ColKind::Str
    } else {
        // SAFETY: `var` is live (caller's guarantee).
        match unsafe { readstat_variable_get_type(var) } {
            t if t == T_INT8 => ColKind::I64 { storage: "int8" },
            t if t == T_INT16 => ColKind::I64 { storage: "int16" },
            t if t == T_INT32 => ColKind::I64 { storage: "int32" },
            t => ColKind::F64 {
                float32: t == T_FLOAT,
            },
        }
    };

    let label_set = if label_set_name.is_null() {
        None
    } else {
        // SAFETY: non-null, so NUL-terminated and live for the call (caller's guarantee).
        let s = rctx
            .decode(unsafe { CStr::from_ptr(label_set_name) })
            .into_owned();
        let st = s.trim();
        if st.is_empty() {
            None
        } else {
            Some(st.to_string())
        }
    };

    let user_missing = {
        // SAFETY: `var` is live (caller's guarantee).
        let missing_count = unsafe { readstat_variable_get_missing_ranges_count(var) };
        if missing_count > 0 {
            let mut values = Vec::new();
            let mut range: Option<Vec<f64>> = None;

            // SAFETY: `var` is live (caller's guarantee).
            let is_string = unsafe { readstat_variable_get_type_class(var) } == TCLASS_STRING;

            for i in 0..missing_count {
                // SAFETY: `var` is live and `i` is below its range count.
                let (lo_val, hi_val) = unsafe {
                    (
                        readstat_variable_get_missing_range_lo(var, i),
                        readstat_variable_get_missing_range_hi(var, i),
                    )
                };

                if is_string {
                    // SAV has no string ranges, so `lo_val` is the code.
                    // SAFETY: a string variable's missing value is a string whose text lives as long as `var`.
                    let sp = unsafe { readstat_string_value(lo_val) };
                    if !sp.is_null() {
                        // SAFETY: non-null ReadStat strings are NUL-terminated.
                        let code = rctx.decode(unsafe { CStr::from_ptr(sp) });
                        // SAV pads codes to 8 bytes; the data values carry no padding.
                        values.push(MissingValue::Str(code.trim_end().to_string()));
                    }
                } else {
                    // SAFETY: a numeric value is read by value; no pointer is followed.
                    let (lo, hi) =
                        unsafe { (readstat_double_value(lo_val), readstat_double_value(hi_val)) };
                    if lo.is_nan() || hi.is_nan() {
                        continue;
                    }

                    if (lo - hi).abs() < 1e-10 {
                        values.push(MissingValue::Num(lo));
                    } else {
                        range = Some(vec![lo, hi]);
                    }
                }
            }

            if !values.is_empty() || range.is_some() {
                Some(UserMissing { values, range })
            } else {
                None
            }
        } else {
            None
        }
    };

    let col = match kind {
        ColKind::Str => ColBuilders {
            kind,
            name: name.clone(),
            label,
            label_set,
            fmt,
            measure,
            user_missing,
            sb: Some(StringBuilder::with_capacity(
                cap,
                cap * STRING_CAPACITY_MULTIPLIER,
            )),
            fb: None,
            ib: None,
        },
        ColKind::F64 { .. } => ColBuilders {
            kind,
            name: name.clone(),
            label,
            label_set,
            fmt,
            measure,
            user_missing,
            sb: None,
            fb: Some(Float64Builder::with_capacity(cap)),
            ib: None,
        },
        ColKind::I64 { .. } => ColBuilders {
            kind,
            name: name.clone(),
            label,
            label_set,
            fmt,
            measure,
            user_missing,
            sb: None,
            fb: None,
            ib: Some(Int64Builder::with_capacity(cap)),
        },
    };

    rctx.var_cols[var_index] = Some(rctx.cols.len());
    rctx.cols.push(col);
    HANDLER_OK
}

/// # Safety
///
/// Called only by ReadStat: `var` is null or live for the call, `value` is live for the
/// call, and `ctx` is as `guard_cb` requires.
pub(crate) unsafe extern "C" fn on_value_cb(
    row: c_int,
    var: *mut readstat_variable_t,
    value: readstat_value_t,
    ctx: *mut c_void,
) -> c_int {
    if var.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; `var` is non-null and live.
    unsafe { guard_cb(ctx, |rctx| on_value_impl(row, var, value, rctx)) }
}

/// # Safety
///
/// `var` must be non-null and live for the call, and so must any string `value` holds.
unsafe fn on_value_impl(
    row: c_int,
    var: *mut readstat_variable_t,
    value: readstat_value_t,
    rctx: &mut ParseCtx,
) -> c_int {
    // Defensive: a negative row index would wrap to a huge usize below.
    if row < 0 {
        return HANDLER_OK;
    }
    let row_us = row as usize;
    rctx.n_rows_seen = rctx.n_rows_seen.max(row_us + 1);

    if row_us < rctx.rows_skip {
        return HANDLER_OK;
    }

    if let Some(nm) = rctx.n_max {
        // First disallowed row is rows_skip + n_max; n_max == 0 emits no rows.
        if row_us >= rctx.rows_skip.saturating_add(nm) {
            return HANDLER_ABORT;
        }
    }

    // Counted before the column lookup, so skipped columns cannot hide a row.
    if rctx.last_counted_row != Some(row_us) {
        rctx.last_counted_row = Some(row_us);
        rctx.n_rows_emitted += 1;
    }

    // SAFETY: `var` is live (caller's guarantee).
    let idx = match usize::try_from(unsafe { readstat_variable_get_index(var) })
        .ok()
        .and_then(|i| rctx.var_cols.get(i).copied().flatten())
    {
        Some(i) => i,
        None => return HANDLER_OK,
    };

    // SAFETY: these read `value`'s tag and type fields; no pointer is followed.
    let (is_tagged, is_missing, vt) = unsafe {
        (
            rctx.detect_tagged && readstat_value_is_tagged_missing(value) != 0,
            readstat_value_is_system_missing(value) != 0,
            rs_value_type(value),
        )
    };
    let is_str = vt == T_STRING || vt == T_STRING_REF;

    // Decoded before borrowing the column builder (decode needs &mut ctx).
    let str_val: Option<Cow<'_, str>> = if !is_tagged && !is_missing && is_str {
        // SAFETY: `value` is a string, so this reads its pointer field.
        let sp = unsafe { readstat_string_value(value) };
        if sp.is_null() {
            None
        } else {
            // SAFETY: ReadStat's string values are NUL-terminated and live for the call.
            Some(rctx.decode(unsafe { CStr::from_ptr(sp) }))
        }
    } else {
        None
    };

    let col = match rctx.cols.get_mut(idx) {
        Some(c) => c,
        None => return HANDLER_OK,
    };

    if is_tagged {
        // SAFETY: reads `value`'s tag field; no pointer is followed.
        let tag_ch = unsafe { readstat_value_tag(value) } as u8 as char;
        let (rows, tags) = rctx
            .tagged
            .entry(col.name.clone())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        rows.push(row as usize);
        tags.push(tag_ch.to_string());
        col.push_missing();
    } else if is_missing {
        col.push_missing();
    } else if is_str {
        match str_val {
            Some(s) => col.push_str(&s),
            None => col.push_missing(),
        }
    } else if vt == T_INT8 || vt == T_INT16 || vt == T_INT32 {
        // SAFETY: a numeric value is read by value; no pointer is followed.
        col.push_i64(unsafe { readstat_int32_value(value) } as i64);
    } else {
        // SAFETY: a numeric value is read by value; no pointer is followed.
        let d = unsafe { readstat_double_value(value) };
        col.push_f64(d);
    }

    HANDLER_OK
}

/// # Safety
///
/// Called only by ReadStat: `note` is null or NUL-terminated, and `ctx` is as `guard_cb`
/// requires.
pub(crate) unsafe extern "C" fn on_note_cb(
    _note_index: c_int,
    note: *const c_char,
    ctx: *mut c_void,
) -> c_int {
    if note.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; `note` is non-null and NUL-terminated.
    unsafe {
        guard_cb(ctx, |rctx| {
            let s = rctx.decode(CStr::from_ptr(note)).into_owned();
            rctx.notes.push(s);
            HANDLER_OK
        })
    }
}

/// # Safety
///
/// Called only by ReadStat: `set_name` and `label` are null or NUL-terminated, any string
/// `val` holds is live for the call, and `ctx` is as `guard_cb` requires.
pub(crate) unsafe extern "C" fn on_value_label_cb(
    set_name: *const c_char,
    val: readstat_value_t,
    label: *const c_char,
    ctx: *mut c_void,
) -> c_int {
    if ctx.is_null() {
        return HANDLER_OK;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; the other arguments are ReadStat's.
    unsafe { guard_cb(ctx, |rctx| on_value_label_impl(set_name, val, label, rctx)) }
}

/// # Safety
///
/// `set_name` and `label` must be null or NUL-terminated, and any string `val` holds
/// must be live for the call.
unsafe fn on_value_label_impl(
    set_name: *const c_char,
    val: readstat_value_t,
    label: *const c_char,
    rctx: &mut ParseCtx,
) -> c_int {
    let set = if set_name.is_null() {
        "__default__".to_string()
    } else {
        // SAFETY: non-null, so NUL-terminated (caller's guarantee).
        rctx.decode(unsafe { CStr::from_ptr(set_name) })
            .trim()
            .to_string()
    };
    let lab = if label.is_null() {
        String::new()
    } else {
        // SAFETY: non-null, so NUL-terminated (caller's guarantee).
        rctx.decode(unsafe { CStr::from_ptr(label) }).into_owned()
    };

    // SAFETY: these read `val`'s missing flag and type; no pointer is followed.
    let (is_missing, vt) = unsafe {
        (
            readstat_value_is_system_missing(val) != 0,
            rs_value_type(val),
        )
    };
    let key = if is_missing {
        "".to_string()
    } else if vt == T_STRING || vt == T_STRING_REF {
        // SAFETY: `val` is a string, so this reads its pointer field.
        let sp = unsafe { readstat_string_value(val) };
        if sp.is_null() {
            String::new()
        } else {
            // SAFETY: non-null and NUL-terminated, live for the call (caller's guarantee).
            rctx.decode(unsafe { CStr::from_ptr(sp) }).into_owned()
        }
    } else {
        // SAFETY: a numeric value is read by value; no pointer is followed.
        format!("{}", unsafe { readstat_double_value(val) })
    };

    let map = rctx.label_sets.entry(set).or_default();
    map.insert(key, lab);
    HANDLER_OK
}

pub(crate) fn finalize_to_ipc(mut ctx: ParseCtx) -> Result<(Vec<u8>, MetaOut)> {
    let internal = |what: &str| Error::Internal(format!("{what} builder missing"));
    let arrow = |e: arrow::error::ArrowError| Error::Internal(format!("Arrow IPC: {e}"));

    {
        let mut seen = HashSet::with_capacity(ctx.cols.len());
        if let Some(dup) = ctx.cols.iter().find(|c| !seen.insert(c.name.as_str())) {
            return Err(Error::Parse(format!(
                "the file has more than one variable named {:?}",
                dup.name
            )));
        }
    }

    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    let mut vars_meta = Vec::with_capacity(ctx.cols.len());

    for mut col in ctx.cols.drain(..) {
        let mut fmeta: HashMap<String, String> = HashMap::new();
        if let Some(lbl) = col.label.as_ref() {
            fmeta.insert("label".to_string(), lbl.clone());
        }
        if let Some(set) = col.label_set.as_ref() {
            fmeta.insert("label_set".to_string(), set.clone());
        }
        if let Some(fmt) = col.fmt.as_ref() {
            fmeta.insert("format".to_string(), fmt.clone());
        }

        match col.kind {
            ColKind::Str => {
                let arr =
                    Arc::new(col.sb.take().ok_or_else(|| internal("string"))?.finish()) as ArrayRef;
                let mut field = Field::new(&col.name, DataType::Utf8, true);
                if !fmeta.is_empty() {
                    field = field.with_metadata(fmeta);
                }
                fields.push(field);
                arrays.push(arr);
                vars_meta.push(VarMeta {
                    name: col.name,
                    label: col.label,
                    label_set: col.label_set,
                    fmt: col.fmt,
                    kind: "string".into(),
                    measure: col.measure,
                    user_missing: col.user_missing,
                });
            }
            ColKind::F64 { float32 } => {
                let arr =
                    Arc::new(col.fb.take().ok_or_else(|| internal("float"))?.finish()) as ArrayRef;
                let mut field = Field::new(&col.name, DataType::Float64, true);
                if !fmeta.is_empty() {
                    field = field.with_metadata(fmeta);
                }
                fields.push(field);
                arrays.push(arr);
                vars_meta.push(VarMeta {
                    name: col.name,
                    label: col.label,
                    label_set: col.label_set,
                    fmt: col.fmt,
                    kind: if float32 { "float" } else { "double" }.into(),
                    measure: col.measure,
                    user_missing: col.user_missing,
                });
            }
            ColKind::I64 { storage } => {
                let arr = Arc::new(col.ib.take().ok_or_else(|| internal("integer"))?.finish())
                    as ArrayRef;
                let mut field = Field::new(&col.name, DataType::Int64, true);
                if !fmeta.is_empty() {
                    field = field.with_metadata(fmeta);
                }
                fields.push(field);
                arrays.push(arr);
                vars_meta.push(VarMeta {
                    name: col.name,
                    label: col.label,
                    label_set: col.label_set,
                    fmt: col.fmt,
                    kind: storage.into(),
                    measure: col.measure,
                    user_missing: col.user_missing,
                });
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays).map_err(arrow)?;

    // Uncompressed: faster to write.
    let write_options = IpcWriteOptions::default();
    let mut buf = Vec::new();
    {
        let mut w =
            FileWriter::try_new_with_options(&mut buf, &schema, write_options).map_err(arrow)?;
        w.write(&batch).map_err(arrow)?;
        w.finish().map_err(arrow)?;
    }

    let vlabels = ctx
        .label_sets
        .into_iter()
        .map(|(set_name, mapping)| ValueLabels { set_name, mapping })
        .collect::<Vec<_>>();

    let mut tagged_specs = Vec::<TaggedSpec>::new();
    for (col, (rows, tags)) in ctx.tagged.into_iter() {
        tagged_specs.push(TaggedSpec { col, rows, tags });
    }

    let meta = MetaOut {
        file_label: ctx.file_label,
        vars: vars_meta,
        value_labels: vlabels,
        user_missing: vec![],
        n_rows: ctx.n_rows_emitted,
        tagged_missings: tagged_specs,
        notes: ctx.notes,
        had_invalid_utf8: ctx.had_invalid_utf8,
        encoding: None,
        catalog_error: None,
    };

    Ok((buf, meta))
}

// ---------- Parser guard ----------

/// A live readstat parser, freed on drop so no return path leaks it.
pub(crate) struct ParserGuard(*mut readstat_sys::readstat_parser_t);

impl ParserGuard {
    pub(crate) fn new() -> Result<Self> {
        // SAFETY: no preconditions; a null return is refused below.
        let p = unsafe { readstat_sys::readstat_parser_init() };
        if p.is_null() {
            return Err(Error::Internal("readstat_parser_init() failed".into()));
        }
        Ok(Self(p))
    }

    pub(crate) fn as_ptr(&self) -> *mut readstat_sys::readstat_parser_t {
        self.0
    }
}

impl Drop for ParserGuard {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `readstat_parser_init` and is freed only here.
        unsafe { readstat_sys::readstat_parser_free(self.0) };
    }
}

// ---------- Writer guard ----------

/// Freed on drop, so an early return cannot leak the writer and everything it owns.
pub(crate) struct WriterGuard(pub(crate) *mut readstat_sys::readstat_writer_t);

impl Drop for WriterGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the guard holds the pointer `readstat_writer_init` returned, freed only here.
            unsafe { readstat_sys::readstat_writer_free(self.0) };
        }
    }
}

/// `Ok` for `READSTAT_OK`, else a write failure naming `what` with ReadStat's text.
#[inline]
pub(crate) fn write_rc(
    rc: readstat_sys::readstat_error_t,
    what: impl FnOnce() -> String,
) -> Result<()> {
    if rc == readstat_sys::readstat_error_e_READSTAT_OK {
        return Ok(());
    }
    Err(Error::Write(format!(
        "{}: {}",
        what(),
        readstat_message(rc)
    )))
}

/// Removed on drop unless `finish` ran, so a failed write leaves no truncated file.
pub(crate) struct OutputFile {
    file: Option<std::fs::File>,
    path: std::path::PathBuf,
}

impl OutputFile {
    pub(crate) fn create(path: &str) -> Result<Self> {
        Ok(Self {
            file: Some(std::fs::File::create(path)?),
            path: path.into(),
        })
    }

    /// The `ctx` pointer for `readstat_begin_writing_*`; valid while `self` lives.
    pub(crate) fn as_ctx(&mut self) -> *mut c_void {
        match self.file.as_mut() {
            Some(f) => f as *mut std::fs::File as *mut c_void,
            None => std::ptr::null_mut(),
        }
    }

    pub(crate) fn finish(mut self) {
        self.file.take();
    }
}

impl Drop for OutputFile {
    fn drop(&mut self) {
        // Closed before removal: Windows cannot delete an open file.
        if self.file.take().is_some() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// ReadStat data sink writing to the `std::fs::File` behind `ctx`.
///
/// # Safety
///
/// Called only by ReadStat: `data` is null or `len` readable bytes, and `ctx` is null or
/// the pointer `OutputFile::as_ctx` returned.
pub(crate) unsafe extern "C" fn data_writer_cb(
    data: *const c_void,
    len: usize,
    ctx: *mut c_void,
) -> isize {
    use std::io::Write;
    // ReadStat stores the result in a size_t, so -1 would pass as success; a short count is the failure.
    if data.is_null() || ctx.is_null() {
        return 0;
    }
    // SAFETY: `ctx` is the `OutputFile`'s `File`, alive and otherwise untouched while writing.
    let file = unsafe { &mut *(ctx as *mut std::fs::File) };
    // SAFETY: ReadStat passes a buffer of `len` readable bytes, valid for the call.
    let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, len) };
    match file.write_all(bytes) {
        Ok(_) => len as isize,
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_limit_is_skip_plus_n_max() {
        assert_eq!(row_limit(0, 5), Some(5));
        assert_eq!(row_limit(10, 5), Some(15));
        assert_eq!(row_limit(0, 0), None);
    }

    #[test]
    fn row_limit_past_c_long_means_no_limit() {
        let c_long_max = ::std::os::raw::c_long::MAX as usize;
        assert_eq!(row_limit(0, c_long_max), Some(::std::os::raw::c_long::MAX));
        assert_eq!(row_limit(1, c_long_max), None);
        assert_eq!(row_limit(usize::MAX, usize::MAX), None);
    }

    #[test]
    fn safe_capacity_uses_default_when_unknown() {
        assert_eq!(safe_capacity(None), DEFAULT_CAPACITY);
        assert_eq!(safe_capacity(Some(0)), DEFAULT_CAPACITY);
    }

    #[test]
    fn safe_capacity_passes_small_counts_through() {
        assert_eq!(safe_capacity(Some(1)), 1);
        assert_eq!(safe_capacity(Some(10_000)), 10_000);
        assert_eq!(safe_capacity(Some(MAX_PREALLOC_ROWS)), MAX_PREALLOC_ROWS);
    }

    #[test]
    fn safe_capacity_clamps_untrusted_huge_counts() {
        // At most ~1 MB of string data per column, not tens of GB.
        assert_eq!(safe_capacity(Some(999_999_999)), MAX_PREALLOC_ROWS);
        assert_eq!(safe_capacity(Some(usize::MAX)), MAX_PREALLOC_ROWS);
        const { assert!(MAX_PREALLOC_ROWS * STRING_CAPACITY_MULTIPLIER <= 16 * 1024 * 1024) };
    }

    #[test]
    fn describe_panic_extracts_messages() {
        let p: Box<dyn std::any::Any + Send> = Box::new("static message");
        assert_eq!(describe_panic(p.as_ref()), "static message");
        let p: Box<dyn std::any::Any + Send> = Box::new(String::from("owned message"));
        assert_eq!(describe_panic(p.as_ref()), "owned message");
        let p: Box<dyn std::any::Any + Send> = Box::new(42_u32);
        assert_eq!(describe_panic(p.as_ref()), "unknown panic");
    }

    #[test]
    fn guard_cb_converts_panic_into_abort_and_stored_error() {
        let mut ctx = ParseCtx {
            cols: Vec::new(),
            var_cols: Vec::new(),
            cols_skip: None,
            rows_skip: 0,
            n_max: None,
            n_rows_seen: 0,
            n_rows_emitted: 0,
            last_counted_row: None,
            had_invalid_utf8: false,
            lossy_utf8: false,
            label_sets: HashMap::new(),
            file_label: None,
            last_err: None,
            tagged: HashMap::new(),
            notes: Vec::new(),
            detect_tagged: false,
            row_capacity: None,
            panic_err: None,
        };
        let ctx_ptr = &mut ctx as *mut ParseCtx as *mut c_void;

        // SAFETY: `ctx_ptr` points to the live `ctx`, untouched until the call returns.
        let rc = unsafe { guard_cb(ctx_ptr, |_| panic!("boom in callback")) };
        assert_eq!(rc, HANDLER_ABORT);
        assert_eq!(ctx.panic_err.as_deref(), Some("boom in callback"));

        ctx.panic_err = None;
        // Re-derived: the write through `ctx` above invalidated the first pointer.
        let ctx_ptr = &mut ctx as *mut ParseCtx as *mut c_void;
        // SAFETY: `ctx_ptr` points to the live `ctx`, untouched until the call returns.
        let rc = unsafe { guard_cb(ctx_ptr, |_| HANDLER_OK) };
        assert_eq!(rc, HANDLER_OK);
        assert!(ctx.panic_err.is_none());
    }

    fn empty_ctx() -> ParseCtx {
        ParseCtx {
            cols: Vec::new(),
            var_cols: Vec::new(),
            cols_skip: None,
            rows_skip: 0,
            n_max: None,
            n_rows_seen: 0,
            n_rows_emitted: 0,
            last_counted_row: None,
            had_invalid_utf8: false,
            lossy_utf8: false,
            label_sets: HashMap::new(),
            file_label: None,
            last_err: None,
            tagged: HashMap::new(),
            notes: Vec::new(),
            detect_tagged: false,
            row_capacity: None,
            panic_err: None,
        }
    }

    fn f64_col(name: &str) -> ColBuilders {
        ColBuilders {
            kind: ColKind::F64 { float32: false },
            name: name.to_string(),
            label: None,
            label_set: None,
            fmt: None,
            measure: None,
            user_missing: None,
            sb: None,
            fb: Some(Float64Builder::new()),
            ib: None,
        }
    }

    use readstat_sys::{
        readstat_error_e_READSTAT_ERROR_CONVERT_BAD_STRING as RC_BAD_STRING,
        readstat_error_e_READSTAT_ERROR_OPEN as RC_OPEN,
        readstat_error_e_READSTAT_ERROR_PARSE as RC_PARSE,
        readstat_error_e_READSTAT_ERROR_UNSUPPORTED_CHARSET as RC_CHARSET,
        readstat_error_e_READSTAT_ERROR_USER_ABORT as RC_ABORT,
        readstat_error_e_READSTAT_OK as RC_OK,
    };

    #[test]
    fn check_parse_accepts_success_and_the_n_max_abort() {
        for rc in [RC_OK, RC_ABORT] {
            assert!(check_parse(rc, &mut empty_ctx(), "f.sav", None).is_ok());
        }
    }

    #[test]
    fn check_parse_reports_a_caught_panic_before_the_return_code() {
        let mut ctx = empty_ctx();
        ctx.panic_err = Some("boom".into());
        let err = check_parse(RC_OK, &mut ctx, "f.sav", None).unwrap_err();
        assert!(
            matches!(&err, Error::Internal(m) if m == "internal error in readstat callback: boom"),
            "{err:?}"
        );
        assert!(ctx.panic_err.is_none());
    }

    #[test]
    fn check_parse_appends_readstats_last_message() {
        let mut ctx = empty_ctx();
        ctx.last_err = Some("bad header".into());
        let err = check_parse(RC_PARSE, &mut ctx, "f.sav", None).unwrap_err();
        let want = "Invalid file, or file has unsupported features (rc=5): bad header";
        assert!(matches!(&err, Error::Parse(m) if m == want), "{err:?}");
    }

    #[test]
    fn check_parse_codes_an_undecodable_string() {
        let err = check_parse(RC_BAD_STRING, &mut empty_ctx(), "f.dta", None).unwrap_err();
        let want =
            "Unable to convert string to the requested encoding (invalid byte sequence) (rc=17)";
        assert!(matches!(&err, Error::BadString(m) if m == want), "{err:?}");
    }

    #[test]
    fn check_parse_blames_an_unsupported_charset_on_the_callers_encoding_only() {
        let want = "File has an unsupported character set (rc=7)";
        let err = check_parse(RC_CHARSET, &mut empty_ctx(), "f.sav", Some("nope")).unwrap_err();
        assert!(
            matches!(&err, Error::UnsupportedEncoding { encoding, detail }
                if encoding == "nope" && detail == want),
            "{err:?}"
        );
        let err = check_parse(RC_CHARSET, &mut empty_ctx(), "f.sav", None).unwrap_err();
        assert!(matches!(&err, Error::Parse(m) if m == want), "{err:?}");
    }

    #[test]
    fn check_parse_turns_an_unopenable_path_into_an_io_error() {
        let missing = std::env::temp_dir().join("svy_io_check_parse_missing.sav");
        let _ = std::fs::remove_file(&missing);
        let err =
            check_parse(RC_OPEN, &mut empty_ctx(), missing.to_str().unwrap(), None).unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound),
            "{err:?}"
        );

        // A path that opens fine leaves ReadStat's own report.
        let present = std::env::temp_dir().join("svy_io_check_parse_present.sav");
        std::fs::write(&present, b"x").unwrap();
        let err =
            check_parse(RC_OPEN, &mut empty_ctx(), present.to_str().unwrap(), None).unwrap_err();
        std::fs::remove_file(&present).unwrap();
        assert!(
            matches!(&err, Error::Parse(m) if m == "Unable to open file (rc=1)"),
            "{err:?}"
        );
    }

    #[test]
    fn finalize_rejects_two_variables_with_one_name() {
        let mut ctx = empty_ctx();
        ctx.cols = vec![f64_col("a"), f64_col("b"), f64_col("a")];
        let err = finalize_to_ipc(ctx).err().unwrap();
        assert!(
            matches!(&err, Error::Parse(m) if m == "the file has more than one variable named \"a\""),
            "{err:?}"
        );
    }

    #[test]
    fn write_rc_passes_ok_and_names_the_failed_step() {
        assert!(write_rc(RC_OK, || unreachable!()).is_ok());
        let err = write_rc(RC_PARSE, || "ending row 3".into()).unwrap_err();
        let want = "ending row 3: Invalid file, or file has unsupported features (rc=5)";
        assert!(matches!(&err, Error::Write(m) if m == want), "{err:?}");
    }

    #[test]
    fn output_file_is_removed_unless_finished() {
        let path = std::env::temp_dir().join("svy_io_output_file.bin");
        let p = path.to_str().unwrap();

        let out = OutputFile::create(p).unwrap();
        assert!(path.exists());
        drop(out);
        assert!(!path.exists());

        OutputFile::create(p).unwrap().finish();
        assert!(path.exists());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn data_writer_cb_reports_a_failed_write_as_zero_bytes() {
        let sink = |data: *const c_void, len: usize, ctx: *mut c_void| {
            // SAFETY: each call passes null or 3 readable bytes, and null or a live `File`.
            unsafe { data_writer_cb(data, len, ctx) }
        };
        let path = std::env::temp_dir().join("svy_io_data_writer_cb.bin");
        std::fs::write(&path, b"").unwrap();
        let mut read_only = std::fs::File::open(&path).unwrap();
        let ctx = &mut read_only as *mut std::fs::File as *mut c_void;
        let bytes = b"abc".as_ptr() as *const c_void;

        assert_eq!(sink(bytes, 3, ctx), 0);
        assert_eq!(sink(std::ptr::null(), 3, ctx), 0);
        assert_eq!(sink(bytes, 3, std::ptr::null_mut()), 0);

        let mut writable = std::fs::File::create(&path).unwrap();
        let ctx = &mut writable as *mut std::fs::File as *mut c_void;
        assert_eq!(sink(bytes, 3, ctx), 3);
        drop(writable);
        assert_eq!(std::fs::read(&path).unwrap(), b"abc");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn output_file_in_a_missing_directory_is_an_io_error() {
        let path = std::env::temp_dir()
            .join("svy_io_no_such_dir")
            .join("x.sav");
        let err = OutputFile::create(path.to_str().unwrap()).err().unwrap();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound),
            "{err:?}"
        );
    }
}
