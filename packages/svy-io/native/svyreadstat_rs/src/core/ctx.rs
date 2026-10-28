use super::*;

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
pub(super) fn safe_capacity(maybe: Option<usize>) -> usize {
    match maybe {
        Some(n) if n > 0 => n.min(MAX_PREALLOC_ROWS),
        _ => DEFAULT_CAPACITY,
    }
}

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
