// native/svyreadstat_rs/src/stata_read.rs
use crate::core::{
    ParseCtx, finalize_to_ipc, on_error_cb, on_metadata_cb, on_note_cb, on_value_cb,
    on_value_label_cb, on_variable_cb,
};
use anyhow::{Result, anyhow};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use readstat_sys::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_void;

const RS_OK: readstat_error_t = readstat_error_e_READSTAT_OK;
const RS_USER_ABORT: readstat_error_t = readstat_error_e_READSTAT_ERROR_USER_ABORT;
const RS_BAD_STRING: readstat_error_t = readstat_error_e_READSTAT_ERROR_CONVERT_BAD_STRING;

/// Stata format code from the file header; None when the header cannot be
/// read, in which case readstat reports the real problem.
fn dta_format(data_path: &str) -> Option<u32> {
    use std::io::Read;
    const TAG: &[u8] = b"<stata_dta><header><release>";
    let mut buf = [0u8; 48];
    let n = std::fs::File::open(data_path).ok()?.read(&mut buf).ok()?;
    let head = &buf[..n];
    if let Some(rest) = head.strip_prefix(TAG) {
        let end = rest.iter().position(|b| *b == b'<')?;
        std::str::from_utf8(&rest[..end]).ok()?.parse().ok()
    } else {
        head.first().map(|b| *b as u32)
    }
}

/// Parse a Stata .dta file into Arrow IPC format.
///
/// Files of format 117 and older declare no encoding. Without an explicit
/// `encoding`, they are validated as strict UTF-8 first (a legacy code page
/// accepts nearly any byte, so it would turn UTF-8 into mojibake silently);
/// only a file that fails that check is read as readstat's Windows-1252
/// default. The resolved encoding is reported in the metadata.
#[inline]
fn parse_dta_impl(
    data_path: &str,
    rows_skip: usize,
    n_max: Option<usize>,
    cols_skip: Option<Vec<String>>,
    encoding: Option<&str>,
    lossy_utf8: bool,
) -> Result<(Vec<u8>, crate::core::MetaOut)> {
    let detect =
        encoding.is_none() && !lossy_utf8 && dta_format(data_path).is_some_and(|f| f < 118);
    let (attempt, resolved) = match encoding {
        _ if detect => (Some("UTF-8"), "utf-8"),
        Some(e) if !lossy_utf8 => (Some(e), e),
        _ => (None, "utf-8"),
    };

    match run_parse(
        data_path,
        rows_skip,
        n_max,
        cols_skip.clone(),
        attempt,
        lossy_utf8,
    ) {
        Ok((ipc, mut meta)) => {
            meta.encoding = Some(resolved.to_string());
            Ok((ipc, meta))
        }
        Err((Some(rc), _)) if detect && rc == RS_BAD_STRING => {
            let (ipc, mut meta) = run_parse(data_path, rows_skip, n_max, cols_skip, None, false)
                .map_err(|(_, e)| e)?;
            meta.encoding = Some("windows-1252".to_string());
            Ok((ipc, meta))
        }
        Err((_, e)) => Err(e),
    }
}

/// One readstat pass. A parse failure carries its readstat code so the
/// caller can tell an encoding rejection from anything else.
fn run_parse(
    data_path: &str,
    rows_skip: usize,
    n_max: Option<usize>,
    cols_skip: Option<Vec<String>>,
    encoding: Option<&str>,
    lossy_utf8: bool,
) -> std::result::Result<(Vec<u8>, crate::core::MetaOut), (Option<readstat_error_t>, anyhow::Error)>
{
    let mut ctx = ParseCtx {
        cols: Vec::with_capacity(64), // Pre-allocate for typical files
        name_to_idx: HashMap::with_capacity(64),
        cols_skip: cols_skip.map(|v| {
            let mut map = HashMap::with_capacity(v.len());
            for k in v {
                map.insert(k, ());
            }
            map
        }),
        rows_skip,
        n_max,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        last_counted_row: None,
        had_invalid_utf8: false,
        lossy_utf8,
        label_sets: HashMap::with_capacity(32), // Pre-allocate
        file_label: None,
        last_err: None,
        tagged: HashMap::with_capacity(16), // Pre-allocate
        notes: Vec::with_capacity(8),
        detect_tagged: true,
        row_capacity: None, // Filled in metadata callback
        panic_err: None,
    };

    unsafe {
        let p = readstat_parser_init();
        if p.is_null() {
            return Err((None, anyhow!("readstat_parser_init() failed")));
        }
        readstat_set_error_handler(p, Some(on_error_cb));
        readstat_set_metadata_handler(p, Some(on_metadata_cb));
        readstat_set_variable_handler(p, Some(on_variable_cb));
        readstat_set_value_handler(p, Some(on_value_cb));

        let enc = crate::core::input_encoding(encoding, lossy_utf8);
        let _keep_enc = match crate::core::configure_parser(p, enc, rows_skip, n_max) {
            Ok(k) => k,
            Err(msg) => {
                readstat_parser_free(p);
                return Err((None, pyo3::exceptions::PyValueError::new_err(msg).into()));
            }
        };
        readstat_set_value_label_handler(p, Some(on_value_label_cb));
        readstat_set_note_handler(p, Some(on_note_cb));

        let c_path = match CString::new(data_path) {
            Ok(c) => c,
            Err(e) => {
                readstat_parser_free(p);
                return Err((None, e.into()));
            }
        };
        let rc = readstat_parse_dta(p, c_path.as_ptr(), &mut ctx as *mut _ as *mut c_void);
        readstat_parser_free(p);

        // A panic caught inside a handler callback is an internal error.
        if let Some(msg) = ctx.panic_err.take() {
            return Err((None, anyhow!("internal error in readstat callback: {msg}")));
        }

        let early_ok = ctx
            .n_max
            .map(|nm| ctx.n_rows_emitted >= nm)
            .unwrap_or(false);

        if rc != RS_OK && !early_ok && rc != RS_USER_ABORT {
            let msg = crate::core::parse_failure(rc, ctx.last_err.take());
            return Err((Some(rc), anyhow!("Failed to parse .dta: {msg}")));
        }
    }

    finalize_to_ipc(ctx).map_err(|e| (None, e))
}

#[pyfunction]
#[pyo3(signature = (data_path, cols_skip=None, n_max=None, rows_skip=0, encoding=None, lossy_utf8=false))]
pub fn df_parse_dta_file<'py>(
    py: Python<'py>,
    data_path: &str,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
    encoding: Option<&str>,
    lossy_utf8: bool,
) -> PyResult<(Py<PyAny>, String)> {
    // Release GIL during parsing for better Python concurrency
    let result =
        py.detach(|| parse_dta_impl(data_path, rows_skip, n_max, cols_skip, encoding, lossy_utf8));

    let (ipc, meta) =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let meta_json = serde_json::to_string(&meta)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let pybytes = PyBytes::new(py, &ipc)
        .into_pyobject(py)
        .unwrap()
        .into_any()
        .unbind();
    Ok((pybytes, meta_json))
}
