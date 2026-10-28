use crate::core::{
    ParseCtx, ParserGuard, check_parse, configure_parser, finalize_to_ipc, on_error_cb,
    on_metadata_cb, on_note_cb, on_value_cb, on_value_label_cb, on_variable_cb,
};
use crate::errors::{Error, Op, Result};
use pyo3::prelude::*;
use readstat_sys::*;
use std::collections::HashMap;
use std::os::raw::c_void;

/// Stata format code from the header; None if unreadable, leaving readstat to report the real problem.
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

/// Format 117 and older declare no encoding: strict UTF-8 is tried before Windows-1252, which would garble UTF-8 silently.
#[inline]
pub(crate) fn parse_dta_impl(
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
        encoding,
        lossy_utf8,
    ) {
        Ok((ipc, mut meta)) => {
            meta.encoding = Some(resolved.to_string());
            Ok((ipc, meta))
        }
        Err(Error::BadString(_)) if detect => {
            let (ipc, mut meta) =
                run_parse(data_path, rows_skip, n_max, cols_skip, None, None, false)?;
            meta.encoding = Some("windows-1252".to_string());
            Ok((ipc, meta))
        }
        Err(e) => Err(e),
    }
}

/// One readstat pass with the `attempt` encoding; `encoding` is the caller's.
fn run_parse(
    data_path: &str,
    rows_skip: usize,
    n_max: Option<usize>,
    cols_skip: Option<Vec<String>>,
    attempt: Option<&str>,
    encoding: Option<&str>,
    lossy_utf8: bool,
) -> Result<(Vec<u8>, crate::core::MetaOut)> {
    let mut ctx = ParseCtx {
        cols: Vec::with_capacity(64),
        var_cols: Vec::with_capacity(64),
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
        label_sets: HashMap::with_capacity(32),
        file_label: None,
        last_err: None,
        tagged: HashMap::with_capacity(16),
        notes: Vec::with_capacity(8),
        detect_tagged: true,
        row_capacity: None, // Filled in metadata callback
        panic_err: None,
    };

    let c_path = crate::core::c_path(data_path)?;
    let parser = ParserGuard::new()?;
    // SAFETY: `parser` is live and each handler matches ReadStat's callback type.
    unsafe {
        readstat_set_error_handler(parser.as_ptr(), Some(on_error_cb));
        readstat_set_metadata_handler(parser.as_ptr(), Some(on_metadata_cb));
        readstat_set_variable_handler(parser.as_ptr(), Some(on_variable_cb));
        readstat_set_value_handler(parser.as_ptr(), Some(on_value_cb));
        readstat_set_value_label_handler(parser.as_ptr(), Some(on_value_label_cb));
        readstat_set_note_handler(parser.as_ptr(), Some(on_note_cb));
    }
    let enc = crate::core::input_encoding(attempt, lossy_utf8);
    let _keep_enc = configure_parser(&parser, enc, rows_skip, n_max)?;
    // SAFETY: `ctx`, the path and `_keep_enc` outlive the parse; only the callbacks touch `ctx`.
    let rc = unsafe {
        readstat_parse_dta(
            parser.as_ptr(),
            c_path.as_ptr(),
            &mut ctx as *mut _ as *mut c_void,
        )
    };
    drop(parser);
    check_parse(rc, &mut ctx, data_path, encoding)?;

    finalize_to_ipc(ctx)
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

    let (ipc, meta) = result.map_err(|e| e.into_py(Op::Read, "DTA", data_path))?;
    crate::core::to_python(py, &ipc, &meta, "DTA", data_path)
}
