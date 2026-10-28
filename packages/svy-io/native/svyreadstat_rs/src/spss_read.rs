// native/svyreadstat_rs/src/spss_read.rs
use pyo3::prelude::*;
use std::collections::HashMap;
use std::os::raw::c_void;

use readstat_sys::{
    readstat_parse_por, readstat_parse_sav, readstat_set_error_handler,
    readstat_set_metadata_handler, readstat_set_value_handler, readstat_set_value_label_handler,
    readstat_set_variable_handler,
};

use crate::core::{
    ParseCtx, ParserGuard, check_parse, configure_parser, finalize_to_ipc, on_error_cb,
    on_metadata_cb, on_value_cb, on_value_label_cb, on_variable_cb,
};
use crate::errors::{Op, Result};

/// Parse SPSS .sav file
#[pyfunction]
#[pyo3(signature = (path, encoding=None, _user_na=false, cols_skip=None, n_max=None, rows_skip=0, lossy_utf8=false))]
#[allow(clippy::too_many_arguments)]
pub fn df_parse_sav_file(
    py: Python<'_>,
    path: &str,
    encoding: Option<&str>,
    _user_na: bool,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
    lossy_utf8: bool,
) -> PyResult<(Py<PyAny>, String)> {
    // Release GIL during parsing for better Python concurrency
    let result =
        py.detach(|| parse_sav_impl(path, encoding, cols_skip, n_max, rows_skip, lossy_utf8));

    let (ipc, meta) = result.map_err(|e| e.into_py(Op::Read, "SAV", path))?;
    crate::core::to_python(py, &ipc, &meta, "SAV", path)
}

/// Parse SPSS portable (.por) file
#[pyfunction]
#[pyo3(signature = (path, encoding=None, _user_na=false, cols_skip=None, n_max=None, rows_skip=0, lossy_utf8=false))]
#[allow(clippy::too_many_arguments)]
pub fn df_parse_por_file(
    py: Python<'_>,
    path: &str,
    encoding: Option<&str>,
    _user_na: bool,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
    lossy_utf8: bool,
) -> PyResult<(Py<PyAny>, String)> {
    // Release GIL during parsing for better Python concurrency
    let result =
        py.detach(|| parse_por_impl(path, encoding, cols_skip, n_max, rows_skip, lossy_utf8));

    let (ipc, meta) = result.map_err(|e| e.into_py(Op::Read, "POR", path))?;
    crate::core::to_python(py, &ipc, &meta, "POR", path)
}

/// Internal implementation for parsing .sav files
#[inline]
fn parse_sav_impl(
    path: &str,
    encoding: Option<&str>,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
    lossy_utf8: bool,
) -> Result<(Vec<u8>, crate::core::MetaOut)> {
    // Pre-calculate skip set for O(1) lookup
    let cols_skip_map = cols_skip.map(|v| {
        let mut map = HashMap::with_capacity(v.len());
        for k in v {
            map.insert(k, ());
        }
        map
    });

    let mut ctx = ParseCtx {
        cols: Vec::with_capacity(128), // SPSS files often have many columns
        var_cols: Vec::with_capacity(128),
        cols_skip: cols_skip_map,
        rows_skip,
        n_max,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        last_counted_row: None,
        had_invalid_utf8: false,
        lossy_utf8,
        label_sets: HashMap::with_capacity(64), // Pre-allocate for value labels
        file_label: None,
        last_err: None,
        tagged: HashMap::new(), // SPSS: no tagged-missing semantics
        notes: Vec::with_capacity(8),
        detect_tagged: false, // SPSS: no tagged-missing semantics
        row_capacity: None,   // Set via on_metadata_cb
        panic_err: None,
    };

    let cpath = crate::core::c_path(path)?;
    let parser = ParserGuard::new()?;
    // SAFETY: `parser` is live and each handler matches ReadStat's callback type.
    unsafe {
        readstat_set_error_handler(parser.as_ptr(), Some(on_error_cb));
        readstat_set_metadata_handler(parser.as_ptr(), Some(on_metadata_cb));
        readstat_set_variable_handler(parser.as_ptr(), Some(on_variable_cb));
        readstat_set_value_handler(parser.as_ptr(), Some(on_value_cb));
        readstat_set_value_label_handler(parser.as_ptr(), Some(on_value_label_cb));
    }
    let enc = crate::core::input_encoding(encoding, lossy_utf8);
    let _keep_enc = configure_parser(&parser, enc, rows_skip, n_max)?;
    // SAFETY: `ctx`, the path and `_keep_enc` outlive the parse, and only the callbacks
    // touch `ctx` until it returns.
    let rc = unsafe {
        readstat_parse_sav(
            parser.as_ptr(),
            cpath.as_ptr(),
            &mut ctx as *mut _ as *mut c_void,
        )
    };
    drop(parser);
    check_parse(rc, &mut ctx, path, encoding)?;

    finalize_to_ipc(ctx)
}

/// Internal implementation for parsing .por files
#[inline]
fn parse_por_impl(
    path: &str,
    encoding: Option<&str>,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
    lossy_utf8: bool,
) -> Result<(Vec<u8>, crate::core::MetaOut)> {
    // Pre-calculate skip set for O(1) lookup
    let cols_skip_map = cols_skip.map(|v| {
        let mut map = HashMap::with_capacity(v.len());
        for k in v {
            map.insert(k, ());
        }
        map
    });

    let mut ctx = ParseCtx {
        cols: Vec::with_capacity(128), // SPSS files often have many columns
        var_cols: Vec::with_capacity(128),
        cols_skip: cols_skip_map,
        rows_skip,
        n_max,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        last_counted_row: None,
        had_invalid_utf8: false,
        lossy_utf8,
        label_sets: HashMap::with_capacity(64), // Pre-allocate for value labels
        file_label: None,
        last_err: None,
        tagged: HashMap::new(), // SPSS: no tagged-missing semantics
        notes: Vec::with_capacity(8),
        detect_tagged: false,
        row_capacity: None, // Set via on_metadata_cb
        panic_err: None,
    };

    let cpath = crate::core::c_path(path)?;
    let parser = ParserGuard::new()?;
    // SAFETY: `parser` is live and each handler matches ReadStat's callback type.
    unsafe {
        readstat_set_error_handler(parser.as_ptr(), Some(on_error_cb));
        readstat_set_metadata_handler(parser.as_ptr(), Some(on_metadata_cb));
        readstat_set_variable_handler(parser.as_ptr(), Some(on_variable_cb));
        readstat_set_value_handler(parser.as_ptr(), Some(on_value_cb));
        readstat_set_value_label_handler(parser.as_ptr(), Some(on_value_label_cb));
    }
    let enc = crate::core::input_encoding(encoding, lossy_utf8);
    let _keep_enc = configure_parser(&parser, enc, rows_skip, n_max)?;
    // SAFETY: `ctx`, the path and `_keep_enc` outlive the parse, and only the callbacks
    // touch `ctx` until it returns.
    let rc = unsafe {
        readstat_parse_por(
            parser.as_ptr(),
            cpath.as_ptr(),
            &mut ctx as *mut _ as *mut c_void,
        )
    };
    drop(parser);
    check_parse(rc, &mut ctx, path, encoding)?;

    finalize_to_ipc(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::Error;

    #[test]
    fn a_file_that_fails_to_parse_errs_even_for_zero_rows() {
        let path = std::env::temp_dir().join("svy_io_garbage_zero_rows.sav");
        std::fs::write(&path, b"not a data file".repeat(20)).unwrap();
        let got = parse_sav_impl(path.to_str().unwrap(), None, None, Some(0), 0, false);
        std::fs::remove_file(&path).unwrap();
        let want = "Invalid file, or file has unsupported features (rc=5)";
        assert!(
            matches!(&got, Err(Error::Parse(m)) if m == want),
            "{:?}",
            got.err()
        );
    }
}
