// native/svyreadstat_rs/src/spss_read.rs
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_void;

use readstat_sys::{
    readstat_error_e_READSTAT_ERROR_USER_ABORT as RS_USER_ABORT,
    readstat_error_e_READSTAT_OK as RS_OK, readstat_parse_por, readstat_parse_sav,
    readstat_parser_free, readstat_parser_init, readstat_set_error_handler,
    readstat_set_metadata_handler, readstat_set_value_handler, readstat_set_value_label_handler,
    readstat_set_variable_handler,
};

use crate::core::{
    ParseCtx, finalize_to_ipc, on_error_cb, on_metadata_cb, on_value_cb, on_value_label_cb,
    on_variable_cb,
};

/// Parse SPSS .sav file
#[pyfunction]
#[pyo3(signature = (path, encoding=None, _user_na=false, cols_skip=None, n_max=None, rows_skip=0))]
pub fn df_parse_sav_file(
    py: Python<'_>,
    path: &str,
    encoding: Option<&str>,
    _user_na: bool,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
) -> PyResult<(Py<PyAny>, String)> {
    // Release GIL during parsing for better Python concurrency
    let result = py.detach(|| parse_sav_impl(path, encoding, cols_skip, n_max, rows_skip));

    let (ipc, meta) = result?;
    let meta_json = serde_json::to_string(&meta).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("JSON serialize metadata: {e}"))
    })?;
    let pybytes = PyBytes::new(py, &ipc)
        .into_pyobject(py)
        .unwrap()
        .into_any()
        .unbind();
    Ok((pybytes, meta_json))
}

/// Parse SPSS portable (.por) file
#[pyfunction]
#[pyo3(signature = (path, encoding=None, _user_na=false, cols_skip=None, n_max=None, rows_skip=0))]
pub fn df_parse_por_file(
    py: Python<'_>,
    path: &str,
    encoding: Option<&str>,
    _user_na: bool,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
) -> PyResult<(Py<PyAny>, String)> {
    // Release GIL during parsing for better Python concurrency
    let result = py.detach(|| parse_por_impl(path, encoding, cols_skip, n_max, rows_skip));

    let (ipc, meta) = result?;
    let meta_json = serde_json::to_string(&meta).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("JSON serialize metadata: {e}"))
    })?;
    let pybytes = PyBytes::new(py, &ipc)
        .into_pyobject(py)
        .unwrap()
        .into_any()
        .unbind();
    Ok((pybytes, meta_json))
}

/// Internal implementation for parsing .sav files
#[inline]
fn parse_sav_impl(
    path: &str,
    encoding: Option<&str>,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
) -> PyResult<(Vec<u8>, crate::core::MetaOut)> {
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
        name_to_idx: HashMap::with_capacity(128),
        cols_skip: cols_skip_map,
        rows_skip,
        n_max,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        last_counted_row: None,
        had_invalid_utf8: false,
        label_sets: HashMap::with_capacity(64), // Pre-allocate for value labels
        file_label: None,
        last_err: None,
        tagged: HashMap::new(), // SPSS: no tagged-missing semantics
        notes: Vec::with_capacity(8),
        detect_tagged: false, // SPSS: no tagged-missing semantics
        row_capacity: None,   // Set via on_metadata_cb
        panic_err: None,
    };

    unsafe {
        let p = readstat_parser_init();
        if p.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "readstat_parser_init() failed",
            ));
        }

        readstat_set_error_handler(p, Some(on_error_cb));
        readstat_set_metadata_handler(p, Some(on_metadata_cb));
        readstat_set_variable_handler(p, Some(on_variable_cb));
        readstat_set_value_handler(p, Some(on_value_cb));
        readstat_set_value_label_handler(p, Some(on_value_label_cb));

        let _keep_enc = match crate::core::configure_parser(p, encoding, rows_skip, n_max) {
            Ok(k) => k,
            Err(msg) => {
                readstat_parser_free(p);
                return Err(pyo3::exceptions::PyValueError::new_err(msg));
            }
        };

        let cpath = CString::new(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid path: {e}")))?;
        let rc = readstat_parse_sav(p, cpath.as_ptr(), &mut ctx as *mut _ as *mut c_void);
        readstat_parser_free(p);

        // A panic caught inside a handler callback is an internal error.
        if let Some(msg) = ctx.panic_err.take() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "internal error in readstat callback: {msg}"
            )));
        }

        let early_ok = ctx
            .n_max
            .map(|nm| ctx.n_rows_emitted >= nm)
            .unwrap_or(false);
        if rc != RS_OK && !early_ok && rc != RS_USER_ABORT {
            let msg = ctx.last_err.take().unwrap_or_else(|| format!("rc={rc}"));
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to parse SAV: {msg}"
            )));
        }
    }

    finalize_to_ipc(ctx)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("finalize_to_ipc: {e}")))
}

/// Internal implementation for parsing .por files
#[inline]
fn parse_por_impl(
    path: &str,
    encoding: Option<&str>,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
) -> PyResult<(Vec<u8>, crate::core::MetaOut)> {
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
        name_to_idx: HashMap::with_capacity(128),
        cols_skip: cols_skip_map,
        rows_skip,
        n_max,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        last_counted_row: None,
        had_invalid_utf8: false,
        label_sets: HashMap::with_capacity(64), // Pre-allocate for value labels
        file_label: None,
        last_err: None,
        tagged: HashMap::new(), // SPSS: no tagged-missing semantics
        notes: Vec::with_capacity(8),
        detect_tagged: false,
        row_capacity: None, // Set via on_metadata_cb
        panic_err: None,
    };

    unsafe {
        let p = readstat_parser_init();
        if p.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "readstat_parser_init() failed",
            ));
        }

        readstat_set_error_handler(p, Some(on_error_cb));
        readstat_set_metadata_handler(p, Some(on_metadata_cb));
        readstat_set_variable_handler(p, Some(on_variable_cb));
        readstat_set_value_handler(p, Some(on_value_cb));
        readstat_set_value_label_handler(p, Some(on_value_label_cb));

        let _keep_enc = match crate::core::configure_parser(p, encoding, rows_skip, n_max) {
            Ok(k) => k,
            Err(msg) => {
                readstat_parser_free(p);
                return Err(pyo3::exceptions::PyValueError::new_err(msg));
            }
        };

        let cpath = CString::new(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid path: {e}")))?;
        let rc = readstat_parse_por(p, cpath.as_ptr(), &mut ctx as *mut _ as *mut c_void);
        readstat_parser_free(p);

        // A panic caught inside a handler callback is an internal error.
        if let Some(msg) = ctx.panic_err.take() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "internal error in readstat callback: {msg}"
            )));
        }

        let early_ok = ctx
            .n_max
            .map(|nm| ctx.n_rows_emitted >= nm)
            .unwrap_or(false);
        if rc != RS_OK && !early_ok && rc != RS_USER_ABORT {
            let msg = ctx.last_err.take().unwrap_or_else(|| format!("rc={rc}"));
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to parse POR: {msg}"
            )));
        }
    }

    finalize_to_ipc(ctx)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("finalize_to_ipc: {e}")))
}
