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
    finalize_to_ipc, on_error_cb, on_metadata_cb, on_value_cb, on_value_label_cb, on_variable_cb,
    ParseCtx,
};

/// Parse SPSS .sav file
#[pyfunction]
#[pyo3(signature = (path, _encoding=None, _user_na=false, cols_skip=None, n_max=None, rows_skip=0))]
pub fn df_parse_sav_file(
    py: Python<'_>,
    path: &str,
    _encoding: Option<&str>, // SPSS SAV stores encoding in the file; unused here
    _user_na: bool,          // reserved for future: user-defined missings
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
) -> PyResult<(PyObject, String)> {
    // Build cols_skip map once
    let cols_skip_map = cols_skip.map(|v| v.into_iter().map(|k| (k, ())).collect());

    // Parse context
    let mut ctx = ParseCtx {
        cols: Vec::new(),
        name_to_idx: HashMap::new(),
        cols_skip: cols_skip_map,
        rows_skip,
        n_max,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        label_sets: HashMap::new(),
        file_label: None,
        last_err: None,
        tagged: HashMap::new(),
        notes: Vec::new(),
        detect_tagged: false, // SPSS has no Stata-style tagged missings
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

        let cpath = CString::new(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid path: {e}")))?;
        let rc = readstat_parse_sav(p, cpath.as_ptr(), &mut ctx as *mut _ as *mut c_void);
        readstat_parser_free(p);

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

    let (ipc, meta) = finalize_to_ipc(ctx)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("finalize_to_ipc: {e}")))?;
    let meta_json = serde_json::to_string(&meta).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("JSON serialize metadata: {e}"))
    })?;
    let pybytes = PyBytes::new_bound(py, &ipc).into_py(py);
    Ok((pybytes, meta_json))
}

/// Parse SPSS portable (.por) file
#[pyfunction]
#[pyo3(signature = (path, _encoding=None, _user_na=false, cols_skip=None, n_max=None, rows_skip=0))]
pub fn df_parse_por_file(
    py: Python<'_>,
    path: &str,
    _encoding: Option<&str>, // POR is ASCII-like; no encoding parameter needed
    _user_na: bool,          // reserved for future: user-defined missings
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
) -> PyResult<(PyObject, String)> {
    let cols_skip_map = cols_skip.map(|v| v.into_iter().map(|k| (k, ())).collect());

    let mut ctx = ParseCtx {
        cols: Vec::new(),
        name_to_idx: HashMap::new(),
        cols_skip: cols_skip_map,
        rows_skip,
        n_max,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        label_sets: HashMap::new(),
        file_label: None,
        last_err: None,
        tagged: HashMap::new(),
        notes: Vec::new(),
        detect_tagged: false,
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

        let cpath = CString::new(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid path: {e}")))?;
        let rc = readstat_parse_por(p, cpath.as_ptr(), &mut ctx as *mut _ as *mut c_void);
        readstat_parser_free(p);

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

    let (ipc, meta) = finalize_to_ipc(ctx)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("finalize_to_ipc: {e}")))?;
    let meta_json = serde_json::to_string(&meta).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("JSON serialize metadata: {e}"))
    })?;
    let pybytes = PyBytes::new_bound(py, &ipc).into_py(py);
    Ok((pybytes, meta_json))
}
