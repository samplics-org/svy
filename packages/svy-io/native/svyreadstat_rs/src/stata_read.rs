// native/svyreadstat_rs/src/stata_read.rs
use crate::core::{
    finalize_to_ipc, on_error_cb, on_metadata_cb, on_note_cb, on_value_cb, on_value_label_cb,
    on_variable_cb, ParseCtx,
};
use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use readstat_sys::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_void;

const RS_OK: readstat_error_t = readstat_error_e_READSTAT_OK;
const RS_USER_ABORT: readstat_error_t = readstat_error_e_READSTAT_ERROR_USER_ABORT;

/// Parse a Stata .dta file into Arrow IPC format
#[inline]
fn parse_dta_impl(
    data_path: &str,
    rows_skip: usize,
    n_max: Option<usize>,
    cols_skip: Option<Vec<String>>,
) -> Result<(Vec<u8>, crate::core::MetaOut)> {
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
        label_sets: HashMap::with_capacity(32), // Pre-allocate
        file_label: None,
        last_err: None,
        tagged: HashMap::with_capacity(16), // Pre-allocate
        notes: Vec::with_capacity(8),
        detect_tagged: true,
        row_capacity: None, // Filled in metadata callback
    };

    unsafe {
        let p = readstat_parser_init();
        if p.is_null() {
            return Err(anyhow!("readstat_parser_init() failed"));
        }
        readstat_set_error_handler(p, Some(on_error_cb));
        readstat_set_metadata_handler(p, Some(on_metadata_cb));
        readstat_set_variable_handler(p, Some(on_variable_cb));
        readstat_set_value_handler(p, Some(on_value_cb));
        readstat_set_value_label_handler(p, Some(on_value_label_cb));
        readstat_set_note_handler(p, Some(on_note_cb));

        let c_path = CString::new(data_path)?;
        let rc = readstat_parse_dta(p, c_path.as_ptr(), &mut ctx as *mut _ as *mut c_void);
        readstat_parser_free(p);

        let early_ok = ctx
            .n_max
            .map(|nm| ctx.n_rows_emitted >= nm)
            .unwrap_or(false);

        if rc != RS_OK && !early_ok && rc != RS_USER_ABORT {
            let msg = ctx.last_err.take().unwrap_or_else(|| format!("rc={rc}"));
            return Err(anyhow!("Failed to parse .dta: {msg}"));
        }
    }

    finalize_to_ipc(ctx)
}

#[pyfunction]
#[pyo3(signature = (data_path, cols_skip=None, n_max=None, rows_skip=0))]
pub fn df_parse_dta_file<'py>(
    py: Python<'py>,
    data_path: &str,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
) -> PyResult<(PyObject, String)> {
    // Release GIL during parsing for better Python concurrency
    let result = py.allow_threads(|| parse_dta_impl(data_path, rows_skip, n_max, cols_skip));

    let (ipc, meta) =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let meta_json = serde_json::to_string(&meta)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let pybytes = PyBytes::new_bound(py, &ipc).into_py(py);
    Ok((pybytes, meta_json))
}
