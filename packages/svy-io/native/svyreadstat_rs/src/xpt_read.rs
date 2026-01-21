use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_void;

use readstat_sys::{
    readstat_error_e_READSTAT_ERROR_USER_ABORT as RS_USER_ABORT,
    readstat_error_e_READSTAT_OK as RS_OK, readstat_parse_xport, readstat_parser_free,
    readstat_parser_init, readstat_set_error_handler, readstat_set_metadata_handler,
    readstat_set_value_handler, readstat_set_value_label_handler, readstat_set_variable_handler,
};

use crate::core::{
    finalize_to_ipc, on_error_cb, on_metadata_cb, on_value_cb, on_value_label_cb, on_variable_cb,
    ParseCtx,
};

fn parse_xpt_impl(
    data_path: &str,
    rows_skip: usize,
    n_max: Option<usize>,
    cols_skip: Option<Vec<String>>,
) -> Result<(Vec<u8>, crate::core::MetaOut)> {
    let mut ctx = ParseCtx {
        cols: Vec::new(),
        name_to_idx: HashMap::new(),
        cols_skip: cols_skip.map(|v| v.into_iter().map(|k| (k, ())).collect()),
        rows_skip,
        n_max,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        label_sets: HashMap::new(),
        file_label: None,
        last_err: None,
        tagged: HashMap::new(),
        notes: Vec::new(),
        detect_tagged: false, // XPT: no tagged-missing semantics
        row_capacity: None,   // set via on_metadata_cb
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

        let rc = readstat_parse_xport(
            p,
            CString::new(data_path)?.as_ptr(),
            &mut ctx as *mut _ as *mut c_void,
        );
        readstat_parser_free(p);

        let early_ok = ctx
            .n_max
            .map(|nm| ctx.n_rows_emitted >= nm)
            .unwrap_or(false);
        if rc != RS_OK && !early_ok && rc != RS_USER_ABORT {
            let msg = ctx.last_err.take().unwrap_or_else(|| format!("rc={rc}"));
            return Err(anyhow!("Failed to parse XPT: {msg}"));
        }
    }

    finalize_to_ipc(ctx)
}

#[pyfunction]
#[pyo3(signature = (data_path, n_max=None, rows_skip=0, cols_skip=None))]
pub fn df_parse_xpt_file<'py>(
    py: Python<'py>,
    data_path: &str,
    n_max: Option<usize>,
    rows_skip: usize,
    cols_skip: Option<Vec<String>>,
) -> PyResult<(PyObject, String)> {
    let (ipc, meta) = parse_xpt_impl(data_path, rows_skip, n_max, cols_skip)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let meta_json = serde_json::to_string(&meta).unwrap();
    let pybytes = PyBytes::new_bound(py, &ipc).into_py(py);
    Ok((pybytes, meta_json))
}
