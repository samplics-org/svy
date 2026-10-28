use pyo3::prelude::*;
use std::collections::HashMap;
use std::os::raw::c_void;

use readstat_sys::{
    readstat_parse_xport, readstat_parser_free, readstat_parser_init, readstat_set_error_handler,
    readstat_set_metadata_handler, readstat_set_value_handler, readstat_set_value_label_handler,
    readstat_set_variable_handler,
};

use crate::core::{
    ParseCtx, check_parse, finalize_to_ipc, on_error_cb, on_metadata_cb, on_value_cb,
    on_value_label_cb, on_variable_cb,
};
use crate::errors::{Error, Op, Result};
use crate::sas_read::{parse_catalog, resolve_format_label_sets};

#[allow(clippy::too_many_arguments)]
fn parse_xpt_impl(
    data_path: &str,
    rows_skip: usize,
    n_max: Option<usize>,
    cols_skip: Option<Vec<String>>,
    encoding: Option<&str>,
    catalog_path: Option<&str>,
    catalog_encoding: Option<&str>,
    lossy_utf8: bool,
) -> Result<(Vec<u8>, crate::core::MetaOut)> {
    let mut ctx = ParseCtx {
        cols: Vec::new(),
        var_cols: Vec::new(),
        cols_skip: cols_skip.map(|v| v.into_iter().map(|k| (k, ())).collect()),
        rows_skip,
        n_max,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        last_counted_row: None,
        had_invalid_utf8: false,
        lossy_utf8,
        label_sets: HashMap::new(),
        file_label: None,
        last_err: None,
        tagged: HashMap::new(),
        notes: Vec::new(),
        detect_tagged: false, // XPT: no tagged-missing semantics
        row_capacity: None,   // set via on_metadata_cb
        panic_err: None,
    };

    let catalog_error = match catalog_path {
        Some(cat_path) => parse_catalog(&mut ctx, cat_path, catalog_encoding, lossy_utf8)?,
        None => None,
    };

    // Before the parser exists, so a bad path cannot return without freeing it.
    let c_path = crate::core::c_path(data_path)?;
    unsafe {
        let p = readstat_parser_init();
        if p.is_null() {
            return Err(Error::Internal("readstat_parser_init() failed".into()));
        }
        readstat_set_error_handler(p, Some(on_error_cb));
        readstat_set_metadata_handler(p, Some(on_metadata_cb));
        readstat_set_variable_handler(p, Some(on_variable_cb));
        readstat_set_value_handler(p, Some(on_value_cb));

        // Defensive row limit for untrusted files (n_max already aborts in
        // the value callback; this stops the C parser earlier too).
        let enc = crate::core::input_encoding(encoding, lossy_utf8);
        let _keep_enc = match crate::core::configure_parser(p, enc, rows_skip, n_max) {
            Ok(k) => k,
            Err(e) => {
                readstat_parser_free(p);
                return Err(e);
            }
        };
        readstat_set_value_label_handler(p, Some(on_value_label_cb));

        let rc = readstat_parse_xport(p, c_path.as_ptr(), &mut ctx as *mut _ as *mut c_void);
        readstat_parser_free(p);
        check_parse(rc, &mut ctx, data_path, encoding)?;
    }

    resolve_format_label_sets(&mut ctx);
    let (ipc, mut meta) = finalize_to_ipc(ctx)?;
    meta.catalog_error = catalog_error;
    Ok((ipc, meta))
}

#[pyfunction]
#[pyo3(signature = (
    data_path,
    n_max=None,
    rows_skip=0,
    cols_skip=None,
    encoding=None,
    lossy_utf8=false,
    catalog_path=None,
    catalog_encoding=None
))]
#[allow(clippy::too_many_arguments)]
pub fn df_parse_xpt_file<'py>(
    py: Python<'py>,
    data_path: &str,
    n_max: Option<usize>,
    rows_skip: usize,
    cols_skip: Option<Vec<String>>,
    encoding: Option<&str>,
    lossy_utf8: bool,
    catalog_path: Option<&str>,
    catalog_encoding: Option<&str>,
) -> PyResult<(Py<PyAny>, String)> {
    let (ipc, meta) = parse_xpt_impl(
        data_path,
        rows_skip,
        n_max,
        cols_skip,
        encoding,
        catalog_path,
        catalog_encoding,
        lossy_utf8,
    )
    .map_err(|e| e.into_py(Op::Read, "XPT", data_path))?;
    crate::core::to_python(py, &ipc, &meta, "XPT", data_path)
}
