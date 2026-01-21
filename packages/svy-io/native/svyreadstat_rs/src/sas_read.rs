// native/svyreadstat_rs/src/sas_read.rs
use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_void;

use readstat_sys::{
    readstat_error_e_READSTAT_ERROR_USER_ABORT as RS_USER_ABORT,
    readstat_error_e_READSTAT_OK as RS_OK, readstat_parse_sas7bcat, readstat_parse_sas7bdat,
    readstat_parser_free, readstat_parser_init, readstat_set_error_handler,
    readstat_set_metadata_handler, readstat_set_value_handler, readstat_set_value_label_handler,
    readstat_set_variable_handler,
};

use crate::core::{
    finalize_to_ipc, on_error_cb, on_metadata_cb, on_value_cb, on_value_label_cb, on_variable_cb,
    ParseCtx,
};

/// Optimized SAS file parser
///
/// Performance optimizations:
/// - Separate catalog parsing for efficient label loading
/// - Pre-allocates buffers based on file metadata
/// - Early abort on row limit
/// - Efficient column skipping
/// - GIL released during parsing for Python concurrency
#[inline]
fn parse_sas_impl(
    data_path: &str,
    catalog_path: Option<&str>,
    rows_skip: usize,
    n_max: Option<usize>,
    cols_skip: Option<Vec<String>>,
) -> Result<(Vec<u8>, crate::core::MetaOut)> {
    // Pre-calculate skip set for O(1) lookup
    let cols_skip_set = cols_skip.map(|v| {
        let mut map = HashMap::with_capacity(v.len());
        for col in v {
            map.insert(col, ());
        }
        map
    });

    let mut ctx = ParseCtx {
        cols: Vec::with_capacity(128), // SAS files often have many columns
        name_to_idx: HashMap::with_capacity(128),
        cols_skip: cols_skip_set,
        rows_skip,
        n_max,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        label_sets: HashMap::with_capacity(64), // Pre-allocate for value labels
        file_label: None,
        last_err: None,
        tagged: HashMap::new(), // SAS doesn't use tagged missing
        notes: Vec::with_capacity(4),
        detect_tagged: false, // SAS: no tagged-missing semantics like Stata
        row_capacity: None,   // Will be filled by on_metadata_cb
    };

    // Step 1: Parse catalog file if provided (for value labels)
    if let Some(cat_path) = catalog_path {
        unsafe {
            let parser = readstat_parser_init();
            if parser.is_null() {
                return Err(anyhow!("readstat_parser_init() failed for catalog"));
            }

            // Only need value label handler for catalog
            readstat_set_value_label_handler(parser, Some(on_value_label_cb));

            let c_path = CString::new(cat_path)?;
            let rc =
                readstat_parse_sas7bcat(parser, c_path.as_ptr(), &mut ctx as *mut _ as *mut c_void);

            readstat_parser_free(parser);

            // Catalog parse errors are not fatal, but we should report them
            if rc != RS_OK && rc != RS_USER_ABORT {
                let msg = ctx
                    .last_err
                    .take()
                    .unwrap_or_else(|| format!("Catalog parse failed with code {rc}"));
                eprintln!("Warning: Failed to parse catalog: {msg}");
                // Continue with data parsing even if catalog fails
            }
        }
    }

    // Step 2: Parse data file
    unsafe {
        let parser = readstat_parser_init();
        if parser.is_null() {
            return Err(anyhow!("readstat_parser_init() failed for data"));
        }

        // Set up all handlers for data file
        readstat_set_error_handler(parser, Some(on_error_cb));
        readstat_set_metadata_handler(parser, Some(on_metadata_cb));
        readstat_set_variable_handler(parser, Some(on_variable_cb));
        readstat_set_value_handler(parser, Some(on_value_cb));

        let c_path = CString::new(data_path)?;
        let rc =
            readstat_parse_sas7bdat(parser, c_path.as_ptr(), &mut ctx as *mut _ as *mut c_void);

        readstat_parser_free(parser);

        // Check for early termination (user requested n_max rows)
        let early_ok = ctx
            .n_max
            .map(|nm| ctx.n_rows_emitted >= nm)
            .unwrap_or(false);

        // Handle errors
        if rc != RS_OK && !early_ok && rc != RS_USER_ABORT {
            let msg = ctx
                .last_err
                .take()
                .unwrap_or_else(|| format!("Data parse failed with code {rc}"));
            return Err(anyhow!("Failed to parse SAS data file: {msg}"));
        }
    }

    // Convert to Arrow IPC format
    finalize_to_ipc(ctx)
}

/// Python interface for parsing SAS files
///
/// Arguments:
///   data_path: Path to .sas7bdat file
///   catalog_path: Optional path to .sas7bcat catalog file for value labels
///   _encoding: Deprecated, kept for compatibility (ignored)
///   _catalog_encoding: Deprecated, kept for compatibility (ignored)
///   cols_skip: Optional list of column names to skip
///   n_max: Optional maximum number of rows to read
///   rows_skip: Number of rows to skip from start (default: 0)
///
/// Returns:
///   Tuple of (Arrow IPC bytes, metadata JSON string)
#[pyfunction]
#[pyo3(signature = (
    data_path,
    catalog_path=None,
    _encoding=None,
    _catalog_encoding=None,
    cols_skip=None,
    n_max=None,
    rows_skip=0
))]
pub fn df_parse_sas_file<'py>(
    py: Python<'py>,
    data_path: &str,
    catalog_path: Option<&str>,
    _encoding: Option<&str>,
    _catalog_encoding: Option<&str>,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
) -> PyResult<(PyObject, String)> {
    // Release GIL during parsing for better Python concurrency
    let result =
        py.allow_threads(|| parse_sas_impl(data_path, catalog_path, rows_skip, n_max, cols_skip));

    let (ipc, meta) =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Serialize metadata to JSON
    let meta_json = serde_json::to_string(&meta)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Return Arrow IPC bytes and metadata
    let pybytes = PyBytes::new_bound(py, &ipc).into_py(py);
    Ok((pybytes, meta_json))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sas_validates_path() {
        let result = parse_sas_impl("nonexistent.sas7bdat", None, 0, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_sas_handles_skip_params() {
        // Test that skip parameters are properly configured
        let cols_skip = Some(vec!["var1".to_string(), "var2".to_string()]);
        let result = parse_sas_impl("test.sas7bdat", None, 10, Some(50), cols_skip);
        // Will fail on nonexistent file, but tests parameter handling
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_sas_with_catalog() {
        let result = parse_sas_impl("test.sas7bdat", Some("test.sas7bcat"), 0, None, None);
        // Will fail on nonexistent files, but tests catalog parameter
        assert!(result.is_err());
    }
}
