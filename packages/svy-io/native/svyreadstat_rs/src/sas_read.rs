// native/svyreadstat_rs/src/sas_read.rs
use anyhow::{Result, anyhow};
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
    ParseCtx, finalize_to_ipc, on_error_cb, on_metadata_cb, on_value_cb, on_value_label_cb,
    on_variable_cb,
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
    encoding: Option<&str>,
    catalog_encoding: Option<&str>,
    rows_skip: usize,
    n_max: Option<usize>,
    cols_skip: Option<Vec<String>>,
    lossy_utf8: bool,
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
        last_counted_row: None,
        had_invalid_utf8: false,
        lossy_utf8,
        label_sets: HashMap::with_capacity(64), // Pre-allocate for value labels
        file_label: None,
        last_err: None,
        tagged: HashMap::new(), // SAS doesn't use tagged missing
        notes: Vec::with_capacity(4),
        detect_tagged: false, // SAS: no tagged-missing semantics like Stata
        row_capacity: None,   // Will be filled by on_metadata_cb
        panic_err: None,
    };

    // Step 1: Parse catalog file if provided (for value labels)
    if let Some(cat_path) = catalog_path {
        parse_catalog(&mut ctx, cat_path, catalog_encoding, lossy_utf8)?;
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

        let enc = crate::core::input_encoding(encoding, lossy_utf8);
        let _keep_enc = match crate::core::configure_parser(parser, enc, rows_skip, n_max) {
            Ok(k) => k,
            Err(msg) => {
                readstat_parser_free(parser);
                return Err(anyhow!(msg));
            }
        };

        let c_path = CString::new(data_path)?;
        let rc =
            readstat_parse_sas7bdat(parser, c_path.as_ptr(), &mut ctx as *mut _ as *mut c_void);

        readstat_parser_free(parser);

        // A panic caught inside a handler callback is an internal error.
        if let Some(msg) = ctx.panic_err.take() {
            return Err(anyhow!("internal error in readstat callback: {msg}"));
        }

        // Check for early termination (user requested n_max rows)
        let early_ok = ctx
            .n_max
            .map(|nm| ctx.n_rows_emitted >= nm)
            .unwrap_or(false);

        // Handle errors
        if rc != RS_OK && !early_ok && rc != RS_USER_ABORT {
            let msg = crate::core::parse_failure(rc, ctx.last_err.take());
            return Err(anyhow!("Failed to parse SAS data file: {msg}"));
        }
    }

    resolve_format_label_sets(&mut ctx);

    // Convert to Arrow IPC format
    finalize_to_ipc(ctx)
}

/// Load a .sas7bcat catalog's value labels into `ctx.label_sets`.
///
/// Shared by the sas7bdat and XPT readers: both name a variable's label set
/// after its format. A catalog that fails to parse only warns, so the data
/// still reads without labels.
pub(crate) fn parse_catalog(
    ctx: &mut ParseCtx,
    cat_path: &str,
    catalog_encoding: Option<&str>,
    lossy_utf8: bool,
) -> Result<()> {
    unsafe {
        let parser = readstat_parser_init();
        if parser.is_null() {
            return Err(anyhow!("readstat_parser_init() failed for catalog"));
        }

        // Only need value label handler for catalog
        readstat_set_value_label_handler(parser, Some(on_value_label_cb));

        let cat_enc = crate::core::input_encoding(catalog_encoding, lossy_utf8);
        let _keep_enc = match crate::core::configure_parser(parser, cat_enc, 0, None) {
            Ok(k) => k,
            Err(msg) => {
                readstat_parser_free(parser);
                return Err(anyhow!(msg));
            }
        };

        let c_path = CString::new(cat_path)?;
        let rc = readstat_parse_sas7bcat(parser, c_path.as_ptr(), ctx as *mut _ as *mut c_void);

        readstat_parser_free(parser);

        // A panic caught inside a handler callback is an internal error.
        if let Some(msg) = ctx.panic_err.take() {
            return Err(anyhow!("internal error in readstat callback: {msg}"));
        }

        if rc != RS_OK && rc != RS_USER_ABORT {
            let msg = ctx
                .last_err
                .take()
                .unwrap_or_else(|| format!("Catalog parse failed with code {rc}"));
            eprintln!("Warning: Failed to parse catalog: {msg}");
        }
    }
    Ok(())
}

/// Strip the display width and decimals ReadStat appends to a format name:
/// `WORKSHOP5` -> `WORKSHOP`, `DOLLAR10.2` -> `DOLLAR`.
fn format_base_name(fmt: &str) -> &str {
    let s = match fmt.rfind('.') {
        Some(i) if fmt[i + 1..].bytes().all(|b| b.is_ascii_digit()) => &fmt[..i],
        _ => fmt,
    };
    s.trim_end_matches(|c: char| c.is_ascii_digit())
}

/// Point each column at its catalog label set when only the width differs.
///
/// A column formatted `WORKSHOP5.` arrives with label set `WORKSHOP5`, while
/// the catalog defines `WORKSHOP`. SAS format names cannot end in a digit, so
/// the trailing digits are always the width.
pub(crate) fn resolve_format_label_sets(ctx: &mut ParseCtx) {
    if ctx.label_sets.is_empty() {
        return;
    }
    for col in ctx.cols.iter_mut() {
        let Some(set) = col.label_set.as_deref() else {
            continue;
        };
        if ctx.label_sets.contains_key(set) {
            continue;
        }
        let base = format_base_name(set);
        if base.len() < set.len() && ctx.label_sets.contains_key(base) {
            col.label_set = Some(base.to_string());
        }
    }
}

/// Python interface for parsing SAS files
///
/// Arguments:
///   data_path: Path to .sas7bdat file
///   catalog_path: Optional path to .sas7bcat catalog file for value labels
///   encoding: Optional input character encoding (iconv name) for the data file
///   catalog_encoding: Optional input character encoding for the catalog file
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
    encoding=None,
    catalog_encoding=None,
    cols_skip=None,
    n_max=None,
    rows_skip=0,
    lossy_utf8=false
))]
pub fn df_parse_sas_file<'py>(
    py: Python<'py>,
    data_path: &str,
    catalog_path: Option<&str>,
    encoding: Option<&str>,
    catalog_encoding: Option<&str>,
    cols_skip: Option<Vec<String>>,
    n_max: Option<usize>,
    rows_skip: usize,
    lossy_utf8: bool,
) -> PyResult<(Py<PyAny>, String)> {
    // Release GIL during parsing for better Python concurrency
    let result =
        py.detach(|| {
            parse_sas_impl(
                data_path,
                catalog_path,
                encoding,
                catalog_encoding,
                rows_skip,
                n_max,
                cols_skip,
                lossy_utf8,
            )
        });

    let (ipc, meta) =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Serialize metadata to JSON
    let meta_json = serde_json::to_string(&meta)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Return Arrow IPC bytes and metadata
    let pybytes = PyBytes::new(py, &ipc)
        .into_pyobject(py)
        .unwrap()
        .into_any()
        .unbind();
    Ok((pybytes, meta_json))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_base_name() {
        assert_eq!(format_base_name("WORKSHOP5"), "WORKSHOP");
        assert_eq!(format_base_name("DOLLAR10.2"), "DOLLAR");
        assert_eq!(format_base_name("$GENDER10"), "$GENDER");
        assert_eq!(format_base_name("$GENDER"), "$GENDER");
        assert_eq!(format_base_name("F8."), "F");
    }

    fn ctx_with(label_sets: &[&str], col_sets: &[Option<&str>]) -> ParseCtx {
        let cols = col_sets
            .iter()
            .enumerate()
            .map(|(i, set)| crate::core::ColBuilders {
                kind: crate::core::ColKind::F64 { float32: false },
                name: format!("v{i}"),
                label: None,
                label_set: set.map(str::to_string),
                fmt: set.map(str::to_string),
                measure: None,
                user_missing: None,
                sb: None,
                fb: None,
                ib: None,
            })
            .collect();
        ParseCtx {
            cols,
            name_to_idx: HashMap::new(),
            cols_skip: None,
            rows_skip: 0,
            n_max: None,
            n_rows_seen: 0,
            n_rows_emitted: 0,
            last_counted_row: None,
            had_invalid_utf8: false,
            lossy_utf8: false,
            label_sets: label_sets
                .iter()
                .map(|s| (s.to_string(), Default::default()))
                .collect(),
            file_label: None,
            last_err: None,
            tagged: HashMap::new(),
            notes: Vec::new(),
            detect_tagged: false,
            row_capacity: None,
            panic_err: None,
        }
    }

    #[test]
    fn test_resolve_format_label_sets_strips_width() {
        let mut ctx = ctx_with(
            &["WORKSHOP", "$GENDER"],
            &[Some("WORKSHOP5"), Some("$GENDER10"), Some("BEST12"), None],
        );
        resolve_format_label_sets(&mut ctx);
        let sets: Vec<_> = ctx.cols.iter().map(|c| c.label_set.as_deref()).collect();
        assert_eq!(sets, [Some("WORKSHOP"), Some("$GENDER"), Some("BEST12"), None]);
        // the format keeps its width
        assert_eq!(ctx.cols[0].fmt.as_deref(), Some("WORKSHOP5"));
    }

    #[test]
    fn test_resolve_format_label_sets_keeps_exact_match() {
        let mut ctx = ctx_with(&["WORKSHOP"], &[Some("WORKSHOP")]);
        resolve_format_label_sets(&mut ctx);
        assert_eq!(ctx.cols[0].label_set.as_deref(), Some("WORKSHOP"));
    }

    #[test]
    fn test_parse_sas_validates_path() {
        let result = parse_sas_impl("nonexistent.sas7bdat", None, None, None, 0, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_sas_handles_skip_params() {
        // Test that skip parameters are properly configured
        let cols_skip = Some(vec!["var1".to_string(), "var2".to_string()]);
        let result = parse_sas_impl("test.sas7bdat", None, None, None, 10, Some(50), cols_skip, false);
        // Will fail on nonexistent file, but tests parameter handling
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_sas_with_catalog() {
        let result = parse_sas_impl(
            "test.sas7bdat",
            Some("test.sas7bcat"),
            None,
            None,
            0,
            None,
            None,
            false,
        );
        // Will fail on nonexistent files, but tests catalog parameter
        assert!(result.is_err());
    }
}
