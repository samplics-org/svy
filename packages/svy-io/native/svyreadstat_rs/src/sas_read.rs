// native/svyreadstat_rs/src/sas_read.rs
use pyo3::prelude::*;
use std::collections::HashMap;
use std::os::raw::c_void;

use readstat_sys::{
    readstat_parse_sas7bcat, readstat_parse_sas7bdat, readstat_set_error_handler,
    readstat_set_metadata_handler, readstat_set_value_handler, readstat_set_value_label_handler,
    readstat_set_variable_handler,
};

use crate::core::{
    ParseCtx, ParserGuard, check_parse, configure_parser, finalize_to_ipc, on_error_cb,
    on_metadata_cb, on_value_cb, on_value_label_cb, on_variable_cb,
};
use crate::errors::{Error, Op, Result};

/// Optimized SAS file parser
///
/// Performance optimizations:
/// - Separate catalog parsing for efficient label loading
/// - Pre-allocates buffers based on file metadata
/// - Early abort on row limit
/// - Efficient column skipping
/// - GIL released during parsing for Python concurrency
#[inline]
#[allow(clippy::too_many_arguments)]
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
        var_cols: Vec::with_capacity(128),
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

    let catalog_error = match catalog_path {
        Some(cat_path) => parse_catalog(&mut ctx, cat_path, catalog_encoding, lossy_utf8)?,
        None => None,
    };

    let c_path = crate::core::c_path(data_path)?;
    let parser = ParserGuard::new()?;
    // SAFETY: `parser` is live and each handler matches ReadStat's callback type.
    unsafe {
        readstat_set_error_handler(parser.as_ptr(), Some(on_error_cb));
        readstat_set_metadata_handler(parser.as_ptr(), Some(on_metadata_cb));
        readstat_set_variable_handler(parser.as_ptr(), Some(on_variable_cb));
        readstat_set_value_handler(parser.as_ptr(), Some(on_value_cb));
    }
    let enc = crate::core::input_encoding(encoding, lossy_utf8);
    let _keep_enc = configure_parser(&parser, enc, rows_skip, n_max)?;
    // SAFETY: `ctx`, the path and `_keep_enc` outlive the parse, and only the callbacks
    // touch `ctx` until it returns.
    let rc = unsafe {
        readstat_parse_sas7bdat(
            parser.as_ptr(),
            c_path.as_ptr(),
            &mut ctx as *mut _ as *mut c_void,
        )
    };
    drop(parser);
    check_parse(rc, &mut ctx, data_path, encoding)?;

    resolve_format_label_sets(&mut ctx);

    let (ipc, mut meta) = finalize_to_ipc(ctx)?;
    meta.catalog_error = catalog_error;
    Ok((ipc, meta))
}

/// Load a .sas7bcat catalog's value labels into `ctx.label_sets`.
///
/// Shared by the sas7bdat and XPT readers: both name a variable's label set
/// after its format. A catalog that fails to parse returns why, so the data
/// still reads without labels.
pub(crate) fn parse_catalog(
    ctx: &mut ParseCtx,
    cat_path: &str,
    catalog_encoding: Option<&str>,
    lossy_utf8: bool,
) -> Result<Option<String>> {
    let c_path = crate::core::c_path(cat_path)?;
    let parser = ParserGuard::new()?;
    // SAFETY: `parser` is live and each handler matches ReadStat's callback type.
    unsafe {
        readstat_set_value_label_handler(parser.as_ptr(), Some(on_value_label_cb));
    }
    let enc = crate::core::input_encoding(catalog_encoding, lossy_utf8);
    let _keep_enc = configure_parser(&parser, enc, 0, None)?;
    // SAFETY: `ctx`, the path and `_keep_enc` outlive the parse, and only the callbacks
    // touch `ctx` until it returns.
    let rc = unsafe {
        readstat_parse_sas7bcat(
            parser.as_ptr(),
            c_path.as_ptr(),
            ctx as *mut _ as *mut c_void,
        )
    };
    drop(parser);
    match check_parse(rc, ctx, cat_path, catalog_encoding) {
        Ok(()) => Ok(None),
        Err(e @ Error::Internal(_)) => Err(e),
        Err(e) => Ok(Some(format!(
            "SAS catalog {cat_path:?} gave no value labels: {e}"
        ))),
    }
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
#[allow(clippy::too_many_arguments)]
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
    let result = py.detach(|| {
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

    let (ipc, meta) = result.map_err(|e| e.into_py(Op::Read, "SAS", data_path))?;
    crate::core::to_python(py, &ipc, &meta, "SAS", data_path)
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
            var_cols: Vec::new(),
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
        assert_eq!(
            sets,
            [Some("WORKSHOP"), Some("$GENDER"), Some("BEST12"), None]
        );
        // the format keeps its width
        assert_eq!(ctx.cols[0].fmt.as_deref(), Some("WORKSHOP5"));
    }

    #[test]
    fn test_resolve_format_label_sets_keeps_exact_match() {
        let mut ctx = ctx_with(&["WORKSHOP"], &[Some("WORKSHOP")]);
        resolve_format_label_sets(&mut ctx);
        assert_eq!(ctx.cols[0].label_set.as_deref(), Some("WORKSHOP"));
    }

    fn fixture(name: &str) -> String {
        format!("{}/../../tests/data/sas/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn test_parse_sas_missing_file_is_an_io_error() {
        let result = parse_sas_impl(
            "nonexistent.sas7bdat",
            None,
            None,
            None,
            0,
            None,
            None,
            false,
        );
        assert!(
            matches!(&result, Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound),
            "{:?}",
            result.err()
        );
    }

    // Expected values from R haven::read_sas on the same file: 8 rows, columns
    // id, workshop, gender, q1..q4.
    #[test]
    fn test_parse_sas_applies_skip_params() {
        let cols_skip = Some(vec!["id".to_string(), "q4".to_string()]);
        let (_, meta) = parse_sas_impl(
            &fixture("hadley.sas7bdat"),
            None,
            None,
            None,
            2,
            Some(3),
            cols_skip,
            false,
        )
        .unwrap();
        assert_eq!(meta.n_rows, 3);
        let names: Vec<&str> = meta.vars.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(names, ["workshop", "gender", "q1", "q2", "q3"]);
    }

    #[test]
    fn test_parse_sas_with_catalog() {
        let (_, meta) = parse_sas_impl(
            &fixture("hadley.sas7bdat"),
            Some(&fixture("formats.sas7bcat")),
            None,
            None,
            0,
            None,
            None,
            false,
        )
        .unwrap();
        assert_eq!(meta.n_rows, 8);
        let set_of = |name: &str| {
            let var = meta.vars.iter().find(|v| v.name == name).unwrap();
            let set = var.label_set.as_deref().unwrap();
            let labels = meta
                .value_labels
                .iter()
                .find(|l| l.set_name == set)
                .unwrap();
            labels.mapping.values().cloned().collect::<Vec<_>>()
        };
        assert_eq!(set_of("gender"), ["Female", "Male"]);
        assert_eq!(set_of("workshop"), ["R", "SAS"]);
    }
}
