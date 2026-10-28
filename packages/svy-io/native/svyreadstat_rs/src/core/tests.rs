use super::*;

#[test]
fn row_limit_is_skip_plus_n_max() {
    assert_eq!(row_limit(0, 5), Some(5));
    assert_eq!(row_limit(10, 5), Some(15));
    assert_eq!(row_limit(0, 0), None);
}

#[test]
fn row_limit_past_c_long_means_no_limit() {
    let c_long_max = ::std::os::raw::c_long::MAX as usize;
    assert_eq!(row_limit(0, c_long_max), Some(::std::os::raw::c_long::MAX));
    assert_eq!(row_limit(1, c_long_max), None);
    assert_eq!(row_limit(usize::MAX, usize::MAX), None);
}

#[test]
fn safe_capacity_uses_default_when_unknown() {
    assert_eq!(safe_capacity(None), DEFAULT_CAPACITY);
    assert_eq!(safe_capacity(Some(0)), DEFAULT_CAPACITY);
}

#[test]
fn safe_capacity_passes_small_counts_through() {
    assert_eq!(safe_capacity(Some(1)), 1);
    assert_eq!(safe_capacity(Some(10_000)), 10_000);
    assert_eq!(safe_capacity(Some(MAX_PREALLOC_ROWS)), MAX_PREALLOC_ROWS);
}

#[test]
fn safe_capacity_clamps_untrusted_huge_counts() {
    // At most ~1 MB of string data per column, not tens of GB.
    assert_eq!(safe_capacity(Some(999_999_999)), MAX_PREALLOC_ROWS);
    assert_eq!(safe_capacity(Some(usize::MAX)), MAX_PREALLOC_ROWS);
    const { assert!(MAX_PREALLOC_ROWS * STRING_CAPACITY_MULTIPLIER <= 16 * 1024 * 1024) };
}

#[test]
fn describe_panic_extracts_messages() {
    let p: Box<dyn std::any::Any + Send> = Box::new("static message");
    assert_eq!(describe_panic(p.as_ref()), "static message");
    let p: Box<dyn std::any::Any + Send> = Box::new(String::from("owned message"));
    assert_eq!(describe_panic(p.as_ref()), "owned message");
    let p: Box<dyn std::any::Any + Send> = Box::new(42_u32);
    assert_eq!(describe_panic(p.as_ref()), "unknown panic");
}

#[test]
fn guard_cb_converts_panic_into_abort_and_stored_error() {
    let mut ctx = ParseCtx {
        cols: Vec::new(),
        var_cols: Vec::new(),
        cols_skip: None,
        rows_skip: 0,
        n_max: None,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        last_counted_row: None,
        had_invalid_utf8: false,
        lossy_utf8: false,
        label_sets: HashMap::new(),
        file_label: None,
        last_err: None,
        tagged: HashMap::new(),
        notes: Vec::new(),
        detect_tagged: false,
        row_capacity: None,
        panic_err: None,
    };
    let ctx_ptr = &mut ctx as *mut ParseCtx as *mut c_void;

    // SAFETY: `ctx_ptr` points to the live `ctx`, untouched until the call returns.
    let rc = unsafe { guard_cb(ctx_ptr, |_| panic!("boom in callback")) };
    assert_eq!(rc, HANDLER_ABORT);
    assert_eq!(ctx.panic_err.as_deref(), Some("boom in callback"));

    ctx.panic_err = None;
    // Re-derived: the write through `ctx` above invalidated the first pointer.
    let ctx_ptr = &mut ctx as *mut ParseCtx as *mut c_void;
    // SAFETY: `ctx_ptr` points to the live `ctx`, untouched until the call returns.
    let rc = unsafe { guard_cb(ctx_ptr, |_| HANDLER_OK) };
    assert_eq!(rc, HANDLER_OK);
    assert!(ctx.panic_err.is_none());
}

fn empty_ctx() -> ParseCtx {
    ParseCtx {
        cols: Vec::new(),
        var_cols: Vec::new(),
        cols_skip: None,
        rows_skip: 0,
        n_max: None,
        n_rows_seen: 0,
        n_rows_emitted: 0,
        last_counted_row: None,
        had_invalid_utf8: false,
        lossy_utf8: false,
        label_sets: HashMap::new(),
        file_label: None,
        last_err: None,
        tagged: HashMap::new(),
        notes: Vec::new(),
        detect_tagged: false,
        row_capacity: None,
        panic_err: None,
    }
}

fn f64_col(name: &str) -> ColBuilders {
    ColBuilders {
        kind: ColKind::F64 { float32: false },
        name: name.to_string(),
        label: None,
        label_set: None,
        fmt: None,
        measure: None,
        user_missing: None,
        sb: None,
        fb: Some(Float64Builder::new()),
        ib: None,
    }
}

use readstat_sys::{
    readstat_error_e_READSTAT_ERROR_CONVERT_BAD_STRING as RC_BAD_STRING,
    readstat_error_e_READSTAT_ERROR_OPEN as RC_OPEN,
    readstat_error_e_READSTAT_ERROR_PARSE as RC_PARSE,
    readstat_error_e_READSTAT_ERROR_UNSUPPORTED_CHARSET as RC_CHARSET,
    readstat_error_e_READSTAT_ERROR_USER_ABORT as RC_ABORT, readstat_error_e_READSTAT_OK as RC_OK,
};

#[test]
fn check_parse_accepts_success_and_the_n_max_abort() {
    for rc in [RC_OK, RC_ABORT] {
        assert!(check_parse(rc, &mut empty_ctx(), "f.sav", None).is_ok());
    }
}

#[test]
fn check_parse_reports_a_caught_panic_before_the_return_code() {
    let mut ctx = empty_ctx();
    ctx.panic_err = Some("boom".into());
    let err = check_parse(RC_OK, &mut ctx, "f.sav", None).unwrap_err();
    assert!(
        matches!(&err, Error::Internal(m) if m == "internal error in readstat callback: boom"),
        "{err:?}"
    );
    assert!(ctx.panic_err.is_none());
}

#[test]
fn check_parse_appends_readstats_last_message() {
    let mut ctx = empty_ctx();
    ctx.last_err = Some("bad header".into());
    let err = check_parse(RC_PARSE, &mut ctx, "f.sav", None).unwrap_err();
    let want = "Invalid file, or file has unsupported features (rc=5): bad header";
    assert!(matches!(&err, Error::Parse(m) if m == want), "{err:?}");
}

#[test]
fn check_parse_codes_an_undecodable_string() {
    let err = check_parse(RC_BAD_STRING, &mut empty_ctx(), "f.dta", None).unwrap_err();
    let want = "Unable to convert string to the requested encoding (invalid byte sequence) (rc=17)";
    assert!(matches!(&err, Error::BadString(m) if m == want), "{err:?}");
}

#[test]
fn check_parse_blames_an_unsupported_charset_on_the_callers_encoding_only() {
    let want = "File has an unsupported character set (rc=7)";
    let err = check_parse(RC_CHARSET, &mut empty_ctx(), "f.sav", Some("nope")).unwrap_err();
    assert!(
        matches!(&err, Error::UnsupportedEncoding { encoding, detail }
            if encoding == "nope" && detail == want),
        "{err:?}"
    );
    let err = check_parse(RC_CHARSET, &mut empty_ctx(), "f.sav", None).unwrap_err();
    assert!(matches!(&err, Error::Parse(m) if m == want), "{err:?}");
}

#[test]
fn check_parse_turns_an_unopenable_path_into_an_io_error() {
    let missing = std::env::temp_dir().join("svy_io_check_parse_missing.sav");
    let _ = std::fs::remove_file(&missing);
    let err = check_parse(RC_OPEN, &mut empty_ctx(), missing.to_str().unwrap(), None).unwrap_err();
    assert!(
        matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound),
        "{err:?}"
    );

    // A path that opens fine leaves ReadStat's own report.
    let present = std::env::temp_dir().join("svy_io_check_parse_present.sav");
    std::fs::write(&present, b"x").unwrap();
    let err = check_parse(RC_OPEN, &mut empty_ctx(), present.to_str().unwrap(), None).unwrap_err();
    std::fs::remove_file(&present).unwrap();
    assert!(
        matches!(&err, Error::Parse(m) if m == "Unable to open file (rc=1)"),
        "{err:?}"
    );
}

#[test]
fn finalize_rejects_two_variables_with_one_name() {
    let mut ctx = empty_ctx();
    ctx.cols = vec![f64_col("a"), f64_col("b"), f64_col("a")];
    let err = finalize_to_ipc(ctx).err().unwrap();
    assert!(
        matches!(&err, Error::Parse(m) if m == "the file has more than one variable named \"a\""),
        "{err:?}"
    );
}
