use super::*;

pub(crate) const HANDLER_OK: c_int = 0;

/// Sets the iconv input encoding and a row limit backing up n_max; keep the returned CString alive through the parse.
pub(crate) fn configure_parser(
    parser: &ParserGuard,
    encoding: Option<&str>,
    rows_skip: usize,
    n_max: Option<usize>,
) -> Result<Option<std::ffi::CString>> {
    let mut keep_enc = None;
    if let Some(enc) = encoding {
        let c = std::ffi::CString::new(enc)
            .map_err(|_| Error::Invalid(format!("encoding {enc:?} contains an embedded NUL")))?;
        // SAFETY: `parser` is live; ReadStat keeps this pointer, so `c` goes back to the caller.
        let rc =
            unsafe { readstat_sys::readstat_set_file_character_encoding(parser.0, c.as_ptr()) };
        if rc != readstat_sys::readstat_error_e_READSTAT_OK {
            return Err(Error::UnsupportedEncoding {
                encoding: enc.to_string(),
                detail: readstat_message(rc),
            });
        }
        keep_enc = Some(c);
    }
    if let Some(limit) = n_max.and_then(|nm| row_limit(rows_skip, nm)) {
        // SAFETY: `parser` is live.
        unsafe { readstat_sys::readstat_set_row_limit(parser.0, limit) };
    }
    Ok(keep_enc)
}

/// `None` for no limit: zero, or past a C `long` (32 bits on Windows), which no file can hold.
pub(super) fn row_limit(rows_skip: usize, n_max: usize) -> Option<::std::os::raw::c_long> {
    let limit = rows_skip.saturating_add(n_max);
    if limit == 0 {
        return None;
    }
    limit.try_into().ok()
}
pub(crate) const HANDLER_ABORT: c_int = 1;

pub(super) const DEFAULT_CAPACITY: usize = 8192;
pub(super) const STRING_CAPACITY_MULTIPLIER: usize = 16;
/// Caps builder pre-allocation only: the header's row count is untrusted and may claim billions.
pub(super) const MAX_PREALLOC_ROWS: usize = 65_536;
