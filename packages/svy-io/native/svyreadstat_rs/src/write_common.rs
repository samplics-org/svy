//! Pieces the SAV, DTA and XPT writers share: Arrow input, cell access and ReadStat output.

use std::io::Cursor;
use std::os::raw::c_void;

use arrow::array::{
    Array, AsArray, DictionaryArray, LargeStringArray, StringArray, StringViewArray,
};
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::record_batch::RecordBatch;

use crate::errors::{Error, Result, readstat_message};

pub(crate) fn ipc_to_batches(buf: &[u8]) -> arrow::error::Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    let schema = if buf.starts_with(b"ARROW1") {
        let mut fr = FileReader::try_new(Cursor::new(buf), None)?;
        for b in fr.by_ref() {
            batches.push(b?);
        }
        fr.schema()
    } else {
        let mut sr = StreamReader::try_new(Cursor::new(buf), None)?;
        for res in sr.by_ref() {
            batches.push(res?);
        }
        sr.schema()
    };
    // A zero-row frame arrives with no batches; the writers still need its columns.
    if batches.is_empty() {
        batches.push(RecordBatch::new_empty(schema));
    }
    Ok(batches)
}

#[inline]
pub(crate) fn is_text_dt(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

pub(crate) fn get_string_value(a: &dyn Array, row: usize) -> Option<&str> {
    if a.is_null(row) {
        return None;
    }
    if let Some(s) = a.as_any().downcast_ref::<StringArray>() {
        Some(s.value(row))
    } else if let Some(s) = a.as_any().downcast_ref::<LargeStringArray>() {
        Some(s.value(row))
    } else if let Some(s) = a.as_any().downcast_ref::<StringViewArray>() {
        Some(s.value(row))
    } else if matches!(a.data_type(), DataType::Dictionary(_, _)) {
        dict_string_at_any(a, row)
    } else {
        None
    }
}

pub(crate) fn dict_string_at_any(a: &dyn Array, row: usize) -> Option<&str> {
    macro_rules! try_dict {
        ($T:ty) => {{
            if let Some(d) = a.as_any().downcast_ref::<DictionaryArray<$T>>() {
                if !is_text_dt(d.values().data_type()) || d.is_null(row) {
                    return None;
                }
                // A corrupt dictionary can hold a negative or out-of-range key; `as usize` would wrap.
                let key_usize = usize::try_from(d.keys().value(row)).ok()?;
                let values = d.values();
                if key_usize >= values.len() {
                    return None;
                }
                return get_string_value(values.as_ref(), key_usize);
            }
        }};
    }
    try_dict!(Int8Type);
    try_dict!(Int16Type);
    try_dict!(Int32Type);
    try_dict!(Int64Type);
    try_dict!(UInt8Type);
    try_dict!(UInt16Type);
    try_dict!(UInt32Type);
    try_dict!(UInt64Type);
    None
}

pub(crate) fn as_f64_opt(a: &dyn Array, row: usize) -> Option<f64> {
    if a.is_null(row) {
        return None;
    }
    macro_rules! down {
        ($T:ty) => {
            a.as_primitive::<$T>().value(row)
        };
    }
    use DataType::*;
    match a.data_type() {
        Float64 => Some(down!(Float64Type)),
        Float32 => Some(down!(Float32Type) as f64),
        Int64 => Some(down!(Int64Type) as f64),
        Int32 => Some(down!(Int32Type) as f64),
        Int16 => Some(down!(Int16Type) as f64),
        Int8 => Some(down!(Int8Type) as f64),
        UInt64 => Some(down!(UInt64Type) as f64),
        UInt32 => Some(down!(UInt32Type) as f64),
        UInt16 => Some(down!(UInt16Type) as f64),
        UInt8 => Some(down!(UInt8Type) as f64),
        Boolean => Some(if a.as_boolean().value(row) { 1.0 } else { 0.0 }),
        _ => None,
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct StringColStats {
    pub(crate) max_len: usize,
    pub(crate) has_nul: bool,
}

pub(crate) fn compute_string_metadata(batches: &[RecordBatch]) -> Vec<Option<StringColStats>> {
    if batches.is_empty() {
        return Vec::new();
    }
    let ncols = batches[0].schema().fields().len();
    let mut all_stats: Vec<Option<StringColStats>> = vec![None; ncols];

    for b in batches {
        for (j, f) in b.schema().fields().iter().enumerate() {
            let col = b.column(j);
            let is_str = is_text_dt(f.data_type())
                || matches!(f.data_type(), DataType::Dictionary(_, v) if is_text_dt(v.as_ref()));
            if !is_str {
                continue;
            }
            let stats = all_stats[j].get_or_insert(StringColStats::default());
            for i in 0..col.len() {
                if let Some(s) = get_string_value(col.as_ref(), i) {
                    let blen = s.len();
                    if blen > stats.max_len {
                        stats.max_len = blen;
                    }
                    if !stats.has_nul && s.as_bytes().contains(&0) {
                        stats.has_nul = true;
                    }
                }
            }
        }
    }
    all_stats
}

// ---------- Writer guard ----------

/// Freed on drop, so an early return cannot leak the writer and everything it owns.
pub(crate) struct WriterGuard(pub(crate) *mut readstat_sys::readstat_writer_t);

impl Drop for WriterGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the guard holds the pointer `readstat_writer_init` returned, freed only here.
            unsafe { readstat_sys::readstat_writer_free(self.0) };
        }
    }
}

/// `Ok` for `READSTAT_OK`, else a write failure naming `what` with ReadStat's text.
#[inline]
pub(crate) fn write_rc(
    rc: readstat_sys::readstat_error_t,
    what: impl FnOnce() -> String,
) -> Result<()> {
    if rc == readstat_sys::readstat_error_e_READSTAT_OK {
        return Ok(());
    }
    Err(Error::Write(format!(
        "{}: {}",
        what(),
        readstat_message(rc)
    )))
}

/// Removed on drop unless `finish` ran, so a failed write leaves no truncated file.
pub(crate) struct OutputFile {
    file: Option<std::fs::File>,
    path: std::path::PathBuf,
}

impl OutputFile {
    pub(crate) fn create(path: &str) -> Result<Self> {
        Ok(Self {
            file: Some(std::fs::File::create(path)?),
            path: path.into(),
        })
    }

    /// The `ctx` pointer for `readstat_begin_writing_*`; valid while `self` lives.
    pub(crate) fn as_ctx(&mut self) -> *mut c_void {
        match self.file.as_mut() {
            Some(f) => f as *mut std::fs::File as *mut c_void,
            None => std::ptr::null_mut(),
        }
    }

    pub(crate) fn finish(mut self) {
        self.file.take();
    }
}

impl Drop for OutputFile {
    fn drop(&mut self) {
        // Closed before removal: Windows cannot delete an open file.
        if self.file.take().is_some() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// ReadStat data sink writing to the `std::fs::File` behind `ctx`.
///
/// # Safety
///
/// Called only by ReadStat: `data` is null or `len` readable bytes, and `ctx` is null or
/// the pointer `OutputFile::as_ctx` returned.
pub(crate) unsafe extern "C" fn data_writer_cb(
    data: *const c_void,
    len: usize,
    ctx: *mut c_void,
) -> isize {
    use std::io::Write;
    // ReadStat stores the result in a size_t, so -1 would pass as success; a short count is the failure.
    if data.is_null() || ctx.is_null() {
        return 0;
    }
    // SAFETY: `ctx` is the `OutputFile`'s `File`, alive and otherwise untouched while writing.
    let file = unsafe { &mut *(ctx as *mut std::fs::File) };
    // SAFETY: ReadStat passes a buffer of `len` readable bytes, valid for the call.
    let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, len) };
    match file.write_all(bytes) {
        Ok(_) => len as isize,
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use readstat_sys::{
        readstat_error_e_READSTAT_ERROR_PARSE as RC_PARSE, readstat_error_e_READSTAT_OK as RC_OK,
    };

    #[test]
    fn write_rc_passes_ok_and_names_the_failed_step() {
        assert!(write_rc(RC_OK, || unreachable!()).is_ok());
        let err = write_rc(RC_PARSE, || "ending row 3".into()).unwrap_err();
        let want = "ending row 3: Invalid file, or file has unsupported features (rc=5)";
        assert!(matches!(&err, Error::Write(m) if m == want), "{err:?}");
    }

    #[test]
    fn output_file_is_removed_unless_finished() {
        let path = std::env::temp_dir().join("svy_io_output_file.bin");
        let p = path.to_str().unwrap();

        let out = OutputFile::create(p).unwrap();
        assert!(path.exists());
        drop(out);
        assert!(!path.exists());

        OutputFile::create(p).unwrap().finish();
        assert!(path.exists());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn data_writer_cb_reports_a_failed_write_as_zero_bytes() {
        let sink = |data: *const c_void, len: usize, ctx: *mut c_void| {
            // SAFETY: each call passes null or 3 readable bytes, and null or a live `File`.
            unsafe { data_writer_cb(data, len, ctx) }
        };
        let path = std::env::temp_dir().join("svy_io_data_writer_cb.bin");
        std::fs::write(&path, b"").unwrap();
        let mut read_only = std::fs::File::open(&path).unwrap();
        let ctx = &mut read_only as *mut std::fs::File as *mut c_void;
        let bytes = b"abc".as_ptr() as *const c_void;

        assert_eq!(sink(bytes, 3, ctx), 0);
        assert_eq!(sink(std::ptr::null(), 3, ctx), 0);
        assert_eq!(sink(bytes, 3, std::ptr::null_mut()), 0);

        let mut writable = std::fs::File::create(&path).unwrap();
        let ctx = &mut writable as *mut std::fs::File as *mut c_void;
        assert_eq!(sink(bytes, 3, ctx), 3);
        drop(writable);
        assert_eq!(std::fs::read(&path).unwrap(), b"abc");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn output_file_in_a_missing_directory_is_an_io_error() {
        let path = std::env::temp_dir()
            .join("svy_io_no_such_dir")
            .join("x.sav");
        let err = OutputFile::create(path.to_str().unwrap()).err().unwrap();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound),
            "{err:?}"
        );
    }
}
