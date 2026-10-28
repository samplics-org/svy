use super::*;

/// A live readstat parser, freed on drop so no return path leaks it.
pub(crate) struct ParserGuard(pub(super) *mut readstat_sys::readstat_parser_t);

impl ParserGuard {
    pub(crate) fn new() -> Result<Self> {
        // SAFETY: no preconditions; a null return is refused below.
        let p = unsafe { readstat_sys::readstat_parser_init() };
        if p.is_null() {
            return Err(Error::Internal("readstat_parser_init() failed".into()));
        }
        Ok(Self(p))
    }

    pub(crate) fn as_ptr(&self) -> *mut readstat_sys::readstat_parser_t {
        self.0
    }
}

impl Drop for ParserGuard {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `readstat_parser_init` and is freed only here.
        unsafe { readstat_sys::readstat_parser_free(self.0) };
    }
}
