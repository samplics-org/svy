//! Reader and writer errors, raised in Python as exceptions carrying a `code`.

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use thiserror::Error;

pyo3::create_exception!(
    svyreadstat_rs,
    ReadStatValueError,
    pyo3::exceptions::PyValueError
);
pyo3::create_exception!(
    svyreadstat_rs,
    ReadStatRuntimeError,
    pyo3::exceptions::PyRuntimeError
);

#[derive(Error, Debug)]
pub(crate) enum Error {
    #[error("{0}")]
    Invalid(String),

    #[error("unsupported encoding {encoding:?}: {detail}")]
    UnsupportedEncoding { encoding: String, detail: String },

    #[error("{0}")]
    BadString(String),

    #[error("{0}")]
    Parse(String),

    #[error("{0}")]
    Write(String),

    #[error("{0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Internal(String),
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy)]
pub(crate) enum Op {
    Read,
    Write,
}

impl Error {
    /// Codes reuse `svy.errors.Code` names where one fits, so Python branches on `e.code`.
    pub(crate) fn code(&self, op: Op) -> &'static str {
        match self {
            Self::Invalid(_) => "INVALID_VALUE",
            Self::UnsupportedEncoding { .. } => "UNSUPPORTED_ENCODING",
            Self::BadString(_) => "READSTAT_BAD_STRING",
            Self::Parse(_) => "READSTAT_PARSE_FAILED",
            Self::Write(_) => "READSTAT_WRITE_FAILED",
            Self::Io(_) => match op {
                Op::Read => "IO_READ_FAILED",
                Op::Write => "IO_WRITE_FAILED",
            },
            Self::Internal(_) => "IO_INTERNAL",
        }
    }

    fn verb(&self, op: Op) -> &'static str {
        match (self, op) {
            (Self::BadString(_) | Self::Parse(_), _) => "parse",
            (_, Op::Read) => "read",
            (_, Op::Write) => "write",
        }
    }

    /// The Python exception, naming the operation, the format and the path.
    pub(crate) fn into_py(self, op: Op, format: &str, path: &str) -> PyErr {
        let code = self.code(op);
        let verb = self.verb(op);
        let what = format!("Failed to {verb} {format} file {path:?}");
        let err = match self {
            Self::Invalid(_) | Self::UnsupportedEncoding { .. } => {
                ReadStatValueError::new_err(format!("{what}: {self}"))
            }
            // OSError(errno, strerror, filename) becomes FileNotFoundError, PermissionError, ...
            Self::Io(e) => match e.raw_os_error() {
                Some(errno) => {
                    let strerror = format!("Failed to {verb} {format} file: {e}");
                    PyOSError::new_err((errno, strerror, path.to_string()))
                }
                None => PyOSError::new_err(format!("{what}: {e}")),
            },
            _ => ReadStatRuntimeError::new_err(format!("{what}: {self}")),
        };
        Python::attach(|py| {
            // A failed setattr leaves an uncoded but still typed exception.
            let _ = err.value(py).setattr("code", code);
        });
        err
    }
}

/// ReadStat's text for a return code, with the code itself.
pub(crate) fn readstat_message(rc: readstat_sys::readstat_error_t) -> String {
    let p = unsafe { readstat_sys::readstat_error_message(rc) };
    if p.is_null() {
        return format!("rc={rc}");
    }
    let text = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy();
    format!("{text} (rc={rc})")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raised(err: Error, op: Op) -> (String, String, String) {
        Python::attach(|py| {
            let e = err.into_py(op, "SAV", "/d/f.sav");
            let v = e.value(py);
            (
                v.get_type().name().unwrap().to_string(),
                v.getattr("code").unwrap().extract().unwrap(),
                v.str().unwrap().to_string(),
            )
        })
    }

    #[test]
    fn input_errors_raise_a_coded_value_error() {
        let got = raised(Error::Invalid("bad label".into()), Op::Write);
        assert_eq!(
            got,
            (
                "ReadStatValueError".into(),
                "INVALID_VALUE".into(),
                "Failed to write SAV file \"/d/f.sav\": bad label".into()
            )
        );
    }

    #[test]
    fn parse_failures_raise_a_coded_runtime_error() {
        let got = raised(Error::BadString("rc=17".into()), Op::Read);
        assert_eq!(
            got,
            (
                "ReadStatRuntimeError".into(),
                "READSTAT_BAD_STRING".into(),
                "Failed to parse SAV file \"/d/f.sav\": rc=17".into()
            )
        );
    }

    #[test]
    fn os_errors_raise_the_errno_subclass_with_the_filename() {
        let io = std::io::Error::from_raw_os_error(2);
        let text = io.to_string();
        Python::attach(|py| {
            let e = Error::Io(io).into_py(Op::Write, "SAV", "/d/f.sav");
            let v = e.value(py);
            assert_eq!(v.get_type().name().unwrap(), "FileNotFoundError");
            assert_eq!(v.getattr("errno").unwrap().extract::<i32>().unwrap(), 2);
            assert_eq!(
                v.getattr("filename").unwrap().extract::<String>().unwrap(),
                "/d/f.sav"
            );
            assert_eq!(
                v.getattr("code").unwrap().extract::<String>().unwrap(),
                "IO_WRITE_FAILED"
            );
            assert_eq!(
                v.getattr("strerror").unwrap().extract::<String>().unwrap(),
                format!("Failed to write SAV file: {text}")
            );
        });
    }
}
