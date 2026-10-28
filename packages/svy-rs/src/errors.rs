//! Kernel errors Python can tell apart: a `ValueError` or `RuntimeError` carrying a `code`.

use pyo3::prelude::*;

pyo3::create_exception!(_internal, KernelValueError, pyo3::exceptions::PyValueError);
pyo3::create_exception!(
    _internal,
    KernelRuntimeError,
    pyo3::exceptions::PyRuntimeError
);

/// An area error the wrappers raise as a coded exception. Codes reuse `svy.errors.Code`
/// names where one fits, so Python branches on `e.code`, never on the message.
pub(crate) trait Coded: std::fmt::Display {
    fn code(&self) -> &'static str;
    /// Input the caller should not have sent (`ValueError`), not a numerical failure.
    fn invalid_input(&self) -> bool;
}

pub(crate) fn to_py_err(e: &impl Coded) -> PyErr {
    let err = if e.invalid_input() {
        KernelValueError::new_err(e.to_string())
    } else {
        KernelRuntimeError::new_err(e.to_string())
    };
    Python::attach(|py| {
        // A failed setattr leaves an uncoded but still typed exception.
        let _ = err.value(py).setattr("code", e.code());
    });
    err
}

/// A polars failure inside a kernel: not the caller's input, and no finer code.
pub(crate) const KERNEL_INTERNAL: &str = "KERNEL_INTERNAL";
