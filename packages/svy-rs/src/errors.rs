//! Kernel errors Python can tell apart: a `ValueError` or `RuntimeError` carrying a `code`.

use pyo3::prelude::*;
pub(crate) use svy_core::errors::{Coded, KERNEL_INTERNAL};

pyo3::create_exception!(_internal, KernelValueError, pyo3::exceptions::PyValueError);
pyo3::create_exception!(
    _internal,
    KernelRuntimeError,
    pyo3::exceptions::PyRuntimeError
);

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
