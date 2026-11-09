//! The error contract every kernel error meets, whatever binding raises it.

/// Error carrying a `code` (a `svy.errors.Code` name where one fits) for callers to branch on.
pub trait Coded: std::fmt::Display {
    fn code(&self) -> &'static str;
    /// Input the caller should not have sent (`ValueError`), not a numerical failure.
    fn invalid_input(&self) -> bool;
}

/// A polars failure inside a kernel: not the caller's input, and no finer code.
pub(crate) const KERNEL_INTERNAL: &str = "KERNEL_INTERNAL";
