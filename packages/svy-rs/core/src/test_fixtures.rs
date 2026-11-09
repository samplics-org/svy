//! The 12-row sample the kernel unit tests share with tests/golden/kernel_unit_r.R.
//!
//! Three strata of two PSUs, unequal weights, and a tie in `Y` (rows 4 and 5).

use polars::prelude::*;

pub(crate) const N: usize = 12;
pub(crate) const STRATA: [i64; N] = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
pub(crate) const PSU: [i64; N] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6];
pub(crate) const W: [f64; N] = [1.5, 2.0, 1.0, 3.0, 2.5, 1.0, 2.0, 1.5, 3.0, 1.0, 2.0, 2.5];
pub(crate) const Y: [f64; N] = [3.1, 4.2, 2.8, 5.0, 4.4, 4.4, 6.1, 5.5, 2.2, 3.3, 4.8, 4.0];
pub(crate) const G2: [u32; N] = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0];
pub(crate) const G3: [u32; N] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
pub(crate) const X2: [f64; N] = [1.0, 2.5, 0.5, 3.0, 2.0, 1.5, 4.0, 3.5, 0.8, 1.2, 2.8, 2.2];
pub(crate) const COUNTS: [f64; N] = [
    0.0, 7.0, 1.0, 12.0, 3.0, 0.0, 15.0, 2.0, 0.0, 9.0, 18.0, 1.0,
];
/// Row (`r`) and column (`cc`) codes of the 2 x 3 table.
pub(crate) const ROW: [u32; N] = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0];
pub(crate) const COL: [u32; N] = [0, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1, 1];

pub(crate) fn strata() -> Column {
    Column::from(Int64Chunked::from_slice("st".into(), &STRATA).into_series())
}

pub(crate) fn psu() -> Column {
    Column::from(Int64Chunked::from_slice("psu".into(), &PSU).into_series())
}

pub(crate) fn floats(name: &str, v: &[f64]) -> Float64Chunked {
    Float64Chunked::from_slice(name.into(), v)
}
