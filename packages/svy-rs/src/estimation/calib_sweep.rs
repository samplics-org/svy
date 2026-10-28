//! Calibration-aware score centering, the analogue of R's `svyrecvar` `postStrata` loop.
//!
//! - Scores are cell-centred: `e_i = w*_i (z_i - z_c(i))`, `z_c` the weighted cell mean of `z`.
//! - Cell means use the PREVIOUS weights, as R does; that matters for raking and GREG only.

use crate::estimation::error::{EResult, EstimationError};
use crate::regression::wols::solve_kxk;

/// Marks a row outside the adjustment: factor 1, no centring.
pub const OUT_OF_SCOPE: u32 = u32::MAX;

/// R's raking sweep count: a fixed number, not a convergence loop.
const RAKE_ITERATIONS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepKind {
    /// One cell coding: poststratification and standardization.
    Cells,
    /// One coding per margin, swept in turn.
    Raking,
    /// Weighted-least-squares residual against the auxiliary matrix.
    Greg,
}

#[derive(Debug, Clone)]
pub struct CalibSweep {
    pub kind: SweepKind,
    /// One code vector per margin. `Cells` and `Greg` use at most one.
    pub cells: Vec<Vec<u32>>,
    pub n_cells: Vec<usize>,
    /// `Greg` auxiliary matrix, column-major; a non-finite entry marks a row outside it.
    pub aux: Vec<Vec<f64>>,
    pub prev_w: Vec<f64>,
    /// Current weights after R's `psw == 0 & oldw == 0 -> psw = 1` guard against dividing by zero.
    pub new_w: Vec<f64>,
    /// False when only the composition was pinned (`shares`), so the grand total stays variable.
    pub pins_total: bool,
}

impl CalibSweep {
    pub fn new(
        kind: SweepKind,
        cells: Vec<Vec<u32>>,
        n_cells: Vec<usize>,
        aux: Vec<Vec<f64>>,
        prev_w: Vec<f64>,
        mut new_w: Vec<f64>,
        pins_total: bool,
    ) -> Self {
        // R's guard only: rows outside a subpopulation are kept (see `apply`).
        for i in 0..new_w.len() {
            if new_w[i] == 0.0 && prev_w[i] == 0.0 {
                new_w[i] = 1.0;
            }
        }
        Self {
            kind,
            cells,
            n_cells,
            aux,
            prev_w,
            new_w,
            pins_total,
        }
    }

    /// Centre `x` in place, with cell means over the FULL sample: the calibration was whole-sample.
    /// Out-of-domain rows add zero score but keep their weight in the denominator, as in R.
    pub fn apply(&self, x: &mut [f64]) {
        // Built from the frame the scores come from, so a mismatch is a kernel bug.
        assert_eq!(
            x.len(),
            self.new_w.len(),
            "calibration sweep length mismatch"
        );
        match self.kind {
            SweepKind::Cells => {
                if self.cells.is_empty() {
                    return;
                }
                if self.pins_total {
                    self.sweep_cells(x, 0);
                } else {
                    self.sweep_shares(x);
                }
            }
            SweepKind::Raking => {
                for _ in 0..RAKE_ITERATIONS {
                    for m in 0..self.cells.len() {
                        self.sweep_raking(x, m);
                    }
                }
            }
            SweepKind::Greg => self.sweep_greg(x),
        }
    }

    /// Ordinary poststratification: subtract the previous-weight cell mean.
    fn sweep_cells(&self, x: &mut [f64], margin: usize) {
        let codes = &self.cells[margin];
        let k = self.n_cells[margin];
        let (mut num, mut den) = (vec![0.0; k], vec![0.0; k]);
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE {
                continue;
            }
            let c = c as usize;
            num[c] += x[i] * self.prev_w[i] / self.new_w[i];
            den[c] += self.prev_w[i];
        }
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE {
                continue;
            }
            let c = c as usize;
            if den[c] != 0.0 {
                x[i] -= (num[c] / den[c]) * self.new_w[i];
            }
        }
    }

    /// `shares` pin composition, not level: the GREG residual on `u_c = delta_c - s_c`, c < k.
    /// `s_c` is not on the record because the achieved composition IS the shares.
    fn sweep_shares(&self, x: &mut [f64]) {
        let codes = &self.cells[0];
        let k = self.n_cells[0];
        if k < 2 {
            return;
        }
        let mut cell_w = vec![0.0; k];
        let mut total = 0.0;
        for (i, &c) in codes.iter().enumerate().take(x.len()) {
            if c == OUT_OF_SCOPE {
                continue;
            }
            cell_w[c as usize] += self.new_w[i];
            total += self.new_w[i];
        }
        if total == 0.0 {
            return;
        }
        let shares: Vec<f64> = cell_w.iter().map(|w| w / total).collect();

        // One column per cell but the last: they are linearly dependent.
        let mut aux = vec![vec![0.0; x.len()]; k - 1];
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE {
                continue;
            }
            for (j, col) in aux.iter_mut().enumerate() {
                col[i] = f64::from(c as usize == j) - shares[j];
            }
        }
        self.wls_residual(x, &aux);
    }

    /// R rakes with the UNWEIGHTED group mean of `x / w`, unlike poststratification; matched.
    fn sweep_raking(&self, x: &mut [f64], margin: usize) {
        let codes = &self.cells[margin];
        let k = self.n_cells[margin];
        let (mut sum, mut cnt) = (vec![0.0; k], vec![0.0f64; k]);
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE {
                continue;
            }
            let c = c as usize;
            sum[c] += x[i] / self.new_w[i];
            cnt[c] += 1.0;
        }
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE {
                continue;
            }
            let c = c as usize;
            if cnt[c] != 0.0 {
                x[i] -= (sum[c] / cnt[c]) * self.new_w[i];
            }
        }
    }

    /// GREG: `w*(z - X b)`, `b` the WLS fit weighted by the PREVIOUS weights, which R's
    /// `qr(X * sqrt(oldw))` unwinds to; the calibrated weights give a close but wrong answer.
    fn sweep_greg(&self, x: &mut [f64]) {
        self.wls_residual(x, &self.aux)
    }

    /// `w*(z - X b)` with `b` the WLS fit of `z` on `X` weighted by prev_w.
    fn wls_residual(&self, x: &mut [f64], aux: &[Vec<f64>]) {
        let k = aux.len();
        if k == 0 {
            return;
        }
        let n = x.len();
        let (mut a, mut b) = (vec![0.0; k * k], vec![0.0; k]);
        for i in 0..n {
            let w = self.prev_w[i];
            if w == 0.0 || !aux.iter().all(|col| col[i].is_finite()) {
                continue;
            }
            let z = x[i] / self.new_w[i];
            for p in 0..k {
                let xp = aux[p][i];
                b[p] += w * xp * z;
                for q in 0..k {
                    a[p * k + q] += w * xp * aux[q][i];
                }
            }
        }
        let Some(beta) = solve_kxk(&a, &b, k) else {
            return;
        };
        for i in 0..n {
            if !aux.iter().all(|col| col[i].is_finite()) {
                continue;
            }
            let mut fit = 0.0;
            for p in 0..k {
                fit += aux[p][i] * beta[p];
            }
            x[i] -= fit * self.new_w[i];
        }
    }
}

// ============================================================================
// Construction from a DataFrame
// ============================================================================

use polars::prelude::*;

/// Column names naming a weight-adjustment record, as passed from Python.
#[derive(Debug, Clone, Default)]
pub struct CalibSpec {
    pub kind: String,
    pub cells_cols: Vec<String>,
    pub aux_cols: Vec<String>,
    pub prev_wgt_col: String,
    pub new_wgt_col: String,
    pub pins_total: bool,
}

fn codes_from(df: &DataFrame, name: &str) -> EResult<(Vec<u32>, usize)> {
    let col = df.column(name)?;
    // Cells arrive as Int32; casting to Int64 first would allocate a second full-length array.
    let s = if matches!(col.dtype(), DataType::Int32) {
        col.clone()
    } else {
        col.cast(&DataType::Int32)?
    };
    let ca = s.i32()?;
    let mut out = Vec::with_capacity(ca.len());
    let mut max = 0i32;
    for v in ca.iter() {
        match v {
            // Null marks a row the adjustment never touched.
            None => out.push(OUT_OF_SCOPE),
            Some(c) if c < 0 => out.push(OUT_OF_SCOPE),
            Some(c) => {
                if c > max {
                    max = c;
                }
                out.push(c as u32)
            }
        }
    }
    Ok((out, (max + 1) as usize))
}

fn floats_from(df: &DataFrame, name: &str) -> EResult<Vec<f64>> {
    let col = df.column(name)?;
    // Always complete Float64 here: memcpy beats an option collect on this per-call path.
    if matches!(col.dtype(), DataType::Float64) {
        let ca = col.f64()?;
        if ca.null_count() == 0
            && let Ok(sl) = ca.cont_slice()
        {
            return Ok(sl.to_vec());
        }
    }
    let s = col.cast(&DataType::Float64)?;
    Ok(s.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

/// Build the sweep for a record Python has already checked still describes the data.
pub fn build_calib_sweep(df: &DataFrame, spec: &CalibSpec) -> EResult<CalibSweep> {
    let kind = match spec.kind.as_str() {
        "poststratification" | "standardization" => SweepKind::Cells,
        "raking" => SweepKind::Raking,
        "calibration" => SweepKind::Greg,
        other => {
            return Err(EstimationError::InvalidChoice {
                what: "calibration kind",
                got: other.to_string(),
                expected: "'poststratification', 'standardization', 'raking' or 'calibration'",
            });
        }
    };

    let prev_w = floats_from(df, &spec.prev_wgt_col)?;
    let new_w = floats_from(df, &spec.new_wgt_col)?;

    let (mut cells, mut n_cells) = (Vec::new(), Vec::new());
    for name in &spec.cells_cols {
        let (c, k) = codes_from(df, name)?;
        cells.push(c);
        n_cells.push(k);
    }

    let mut aux = Vec::new();
    for name in &spec.aux_cols {
        aux.push(floats_from(df, name)?);
    }

    match kind {
        SweepKind::Greg if aux.is_empty() => {
            return Err(EstimationError::Empty(
                "a calibration sweep needs calib_aux",
            ));
        }
        SweepKind::Cells | SweepKind::Raking if cells.is_empty() => {
            return Err(EstimationError::Empty("a cell sweep needs calib_cells"));
        }
        _ => {}
    }

    Ok(CalibSweep::new(
        kind,
        cells,
        n_cells,
        aux,
        prev_w,
        new_w,
        spec.pins_total,
    ))
}

/// The sweep for a wrapper's `calib_*` arguments; `None` when no record was passed.
/// `new_wgt` is `weight_col` except under a subpopulation filter: then it is the pre-zeroing copy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_calib(
    df: &DataFrame,
    weight_col: &str,
    kind: Option<String>,
    cells: Option<Vec<String>>,
    aux: Option<Vec<String>>,
    prev_wgt: Option<String>,
    pins_total: Option<bool>,
    new_wgt: Option<String>,
) -> EResult<Option<CalibSweep>> {
    let Some(kind) = kind else {
        return Ok(None);
    };
    let spec = CalibSpec {
        kind,
        cells_cols: cells.unwrap_or_default(),
        aux_cols: aux.unwrap_or_default(),
        prev_wgt_col: prev_wgt.ok_or(EstimationError::Empty("calib_kind needs calib_prev_wgt"))?,
        new_wgt_col: new_wgt.unwrap_or_else(|| weight_col.to_string()),
        pins_total: pins_total.unwrap_or(true),
    };
    build_calib_sweep(df, &spec).map(Some)
}

/// Centre a `Float64Chunked` score column, so `taylor_variance` callers receive centred scores.
pub fn sweep_scores(scores: &Float64Chunked, calib: Option<&CalibSweep>) -> Float64Chunked {
    match calib {
        None => scores.clone(),
        Some(c) => {
            let mut v: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
            c.apply(&mut v);
            Float64Chunked::from_slice(scores.name().clone(), &v)
        }
    }
}
