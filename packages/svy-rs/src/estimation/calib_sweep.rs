// src/estimation/calib_sweep.rs
//! Calibration-aware score centering, the analogue of R's `svyrecvar`
//! `postStrata` loop.
//!
//! A calibrating adjustment pins quantities in the population, which removes
//! sampling variability the uncentred estimator still charges for. Feeding the
//! stratified PSU-total formula the *cell-centred* scores accounts for it:
//!
//! ```text
//! e_i = w*_i (z_i - z_c(i))     z_c = sum_{i in c} w_i z_i / sum_{i in c} w_i
//! ```
//!
//! Cell means use the PREVIOUS weights throughout. For a plain
//! poststratification that choice is immaterial -- the adjustment factor is
//! constant within a cell and cancels -- but it changes the answer wherever the
//! factor varies inside a sweep's cell: raking after the first margin, and GREG,
//! whose factor varies row by row. R uses the previous weights in every branch.
//!
//! Verified against survey 4.5 on apiclus1 before being written here: cells
//! 23.875658304826, GREG 3.295880912157, raking 23.745841047537, each matching
//! R to twelve significant figures, and a poststratified margin total giving
//! exactly zero.

use crate::regression::wols::solve_kxk;

/// Marks a row outside the adjustment: factor 1, no centring.
pub const OUT_OF_SCOPE: u32 = u32::MAX;

/// How many iterations R runs its raking sweep. Not a convergence loop -- a
/// fixed count, matched here so the two agree.
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
    /// Auxiliary matrix, column-major, for `Greg`. Non-finite entries mark a
    /// row outside the calibration.
    pub aux: Vec<Vec<f64>>,
    pub prev_w: Vec<f64>,
    /// Current weights, already carrying R's `psw == 0 & oldw == 0 -> psw = 1`
    /// guard so zero-weight rows cannot divide by zero.
    pub new_w: Vec<f64>,
    /// False when only the composition was pinned (`shares`), leaving the
    /// grand total's variability intact.
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

    /// Centre `x` in place.
    pub fn apply(&self, x: &mut [f64]) {
        if x.len() != self.new_w.len() {
            return;
        }
        match self.kind {
            SweepKind::Cells => {
                if self.cells.is_empty() {
                    return;
                }
                if self.pins_total {
                    self.sweep_cells(x, 0);
                } else {
                    // `shares` pin k-1 contrasts, not k totals. Removing the
                    // cell means also removes the grand total, which is still
                    // estimated, so add that component back:
                    //   x - (P_cells x - P_global x)
                    // With sweep_* returning x - P x, that is
                    //   sweep_cells(x) + P_global x.
                    let orig = x.to_vec();
                    let mut global = orig.clone();
                    self.sweep_global(&mut global);
                    self.sweep_cells(x, 0);
                    for i in 0..x.len() {
                        x[i] += orig[i] - global[i];
                    }
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

    /// The same sweep with every in-scope row in one cell.
    fn sweep_global(&self, x: &mut [f64]) {
        let codes = &self.cells[0];
        let (mut num, mut den) = (0.0, 0.0);
        for i in 0..x.len() {
            if codes[i] == OUT_OF_SCOPE {
                continue;
            }
            num += x[i] * self.prev_w[i] / self.new_w[i];
            den += self.prev_w[i];
        }
        if den == 0.0 {
            return;
        }
        for i in 0..x.len() {
            if codes[i] != OUT_OF_SCOPE {
                x[i] -= (num / den) * self.new_w[i];
            }
        }
    }

    /// R rakes with the UNWEIGHTED group mean of `x / w`, unlike the
    /// poststratification branch. Matched deliberately.
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

    /// GREG: `w*(z - X b)` with `b` the WLS fit of `z` on `X` weighted by the
    /// PREVIOUS weights. R stores `qr(X * sqrt(oldw))` and `w = g * sqrt(oldw)`,
    /// which unwinds to exactly this; using the calibrated weights instead is
    /// close but wrong (3.2631 against 3.2959 on apiclus1).
    fn sweep_greg(&self, x: &mut [f64]) {
        let k = self.aux.len();
        if k == 0 {
            return;
        }
        let n = x.len();
        let (mut a, mut b) = (vec![0.0; k * k], vec![0.0; k]);
        for i in 0..n {
            let w = self.prev_w[i];
            if w == 0.0 || !self.aux.iter().all(|col| col[i].is_finite()) {
                continue;
            }
            let z = x[i] / self.new_w[i];
            for p in 0..k {
                let xp = self.aux[p][i];
                b[p] += w * xp * z;
                for q in 0..k {
                    a[p * k + q] += w * xp * self.aux[q][i];
                }
            }
        }
        let Some(beta) = solve_kxk(&a, &b, k) else {
            return;
        };
        for i in 0..n {
            if !self.aux.iter().all(|col| col[i].is_finite()) {
                continue;
            }
            let mut fit = 0.0;
            for p in 0..k {
                fit += self.aux[p][i] * beta[p];
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

fn codes_from(df: &DataFrame, name: &str) -> PolarsResult<(Vec<u32>, usize)> {
    let s = df.column(name)?.cast(&DataType::Int64)?;
    let ca = s.i64()?;
    let mut out = Vec::with_capacity(ca.len());
    let mut max = 0i64;
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

fn floats_from(df: &DataFrame, name: &str) -> PolarsResult<Vec<f64>> {
    let s = df.column(name)?.cast(&DataType::Float64)?;
    Ok(s.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

/// Build a sweep, or `None` when the record is provenance-only, its columns are
/// gone, or the active weight no longer matches what the record describes.
///
/// Returning `None` is the documented fallback: variance reverts to treating
/// weights as fixed, which is what it does today, rather than centring against
/// a structure that may no longer hold.
pub fn build_calib_sweep(df: &DataFrame, spec: &CalibSpec) -> Option<CalibSweep> {
    let kind = match spec.kind.as_str() {
        "poststratification" | "standardization" => SweepKind::Cells,
        "raking" => SweepKind::Raking,
        "calibration" => SweepKind::Greg,
        _ => return None,
    };

    let prev_w = floats_from(df, &spec.prev_wgt_col).ok()?;
    let new_w = floats_from(df, &spec.new_wgt_col).ok()?;

    let (mut cells, mut n_cells) = (Vec::new(), Vec::new());
    for name in &spec.cells_cols {
        let (c, k) = codes_from(df, name).ok()?;
        cells.push(c);
        n_cells.push(k);
    }

    let mut aux = Vec::new();
    for name in &spec.aux_cols {
        aux.push(floats_from(df, name).ok()?);
    }

    match kind {
        SweepKind::Greg if aux.is_empty() => return None,
        SweepKind::Cells | SweepKind::Raking if cells.is_empty() => return None,
        _ => {}
    }

    Some(CalibSweep::new(
        kind,
        cells,
        n_cells,
        aux,
        prev_w,
        new_w,
        spec.pins_total,
    ))
}
